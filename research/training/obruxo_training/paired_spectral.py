"""Learn signed local sound changes from aligned, training-only native pairs."""

import torch


def pair_spec():
    return {"format": "aligned_control_change_v1", "weight": .25, "batch": "four_pairs_eight_recordings",
            "continuous_changes": "one_active_control_per_pair_cycling_over_nine_controls",
            "target": "signed_log1p_magnitude_and_envelope_differences", "log_floor": .0001,
            "normalization": "mean_absolute_native_difference_floor_0.05", "difference_floor": .05,
            "spectral_weight": 1., "envelope_weight": .1, "side_weight": .1,
            "sampling": "identical_absolute_times_within_each_pair", "minimum_complete_pair_fraction": .9}


def complete_pairs(cache):
    positions = {row["sequence"]: i for i, row in enumerate(cache.rows)}
    return [(positions[sequence - 4], position) for sequence, position in positions.items()
            if sequence % 8 >= 4 and sequence - 4 in positions]


def require_aligned_pairs(records):
    if len(records) % 2:
        raise ValueError("paired batch needs complete original/changed records")
    for first, second in zip(records[::2], records[1::2], strict=True):
        if (first["pair_id"] != second["pair_id"] or first["source_state_id"] != second["source_state_id"]
                or not torch.equal(first["performance"], second["performance"])
                or not torch.equal(first["measurements"]["lengths"], second["measurements"]["lengths"])
                or any(not torch.equal(a, b) for a, b in zip(first["times"], second["times"], strict=True))):
            raise ValueError("paired examples require identical source ancestry, MIDI and absolute frame times")


def paired_change_loss(prediction, target):
    """Each adjacent batch pair is original then changed; native targets remain detached."""
    if len(target["lengths"]) % 2 or len(target["lengths"]) == 0:
        raise ValueError("paired objective needs a nonempty even batch")
    def difference_error(predicted, expected, mask):
        if not torch.equal(mask[::2], mask[1::2]):
            raise ValueError("paired time masks differ")
        predicted, expected = torch.log1p(predicted / .0001), torch.log1p(expected.detach() / .0001)
        proposed, actual = predicted[1::2] - predicted[::2], expected[1::2] - expected[::2]
        weight = mask[::2].expand_as(actual)
        dims = tuple(range(1, actual.ndim))
        count = weight.sum(dims).clamp_min(1)
        scale = ((actual.abs() * weight).sum(dims) / count).clamp_min(.05)
        return ((proposed - actual).abs() * weight).sum(dims) / count / scale
    spectra = [difference_error(predicted, expected, mask[:, None, None])
               for predicted, expected, mask in zip(prediction["magnitudes"], target["magnitudes"], target["masks"], strict=True)]
    predicted, expected, mask = prediction["envelope"], target["envelope"], target["envelope_mask"][:, None]
    envelope = difference_error(predicted[:, :2], expected[:, :2], mask)
    side = difference_error(predicted[:, 2:], expected[:, 2:], mask)
    return torch.stack(spectra).mean(0) + .1 * envelope + .1 * side

"""Preserve note transitions and the complete envelope within a bounded spectral sample."""

import torch

from obruxo_data.midi import TempoMap

from .performances import CAPABILITIES
from .spectral import FFT_SIZES, SAMPLE_RATE


def event_frame_spec():
    return {"format": "note_event_spectra_full_envelope_v1", "spectral_frames": 32,
            "event_offsets_seconds": [0., .025, .1], "maximum_unique_note_events": 8,
            "event_selection": "nearest_valid_center_ties_choose_earlier_deduplicate",
            "remaining_frames": "uniform_without_replacement_at_least_eight",
            "envelope_frames": "all_valid", "pair_alignment": "same_MIDI_length_and_seed",
            "weighting": "equal_selected_frames_not_an_unbiased_full_spectral_estimate"}


def note_event_seconds(performance):
    performance.validate(CAPABILITIES).require_valid()
    timing = TempoMap.from_performance(performance)
    ticks = sorted({tick for note in performance.note_spans() for tick in (note.start_tick, note.end_tick)})
    if not 1 <= len(ticks) <= event_frame_spec()["maximum_unique_note_events"]:
        raise ValueError("event-frame training supports at most eight unique note start/end times")
    return [float(timing.tick_to_seconds(tick)) for tick in ticks]


def sample_event_frames(target, performance, generator):
    from .temporal_proxy import full_times
    if len(target["lengths"]) != 1:
        raise ValueError("sample one measured recording at a time")
    spec, events, times = event_frame_spec(), note_event_seconds(performance), full_times(target)
    if max(events) >= float(target["lengths"][0]) / SAMPLE_RATE:
        raise ValueError("recording must extend beyond the final note event")
    values, selected_times, masks = [], [], []
    for index, (value, mask) in enumerate(zip((*target["magnitudes"], target["envelope"]),
                                            (*target["masks"], target["envelope_mask"]), strict=True)):
        valid = torch.nonzero(mask[0], as_tuple=False).flatten()
        selected = valid
        if index < len(FFT_SIZES):
            if len(valid) < spec["spectral_frames"]:
                raise ValueError("recording has too few valid spectral frames")
            queries = times[index].new_tensor([event + offset for event in events for offset in spec["event_offsets_seconds"]])
            centers = times[index][0, valid]
            anchors = valid[(centers[:, None] - queries[None]).abs().argmin(0)].unique(sorted=True)
            remaining = valid[~torch.isin(valid, anchors)]
            count = spec["spectral_frames"] - len(anchors)
            selected = torch.cat((anchors, remaining[torch.randperm(len(remaining), generator=generator)[:count]])).sort().values
        values.append(value[..., selected].clone())
        selected_times.append(times[index][:, selected].clone())
        masks.append(torch.ones(1, len(selected), dtype=torch.bool, device=value.device))
    return ({"magnitudes": tuple(values[:3]), "masks": tuple(masks[:3]), "envelope": values[3],
             "envelope_mask": masks[3], "lengths": target["lengths"].clone()}, tuple(selected_times))

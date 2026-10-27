import copy

import numpy as np
import pytest

from obruxo_training.basic_level_diagnosis import ARMS, match_recordings, summarize


class EnergyMeter:
    """Test gain/headroom algebra only; real runs require two BS.1770 meters."""
    def measure(self, audio):
        return {"integrated_lufs": 20 * np.log10(np.sqrt(np.mean(audio.astype(np.float64) ** 2))),
                "true_peak_upper_dbtp": 20 * np.log10(np.abs(audio).max())}


def recordings():
    signal = (.1 * np.sin(np.arange(44100) * .06)).astype(np.float32)
    stereo = np.stack((signal, .7 * signal), axis=1)
    return {name: stereo * factor for name, factor in zip(("reference", *ARMS), (1., 2., .25, .7), strict=True)}


def test_all_four_clips_get_one_static_gain_and_identical_shared_target():
    audio, meter = recordings(), EnergyMeter()
    matched, proof = match_recordings(audio, meter, lambda value: meter.measure(value)["integrated_lufs"])
    for arm, original in audio.items():
        gain = 10 ** ((proof["individual_gain_db"][arm] + proof["common_gain_db"]) / 20)
        assert np.array_equal(matched[arm], (original.astype(np.float64) * gain).astype(np.float32))
        assert np.allclose(matched[arm], matched["reference"], rtol=1e-6, atol=1e-8)
    assert proof["independent_spread_lu"] < .00001


def test_high_crest_clip_lowers_all_four_targets_without_limiting_any_envelope():
    audio, meter = recordings(), EnergyMeter()
    audio["spectral"] = np.zeros_like(audio["reference"])
    audio["spectral"][100] = .8
    matched, proof = match_recordings(audio, meter, lambda value: meter.measure(value)["integrated_lufs"])
    assert proof["common_gain_db"] < 0
    assert np.count_nonzero(matched["spectral"]) == 2
    assert max(row["true_peak_upper_dbtp"] for row in proof["after"].values()) <= -1
    assert proof["independent_spread_lu"] < .00001


def test_independent_meter_disagreement_and_incomplete_comparison_are_rejected():
    audio, meter = recordings(), EnergyMeter()
    with pytest.raises(ValueError, match="tolerance"):
        match_recordings(audio, meter, lambda value: meter.measure(value)["integrated_lufs"] + .3)
    audio.pop("initial")
    with pytest.raises(ValueError, match="four complete"):
        match_recordings(audio, meter, lambda _: 0.)


def test_gain_solver_corrects_absolute_gate_changes_without_dynamic_processing():
    class GatedMeter(EnergyMeter):
        def measure(self, audio):
            result = super().measure(audio)
            result["integrated_lufs"] += 1. if np.abs(audio).max() < .05 else 0.
            return result

    audio, meter = recordings(), GatedMeter()
    matched, proof = match_recordings(audio, meter, lambda value: meter.measure(value)["integrated_lufs"])
    assert 1 < len(proof["rounds"]) <= 4 and proof["independent_spread_lu"] < .00001
    for arm, original in audio.items():
        gain = 10 ** ((proof["individual_gain_db"][arm] + proof["common_gain_db"]) / 20)
        assert np.array_equal(matched[arm], (original.astype(np.float64) * gain).astype(np.float32))


def observation(state, level):
    costs = {arm: {"spectral": level, "mrstft": level} for arm in ARMS}
    return {"group": "held_note_own", "state_id": state, "raw": copy.deepcopy(costs), "matched": copy.deepcopy(costs),
            "matching": {"before": {name: {"integrated_lufs": -23. if name == "reference" else -20.} for name in ("reference", *ARMS)}}}


def test_summary_weights_independent_patches_equally_and_keeps_level_error_separate():
    rows = [observation("a", 2.), observation("a", 4.), observation("b", 9.)]
    summary = summarize(rows)["held_note_own"]
    assert summary["independent_patches"] == 2 and summary["observations"] == 3
    assert summary["means"]["spectral"]["raw_spectral"] == 6.
    assert summary["means"]["spectral"]["absolute_loudness_error_lu"] == 3.


def test_level_matched_numeric_reversal_does_not_replace_original_scores():
    row = observation("a", 1.)
    row["raw"]["spectral"]["spectral"] = 2.
    row["matched"]["spectral"]["spectral"] = .5
    ratio = summarize([row])["held_note_own"]["sound_vs_parameter_ratios"]
    assert ratio["raw_spectral"] == 2. and ratio["matched_spectral"] == .5
    assert row["raw"]["spectral"]["spectral"] == 2.

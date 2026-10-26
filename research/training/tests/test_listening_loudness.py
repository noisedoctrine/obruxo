import json
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from obruxo_training.listening import digest, prepare, summarize, write_json
from obruxo_training.listening_loudness import POLICY, LoudnessMeter, match_triplet


@pytest.fixture(scope="module")
def meter():
    path = os.environ.get("OBRUXO_TEST_FFMPEG")
    if not path:
        pytest.skip("set OBRUXO_TEST_FFMPEG to run the real loudness-meter checks")
    return LoudnessMeter(Path(path))


def sine():
    mono = (.1 * np.sin(2 * np.pi * 1000 * np.arange(88200) / 44100)).astype(np.float32)
    return np.column_stack((mono, mono))


def test_meter_calibration_and_constant_gain_relationship(meter):
    audio = sine()
    first, quieter = meter.measure(audio), meter.measure(audio * .1)
    assert first["integrated_lufs"] == pytest.approx(-20., abs=.1)
    assert first["integrated_lufs"] - quieter["integrated_lufs"] == pytest.approx(20., abs=.01)


def test_matching_large_level_gaps_preserves_relative_waveforms_and_stereo(meter):
    base = sine()
    originals = {"reference": base, "candidate": base * .01, "control": base * 3}
    matched, report = match_triplet(originals, meter)
    assert report["maximum_spread_lu"] <= POLICY["maximum_spread_lu"]
    for role, actual in matched.items():
        gain = 10**((report["individual_gain_db"][role] + report["common_gain_db"]) / 20)
        assert actual.dtype == np.float32 and actual.shape == base.shape
        np.testing.assert_allclose(actual, originals[role].astype(np.float64) * gain, atol=1e-7)
        np.testing.assert_allclose(actual / np.max(np.abs(actual)), matched["reference"] / np.max(np.abs(matched["reference"])), atol=2e-7)
        assert report["after"][role]["true_peak_upper_dbtp"] <= -1.


def test_peak_headroom_uses_one_common_attenuation_without_limiting(meter):
    transient = np.zeros_like(sine())
    transient[10000:10004] = .8
    originals = {"reference": sine(), "candidate": transient, "control": sine() * .3}
    matched, report = match_triplet(originals, meter)
    assert report["common_gain_db"] < 0
    assert report["maximum_spread_lu"] <= .1
    assert all(row["true_peak_upper_dbtp"] <= -1 for row in report["after"].values())
    assert np.count_nonzero(matched["candidate"]) == np.count_nonzero(transient)
    assert np.max(np.abs(matched["candidate"])) < 1


def test_timbre_packet_matches_levels_by_default_and_binds_processing_to_identity(meter, tmp_path):
    assets = {}
    for role, scale in (("reference", 1), ("candidate", .1), ("control", 2)):
        path = tmp_path / f"{role}.wav"
        wavfile.write(path, 44100, sine() * scale)
        assets[role] = {"path": path.name, "sha256": digest(path)}
    manifest = tmp_path / "inputs.json"
    write_json(manifest, {"format": "blind_pairwise_listening_inputs_v1", "cases": [{"family_id": "family", "role": "reference", **assets}]})
    output = tmp_path / "matched"
    result = prepare(manifest, output, seed=7, ffmpeg=meter.path)
    key = json.loads((output / "answer-key.json").read_bytes())
    assert key["packet"]["condition"]["purpose"] == "timbre"
    assert key["processing"]["cases"][0]["maximum_spread_lu"] <= .1
    assert all(digest(tmp_path / row["path"]) == row["sha256"] for row in assets.values())
    response = tmp_path / "responses.json"
    write_json(response, {"format": "blind_pairwise_listening_responses_v1", "packet_id": result["packet_id"],
                          "judgments": [{"trial_id": "trial-001", "preferred": "tie", "usable": "both"}]})
    assert summarize(output / "answer-key.json", response)["condition"]["purpose"] == "timbre"
    key["processing"]["cases"][0]["common_gain_db"] -= 1
    write_json(output / "answer-key.json", key)
    with pytest.raises(ValueError, match="packet"):
        summarize(output / "answer-key.json", response)
    raw = prepare(manifest, tmp_path / "raw", seed=7, purpose="level_accuracy")
    assert raw["packet_id"] != result["packet_id"]


def test_unmeasurable_audio_and_missing_meter_do_not_silently_produce_tone_packets(meter, tmp_path):
    with pytest.raises(ValueError):
        meter.measure(np.zeros_like(sine()))
    with pytest.raises(ValueError, match="three aligned"):
        match_triplet({"reference": sine()}, meter)
    manifest = tmp_path / "inputs.json"
    write_json(manifest, {"format": "blind_pairwise_listening_inputs_v1", "cases": [{}]})
    with pytest.raises(ValueError, match="executable is missing"):
        prepare(manifest, tmp_path / "bad", ffmpeg=tmp_path / "missing.exe")
    assert not (tmp_path / "bad").exists()

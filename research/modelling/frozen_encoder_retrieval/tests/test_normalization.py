from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))

from frozen_retrieval.normalization import (  # noqa: E402
    ACTIVE_FLOOR_DB, MAX_NORMALIZED_PEAK, MIN_ACTIVE_RMS, WINDOW_SECONDS,
    measure_active_rms, normalize_active_rms, window_size_for_sample_rate,
)


def _reference_active_rms(samples: np.ndarray, sample_rate: int) -> float:
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    window = window_size_for_sample_rate(sample_rate)
    nonzero = np.flatnonzero(np.any(values != 0.0, axis=1))
    if not len(nonzero):
        raise ValueError("silence")
    energies = []
    for start in range(int(nonzero[0]) - window + 1, int(nonzero[-1]) + 1):
        clip = values[max(0, start):min(len(values), start + window)]
        energies.append(float(np.square(clip).mean(axis=1).sum() / window))
    energies = np.asarray(energies)
    peak = energies.max()
    return float(np.sqrt(energies[energies >= peak * 10 ** (ACTIVE_FLOOR_DB / 10)].mean()))


@pytest.mark.parametrize("leading,trailing", [(0, 0), (1, 1), (17, 913), (882, 2205)])
def test_padding_by_arbitrary_zero_counts_does_not_change_gain(leading, trailing):
    rng = np.random.default_rng(164)
    audio = (rng.normal(0.0, 0.03, (4097, 2))).astype(np.float32)
    base, base_info = normalize_active_rms(audio, 44_100, 0.08)
    padded = np.pad(audio, ((leading, trailing), (0, 0)))
    actual, info = normalize_active_rms(padded, 44_100, 0.08)
    assert info["applied_gain"] == pytest.approx(base_info["applied_gain"], rel=1e-12)
    np.testing.assert_array_equal(actual[leading:leading + len(audio)], base)


def test_sliding_windows_cover_transients_on_either_side_of_old_frame_boundary():
    left = np.zeros((4_000, 2), dtype=np.float32)
    left[881] = (0.8, 0.2)
    shifted = np.pad(left, ((1, 0), (0, 0)))
    left_normalized, left_info = normalize_active_rms(left, 44_100, 0.02)
    shifted_normalized, shifted_info = normalize_active_rms(shifted, 44_100, 0.02)
    assert left_info["window_size_samples"] == 882
    assert shifted_info["applied_gain"] == pytest.approx(left_info["applied_gain"], rel=1e-12)
    np.testing.assert_array_equal(shifted_normalized[1:], left_normalized)
    assert left_info["active_window_count"] > 1


def test_stereo_rms_uses_energy_from_both_channels():
    audio = np.zeros((50_000, 2), dtype=np.float32)
    audio[:, 0] = 0.1
    measured = measure_active_rms(audio, 44_100)
    assert measured["active_rms"] == pytest.approx(_reference_active_rms(audio, 44_100), rel=1e-12)
    assert measured["active_rms"] == pytest.approx(0.1 / np.sqrt(2), rel=0.02)


def test_relative_gate_excludes_a_long_quiet_release_tail():
    body = np.ones(3_000, dtype=np.float32) * 0.2
    release = np.ones(44_100, dtype=np.float32) * 0.001
    base = measure_active_rms(body, 44_100)
    with_tail = measure_active_rms(np.concatenate((body, release)), 44_100)
    assert base["active_rms"] == pytest.approx(with_tail["active_rms"], rel=0.01)
    assert with_tail["active_window_count"] < len(body) + len(release)


def test_very_quiet_finite_signal_is_scale_invariant():
    audio = np.zeros(1_500, dtype=np.float64)
    audio[700:705] = 1e-6
    reference, info = normalize_active_rms(audio, 44_100, 0.04)
    tiny, tiny_info = normalize_active_rms(audio * 1e-3, 44_100, 0.04)
    assert tiny_info["applied_gain"] / info["applied_gain"] == pytest.approx(1e3, rel=1e-10)
    np.testing.assert_allclose(reference, tiny, rtol=2e-7, atol=1e-8)


def test_nonfinite_silent_near_zero_and_unsafe_fixed_target_fail_explicitly():
    with pytest.raises(ValueError, match="nonfinite"):
        measure_active_rms(np.array([0.0, np.nan]), 44_100)
    with pytest.raises(ValueError, match="silent"):
        measure_active_rms(np.zeros(1_000), 44_100)
    with pytest.raises(ValueError, match="minimum active-RMS floor"):
        normalize_active_rms(np.full(1_000, MIN_ACTIVE_RMS / 10), 44_100, 0.1)
    with pytest.raises(ValueError, match="peak ceiling"):
        normalize_active_rms(np.array([1.0, 0.0], dtype=np.float32), 44_100, 0.95)


def test_out_of_unit_range_float_input_is_reported_and_safely_scaled_without_clipping():
    audio = np.full((2_000, 2), 1.4, dtype=np.float32)
    normalized, info = normalize_active_rms(audio, 44_100, 0.1)
    assert info["source_peak_above_unit_range"] is True
    assert info["output_peak"] <= MAX_NORMALIZED_PEAK
    assert float(np.max(np.abs(normalized))) <= MAX_NORMALIZED_PEAK
    assert info["active_rms"] > 0.9


def test_numba_measurement_matches_simple_zero_extended_reference():
    rng = np.random.default_rng(18)
    audio = np.zeros((12_000, 2), dtype=np.float64)
    audio[533:10_020] = rng.normal(0.0, 0.025, (9_487, 2))
    audio[9_900:10_020] *= 0.001
    assert WINDOW_SECONDS == 0.02
    assert measure_active_rms(audio, 44_100)["active_rms"] == pytest.approx(
        _reference_active_rms(audio, 44_100), rel=1e-12)


def test_encoder_audio_normalizes_stereo_before_downmix_and_resampling(tmp_path):
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    from frozen_retrieval.encoders import FrozenAdapter

    rng = np.random.default_rng(164)
    audio = rng.normal(0.0, 0.08, (132_300, 2)).astype(np.float32)
    audio[:, 1] *= 0.35
    wav = tmp_path / "stereo.wav"
    wavfile.write(wav, 44_100, audio)
    adapter = FrozenAdapter("clap_music", tmp_path / "unused.pt", {}, "unused", "cpu", None, None, np)
    target = 0.05
    actual, details = adapter._audio(wav, active_rms_target=target)
    normalized_stereo, expected_details = normalize_active_rms(audio, 44_100, target)
    expected = resample_poly(normalized_stereo.mean(axis=1), 160, 147).astype(np.float32)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-7)
    assert details["active_rms"] == pytest.approx(expected_details["active_rms"], rel=1e-12)
    assert details["target_active_rms"] == target

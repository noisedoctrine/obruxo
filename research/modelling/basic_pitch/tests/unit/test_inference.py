from __future__ import annotations

import numpy as np
from obruxo_basic_pitch.constants import (
    AUDIO_N_SAMPLES,
    AUDIO_SAMPLE_RATE,
    FFT_HOP,
    OVERLAP,
)
from obruxo_basic_pitch.inference import _window_audio, unwrap_window_outputs


def test_final_window_stays_on_fixed_hop_grid_and_zero_pads_tail() -> None:
    samples = np.arange(AUDIO_N_SAMPLES + 100, dtype=np.float32)
    windows = _window_audio(samples)
    hop = AUDIO_N_SAMPLES - OVERLAP * FFT_HOP
    leading = (OVERLAP * FFT_HOP) // 2

    assert windows.shape == (2, AUDIO_N_SAMPLES, 1)
    expected_tail = np.pad(samples[hop - leading :], (0, AUDIO_N_SAMPLES - (samples.shape[0] - hop + leading)))
    np.testing.assert_array_equal(windows[1, :, 0], expected_tail)
    assert windows[1, 0, 0] == samples[hop - leading]
    assert windows[1, -1, 0] == 0.0


def test_unwrap_crops_each_window_then_trims_once_to_original_duration() -> None:
    output = {
        "note": np.arange(2 * 172 * 88, dtype=np.float32).reshape(2, 172, 88),
        "onset": np.zeros((2, 172, 88), dtype=np.float32),
        "contour": np.zeros((2, 172, 264), dtype=np.float32),
    }

    original_sample_count = AUDIO_SAMPLE_RATE * 2 + 100
    unwrapped = unwrap_window_outputs(output, original_sample_count=original_sample_count)
    expected_frames = original_sample_count * (AUDIO_SAMPLE_RATE // FFT_HOP) // AUDIO_SAMPLE_RATE

    assert {name: value.shape for name, value in unwrapped.items()} == {
        "note": (expected_frames, 88),
        "onset": (expected_frames, 88),
        "contour": (expected_frames, 264),
    }
    np.testing.assert_array_equal(unwrapped["note"][0], output["note"][0, 15])
    np.testing.assert_array_equal(unwrapped["note"][141], output["note"][0, 156])
    np.testing.assert_array_equal(unwrapped["note"][142], output["note"][1, 15])

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from obruxo_basic_pitch.constants import (
    AUDIO_N_SAMPLES,
    AUDIO_SAMPLE_RATE,
    FFT_HOP,
    OVERLAP,
)
from obruxo_basic_pitch.inference import (
    _window_audio,
    prepare_wav,
    unwrap_window_outputs,
)
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[2]


def test_prepare_wav_reads_and_resamples_without_source_side_effects() -> None:
    path = ROOT / "outputs" / f".test-source-{os.getpid()}.wav"
    assert not path.exists()
    samples = np.stack((np.arange(2_205, dtype=np.int16), np.zeros(2_205, dtype=np.int16)), axis=1)
    try:
        wavfile.write(path, 11_025, samples)
        before = path.stat()
        prepared = prepare_wav(path)
        after = path.stat()
        assert after.st_size == before.st_size
        assert after.st_mtime_ns == before.st_mtime_ns
        assert prepared.sample_rate == AUDIO_SAMPLE_RATE
        assert prepared.original_sample_count == 4_410
        assert prepared.windows.dtype == np.float32
        assert prepared.windows.shape[1:] == (AUDIO_N_SAMPLES, 1)
        assert not path.with_suffix(".wav.tmp").exists()
    finally:
        if path.exists():
            path.unlink()


def test_unwrap_window_outputs_drops_overlap_and_trims_duration() -> None:
    output = {
        "note": np.arange(2 * 172 * 88, dtype=np.float32).reshape(2, 172, 88),
        "onset": np.zeros((2, 172, 88), dtype=np.float32),
        "contour": np.zeros((2, 172, 264), dtype=np.float32),
    }

    unwrapped = unwrap_window_outputs(output, original_sample_count=AUDIO_SAMPLE_RATE * 2)

    assert {name: value.shape for name, value in unwrapped.items()} == {
        "note": (172, 88),
        "onset": (172, 88),
        "contour": (172, 264),
    }
    np.testing.assert_array_equal(unwrapped["note"][0], output["note"][0, 15])
    np.testing.assert_array_equal(unwrapped["note"][-1], output["note"][1, 44])


def test_final_window_stays_on_fixed_hop_grid_and_zero_pads_tail() -> None:
    samples = np.arange(AUDIO_N_SAMPLES + 100, dtype=np.float32)
    windows = _window_audio(samples)
    hop = AUDIO_N_SAMPLES - OVERLAP * FFT_HOP
    leading = (OVERLAP * FFT_HOP) // 2

    assert windows.shape == (2, AUDIO_N_SAMPLES, 1)
    expected_tail = np.pad(
        samples[hop - leading :],
        (0, AUDIO_N_SAMPLES - (samples.shape[0] - hop + leading)),
    )
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
    unwrapped = unwrap_window_outputs(
        output, original_sample_count=original_sample_count
    )
    expected_frames = (
        original_sample_count * (AUDIO_SAMPLE_RATE // FFT_HOP) // AUDIO_SAMPLE_RATE
    )

    assert {name: value.shape for name, value in unwrapped.items()} == {
        "note": (expected_frames, 88),
        "onset": (expected_frames, 88),
        "contour": (expected_frames, 264),
    }
    np.testing.assert_array_equal(unwrapped["note"][0], output["note"][0, 15])
    np.testing.assert_array_equal(unwrapped["note"][141], output["note"][0, 156])
    np.testing.assert_array_equal(unwrapped["note"][142], output["note"][1, 15])

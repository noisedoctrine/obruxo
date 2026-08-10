"""Shared read-only audio preparation and posterior-window unwrapping."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile

from .constants import ANNOTATIONS_FPS, AUDIO_N_SAMPLES, AUDIO_SAMPLE_RATE


@dataclass(frozen=True)
class PreparedAudio:
    sample_rate: int
    original_sample_count: int
    audio_seconds: float
    windows: np.ndarray


def _window_audio(samples: np.ndarray) -> np.ndarray:
    padded = np.concatenate((np.zeros(3840, dtype=np.float32), samples))
    hop = AUDIO_N_SAMPLES - 30 * 256
    starts = list(range(0, max(1, padded.shape[0] - AUDIO_N_SAMPLES + 1), hop))
    if starts[-1] + AUDIO_N_SAMPLES < padded.shape[0]:
        starts.append(padded.shape[0] - AUDIO_N_SAMPLES)
    windows = []
    for start in starts:
        window = padded[start : start + AUDIO_N_SAMPLES]
        if window.shape[0] < AUDIO_N_SAMPLES:
            window = np.pad(window, (0, AUDIO_N_SAMPLES - window.shape[0]))
        windows.append(window)
    return np.ascontiguousarray(np.stack(windows, axis=0)[:, :, None], dtype=np.float32)


def prepare_wav(path: Path) -> PreparedAudio:
    """Read a WAV source without writing beside it and prepare shared windows."""
    source = Path(path).resolve(strict=True)
    sample_rate, audio = wavfile.read(source)
    decoded = np.asarray(audio)
    if decoded.ndim == 2:
        samples = decoded.astype(np.float32).mean(axis=1)
    else:
        samples = decoded.astype(np.float32)
    if np.issubdtype(decoded.dtype, np.integer):
        samples /= np.iinfo(decoded.dtype).max
    if sample_rate != AUDIO_SAMPLE_RATE:
        gcd = np.gcd(sample_rate, AUDIO_SAMPLE_RATE)
        samples = signal.resample_poly(samples, AUDIO_SAMPLE_RATE // gcd, sample_rate // gcd).astype(np.float32)
    samples = np.ascontiguousarray(samples, dtype=np.float32)
    return PreparedAudio(
        sample_rate=AUDIO_SAMPLE_RATE,
        original_sample_count=int(samples.shape[0]),
        audio_seconds=float(samples.shape[0] / AUDIO_SAMPLE_RATE),
        windows=_window_audio(samples),
    )


def unwrap_window_outputs(
    output: Mapping[str, np.ndarray],
    *,
    original_sample_count: int,
) -> dict[str, np.ndarray]:
    """Drop the 15-frame window margins and trim to the source duration."""
    target_frames = int(original_sample_count * ANNOTATIONS_FPS // AUDIO_SAMPLE_RATE)
    unwrapped = {}
    for name, value in output.items():
        array = np.asarray(value)
        if array.ndim != 3 or array.shape[1] != 172:
            raise ValueError(f"expected windowed posterior {name} [N,172,F], got {array.shape}")
        unwrapped[name] = np.ascontiguousarray(array[:, 15:-15, :].reshape(-1, array.shape[2])[:target_frames])
    return unwrapped

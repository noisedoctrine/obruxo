"""Thin Timbre-Trap frame-output normalization preserving paper semantics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..artifacts import (
    ArtifactUnavailable,
    ModelSpec,
    verify_checkout,
    verify_checkpoint,
)
from ..types import FramePitchPrediction, TranscriptionOutput

PAPER_ACTIVATION_THRESHOLD = 0.5
PITCH_TOLERANCE_CENTS = 50.0


def peak_pick_then_threshold(activations: np.ndarray) -> np.ndarray:
    """Apply native-frequency peak picking, then the fixed paper threshold."""
    values = np.asarray(activations, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("Timbre-Trap activations must have shape [frames, native_pitches]")
    if not np.isfinite(values).all():
        raise ValueError("Timbre-Trap activations must be finite")
    if values.shape[1] == 0:
        return np.zeros(values.shape, dtype=bool)
    left = np.concatenate((values[:, :1], values[:, :-1]), axis=1)
    right = np.concatenate((values[:, 1:], values[:, -1:]), axis=1)
    peaks = (values >= left) & (values >= right)
    return peaks & (values >= PAPER_ACTIVATION_THRESHOLD)


def _nearest_native_frame(native_times: np.ndarray, time_seconds: float) -> int:
    right = int(np.searchsorted(native_times, time_seconds, side="left"))
    if right == 0:
        return 0
    if right == native_times.size:
        return native_times.size - 1
    left = right - 1
    left_distance = abs(time_seconds - float(native_times[left]))
    right_distance = abs(float(native_times[right]) - time_seconds)
    return left if left_distance <= right_distance else right


def native_to_common_grid(
    native_times: np.ndarray,
    native_pitch_centers_hz: np.ndarray,
    native_active: np.ndarray,
    common_times: np.ndarray,
) -> FramePitchPrediction:
    """Map decided native peaks to #25's 88-pitch grid without re-thresholding."""
    times = np.asarray(native_times, dtype=np.float64)
    centers = np.asarray(native_pitch_centers_hz, dtype=np.float64)
    active = np.asarray(native_active, dtype=bool)
    common = np.asarray(common_times, dtype=np.float64)
    if times.ndim != 1 or centers.ndim != 1 or active.shape != (times.size, centers.size):
        raise ValueError("native Timbre-Trap output has inconsistent shapes")
    if common.ndim != 1 or not np.isfinite(times).all() or not np.isfinite(common).all():
        raise ValueError("native and common frame times must be finite vectors")
    if times.size and np.any(np.diff(times) < 0) or common.size and np.any(np.diff(common) < 0):
        raise ValueError("frame times must be non-decreasing")
    mapped: dict[int, list[int]] = {}
    for index, frequency in enumerate(centers):
        if not np.isfinite(frequency) or frequency <= 0:
            continue
        midi = 69.0 + 12.0 * np.log2(frequency / 440.0)
        nearest = int(np.rint(midi))
        if 21 <= nearest <= 108 and abs(midi - nearest) * 100.0 <= PITCH_TOLERANCE_CENTS:
            mapped.setdefault(nearest, []).append(index)
    result = np.zeros((common.size, 88), dtype=bool)
    for common_index, time_seconds in enumerate(common):
        if times.size == 0:
            continue
        native_index = _nearest_native_frame(times, float(time_seconds))
        for midi, native_indices in mapped.items():
            result[common_index, midi - 21] = bool(np.any(active[native_index, native_indices]))
    return FramePitchPrediction(common, result)


def normalize_frame_output(
    native_times: np.ndarray,
    native_pitch_centers_hz: np.ndarray,
    activations: np.ndarray,
    common_times: np.ndarray,
) -> TranscriptionOutput:
    active = peak_pick_then_threshold(activations)
    return TranscriptionOutput(None, native_to_common_grid(native_times, native_pitch_centers_hz, active, common_times))


class TimbreTrapAdapter:
    def __init__(self, spec: ModelSpec, source_root: Path | None, checkpoint: Path | None) -> None:
        self.spec = spec
        self.source_root = None if source_root is None else Path(source_root)
        self.checkpoint = None if checkpoint is None else Path(checkpoint)

    def preflight(self) -> None:
        if self.source_root is None or self.checkpoint is None:
            raise ArtifactUnavailable("dependency_unavailable")
        verify_checkout(self.spec, self.source_root)
        verify_checkpoint(self.spec, self.checkpoint)

    def load(self) -> None:
        self.preflight()
        try:
            __import__("torch")
            __import__("timbre_trap")
        except (ImportError, OSError) as exc:
            raise ArtifactUnavailable("dependency_unavailable") from exc

    def transcribe(self, _audio: Path) -> Any:
        raise ArtifactUnavailable("official Timbre-Trap runtime is not present")

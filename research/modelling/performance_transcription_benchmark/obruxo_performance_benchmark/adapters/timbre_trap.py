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
from ..types import (
    FramePitchPrediction,
    TranscriptionOutput,
    common_frame_count,
    common_frame_times,
)

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
        self.model: Any | None = None
        self.bound_model: Any | None = None

    def preflight(self) -> None:
        if self.source_root is None or self.checkpoint is None:
            raise ArtifactUnavailable("dependency_unavailable")
        verify_checkout(self.spec, self.source_root)
        verify_checkpoint(self.spec, self.checkpoint)

    @property
    def active_model(self) -> Any | None:
        return self.bound_model if self.bound_model is not None else self.model

    def load(self, device: str = "cpu") -> None:
        self.preflight()
        try:
            import sys

            import torch

            root = self.source_root.resolve(strict=True)
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            from timbre_trap.framework.modules import TimbreTrap
        except (ImportError, OSError) as exc:
            raise ArtifactUnavailable("dependency_unavailable") from exc
        if device not in {"cpu", "xpu"}:
            raise ValueError("Timbre-Trap supports only the requested CPU/XPU routes")
        torch_device = torch.device(device)
        settings = dict(self.spec.stock_inference)
        model = TimbreTrap(
            sample_rate=int(settings["sample_rate"]),
            n_octaves=int(settings["cqt_octaves"]),
            bins_per_octave=int(settings["cqt_bins_per_octave"]),
            secs_per_block=int(settings["secs_per_block"]),
            latent_size=int(settings["latent_size"]),
            model_complexity=int(settings["model_complexity"]),
            skip_connections=bool(settings["skip_connections"]),
        )
        state = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
            state = state["state_dict"]
        model.load_state_dict(state, strict=True)
        model.to(torch_device).eval()
        self.model = model
        self.bound_model = model

    def bind_model(self, model: Any) -> None:
        self.bound_model = model
        if hasattr(model, "eval"):
            model.eval()

    @staticmethod
    def _read_audio(path: Path, sample_rate: int) -> Any:
        from scipy import signal
        from scipy.io import wavfile

        source_rate, values = wavfile.read(Path(path).resolve(strict=True))
        raw = np.asarray(values)
        samples = raw.astype(np.float32)
        if raw.ndim == 2:
            samples = samples.mean(axis=1)
        if np.issubdtype(raw.dtype, np.integer):
            samples /= np.iinfo(raw.dtype).max
        if source_rate != sample_rate:
            gcd = np.gcd(source_rate, sample_rate)
            samples = signal.resample_poly(samples, sample_rate // gcd, source_rate // gcd).astype(np.float32)
        if samples.ndim != 1 or not np.isfinite(samples).all():
            raise ValueError("audio must be finite mono samples")
        return samples

    def transcribe(self, audio: Path) -> TranscriptionOutput:
        if self.active_model is None:
            self.load()
        import torch

        model = self.active_model
        sample_rate = int(self.spec.stock_inference["sample_rate"])
        samples = self._read_audio(Path(audio), sample_rate)
        with torch.inference_mode():
            activations = model.transcribe(torch.from_numpy(samples[None, :]).to(next(model.parameters()).device))
        values = activations.detach().float().cpu().numpy()
        if values.ndim == 3:
            values = values[0].T
        if values.ndim != 2:
            raise ValueError(f"Timbre-Trap returned unexpected activations {values.shape}")
        sli_cq = getattr(model, "sliCQ", None)
        if sli_cq is None:
            raise ValueError("Timbre-Trap model does not expose its native SliCQT grid")
        native_times = np.asarray(sli_cq.get_times(values.shape[0]), dtype=np.float64)
        native_midi = np.asarray(sli_cq.get_midi_freqs(), dtype=np.float64)
        native_hz = 440.0 * np.power(2.0, (native_midi - 69.0) / 12.0)
        frame_count = self._frame_count(Path(audio))
        return normalize_frame_output(native_times, native_hz, values, common_frame_times(frame_count))

    @staticmethod
    def _frame_count(path: Path) -> int:
        from scipy.io import wavfile

        source_rate, values = wavfile.read(Path(path).resolve(strict=True))
        count = int(np.asarray(values).shape[0])
        sample_count = round(count * 22050 / source_rate) if source_rate != 22050 else count
        return common_frame_count(sample_count)

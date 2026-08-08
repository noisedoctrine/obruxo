from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, TYPE_CHECKING

from obruxo_data.errors import Diagnostic, Severity

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class AudioQualityConfig:
    silence_rms: float = 1e-6
    clipping_threshold: float = 1.0
    tail_seconds: float = 0.25
    tail_rms_warning: float = 1e-3


@dataclass(frozen=True)
class RepeatabilityTolerance:
    rms_relative: float = 0.01
    peak_absolute: float = 0.10
    log_spectral_rmse: float = 0.35


def audio_float32_sha256(audio: "np.ndarray") -> str:
    import numpy as np

    buffer = np.ascontiguousarray(audio.astype("<f4", copy=False)).tobytes()
    return hashlib.sha256(buffer).hexdigest()


def compare_audio(first: "np.ndarray", second: "np.ndarray", *,
                  tolerance: RepeatabilityTolerance | None = None) -> dict[str, Any]:
    import numpy as np

    resolved = tolerance or RepeatabilityTolerance()
    if first.shape != second.shape:
        return {"within_tolerance": False, "first_shape": list(first.shape), "second_shape": list(second.shape)}
    left = first.astype(np.float64, copy=False)
    right = second.astype(np.float64, copy=False)
    left_rms = float(np.sqrt(np.mean(left * left))) if left.size else 0.0
    right_rms = float(np.sqrt(np.mean(right * right))) if right.size else 0.0
    rms_relative = abs(left_rms - right_rms) / max(left_rms, right_rms, 1e-12)
    waveform_rmse = float(np.sqrt(np.mean((left - right) ** 2))) if left.size else 0.0
    waveform_relative_rmse = waveform_rmse / max(left_rms, right_rms, 1e-12)
    peak_absolute = abs(float(np.max(np.abs(left), initial=0.0)) - float(np.max(np.abs(right), initial=0.0)))
    left_spectrum = np.abs(np.fft.rfft(left, axis=0))
    right_spectrum = np.abs(np.fft.rfft(right, axis=0))
    log_spectral_rmse = float(np.sqrt(np.mean((np.log1p(left_spectrum) - np.log1p(right_spectrum)) ** 2)))
    within = (
        rms_relative <= resolved.rms_relative
        and peak_absolute <= resolved.peak_absolute
        and log_spectral_rmse <= resolved.log_spectral_rmse
    )
    return {
        "within_tolerance": within,
        "rms_relative": rms_relative,
        "waveform_rmse": waveform_rmse,
        "waveform_relative_rmse": waveform_relative_rmse,
        "peak_absolute": peak_absolute,
        "log_spectral_rmse": log_spectral_rmse,
        "tolerance": {
            "rms_relative": resolved.rms_relative,
            "peak_absolute": resolved.peak_absolute,
            "log_spectral_rmse": resolved.log_spectral_rmse,
        },
    }


def analyze_audio(audio: "np.ndarray", *, sample_rate: int, expected_frames: int, expected_channels: int,
                  config: AudioQualityConfig | None = None) -> tuple[dict[str, Any], tuple[Diagnostic, ...]]:
    import numpy as np

    resolved = config or AudioQualityConfig()
    diagnostics = []
    actual_frames = int(audio.shape[0]) if audio.ndim >= 1 else 0
    actual_channels = int(audio.shape[1]) if audio.ndim == 2 else 0
    finite = bool(np.isfinite(audio).all())
    safe = np.nan_to_num(audio.astype(np.float64, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    absolute = np.abs(safe)
    peak = float(absolute.max(initial=0.0))
    clipping_count = int(np.count_nonzero(absolute >= resolved.clipping_threshold))
    rms = float(np.sqrt(np.mean(safe * safe))) if safe.size else 0.0
    dc_offset = [float(item) for item in np.mean(safe, axis=0)] if safe.ndim == 2 and safe.size else []
    tail_frames = min(actual_frames, round(resolved.tail_seconds * sample_rate))
    tail = safe[-tail_frames:] if tail_frames else safe[:0]
    tail_rms = float(np.sqrt(np.mean(tail * tail))) if tail.size else 0.0
    digest = audio_float32_sha256(audio)
    if actual_frames != expected_frames:
        diagnostics.append(Diagnostic("audio.frames", Severity.ERROR, "rendered frame count does not match request", context={"expected": expected_frames, "actual": actual_frames}))
    if actual_channels != expected_channels:
        diagnostics.append(Diagnostic("audio.channels", Severity.ERROR, "rendered channel count does not match request", context={"expected": expected_channels, "actual": actual_channels}))
    if not finite:
        diagnostics.append(Diagnostic("audio.non_finite", Severity.ERROR, "render contains NaN or infinite samples"))
    if rms <= resolved.silence_rms:
        diagnostics.append(Diagnostic("audio.silence", Severity.WARNING, "render is likely silent", context={"rms": rms}))
    if clipping_count:
        diagnostics.append(Diagnostic("audio.clipping", Severity.WARNING, "render reaches or exceeds the clipping threshold", context={"count": clipping_count, "peak": peak}))
    if tail_rms > resolved.tail_rms_warning:
        diagnostics.append(Diagnostic("audio.tail_truncation", Severity.WARNING, "render tail may be truncated", context={"tail_rms": tail_rms}))
    qa = {
        "expected_frames": expected_frames,
        "actual_frames": actual_frames,
        "expected_channels": expected_channels,
        "actual_channels": actual_channels,
        "finite": finite,
        "peak": peak,
        "clipping_count": clipping_count,
        "rms": rms,
        "silence_threshold": resolved.silence_rms,
        "dc_offset": dc_offset,
        "tail_frames": tail_frames,
        "tail_rms": tail_rms,
        "tail_rms_warning": resolved.tail_rms_warning,
        "audio_float32_sha256": digest,
    }
    return qa, tuple(diagnostics)

from __future__ import annotations

import numpy as np
from scipy import signal

CONFIG = {"version": "quick_metrics_v1", "fft_sizes": [512, 1024, 2048], "hop_fraction": 0.25,
          "window": "hann_periodic", "stft_scaling": "spectrum", "boundary": None, "padded": False,
          "epsilon": 1e-7, "envelope_window": 1024, "envelope_hop": 256,
          "normalization": "none", "tolerance_absolute": 1e-10, "tolerance_relative": 1e-6}


def distances(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2 or len(left) < 2048:
        raise ValueError("requires equal [frames, channels] arrays with at least 2048 frames")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("nonfinite audio cannot be scored")
    errors = []
    epsilon = CONFIG["epsilon"]
    for size in CONFIG["fft_sizes"]:
        for channel in range(left.shape[1]):
            kwargs = dict(window="hann", nperseg=size, noverlap=3 * size // 4, boundary=None, padded=False, scaling="spectrum")
            a = np.abs(signal.stft(left[:, channel], **kwargs)[2])
            b = np.abs(signal.stft(right[:, channel], **kwargs)[2])
            convergence = np.linalg.norm(a - b) / max(np.linalg.norm(a), epsilon)
            errors.append(convergence + np.mean(np.abs(np.log(a + epsilon) - np.log(b + epsilon))))
    def envelope(audio):
        windows = np.lib.stride_tricks.sliding_window_view(audio, 1024, axis=0)[::256]
        return np.sqrt(np.mean(windows**2, axis=-1))
    return {"waveform_rmse": float(np.sqrt(np.mean((left - right)**2))),
            "spectral_distance": float(np.mean(errors)),
            "envelope_mae": float(np.mean(np.abs(envelope(left) - envelope(right))))}


def separation(unchanged: list[float], changed: list[float]) -> dict:
    if len(unchanged) != 3 or len(changed) != 2 or not np.isfinite(unchanged + changed).all():
        return {"interpretation": "insufficient_valid_comparisons"}
    maximum = max(unchanged + changed)
    tolerance = CONFIG["tolerance_absolute"] + CONFIG["tolerance_relative"] * maximum
    return {"unchanged_min": min(unchanged), "unchanged_max": max(unchanged),
            "changed_min": min(changed), "changed_max": max(changed), "numerical_tolerance": tolerance,
            "interpretation": "observed_separation" if min(changed) > max(unchanged) + tolerance else "overlap_inconclusive"}

"""Vital-compatible sampling of drawable LFO curves.

The interpolation follows Vital's LineGenerator: optional smoothstep followed by
the clipped exponential power scale, then linear interpolation between y values.
We retain the serialized/display y orientation; inversion in Vital's internal
buffer does not affect reconstruction distances.
"""

from __future__ import annotations

import numpy as np

from .model import LfoShape


def power_scale(t: np.ndarray, power: np.ndarray) -> np.ndarray:
    t, power = np.broadcast_arrays(
        np.asarray(t, dtype=np.float64), np.asarray(power, dtype=np.float64)
    )
    result = t.copy()
    curved = np.abs(power) >= 0.01
    if np.any(curved):
        numerator = np.expm1(power[curved] * t[curved])
        denominator = np.expm1(power[curved])
        result[curved] = numerator / denominator
    return np.clip(result, 0.0, 1.0)


def sample_shape(shape: LfoShape, resolution: int = 1024) -> np.ndarray:
    """Sample a shape at uniformly spaced phases in [0, 1)."""
    if resolution < 2:
        raise ValueError("resolution must be at least 2")

    phase = np.arange(resolution, dtype=np.float64) / resolution
    x = shape.points[:, 0]
    y = shape.points[:, 1]

    # Vital treats a zero-width segment as a jump to its second point. Using the
    # first point at or to the right of a phase reproduces that boundary choice.
    right = np.searchsorted(x, phase, side="left")
    right = np.clip(right, 1, len(x) - 1)
    left = right - 1
    width = x[right] - x[left]
    local = np.divide(
        phase - x[left], width, out=np.ones_like(phase), where=width > 0.0
    )
    local = np.clip(local, 0.0, 1.0)
    if shape.smooth:
        local = local * local * (3.0 - 2.0 * local)
    local = power_scale(local, shape.powers[left])
    return y[left] + local * (y[right] - y[left])


def resample_curve(values: np.ndarray, size: int) -> np.ndarray:
    """Periodically resample a dense curve to a new fixed grid."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("values must be a one-dimensional sampled curve")
    if size < 2:
        raise ValueError("size must be at least 2")
    source_x = np.arange(len(values) + 1, dtype=np.float64) / len(values)
    source_y = np.concatenate([values, values[:1]])
    target_x = np.arange(size, dtype=np.float64) / size
    return np.interp(target_x, source_x, source_y)


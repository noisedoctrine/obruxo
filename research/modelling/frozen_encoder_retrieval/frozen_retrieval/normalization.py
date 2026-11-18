from __future__ import annotations

import math

import numpy as np
from numba import njit


NORMALIZATION_VERSION = "active_rms_sliding20ms_relative_minus40db_v1"
WINDOW_SECONDS = 0.020
ACTIVE_FLOOR_DB = -40.0
MIN_ACTIVE_RMS = 1e-12
MAX_NORMALIZED_PEAK = 0.95


@njit(cache=True)
def _measure_active_rms(samples: np.ndarray, window_size: int) -> tuple[float, float, int, int, int]:
    frames, channels = samples.shape
    first, last = frames, -1
    peak_sample = 0.0
    for frame in range(frames):
        sample_energy = 0.0
        nonzero = False
        for channel in range(channels):
            value = samples[frame, channel]
            if value != 0.0:
                nonzero = True
            sample_energy += value * value
            magnitude = abs(value)
            if magnitude > peak_sample:
                peak_sample = magnitude
        if nonzero:
            if first == frames:
                first = frame
            last = frame

    if last < 0:
        return 0.0, peak_sample, 0, first, last

    starts = first - window_size + 1
    stop = last + 1
    rolling_sum = 0.0
    peak_energy = 0.0
    for start in range(starts, stop):
        entering = start + window_size - 1
        if 0 <= entering < frames:
            for channel in range(channels):
                value = samples[entering, channel]
                rolling_sum += value * value / channels
        leaving = start - 1
        if 0 <= leaving < frames:
            for channel in range(channels):
                value = samples[leaving, channel]
                rolling_sum -= value * value / channels
        energy = max(0.0, rolling_sum / window_size)
        if energy > peak_energy:
            peak_energy = energy

    if peak_energy <= 0.0:
        return 0.0, peak_sample, 0, first, last
    threshold = peak_energy * 1e-4
    active_energy_sum = 0.0
    active_count = 0
    rolling_sum = 0.0
    for start in range(starts, stop):
        entering = start + window_size - 1
        if 0 <= entering < frames:
            for channel in range(channels):
                value = samples[entering, channel]
                rolling_sum += value * value / channels
        leaving = start - 1
        if 0 <= leaving < frames:
            for channel in range(channels):
                value = samples[leaving, channel]
                rolling_sum -= value * value / channels
        energy = max(0.0, rolling_sum / window_size)
        if energy >= threshold:
            active_energy_sum += energy
            active_count += 1

    if active_count == 0:
        return 0.0, peak_sample, 0, first, last
    return math.sqrt(active_energy_sum / active_count), peak_sample, active_count, first, last


@njit(cache=True)
def _apply_gain(samples: np.ndarray, gain: float) -> np.ndarray:
    result = np.empty(samples.shape, dtype=np.float32)
    for frame in range(samples.shape[0]):
        for channel in range(samples.shape[1]):
            result[frame, channel] = np.float32(samples[frame, channel] * gain)
    return result


def window_size_for_sample_rate(sample_rate: int) -> int:
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("sample rate must be a positive integer")
    return max(1, int(round(sample_rate * WINDOW_SECONDS)))


def measure_active_rms(samples: np.ndarray, sample_rate: int) -> dict:
    """Measure padding-invariant sliding active RMS on mono or frames-by-channels audio."""
    values = np.asarray(samples)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0 or np.iscomplexobj(values):
        raise ValueError("audio must be non-empty real mono or frames-by-channels samples")
    values = np.ascontiguousarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("audio contains nonfinite samples")
    window_size = window_size_for_sample_rate(sample_rate)
    active_rms, peak_sample, active_count, first, last = _measure_active_rms(values, window_size)
    if active_count == 0 or not math.isfinite(active_rms) or active_rms < MIN_ACTIVE_RMS:
        raise ValueError("audio is silent or below the minimum active-RMS floor")
    return {"active_rms": float(active_rms), "peak_sample": float(peak_sample),
            "peak_to_active_rms": float(peak_sample / active_rms), "active_window_count": int(active_count),
            "window_size_samples": window_size, "first_nonzero_sample": int(first),
            "last_nonzero_sample": int(last)}


def normalize_active_rms(samples: np.ndarray, sample_rate: int, target_active_rms: float) -> tuple[np.ndarray, dict]:
    """Apply one fixed scalar gain using the shared padding-invariant active-RMS contract."""
    if not math.isfinite(target_active_rms) or target_active_rms <= 0.0:
        raise ValueError("target active RMS must be finite and positive")
    values = np.asarray(samples)
    was_mono = values.ndim == 1
    if was_mono:
        values = values[:, None]
    details = measure_active_rms(values, sample_rate)
    gain = target_active_rms / details["active_rms"]
    output_peak = details["peak_sample"] * gain
    if not math.isfinite(gain) or not math.isfinite(output_peak):
        raise ValueError("normalization gain or output peak is nonfinite")
    if output_peak > MAX_NORMALIZED_PEAK:
        raise ValueError("fixed target would exceed the shared normalized peak ceiling")
    normalized = _apply_gain(np.ascontiguousarray(values, dtype=np.float64), gain)
    if not np.isfinite(normalized).all() or float(np.max(np.abs(normalized))) > MAX_NORMALIZED_PEAK + 1e-6:
        raise ValueError("normalized audio is nonfinite or exceeds the shared peak ceiling")
    if was_mono:
        normalized = normalized[:, 0]
    return normalized, {**details, "version": NORMALIZATION_VERSION,
                        "sample_rate": int(sample_rate), "target_active_rms": float(target_active_rms),
                        "applied_gain": float(gain), "applied_gain_db": float(20.0 * math.log10(gain)),
                        "output_peak": float(output_peak),
                        "source_peak_above_unit_range": bool(details["peak_sample"] > 1.0)}

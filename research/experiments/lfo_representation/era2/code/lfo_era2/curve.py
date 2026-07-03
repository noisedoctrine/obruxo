"""Curve utilities for Era 2 LFO representation work."""

from __future__ import annotations

import numpy as np


def as_curve_matrix(values: np.ndarray) -> np.ndarray:
    """Return a float32 matrix with shape [rows, resolution]."""
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError("curves must have shape [rows, resolution] with resolution >= 2")
    return array


def resample_curve(values: np.ndarray, size: int) -> np.ndarray:
    """Periodically resample a dense curve to a new fixed grid."""
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 1 or len(array) < 2:
        raise ValueError("values must be a one-dimensional sampled curve")
    if size < 2:
        raise ValueError("size must be at least 2")
    source_x = np.arange(len(array) + 1, dtype=np.float32) / len(array)
    source_y = np.concatenate([array, array[:1]])
    target_x = np.arange(size, dtype=np.float32) / size
    return np.interp(target_x, source_x, source_y).astype(np.float32)


def circular_shift(curves: np.ndarray, phases: np.ndarray | float) -> np.ndarray:
    """Shift curves by cycle-fraction phases using periodic linear interpolation."""
    array = np.asarray(curves, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim < 2:
        raise ValueError("curves must have a sampled resolution axis")

    width = array.shape[-1]
    leading_shape = array.shape[:-1]
    phase_array = np.asarray(phases, dtype=np.float32)
    phase_array = np.broadcast_to(phase_array, leading_shape).reshape(-1)
    flat = array.reshape(-1, width)

    positions = (np.arange(width, dtype=np.float32)[None, :] - phase_array[:, None] * width) % width
    left = np.floor(positions).astype(np.int64)
    right = (left + 1) % width
    frac = positions - left

    rows = np.arange(flat.shape[0])[:, None]
    shifted = flat[rows, left] * (1.0 - frac) + flat[rows, right] * frac
    return shifted.reshape(array.shape).astype(np.float32)


def phase_shift_bank(curves: np.ndarray, phase_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Return [code, phase, resolution] shifted curves and their phase values."""
    if phase_bins < 1:
        raise ValueError("phase_bins must be at least 1")
    matrix = as_curve_matrix(curves)
    phases = np.arange(phase_bins, dtype=np.float32) / float(phase_bins)
    bank = np.stack([circular_shift(matrix, phase) for phase in phases], axis=1)
    return bank.astype(np.float32), phases


def synthetic_base_dictionary(size: int = 32, resolution: int = 64) -> np.ndarray:
    """Create a deterministic smoke-test base dictionary."""
    if size < 1:
        raise ValueError("size must be positive")
    if resolution < 2:
        raise ValueError("resolution must be at least 2")

    x = np.arange(resolution, dtype=np.float32) / float(resolution)
    curves: list[np.ndarray] = []
    for index in range(size):
        phase = index / float(size)
        xp = (x + phase) % 1.0
        kind = index % 8
        if index == 0:
            values = np.full_like(x, 0.5)
        elif kind == 0:
            values = xp
        elif kind == 1:
            values = 1.0 - xp
        elif kind == 2:
            values = 1.0 - np.abs(2.0 * xp - 1.0)
        elif kind == 3:
            values = (xp >= 0.5).astype(np.float32)
        elif kind == 4:
            values = 0.5 + 0.45 * np.sin(2.0 * np.pi * xp)
        elif kind == 5:
            values = 0.5 + 0.35 * np.sin(4.0 * np.pi * xp)
        elif kind == 6:
            values = np.floor(xp * 4.0) / 3.0
        else:
            values = np.clip(0.5 + 0.6 * (xp - 0.5) ** 3 * 4.0, 0.0, 1.0)
        curves.append(np.clip(values, 0.0, 1.0).astype(np.float32))
    return np.stack(curves).astype(np.float32)


def synthetic_residual_dictionaries(
    residual_layer_count: int,
    width: int,
    resolution: int = 64,
) -> list[np.ndarray]:
    """Create deterministic small residual-layer dictionaries for smoke tests."""
    if residual_layer_count < 1:
        raise ValueError("residual_layer_count must be positive")
    if width < 1:
        raise ValueError("width must be positive")
    x = np.arange(resolution, dtype=np.float32) / float(resolution)
    layers: list[np.ndarray] = []
    for residual_layer in range(residual_layer_count):
        amplitude = 0.055 / float(residual_layer + 1)
        atoms = [np.zeros(resolution, dtype=np.float32)]
        for atom_index in range(1, width):
            frequency = 1 + ((atom_index + residual_layer) % 4)
            phase = (atom_index * 0.173 + residual_layer * 0.071) % 1.0
            xp = (x + phase) % 1.0
            if atom_index % 3 == 0:
                atom = amplitude * (2.0 * xp - 1.0)
            elif atom_index % 3 == 1:
                atom = amplitude * np.sin(2.0 * np.pi * frequency * xp)
            else:
                atom = amplitude * (1.0 - np.abs(2.0 * xp - 1.0) - 0.5)
            atoms.append(atom.astype(np.float32))
        layers.append(np.stack(atoms).astype(np.float32))
    return layers


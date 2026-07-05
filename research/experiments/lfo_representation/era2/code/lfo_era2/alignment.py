"""Phase and residual-gain alignment helpers for Era 2 diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .accelerator import BackendPreference, NearestResult, nearest_indices
from .curve import as_curve_matrix, circular_shift, phase_shift_bank


EPS64 = 1e-15
GAIN_BOUNDS = (-2.0, 2.0)


@dataclass(frozen=True)
class AlignmentChoice:
    indices: np.ndarray
    phases: np.ndarray
    gains: np.ndarray
    values: np.ndarray
    losses: np.ndarray
    backend_used: str


@dataclass(frozen=True)
class AlignmentMatrix:
    losses: np.ndarray
    phases: np.ndarray
    gains: np.ndarray


def best_alignment(
    targets: np.ndarray,
    codes: np.ndarray,
    *,
    phase_policy: str,
    gain_policy: str = "fixed",
    backend: BackendPreference = "auto",
    chunk_size: int = 256,
    phase_candidate_count: int | None = None,
) -> AlignmentChoice:
    target_matrix = as_curve_matrix(targets)
    code_matrix = as_curve_matrix(codes)
    if target_matrix.shape[1] != code_matrix.shape[1]:
        raise ValueError("targets and codes must share one resolution")
    if gain_policy not in {"fixed", "optimized"}:
        raise ValueError("gain_policy must be fixed or optimized")
    if phase_policy == "fft_lattice" and gain_policy == "fixed":
        return _best_lattice_fixed(
            target_matrix,
            code_matrix,
            backend=backend,
            chunk_size=chunk_size,
            phase_candidate_count=phase_candidate_count,
        )
    matrix = alignment_matrix(
        target_matrix,
        code_matrix,
        phase_policy=phase_policy,
        gain_policy=gain_policy,
        chunk_size=chunk_size,
        phase_candidate_count=phase_candidate_count,
    )
    indices = np.argmin(matrix.losses, axis=1).astype(np.int32)
    rows = np.arange(len(indices))
    phases = matrix.phases[rows, indices].astype(np.float32)
    gains = matrix.gains[rows, indices].astype(np.float32)
    values = circular_shift(code_matrix[indices], phases) * gains[:, None]
    return AlignmentChoice(
        indices=indices,
        phases=phases,
        gains=gains,
        values=values.astype(np.float32),
        losses=matrix.losses[rows, indices].astype(np.float32),
        backend_used=phase_policy,
    )


def alignment_matrix(
    targets: np.ndarray,
    codes: np.ndarray,
    *,
    phase_policy: str,
    gain_policy: str = "fixed",
    chunk_size: int = 256,
    phase_candidate_count: int | None = None,
) -> AlignmentMatrix:
    target_matrix = as_curve_matrix(targets)
    code_matrix = as_curve_matrix(codes)
    if target_matrix.shape[1] != code_matrix.shape[1]:
        raise ValueError("targets and codes must share one resolution")
    if phase_policy == "exact":
        fixed_gain = None if gain_policy == "optimized" else 1.0
        return exact_alignment_matrix(target_matrix, code_matrix, fixed_gain=fixed_gain)
    if phase_policy == "fft_lattice":
        return lattice_alignment_matrix(
            target_matrix,
            code_matrix,
            gain_policy=gain_policy,
            chunk_size=chunk_size,
            phase_candidate_count=phase_candidate_count,
        )
    raise ValueError(f"unsupported phase_policy: {phase_policy}")


def exact_alignment_matrix(
    targets: np.ndarray,
    codes: np.ndarray,
    *,
    fixed_gain: float | None = 1.0,
    gain_bounds: tuple[float, float] = GAIN_BOUNDS,
) -> AlignmentMatrix:
    target_matrix = np.asarray(targets, dtype=np.float64)
    code_matrix = np.asarray(codes, dtype=np.float64)
    correlations = _correlations_numpy(target_matrix, code_matrix)
    a = correlations
    b = np.roll(correlations, -1, axis=-1) - correlations
    c0, d0, e0 = _interval_constants_numpy(code_matrix)
    c = c0[None, :, None]
    d = d0[None, :, None]
    e = e0[None, :, None]
    zero_codes = c0 <= EPS64
    energy = np.sum(target_matrix * target_matrix, axis=1)[:, None]
    best_error = np.full(a.shape[:2], np.inf, dtype=np.float64)
    best_phase = np.zeros(a.shape[:2], dtype=np.float64)
    best_gain = np.zeros(a.shape[:2], dtype=np.float64)
    for delta in _candidate_deltas_numpy(a, b, c, d, e, gain_bounds, fixed_gain):
        valid = np.isfinite(delta) & (delta >= 0.0) & (delta <= 1.0)
        numerator = a + b * delta
        denominator = c + d * delta + e * delta * delta
        if fixed_gain is None:
            gain = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > EPS64)
            gain = np.clip(gain, *gain_bounds)
        else:
            gain = np.full_like(numerator, fixed_gain)
        mse = (energy[..., None] - 2.0 * gain * numerator + gain * gain * denominator) / target_matrix.shape[1]
        mse[~valid] = np.inf
        interval = np.argmin(mse, axis=2)
        rows = np.arange(len(target_matrix))[:, None]
        cols = np.arange(a.shape[1])[None]
        value = mse[rows, cols, interval]
        improved = value < best_error
        best_error[improved] = value[improved]
        selected_delta = delta[rows, cols, interval]
        selected_gain = gain[rows, cols, interval]
        selected_phase = ((interval + selected_delta) / target_matrix.shape[1]) % 1.0
        best_phase[improved] = selected_phase[improved]
        best_gain[improved] = selected_gain[improved]
    if np.any(zero_codes):
        best_error[:, zero_codes] = energy
        best_phase[:, zero_codes] = 0.0
        best_gain[:, zero_codes] = 0.0
    return AlignmentMatrix(
        losses=np.maximum(best_error, 0.0).astype(np.float32),
        phases=best_phase.astype(np.float32),
        gains=best_gain.astype(np.float32),
    )


def lattice_alignment_matrix(
    targets: np.ndarray,
    codes: np.ndarray,
    *,
    gain_policy: str = "fixed",
    chunk_size: int = 256,
    phase_candidate_count: int | None = None,
) -> AlignmentMatrix:
    target_matrix = as_curve_matrix(targets)
    code_matrix = as_curve_matrix(codes)
    phase_count = int(phase_candidate_count or target_matrix.shape[1])
    bank, phase_values = phase_shift_bank(code_matrix, phase_count)
    flat_bank = bank.reshape(bank.shape[0] * bank.shape[1], bank.shape[2]).astype(np.float32)
    losses = np.empty((len(target_matrix), len(code_matrix)), dtype=np.float32)
    phases = np.empty_like(losses)
    gains = np.empty_like(losses)
    flat_energy = np.sum(flat_bank * flat_bank, axis=1, dtype=np.float64)
    for start in range(0, len(target_matrix), max(1, int(chunk_size))):
        stop = min(start + max(1, int(chunk_size)), len(target_matrix))
        batch = target_matrix[start:stop].astype(np.float64)
        dot = batch @ flat_bank.T.astype(np.float64)
        target_energy = np.sum(batch * batch, axis=1, dtype=np.float64)[:, None]
        if gain_policy == "optimized":
            flat_gain = np.divide(dot, flat_energy[None, :], out=np.zeros_like(dot), where=flat_energy[None, :] > EPS64)
            flat_gain = np.clip(flat_gain, *GAIN_BOUNDS)
        elif gain_policy == "fixed":
            flat_gain = np.ones_like(dot)
        else:
            raise ValueError("gain_policy must be fixed or optimized")
        flat_mse = (target_energy - 2.0 * flat_gain * dot + flat_gain * flat_gain * flat_energy[None, :]) / float(target_matrix.shape[1])
        cube = flat_mse.reshape(len(batch), len(code_matrix), phase_count)
        choice = np.argmin(cube, axis=2)
        rows = np.arange(len(batch))[:, None]
        cols = np.arange(len(code_matrix))[None, :]
        flat_choice = cols * phase_count + choice
        losses[start:stop] = np.maximum(cube[rows, cols, choice], 0.0).astype(np.float32)
        phases[start:stop] = phase_values[choice].astype(np.float32)
        gains[start:stop] = flat_gain[rows, flat_choice].astype(np.float32)
    zero_codes = np.sum(code_matrix * code_matrix, axis=1) <= EPS64
    if np.any(zero_codes):
        losses[:, zero_codes] = np.mean(target_matrix * target_matrix, axis=1)[:, None]
        phases[:, zero_codes] = 0.0
        gains[:, zero_codes] = 0.0
    return AlignmentMatrix(losses=losses, phases=phases, gains=gains)


def _best_lattice_fixed(
    targets: np.ndarray,
    codes: np.ndarray,
    *,
    backend: BackendPreference,
    chunk_size: int,
    phase_candidate_count: int | None,
) -> AlignmentChoice:
    phase_count = int(phase_candidate_count or targets.shape[1])
    bank, phases = phase_shift_bank(codes, phase_count)
    flat_bank = bank.reshape(bank.shape[0] * bank.shape[1], bank.shape[2])
    result = nearest_indices(targets, flat_bank, backend=backend, chunk_size=chunk_size)
    code_index = (result.indices // phase_count).astype(np.int32)
    phase_index = (result.indices % phase_count).astype(np.int32)
    selected = flat_bank[result.indices]
    zero = np.sum(codes[code_index] * codes[code_index], axis=1) <= EPS64
    gains = np.ones(len(code_index), dtype=np.float32)
    gains[zero] = 0.0
    return AlignmentChoice(
        indices=code_index,
        phases=phases[phase_index].astype(np.float32),
        gains=gains,
        values=selected.astype(np.float32),
        losses=result.losses.astype(np.float32),
        backend_used=result.backend_used,
    )


def _correlations_numpy(targets: np.ndarray, codes: np.ndarray) -> np.ndarray:
    target_fft = np.fft.rfft(targets, axis=-1)
    code_fft = np.fft.rfft(codes, axis=-1)
    return np.fft.irfft(target_fft[:, None] * np.conj(code_fft[None]), n=targets.shape[-1], axis=-1).real


def _interval_constants_numpy(codes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    following = np.roll(codes, 1, axis=-1)
    difference = following - codes
    c = np.sum(codes * codes, axis=-1)
    d = 2.0 * np.sum(codes * difference, axis=-1)
    e = np.sum(difference * difference, axis=-1)
    return c, d, e


def _candidate_deltas_numpy(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    e: np.ndarray,
    gain_bounds: tuple[float, float],
    fixed_gain: float | None,
) -> list[np.ndarray]:
    zero = np.zeros_like(a)
    candidates = [zero]
    if fixed_gain is not None:
        vertex = np.divide(
            b / fixed_gain - d / 2.0,
            e,
            out=np.full_like(a, np.nan),
            where=np.abs(e) > EPS64,
        )
        candidates.append(vertex)
        return candidates

    denominator = b * d - 2.0 * a * e
    stationary = np.divide(
        a * d - 2.0 * b * c,
        denominator,
        out=np.full_like(a, np.nan),
        where=np.abs(denominator) > EPS64,
    )
    candidates.append(stationary)
    for bound in gain_bounds:
        if abs(bound) > EPS64:
            candidates.append(
                np.divide(
                    b / bound - d / 2.0,
                    e,
                    out=np.full_like(a, np.nan),
                    where=np.abs(e) > EPS64,
                )
            )
        qa = bound * e
        qb = bound * d - b
        qc = bound * c - a
        discriminant = qb * qb - 4.0 * qa * qc
        root = np.sqrt(np.maximum(discriminant, 0.0))
        for sign in (-1.0, 1.0):
            candidates.append(
                np.divide(
                    -qb + sign * root,
                    2.0 * qa,
                    out=np.full_like(a, np.nan),
                    where=(np.abs(qa) > EPS64) & (discriminant >= 0.0),
                )
            )
    return candidates


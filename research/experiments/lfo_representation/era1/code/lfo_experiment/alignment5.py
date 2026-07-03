"""Per-code continuous phase/gain alignment solvers for Experiment 5."""

from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Iterable

import numpy as np


EPS64 = 1e-15


def _align_progress(message: str) -> None:
    if os.environ.get("LFO_ALIGN_PROGRESS", "0").strip().lower() not in {"0", "false", "no"}:
        print(message, flush=True)


def _align_warning(message: str) -> None:
    print(message, flush=True)


@dataclass
class AlignmentResult:
    error: np.ndarray  # [target, code] MSE after independent optimization
    phase: np.ndarray  # [target, code], cycles in [0, 1)
    gain: np.ndarray  # [target, code]


@dataclass
class TorchAlignmentResult:
    error: object  # torch.Tensor [target, code]
    phase: object  # torch.Tensor [target, code], cycles in [0, 1)
    gain: object  # torch.Tensor [target, code]


def select_best_code(result: AlignmentResult) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select codes only after every code has its own optimized alignment."""
    code = np.argmin(result.error, axis=1)
    rows = np.arange(len(code))
    return code, result.error[rows, code], result.phase[rows, code], result.gain[rows, code]


def _correlations_numpy(targets: np.ndarray, codes: np.ndarray) -> np.ndarray:
    target_fft = np.fft.rfft(targets, axis=-1)
    if codes.ndim == 2:
        code_fft = np.fft.rfft(codes, axis=-1)
        return np.fft.irfft(target_fft[:, None] * np.conj(code_fft[None]), n=targets.shape[-1], axis=-1).real
    code_fft = np.fft.rfft(codes, axis=-1)
    return np.fft.irfft(target_fft[:, None] * np.conj(code_fft), n=targets.shape[-1], axis=-1).real


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
            candidates.append(np.divide(
                b / bound - d / 2.0,
                e,
                out=np.full_like(a, np.nan),
                where=np.abs(e) > EPS64,
            ))
        qa = bound * e
        qb = bound * d - b
        qc = bound * c - a
        discriminant = qb * qb - 4.0 * qa * qc
        root = np.sqrt(np.maximum(discriminant, 0.0))
        for sign in (-1.0, 1.0):
            candidates.append(np.divide(
                -qb + sign * root,
                2.0 * qa,
                out=np.full_like(a, np.nan),
                where=(np.abs(qa) > EPS64) & (discriminant >= 0.0),
            ))
    return candidates


def exact_align_cpu(
    targets: np.ndarray,
    codes: np.ndarray,
    *,
    gain_bounds: tuple[float, float] = (-2.0, 2.0),
    fixed_gain: float | None = None,
) -> AlignmentResult:
    """Global fixed-code optimum for periodic piecewise-linear interpolation."""
    targets = np.asarray(targets, dtype=np.float64)
    codes = np.asarray(codes, dtype=np.float64)
    correlations = _correlations_numpy(targets, codes)
    a = correlations
    b = np.roll(correlations, -1, axis=-1) - correlations
    c0, d0, e0 = _interval_constants_numpy(codes)
    if codes.ndim == 2:
        c = c0[None, :, None]
        d = d0[None, :, None]
        e = e0[None, :, None]
        zero_codes = c0 <= EPS64
    else:
        c = c0[..., None]
        d = d0[..., None]
        e = e0[..., None]
        zero_codes = c0 <= EPS64
    energy = np.sum(targets * targets, axis=1)[:, None]
    best_error = np.full(a.shape[:2], np.inf, dtype=np.float64)
    best_phase = np.zeros(a.shape[:2], dtype=np.float64)
    best_gain = np.zeros(a.shape[:2], dtype=np.float64)
    shifts = np.arange(targets.shape[1], dtype=np.float64)[None, None, :]
    for delta in _candidate_deltas_numpy(a, b, c, d, e, gain_bounds, fixed_gain):
        valid = np.isfinite(delta) & (delta >= 0.0) & (delta <= 1.0)
        numerator = a + b * delta
        denominator = c + d * delta + e * delta * delta
        if fixed_gain is None:
            gain = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > EPS64)
            gain = np.clip(gain, *gain_bounds)
        else:
            gain = np.full_like(numerator, fixed_gain)
        mse = (energy[..., None] - 2.0 * gain * numerator + gain * gain * denominator) / targets.shape[1]
        mse[~valid] = np.inf
        interval = np.argmin(mse, axis=2)
        rows = np.arange(len(targets))[:, None]
        cols = np.arange(a.shape[1])[None]
        value = mse[rows, cols, interval]
        improved = value < best_error
        best_error[improved] = value[improved]
        selected_delta = delta[rows, cols, interval]
        selected_gain = gain[rows, cols, interval]
        selected_phase = ((interval + selected_delta) / targets.shape[1]) % 1.0
        best_phase[improved] = selected_phase[improved]
        best_gain[improved] = selected_gain[improved]
    if np.any(zero_codes):
        if codes.ndim == 2:
            best_error[:, zero_codes] = energy
            best_phase[:, zero_codes] = 0.0
            best_gain[:, zero_codes] = 0.0
        else:
            best_error[zero_codes] = energy.repeat(best_error.shape[1], axis=1)[zero_codes]
            best_phase[zero_codes] = 0.0
            best_gain[zero_codes] = 0.0
    return AlignmentResult(
        np.maximum(best_error, 0.0).astype(np.float64),
        best_phase.astype(np.float64),
        best_gain.astype(np.float64),
    )


def dense_align_cpu(
    targets: np.ndarray,
    codes: np.ndarray,
    positions: int,
    *,
    gain_bounds: tuple[float, float] = (-2.0, 2.0),
    fixed_gain: float | None = None,
) -> AlignmentResult:
    """Dense phase-grid reference evaluated from interval polynomials."""
    targets = np.asarray(targets, dtype=np.float64)
    codes = np.asarray(codes, dtype=np.float64)
    correlations = _correlations_numpy(targets, codes)
    width = targets.shape[1]
    phase_index = np.arange(positions, dtype=np.float64) * width / positions
    interval = np.floor(phase_index).astype(np.int64) % width
    delta = phase_index - np.floor(phase_index)
    a = correlations[..., interval]
    b = np.roll(correlations, -1, axis=-1)[..., interval] - a
    c0, d0, e0 = _interval_constants_numpy(codes)
    if codes.ndim == 2:
        c, d, e = c0[None, :, None], d0[None, :, None], e0[None, :, None]
    else:
        c, d, e = c0[..., None], d0[..., None], e0[..., None]
    numerator = a + b * delta
    denominator = c + d * delta + e * delta * delta
    if fixed_gain is None:
        gain = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > EPS64)
        gain = np.clip(gain, *gain_bounds)
    else:
        gain = np.full_like(numerator, fixed_gain)
    energy = np.sum(targets * targets, axis=1)[:, None, None]
    mse = (energy - 2 * gain * numerator + gain * gain * denominator) / width
    choice = np.argmin(mse, axis=2)
    rows = np.arange(len(targets))[:, None]
    cols = np.arange(mse.shape[1])[None]
    return AlignmentResult(
        np.maximum(mse[rows, cols, choice], 0.0),
        (choice.astype(np.float64) / positions) % 1.0,
        gain[rows, cols, choice],
    )


def fft128_local9_cpu(
    targets: np.ndarray,
    codes: np.ndarray,
    *,
    gain_bounds: tuple[float, float] = (-2.0, 2.0),
    fixed_gain: float | None = None,
) -> AlignmentResult:
    """Experiment 4-style 128-grid search followed by nine local samples per code."""
    width = targets.shape[1]
    coarse = dense_align_cpu(targets, codes, 128, gain_bounds=gain_bounds, fixed_gain=fixed_gain)
    offsets = np.linspace(-1.0 / 128.0, 1.0 / 128.0, 9)
    best = AlignmentResult(coarse.error.copy(), coarse.phase.copy(), coarse.gain.copy())
    for offset in offsets:
        phase = (coarse.phase + offset) % 1.0
        phase_index = phase * width
        interval = np.floor(phase_index).astype(np.int64) % width
        delta = phase_index - np.floor(phase_index)
        corr = _correlations_numpy(np.asarray(targets, np.float64), np.asarray(codes, np.float64))
        rows = np.arange(len(targets))[:, None]
        cols = np.arange(corr.shape[1])[None]
        a = corr[rows, cols, interval]
        following = np.roll(corr, -1, axis=-1)[rows, cols, interval]
        numerator = a + (following - a) * delta
        c0, d0, e0 = _interval_constants_numpy(np.asarray(codes, np.float64))
        c = c0[None] if codes.ndim == 2 else c0
        d = d0[None] if codes.ndim == 2 else d0
        e = e0[None] if codes.ndim == 2 else e0
        denominator = c + d * delta + e * delta * delta
        if fixed_gain is None:
            gain = np.clip(np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > EPS64), *gain_bounds)
        else:
            gain = np.full_like(numerator, fixed_gain)
        energy = np.sum(targets * targets, axis=1)[:, None]
        error = (energy - 2 * gain * numerator + gain * gain * denominator) / width
        improved = error < best.error
        best.error[improved] = error[improved]
        best.phase[improved] = phase[improved]
        best.gain[improved] = gain[improved]
    return best


def circular_shift_torch(values, phase):
    """Fractionally roll periodic rows by phase cycles using torch gather."""
    import torch

    array = values.to(dtype=torch.float32)
    was_vector = array.ndim == 1
    if was_vector:
        array = array[None, :]
    leading_shape = array.shape[:-1]
    width = array.shape[-1]
    flat = array.reshape(-1, width)
    phases = torch.broadcast_to(phase.to(device=array.device, dtype=torch.float32), leading_shape).reshape(-1)
    position = (torch.arange(width, device=array.device, dtype=torch.float32)[None, :] - phases[:, None] * width) % width
    left = torch.floor(position).to(torch.long)
    fraction = position - left.to(torch.float32)
    right = torch.remainder(left + 1, width)
    shifted = torch.gather(flat, 1, left) * (1.0 - fraction) + torch.gather(flat, 1, right) * fraction
    shifted = shifted.reshape(*leading_shape, width)
    return shifted[0] if was_vector else shifted


def _candidate_deltas_torch(a, b, c, d, e, gain_bounds, fixed_gain):
    import torch
    nan = torch.full_like(a, torch.nan)
    yield torch.zeros_like(a)
    if fixed_gain is not None:
        yield torch.where(torch.abs(e) > 1e-12, (b / fixed_gain - d / 2) / e, nan)
        return
    denominator = b * d - 2 * a * e
    yield torch.where(torch.abs(denominator) > 1e-12, (a * d - 2 * b * c) / denominator, nan)
    for bound in gain_bounds:
        if abs(bound) > 1e-12:
            yield torch.where(torch.abs(e) > 1e-12, (b / bound - d / 2) / e, nan)
        qa, qb, qc = bound * e, bound * d - b, bound * c - a
        discriminant = qb * qb - 4 * qa * qc
        root = torch.sqrt(torch.clamp(discriminant, min=0))
        for sign in (-1.0, 1.0):
            yield torch.where(
                (torch.abs(qa) > 1e-12) & (discriminant >= 0), (-qb + sign * root) / (2 * qa), nan
            )


def _exact_align_torch_unbatched(
    targets,
    codes,
    *,
    gain_bounds: tuple[float, float] = (-2.0, 2.0),
    fixed_gain: float | None = None,
    device: str = "xpu:0",
) -> TorchAlignmentResult:
    """Batched float32 implementation of the analytic interval solver."""
    import torch

    with torch.no_grad():
        target = torch.as_tensor(targets, dtype=torch.float32, device=device)
        code = torch.as_tensor(codes, dtype=torch.float32, device=device)
    target_fft = torch.fft.rfft(target, dim=-1)
    code_fft = torch.fft.rfft(code, dim=-1)
    if code.ndim == 2:
        corr = torch.fft.irfft(target_fft[:, None] * torch.conj(code_fft[None]), n=target.shape[-1], dim=-1)
        following_code = torch.roll(code, 1, -1)
        diff = following_code - code
        c = torch.sum(code * code, -1)[None, :, None]
        d = (2 * torch.sum(code * diff, -1))[None, :, None]
        e = torch.sum(diff * diff, -1)[None, :, None]
        zero = c[..., 0] <= 1e-12
    else:
        corr = torch.fft.irfft(target_fft[:, None] * torch.conj(code_fft), n=target.shape[-1], dim=-1)
        diff = torch.roll(code, 1, -1) - code
        c = torch.sum(code * code, -1)[..., None]
        d = (2 * torch.sum(code * diff, -1))[..., None]
        e = torch.sum(diff * diff, -1)[..., None]
        zero = c[..., 0] <= 1e-12
    a = corr
    b = torch.roll(corr, -1, -1) - corr
    energy = torch.sum(target * target, -1)[:, None]
    best_error = torch.full(a.shape[:2], torch.inf, device=device)
    best_phase = torch.zeros(a.shape[:2], device=device)
    best_gain = torch.zeros(a.shape[:2], device=device)
    for delta in _candidate_deltas_torch(a, b, c, d, e, gain_bounds, fixed_gain):
        valid = torch.isfinite(delta) & (delta >= 0) & (delta <= 1)
        numerator = a + b * delta
        denominator = c + d * delta + e * delta * delta
        if fixed_gain is None:
            gain = torch.clamp(torch.where(denominator > 1e-12, numerator / denominator, 0), *gain_bounds)
        else:
            gain = torch.full_like(numerator, fixed_gain)
        mse = (energy[..., None] - 2 * gain * numerator + gain * gain * denominator) / target.shape[-1]
        mse = torch.where(valid, mse, torch.inf)
        value, interval = torch.min(mse, -1)
        chosen_delta = torch.gather(delta, -1, interval[..., None])[..., 0]
        chosen_gain = torch.gather(gain, -1, interval[..., None])[..., 0]
        chosen_phase = (interval + chosen_delta) / target.shape[-1]
        improved = value < best_error
        best_error = torch.where(improved, value, best_error)
        best_phase = torch.where(improved, chosen_phase, best_phase)
        best_gain = torch.where(improved, chosen_gain, best_gain)
        del valid, numerator, denominator, gain, mse, value, interval, chosen_delta, chosen_gain, chosen_phase, improved, delta
    if code.ndim == 2 and torch.any(zero):
        best_error[:, zero[0]] = energy
        best_phase[:, zero[0]] = 0
        best_gain[:, zero[0]] = 0
    elif code.ndim == 3 and torch.any(zero):
        best_error = torch.where(zero, energy.expand_as(best_error), best_error)
        best_phase = torch.where(zero, 0, best_phase)
        best_gain = torch.where(zero, 0, best_gain)
    error = torch.clamp(best_error, min=0)
    phase = torch.remainder(best_phase, 1)
    gain = best_gain
    del target, code, target_fft, code_fft, corr, a, b, energy, best_error, best_phase, best_gain
    if "diff" in locals():
        del diff
    if "c" in locals():
        del c, d, e, zero
    return TorchAlignmentResult(error, phase, gain)


def exact_align_torch_tensors(
    targets,
    codes,
    *,
    gain_bounds: tuple[float, float] = (-2.0, 2.0),
    fixed_gain: float | None = None,
    device: str = "xpu:0",
    batch_size: int | None = None,
) -> TorchAlignmentResult:
    """Torch alignment wrapper that keeps results resident on the requested device."""
    import torch

    targets_tensor = torch.as_tensor(targets, dtype=torch.float32, device=device)
    codes_tensor = torch.as_tensor(codes, dtype=torch.float32, device=device)
    if batch_size is None:
        batch_size = int(os.environ.get("LFO_TORCH_ALIGN_BATCH_SIZE", "256"))
    batch_size = max(1, int(batch_size))
    progress_seconds = float(os.environ.get("LFO_ALIGN_PROGRESS_SECONDS", "20"))
    started = time.perf_counter()
    last_progress = started
    code_count = codes_tensor.shape[1] if codes_tensor.ndim == 3 else codes_tensor.shape[0]
    if len(targets_tensor) > batch_size:
        _align_progress(
            f"align[{device}]: targets={len(targets_tensor):,}, codes={code_count:,}, "
            f"resolution={targets_tensor.shape[-1]:,}, chunk={batch_size}"
        )
    if len(targets_tensor) <= batch_size:
        return _exact_align_torch_unbatched(
            targets_tensor,
            codes_tensor,
            gain_bounds=gain_bounds,
            fixed_gain=fixed_gain,
            device=device,
        )

    error_parts = []
    phase_parts = []
    gain_parts = []
    start = 0
    current_batch = batch_size
    while start < len(targets_tensor):
        stop = min(start + current_batch, len(targets_tensor))
        try:
            part = _exact_align_torch_unbatched(
                targets_tensor[start:stop],
                codes_tensor[start:stop] if codes_tensor.ndim == 3 and len(codes_tensor) == len(targets_tensor) else codes_tensor,
                gain_bounds=gain_bounds,
                fixed_gain=fixed_gain,
                device=device,
            )
        except RuntimeError as exc:
            message = str(exc).lower()
            is_oom = "out of memory" in message or "oom" in message
            if not is_oom or current_batch == 1:
                raise
            current_batch = max(1, current_batch // 2)
            _align_warning(
                f"align[{device}]: out of memory; retrying from target {start:,} "
                f"with smaller chunk={current_batch}"
            )
            continue
        error_parts.append(part.error)
        phase_parts.append(part.phase)
        gain_parts.append(part.gain)
        del part
        start = stop
        now = time.perf_counter()
        if now - last_progress >= progress_seconds or start >= len(targets_tensor):
            elapsed = now - started
            rate = start / elapsed if elapsed > 0 else 0.0
            remaining = (len(targets_tensor) - start) / rate if rate > 0 else float("nan")
            _align_progress(
                f"align[{device}]: {start:,}/{len(targets_tensor):,} targets "
                f"({start / len(targets_tensor):.1%}), chunk={current_batch}, "
                f"elapsed={elapsed:.1f}s, eta={remaining:.1f}s"
            )
            last_progress = now
    return TorchAlignmentResult(
        torch.cat(error_parts, dim=0),
        torch.cat(phase_parts, dim=0),
        torch.cat(gain_parts, dim=0),
    )


def exact_align_torch(
    targets: np.ndarray,
    codes: np.ndarray,
    *,
    gain_bounds: tuple[float, float] = (-2.0, 2.0),
    fixed_gain: float | None = None,
    device: str = "xpu:0",
    batch_size: int | None = None,
) -> AlignmentResult:
    """Batched PyTorch alignment wrapper that returns NumPy arrays."""
    result = exact_align_torch_tensors(
        targets,
        codes,
        gain_bounds=gain_bounds,
        fixed_gain=fixed_gain,
        device=device,
        batch_size=batch_size,
    )
    return AlignmentResult(
        result.error.detach().cpu().numpy().copy(),
        result.phase.detach().cpu().numpy().copy(),
        result.gain.detach().cpu().numpy().copy(),
    )


def exact_align_xpu(
    targets: np.ndarray,
    codes: np.ndarray,
    *,
    gain_bounds: tuple[float, float] = (-2.0, 2.0),
    fixed_gain: float | None = None,
    device: str = "xpu:0",
) -> AlignmentResult:
    """Compatibility wrapper for existing Intel XPU callers."""
    return exact_align_torch(
        targets,
        codes,
        gain_bounds=gain_bounds,
        fixed_gain=fixed_gain,
        device=device,
    )


def phase_distance(a: np.ndarray, b: np.ndarray, symmetry: int = 1) -> np.ndarray:
    difference = np.abs((a - b) * symmetry) % 1.0
    return np.minimum(difference, 1.0 - difference) / symmetry


def _shift_code_grid(code: np.ndarray, phases: np.ndarray) -> np.ndarray:
    width = len(code)
    position = (np.arange(width)[None] - phases[:, None] * width) % width
    left = np.floor(position).astype(np.int64) % width
    fraction = position - left
    return code[left] * (1.0 - fraction) + code[(left + 1) % width] * fraction


def clipped_grid_reference(
    targets: np.ndarray,
    prefixes: np.ndarray,
    codes: np.ndarray,
    *,
    positions: int = 4096,
    top_peaks: int = 8,
    refine_rounds: int = 3,
    gain_bounds: tuple[float, float] = (-2.0, 2.0),
) -> AlignmentResult:
    """High-resolution clipped-prefix reference with deterministic local refinement.

    This is intentionally numerical: clipping changes active samples as phase and
    gain move, so the residual-space analytic result is no longer the full objective.
    """
    targets = np.asarray(targets, np.float64)
    prefixes = np.asarray(prefixes, np.float64)
    codes = np.asarray(codes, np.float64)
    if codes.ndim == 2:
        codes = np.broadcast_to(codes[None], (len(targets), *codes.shape))
    b, k, width = codes.shape
    peak_error = np.full((b, k, top_peaks), np.inf)
    peak_phase = np.zeros((b, k, top_peaks))
    peak_gain = np.zeros((b, k, top_peaks))
    residual = targets - prefixes
    for code_index in range(k):
        for start in range(0, positions, 128):
            indices = np.arange(start, min(start + 128, positions))
            phases = indices / positions
            for row in range(b):
                shifted = _shift_code_grid(codes[row, code_index], phases)
                denominator = np.sum(shifted * shifted, axis=1)
                gain = np.clip(
                    np.divide(shifted @ residual[row], denominator, out=np.zeros(len(phases)), where=denominator > EPS64),
                    *gain_bounds,
                )
                reconstruction = np.clip(prefixes[row][None] + gain[:, None] * shifted, 0.0, 1.0)
                error = np.mean((reconstruction - targets[row]) ** 2, axis=1)
                all_error = np.concatenate([peak_error[row, code_index], error])
                all_phase = np.concatenate([peak_phase[row, code_index], phases])
                all_gain = np.concatenate([peak_gain[row, code_index], gain])
                keep = np.argpartition(all_error, top_peaks - 1)[:top_peaks]
                order = keep[np.argsort(all_error[keep])]
                peak_error[row, code_index] = all_error[order]
                peak_phase[row, code_index] = all_phase[order]
                peak_gain[row, code_index] = all_gain[order]

    best_error = peak_error[..., 0].copy()
    best_phase = peak_phase[..., 0].copy()
    best_gain = peak_gain[..., 0].copy()
    for row in range(b):
        for code_index in range(k):
            if np.mean(codes[row, code_index] ** 2) <= EPS64:
                best_error[row, code_index] = np.mean((prefixes[row] - targets[row]) ** 2)
                best_phase[row, code_index] = 0.0
                best_gain[row, code_index] = 0.0
                continue
            for peak in range(top_peaks):
                center_phase = peak_phase[row, code_index, peak]
                center_gain = peak_gain[row, code_index, peak]
                phase_radius = 1.0 / positions
                gain_radius = .25
                candidate_best = peak_error[row, code_index, peak]
                for _ in range(refine_rounds):
                    phases = (center_phase + np.linspace(-phase_radius, phase_radius, 9)) % 1.0
                    gains = np.clip(center_gain + np.linspace(-gain_radius, gain_radius, 9), *gain_bounds)
                    shifted = _shift_code_grid(codes[row, code_index], phases)
                    reconstruction = np.clip(
                        prefixes[row][None, None] + gains[None, :, None] * shifted[:, None, :], 0.0, 1.0
                    )
                    error = np.mean((reconstruction - targets[row]) ** 2, axis=2)
                    phase_choice, gain_choice = np.unravel_index(np.argmin(error), error.shape)
                    if error[phase_choice, gain_choice] < candidate_best:
                        candidate_best = error[phase_choice, gain_choice]
                        center_phase = phases[phase_choice]
                        center_gain = gains[gain_choice]
                    phase_radius /= 4.0
                    gain_radius /= 4.0
                if candidate_best < best_error[row, code_index]:
                    best_error[row, code_index] = candidate_best
                    best_phase[row, code_index] = center_phase
                    best_gain[row, code_index] = center_gain
    return AlignmentResult(best_error, best_phase % 1.0, best_gain)

"""Experiment 7: additive LFO codebook finalization.

Experiment 7 deliberately splits into:

* 7A: compare residual construction policies at one representative budget.
* 7B: after human review, sweep K/D budgets for an explicitly selected policy.

The decoder differs from earlier phase experiments in one important way: it
uses one final hard clip only.  Intermediate prefixes remain unclipped so the
oracle measures the representation we intend to discuss.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .alignment5 import (
    AlignmentResult,
    circular_shift_torch,
    exact_align_cpu,
    exact_align_torch_tensors,
    exact_align_xpu,
)
from .experiment6 import (
    NODE_THRESHOLDS,
    RMSE_THRESHOLDS,
    _assert_author_split,
    _conditions_for,
    _evaluate_reconstruction,
    _format_duration,
    _output_cost,
    _resample_chain,
    _safe_id,
    _structured_weight,
    _write_progress,
    summarize_results,
)
from .phase4 import PhaseChain, PhaseEncoding, circular_shift
from .stacked import TOPOLOGY_NAMES, CurveDataset, load_curve_dataset, load_stock_curves


SEED = 20260707
EXPERIMENT7A_RESOLUTION = 960
FINAL_EVAL_RESOLUTION = 1920
EVAL_RESOLUTION = FINAL_EVAL_RESOLUTION
STOCK_RMSE_THRESHOLD = 0.005
STOCK_POINTWISE_THRESHOLD = 0.02
GAIN_BOUNDS = (-2.0, 2.0)
CONSTRUCTION_STRATEGIES = (
    "frequency_first",
    "greedy_global_improvement",
    "tail_aware_greedy",
    "common_then_tail",
    "topology_balanced_common_then_tail",
)
MODIFIER_POLICIES = ("none", "global_offset", "base_gain", "base_gain_global_offset")
RESIDUAL_CLIP_POLICIES = (
    "final_only",
    "intermediate_m11_final_01",
    "unipolar_guard_each_layer",
    "bipolar_guard_each_layer",
    "headroom_guard_025_each_layer",
    "headroom_guard_050_each_layer",
    "base_unipolar_residual_bipolar",
    "residual_depth_limiter_025",
    "residual_depth_limiter_050",
)
EXPERIMENT8_RESOLUTION = 120
EXPERIMENT8_SAMPLE_FRACTION = 1.0 / 3.0
EXPERIMENT8_CONSTRUCTION_RECIPE = "topology_balanced_common_then_tail"
EXPERIMENT8_MODIFIER_LABELS = {
    "phase_only": "none",
    "phase_gain": "base_gain",
    "phase_offset": "global_offset",
    "phase_gain_offset": "base_gain_global_offset",
}
EXPERIMENT9_RESOLUTION = 120
EXPERIMENT9_SAMPLE_FRACTION = 1.0 / 3.0
EXPERIMENT9_CONSTRUCTION_RECIPE = EXPERIMENT8_CONSTRUCTION_RECIPE
EXPERIMENT9_RESIDUAL_WIDTH = 8
EXPERIMENT9_RESIDUAL_DEPTH = 16
EXPERIMENT9_EPS = 1e-4
EXPERIMENT9_BUDGET_BASELINE_WIDTH = 8
EXPERIMENT9_BUDGET_ANCHOR_DEPTHS = (24, 32, 48, 64)
EXPERIMENT9_BUDGET_WIDTHS = (4, 6)
EXPERIMENT9_CLIP_POLICIES = (
    "final_only",
    "unipolar_guard_each_layer",
    "bipolar_guard_each_layer",
    "headroom_guard_025_each_layer",
    "headroom_guard_050_each_layer",
    "base_unipolar_residual_bipolar",
    "residual_depth_limiter_025",
    "residual_depth_limiter_050",
)
EXPERIMENT9_SNAP_POLICIES = (
    "none",
    "data_snap_rails",
    "data_snap_dyadic_1",
    "data_snap_dyadic_2",
    "data_snap_dyadic_2_triadic_1",
)
_DATASET_CACHE: dict[tuple[str, int], CurveDataset] = {}
_STOCK_CACHE: dict[tuple[str, int], tuple[list[str], np.ndarray]] = {}


@dataclass(frozen=True)
class Experiment7Policy:
    construction_strategy: str
    modifier_policy: str = "none"
    stock_rmse_threshold: float = STOCK_RMSE_THRESHOLD
    stock_pointwise_threshold: float = STOCK_POINTWISE_THRESHOLD
    base_medoids_source: str = "leftover_frequency"
    selection_cutover_layer: int | None = None
    residual_clip_policy: str = "final_only"
    affine_scope: str = "legacy"
    affine_modulation: str = "legacy"
    range_normalization: bool = False


@dataclass
class Experiment7Job:
    job_id: str
    experiment: str
    policy: Experiment7Policy
    k: int
    d: int
    eval_resolution: int
    beam_width: int
    batch_size: int
    seed: int
    weight: float
    modifier_label: str | None = None
    sample_fraction: float = 1.0
    sample_seed: int | None = None
    estimated_peak_memory_mb: float = 0.0
    experiment9_section: str = ""
    target_scope: str = ""
    affine_modulation: str = ""
    normalization_label: str = ""
    decoder_hygiene_policy: str = ""
    snap_policy: str = "none"
    budget_anchor_width: int = 0
    budget_anchor_depth: int = 0
    budget_anchor_head_outputs: int = 0
    budget_actual_head_outputs: int = 0

    @property
    def label(self) -> str:
        if self.experiment in {"8", "9"}:
            experiment_label = "8" if self.experiment == "8" else "9"
            return (
                f"Experiment {experiment_label} screen {self.policy.construction_strategy} "
                f"{self.modifier_label or self.policy.modifier_policy} W{self.k}D{self.residual_depth} "
                f"{self.policy.residual_clip_policy}"
            )
        return (
            f"Experiment {self.experiment} {self.policy.construction_strategy} "
            f"K{self.k} D{self.d} {self.policy.modifier_policy}"
        )

    @property
    def residual_depth(self) -> int:
        return self.d * 2 if self.experiment in {"8", "9"} else self.d


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _log_progress(message: str) -> None:
    if os.environ.get("LFO_PROGRESS", "1").strip().lower() not in {"0", "false", "no"}:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def _effort(done: float, total: float) -> str:
    if total <= 0:
        return "n/a"
    return f"{max(0.0, min(1.0, done / total)):.1%}"


def _load_cached_dataset(catalog_path: Path, resolution: int) -> CurveDataset:
    key = (str(catalog_path.resolve()), int(resolution))
    if key not in _DATASET_CACHE:
        _DATASET_CACHE[key] = load_curve_dataset(catalog_path, resolution=resolution)
    return _DATASET_CACHE[key]


def _load_cached_stock(codebook_path: Path, resolution: int) -> tuple[list[str], np.ndarray]:
    key = (str(codebook_path.resolve()), int(resolution))
    if key not in _STOCK_CACHE:
        _STOCK_CACHE[key] = load_stock_curves(codebook_path, resolution=resolution)
    return _STOCK_CACHE[key]


def _torch_align_device() -> str | None:
    device_preference = os.environ.get("LFO_ALIGN_DEVICE", "auto").strip().lower()
    if device_preference in {"cpu", "numpy"}:
        return None
    if device_preference not in {"auto", "xpu"}:
        raise RuntimeError(f"unsupported LFO_ALIGN_DEVICE={device_preference!r}; use auto, xpu, or cpu")
    try:
        import torch

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu:0"
    except Exception:
        if device_preference == "xpu":
            raise
    if device_preference == "xpu":
        raise RuntimeError("LFO_ALIGN_DEVICE requested XPU, but torch.xpu is unavailable")
    return None


def _release_xpu_temporaries(device: str) -> None:
    if not str(device).startswith("xpu"):
        return
    try:
        import torch

        if hasattr(torch, "xpu"):
            torch.xpu.synchronize()
            if hasattr(torch.xpu, "empty_cache"):
                torch.xpu.empty_cache()
    except Exception:
        # Cleanup is a stability hint; alignment/evaluation errors should still
        # come from the operation that failed, not from cache maintenance.
        return


def _exact_align(targets: np.ndarray, codes: np.ndarray, *, fixed_gain: float | None = None) -> AlignmentResult:
    device = _torch_align_device()
    if device is not None:
        return exact_align_xpu(targets, codes, fixed_gain=fixed_gain, gain_bounds=GAIN_BOUNDS, device=device)
    return exact_align_cpu(targets, codes, fixed_gain=fixed_gain, gain_bounds=GAIN_BOUNDS)


def _clip_intermediate_np(values: np.ndarray, residual_clip_policy: str) -> np.ndarray:
    if residual_clip_policy == "final_only":
        return values
    if residual_clip_policy in {"intermediate_m11_final_01", "bipolar_guard_each_layer"}:
        return np.clip(values, -1.0, 1.0)
    if residual_clip_policy == "unipolar_guard_each_layer":
        return np.clip(values, 0.0, 1.0)
    if residual_clip_policy == "headroom_guard_025_each_layer":
        return np.clip(values, -0.25, 1.25)
    if residual_clip_policy == "headroom_guard_050_each_layer":
        return np.clip(values, -0.5, 1.5)
    if residual_clip_policy in {
        "base_unipolar_residual_bipolar",
        "residual_depth_limiter_025",
        "residual_depth_limiter_050",
    }:
        return values
    raise ValueError(f"unsupported residual clip policy: {residual_clip_policy}")


def _clip_intermediate_torch(values, residual_clip_policy: str):
    if residual_clip_policy == "final_only":
        return values
    if residual_clip_policy in {"intermediate_m11_final_01", "bipolar_guard_each_layer"}:
        return values.clamp(-1.0, 1.0)
    if residual_clip_policy == "unipolar_guard_each_layer":
        return values.clamp(0.0, 1.0)
    if residual_clip_policy == "headroom_guard_025_each_layer":
        return values.clamp(-0.25, 1.25)
    if residual_clip_policy == "headroom_guard_050_each_layer":
        return values.clamp(-0.5, 1.5)
    if residual_clip_policy in {
        "base_unipolar_residual_bipolar",
        "residual_depth_limiter_025",
        "residual_depth_limiter_050",
    }:
        return values
    raise ValueError(f"unsupported residual clip policy: {residual_clip_policy}")


def _apply_decoder_step_np(
    prefix: np.ndarray,
    addition: np.ndarray,
    residual_clip_policy: str,
    *,
    is_base: bool = False,
) -> np.ndarray:
    candidate_addition = addition
    if not is_base and residual_clip_policy == "residual_depth_limiter_025":
        candidate_addition = np.clip(addition, -0.25, 0.25)
    elif not is_base and residual_clip_policy == "residual_depth_limiter_050":
        candidate_addition = np.clip(addition, -0.5, 0.5)
    candidate = prefix + candidate_addition
    if residual_clip_policy == "base_unipolar_residual_bipolar":
        return np.clip(candidate, 0.0, 1.0) if is_base else np.clip(candidate, -1.0, 1.0)
    return _clip_intermediate_np(candidate, residual_clip_policy)


def _apply_decoder_step_torch(prefix, addition, residual_clip_policy: str, *, is_base: bool = False):
    candidate_addition = addition
    if not is_base and residual_clip_policy == "residual_depth_limiter_025":
        candidate_addition = addition.clamp(-0.25, 0.25)
    elif not is_base and residual_clip_policy == "residual_depth_limiter_050":
        candidate_addition = addition.clamp(-0.5, 0.5)
    candidate = prefix + candidate_addition
    if residual_clip_policy == "base_unipolar_residual_bipolar":
        return candidate.clamp(0.0, 1.0) if is_base else candidate.clamp(-1.0, 1.0)
    return _clip_intermediate_torch(candidate, residual_clip_policy)


def _affine_applies(policy: Experiment7Policy, target: str) -> bool:
    if policy.affine_scope == "legacy":
        return False
    return policy.affine_scope == f"{target}_only" or policy.affine_scope == "base_and_residuals"


def _gain_enabled(policy: Experiment7Policy, target: str) -> bool:
    if policy.affine_scope == "legacy":
        return target == "residual"
    return _affine_applies(policy, target) and "gain" in policy.affine_modulation


def _offset_enabled(policy: Experiment7Policy, target: str) -> bool:
    return _affine_applies(policy, target) and "offset" in policy.affine_modulation


def _normalization_enabled(policy: Experiment7Policy, target: str) -> bool:
    return bool(policy.range_normalization and _affine_applies(policy, target))


def _range_normalize_np(values: np.ndarray, *, target: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=np.float32)
    lo = np.min(array, axis=1)
    hi = np.max(array, axis=1)
    if target == "base":
        offset = lo
        gain = hi - lo
    else:
        offset = (hi + lo) * 0.5
        gain = (hi - lo) * 0.5
    flat = gain < EXPERIMENT9_EPS
    normalized = np.divide(
        array - offset[:, None],
        gain[:, None],
        out=np.zeros_like(array, dtype=np.float32),
        where=gain[:, None] >= EXPERIMENT9_EPS,
    )
    if target == "residual":
        normalized = np.clip(normalized, -1.0, 1.0)
    else:
        normalized = np.clip(normalized, 0.0, 1.0)
    return normalized.astype(np.float32), offset.astype(np.float32), gain.astype(np.float32), flat


def _offset_for_prediction_np(targets: np.ndarray, predicted: np.ndarray, enabled: bool) -> np.ndarray:
    if not enabled:
        return np.zeros(predicted.shape[:-1], dtype=np.float32)
    return np.clip(np.mean(targets[..., None, :] - predicted, axis=-1), -2.0, 2.0).astype(np.float32)


def _experiment9_snap_candidates(policy: str) -> np.ndarray:
    grids = {
        "none": (),
        "data_snap_rails": (0.0, 1.0),
        "data_snap_dyadic_1": (0.0, 0.5, 1.0),
        "data_snap_dyadic_2": (0.0, 0.25, 0.5, 0.75, 1.0),
        "data_snap_dyadic_2_triadic_1": (0.0, 0.25, 1.0 / 3.0, 0.5, 2.0 / 3.0, 0.75, 1.0),
    }
    if policy not in grids:
        raise ValueError(f"unsupported Experiment 9 snap policy: {policy}")
    return np.asarray(grids[policy], dtype=np.float32)


def _infer_snap_schwarzschild(dataset: CurveDataset, snap_policy: str) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    candidates = _experiment9_snap_candidates(snap_policy)
    if len(candidates) == 0:
        return candidates, np.zeros(0, dtype=np.float32), {
            "snap_anchor_count": 0.0,
            "snap_radius_median": 0.0,
        }
    curves = dataset.curves[dataset.train_indices].astype(np.float32)
    slope = np.abs(np.roll(curves, -1, axis=1) - np.roll(curves, 1, axis=1)) * 0.5
    weights = 1.0 / (0.005 + slope)
    values = curves.reshape(-1)
    flat_weights = weights.reshape(-1)
    total_weight = float(np.sum(flat_weights)) + 1e-12
    anchors = []
    radii = []
    for anchor in candidates:
        distance = np.abs(values - float(anchor))
        near = distance <= 0.08
        support = float(np.sum(flat_weights[near]) / total_weight) if np.any(near) else 0.0
        if anchor not in {0.0, 1.0} and support < 0.003:
            continue
        radius = float(np.quantile(distance[near], 0.80)) if np.any(near) else 0.0075
        anchors.append(float(anchor))
        radii.append(float(np.clip(radius, 0.0075, 0.04)))
    anchor_array = np.asarray(anchors, dtype=np.float32)
    radius_array = np.asarray(radii, dtype=np.float32)
    return anchor_array, radius_array, {
        "snap_anchor_count": float(len(anchor_array)),
        "snap_radius_median": float(np.median(radius_array)) if len(radius_array) else 0.0,
    }


def _apply_snap_schwarzschild(
    values: np.ndarray,
    anchors: np.ndarray,
    radii: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    if len(anchors) == 0:
        return values, {"snap_changed_value_rate": 0.0, "snap_mean_abs_delta": 0.0}
    distances = np.abs(values[..., None] - anchors.reshape((1,) * values.ndim + (-1,)))
    normalized = np.divide(
        distances,
        radii.reshape((1,) * values.ndim + (-1,)),
        out=np.full_like(distances, np.inf, dtype=np.float32),
        where=radii.reshape((1,) * values.ndim + (-1,)) > 0,
    )
    eligible = distances <= radii.reshape((1,) * values.ndim + (-1,))
    normalized[~eligible] = np.inf
    choice = np.argmin(normalized, axis=-1)
    best = np.take(anchors, choice)
    changed = np.isfinite(np.min(normalized, axis=-1))
    snapped = values.copy()
    snapped[changed] = best[changed]
    delta = np.abs(snapped - values)
    return snapped.astype(np.float32), {
        "snap_changed_value_rate": float(np.mean(changed)),
        "snap_mean_abs_delta": float(np.mean(delta)),
    }


def _finalize_experiment9(values: np.ndarray, anchors: np.ndarray, radii: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    clipped = np.clip(values, 0.0, 1.0).astype(np.float32)
    return _apply_snap_schwarzschild(clipped, anchors, radii)


def _align_affine_np(
    targets: np.ndarray,
    codes: np.ndarray,
    *,
    gain_enabled: bool,
    offset_enabled: bool,
) -> tuple[AlignmentResult, np.ndarray]:
    aligned = exact_align_cpu(
        targets,
        codes,
        gain_bounds=GAIN_BOUNDS,
        fixed_gain=None if gain_enabled else 1.0,
    )
    shifted = circular_shift(
        np.broadcast_to(codes[None], (len(targets), *codes.shape)).reshape(len(targets) * len(codes), codes.shape[-1]),
        aligned.phase.reshape(len(targets) * len(codes)),
    ).reshape(len(targets), len(codes), codes.shape[-1])
    prediction = shifted * aligned.gain[:, :, None]
    offsets = _offset_for_prediction_np(targets, prediction, offset_enabled)
    if offset_enabled:
        prediction = prediction + offsets[:, :, None]
        aligned.error = np.mean((targets[:, None, :] - prediction) ** 2, axis=2)
    offsets[:, 0] = 0.0
    aligned.phase[:, 0] = 0.0
    aligned.gain[:, 0] = 0.0
    return aligned, offsets.astype(np.float32)


def _experiment9_align_target(
    values: np.ndarray,
    policy: Experiment7Policy,
    *,
    target: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if _normalization_enabled(policy, target):
        return _range_normalize_np(values, target=target)
    return (
        values.astype(np.float32),
        np.zeros(len(values), dtype=np.float32),
        np.ones(len(values), dtype=np.float32),
        np.zeros(len(values), dtype=bool),
    )


def _experiment9_head_scalars(policy: Experiment7Policy, stage_count: int) -> int:
    scalars = 1 + stage_count  # base phase + residual phases
    if _affine_applies(policy, "base"):
        scalars += int("gain" in policy.affine_modulation) + int("offset" in policy.affine_modulation)
    if _affine_applies(policy, "residual"):
        scalars += stage_count * (
            int("gain" in policy.affine_modulation) + int("offset" in policy.affine_modulation)
        )
    return scalars


def _experiment9_phase_head_outputs(width: int, residual_depth: int) -> int:
    return 33 + int(residual_depth) * (int(width) + 1)


def _closest_even_depth_for_head_budget(*, width: int, target_head_outputs: int) -> int:
    raw_depth = max(2.0, (float(target_head_outputs) - 33.0) / float(int(width) + 1))
    lower = max(2, int(np.floor(raw_depth / 2.0)) * 2)
    upper = max(2, int(np.ceil(raw_depth / 2.0)) * 2)
    candidates = sorted({lower, upper})
    return min(
        candidates,
        key=lambda depth: (
            abs(_experiment9_phase_head_outputs(width, depth) - int(target_head_outputs)),
            depth,
        ),
    )


def _xpu_shift_impl() -> str:
    value = os.environ.get("LFO_XPU_SHIFT_IMPL", "roll_bank").strip().lower()
    if value not in {"roll_bank", "gather"}:
        raise RuntimeError(f"unsupported LFO_XPU_SHIFT_IMPL={value!r}; use roll_bank or gather")
    return value


def _roll_bank_torch(codes):
    import torch

    width = int(codes.shape[-1])
    return torch.stack([torch.roll(codes, shifts=shift, dims=-1) for shift in range(width)], dim=-2)


def _circular_shift_roll_bank_torch(codes, phase, roll_bank=None):
    import torch

    values = codes.to(dtype=torch.float32)
    was_vector = values.ndim == 1
    if was_vector:
        values = values[None, :]
    leading_shape = values.shape[:-1]
    width = int(values.shape[-1])
    flat = values.reshape(-1, width)
    phase_flat = torch.broadcast_to(phase.to(device=values.device, dtype=torch.float32), leading_shape).reshape(-1)
    if roll_bank is None:
        bank = _roll_bank_torch(flat)
    else:
        bank = roll_bank.reshape(-1, width, width)
    phase_position = phase_flat * width
    left = torch.remainder(torch.floor(phase_position).to(torch.long), width)
    fraction = phase_position - torch.floor(phase_position)
    positions = torch.arange(width, device=values.device)
    right = torch.remainder(left + 1, width)
    weights = (
        (positions[None, :] == left[:, None]).to(torch.float32) * (1.0 - fraction[:, None])
        + (positions[None, :] == right[:, None]).to(torch.float32) * fraction[:, None]
    )
    shifted = torch.einsum("bi,biw->bw", weights, bank).reshape(*leading_shape, width)
    return shifted[0] if was_vector else shifted


def _circular_shift_xpu_torch(codes, phase, roll_bank=None):
    if _xpu_shift_impl() == "gather":
        return circular_shift_torch(codes, phase)
    return _circular_shift_roll_bank_torch(codes, phase, roll_bank=roll_bank)


def _sample_hash(train_indices: np.ndarray, validation_indices: np.ndarray, *, fraction: float, seed: int) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(train_indices, dtype=np.int64).tobytes())
    digest.update(np.asarray(validation_indices, dtype=np.int64).tobytes())
    digest.update(str(float(fraction)).encode("utf-8"))
    digest.update(str(int(seed)).encode("utf-8"))
    return digest.hexdigest()[:16]


def _sample_indices(indices: np.ndarray, *, fraction: float, seed: int, salt: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int32)
    count = max(1, int(round(len(indices) * fraction)))
    rng = np.random.default_rng(int(seed) + int(salt))
    choice = np.sort(rng.choice(indices, size=min(count, len(indices)), replace=False))
    return choice.astype(np.int32)


def _apply_screen_sample(
    output_dir: Path,
    dataset: CurveDataset,
    *,
    fraction: float,
    seed: int,
) -> tuple[CurveDataset, str]:
    sample_path = output_dir / "screen_sample_indices.npz"
    if sample_path.exists():
        try:
            cached = np.load(sample_path)
            if (
                int(cached["seed"]) == int(seed)
                and abs(float(cached["fraction"]) - float(fraction)) < 1e-12
                and int(cached["train_source_count"]) == len(dataset.train_indices)
                and int(cached["validation_source_count"]) == len(dataset.validation_indices)
            ):
                train_indices = cached["train_indices"].astype(np.int32, copy=False)
                validation_indices = cached["validation_indices"].astype(np.int32, copy=False)
                sample_hash = str(cached["sample_hash"].item())
                return replace(dataset, train_indices=train_indices, validation_indices=validation_indices), sample_hash
        except Exception as exc:
            _log_progress(f"Ignoring invalid Experiment 8 sample cache {sample_path}: {exc}")

    train_indices = _sample_indices(dataset.train_indices, fraction=fraction, seed=seed, salt=11)
    validation_indices = _sample_indices(dataset.validation_indices, fraction=fraction, seed=seed, salt=29)
    sample_hash = _sample_hash(train_indices, validation_indices, fraction=fraction, seed=seed)
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    _write_npz_atomic(
        sample_path,
        train_indices=train_indices,
        validation_indices=validation_indices,
        seed=np.asarray(seed, dtype=np.int64),
        fraction=np.asarray(float(fraction), dtype=np.float64),
        train_source_count=np.asarray(len(dataset.train_indices), dtype=np.int64),
        validation_source_count=np.asarray(len(dataset.validation_indices), dtype=np.int64),
        sample_hash=np.asarray(sample_hash),
    )
    return replace(dataset, train_indices=train_indices, validation_indices=validation_indices), sample_hash


def _checkpoint_dir(output_dir: Path, job_id: str) -> Path:
    return output_dir / "checkpoints" / _safe_id(job_id)


def _checkpoint_done(output_dir: Path, job_id: str) -> bool:
    directory = _checkpoint_dir(output_dir, job_id)
    return (
        (directory / "DONE.txt").exists()
        and (directory / "result.csv").exists()
        and (directory / "paths.csv").exists()
        and (directory / "usage.csv").exists()
        and (directory / "chain" / "manifest.json").exists()
    )


def _policy_payload(policy: Experiment7Policy) -> dict[str, object]:
    payload = {
        "construction_strategy": policy.construction_strategy,
        "modifier_policy": policy.modifier_policy,
        "stock_rmse_threshold": policy.stock_rmse_threshold,
        "stock_pointwise_threshold": policy.stock_pointwise_threshold,
        "base_medoids_source": policy.base_medoids_source,
        "selection_cutover_layer": policy.selection_cutover_layer,
    }
    if policy.residual_clip_policy != "final_only":
        payload["residual_clip_policy"] = policy.residual_clip_policy
    return payload


def _write_checkpoint(
    output_dir: Path,
    job: Experiment7Job,
    chain: PhaseChain,
    result: pd.DataFrame,
    subsets: pd.DataFrame,
    paths: pd.DataFrame,
    usage: pd.DataFrame,
    construction: pd.DataFrame,
) -> None:
    directory = _checkpoint_dir(output_dir, job.job_id)
    directory.mkdir(parents=True, exist_ok=True)
    done = directory / "DONE.txt"
    if done.exists():
        done.unlink()
    chain.save(directory / "chain")
    result.to_csv(directory / "result.csv", index=False)
    subsets.to_csv(directory / "subsets.csv", index=False)
    paths.to_csv(directory / "paths.csv", index=False)
    usage.to_csv(directory / "usage.csv", index=False)
    construction.to_csv(directory / "construction.csv", index=False)
    manifest = {
        "job_id": job.job_id,
        "experiment": job.experiment,
        "label": job.label,
        "k": job.k,
        "d": job.d,
        "residual_width": job.k,
        "residual_depth": job.residual_depth,
        "modifier_label": job.modifier_label or job.policy.modifier_policy,
        "residual_clip_policy": job.policy.residual_clip_policy,
        "sample_fraction": job.sample_fraction,
        "sample_seed": job.sample_seed,
        "estimated_peak_memory_mb": job.estimated_peak_memory_mb,
        "actual_residual_stages": len(chain.stages),
        "eval_resolution": job.eval_resolution,
        "beam_width": job.beam_width,
        "batch_size": job.batch_size,
        "seed": job.seed,
        "weight": job.weight,
        "policy": _policy_payload(job.policy),
        "completed_at": _now_iso(),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    done.write_text(f"{_now_iso()}\n", encoding="utf-8")


def _load_checkpoint(output_dir: Path, job_id: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    directory = _checkpoint_dir(output_dir, job_id)
    return (
        pd.read_csv(directory / "result.csv", low_memory=False),
        pd.read_csv(directory / "subsets.csv", low_memory=False),
        pd.read_csv(directory / "paths.csv", low_memory=False),
        pd.read_csv(directory / "usage.csv", low_memory=False),
        pd.read_csv(directory / "construction.csv", low_memory=False),
    )


def _file_fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _training_cache_key(
    *,
    experiment: str,
    policy: Experiment7Policy,
    k: int,
    d: int,
    eval_resolution: int,
    seed: int,
    quick: bool,
    catalog_path: Path,
    codebook_path: Path,
    sample_hash: str = "",
) -> str:
    payload = {
        "experiment": experiment,
        "policy": _policy_payload(replace(policy, modifier_policy="none")),
        "k": int(k),
        "d": int(d),
        "eval_resolution": int(eval_resolution),
        "seed": int(seed),
        "quick": bool(quick),
        "catalog": _file_fingerprint(catalog_path),
        "codebook": _file_fingerprint(codebook_path),
    }
    if sample_hash:
        payload["sample_hash"] = sample_hash
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:20]


def _training_cache_dir(output_dir: Path, cache_key: str) -> Path:
    return output_dir / "training_cache" / cache_key


def _cache_every() -> int:
    return max(0, int(os.environ.get("LFO_CACHE_EVERY", "10")))


def _load_trained_cache(output_dir: Path, cache_key: str) -> tuple[PhaseChain, pd.DataFrame, float] | None:
    directory = _training_cache_dir(output_dir, cache_key)
    done = directory / "DONE_TRAINING.txt"
    if not done.exists() or not (directory / "chain" / "manifest.json").exists():
        return None
    chain = PhaseChain.load(directory / "chain")
    construction = pd.read_csv(directory / "construction.csv", low_memory=False)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    return chain, construction, float(manifest.get("elapsed_seconds", 0.0) or 0.0)


def _write_training_cache(
    output_dir: Path,
    cache_key: str,
    *,
    chain: PhaseChain,
    construction: pd.DataFrame,
    prefix: np.ndarray | None = None,
    phase: str,
    elapsed_seconds: float,
    complete: bool,
) -> None:
    directory = _training_cache_dir(output_dir, cache_key)
    directory.mkdir(parents=True, exist_ok=True)
    done = directory / "DONE_TRAINING.txt"
    if done.exists():
        done.unlink()
    chain.save(directory / "chain")
    construction.to_csv(directory / "construction.csv", index=False)
    if prefix is not None:
        np.save(directory / "prefix.npy", prefix.astype(np.float32))
    manifest = {
        "cache_key": cache_key,
        "phase": phase,
        "complete": bool(complete),
        "elapsed_seconds": float(elapsed_seconds),
        "updated_at": _now_iso(),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if complete:
        done.write_text(f"{_now_iso()}\n", encoding="utf-8")


def _load_partial_training_cache(output_dir: Path, cache_key: str) -> tuple[PhaseChain, pd.DataFrame, np.ndarray, dict[str, object]] | None:
    directory = _training_cache_dir(output_dir, cache_key)
    manifest_path = directory / "manifest.json"
    prefix_path = directory / "prefix.npy"
    if not manifest_path.exists() or not prefix_path.exists() or not (directory / "chain" / "manifest.json").exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if bool(manifest.get("complete")):
        return None
    chain = PhaseChain.load(directory / "chain")
    construction = pd.read_csv(directory / "construction.csv", low_memory=False)
    prefix = np.load(prefix_path).astype(np.float32, copy=False)
    return chain, construction, prefix, manifest


def _inflight_cache_path(cache_output_dir: Path | None, cache_key: str | None, name: str) -> Path | None:
    if cache_output_dir is None or cache_key is None:
        return None
    safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)
    return _training_cache_dir(cache_output_dir, cache_key) / "inflight" / f"{safe_name}.npz"


def _write_npz_atomic(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.savez(handle, **arrays)
    tmp.replace(path)


def _encoding_cache_path(output_dir: Path, job_id: str) -> Path:
    return output_dir / "encoding_cache" / _safe_id(job_id) / "encoding.npz"


def _load_encoding_progress(
    cache_path: Path | None,
    *,
    n: int,
    stage_count: int,
    batch_size: int,
) -> tuple[int, np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray], list[np.ndarray]] | None:
    if cache_path is None or not cache_path.exists():
        return None
    try:
        cached = np.load(cache_path)
        if (
            int(cached["n"]) != n
            or int(cached["stage_count"]) != stage_count
            or int(cached["batch_size"]) != batch_size
        ):
            return None
        completed = min(n, int(cached["completed"]))
        stage_indices = cached["stage_indices"]
        stage_phases = cached["stage_phases"]
        stage_gains = cached["stage_gains"]
        if stage_indices.shape != (stage_count, n):
            return None
        _log_progress(f"Resuming cached validation encoding {cache_path.parent.name}: {completed:,}/{n:,}")
        return (
            completed,
            cached["base_indices"].astype(np.int16, copy=False),
            cached["base_phases"].astype(np.float32, copy=False),
            [stage_indices[i].astype(np.int16, copy=False) for i in range(stage_count)],
            [stage_phases[i].astype(np.float32, copy=False) for i in range(stage_count)],
            [stage_gains[i].astype(np.float32, copy=False) for i in range(stage_count)],
        )
    except Exception as exc:
        _log_progress(f"Ignoring invalid validation encoding cache {cache_path}: {exc}")
        return None


def _write_encoding_progress(
    cache_path: Path | None,
    *,
    completed: int,
    n: int,
    stage_count: int,
    batch_size: int,
    out_base: np.ndarray,
    out_base_phase: np.ndarray,
    out_indices: list[np.ndarray],
    out_phases: list[np.ndarray],
    out_gains: list[np.ndarray],
) -> None:
    if cache_path is None:
        return
    _write_npz_atomic(
        cache_path,
        completed=np.asarray(completed, dtype=np.int64),
        n=np.asarray(n, dtype=np.int64),
        stage_count=np.asarray(stage_count, dtype=np.int64),
        batch_size=np.asarray(batch_size, dtype=np.int64),
        base_indices=out_base,
        base_phases=out_base_phase,
        stage_indices=np.stack(out_indices) if out_indices else np.empty((0, n), dtype=np.int16),
        stage_phases=np.stack(out_phases) if out_phases else np.empty((0, n), dtype=np.float32),
        stage_gains=np.stack(out_gains) if out_gains else np.empty((0, n), dtype=np.float32),
    )


def _estimate_peak_memory_mb(
    *,
    train_count: int,
    validation_count: int,
    resolution: int,
    residual_width: int,
    residual_depth: int,
    beam_width: int,
    batch_size: int,
) -> float:
    float_bytes = 4
    int_bytes = 2
    train_stage_batch = max(1, int(os.environ.get("LFO_TRAIN_STAGE_BATCH_SIZE", "256")))
    eval_rows = max(1, int(batch_size)) * max(1, int(beam_width)) * max(1, int(residual_width))
    train_rows = train_stage_batch * max(1, int(residual_width))
    eval_scratch = eval_rows * int(resolution) * float_bytes * 8
    train_scratch = train_rows * int(resolution) * float_bytes * 6
    dataset_arrays = (int(train_count) + int(validation_count)) * int(resolution) * float_bytes * 3
    dictionary_arrays = (32 + int(residual_depth) * int(residual_width) * len(TOPOLOGY_NAMES)) * int(resolution) * float_bytes
    encoding_arrays = int(validation_count) * (2 * int(residual_depth) * float_bytes + int(residual_depth) * int_bytes + 8)
    return float((eval_scratch + train_scratch + dataset_arrays + dictionary_arrays + encoding_arrays) / (1024 * 1024))


def _memory_budget_mb() -> float:
    return float(os.environ.get("LFO_EXPERIMENT8_MEMORY_BUDGET_MB", "4096"))


def _check_memory_budget(job: Experiment7Job, *, train_count: int, validation_count: int) -> float:
    if job.experiment != "8":
        return 0.0
    estimate = _estimate_peak_memory_mb(
        train_count=train_count,
        validation_count=validation_count,
        resolution=job.eval_resolution,
        residual_width=job.k,
        residual_depth=job.residual_depth,
        beam_width=job.beam_width,
        batch_size=job.batch_size,
    )
    budget = _memory_budget_mb()
    if estimate > budget:
        suggested_batch = max(1, job.batch_size // 2)
        suggested_train_batch = max(1, int(os.environ.get("LFO_TRAIN_STAGE_BATCH_SIZE", "256")) // 2)
        raise RuntimeError(
            f"{job.label}: estimated peak memory {estimate:.1f} MB exceeds budget {budget:.1f} MB. "
            f"Try --batch-size {suggested_batch} and --train-stage-batch-size {suggested_train_batch}, "
            "or raise LFO_EXPERIMENT8_MEMORY_BUDGET_MB after checking available memory."
        )
    return estimate


def _prepare_dataset_for_job(output_dir: Path, dataset: CurveDataset, job: Experiment7Job) -> tuple[CurveDataset, str]:
    if job.experiment == "8":
        sample_seed = job.sample_seed if job.sample_seed is not None else job.seed
        return _apply_screen_sample(
            output_dir,
            dataset,
            fraction=job.sample_fraction,
            seed=sample_seed,
        )
    return dataset, ""


def _jobs_signature(jobs: list[Experiment7Job]) -> str:
    payload = [
        {
            "job_id": job.job_id,
            "experiment": job.experiment,
            "k": job.k,
            "d": job.d,
            "eval_resolution": job.eval_resolution,
            "beam_width": job.beam_width,
            "seed": job.seed,
            "policy": _policy_payload(job.policy),
            "modifier_label": job.modifier_label,
            "sample_fraction": job.sample_fraction,
            "sample_seed": job.sample_seed,
        }
        for job in jobs
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _progress_from_jobs(output_dir: Path, jobs: list[Experiment7Job], experiment: str, *, started_at: str | None = None) -> dict[str, object]:
    rows = []
    completed = 0
    completed_weight = 0.0
    total_weight = float(sum(job.weight for job in jobs))
    for job in jobs:
        done = _checkpoint_done(output_dir, job.job_id)
        completed += int(done)
        completed_weight += job.weight if done else 0.0
        rows.append(
            {
                "job_id": job.job_id,
                "label": job.label,
                "k": job.k,
                "d": job.d,
                "residual_width": job.k,
                "residual_depth": job.residual_depth,
                "modifier_label": job.modifier_label or job.policy.modifier_policy,
                "residual_clip_policy": job.policy.residual_clip_policy,
                "sample_fraction": job.sample_fraction,
                "sample_seed": job.sample_seed,
                "eval_resolution": job.eval_resolution,
                "weight": job.weight,
                "status": "completed" if done else "pending",
            }
        )
    return {
        "experiment": experiment,
        "run_signature": _jobs_signature(jobs),
        "started_at": started_at or _now_iso(),
        "total_jobs": len(jobs),
        "completed_jobs": completed,
        "total_weight": total_weight,
        "completed_weight": completed_weight,
        "jobs": rows,
    }


def experiment7_status(output_dir: Path, experiment: str) -> str:
    progress_path = output_dir / "progress.json"
    if not progress_path.exists():
        return f"No Experiment {experiment} progress manifest found at {progress_path}"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    jobs = progress.get("jobs", [])
    completed_jobs = 0
    completed_weight = 0.0
    total_weight = float(progress.get("total_weight", 0.0) or 0.0)
    for job in jobs:
        done = _checkpoint_done(output_dir, str(job["job_id"]))
        job["status"] = "completed" if done else job.get("status", "pending")
        if done:
            completed_jobs += 1
            completed_weight += float(job.get("weight", 0.0) or 0.0)
    total_jobs = int(progress.get("total_jobs", len(jobs)) or len(jobs))
    elapsed_seconds = None
    started_at = progress.get("started_at")
    if isinstance(started_at, str):
        try:
            elapsed_seconds = (
                datetime.now(timezone.utc).astimezone() - datetime.fromisoformat(started_at)
            ).total_seconds()
        except ValueError:
            elapsed_seconds = None
    eta_seconds = None
    if elapsed_seconds is not None and completed_weight > 0 and total_weight > completed_weight:
        eta_seconds = elapsed_seconds * (total_weight - completed_weight) / completed_weight
    count_fraction = completed_jobs / total_jobs if total_jobs else 0.0
    weight_fraction = completed_weight / total_weight if total_weight > 0 else 0.0
    running = [job for job in jobs if job.get("status") == "running"]
    pending = [job for job in jobs if job.get("status") != "completed"]
    marker = output_dir / f"COMPLETED_EXPERIMENT_{experiment}.txt"
    lines = [
        f"Experiment {experiment} progress: {completed_jobs}/{total_jobs} jobs ({count_fraction:.1%})",
        f"Estimated workload: {completed_weight:.2f}/{total_weight:.2f} units ({weight_fraction:.1%})",
        f"Elapsed: {_format_duration(elapsed_seconds)}",
        f"Estimated time remaining: {_format_duration(eta_seconds)}",
        f"Completion marker: {'yes' if marker.exists() else 'no'}",
    ]
    if running:
        lines.append("Running:")
        lines.extend(f"  - {job.get('label', job.get('job_id'))}" for job in running[:8])
    if pending:
        lines.append("Next pending:")
        lines.extend(f"  - {job.get('label', job.get('job_id'))}" for job in pending[:8])
    return "\n".join(lines)


def _top_beam(error: np.ndarray, beam_width: int) -> np.ndarray:
    width = min(beam_width, error.shape[1])
    choices = np.argpartition(error, width - 1, axis=1)[:, :width]
    rows = np.arange(error.shape[0])[:, None]
    order = np.argsort(error[rows, choices], axis=1)
    return np.take_along_axis(choices, order, axis=1)


def _align_stage_grouped(residual: np.ndarray, stage: np.ndarray, conditions: np.ndarray, beam_width: int) -> AlignmentResult:
    dictionaries = stage[np.repeat(conditions, beam_width)]
    return _exact_align(residual, dictionaries)


def _decode_raw(
    chain: PhaseChain,
    encoding: PhaseEncoding,
    conditions: np.ndarray,
    *,
    residual_clip_policy: str = "final_only",
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    base = circular_shift(chain.bases[encoding.base_indices], encoding.base_phases)
    raw = base.copy()
    additions: list[np.ndarray] = []
    rows = np.arange(len(raw))
    raw = _apply_decoder_step_np(np.zeros_like(raw), raw, residual_clip_policy, is_base=True)
    for stage_i, stage in enumerate(chain.stages):
        code = stage[conditions, encoding.stage_indices[stage_i]]
        addition = circular_shift(code, encoding.stage_phases[stage_i])
        addition *= encoding.stage_gains[stage_i][:, None]
        raw = _apply_decoder_step_np(raw, addition, residual_clip_policy)
        additions.append(addition)
    return raw, base, additions


def _apply_modifier_policy(
    targets: np.ndarray,
    raw: np.ndarray,
    base: np.ndarray,
    policy: str,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
    n = len(targets)
    base_gain = np.ones(n, dtype=np.float32)
    global_offset = np.zeros(n, dtype=np.float32)
    adjusted = raw.astype(np.float32)
    if policy in {"global_offset", "base_gain_global_offset"}:
        offset = np.clip(np.mean(targets - adjusted, axis=1), -1.0, 1.0)
        global_offset = offset.astype(np.float32)
        adjusted = adjusted + global_offset[:, None]
    if policy in {"base_gain", "base_gain_global_offset"}:
        residual_sum = adjusted - base
        grid = np.linspace(0.0, 2.0, 81, dtype=np.float32)
        best_error = np.full(n, np.inf, dtype=np.float64)
        best_gain = np.ones(n, dtype=np.float32)
        best_recon = np.clip(adjusted, 0.0, 1.0)
        for value in grid:
            candidate_raw = residual_sum + value * base
            candidate = np.clip(candidate_raw, 0.0, 1.0)
            error = np.mean((targets - candidate) ** 2, axis=1)
            improved = error < best_error
            best_error[improved] = error[improved]
            best_gain[improved] = value
            best_recon[improved] = candidate[improved]
        base_gain = best_gain
        adjusted = residual_sum + base_gain[:, None] * base
        if policy == "base_gain_global_offset":
            # One deterministic refinement pass after base gain.
            offset = np.clip(np.mean(targets - adjusted, axis=1), -1.0, 1.0)
            global_offset = (global_offset + offset).astype(np.float32)
            adjusted = adjusted + offset[:, None]
    reconstructed = np.clip(adjusted, 0.0, 1.0).astype(np.float32)
    values = {"base_gain": base_gain, "global_offset": global_offset}
    stats = {
        "base_gain_saturated_share": float(np.mean((base_gain <= 1e-6) | (base_gain >= 2.0 - 1e-6))),
        "global_offset_abs_median": float(np.median(np.abs(global_offset))),
    }
    return reconstructed, values, stats


def _encode_final_clip_beam_torch(
    targets: np.ndarray,
    chain: PhaseChain,
    conditions: np.ndarray,
    *,
    beam_width: int,
    batch_size: int,
    device: str,
    progress_label: str | None = None,
    progress_offset: float = 0.0,
    progress_scale: float = 1.0,
    cache_path: Path | None = None,
    residual_clip_policy: str = "final_only",
) -> tuple[np.ndarray, PhaseEncoding, dict[str, float]]:
    import torch

    n, resolution = targets.shape
    stage_count = len(chain.stages)
    out_base = np.empty(n, dtype=np.int16)
    out_base_phase = np.zeros(n, dtype=np.float32)
    out_indices = [np.empty(n, dtype=np.int16) for _ in chain.stages]
    out_phases = [np.zeros(n, dtype=np.float32) for _ in chain.stages]
    out_gains = [np.zeros(n, dtype=np.float32) for _ in chain.stages]
    completed = 0
    cached = _load_encoding_progress(cache_path, n=n, stage_count=stage_count, batch_size=batch_size)
    if cached is not None:
        completed, out_base, out_base_phase, out_indices, out_phases, out_gains = cached
    bases_t = torch.as_tensor(chain.bases, dtype=torch.float32, device=device)
    bases_roll_bank = _roll_bank_torch(bases_t)
    stages_t = [torch.as_tensor(stage, dtype=torch.float32, device=device) for stage in chain.stages]
    stages_roll_bank = [_roll_bank_torch(stage) for stage in stages_t]
    total_batches = max(1, math.ceil(n / batch_size))
    total_stage_steps = total_batches * max(1, stage_count)
    completed_stage_steps = 0
    eval_log_every = max(1, int(os.environ.get("LFO_EVAL_PROGRESS_EVERY", "10")))

    with torch.no_grad():
        for batch_index, start in enumerate(range(0, n, batch_size), start=1):
            stop = min(start + batch_size, n)
            if stop <= completed:
                completed_stage_steps += max(1, stage_count)
                continue
            batch_t = torch.as_tensor(targets[start:stop], dtype=torch.float32, device=device)
            conditions_t = torch.as_tensor(conditions[start:stop], dtype=torch.long, device=device)
            b = batch_t.shape[0]
            should_log_batch = (
                progress_label is not None
                and (batch_index == 1 or batch_index == total_batches or batch_index % eval_log_every == 0)
            )
            if should_log_batch:
                effort = progress_offset + progress_scale * (completed_stage_steps / total_stage_steps)
                _log_progress(
                    f"{progress_label}: eval batch {batch_index}/{total_batches} "
                    f"curves {start + 1:,}-{stop:,}/{n:,}; effort~{effort:.1%}"
                )

            base_align = exact_align_torch_tensors(batch_t, bases_t, fixed_gain=1.0, gain_bounds=GAIN_BOUNDS, device=device)
            beam = min(beam_width, base_align.error.shape[1])
            base_error, base_choices = torch.topk(base_align.error, k=beam, dim=1, largest=False, sorted=True)
            del base_error
            base_phases = torch.gather(base_align.phase, 1, base_choices)
            raw_prefix = _circular_shift_xpu_torch(
                bases_t[base_choices.reshape(-1)],
                base_phases.reshape(-1),
                roll_bank=bases_roll_bank[base_choices.reshape(-1)],
            ).reshape(b, beam, resolution)
            raw_prefix = _apply_decoder_step_torch(
                torch.zeros_like(raw_prefix),
                raw_prefix,
                residual_clip_policy,
                is_base=True,
            )
            base_paths = base_choices.to(torch.long)
            base_phase_paths = base_phases.to(torch.float32)
            max_paths = max(beam, beam_width)
            index_paths = torch.empty((b, max_paths, stage_count), dtype=torch.long, device=device)
            phase_paths = torch.empty((b, max_paths, stage_count), dtype=torch.float32, device=device)
            gain_paths = torch.empty((b, max_paths, stage_count), dtype=torch.float32, device=device)

            for stage_i, (stage_t, stage_bank_t) in enumerate(zip(stages_t, stages_roll_bank, strict=True)):
                should_log_stage = should_log_batch and (
                    stage_i == 0 or stage_i + 1 == stage_count or (stage_i + 1) % 4 == 0
                )
                if should_log_stage:
                    effort = progress_offset + progress_scale * (completed_stage_steps / total_stage_steps)
                    _log_progress(
                        f"{progress_label}: eval batch {batch_index}/{total_batches}, "
                        f"stage {stage_i + 1}/{stage_count} ({chain.stage_labels[stage_i]}); "
                        f"effort~{effort:.1%}"
                    )
                bw = raw_prefix.shape[1]
                k = stage_t.shape[1]
                residual = (batch_t[:, None, :] - raw_prefix).reshape(b * bw, resolution)
                dictionaries = stage_t[conditions_t]
                repeated = dictionaries.repeat_interleave(bw, dim=0)
                aligned = exact_align_torch_tensors(residual, repeated, gain_bounds=GAIN_BOUNDS, device=device)
                repeated_bank = stage_bank_t[conditions_t].repeat_interleave(bw, dim=0)
                shifted = _circular_shift_xpu_torch(repeated, aligned.phase, roll_bank=repeated_bank)
                additions = (shifted * aligned.gain[:, :, None]).reshape(b, bw, k, resolution)
                candidate_state = _apply_decoder_step_torch(
                    raw_prefix[:, :, None, :],
                    additions,
                    residual_clip_policy,
                )
                candidate_recon = torch.clamp(candidate_state, 0.0, 1.0)
                mse = torch.mean((batch_t[:, None, None, :] - candidate_recon) ** 2, dim=3)
                previous = torch.mean((batch_t[:, None, :] - torch.clamp(raw_prefix, 0.0, 1.0)) ** 2, dim=2)
                mse[:, :, 0] = torch.minimum(mse[:, :, 0], previous)
                candidate_state[:, :, 0, :] = raw_prefix
                phase3 = aligned.phase.reshape(b, bw, k)
                gain3 = aligned.gain.reshape(b, bw, k)
                phase3[:, :, 0] = 0.0
                gain3[:, :, 0] = 0.0

                flat = mse.reshape(b, -1)
                next_width = min(beam_width, flat.shape[1])
                _, choice = torch.topk(flat, k=next_width, dim=1, largest=False, sorted=True)
                parent = torch.div(choice, k, rounding_mode="floor")
                code = choice % k
                batch_rows = torch.arange(b, device=device)[:, None]
                raw_prefix = candidate_state[batch_rows, parent, code]
                base_paths = torch.gather(base_paths, 1, parent)
                base_phase_paths = torch.gather(base_phase_paths, 1, parent)
                if stage_i:
                    gather_previous = parent[:, :, None].expand(-1, -1, stage_i)
                    index_paths[:, :next_width, :stage_i] = torch.gather(index_paths[:, :bw, :stage_i], 1, gather_previous)
                    phase_paths[:, :next_width, :stage_i] = torch.gather(phase_paths[:, :bw, :stage_i], 1, gather_previous)
                    gain_paths[:, :next_width, :stage_i] = torch.gather(gain_paths[:, :bw, :stage_i], 1, gather_previous)
                selected_phase = phase3[batch_rows, parent, code]
                selected_gain = gain3[batch_rows, parent, code]
                index_paths[:, :next_width, stage_i] = code
                phase_paths[:, :next_width, stage_i] = selected_phase
                gain_paths[:, :next_width, stage_i] = selected_gain
                raw_prefix = raw_prefix[:, :next_width]
                base_paths = base_paths[:, :next_width]
                base_phase_paths = base_phase_paths[:, :next_width]
                del (
                    residual,
                    dictionaries,
                    repeated,
                    repeated_bank,
                    aligned,
                    shifted,
                    additions,
                    candidate_raw,
                    candidate_state,
                    candidate_recon,
                    mse,
                    previous,
                    phase3,
                    gain3,
                    flat,
                    choice,
                    parent,
                    code,
                    batch_rows,
                    selected_phase,
                    selected_gain,
                )
                _release_xpu_temporaries(device)
                completed_stage_steps += 1

            out_base[start:stop] = base_paths[:, 0].detach().cpu().numpy().astype(np.int16)
            out_base_phase[start:stop] = base_phase_paths[:, 0].detach().cpu().numpy().astype(np.float32)
            final_indices = index_paths[:, 0, :stage_count].detach().cpu().numpy().astype(np.int16)
            final_phases = phase_paths[:, 0, :stage_count].detach().cpu().numpy().astype(np.float32)
            final_gains = gain_paths[:, 0, :stage_count].detach().cpu().numpy().astype(np.float32)
            for stage_i in range(stage_count):
                out_indices[stage_i][start:stop] = final_indices[:, stage_i]
                out_phases[stage_i][start:stop] = final_phases[:, stage_i] % 1.0
                out_gains[stage_i][start:stop] = final_gains[:, stage_i]
            if cache_path is not None and _cache_every() and (batch_index % _cache_every() == 0 or stop == n):
                _write_encoding_progress(
                    cache_path,
                    completed=stop,
                    n=n,
                    stage_count=stage_count,
                    batch_size=batch_size,
                    out_base=out_base,
                    out_base_phase=out_base_phase,
                    out_indices=out_indices,
                    out_phases=out_phases,
                    out_gains=out_gains,
                )
            del (
                batch_t,
                conditions_t,
                base_align,
                base_choices,
                base_phases,
                raw_prefix,
                base_paths,
                base_phase_paths,
                index_paths,
                phase_paths,
                gain_paths,
                final_indices,
                final_phases,
                final_gains,
            )
            _release_xpu_temporaries(device)

    encoding = PhaseEncoding(out_base, out_base_phase % 1.0, out_indices, out_phases, out_gains)
    raw, _, _ = _decode_raw(chain, encoding, conditions, residual_clip_policy=residual_clip_policy)
    reconstructed = np.clip(raw, 0.0, 1.0)
    noop = {f"stage_{i+1}_noop_rate": float(np.mean(values == 0)) for i, values in enumerate(out_indices)}
    return reconstructed.astype(np.float32), encoding, noop


def encode_final_clip_beam(
    targets: np.ndarray,
    chain: PhaseChain,
    conditions: np.ndarray,
    *,
    beam_width: int,
    batch_size: int,
    progress_label: str | None = None,
    progress_offset: float = 0.0,
    progress_scale: float = 1.0,
    cache_path: Path | None = None,
    residual_clip_policy: str = "final_only",
) -> tuple[np.ndarray, PhaseEncoding, dict[str, float]]:
    device = _torch_align_device()
    if device is not None:
        return _encode_final_clip_beam_torch(
            targets,
            chain,
            conditions,
            beam_width=beam_width,
            batch_size=batch_size,
            device=device,
            progress_label=progress_label,
            progress_offset=progress_offset,
            progress_scale=progress_scale,
            cache_path=cache_path,
            residual_clip_policy=residual_clip_policy,
        )

    n, resolution = targets.shape
    out_base = np.empty(n, dtype=np.int16)
    out_base_phase = np.zeros(n, dtype=np.float32)
    out_indices = [np.empty(n, dtype=np.int16) for _ in chain.stages]
    out_phases = [np.zeros(n, dtype=np.float32) for _ in chain.stages]
    out_gains = [np.zeros(n, dtype=np.float32) for _ in chain.stages]
    completed = 0
    cached = _load_encoding_progress(cache_path, n=n, stage_count=len(chain.stages), batch_size=batch_size)
    if cached is not None:
        completed, out_base, out_base_phase, out_indices, out_phases, out_gains = cached
    total_batches = max(1, math.ceil(n / batch_size))
    total_stage_steps = total_batches * max(1, len(chain.stages))
    completed_stage_steps = 0
    eval_log_every = max(1, int(os.environ.get("LFO_EVAL_PROGRESS_EVERY", "10")))

    for batch_index, start in enumerate(range(0, n, batch_size), start=1):
        stop = min(start + batch_size, n)
        if stop <= completed:
            completed_stage_steps += max(1, len(chain.stages))
            continue
        batch = targets[start:stop]
        b = len(batch)
        should_log_batch = (
            progress_label is not None
            and (batch_index == 1 or batch_index == total_batches or batch_index % eval_log_every == 0)
        )
        if should_log_batch:
            effort = progress_offset + progress_scale * (completed_stage_steps / total_stage_steps)
            _log_progress(
                f"{progress_label}: eval batch {batch_index}/{total_batches} "
                f"curves {start + 1:,}-{stop:,}/{n:,}; effort~{effort:.1%}"
            )
        base_align = _exact_align(batch, chain.bases, fixed_gain=1.0)
        base_choices = _top_beam(base_align.error, beam_width)
        rows = np.arange(b)[:, None]
        base_phases = base_align.phase[rows, base_choices]
        beam = base_choices.shape[1]
        raw_prefix = circular_shift(
            chain.bases[base_choices.reshape(-1)],
            base_phases.reshape(-1),
        ).reshape(b, beam, resolution).astype(np.float32)
        raw_prefix = _apply_decoder_step_np(
            np.zeros_like(raw_prefix),
            raw_prefix,
            residual_clip_policy,
            is_base=True,
        )
        base_paths = base_choices.astype(np.int16)
        base_phase_paths = base_phases.astype(np.float32)
        max_paths = max(beam, beam_width)
        index_paths = np.empty((b, max_paths, len(chain.stages)), dtype=np.int16)
        phase_paths = np.empty((b, max_paths, len(chain.stages)), dtype=np.float32)
        gain_paths = np.empty((b, max_paths, len(chain.stages)), dtype=np.float32)

        for stage_i, stage in enumerate(chain.stages):
            should_log_stage = should_log_batch and (
                stage_i == 0 or stage_i + 1 == len(chain.stages) or (stage_i + 1) % 4 == 0
            )
            if should_log_stage:
                effort = progress_offset + progress_scale * (completed_stage_steps / total_stage_steps)
                _log_progress(
                    f"{progress_label}: eval batch {batch_index}/{total_batches}, "
                    f"stage {stage_i + 1}/{len(chain.stages)} ({chain.stage_labels[stage_i]}); "
                    f"effort~{effort:.1%}"
                )
            bw = raw_prefix.shape[1]
            residual = (batch[:, None, :] - raw_prefix).reshape(b * bw, resolution)
            aligned = _align_stage_grouped(residual, stage, conditions[start:stop], bw)
            k = stage.shape[1]
            dictionaries = stage[conditions[start:stop]]
            repeated = np.repeat(dictionaries, bw, axis=0)
            shifted = circular_shift(
                repeated.reshape(b * bw * k, resolution),
                aligned.phase.reshape(b * bw * k),
            ).reshape(b * bw, k, resolution)
            additions = (shifted * aligned.gain[:, :, None]).reshape(b, bw, k, resolution)
            candidate_state = _apply_decoder_step_np(
                raw_prefix[:, :, None, :],
                additions,
                residual_clip_policy,
            )
            candidate_recon = np.clip(candidate_state, 0.0, 1.0)
            mse = np.mean((batch[:, None, None, :] - candidate_recon) ** 2, axis=3)
            previous = np.mean((batch[:, None, :] - np.clip(raw_prefix, 0.0, 1.0)) ** 2, axis=2)
            mse[:, :, 0] = np.minimum(mse[:, :, 0], previous)
            candidate_state[:, :, 0, :] = raw_prefix
            aligned.phase = aligned.phase.reshape(b, bw, k)
            aligned.gain = aligned.gain.reshape(b, bw, k)
            aligned.phase[:, :, 0] = 0.0
            aligned.gain[:, :, 0] = 0.0

            flat = mse.reshape(b, -1)
            next_width = min(beam_width, flat.shape[1])
            choice = np.argpartition(flat, next_width - 1, axis=1)[:, :next_width]
            choice = np.take_along_axis(choice, np.argsort(flat[rows, choice], axis=1), axis=1)
            parent = choice // k
            code = choice % k
            raw_prefix = candidate_state[rows, parent, code]
            base_paths = base_paths[rows, parent]
            base_phase_paths = base_phase_paths[rows, parent]
            if stage_i:
                index_paths[:, :next_width, :stage_i] = index_paths[rows, parent, :stage_i]
                phase_paths[:, :next_width, :stage_i] = phase_paths[rows, parent, :stage_i]
                gain_paths[:, :next_width, :stage_i] = gain_paths[rows, parent, :stage_i]
            index_paths[:, :next_width, stage_i] = code.astype(np.int16)
            phase_paths[:, :next_width, stage_i] = aligned.phase[rows, parent, code].astype(np.float32)
            gain_paths[:, :next_width, stage_i] = aligned.gain[rows, parent, code].astype(np.float32)
            raw_prefix = raw_prefix[:, :next_width]
            base_paths = base_paths[:, :next_width]
            base_phase_paths = base_phase_paths[:, :next_width]
            completed_stage_steps += 1

        out_base[start:stop] = base_paths[:, 0]
        out_base_phase[start:stop] = base_phase_paths[:, 0]
        for stage_i in range(len(chain.stages)):
            out_indices[stage_i][start:stop] = index_paths[:, 0, stage_i]
            out_phases[stage_i][start:stop] = phase_paths[:, 0, stage_i] % 1.0
            out_gains[stage_i][start:stop] = gain_paths[:, 0, stage_i]
        if cache_path is not None and _cache_every() and (batch_index % _cache_every() == 0 or stop == n):
            _write_encoding_progress(
                cache_path,
                completed=stop,
                n=n,
                stage_count=len(chain.stages),
                batch_size=batch_size,
                out_base=out_base,
                out_base_phase=out_base_phase,
                out_indices=out_indices,
                out_phases=out_phases,
                out_gains=out_gains,
            )

    encoding = PhaseEncoding(out_base, out_base_phase % 1.0, out_indices, out_phases, out_gains)
    raw, _, _ = _decode_raw(chain, encoding, conditions, residual_clip_policy=residual_clip_policy)
    reconstructed = np.clip(raw, 0.0, 1.0)
    noop = {f"stage_{i+1}_noop_rate": float(np.mean(values == 0)) for i, values in enumerate(out_indices)}
    return reconstructed.astype(np.float32), encoding, noop


def _group_representatives(dataset: CurveDataset, indices: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    frame = dataset.frame.iloc[indices]
    rows = []
    representatives = []
    for signature, group in frame.groupby("shape_signature", sort=False):
        local_indices = group.index.to_numpy(dtype=np.int32)
        dataset_indices = np.asarray(local_indices, dtype=np.int32)
        count = len(dataset_indices)
        curves = dataset.curves[dataset_indices]
        centroid = np.mean(curves, axis=0)
        choice_local = int(np.argmin(np.mean((curves - centroid[None]) ** 2, axis=1)))
        representative = int(dataset_indices[choice_local])
        representatives.append(representative)
        rows.append(
            {
                "shape_signature": signature,
                "count": count,
                "representative_dataset_index": representative,
                "topology": TOPOLOGY_NAMES[int(dataset.topology[representative])],
            }
        )
    table = pd.DataFrame(rows).sort_values(["count", "shape_signature"], ascending=[False, True])
    return np.asarray(representatives, dtype=np.int32), table


def _stock_explained_mask(
    dataset: CurveDataset,
    stock: np.ndarray,
    train_indices: np.ndarray,
    *,
    rmse_threshold: float,
    pointwise_threshold: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    targets = dataset.curves[train_indices]
    aligned = _exact_align(targets, stock, fixed_gain=1.0)
    choice = np.argmin(aligned.error, axis=1)
    rows = np.arange(len(targets))
    shifted = circular_shift(stock[choice], aligned.phase[rows, choice])
    error = shifted - targets
    rmse = np.sqrt(np.mean(error * error, axis=1))
    max_error = np.max(np.abs(error), axis=1)
    explained = (rmse <= rmse_threshold) & (max_error <= pointwise_threshold)
    audit = pd.DataFrame(
        {
            "dataset_index": train_indices,
            "stock_code": choice,
            "stock_phase": aligned.phase[rows, choice],
            "stock_rmse": rmse,
            "stock_max_abs_error": max_error,
            "stock_explained": explained,
        }
    )
    return explained, audit


def _build_bases(
    dataset: CurveDataset,
    stock: np.ndarray,
    *,
    policy: Experiment7Policy,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    train_indices = dataset.train_indices
    explained, stock_audit = _stock_explained_mask(
        dataset,
        stock,
        train_indices,
        rmse_threshold=policy.stock_rmse_threshold,
        pointwise_threshold=policy.stock_pointwise_threshold,
    )
    leftover = train_indices[~explained]
    if len(leftover) < 16:
        leftover = train_indices
    _, frequency = _group_representatives(dataset, leftover)
    chosen = frequency.head(16)["representative_dataset_index"].to_numpy(dtype=np.int32)
    if len(chosen) < 16:
        rng = np.random.default_rng(seed)
        pool = np.setdiff1d(train_indices, chosen, assume_unique=False)
        extra = rng.choice(pool, size=16 - len(chosen), replace=False)
        chosen = np.concatenate([chosen, extra.astype(np.int32)])
    bases = np.concatenate(
        [
            stock[:15],
            dataset.curves[chosen[:16]],
            np.zeros((1, dataset.curves.shape[1]), dtype=np.float32),
        ]
    ).astype(np.float32)
    sources = np.concatenate(
        [
            np.full(15, -2, dtype=np.int32),
            chosen[:16].astype(np.int32),
            np.asarray([-1], dtype=np.int32),
        ]
    )
    base_rows = []
    for index in range(15):
        base_rows.append({"base_index": index, "kind": "stock", "dataset_index": -2})
    for offset, source in enumerate(chosen[:16], start=15):
        base_rows.append(
            {
                "base_index": offset,
                "kind": "leftover_medoid",
                "dataset_index": int(source),
                "shape_signature": str(dataset.frame.iloc[int(source)].shape_signature),
            }
        )
    base_rows.append({"base_index": 31, "kind": "zero", "dataset_index": -1})
    base_audit = pd.DataFrame(base_rows)
    base_audit["stock_discarded_train_shapes"] = int(np.sum(explained))
    base_audit["stock_leftover_train_shapes"] = int(len(leftover))
    return bases, sources, pd.concat([base_audit, stock_audit.head(0)], ignore_index=True, sort=False)


def _candidate_pool(
    dataset: CurveDataset,
    local_indices: np.ndarray,
    residual: np.ndarray,
    strategy: str,
    rng: np.random.Generator,
    *,
    limit: int,
    topology_balanced: bool,
) -> np.ndarray:
    if len(local_indices) == 0:
        raise ValueError("empty residual candidate pool")
    frame = dataset.frame.iloc[local_indices]
    energy = np.mean(residual * residual, axis=1)
    grouped = []
    local_position = {int(dataset_index): position for position, dataset_index in enumerate(local_indices)}
    for signature, group in frame.groupby("shape_signature", sort=False):
        positions = np.asarray([local_position[int(idx)] for idx in group.index], dtype=np.int32)
        if len(positions) == 0:
            continue
        best = int(positions[np.argmax(energy[positions])])
        grouped.append(
            {
                "local": best,
                "count": len(positions),
                "energy": float(energy[best]),
                "topology": int(dataset.topology[int(local_indices[best])]),
                "signature": signature,
            }
        )
    table = pd.DataFrame(grouped)
    if table.empty:
        return np.arange(min(len(local_indices), limit), dtype=np.int32)
    if strategy == "frequency_first":
        table = table.sort_values(["count", "energy", "signature"], ascending=[False, False, True])
    elif topology_balanced:
        parts = []
        per_topology = max(1, math.ceil(limit / len(TOPOLOGY_NAMES)))
        for topology in range(len(TOPOLOGY_NAMES)):
            part = table[table["topology"] == topology].sort_values(["count", "energy"], ascending=[False, False])
            parts.append(part.head(per_topology))
        table = pd.concat(parts).drop_duplicates("local").sort_values(["count", "energy"], ascending=[False, False])
    else:
        table = table.sort_values(["energy", "count", "signature"], ascending=[False, False, True])
    selected = table["local"].to_numpy(dtype=np.int32)[:limit]
    if len(selected) < min(limit, len(local_indices)):
        remaining = np.setdiff1d(np.arange(len(local_indices), dtype=np.int32), selected, assume_unique=False)
        rng.shuffle(remaining)
        selected = np.concatenate([selected, remaining[: limit - len(selected)]])
    return selected.astype(np.int32)


def _selection_mode(
    strategy: str,
    named_layer: int,
    total_layers: int,
    *,
    selection_cutover_layer: int | None = None,
) -> str:
    if strategy == "frequency_first":
        return "frequency"
    if strategy == "greedy_global_improvement":
        return "global"
    if strategy == "tail_aware_greedy":
        return "tail"
    if strategy in {"common_then_tail", "topology_balanced_common_then_tail"}:
        cutoff = selection_cutover_layer if selection_cutover_layer is not None else max(1, total_layers // 2)
        return "global" if named_layer <= max(1, int(cutoff)) else "tail"
    raise ValueError(f"unknown construction strategy: {strategy}")


def _fit_residual_codes(
    dataset: CurveDataset,
    local_indices: np.ndarray,
    residual: np.ndarray,
    k: int,
    *,
    strategy: str,
    named_layer: int,
    total_layers: int,
    seed: int,
    topology_balanced: bool,
    selection_cutover_layer: int | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    nonzero = k - 1
    if nonzero <= 0:
        return (
            np.zeros((1, residual.shape[1]), dtype=np.float32),
            np.asarray([-1], dtype=np.int32),
            pd.DataFrame(),
        )
    limit = min(len(local_indices), max(64, nonzero * 24))
    pool = _candidate_pool(
        dataset,
        local_indices,
        residual,
        strategy,
        rng,
        limit=limit,
        topology_balanced=topology_balanced,
    )
    candidate_indices = local_indices[pool]
    candidate_values = residual[pool].astype(np.float32)
    if strategy == "frequency_first":
        chosen_pool = np.arange(min(nonzero, len(pool)), dtype=np.int32)
    else:
        aligned = _exact_align(residual, candidate_values)
        current = np.mean(residual * residual, axis=1)
        chosen_pool = []
        mode = _selection_mode(
            strategy,
            named_layer,
            total_layers,
            selection_cutover_layer=selection_cutover_layer,
        )
        for _ in range(min(nonzero, len(pool))):
            improvement = np.maximum(current[:, None] - aligned.error, 0.0)
            if mode == "tail":
                cutoff = np.quantile(current, 0.80)
                weights = np.where(current >= cutoff, 1.0 + current / (np.mean(current) + 1e-12), 0.10)
                utility = np.sum(improvement * weights[:, None], axis=0)
            else:
                utility = np.sum(improvement, axis=0)
            if chosen_pool:
                utility[np.asarray(chosen_pool, dtype=np.int32)] = -np.inf
            choice = int(np.argmax(utility))
            if not np.isfinite(utility[choice]):
                break
            chosen_pool.append(choice)
            current = np.minimum(current, aligned.error[:, choice])
        chosen_pool = np.asarray(chosen_pool, dtype=np.int32)
    if len(chosen_pool) < nonzero:
        remaining = np.setdiff1d(np.arange(len(pool), dtype=np.int32), chosen_pool, assume_unique=False)
        chosen_pool = np.concatenate([chosen_pool, remaining[: nonzero - len(chosen_pool)]])
    selected_values = candidate_values[chosen_pool[:nonzero]]
    sources = candidate_indices[chosen_pool[:nonzero]].astype(np.int32)
    codes = np.concatenate([np.zeros((1, residual.shape[1]), dtype=np.float32), selected_values]).astype(np.float32)
    source_array = np.concatenate([np.asarray([-1], dtype=np.int32), sources])
    audit = pd.DataFrame(
        {
            "code": np.arange(len(source_array)),
            "dataset_index": source_array,
            "is_noop": np.arange(len(source_array)) == 0,
            "selection_mode": _selection_mode(
                strategy,
                named_layer,
                total_layers,
                selection_cutover_layer=selection_cutover_layer,
            ),
            "selection_cutover_layer": selection_cutover_layer,
        }
    )
    return codes, source_array, audit


def _score_training_stage_cpu(
    prefix: np.ndarray,
    targets: np.ndarray,
    codes: np.ndarray,
    phase: np.ndarray,
    gain: np.ndarray,
    *,
    cache_path: Path | None = None,
    residual_clip_policy: str = "final_only",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_batch = max(1, int(os.environ.get("LFO_TRAIN_STAGE_BATCH_SIZE", "256")))
    completed = 0
    updated = np.empty_like(prefix, dtype=np.float32)
    choices = np.empty(len(targets), dtype=np.int16)
    phases = np.empty(len(targets), dtype=np.float32)
    gains = np.empty(len(targets), dtype=np.float32)
    if cache_path is not None and cache_path.exists():
        try:
            cached = np.load(cache_path)
            if tuple(cached["updated"].shape) == tuple(prefix.shape) and int(cached["target_count"]) == len(targets):
                completed = min(len(targets), int(cached["completed"]))
                updated[:completed] = cached["updated"][:completed]
                choices[:completed] = cached["choices"][:completed]
                phases[:completed] = cached["phases"][:completed]
                gains[:completed] = cached["gains"][:completed]
                if completed:
                    _log_progress(f"Resuming cached training apply {cache_path.name}: {completed:,}/{len(targets):,}")
        except Exception as exc:
            _log_progress(f"Ignoring invalid training apply cache {cache_path}: {exc}")
            completed = 0
    cache_every = _cache_every()
    rows_cache: dict[int, np.ndarray] = {}
    for chunk_index, start in enumerate(range(0, len(targets), train_batch), start=1):
        stop = min(start + train_batch, len(targets))
        if stop <= completed:
            continue
        chunk_prefix = prefix[start:stop]
        chunk_targets = targets[start:stop]
        chunk_phase = phase[start:stop]
        chunk_gain = gain[start:stop]
        chunk_size = stop - start
        rows = rows_cache.setdefault(chunk_size, np.arange(chunk_size))
        shifted = circular_shift(
            np.broadcast_to(codes[None], (chunk_size, *codes.shape)).reshape(chunk_size * len(codes), codes.shape[1]),
            chunk_phase.reshape(chunk_size * len(codes)),
        ).reshape(chunk_size, len(codes), codes.shape[1])
        score_state = _apply_decoder_step_np(
            chunk_prefix[:, None, :],
            chunk_gain[:, :, None] * shifted,
            residual_clip_policy,
        )
        score = np.mean((chunk_targets[:, None, :] - np.clip(score_state, 0.0, 1.0)) ** 2, axis=2)
        previous = np.mean((chunk_targets - np.clip(chunk_prefix, 0.0, 1.0)) ** 2, axis=1)
        score[:, 0] = np.minimum(score[:, 0], previous)
        choice = np.argmin(score, axis=1)
        choice[score[rows, choice] >= previous - 1e-12] = 0
        selected_phase = chunk_phase[rows, choice].astype(np.float32)
        selected_gain = chunk_gain[rows, choice].astype(np.float32)
        selected_phase[choice == 0] = 0.0
        selected_gain[choice == 0] = 0.0
        addition = circular_shift(codes[choice], selected_phase) * selected_gain[:, None]
        updated[start:stop] = _apply_decoder_step_np(chunk_prefix, addition, residual_clip_policy)
        choices[start:stop] = choice.astype(np.int16)
        phases[start:stop] = selected_phase
        gains[start:stop] = selected_gain
        if cache_path is not None and cache_every and (chunk_index % cache_every == 0 or stop == len(targets)):
            _write_npz_atomic(
                cache_path,
                completed=np.asarray(stop, dtype=np.int64),
                target_count=np.asarray(len(targets), dtype=np.int64),
                updated=updated,
                choices=choices,
                phases=phases,
                gains=gains,
            )
    if cache_path is not None and cache_path.exists():
        cache_path.unlink()
    return updated, choices, phases, gains


def _apply_training_stage_torch(
    prefix: np.ndarray,
    targets: np.ndarray,
    codes: np.ndarray,
    *,
    device: str,
    cache_path: Path | None = None,
    residual_clip_policy: str = "final_only",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import torch

    train_batch = max(1, int(os.environ.get("LFO_TRAIN_STAGE_BATCH_SIZE", "256")))
    completed = 0
    updated = np.empty_like(prefix, dtype=np.float32)
    choices = np.empty(len(targets), dtype=np.int16)
    phases = np.empty(len(targets), dtype=np.float32)
    gains = np.empty(len(targets), dtype=np.float32)
    if cache_path is not None and cache_path.exists():
        try:
            cached = np.load(cache_path)
            if tuple(cached["updated"].shape) == tuple(prefix.shape) and int(cached["target_count"]) == len(targets):
                completed = min(len(targets), int(cached["completed"]))
                updated[:completed] = cached["updated"][:completed]
                choices[:completed] = cached["choices"][:completed]
                phases[:completed] = cached["phases"][:completed]
                gains[:completed] = cached["gains"][:completed]
                if completed:
                    _log_progress(f"Resuming cached training apply {cache_path.name}: {completed:,}/{len(targets):,}")
        except Exception as exc:
            _log_progress(f"Ignoring invalid training apply cache {cache_path}: {exc}")
            completed = 0
    cache_every = _cache_every()
    codes_t = torch.as_tensor(codes, dtype=torch.float32, device=device)
    codes_roll_bank = _roll_bank_torch(codes_t)
    with torch.no_grad():
        for chunk_index, start in enumerate(range(0, len(targets), train_batch), start=1):
            stop = min(start + train_batch, len(targets))
            if stop <= completed:
                continue
            chunk_prefix = torch.as_tensor(prefix[start:stop], dtype=torch.float32, device=device)
            chunk_targets = torch.as_tensor(targets[start:stop], dtype=torch.float32, device=device)
            chunk_size = stop - start
            residual = chunk_targets - chunk_prefix
            aligned = exact_align_torch_tensors(residual, codes_t, gain_bounds=GAIN_BOUNDS, device=device)
            shifted = _circular_shift_xpu_torch(
                codes_t[None, :, :].expand(chunk_size, -1, -1),
                aligned.phase,
                roll_bank=codes_roll_bank[None, :, :, :].expand(chunk_size, -1, -1, -1),
            )
            score_state = _apply_decoder_step_torch(
                chunk_prefix[:, None, :],
                aligned.gain[:, :, None] * shifted,
                residual_clip_policy,
            )
            score = torch.mean((chunk_targets[:, None, :] - torch.clamp(score_state, 0.0, 1.0)) ** 2, dim=2)
            previous = torch.mean((chunk_targets - torch.clamp(chunk_prefix, 0.0, 1.0)) ** 2, dim=1)
            score[:, 0] = torch.minimum(score[:, 0], previous)
            choice = torch.argmin(score, dim=1)
            rows = torch.arange(chunk_size, device=device)
            choice_score = score[rows, choice]
            choice = torch.where(choice_score >= previous - 1e-12, torch.zeros_like(choice), choice)
            selected_phase = aligned.phase[rows, choice].to(torch.float32)
            selected_gain = aligned.gain[rows, choice].to(torch.float32)
            noop = choice == 0
            selected_phase = torch.where(noop, torch.zeros_like(selected_phase), selected_phase)
            selected_gain = torch.where(noop, torch.zeros_like(selected_gain), selected_gain)
            addition = _circular_shift_xpu_torch(
                codes_t[choice],
                selected_phase,
                roll_bank=codes_roll_bank[choice],
            ) * selected_gain[:, None]
            updated_chunk = _apply_decoder_step_torch(chunk_prefix, addition, residual_clip_policy)
            updated[start:stop] = updated_chunk.detach().cpu().numpy().astype(np.float32)
            choices[start:stop] = choice.detach().cpu().numpy().astype(np.int16)
            phases[start:stop] = selected_phase.detach().cpu().numpy().astype(np.float32)
            gains[start:stop] = selected_gain.detach().cpu().numpy().astype(np.float32)
            if cache_path is not None and cache_every and (chunk_index % cache_every == 0 or stop == len(targets)):
                _write_npz_atomic(
                    cache_path,
                    completed=np.asarray(stop, dtype=np.int64),
                    target_count=np.asarray(len(targets), dtype=np.int64),
                    updated=updated,
                    choices=choices,
                    phases=phases,
                    gains=gains,
                )
            del (
                chunk_prefix,
                chunk_targets,
                residual,
                aligned,
                shifted,
                score_raw,
                score_state,
                score,
                previous,
                choice,
                rows,
                choice_score,
                selected_phase,
                selected_gain,
                noop,
                addition,
                updated_chunk,
            )
            _release_xpu_temporaries(device)
    if cache_path is not None and cache_path.exists():
        cache_path.unlink()
    del codes_t, codes_roll_bank
    _release_xpu_temporaries(device)
    return updated, choices, phases, gains


def _apply_training_stage(
    prefix: np.ndarray,
    targets: np.ndarray,
    codes: np.ndarray,
    *,
    cache_path: Path | None = None,
    residual_clip_policy: str = "final_only",
    policy: Experiment7Policy | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if policy is not None and policy.affine_scope != "legacy":
        return _apply_training_stage_experiment9(
            prefix,
            targets,
            codes,
            policy=policy,
            cache_path=cache_path,
        )
    device = _torch_align_device()
    if device is not None:
        return _apply_training_stage_torch(
            prefix,
            targets,
            codes,
            device=device,
            cache_path=cache_path,
            residual_clip_policy=residual_clip_policy,
        )

    residual = targets - prefix
    aligned = _exact_align(residual, codes)
    return _score_training_stage_cpu(
        prefix,
        targets,
        codes,
        aligned.phase,
        aligned.gain,
        cache_path=cache_path,
        residual_clip_policy=residual_clip_policy,
    )


def _apply_training_stage_experiment9(
    prefix: np.ndarray,
    targets: np.ndarray,
    codes: np.ndarray,
    *,
    policy: Experiment7Policy,
    cache_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_batch = max(1, int(os.environ.get("LFO_TRAIN_STAGE_BATCH_SIZE", "256")))
    completed = 0
    updated = np.empty_like(prefix, dtype=np.float32)
    choices = np.empty(len(targets), dtype=np.int16)
    phases = np.empty(len(targets), dtype=np.float32)
    gains = np.empty(len(targets), dtype=np.float32)
    if cache_path is not None and cache_path.exists():
        try:
            cached = np.load(cache_path)
            if tuple(cached["updated"].shape) == tuple(prefix.shape) and int(cached["target_count"]) == len(targets):
                completed = min(len(targets), int(cached["completed"]))
                updated[:completed] = cached["updated"][:completed]
                choices[:completed] = cached["choices"][:completed]
                phases[:completed] = cached["phases"][:completed]
                gains[:completed] = cached["gains"][:completed]
                if completed:
                    _log_progress(f"Resuming cached training apply {cache_path.name}: {completed:,}/{len(targets):,}")
        except Exception as exc:
            _log_progress(f"Ignoring invalid training apply cache {cache_path}: {exc}")
            completed = 0
    cache_every = _cache_every()
    gain_enabled = _gain_enabled(policy, "residual")
    offset_enabled = _offset_enabled(policy, "residual")
    rows_cache: dict[int, np.ndarray] = {}
    for chunk_index, start in enumerate(range(0, len(targets), train_batch), start=1):
        stop = min(start + train_batch, len(targets))
        if stop <= completed:
            continue
        chunk_prefix = prefix[start:stop]
        chunk_targets = targets[start:stop]
        residual = chunk_targets - chunk_prefix
        target_align, norm_offset, norm_gain, _ = _experiment9_align_target(residual, policy, target="residual")
        aligned, affine_offsets = _align_affine_np(
            target_align,
            codes,
            gain_enabled=gain_enabled,
            offset_enabled=offset_enabled,
        )
        chunk_size = stop - start
        rows = rows_cache.setdefault(chunk_size, np.arange(chunk_size))
        shifted = circular_shift(
            np.broadcast_to(codes[None], (chunk_size, *codes.shape)).reshape(chunk_size * len(codes), codes.shape[1]),
            aligned.phase.reshape(chunk_size * len(codes)),
        ).reshape(chunk_size, len(codes), codes.shape[1])
        normalized_prediction = aligned.gain[:, :, None] * shifted + affine_offsets[:, :, None]
        correction = norm_offset[:, None, None] + norm_gain[:, None, None] * normalized_prediction
        score_state = _apply_decoder_step_np(chunk_prefix[:, None, :], correction, policy.residual_clip_policy)
        score = np.mean((chunk_targets[:, None, :] - np.clip(score_state, 0.0, 1.0)) ** 2, axis=2)
        previous = np.mean((chunk_targets - np.clip(chunk_prefix, 0.0, 1.0)) ** 2, axis=1)
        score[:, 0] = np.minimum(score[:, 0], previous)
        choice = np.argmin(score, axis=1)
        choice[score[rows, choice] >= previous - 1e-12] = 0
        selected_phase = aligned.phase[rows, choice].astype(np.float32)
        selected_gain = aligned.gain[rows, choice].astype(np.float32)
        selected_phase[choice == 0] = 0.0
        selected_gain[choice == 0] = 0.0
        selected_prediction = normalized_prediction[rows, choice]
        selected_prediction[choice == 0] = 0.0
        selected_correction = norm_offset[:, None] + norm_gain[:, None] * selected_prediction
        selected_correction[choice == 0] = 0.0
        updated[start:stop] = _apply_decoder_step_np(chunk_prefix, selected_correction, policy.residual_clip_policy)
        choices[start:stop] = choice.astype(np.int16)
        phases[start:stop] = selected_phase
        gains[start:stop] = selected_gain
        if cache_path is not None and cache_every and (chunk_index % cache_every == 0 or stop == len(targets)):
            _write_npz_atomic(
                cache_path,
                completed=np.asarray(stop, dtype=np.int64),
                target_count=np.asarray(len(targets), dtype=np.int64),
                updated=updated,
                choices=choices,
                phases=phases,
                gains=gains,
            )
    if cache_path is not None and cache_path.exists():
        cache_path.unlink()
    return updated, choices, phases, gains


def train_experiment7_chain(
    dataset: CurveDataset,
    stock: np.ndarray,
    *,
    policy: Experiment7Policy,
    k: int,
    d: int,
    seed: int,
    cache_output_dir: Path | None = None,
    cache_key: str | None = None,
) -> tuple[PhaseChain, pd.DataFrame]:
    if policy.construction_strategy not in CONSTRUCTION_STRATEGIES:
        raise ValueError(f"unsupported construction strategy: {policy.construction_strategy}")
    if policy.residual_clip_policy not in RESIDUAL_CLIP_POLICIES:
        raise ValueError(f"unsupported residual clip policy: {policy.residual_clip_policy}")
    started_training = time.perf_counter()
    _log_progress(
        f"Training chain strategy={policy.construction_strategy} K={k} D={d} "
        f"train={len(dataset.train_indices):,} resolution={dataset.curves.shape[1]}"
    )
    training_steps_total = 2 + d * (2 + len(TOPOLOGY_NAMES) * 2)
    training_steps_done = 0
    train_indices = dataset.train_indices
    train = dataset.curves[train_indices]
    train_topology = dataset.topology[train_indices].astype(np.int32)
    topology_balanced = policy.construction_strategy == "topology_balanced_common_then_tail"
    resume_layer = 1
    resume_branch = "shared"
    partial_cache = (
        _load_partial_training_cache(cache_output_dir, cache_key)
        if cache_output_dir is not None and cache_key is not None
        else None
    )
    if partial_cache is not None:
        partial_chain, partial_construction, prefix, manifest = partial_cache
        bases = partial_chain.bases
        base_sources = partial_chain.base_sources
        stages = list(partial_chain.stages)
        sources = list(partial_chain.stage_sources)
        rotations = list(partial_chain.canonical_rotations)
        labels = list(partial_chain.stage_labels)
        layers = list(partial_chain.stage_layers)
        branches = list(partial_chain.stage_branches)
        construction_rows = [partial_construction]
        phase = str(manifest.get("phase", "base_aligned"))
        if phase == "base_aligned":
            resume_layer, resume_branch = 1, "shared"
            training_steps_done = 2
        elif phase.startswith("layer_"):
            _, layer_text, branch = phase.split("_", 2)
            phase_layer = int(layer_text)
            if branch == "shared":
                resume_layer, resume_branch = phase_layer, "topology"
                training_steps_done = 2 + (phase_layer - 1) * (2 + len(TOPOLOGY_NAMES) * 2) + 2
            else:
                resume_layer, resume_branch = phase_layer + 1, "shared"
                training_steps_done = 2 + phase_layer * (2 + len(TOPOLOGY_NAMES) * 2)
        _log_progress(
            f"Resuming partial training cache phase={phase} "
            f"at layer {resume_layer}/{d} branch={resume_branch}"
        )
    else:
        _log_progress(
            f"Training step {training_steps_done}/{training_steps_total} "
            f"(effort~{_effort(training_steps_done, training_steps_total)}): "
            "building base codebook: stock discard + leftover medoids"
        )
        bases, base_sources, base_audit = _build_bases(dataset, stock, policy=policy, seed=seed)
        training_steps_done += 1
        _log_progress(
            f"Training step {training_steps_done}/{training_steps_total} "
            f"(effort~{_effort(training_steps_done, training_steps_total)}): "
            f"aligning {len(train):,} training curves to {len(bases)} base codes"
        )
        if policy.affine_scope == "legacy":
            base_align = _exact_align(train, bases, fixed_gain=1.0)
            base_choice = np.argmin(base_align.error, axis=1)
            rows = np.arange(len(train))
            prefix = circular_shift(bases[base_choice], base_align.phase[rows, base_choice]).astype(np.float32)
            prefix = _apply_decoder_step_np(np.zeros_like(prefix), prefix, policy.residual_clip_policy, is_base=True)
        else:
            base_target, norm_offset, norm_gain, _ = _experiment9_align_target(train, policy, target="base")
            base_align, base_offsets = _align_affine_np(
                base_target,
                bases,
                gain_enabled=_gain_enabled(policy, "base"),
                offset_enabled=_offset_enabled(policy, "base"),
            )
            base_choice = np.argmin(base_align.error, axis=1)
            rows = np.arange(len(train))
            shifted_base = circular_shift(bases[base_choice], base_align.phase[rows, base_choice])
            normalized_base = base_align.gain[rows, base_choice, None] * shifted_base + base_offsets[rows, base_choice, None]
            prefix = norm_offset[:, None] + norm_gain[:, None] * normalized_base
            prefix = _apply_decoder_step_np(np.zeros_like(prefix), prefix.astype(np.float32), policy.residual_clip_policy, is_base=True)
        training_steps_done += 1
        _log_progress(
            f"Training step {training_steps_done}/{training_steps_total} "
            f"(effort~{_effort(training_steps_done, training_steps_total)}): base alignment complete"
        )
        stages = []
        sources = []
        rotations = []
        labels = []
        layers = []
        branches = []
        construction_rows = [base_audit.assign(stage="base", branch="base", layer=0)]

    def cache_phase(phase: str, complete: bool = False) -> None:
        if cache_output_dir is None or cache_key is None:
            return
        partial = PhaseChain(
            _chain_name(policy, k=k, d=d),
            bases,
            tuple(stages),
            base_sources,
            tuple(sources),
            tuple(labels),
            True,
            tuple(layers),
            tuple(branches),
            tuple(rotations),
        )
        construction = pd.concat(construction_rows, ignore_index=True, sort=False)
        construction["construction_strategy"] = policy.construction_strategy
        construction["modifier_policy"] = policy.modifier_policy
        construction["residual_clip_policy"] = policy.residual_clip_policy
        construction["k"] = k
        construction["d"] = d
        construction["residual_depth"] = d * 2
        _write_training_cache(
            cache_output_dir,
            cache_key,
            chain=partial,
            construction=construction,
            prefix=prefix,
            phase=phase,
            elapsed_seconds=time.perf_counter() - started_training,
            complete=complete,
        )

    if partial_cache is None:
        cache_phase("base_aligned")
    for layer in range(1, d + 1):
        if layer < resume_layer:
            continue
        if not (layer == resume_layer and resume_branch == "topology"):
            _log_progress(
                f"Training step {training_steps_done}/{training_steps_total} "
                f"(effort~{_effort(training_steps_done, training_steps_total)}): "
                f"layer {layer}/{d} fitting shared residual codes"
            )
            shared_residual = train - prefix
            fit_shared_residual = (
                _experiment9_align_target(shared_residual, policy, target="residual")[0]
                if _normalization_enabled(policy, "residual")
                else shared_residual
            )
            shared_codes, shared_sources_local, shared_audit = _fit_residual_codes(
                dataset,
                train_indices,
                fit_shared_residual,
                k,
                strategy=policy.construction_strategy,
                named_layer=layer,
                total_layers=d,
                seed=seed + layer * 101,
                topology_balanced=topology_balanced,
                selection_cutover_layer=policy.selection_cutover_layer,
            )
            training_steps_done += 1
            _log_progress(
                f"Training step {training_steps_done}/{training_steps_total} "
                f"(effort~{_effort(training_steps_done, training_steps_total)}): "
                f"layer {layer}/{d} applying shared residual stage"
            )
            prefix, _, _, _ = _apply_training_stage(
                prefix,
                train,
                shared_codes,
                cache_path=_inflight_cache_path(
                    cache_output_dir,
                    cache_key,
                    f"layer_{layer}_shared_apply_{policy.residual_clip_policy}",
                ),
                residual_clip_policy=policy.residual_clip_policy,
                policy=policy,
            )
            training_steps_done += 1
            shared_stage = np.repeat(shared_codes[None], len(TOPOLOGY_NAMES), axis=0)
            shared_source_stage = np.repeat(shared_sources_local[None], len(TOPOLOGY_NAMES), axis=0)
            stages.append(shared_stage)
            sources.append(shared_source_stage)
            rotations.append(np.zeros_like(shared_source_stage, dtype=np.float32))
            labels.append(f"layer_{layer}_shared")
            layers.append(layer)
            branches.append("shared")
            construction_rows.append(shared_audit.assign(stage=f"layer_{layer}_shared", branch="shared", layer=layer))
            cache_phase(f"layer_{layer}_shared")

        topology_stage = []
        topology_sources = []
        topology_rotation = []
        topology_additions = np.zeros_like(prefix)
        for condition in range(len(TOPOLOGY_NAMES)):
            members = np.flatnonzero(train_topology == condition)
            _log_progress(
                f"Training step {training_steps_done}/{training_steps_total} "
                f"(effort~{_effort(training_steps_done, training_steps_total)}): "
                f"layer {layer}/{d} fitting topology={TOPOLOGY_NAMES[condition]} "
                f"members={len(members):,}"
            )
            if len(members) == 0:
                codes = np.zeros((k, train.shape[1]), dtype=np.float32)
                source = np.full(k, -1, dtype=np.int32)
                audit = pd.DataFrame({"code": np.arange(k), "dataset_index": source, "is_noop": np.arange(k) == 0})
                training_steps_done += 2
            else:
                residual = train[members] - prefix[members]
                fit_residual = (
                    _experiment9_align_target(residual, policy, target="residual")[0]
                    if _normalization_enabled(policy, "residual")
                    else residual
                )
                codes, source, audit = _fit_residual_codes(
                    dataset,
                    train_indices[members],
                    fit_residual,
                    k,
                    strategy=policy.construction_strategy,
                    named_layer=layer,
                    total_layers=d,
                    seed=seed + 5000 + layer * 101 + condition,
                    topology_balanced=False,
                    selection_cutover_layer=policy.selection_cutover_layer,
                )
                training_steps_done += 1
                _log_progress(
                    f"Training step {training_steps_done}/{training_steps_total} "
                    f"(effort~{_effort(training_steps_done, training_steps_total)}): "
                    f"layer {layer}/{d} applying topology={TOPOLOGY_NAMES[condition]} residual stage"
                )
                updated, choice, phase, gain = _apply_training_stage(
                    prefix[members],
                    train[members],
                    codes,
                    cache_path=_inflight_cache_path(
                        cache_output_dir,
                        cache_key,
                        f"layer_{layer}_topology_{TOPOLOGY_NAMES[condition]}_apply_{policy.residual_clip_policy}",
                    ),
                    residual_clip_policy=policy.residual_clip_policy,
                    policy=policy,
                )
                topology_additions[members] = updated - prefix[members]
                training_steps_done += 1
            topology_stage.append(codes)
            topology_sources.append(source)
            topology_rotation.append(np.zeros(k, dtype=np.float32))
            construction_rows.append(
                audit.assign(
                    stage=f"layer_{layer}_topology",
                    branch=f"topology_{TOPOLOGY_NAMES[condition]}",
                    layer=layer,
                )
            )
        prefix = prefix + topology_additions
        stages.append(np.asarray(topology_stage, dtype=np.float32))
        sources.append(np.asarray(topology_sources, dtype=np.int32))
        rotations.append(np.asarray(topology_rotation, dtype=np.float32))
        labels.append(f"layer_{layer}_topology")
        layers.append(layer)
        branches.append("topology")
        cache_phase(f"layer_{layer}_topology")
        _log_progress(
            f"Layer {layer}/{d}: complete; training effort~{_effort(training_steps_done, training_steps_total)}"
        )

    chain = PhaseChain(
        _chain_name(policy, k=k, d=d),
        bases,
        tuple(stages),
        base_sources,
        tuple(sources),
        tuple(labels),
        True,
        tuple(layers),
        tuple(branches),
        tuple(rotations),
    )
    construction = pd.concat(construction_rows, ignore_index=True, sort=False)
    construction["construction_strategy"] = policy.construction_strategy
    construction["modifier_policy"] = policy.modifier_policy
    construction["residual_clip_policy"] = policy.residual_clip_policy
    construction["k"] = k
    construction["d"] = d
    construction["residual_depth"] = d * 2
    _log_progress("Training chain complete")
    if cache_output_dir is not None and cache_key is not None:
        _write_training_cache(
            cache_output_dir,
            cache_key,
            chain=chain,
            construction=construction,
            prefix=prefix,
            phase="complete",
            elapsed_seconds=time.perf_counter() - started_training,
            complete=True,
        )
    return chain, construction


def _modifier_extra_scalars(modifier_policy: str) -> int:
    return {
        "none": 0,
        "global_offset": 1,
        "base_gain": 1,
        "base_gain_global_offset": 2,
    }[modifier_policy]


def _prediction_burden(chain: PhaseChain, modifier_policy: str, encoding: PhaseEncoding | None = None) -> dict[str, object]:
    stage_count = len(chain.stages)
    categorical_outputs = 1 + stage_count
    continuous_outputs = 1 + 2 * stage_count + _modifier_extra_scalars(modifier_policy)
    burden: dict[str, object] = {
        "categorical_outputs": int(categorical_outputs),
        "continuous_outputs": int(continuous_outputs),
        "predicted_outputs": int(categorical_outputs + continuous_outputs),
    }
    if encoding is not None and encoding.stage_indices:
        active_layers = np.sum(np.stack([values != 0 for values in encoding.stage_indices], axis=1), axis=1)
        burden["active_residual_layers_median"] = float(np.median(active_layers))
        burden["active_residual_layers_p95"] = float(np.percentile(active_layers, 95))
        burden["active_residual_outputs_median"] = float(np.median(active_layers * 3))
        burden["active_residual_outputs_p95"] = float(np.percentile(active_layers * 3, 95))
    else:
        burden["active_residual_layers_median"] = 0.0
        burden["active_residual_layers_p95"] = 0.0
        burden["active_residual_outputs_median"] = 0.0
        burden["active_residual_outputs_p95"] = 0.0
    return burden


def _complexity_with_modifiers(
    chain: PhaseChain,
    modifier_policy: str,
    *,
    residual_clip_policy: str = "final_only",
    encoding: PhaseEncoding | None = None,
) -> dict[str, object]:
    complexity = _output_cost(chain)
    extras = _modifier_extra_scalars(modifier_policy)
    complexity["continuous_scalars"] = int(complexity["continuous_scalars"]) + extras
    complexity["dense_outputs"] = int(complexity["dense_outputs"]) + extras
    complexity["modifier_policy"] = modifier_policy
    complexity["residual_clip_policy"] = residual_clip_policy
    complexity["final_clip_only"] = residual_clip_policy == "final_only"
    complexity.update(_prediction_burden(chain, modifier_policy, encoding))
    named_depth = max(chain.stage_layers) if chain.stage_layers else 0
    width = chain.stage_widths[0] if chain.stage_widths else 0
    logical_stored_codes = len(chain.bases) + named_depth * width * (1 + len(TOPOLOGY_NAMES))
    complexity["logical_stored_codes"] = int(logical_stored_codes)
    complexity["logical_stored_floats"] = int(logical_stored_codes * chain.bases.shape[1])
    complexity["logical_stored_bytes_float32"] = int(logical_stored_codes * chain.bases.shape[1] * 4)
    return complexity


def _evaluate_experiment7_encoding(
    dataset: CurveDataset,
    indices: np.ndarray,
    chain: PhaseChain,
    conditions: np.ndarray,
    encoding: PhaseEncoding,
    noop: dict[str, float],
    *,
    modifier_policy: str,
    modifier_label: str | None = None,
    residual_clip_policy: str = "final_only",
    eval_resolution: int,
    beam_width: int,
    elapsed_seconds: float,
    sample_fraction: float = 1.0,
    sample_hash: str = "",
    estimated_peak_memory_mb: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw, base, _ = _decode_raw(chain, encoding, conditions, residual_clip_policy=residual_clip_policy)
    reconstructed, modifier_values, modifier_stats = _apply_modifier_policy(
        dataset.curves[indices], raw, base, modifier_policy
    )
    complexity = {
        **_complexity_with_modifiers(
            chain,
            modifier_policy,
            residual_clip_policy=residual_clip_policy,
            encoding=encoding,
        ),
        **noop,
        **modifier_stats,
    }
    public_modifier = modifier_label or modifier_policy
    config = f"{chain.name}_{public_modifier}_{residual_clip_policy}_bw{beam_width}_eval{eval_resolution}"
    result, subsets = _evaluate_reconstruction(
        dataset,
        indices,
        reconstructed,
        configuration=config,
        family="phase_additive_final_clip",
        candidate=chain.name,
        depth=len(chain.stages),
        eval_resolution=eval_resolution,
        training_feature_grid=128,
        complexity=complexity,
        elapsed_seconds=elapsed_seconds,
    )
    result["named_depth"] = max(chain.stage_layers) if chain.stage_layers else 0
    result["k"] = chain.stage_widths[0] if chain.stage_widths else 0
    result["residual_width"] = result["k"]
    result["residual_depth"] = len(chain.stages)
    result["construction_strategy"] = chain.name.removeprefix("exp7_").split("_k", 1)[0]
    result["modifier_policy"] = modifier_policy
    result["modifier_label"] = public_modifier
    result["residual_clip_policy"] = residual_clip_policy
    result["sample_fraction"] = float(sample_fraction)
    result["sample_hash"] = sample_hash
    result["estimated_peak_memory_mb"] = float(estimated_peak_memory_mb)
    paths = result[["dataset_index", "configuration", "family", "candidate", "eval_resolution"]].copy()
    paths["base_index"] = encoding.base_indices
    paths["base_phase"] = encoding.base_phases
    paths["base_gain"] = modifier_values["base_gain"]
    paths["global_offset"] = modifier_values["global_offset"]
    for stage_i in range(len(chain.stages)):
        paths[f"stage_{stage_i+1}_index"] = encoding.stage_indices[stage_i]
        paths[f"stage_{stage_i+1}_phase"] = encoding.stage_phases[stage_i]
        paths[f"stage_{stage_i+1}_gain"] = encoding.stage_gains[stage_i]
        paths[f"stage_{stage_i+1}_label"] = chain.stage_labels[stage_i]
    usage_rows = []
    for stage_i, values in enumerate(encoding.stage_indices):
        for code, count in enumerate(np.bincount(values, minlength=chain.stage_widths[stage_i])):
            usage_rows.append(
                {
                    "configuration": config,
                    "candidate": chain.name,
                    "stage": chain.stage_labels[stage_i],
                    "layer": chain.stage_layers[stage_i],
                    "branch": chain.stage_branches[stage_i],
                    "code": int(code),
                    "uses": int(count),
                    "is_noop": code == 0,
                }
            )
    return result, subsets, paths, pd.DataFrame(usage_rows)


def evaluate_experiment7_chain(
    dataset: CurveDataset,
    indices: np.ndarray,
    chain: PhaseChain,
    *,
    modifier_policy: str,
    modifier_label: str | None = None,
    residual_clip_policy: str = "final_only",
    eval_resolution: int,
    beam_width: int,
    batch_size: int,
    elapsed_training_seconds: float,
    progress_label: str | None = None,
    progress_offset: float = 0.0,
    progress_scale: float = 1.0,
    encoding_cache_path: Path | None = None,
    sample_fraction: float = 1.0,
    sample_hash: str = "",
    estimated_peak_memory_mb: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    started = time.perf_counter()
    chain = _resample_chain(chain, eval_resolution)
    conditions = _conditions_for(dataset, chain, indices)
    _, encoding, noop = encode_final_clip_beam(
        dataset.curves[indices],
        chain,
        conditions,
        beam_width=beam_width,
        batch_size=batch_size,
        progress_label=progress_label,
        progress_offset=progress_offset,
        progress_scale=progress_scale,
        cache_path=encoding_cache_path,
        residual_clip_policy=residual_clip_policy,
    )
    elapsed = time.perf_counter() - started + elapsed_training_seconds
    return _evaluate_experiment7_encoding(
        dataset,
        indices,
        chain,
        conditions,
        encoding,
        noop,
        modifier_policy=modifier_policy,
        modifier_label=modifier_label,
        residual_clip_policy=residual_clip_policy,
        eval_resolution=eval_resolution,
        beam_width=beam_width,
        elapsed_seconds=elapsed,
        sample_fraction=sample_fraction,
        sample_hash=sample_hash,
        estimated_peak_memory_mb=estimated_peak_memory_mb,
    )


def encode_experiment9_beam(
    targets: np.ndarray,
    chain: PhaseChain,
    conditions: np.ndarray,
    *,
    policy: Experiment7Policy,
    beam_width: int,
    batch_size: int,
    snap_anchors: np.ndarray,
    snap_radii: np.ndarray,
    progress_label: str | None = None,
) -> tuple[np.ndarray, PhaseEncoding, dict[str, float]]:
    n, resolution = targets.shape
    stage_count = len(chain.stages)
    out_base = np.empty(n, dtype=np.int16)
    out_base_phase = np.zeros(n, dtype=np.float32)
    out_base_gain = np.ones(n, dtype=np.float32)
    out_base_offset = np.zeros(n, dtype=np.float32)
    out_indices = [np.empty(n, dtype=np.int16) for _ in chain.stages]
    out_phases = [np.zeros(n, dtype=np.float32) for _ in chain.stages]
    out_gains = [np.zeros(n, dtype=np.float32) for _ in chain.stages]
    out_offsets = [np.zeros(n, dtype=np.float32) for _ in chain.stages]
    total_batches = max(1, math.ceil(n / batch_size))
    eval_log_every = max(1, int(os.environ.get("LFO_EVAL_PROGRESS_EVERY", "10")))
    for batch_index, start in enumerate(range(0, n, batch_size), start=1):
        stop = min(start + batch_size, n)
        batch = targets[start:stop].astype(np.float32)
        b = len(batch)
        should_log = (
            progress_label is not None
            and (batch_index == 1 or batch_index == total_batches or batch_index % eval_log_every == 0)
        )
        if should_log:
            _log_progress(f"{progress_label}: Experiment 9 eval batch {batch_index}/{total_batches}")
        base_target, base_norm_offset, base_norm_gain, _ = _experiment9_align_target(batch, policy, target="base")
        base_align, base_offsets = _align_affine_np(
            base_target,
            chain.bases,
            gain_enabled=_gain_enabled(policy, "base"),
            offset_enabled=_offset_enabled(policy, "base"),
        )
        beam = min(beam_width, base_align.error.shape[1])
        base_choices = _top_beam(base_align.error, beam)
        rows = np.arange(b)[:, None]
        base_phases = base_align.phase[rows, base_choices]
        base_gains = base_align.gain[rows, base_choices]
        selected_base_offsets = base_offsets[rows, base_choices]
        shifted_base = circular_shift(
            chain.bases[base_choices.reshape(-1)],
            base_phases.reshape(-1),
        ).reshape(b, beam, resolution)
        normalized_base = base_gains[:, :, None] * shifted_base + selected_base_offsets[:, :, None]
        raw_prefix = base_norm_offset[:, None, None] + base_norm_gain[:, None, None] * normalized_base
        raw_prefix = _apply_decoder_step_np(
            np.zeros_like(raw_prefix),
            raw_prefix.astype(np.float32),
            policy.residual_clip_policy,
            is_base=True,
        )
        base_paths = base_choices.astype(np.int16)
        base_phase_paths = base_phases.astype(np.float32)
        base_gain_paths = base_gains.astype(np.float32)
        base_offset_paths = selected_base_offsets.astype(np.float32)
        max_paths = max(beam, beam_width)
        index_paths = np.empty((b, max_paths, stage_count), dtype=np.int16)
        phase_paths = np.empty((b, max_paths, stage_count), dtype=np.float32)
        gain_paths = np.empty((b, max_paths, stage_count), dtype=np.float32)
        offset_paths = np.empty((b, max_paths, stage_count), dtype=np.float32)
        for stage_i, stage in enumerate(chain.stages):
            bw = raw_prefix.shape[1]
            k = stage.shape[1]
            residual = (batch[:, None, :] - raw_prefix).reshape(b * bw, resolution)
            target_align, norm_offset, norm_gain, _ = _experiment9_align_target(residual, policy, target="residual")
            dictionaries = stage[conditions[start:stop]]
            repeated = np.repeat(dictionaries, bw, axis=0)
            aligned = exact_align_cpu(
                target_align,
                repeated,
                gain_bounds=GAIN_BOUNDS,
                fixed_gain=None if _gain_enabled(policy, "residual") else 1.0,
            )
            shifted = circular_shift(
                repeated.reshape(b * bw * k, resolution),
                aligned.phase.reshape(b * bw * k),
            ).reshape(b * bw, k, resolution)
            normalized_prediction = aligned.gain[:, :, None] * shifted
            affine_offsets = _offset_for_prediction_np(target_align, normalized_prediction, _offset_enabled(policy, "residual"))
            normalized_prediction = normalized_prediction + affine_offsets[:, :, None]
            corrections = norm_offset[:, None, None] + norm_gain[:, None, None] * normalized_prediction
            candidate_state = _apply_decoder_step_np(
                raw_prefix[:, :, None, :],
                corrections.reshape(b, bw, k, resolution),
                policy.residual_clip_policy,
            )
            candidate_recon, _ = _finalize_experiment9(candidate_state, snap_anchors, snap_radii)
            mse = np.mean((batch[:, None, None, :] - candidate_recon) ** 2, axis=3)
            previous_recon, _ = _finalize_experiment9(raw_prefix, snap_anchors, snap_radii)
            previous = np.mean((batch[:, None, :] - previous_recon) ** 2, axis=2)
            mse[:, :, 0] = np.minimum(mse[:, :, 0], previous)
            candidate_state[:, :, 0, :] = raw_prefix
            phase3 = aligned.phase.reshape(b, bw, k)
            gain3 = aligned.gain.reshape(b, bw, k)
            offset3 = affine_offsets.reshape(b, bw, k)
            phase3[:, :, 0] = 0.0
            gain3[:, :, 0] = 0.0
            offset3[:, :, 0] = 0.0
            flat = mse.reshape(b, -1)
            next_width = min(beam_width, flat.shape[1])
            choice = np.argpartition(flat, next_width - 1, axis=1)[:, :next_width]
            choice = np.take_along_axis(choice, np.argsort(flat[rows, choice], axis=1), axis=1)
            parent = choice // k
            code = choice % k
            raw_prefix = candidate_state[rows, parent, code]
            base_paths = base_paths[rows, parent]
            base_phase_paths = base_phase_paths[rows, parent]
            base_gain_paths = base_gain_paths[rows, parent]
            base_offset_paths = base_offset_paths[rows, parent]
            if stage_i:
                index_paths[:, :next_width, :stage_i] = index_paths[rows, parent, :stage_i]
                phase_paths[:, :next_width, :stage_i] = phase_paths[rows, parent, :stage_i]
                gain_paths[:, :next_width, :stage_i] = gain_paths[rows, parent, :stage_i]
                offset_paths[:, :next_width, :stage_i] = offset_paths[rows, parent, :stage_i]
            index_paths[:, :next_width, stage_i] = code.astype(np.int16)
            phase_paths[:, :next_width, stage_i] = phase3[rows, parent, code].astype(np.float32)
            gain_paths[:, :next_width, stage_i] = gain3[rows, parent, code].astype(np.float32)
            offset_paths[:, :next_width, stage_i] = offset3[rows, parent, code].astype(np.float32)
            raw_prefix = raw_prefix[:, :next_width]
            base_paths = base_paths[:, :next_width]
            base_phase_paths = base_phase_paths[:, :next_width]
            base_gain_paths = base_gain_paths[:, :next_width]
            base_offset_paths = base_offset_paths[:, :next_width]
        final, snap_stats = _finalize_experiment9(raw_prefix[:, 0, :], snap_anchors, snap_radii)
        out_base[start:stop] = base_paths[:, 0]
        out_base_phase[start:stop] = base_phase_paths[:, 0] % 1.0
        out_base_gain[start:stop] = base_gain_paths[:, 0]
        out_base_offset[start:stop] = base_offset_paths[:, 0]
        for stage_i in range(stage_count):
            out_indices[stage_i][start:stop] = index_paths[:, 0, stage_i]
            out_phases[stage_i][start:stop] = phase_paths[:, 0, stage_i] % 1.0
            out_gains[stage_i][start:stop] = gain_paths[:, 0, stage_i]
            out_offsets[stage_i][start:stop] = offset_paths[:, 0, stage_i]
        if start == 0:
            reconstructed = np.empty_like(targets, dtype=np.float32)
            snap_changed = []
            snap_delta = []
        reconstructed[start:stop] = final
        snap_changed.append(snap_stats["snap_changed_value_rate"])
        snap_delta.append(snap_stats["snap_mean_abs_delta"])
    encoding = PhaseEncoding(out_base, out_base_phase, out_indices, out_phases, out_gains)
    encoding.base_gains = out_base_gain
    encoding.base_offsets = out_base_offset
    encoding.stage_offsets = out_offsets
    noop = {f"stage_{i+1}_noop_rate": float(np.mean(values == 0)) for i, values in enumerate(out_indices)}
    noop["snap_changed_value_rate"] = float(np.mean(snap_changed)) if snap_changed else 0.0
    noop["snap_mean_abs_delta"] = float(np.mean(snap_delta)) if snap_delta else 0.0
    return reconstructed.astype(np.float32), encoding, noop


def evaluate_experiment9_chain(
    dataset: CurveDataset,
    indices: np.ndarray,
    chain: PhaseChain,
    *,
    job: Experiment7Job,
    elapsed_training_seconds: float,
    snap_anchors: np.ndarray,
    snap_radii: np.ndarray,
    snap_stats: dict[str, float],
    progress_label: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    started = time.perf_counter()
    chain = _resample_chain(chain, job.eval_resolution)
    conditions = _conditions_for(dataset, chain, indices)
    reconstructed, encoding, stats = encode_experiment9_beam(
        dataset.curves[indices],
        chain,
        conditions,
        policy=job.policy,
        beam_width=job.beam_width,
        batch_size=job.batch_size,
        snap_anchors=snap_anchors,
        snap_radii=snap_radii,
        progress_label=progress_label,
    )
    scalar_outputs = _experiment9_head_scalars(job.policy, len(chain.stages))
    complexity = {
        **_output_cost(chain),
        **stats,
        **snap_stats,
        "continuous_scalars": scalar_outputs,
        "dense_outputs": int(_output_cost(chain)["categorical_logits"]) + int(scalar_outputs),
        "categorical_outputs": 1 + len(chain.stages),
        "continuous_outputs": scalar_outputs,
        "predicted_outputs": (1 + len(chain.stages)) + scalar_outputs,
        "modifier_policy": job.policy.modifier_policy,
        "modifier_label": job.modifier_label or job.policy.modifier_policy,
        "residual_clip_policy": job.policy.residual_clip_policy,
        "final_clip_only": job.policy.residual_clip_policy == "final_only",
        "base_gain_saturated_share": float(np.mean((encoding.base_gains <= 1e-6) | (encoding.base_gains >= 2.0 - 1e-6))),
        "global_offset_abs_median": 0.0,
        "gain_median": float(np.median(np.concatenate([encoding.base_gains, *encoding.stage_gains]))),
        "gain_p95": float(np.percentile(np.abs(np.concatenate([encoding.base_gains, *encoding.stage_gains])), 95)),
        "offset_abs_median": float(np.median(np.abs(np.concatenate([encoding.base_offsets, *encoding.stage_offsets])))),
        "offset_abs_p95": float(np.percentile(np.abs(np.concatenate([encoding.base_offsets, *encoding.stage_offsets])), 95)),
        "gain_under_eps_rate": float(np.mean(np.abs(np.concatenate([encoding.base_gains, *encoding.stage_gains])) < EXPERIMENT9_EPS)),
    }
    public_modifier = job.modifier_label or job.policy.modifier_policy
    config = f"{chain.name}_{public_modifier}_{job.policy.residual_clip_policy}_{job.snap_policy}_bw{job.beam_width}_eval{job.eval_resolution}"
    elapsed = elapsed_training_seconds + (time.perf_counter() - started)
    result, subsets = _evaluate_reconstruction(
        dataset,
        indices,
        reconstructed,
        configuration=config,
        family="experiment9_quick_screen",
        candidate=chain.name,
        depth=len(chain.stages),
        eval_resolution=job.eval_resolution,
        training_feature_grid=128,
        complexity=complexity,
        elapsed_seconds=elapsed,
    )
    result["named_depth"] = max(chain.stage_layers) if chain.stage_layers else 0
    result["k"] = chain.stage_widths[0] if chain.stage_widths else 0
    result["residual_width"] = result["k"]
    result["residual_depth"] = len(chain.stages)
    result["construction_strategy"] = job.policy.construction_strategy
    result["modifier_policy"] = job.policy.modifier_policy
    result["modifier_label"] = public_modifier
    result["residual_clip_policy"] = job.policy.residual_clip_policy
    result["sample_fraction"] = float(job.sample_fraction)
    result["sample_hash"] = ""
    result["estimated_peak_memory_mb"] = float(job.estimated_peak_memory_mb)
    result["experiment9_section"] = job.experiment9_section
    result["target_scope"] = job.target_scope
    result["affine_modulation"] = job.affine_modulation
    result["normalization_label"] = job.normalization_label
    result["decoder_hygiene_policy"] = job.decoder_hygiene_policy
    result["snap_policy"] = job.snap_policy
    result["budget_anchor_width"] = job.budget_anchor_width
    result["budget_anchor_depth"] = job.budget_anchor_depth
    result["budget_anchor_head_outputs"] = job.budget_anchor_head_outputs
    result["budget_actual_head_outputs"] = job.budget_actual_head_outputs
    paths = result[["dataset_index", "configuration", "family", "candidate", "eval_resolution"]].copy()
    paths["base_index"] = encoding.base_indices
    paths["base_phase"] = encoding.base_phases
    paths["base_gain"] = encoding.base_gains
    paths["base_offset"] = encoding.base_offsets
    paths["global_offset"] = 0.0
    for stage_i in range(len(chain.stages)):
        paths[f"stage_{stage_i+1}_index"] = encoding.stage_indices[stage_i]
        paths[f"stage_{stage_i+1}_phase"] = encoding.stage_phases[stage_i]
        paths[f"stage_{stage_i+1}_gain"] = encoding.stage_gains[stage_i]
        paths[f"stage_{stage_i+1}_offset"] = encoding.stage_offsets[stage_i]
        paths[f"stage_{stage_i+1}_label"] = chain.stage_labels[stage_i]
    usage_rows = []
    for stage_i, values in enumerate(encoding.stage_indices):
        for code, count in enumerate(np.bincount(values, minlength=chain.stage_widths[stage_i])):
            usage_rows.append(
                {
                    "configuration": config,
                    "candidate": chain.name,
                    "stage": chain.stage_labels[stage_i],
                    "layer": chain.stage_layers[stage_i],
                    "branch": chain.stage_branches[stage_i],
                    "code": int(code),
                    "uses": int(count),
                    "is_noop": code == 0,
                }
            )
    return result, subsets, paths, pd.DataFrame(usage_rows)


def _prefix_chain(chain: PhaseChain, *, named_depth: int, name: str) -> PhaseChain:
    keep = [index for index, layer in enumerate(chain.stage_layers) if int(layer) <= int(named_depth)]
    return PhaseChain(
        name,
        chain.bases,
        tuple(chain.stages[index] for index in keep),
        chain.base_sources,
        tuple(chain.stage_sources[index] for index in keep),
        tuple(chain.stage_labels[index] for index in keep),
        chain.topology_conditioned,
        tuple(chain.stage_layers[index] for index in keep),
        tuple(chain.stage_branches[index] for index in keep),
        tuple(chain.canonical_rotations[index] for index in keep),
    )


def _chain_name(policy: Experiment7Policy, *, k: int, d: int) -> str:
    suffix = "" if policy.residual_clip_policy == "final_only" else f"_{policy.residual_clip_policy}"
    return f"exp7_{policy.construction_strategy}_k{k}_d{d}{suffix}"


def _train_or_load_experiment7_chain(
    dataset: CurveDataset,
    stock: np.ndarray,
    *,
    catalog_path: Path,
    codebook_path: Path,
    output_dir: Path,
    experiment: str,
    policy: Experiment7Policy,
    k: int,
    d: int,
    seed: int,
    quick: bool,
    eval_resolution: int,
    sample_hash: str = "",
) -> tuple[PhaseChain, pd.DataFrame, float]:
    cache_key = _training_cache_key(
        experiment=experiment,
        policy=policy,
        k=k,
        d=d,
        eval_resolution=eval_resolution,
        seed=seed,
        quick=quick,
        catalog_path=catalog_path,
        codebook_path=codebook_path,
        sample_hash=sample_hash,
    )
    cached = _load_trained_cache(output_dir, cache_key)
    if cached is not None:
        chain, construction, elapsed = cached
        _log_progress(
            f"Reusing cached training chain key={cache_key} "
            f"strategy={policy.construction_strategy} K={k} D={d}"
        )
        return chain, construction, elapsed
    started = time.perf_counter()
    chain, construction = train_experiment7_chain(
        dataset,
        stock,
        policy=policy,
        k=k,
        d=d,
        seed=seed,
        cache_output_dir=output_dir,
        cache_key=cache_key,
    )
    return chain, construction, time.perf_counter() - started


def _run_job(
    catalog_path: Path,
    codebook_path: Path,
    output_dir: Path,
    job: Experiment7Job,
    *,
    max_shapes: int | None,
    quick: bool,
) -> str:
    if _checkpoint_done(output_dir, job.job_id):
        return job.job_id
    _log_progress(f"{job.label}: loading curve dataset")
    dataset = _load_cached_dataset(catalog_path, job.eval_resolution)
    _assert_author_split(dataset)
    dataset, sample_hash = _prepare_dataset_for_job(output_dir, dataset, job)
    if quick:
        dataset = replace(
            dataset,
            train_indices=dataset.train_indices[: min(len(dataset.train_indices), 160)],
            validation_indices=dataset.validation_indices[: min(len(dataset.validation_indices), 96)],
        )
    _, stock = _load_cached_stock(codebook_path, job.eval_resolution)
    stock = stock[:15]
    _log_progress(
        f"{job.label}: dataset ready train={len(dataset.train_indices):,} "
        f"validation={len(dataset.validation_indices):,}"
    )
    chain, construction, train_elapsed = _train_or_load_experiment7_chain(
        dataset,
        stock,
        catalog_path=catalog_path,
        codebook_path=codebook_path,
        output_dir=output_dir,
        experiment=job.experiment,
        policy=replace(job.policy, modifier_policy="none"),
        k=job.k,
        d=job.d,
        seed=job.seed,
        quick=quick,
        eval_resolution=job.eval_resolution,
        sample_hash=sample_hash,
    )
    indices = dataset.validation_indices.copy()
    if max_shapes is not None:
        indices = indices[:max_shapes]
    if quick:
        indices = indices[: min(len(indices), 96)]
    _log_progress(f"{job.label}: evaluating held-out curves={len(indices):,}")
    result, subsets, paths, usage = evaluate_experiment7_chain(
        dataset,
        indices,
        chain,
        modifier_policy=job.policy.modifier_policy,
        modifier_label=job.modifier_label,
        residual_clip_policy=job.policy.residual_clip_policy,
        eval_resolution=job.eval_resolution,
        beam_width=job.beam_width,
        batch_size=job.batch_size,
        elapsed_training_seconds=train_elapsed,
        progress_label=job.label,
        progress_offset=0.55,
        progress_scale=0.45,
        encoding_cache_path=_encoding_cache_path(output_dir, job.job_id),
        sample_fraction=job.sample_fraction,
        sample_hash=sample_hash,
        estimated_peak_memory_mb=job.estimated_peak_memory_mb,
    )
    _log_progress(f"{job.label}: writing checkpoint")
    _write_checkpoint(output_dir, job, chain, result, subsets, paths, usage, construction)
    return job.job_id


def _encoding_from_paths(paths: pd.DataFrame, chain: PhaseChain) -> PhaseEncoding:
    stage_indices = []
    stage_phases = []
    stage_gains = []
    for stage_i in range(len(chain.stages)):
        stage_indices.append(paths[f"stage_{stage_i+1}_index"].to_numpy(dtype=np.int16))
        stage_phases.append(paths[f"stage_{stage_i+1}_phase"].to_numpy(dtype=np.float32))
        stage_gains.append(paths[f"stage_{stage_i+1}_gain"].to_numpy(dtype=np.float32))
    return PhaseEncoding(
        paths["base_index"].to_numpy(dtype=np.int16),
        paths["base_phase"].to_numpy(dtype=np.float32),
        stage_indices,
        stage_phases,
        stage_gains,
    )


def _job_group_key(job: Experiment7Job) -> tuple[object, ...]:
    return (
        job.experiment,
        job.policy.construction_strategy,
        job.policy.stock_rmse_threshold,
        job.policy.stock_pointwise_threshold,
        job.policy.base_medoids_source,
        job.policy.selection_cutover_layer,
        job.policy.residual_clip_policy,
        job.k,
        job.d,
        job.eval_resolution,
        job.beam_width,
        job.batch_size,
        job.seed,
        job.sample_fraction,
        job.sample_seed,
    )


def _job_group_key_7b(job: Experiment7Job) -> tuple[object, ...]:
    return (
        job.experiment,
        job.policy.construction_strategy,
        job.policy.modifier_policy,
        job.policy.stock_rmse_threshold,
        job.policy.stock_pointwise_threshold,
        job.policy.base_medoids_source,
        job.policy.selection_cutover_layer,
        job.policy.residual_clip_policy,
        job.k,
        job.eval_resolution,
        job.beam_width,
        job.batch_size,
        job.seed,
        job.sample_fraction,
        job.sample_seed,
    )


def _job_group_key_8_modifiers(job: Experiment7Job) -> tuple[object, ...]:
    return (
        job.experiment,
        job.policy.construction_strategy,
        job.policy.stock_rmse_threshold,
        job.policy.stock_pointwise_threshold,
        job.policy.base_medoids_source,
        job.policy.selection_cutover_layer,
        job.policy.residual_clip_policy,
        job.k,
        job.d,
        job.eval_resolution,
        job.beam_width,
        job.batch_size,
        job.seed,
        job.sample_fraction,
        job.sample_seed,
    )


def _experiment8_group_for(job: Experiment7Job, jobs: list[Experiment7Job], output_dir: Path) -> tuple[str, list[Experiment7Job]]:
    modifier_group = [
        candidate for candidate in jobs
        if _job_group_key_8_modifiers(candidate) == _job_group_key_8_modifiers(job)
    ]
    pending_modifier_group = [candidate for candidate in modifier_group if not _checkpoint_done(output_dir, candidate.job_id)]
    if len({candidate.policy.modifier_policy for candidate in pending_modifier_group}) > 1:
        return "modifiers", modifier_group
    prefix_group = [
        candidate for candidate in jobs
        if _job_group_key_7b(candidate) == _job_group_key_7b(job)
    ]
    return "prefixes", prefix_group


def _run_7a_job_group(
    catalog_path: Path,
    codebook_path: Path,
    output_dir: Path,
    jobs: list[Experiment7Job],
    *,
    max_shapes: int | None,
    quick: bool,
) -> list[str]:
    pending = [job for job in jobs if not _checkpoint_done(output_dir, job.job_id)]
    if not pending:
        return [job.job_id for job in jobs]
    reference = jobs[0]
    _log_progress(
        f"Grouped 7A execution: strategy={reference.policy.construction_strategy}, "
        f"modifiers={', '.join(job.policy.modifier_policy for job in pending)}"
    )
    _log_progress(f"{reference.label}: loading curve dataset")
    dataset = _load_cached_dataset(catalog_path, reference.eval_resolution)
    _assert_author_split(dataset)
    dataset, sample_hash = _prepare_dataset_for_job(output_dir, dataset, reference)
    if quick:
        dataset = replace(
            dataset,
            train_indices=dataset.train_indices[: min(len(dataset.train_indices), 160)],
            validation_indices=dataset.validation_indices[: min(len(dataset.validation_indices), 96)],
        )
    for grouped in pending:
        grouped.estimated_peak_memory_mb = _check_memory_budget(
            grouped,
            train_count=len(dataset.train_indices),
            validation_count=len(dataset.validation_indices),
        )
    indices = dataset.validation_indices.copy()
    if max_shapes is not None:
        indices = indices[:max_shapes]
    if quick:
        indices = indices[: min(len(indices), 96)]

    completed = [job for job in jobs if _checkpoint_done(output_dir, job.job_id)]
    if completed:
        source_job = next((job for job in completed if job.policy.modifier_policy == "none"), completed[0])
        _log_progress(
            f"Reusing existing encoded paths from {source_job.label} for pending modifiers"
        )
        source_dir = _checkpoint_dir(output_dir, source_job.job_id)
        source_result, _, source_paths, _, source_construction = _load_checkpoint(output_dir, source_job.job_id)
        chain = PhaseChain.load(source_dir / "chain")
        chain = _resample_chain(chain, reference.eval_resolution)
        indices = source_paths["dataset_index"].to_numpy(dtype=np.int32)
        conditions = _conditions_for(dataset, chain, indices)
        encoding = _encoding_from_paths(source_paths, chain)
        noop = {f"stage_{i+1}_noop_rate": float(np.mean(values == 0)) for i, values in enumerate(encoding.stage_indices)}
        elapsed = float(source_result["elapsed_seconds"].iloc[0]) if "elapsed_seconds" in source_result else 0.0
        construction = source_construction
    else:
        _, stock = _load_cached_stock(codebook_path, reference.eval_resolution)
        stock = stock[:15]
        _log_progress(
            f"{reference.label}: dataset ready train={len(dataset.train_indices):,} "
            f"validation={len(dataset.validation_indices):,}; training once for {len(pending)} modifiers"
        )
        chain, construction, train_elapsed = _train_or_load_experiment7_chain(
            dataset,
            stock,
            catalog_path=catalog_path,
            codebook_path=codebook_path,
            output_dir=output_dir,
            experiment=reference.experiment,
            policy=replace(reference.policy, modifier_policy="none"),
            k=reference.k,
            d=reference.d,
            seed=reference.seed,
            quick=quick,
            eval_resolution=reference.eval_resolution,
            sample_hash=sample_hash,
        )
        chain = _resample_chain(chain, reference.eval_resolution)
        conditions = _conditions_for(dataset, chain, indices)
        _log_progress(
            f"{reference.label}: encoding held-out curves once for modifiers={len(pending)} "
            f"curves={len(indices):,}"
        )
        encode_started = time.perf_counter()
        _, encoding, noop = encode_final_clip_beam(
            dataset.curves[indices],
            chain,
            conditions,
            beam_width=reference.beam_width,
            batch_size=reference.batch_size,
            progress_label=f"{reference.label} shared encode",
            progress_offset=0.55,
            progress_scale=0.45,
            cache_path=_encoding_cache_path(output_dir, reference.job_id),
            residual_clip_policy=reference.policy.residual_clip_policy,
        )
        elapsed = train_elapsed + (time.perf_counter() - encode_started)

    completed_ids = []
    for index, job in enumerate(pending, start=1):
        _log_progress(
            f"{job.label}: evaluating modifier {index}/{len(pending)} from shared encoded paths"
        )
        result, subsets, paths, usage = _evaluate_experiment7_encoding(
            dataset,
            indices,
            chain,
            conditions,
            encoding,
            noop,
            modifier_policy=job.policy.modifier_policy,
            modifier_label=job.modifier_label,
            residual_clip_policy=job.policy.residual_clip_policy,
            eval_resolution=job.eval_resolution,
            beam_width=job.beam_width,
            elapsed_seconds=elapsed,
            sample_fraction=job.sample_fraction,
            sample_hash=sample_hash,
            estimated_peak_memory_mb=job.estimated_peak_memory_mb,
        )
        _write_checkpoint(
            output_dir,
            job,
            chain,
            result,
            subsets,
            paths,
            usage,
            construction.assign(
                modifier_policy=job.policy.modifier_policy,
                modifier_label=job.modifier_label or job.policy.modifier_policy,
                residual_clip_policy=job.policy.residual_clip_policy,
                residual_depth=job.residual_depth,
            ),
        )
        completed_ids.append(job.job_id)
        _log_progress(f"{job.label}: checkpoint complete from grouped execution")
    return completed_ids


def _run_7b_job_group(
    catalog_path: Path,
    codebook_path: Path,
    output_dir: Path,
    jobs: list[Experiment7Job],
    *,
    max_shapes: int | None,
    quick: bool,
) -> list[str]:
    pending = sorted([job for job in jobs if not _checkpoint_done(output_dir, job.job_id)], key=lambda item: item.d)
    if not pending:
        return [job.job_id for job in jobs]
    reference = pending[-1]
    max_depth = max(job.d for job in jobs)
    _log_progress(
        f"Grouped 7B execution: strategy={reference.policy.construction_strategy}, "
        f"modifier={reference.policy.modifier_policy}, K={reference.k}, max D={max_depth}, "
        f"pending prefixes={','.join(str(job.d) for job in pending)}"
    )
    dataset = _load_cached_dataset(catalog_path, reference.eval_resolution)
    _assert_author_split(dataset)
    dataset, sample_hash = _prepare_dataset_for_job(output_dir, dataset, reference)
    if quick:
        dataset = replace(
            dataset,
            train_indices=dataset.train_indices[: min(len(dataset.train_indices), 160)],
            validation_indices=dataset.validation_indices[: min(len(dataset.validation_indices), 96)],
        )
    for grouped in pending:
        grouped.estimated_peak_memory_mb = _check_memory_budget(
            grouped,
            train_count=len(dataset.train_indices),
            validation_count=len(dataset.validation_indices),
        )
    indices = dataset.validation_indices.copy()
    if max_shapes is not None:
        indices = indices[:max_shapes]
    if quick:
        indices = indices[: min(len(indices), 96)]
    _, stock = _load_cached_stock(codebook_path, reference.eval_resolution)
    stock = stock[:15]
    _log_progress(
        f"7B K{reference.k}: dataset ready train={len(dataset.train_indices):,} "
        f"validation={len(dataset.validation_indices):,}; training once to D{max_depth}"
    )
    max_chain, max_construction, train_elapsed = _train_or_load_experiment7_chain(
        dataset,
        stock,
        catalog_path=catalog_path,
        codebook_path=codebook_path,
        output_dir=output_dir,
        experiment=reference.experiment,
        policy=replace(reference.policy, modifier_policy="none"),
        k=reference.k,
        d=max_depth,
        seed=reference.seed,
        quick=quick,
        eval_resolution=reference.eval_resolution,
        sample_hash=sample_hash,
    )

    completed_ids = []
    for index, job in enumerate(pending, start=1):
        prefix = _prefix_chain(
            max_chain,
            named_depth=job.d,
            name=_chain_name(job.policy, k=job.k, d=job.d),
        )
        layer_values = pd.to_numeric(
            max_construction["layer"]
            if "layer" in max_construction.columns
            else pd.Series(np.zeros(len(max_construction)), index=max_construction.index),
            errors="coerce",
        ).fillna(0).astype(int)
        construction = max_construction[layer_values <= job.d].copy()
        construction["d"] = job.d
        construction["residual_depth"] = job.residual_depth
        construction["modifier_policy"] = job.policy.modifier_policy
        construction["modifier_label"] = job.modifier_label or job.policy.modifier_policy
        construction["residual_clip_policy"] = job.policy.residual_clip_policy
        _log_progress(
            f"{job.label}: evaluating prefix {index}/{len(pending)} "
            f"curves={len(indices):,}"
        )
        result, subsets, paths, usage = evaluate_experiment7_chain(
            dataset,
            indices,
            prefix,
            modifier_policy=job.policy.modifier_policy,
            modifier_label=job.modifier_label,
            residual_clip_policy=job.policy.residual_clip_policy,
            eval_resolution=job.eval_resolution,
            beam_width=job.beam_width,
            batch_size=job.batch_size,
            elapsed_training_seconds=train_elapsed * (job.d / max_depth),
            progress_label=job.label,
            progress_offset=0.0,
            progress_scale=1.0,
            encoding_cache_path=_encoding_cache_path(output_dir, job.job_id),
            sample_fraction=job.sample_fraction,
            sample_hash=sample_hash,
            estimated_peak_memory_mb=job.estimated_peak_memory_mb,
        )
        _write_checkpoint(output_dir, job, prefix, result, subsets, paths, usage, construction)
        completed_ids.append(job.job_id)
        _log_progress(f"{job.label}: checkpoint complete from grouped 7B execution")
    return completed_ids


def _make_7a_jobs(*, quick: bool, beam_width: int, seed: int, batch_size: int | None = None) -> list[Experiment7Job]:
    strategies = CONSTRUCTION_STRATEGIES[:2] if quick else CONSTRUCTION_STRATEGIES
    modifiers = ("none", "global_offset") if quick else MODIFIER_POLICIES
    resolution = 512 if quick else EXPERIMENT7A_RESOLUTION
    jobs = []
    for strategy in strategies:
        for modifier in modifiers:
            policy = Experiment7Policy(strategy, modifier)
            job_id = f"7A_{strategy}_{modifier}_k12_d8_bw{beam_width}_eval{resolution}"
            jobs.append(
                Experiment7Job(
                    job_id=job_id,
                    experiment="7A",
                    policy=policy,
                    k=12,
                    d=8,
                    eval_resolution=resolution,
                    beam_width=min(beam_width, 8 if quick else beam_width),
                    batch_size=max(1, int(batch_size)) if batch_size is not None else (2 if quick else 6),
                    seed=seed,
                    weight=_structured_weight(
                        _dummy_chain_for_weight(k=12, d=8, resolution=resolution),
                        resolution,
                        beam_width,
                    ),
                )
            )
    return jobs


def _valid_7b_pairs(quick: bool) -> list[tuple[int, int]]:
    if quick:
        return [(8, 4), (12, 4)]
    pairs = []
    for k in (4, 8, 12, 16, 20):
        for d in (4, 6, 8, 10, 12, 14, 16, 20, 24):
            if k == 4 and d < 16:
                continue
            if d >= 16 and k > 12:
                continue
            if k < 8 and d < 16:
                continue
            pairs.append((k, d))
    return pairs


def _dummy_chain_for_weight(k: int, d: int, resolution: int) -> PhaseChain:
    bases = np.zeros((32, resolution), dtype=np.float32)
    stages = tuple(np.zeros((3, k, resolution), dtype=np.float32) for _ in range(d * 2))
    sources = tuple(np.zeros((3, k), dtype=np.int32) for _ in range(d * 2))
    return PhaseChain(
        "weight",
        bases,
        stages,
        np.zeros(32, dtype=np.int32),
        sources,
        tuple(f"stage_{i}" for i in range(d * 2)),
        True,
        tuple((i // 2) + 1 for i in range(d * 2)),
        tuple("shared" if i % 2 == 0 else "topology" for i in range(d * 2)),
        tuple(np.zeros((3, k), dtype=np.float32) for _ in range(d * 2)),
    )


def _load_7b_policy(config_path: Path) -> Experiment7Policy:
    if not config_path.exists():
        raise SystemExit(f"Missing Experiment 7B config: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    missing = [
        key
        for key in ("construction_strategy", "modifier_policy", "stock_discard_thresholds", "base_medoids_source")
        if key not in payload
    ]
    if missing:
        raise SystemExit(
            "Experiment 7B requires an explicit policy selected after reviewing 7A. "
            f"Missing keys in {config_path}: {', '.join(missing)}"
        )
    thresholds = payload["stock_discard_thresholds"]
    strategy = str(payload["construction_strategy"])
    cutover = payload.get("selection_cutover_layer")
    if cutover is None and strategy in {"common_then_tail", "topology_balanced_common_then_tail"}:
        cutover = 4
    return Experiment7Policy(
        construction_strategy=strategy,
        modifier_policy=str(payload["modifier_policy"]),
        stock_rmse_threshold=float(thresholds["rmse"]),
        stock_pointwise_threshold=float(thresholds["pointwise_max_error"]),
        base_medoids_source=str(payload["base_medoids_source"]),
        selection_cutover_layer=None if cutover is None else int(cutover),
    )


def _make_7b_jobs(
    *,
    policy: Experiment7Policy,
    quick: bool,
    beam_width: int,
    seed: int,
    batch_size: int | None = None,
) -> list[Experiment7Job]:
    pairs = _valid_7b_pairs(quick)
    resolution = 512 if quick else FINAL_EVAL_RESOLUTION
    rng = np.random.default_rng(seed)
    order = np.arange(len(pairs))
    rng.shuffle(order)
    jobs = []
    for position in order:
        k, d = pairs[int(position)]
        job_id = f"7B_{policy.construction_strategy}_{policy.modifier_policy}_k{k}_d{d}_bw{beam_width}_eval{resolution}"
        jobs.append(
            Experiment7Job(
                job_id=job_id,
                experiment="7B",
                policy=policy,
                k=k,
                d=d,
                eval_resolution=resolution,
                beam_width=min(beam_width, 8 if quick else beam_width),
                batch_size=max(1, int(batch_size)) if batch_size is not None else (2 if quick else 6),
                seed=seed + k * 100,
                weight=_structured_weight(
                    _dummy_chain_for_weight(k=k, d=d, resolution=resolution),
                    resolution,
                    beam_width,
                ),
            )
        )
    return jobs


def _experiment8_size_pairs() -> list[tuple[int, int]]:
    widths = (8, 16, 24, 32)
    depths = (4, 8, 12, 16, 20, 24, 28, 32)
    return [(width, depth) for width in widths for depth in depths if 128 <= width * depth <= 576]


def _make_experiment8_job(
    *,
    width: int,
    residual_depth: int,
    modifier_label: str,
    residual_clip_policy: str,
    beam_width: int,
    seed: int,
    batch_size: int | None,
) -> Experiment7Job:
    if residual_depth % 2:
        raise ValueError(f"Experiment 8 residual depth must be even: {residual_depth}")
    if modifier_label not in EXPERIMENT8_MODIFIER_LABELS:
        raise ValueError(f"unsupported Experiment 8 modifier label: {modifier_label}")
    if residual_clip_policy not in RESIDUAL_CLIP_POLICIES:
        raise ValueError(f"unsupported residual clip policy: {residual_clip_policy}")
    modifier_policy = EXPERIMENT8_MODIFIER_LABELS[modifier_label]
    internal_depth = residual_depth // 2
    policy = Experiment7Policy(
        EXPERIMENT8_CONSTRUCTION_RECIPE,
        modifier_policy,
        selection_cutover_layer=4,
        residual_clip_policy=residual_clip_policy,
    )
    job_id = (
        f"8_screen_{EXPERIMENT8_CONSTRUCTION_RECIPE}_{modifier_label}_"
        f"W{width}D{residual_depth}_{residual_clip_policy}_"
        f"bw{beam_width}_eval{EXPERIMENT8_RESOLUTION}_sample33_seed{seed}"
    )
    return Experiment7Job(
        job_id=job_id,
        experiment="8",
        policy=policy,
        k=width,
        d=internal_depth,
        eval_resolution=EXPERIMENT8_RESOLUTION,
        beam_width=beam_width,
        batch_size=max(1, int(batch_size)) if batch_size is not None else 6,
        seed=seed + width * 100 + residual_depth,
        weight=_structured_weight(
            _dummy_chain_for_weight(k=width, d=internal_depth, resolution=EXPERIMENT8_RESOLUTION),
            EXPERIMENT8_RESOLUTION,
            beam_width,
        ),
        modifier_label=modifier_label,
        sample_fraction=EXPERIMENT8_SAMPLE_FRACTION,
        sample_seed=seed,
    )


def _make_experiment8_screen_jobs(*, beam_width: int, seed: int, batch_size: int | None = None) -> list[Experiment7Job]:
    jobs: list[Experiment7Job] = []
    seen: set[str] = set()

    def add(job: Experiment7Job) -> None:
        if job.job_id not in seen:
            jobs.append(job)
            seen.add(job.job_id)

    for width, residual_depth in _experiment8_size_pairs():
        add(
            _make_experiment8_job(
                width=width,
                residual_depth=residual_depth,
                modifier_label="phase_only",
                residual_clip_policy="final_only",
                beam_width=beam_width,
                seed=seed,
                batch_size=batch_size,
            )
        )

    for modifier_label in ("phase_only", "phase_gain", "phase_offset", "phase_gain_offset"):
        add(
            _make_experiment8_job(
                width=12,
                residual_depth=16,
                modifier_label=modifier_label,
                residual_clip_policy="final_only",
                beam_width=beam_width,
                seed=seed,
                batch_size=batch_size,
            )
        )

    add(
        _make_experiment8_job(
            width=12,
            residual_depth=16,
            modifier_label="phase_gain",
            residual_clip_policy="intermediate_m11_final_01",
            beam_width=beam_width,
            seed=seed,
            batch_size=batch_size,
        )
    )
    return jobs


def _make_experiment9_job(
    *,
    section: str,
    modifier_label: str,
    residual_width: int = EXPERIMENT9_RESIDUAL_WIDTH,
    residual_depth: int = EXPERIMENT9_RESIDUAL_DEPTH,
    target_scope: str = "",
    affine_modulation: str = "",
    normalization_label: str = "raw",
    decoder_hygiene_policy: str = "bipolar_guard_each_layer",
    snap_policy: str = "none",
    budget_anchor_width: int = 0,
    budget_anchor_depth: int = 0,
    budget_anchor_head_outputs: int = 0,
    beam_width: int,
    seed: int,
    batch_size: int | None,
) -> Experiment7Job:
    if decoder_hygiene_policy not in RESIDUAL_CLIP_POLICIES:
        raise ValueError(f"unsupported Experiment 9 decoder hygiene policy: {decoder_hygiene_policy}")
    if snap_policy not in EXPERIMENT9_SNAP_POLICIES:
        raise ValueError(f"unsupported Experiment 9 snap policy: {snap_policy}")
    if residual_depth % 2:
        raise ValueError(f"Experiment 9 residual depth must be even: {residual_depth}")
    internal_depth = residual_depth // 2
    if section == "9A":
        policy = Experiment7Policy(
            EXPERIMENT9_CONSTRUCTION_RECIPE,
            "none",
            selection_cutover_layer=4,
            residual_clip_policy=decoder_hygiene_policy,
            affine_scope=target_scope,
            affine_modulation=affine_modulation,
            range_normalization=normalization_label == "range_normalized",
        )
    else:
        policy = Experiment7Policy(
            EXPERIMENT9_CONSTRUCTION_RECIPE,
            "none",
            selection_cutover_layer=4,
            residual_clip_policy=decoder_hygiene_policy,
        )
    key_parts = [
        "9_screen",
        section,
        modifier_label,
        target_scope or "phase_only",
        affine_modulation or "phase_only",
        normalization_label,
        decoder_hygiene_policy,
        snap_policy,
        f"W{residual_width}D{residual_depth}",
        f"bw{beam_width}",
        f"eval{EXPERIMENT9_RESOLUTION}",
        f"sample33_seed{seed}",
    ]
    job_id = "_".join(_safe_id(part) for part in key_parts if part)
    job_seed = seed + residual_width * 100 if section == "9D" else seed + len(job_id) * 7
    return Experiment7Job(
        job_id=job_id,
        experiment="9",
        policy=policy,
        k=residual_width,
        d=internal_depth,
        eval_resolution=EXPERIMENT9_RESOLUTION,
        beam_width=beam_width,
        batch_size=max(1, int(batch_size)) if batch_size is not None else 6,
        seed=job_seed,
        weight=_structured_weight(
            _dummy_chain_for_weight(
                k=residual_width,
                d=internal_depth,
                resolution=EXPERIMENT9_RESOLUTION,
            ),
            EXPERIMENT9_RESOLUTION,
            beam_width,
        ),
        modifier_label=modifier_label,
        sample_fraction=EXPERIMENT9_SAMPLE_FRACTION,
        sample_seed=seed,
        experiment9_section=section,
        target_scope=target_scope,
        affine_modulation=affine_modulation,
        normalization_label=normalization_label,
        decoder_hygiene_policy=decoder_hygiene_policy,
        snap_policy=snap_policy,
        budget_anchor_width=budget_anchor_width,
        budget_anchor_depth=budget_anchor_depth,
        budget_anchor_head_outputs=budget_anchor_head_outputs,
        budget_actual_head_outputs=_experiment9_phase_head_outputs(residual_width, residual_depth),
    )


def _make_experiment9_budget_jobs(*, beam_width: int, seed: int, batch_size: int | None) -> list[Experiment7Job]:
    jobs: list[Experiment7Job] = []
    for anchor_depth in EXPERIMENT9_BUDGET_ANCHOR_DEPTHS:
        target_head_outputs = _experiment9_phase_head_outputs(EXPERIMENT9_BUDGET_BASELINE_WIDTH, anchor_depth)
        for width in EXPERIMENT9_BUDGET_WIDTHS:
            residual_depth = _closest_even_depth_for_head_budget(
                width=width,
                target_head_outputs=target_head_outputs,
            )
            jobs.append(
                _make_experiment9_job(
                    section="9D",
                    modifier_label=f"phase_only_W{width}_budget_W8D{anchor_depth}",
                    residual_width=width,
                    residual_depth=residual_depth,
                    normalization_label="raw",
                    decoder_hygiene_policy="final_only",
                    snap_policy="none",
                    budget_anchor_width=EXPERIMENT9_BUDGET_BASELINE_WIDTH,
                    budget_anchor_depth=anchor_depth,
                    budget_anchor_head_outputs=target_head_outputs,
                    beam_width=beam_width,
                    seed=seed,
                    batch_size=batch_size,
                )
            )
    return jobs


def _make_experiment9_screen_jobs(*, beam_width: int, seed: int, batch_size: int | None = None) -> list[Experiment7Job]:
    jobs: list[Experiment7Job] = []
    for target_scope in ("base_only", "residuals_only", "base_and_residuals"):
        for modulation in ("phase_gain", "phase_offset", "phase_gain_offset"):
            for normalization in ("raw", "range_normalized"):
                jobs.append(
                    _make_experiment9_job(
                        section="9A",
                        modifier_label=f"{target_scope}_{modulation}_{normalization}",
                        target_scope=target_scope,
                        affine_modulation=modulation,
                        normalization_label=normalization,
                        decoder_hygiene_policy="bipolar_guard_each_layer",
                        snap_policy="none",
                        beam_width=beam_width,
                        seed=seed,
                        batch_size=batch_size,
                    )
                )
    for policy_name in EXPERIMENT9_CLIP_POLICIES:
        jobs.append(
            _make_experiment9_job(
                section="9B",
                modifier_label=f"phase_only_{policy_name}",
                normalization_label="raw",
                decoder_hygiene_policy=policy_name,
                snap_policy="none",
                beam_width=beam_width,
                seed=seed,
                batch_size=batch_size,
            )
        )
    for snap_policy in EXPERIMENT9_SNAP_POLICIES:
        jobs.append(
            _make_experiment9_job(
                section="9C",
                modifier_label=f"phase_only_{snap_policy}",
                normalization_label="raw",
                decoder_hygiene_policy="bipolar_guard_each_layer",
                snap_policy=snap_policy,
                beam_width=beam_width,
                seed=seed,
                batch_size=batch_size,
            )
        )
    jobs.extend(_make_experiment9_budget_jobs(beam_width=beam_width, seed=seed, batch_size=batch_size))
    return jobs


def _run_experiment9_job(
    catalog_path: Path,
    codebook_path: Path,
    output_dir: Path,
    job: Experiment7Job,
    *,
    max_shapes: int | None,
    quick: bool,
) -> str:
    if _checkpoint_done(output_dir, job.job_id):
        return job.job_id
    _log_progress(f"{job.label}: loading curve dataset")
    dataset = _load_cached_dataset(catalog_path, job.eval_resolution)
    _assert_author_split(dataset)
    dataset, sample_hash = _prepare_dataset_for_job(output_dir, dataset, job)
    if quick:
        dataset = replace(
            dataset,
            train_indices=dataset.train_indices[: min(len(dataset.train_indices), 160)],
            validation_indices=dataset.validation_indices[: min(len(dataset.validation_indices), 96)],
        )
    job.estimated_peak_memory_mb = _check_memory_budget(
        job,
        train_count=len(dataset.train_indices),
        validation_count=len(dataset.validation_indices),
    )
    _, stock = _load_cached_stock(codebook_path, job.eval_resolution)
    stock = stock[:15]
    _log_progress(
        f"{job.label}: dataset ready train={len(dataset.train_indices):,} "
        f"validation={len(dataset.validation_indices):,}"
    )
    chain, construction, train_elapsed = _train_or_load_experiment7_chain(
        dataset,
        stock,
        catalog_path=catalog_path,
        codebook_path=codebook_path,
        output_dir=output_dir,
        experiment=job.experiment,
        policy=job.policy,
        k=job.k,
        d=job.d,
        seed=job.seed,
        quick=quick,
        eval_resolution=job.eval_resolution,
        sample_hash=sample_hash,
    )
    snap_anchors, snap_radii, snap_stats = _infer_snap_schwarzschild(dataset, job.snap_policy)
    validation_indices = dataset.validation_indices.copy()
    train_indices = dataset.train_indices.copy()
    if max_shapes is not None:
        validation_indices = validation_indices[:max_shapes]
        train_indices = train_indices[:max_shapes]
    if quick:
        validation_indices = validation_indices[: min(len(validation_indices), 96)]
        train_indices = train_indices[: min(len(train_indices), 96)]
    _log_progress(f"{job.label}: evaluating validation curves={len(validation_indices):,}")
    result, subsets, paths, usage = evaluate_experiment9_chain(
        dataset,
        validation_indices,
        chain,
        job=job,
        elapsed_training_seconds=train_elapsed,
        snap_anchors=snap_anchors,
        snap_radii=snap_radii,
        snap_stats=snap_stats,
        progress_label=job.label,
    )
    _log_progress(f"{job.label}: evaluating train curves={len(train_indices):,}")
    train_result, _, _, _ = evaluate_experiment9_chain(
        dataset,
        train_indices,
        chain,
        job=job,
        elapsed_training_seconds=0.0,
        snap_anchors=snap_anchors,
        snap_radii=snap_radii,
        snap_stats=snap_stats,
        progress_label=None,
    )
    train_metrics = {
        "train_rmse_median": float(np.median(train_result["rmse"])),
        "train_rmse_p95": float(np.percentile(train_result["rmse"], 95)),
        "train_rmse_p99": float(np.percentile(train_result["rmse"], 99)),
        "validation_rmse_median": float(np.median(result["rmse"])),
        "validation_rmse_p95": float(np.percentile(result["rmse"], 95)),
        "validation_rmse_p99": float(np.percentile(result["rmse"], 99)),
    }
    train_metrics["generalization_gap_p95"] = train_metrics["validation_rmse_p95"] - train_metrics["train_rmse_p95"]
    for key, value in train_metrics.items():
        result[key] = value
    construction = construction.assign(
        modifier_policy=job.policy.modifier_policy,
        modifier_label=job.modifier_label or job.policy.modifier_policy,
        residual_clip_policy=job.policy.residual_clip_policy,
        residual_depth=job.residual_depth,
        experiment9_section=job.experiment9_section,
        target_scope=job.target_scope,
        affine_modulation=job.affine_modulation,
        normalization_label=job.normalization_label,
        decoder_hygiene_policy=job.decoder_hygiene_policy,
        snap_policy=job.snap_policy,
        budget_anchor_width=job.budget_anchor_width,
        budget_anchor_depth=job.budget_anchor_depth,
        budget_anchor_head_outputs=job.budget_anchor_head_outputs,
        budget_actual_head_outputs=job.budget_actual_head_outputs,
    )
    _log_progress(f"{job.label}: writing checkpoint")
    _write_checkpoint(output_dir, job, chain, result, subsets, paths, usage, construction)
    return job.job_id


def _run_experiment9_prefix_group(
    catalog_path: Path,
    codebook_path: Path,
    output_dir: Path,
    jobs: list[Experiment7Job],
    *,
    max_shapes: int | None,
    quick: bool,
) -> list[str]:
    pending = sorted([job for job in jobs if not _checkpoint_done(output_dir, job.job_id)], key=lambda item: item.d)
    if not pending:
        return [job.job_id for job in jobs]
    reference = pending[-1]
    max_depth = max(job.d for job in jobs)
    _log_progress(
        f"Grouped Experiment 9D execution: W={reference.k}, max residual D={max_depth * 2}, "
        f"pending residual depths={','.join(str(job.residual_depth) for job in pending)}"
    )
    dataset = _load_cached_dataset(catalog_path, reference.eval_resolution)
    _assert_author_split(dataset)
    dataset, sample_hash = _prepare_dataset_for_job(output_dir, dataset, reference)
    if quick:
        dataset = replace(
            dataset,
            train_indices=dataset.train_indices[: min(len(dataset.train_indices), 160)],
            validation_indices=dataset.validation_indices[: min(len(dataset.validation_indices), 96)],
        )
    for grouped in pending:
        grouped.estimated_peak_memory_mb = _check_memory_budget(
            grouped,
            train_count=len(dataset.train_indices),
            validation_count=len(dataset.validation_indices),
        )
    validation_indices = dataset.validation_indices.copy()
    train_indices = dataset.train_indices.copy()
    if max_shapes is not None:
        validation_indices = validation_indices[:max_shapes]
        train_indices = train_indices[:max_shapes]
    if quick:
        validation_indices = validation_indices[: min(len(validation_indices), 96)]
        train_indices = train_indices[: min(len(train_indices), 96)]
    _, stock = _load_cached_stock(codebook_path, reference.eval_resolution)
    stock = stock[:15]
    _log_progress(
        f"9D W{reference.k}: dataset ready train={len(dataset.train_indices):,} "
        f"validation={len(dataset.validation_indices):,}; training once to residual D{max_depth * 2}"
    )
    max_chain, max_construction, train_elapsed = _train_or_load_experiment7_chain(
        dataset,
        stock,
        catalog_path=catalog_path,
        codebook_path=codebook_path,
        output_dir=output_dir,
        experiment=reference.experiment,
        policy=reference.policy,
        k=reference.k,
        d=max_depth,
        seed=reference.seed,
        quick=quick,
        eval_resolution=reference.eval_resolution,
        sample_hash=sample_hash,
    )
    snap_anchors, snap_radii, snap_stats = _infer_snap_schwarzschild(dataset, "none")
    completed_ids = []
    for index, job in enumerate(pending, start=1):
        prefix = _prefix_chain(
            max_chain,
            named_depth=job.d,
            name=_chain_name(job.policy, k=job.k, d=job.d),
        )
        layer_values = pd.to_numeric(
            max_construction["layer"]
            if "layer" in max_construction.columns
            else pd.Series(np.zeros(len(max_construction)), index=max_construction.index),
            errors="coerce",
        ).fillna(0).astype(int)
        construction = max_construction[layer_values <= job.d].copy()
        _log_progress(
            f"{job.label}: evaluating 9D prefix {index}/{len(pending)} "
            f"validation={len(validation_indices):,}, train={len(train_indices):,}"
        )
        result, subsets, paths, usage = evaluate_experiment9_chain(
            dataset,
            validation_indices,
            prefix,
            job=job,
            elapsed_training_seconds=train_elapsed * (job.d / max_depth),
            snap_anchors=snap_anchors,
            snap_radii=snap_radii,
            snap_stats=snap_stats,
            progress_label=job.label,
        )
        train_result, _, _, _ = evaluate_experiment9_chain(
            dataset,
            train_indices,
            prefix,
            job=job,
            elapsed_training_seconds=0.0,
            snap_anchors=snap_anchors,
            snap_radii=snap_radii,
            snap_stats=snap_stats,
            progress_label=None,
        )
        train_metrics = {
            "train_rmse_median": float(np.median(train_result["rmse"])),
            "train_rmse_p95": float(np.percentile(train_result["rmse"], 95)),
            "train_rmse_p99": float(np.percentile(train_result["rmse"], 99)),
            "validation_rmse_median": float(np.median(result["rmse"])),
            "validation_rmse_p95": float(np.percentile(result["rmse"], 95)),
            "validation_rmse_p99": float(np.percentile(result["rmse"], 99)),
        }
        train_metrics["generalization_gap_p95"] = train_metrics["validation_rmse_p95"] - train_metrics["train_rmse_p95"]
        for key, value in train_metrics.items():
            result[key] = value
        construction = construction.assign(
            modifier_policy=job.policy.modifier_policy,
            modifier_label=job.modifier_label or job.policy.modifier_policy,
            residual_clip_policy=job.policy.residual_clip_policy,
            residual_depth=job.residual_depth,
            experiment9_section=job.experiment9_section,
            target_scope=job.target_scope,
            affine_modulation=job.affine_modulation,
            normalization_label=job.normalization_label,
            decoder_hygiene_policy=job.decoder_hygiene_policy,
            snap_policy=job.snap_policy,
            budget_anchor_width=job.budget_anchor_width,
            budget_anchor_depth=job.budget_anchor_depth,
            budget_anchor_head_outputs=job.budget_anchor_head_outputs,
            budget_actual_head_outputs=job.budget_actual_head_outputs,
        )
        _write_checkpoint(output_dir, job, prefix, result, subsets, paths, usage, construction)
        completed_ids.append(job.job_id)
        _log_progress(f"{job.label}: checkpoint complete from grouped 9D execution")
    return completed_ids


def _run_jobs(
    catalog_path: Path,
    codebook_path: Path,
    output_dir: Path,
    jobs: list[Experiment7Job],
    *,
    experiment: str,
    max_shapes: int | None,
    quick: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress = _progress_from_jobs(output_dir, jobs, experiment)
    existing = output_dir / "progress.json"
    if existing.exists():
        old = json.loads(existing.read_text(encoding="utf-8"))
        if old.get("run_signature") == progress["run_signature"]:
            progress["started_at"] = old.get("started_at", progress["started_at"])
    _write_progress(output_dir, progress)
    for job in jobs:
        if _checkpoint_done(output_dir, job.job_id):
            print(f"Already completed {job.label}", flush=True)
            continue
        group_kind = ""
        if experiment == "7B":
            grouped_jobs = [candidate for candidate in jobs if _job_group_key_7b(candidate) == _job_group_key_7b(job)]
        elif experiment == "8":
            group_kind, grouped_jobs = _experiment8_group_for(job, jobs, output_dir)
        elif experiment == "9" and job.experiment9_section == "9D":
            grouped_jobs = [
                candidate for candidate in jobs
                if candidate.experiment9_section == "9D"
                and _job_group_key_7b(candidate) == _job_group_key_7b(job)
            ]
        else:
            grouped_jobs = [candidate for candidate in jobs if _job_group_key(candidate) == _job_group_key(job)]
        progress = _progress_from_jobs(output_dir, jobs, experiment, started_at=str(progress.get("started_at")))
        for row in progress["jobs"]:
            if row["job_id"] in {grouped.job_id for grouped in grouped_jobs if not _checkpoint_done(output_dir, grouped.job_id)}:
                row["status"] = "running"
        _write_progress(output_dir, progress)
        if experiment == "7A":
            labels = ", ".join(grouped.policy.modifier_policy for grouped in grouped_jobs if not _checkpoint_done(output_dir, grouped.job_id))
            print(f"{job.label} grouped modifiers: {labels}", flush=True)
            completed_ids = _run_7a_job_group(
                catalog_path,
                codebook_path,
                output_dir,
                grouped_jobs,
                max_shapes=max_shapes,
                quick=quick,
            )
            print(f"Completed grouped 7A checkpoints: {len(completed_ids)}", flush=True)
        elif experiment == "7B":
            labels = ", ".join(f"D{grouped.d}" for grouped in grouped_jobs if not _checkpoint_done(output_dir, grouped.job_id))
            print(f"{job.label} grouped prefixes: {labels}", flush=True)
            completed_ids = _run_7b_job_group(
                catalog_path,
                codebook_path,
                output_dir,
                grouped_jobs,
                max_shapes=max_shapes,
                quick=quick,
            )
            print(f"Completed grouped 7B checkpoints: {len(completed_ids)}", flush=True)
        elif experiment == "8":
            if group_kind == "modifiers":
                labels = ", ".join(
                    grouped.modifier_label or grouped.policy.modifier_policy
                    for grouped in grouped_jobs
                    if not _checkpoint_done(output_dir, grouped.job_id)
                )
                print(f"{job.label} grouped modifiers: {labels}", flush=True)
                completed_ids = _run_7a_job_group(
                    catalog_path,
                    codebook_path,
                    output_dir,
                    grouped_jobs,
                    max_shapes=max_shapes,
                    quick=quick,
                )
            else:
                labels = ", ".join(
                    f"D{grouped.residual_depth}"
                    for grouped in grouped_jobs
                    if not _checkpoint_done(output_dir, grouped.job_id)
                )
                print(f"{job.label} grouped prefixes: {labels}", flush=True)
                completed_ids = _run_7b_job_group(
                    catalog_path,
                    codebook_path,
                    output_dir,
                    grouped_jobs,
                    max_shapes=max_shapes,
                    quick=quick,
                )
            print(f"Completed grouped Experiment 8 checkpoints: {len(completed_ids)}", flush=True)
        elif experiment == "9":
            if job.experiment9_section == "9D":
                labels = ", ".join(
                    f"W{grouped.k}D{grouped.residual_depth}->W{grouped.budget_anchor_width}D{grouped.budget_anchor_depth}"
                    for grouped in grouped_jobs
                    if not _checkpoint_done(output_dir, grouped.job_id)
                )
                print(f"{job.label} grouped 9D prefixes: {labels}", flush=True)
                completed_ids = _run_experiment9_prefix_group(
                    catalog_path,
                    codebook_path,
                    output_dir,
                    grouped_jobs,
                    max_shapes=max_shapes,
                    quick=quick,
                )
                print(f"Completed grouped Experiment 9D checkpoints: {len(completed_ids)}", flush=True)
            else:
                print(job.label, flush=True)
                _run_experiment9_job(catalog_path, codebook_path, output_dir, job, max_shapes=max_shapes, quick=quick)
                print(f"Completed {job.label}", flush=True)
        else:
            print(job.label, flush=True)
            _run_job(catalog_path, codebook_path, output_dir, job, max_shapes=max_shapes, quick=quick)
            print(f"Completed {job.label}", flush=True)
        progress = _progress_from_jobs(output_dir, jobs, experiment, started_at=str(progress.get("started_at")))
        _write_progress(output_dir, progress)


def _collect_completed(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    checkpoint_root = output_dir / "checkpoints"
    result_frames = []
    subset_frames = []
    path_frames = []
    usage_frames = []
    construction_frames = []
    if checkpoint_root.exists():
        for done in sorted(checkpoint_root.glob("*/DONE.txt")):
            result, subsets, paths, usage, construction = _load_checkpoint(output_dir, done.parent.name)
            result_frames.append(result)
            subset_frames.append(subsets)
            path_frames.append(paths)
            usage_frames.append(usage)
            construction_frames.append(construction)
    empty = pd.DataFrame()
    return (
        pd.concat(result_frames, ignore_index=True) if result_frames else empty,
        pd.concat(subset_frames, ignore_index=True) if subset_frames else empty,
        pd.concat(path_frames, ignore_index=True) if path_frames else empty,
        pd.concat(usage_frames, ignore_index=True) if usage_frames else empty,
        pd.concat(construction_frames, ignore_index=True) if construction_frames else empty,
    )


def _write_basic_plots(summary: pd.DataFrame, output_dir: Path, experiment: str) -> None:
    plots = output_dir / "analytics" / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    if summary.empty:
        return
    selected = summary.sort_values("rmse_p95")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(selected["dense_outputs"], selected["rmse_p95"], c=selected["rmse_median"], cmap="viridis")
    for row in selected.head(12).itertuples(index=False):
        ax.annotate(str(row.configuration).replace("_bw", "\n bw"), (row.dense_outputs, row.rmse_p95), fontsize=7)
    ax.set_title(f"Experiment {experiment}: parameter efficiency")
    ax.set_xlabel("dense outputs")
    ax.set_ylabel("P95 RMSE")
    fig.tight_layout()
    fig.savefig(plots / "parameter_efficiency.png", dpi=150)
    plt.close(fig)

    if experiment != "8" and {"k", "named_depth", "rmse_p95"}.issubset(summary.columns):
        pivot = summary.pivot_table(index="named_depth", columns="k", values="rmse_p95", aggfunc="min")
        if not pivot.empty:
            fig, ax = plt.subplots(figsize=(8, 6))
            image = ax.imshow(pivot.to_numpy(), aspect="auto", origin="lower")
            ax.set_xticks(range(len(pivot.columns)), [str(value) for value in pivot.columns])
            ax.set_yticks(range(len(pivot.index)), [str(value) for value in pivot.index])
            ax.set_xlabel("K")
            ax.set_ylabel("D")
            ax.set_title(f"Experiment {experiment}: K/D P95 RMSE")
            fig.colorbar(image, ax=ax, label="P95 RMSE")
            fig.tight_layout()
            fig.savefig(plots / "kd_heatmap.png", dpi=150)
            plt.close(fig)
    if experiment == "7B" and {"k", "named_depth", "rmse_p95", "dense_outputs"}.issubset(summary.columns):
        frame = summary.sort_values(["k", "named_depth"]).copy()
        if not frame.empty:
            fig, ax = plt.subplots(figsize=(9, 5))
            for k, group in frame.groupby("k"):
                group = group.sort_values("named_depth")
                ax.plot(group["named_depth"], group["rmse_p95"], marker="o", label=f"K{k}")
            ax.set_title("Experiment 7B: depth vs P95 RMSE")
            ax.set_xlabel("Additive depth D")
            ax.set_ylabel("P95 RMSE")
            ax.legend()
            fig.tight_layout()
            fig.savefig(plots / "depth_vs_p95.png", dpi=150)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(9, 5))
            for depth, group in frame.groupby("named_depth"):
                group = group.sort_values("k")
                if len(group) >= 2:
                    ax.plot(group["k"], group["rmse_p95"], marker="o", label=f"D{int(depth)}")
            ax.set_title("Experiment 7B: width vs P95 RMSE")
            ax.set_xlabel("Residual width K")
            ax.set_ylabel("P95 RMSE")
            if ax.lines:
                ax.legend(ncol=2, fontsize=8)
            fig.tight_layout()
            fig.savefig(plots / "width_vs_p95.png", dpi=150)
            plt.close(fig)

            for metric in ("rmse_median", "rmse_p95", "rmse_p99", "node_max_error_p95"):
                if metric not in frame.columns:
                    continue
                fig, ax = plt.subplots(figsize=(9, 5))
                ax.scatter(frame["dense_outputs"], frame[metric], c=frame["named_depth"], cmap="plasma")
                for row in frame.itertuples(index=False):
                    ax.annotate(f"K{int(row.k)}D{int(row.named_depth)}", (row.dense_outputs, getattr(row, metric)), fontsize=7)
                ax.set_title(f"Experiment 7B: {metric} vs dense outputs")
                ax.set_xlabel("Dense outputs")
                ax.set_ylabel(metric)
                fig.tight_layout()
                fig.savefig(plots / f"{metric}_vs_dense_outputs.png", dpi=150)
                plt.close(fig)

            coverage_columns = [column for column in frame.columns if column.startswith("all_nodes_under_")]
            if coverage_columns:
                fig, ax = plt.subplots(figsize=(9, 5))
                for column in coverage_columns:
                    ax.scatter(frame["dense_outputs"], frame[column], label=column.removeprefix("all_nodes_under_"), alpha=0.75)
                ax.set_title("Experiment 7B: pointwise/node threshold coverage")
                ax.set_xlabel("Dense outputs")
                ax.set_ylabel("Share of held-out LFOs")
                ax.legend(title="max node error <=", fontsize=8)
                fig.tight_layout()
                fig.savefig(plots / "node_threshold_coverage_vs_dense_outputs.png", dpi=150)
                plt.close(fig)
    if experiment == "8" and {"residual_width", "residual_depth"}.issubset(summary.columns):
        screen = summary.copy()
        screen["residual_width"] = pd.to_numeric(screen["residual_width"], errors="coerce")
        screen["residual_depth"] = pd.to_numeric(screen["residual_depth"], errors="coerce")
        size = screen[
            (screen.get("modifier_label", "") == "phase_only")
            & (screen.get("residual_clip_policy", "") == "final_only")
        ].copy()
        for metric in ("rmse_median", "rmse_p95", "elapsed_seconds", "estimated_peak_memory_mb"):
            if metric not in size.columns or size.empty:
                continue
            pivot = size.pivot_table(index="residual_depth", columns="residual_width", values=metric, aggfunc="min")
            if pivot.empty:
                continue
            fig, ax = plt.subplots(figsize=(8, 6))
            image = ax.imshow(pivot.to_numpy(), aspect="auto", origin="lower")
            ax.set_xticks(range(len(pivot.columns)), [f"W{int(value)}" for value in pivot.columns])
            ax.set_yticks(range(len(pivot.index)), [f"D{int(value)}" for value in pivot.index])
            ax.set_xlabel("Residual width")
            ax.set_ylabel("Residual depth")
            ax.set_title(f"Experiment 8 screen: {metric} over W/D")
            fig.colorbar(image, ax=ax, label=metric)
            fig.tight_layout()
            fig.savefig(plots / f"experiment8_{metric}_wd_heatmap.png", dpi=150)
            plt.close(fig)
        for metric in ("rmse_median", "rmse_p95"):
            if metric not in screen.columns or "predicted_outputs" not in screen.columns:
                continue
            fig, ax = plt.subplots(figsize=(9, 5))
            for label, group in screen.groupby("modifier_label" if "modifier_label" in screen.columns else "modifier_policy"):
                ax.scatter(group["predicted_outputs"], group[metric], label=str(label), alpha=0.75)
            ax.set_title(f"Experiment 8 screen: {metric} vs prediction burden")
            ax.set_xlabel("Predicted outputs per LFO")
            ax.set_ylabel(metric)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(plots / f"experiment8_{metric}_vs_predicted_outputs.png", dpi=150)
            plt.close(fig)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows yet._"
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def run_experiment7_analysis(output_dir: Path, *, experiment: str) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results, subsets, paths, usage, construction = _collect_completed(output_dir)
    analytics = output_dir / "analytics"
    analytics.mkdir(parents=True, exist_ok=True)
    if results.empty:
        report = output_dir / f"EXPERIMENT_{experiment}_FINDINGS.md"
        report.write_text(f"# Experiment {experiment} Findings\n\nNo completed jobs yet.\n", encoding="utf-8")
        return {"analytics_dir": analytics, "report": report}
    summary, thresholds, topology = summarize_results(results, subsets)
    # Preserve Experiment 7-specific columns that Experiment 6's generic
    # summarizer does not group by.
    agg: dict[str, tuple[str, str]] = {
        "k": ("k", "first"),
        "named_depth": ("named_depth", "first"),
        "construction_strategy": ("construction_strategy", "first"),
        "modifier_policy": ("modifier_policy", "first"),
        "logical_stored_codes": (
            "logical_stored_codes" if "logical_stored_codes" in results.columns else "stored_codes",
            "first",
        ),
        "logical_stored_floats": (
            "logical_stored_floats" if "logical_stored_floats" in results.columns else "stored_floats",
            "first",
        ),
        "logical_stored_bytes_float32": (
            "logical_stored_bytes_float32" if "logical_stored_bytes_float32" in results.columns else "stored_bytes_float32",
            "first",
        ),
    }
    for column in (
        "residual_width",
        "residual_depth",
        "modifier_label",
        "residual_clip_policy",
        "sample_fraction",
        "sample_hash",
        "estimated_peak_memory_mb",
        "categorical_outputs",
        "continuous_outputs",
        "predicted_outputs",
        "active_residual_layers_median",
        "active_residual_layers_p95",
        "active_residual_outputs_median",
        "active_residual_outputs_p95",
        "experiment9_section",
        "target_scope",
        "affine_modulation",
        "normalization_label",
        "decoder_hygiene_policy",
        "snap_policy",
        "budget_anchor_width",
        "budget_anchor_depth",
        "budget_anchor_head_outputs",
        "budget_actual_head_outputs",
        "train_rmse_median",
        "train_rmse_p95",
        "train_rmse_p99",
        "validation_rmse_median",
        "validation_rmse_p95",
        "validation_rmse_p99",
        "generalization_gap_p95",
        "gain_median",
        "gain_p95",
        "offset_abs_median",
        "offset_abs_p95",
        "gain_under_eps_rate",
        "snap_anchor_count",
        "snap_radius_median",
        "snap_changed_value_rate",
        "snap_mean_abs_delta",
    ):
        if column in results.columns:
            agg[column] = (column, "first")
    for column in sorted(column for column in results.columns if column.startswith("stage_") and column.endswith("_noop_rate")):
        agg[column] = (column, "first")
    extras = results.groupby("configuration", as_index=False).agg(**agg)
    summary = summary.merge(extras, on="configuration", how="left")
    summary.to_csv(analytics / "summary.csv", index=False)
    thresholds.to_csv(analytics / "thresholds.csv", index=False)
    topology.to_csv(analytics / "topology.csv", index=False)
    paths.to_csv(analytics / "paths.csv", index=False)
    usage.to_csv(analytics / "usage.csv", index=False)
    construction.to_csv(analytics / "construction.csv", index=False)
    _write_basic_plots(summary, output_dir, experiment)
    marker = output_dir / f"COMPLETED_EXPERIMENT_{experiment}.txt"
    status = "complete" if marker.exists() else "partial"
    best = summary.sort_values(["rmse_p95", "rmse_median"]).head(15)
    shortlist_columns = [
        "configuration",
        "construction_strategy",
        "modifier_policy",
        "modifier_label",
        "residual_clip_policy",
        "residual_width",
        "residual_depth",
        "k",
        "named_depth",
        "dense_outputs",
        "predicted_outputs",
        "rmse_median",
        "rmse_p95",
        "rmse_p99",
        "max_error_p95",
        "node_max_error_p95",
        "estimated_peak_memory_mb",
    ]
    shortlist = best[[column for column in shortlist_columns if column in best.columns]]
    if experiment == "7A":
        candidates_dir = output_dir / "candidate_7b_configs"
        candidates_dir.mkdir(parents=True, exist_ok=True)
        for row in shortlist.drop_duplicates(["construction_strategy", "modifier_policy"]).head(8).itertuples(index=False):
            payload = {
                "construction_strategy": row.construction_strategy,
                "modifier_policy": row.modifier_policy,
                "stock_discard_thresholds": {
                    "rmse": STOCK_RMSE_THRESHOLD,
                    "pointwise_max_error": STOCK_POINTWISE_THRESHOLD,
                },
                "base_medoids_source": "leftover_frequency",
                "selection_cutover_layer": 4
                if row.construction_strategy in {"common_then_tail", "topology_balanced_common_then_tail"}
                else None,
                "notes": "Generated from Experiment 7A shortlist; review before running 7B.",
            }
            path = candidates_dir / f"experiment7b_{_safe_id(str(row.construction_strategy))}_{_safe_id(str(row.modifier_policy))}.json"
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report = output_dir / f"EXPERIMENT_{experiment}_FINDINGS.md"
    plot_rel = "analytics/plots/parameter_efficiency.png"
    if experiment == "8":
        heatmap_rel = "analytics/plots/experiment8_rmse_p95_wd_heatmap.png"
        intro = """## Questions

- How much quality do we gain from wider residual layers?
- How much quality do we gain from more residual layers?
- Where is the useful `W x D` size band?
- Which settings are most parameter-efficient for the downstream model to predict?
- Do gain and/or offset help once phase alignment is always enabled?
- Does inter-layer clipping help in the phase+gain setting?
- Which settings should move into the full follow-up experiment?
"""
        heatmap_title = "![W/D P95 heatmap]"
        reminder = (
            "Experiment 8 is a 120-point, beam-4 screen on a fixed 1/3 train/validation sample. "
            "`W` is residual codebook width and `D` is actual residual-layer count."
        )
    else:
        heatmap_rel = "analytics/plots/kd_heatmap.png"
        intro = "## Current read"
        heatmap_title = "![K/D heatmap]"
        reminder = (
            "Experiment 7 uses one final hard clip only. Residual no-op is included inside K. "
            "Phase is serialized as a cycle fraction in `[0, 1)`."
        )
    report.write_text(
        f"""# Experiment {experiment} Findings ({status})

This report is safe to read while the run is still in progress. Completed jobs only are included.

{intro}

The table below is sorted by held-out P95 RMSE. For 7A, treat this as a shortlist for discussion, not an automatic 7B choice.

![Parameter efficiency]({plot_rel})

{heatmap_title}({heatmap_rel})

{_markdown_table(shortlist)}

## Files

- `analytics/summary.csv`
- `analytics/thresholds.csv`
- `analytics/topology.csv`
- `analytics/usage.csv`
- `analytics/construction.csv`
- `analytics/paths.csv`

## Reminder

{reminder}
""",
        encoding="utf-8",
    )
    return {"analytics_dir": analytics, "report": report}


def run_experiment7a(
    catalog_path: Path,
    codebook_path: Path,
    output_dir: Path,
    *,
    quick: bool = False,
    beam_width: int = 32,
    batch_size: int | None = None,
    max_shapes: int | None = None,
    seed: int = SEED,
) -> None:
    jobs = _make_7a_jobs(quick=quick, beam_width=beam_width, seed=seed, batch_size=batch_size)
    _run_jobs(catalog_path, codebook_path, output_dir, jobs, experiment="7A", max_shapes=max_shapes, quick=quick)
    run_experiment7_analysis(output_dir, experiment="7A")


def run_experiment7b(
    catalog_path: Path,
    codebook_path: Path,
    output_dir: Path,
    *,
    config_path: Path | None,
    quick: bool = False,
    beam_width: int = 4,
    batch_size: int | None = None,
    max_shapes: int | None = None,
    seed: int = 7267,
) -> None:
    if config_path is None:
        raise SystemExit(
            "Experiment 7B requires an explicit --config selected after reviewing Experiment 7A. "
            "Run experiment7a_analysis and choose one of artifacts/additive_finalization_7a/"
            "candidate_7b_configs/*.json, or create your own config."
        )
    policy = _load_7b_policy(config_path)
    jobs = _make_7b_jobs(policy=policy, quick=quick, beam_width=beam_width, seed=seed, batch_size=batch_size)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "selected_7a_policy.json").write_text(
        json.dumps({"config_path": str(config_path), "policy": _policy_payload(policy)}, indent=2),
        encoding="utf-8",
    )
    _run_jobs(catalog_path, codebook_path, output_dir, jobs, experiment="7B", max_shapes=max_shapes, quick=quick)
    run_experiment7_analysis(output_dir, experiment="7B")


def run_experiment8_screen(
    catalog_path: Path,
    codebook_path: Path,
    output_dir: Path,
    *,
    beam_width: int = 4,
    batch_size: int | None = None,
    max_shapes: int | None = None,
    seed: int = 7267,
) -> None:
    jobs = _make_experiment8_screen_jobs(
        beam_width=beam_width,
        seed=seed,
        batch_size=batch_size,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "experiment8_screen_plan.json").write_text(
        json.dumps(
            {
                "construction_recipe": EXPERIMENT8_CONSTRUCTION_RECIPE,
                "resolution": EXPERIMENT8_RESOLUTION,
                "beam_width": beam_width,
                "sample_fraction": EXPERIMENT8_SAMPLE_FRACTION,
                "sample_seed": seed,
                "size_pairs": [{"residual_width": width, "residual_depth": depth} for width, depth in _experiment8_size_pairs()],
                "modifier_labels": EXPERIMENT8_MODIFIER_LABELS,
                "clip_policies": RESIDUAL_CLIP_POLICIES,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _run_jobs(catalog_path, codebook_path, output_dir, jobs, experiment="8", max_shapes=max_shapes, quick=False)
    run_experiment7_analysis(output_dir, experiment="8")


def run_experiment9_screen(
    catalog_path: Path,
    codebook_path: Path,
    output_dir: Path,
    *,
    beam_width: int = 4,
    batch_size: int | None = None,
    max_shapes: int | None = None,
    seed: int = 7267,
) -> None:
    jobs = _make_experiment9_screen_jobs(
        beam_width=beam_width,
        seed=seed,
        batch_size=batch_size,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "experiment9_screen_plan.json").write_text(
        json.dumps(
            {
                "construction_recipe": EXPERIMENT9_CONSTRUCTION_RECIPE,
                "resolution": EXPERIMENT9_RESOLUTION,
                "beam_width": beam_width,
                "sample_fraction": EXPERIMENT9_SAMPLE_FRACTION,
                "sample_seed": seed,
                "residual_width": EXPERIMENT9_RESIDUAL_WIDTH,
                "residual_depth": EXPERIMENT9_RESIDUAL_DEPTH,
                "sections": {
                    "9A": {
                        "target_scope": ["base_only", "residuals_only", "base_and_residuals"],
                        "modulation": ["phase_gain", "phase_offset", "phase_gain_offset"],
                        "normalization": ["raw", "range_normalized"],
                    },
                    "9B": {"decoder_hygiene_policy": list(EXPERIMENT9_CLIP_POLICIES)},
                    "9C": {"snap_policy": list(EXPERIMENT9_SNAP_POLICIES)},
                    "9D": {
                        "question": "equivalent phase-only output-head budget for narrow residual widths",
                        "baseline_width": EXPERIMENT9_BUDGET_BASELINE_WIDTH,
                        "anchor_depths": list(EXPERIMENT9_BUDGET_ANCHOR_DEPTHS),
                        "tested_widths": list(EXPERIMENT9_BUDGET_WIDTHS),
                        "decoder_hygiene_policy": "final_only",
                    },
                },
                "job_count": len(jobs),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _run_jobs(catalog_path, codebook_path, output_dir, jobs, experiment="9", max_shapes=max_shapes, quick=False)
    run_experiment7_analysis(output_dir, experiment="9")

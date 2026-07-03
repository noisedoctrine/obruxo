"""Experiment 6: codebook-generation approach selection.

This experiment does not freeze the production LFO codebook.  It evaluates the
candidate families we might use to create that codebook and writes a decision
packet: reconstruction quality, threshold coverage, editor-node preservation,
complexity, Pareto plots, and pseudo-AIC/BIC diagnostics.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
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

from .alignment5 import AlignmentResult, exact_align_cpu, exact_align_xpu, select_best_code
from .phase4 import (
    PhaseChain,
    PhaseEncoding,
    circular_shift,
    compose_additive,
    compose_partitioned,
    compose_switch,
    decode_phase_chain,
)
from .stacked import TOPOLOGY_NAMES, CurveDataset, load_curve_dataset, metric_arrays


RMSE_THRESHOLDS = (1e-6, 0.005, 0.01, 0.02, 0.05, 0.10)
NODE_THRESHOLDS = (0.005, 0.01, 0.02, 0.05, 0.10)
GRID_BASELINES = (32, 48, 64, 96, 128, 192)
EVAL_RESOLUTIONS = (1024, 1920)
SEED = 20260626


@dataclass
class CandidateJob:
    job_id: str
    kind: str
    eval_resolution: int
    beam_width: int
    batch_size: int
    training_feature_grid: int
    weight: float
    width: int | None = None
    chain: PhaseChain | None = None

    @property
    def label(self) -> str:
        if self.kind == "direct_grid":
            return f"Direct grid {self.width} @ eval {self.eval_resolution}"
        assert self.chain is not None
        return f"Structured {self.chain.name} @ eval {self.eval_resolution}"


def _bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def _checkpoint_dir(output_dir: Path, job_id: str) -> Path:
    return output_dir / "checkpoints" / _safe_id(job_id)


def _checkpoint_done(output_dir: Path, job_id: str) -> bool:
    directory = _checkpoint_dir(output_dir, job_id)
    return (
        (directory / "DONE.txt").exists()
        and (directory / "result.csv").exists()
        and (directory / "subsets.csv").exists()
        and (directory / "paths.csv").exists()
    )


def _write_checkpoint(
    output_dir: Path,
    job: CandidateJob,
    result: pd.DataFrame,
    subsets: pd.DataFrame,
    paths: pd.DataFrame,
    usage: pd.DataFrame | None,
) -> None:
    directory = _checkpoint_dir(output_dir, job.job_id)
    directory.mkdir(parents=True, exist_ok=True)
    done = directory / "DONE.txt"
    if done.exists():
        done.unlink()
    result.to_csv(directory / "result.csv", index=False)
    subsets.to_csv(directory / "subsets.csv", index=False)
    paths.to_csv(directory / "paths.csv", index=False)
    if usage is not None and not usage.empty:
        usage.to_csv(directory / "usage.csv", index=False)
    else:
        pd.DataFrame().to_csv(directory / "usage.csv", index=False)
    manifest = {
        "job_id": job.job_id,
        "kind": job.kind,
        "label": job.label,
        "eval_resolution": job.eval_resolution,
        "beam_width": job.beam_width,
        "batch_size": job.batch_size,
        "training_feature_grid": job.training_feature_grid,
        "weight": job.weight,
        "width": job.width,
        "chain": None if job.chain is None else job.chain.name,
        "completed_at": _now_iso(),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    done.write_text(f"{_now_iso()}\n", encoding="utf-8")


def _load_checkpoint(output_dir: Path, job: CandidateJob) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    directory = _checkpoint_dir(output_dir, job.job_id)
    result = pd.read_csv(directory / "result.csv", low_memory=False)
    subsets = pd.read_csv(directory / "subsets.csv", low_memory=False)
    paths = pd.read_csv(directory / "paths.csv", low_memory=False)
    usage_path = directory / "usage.csv"
    try:
        usage = pd.read_csv(usage_path, low_memory=False) if usage_path.exists() else pd.DataFrame()
    except pd.errors.EmptyDataError:
        usage = pd.DataFrame()
    return result, subsets, paths, usage


def _format_duration(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(seconds):
        return "unknown"
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {sec}s"
    if minutes:
        return f"{minutes}m {sec}s"
    return f"{sec}s"


def _read_progress(output_dir: Path) -> dict[str, object] | None:
    path = output_dir / "progress.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_progress(output_dir: Path, progress: dict[str, object]) -> None:
    progress["updated_at"] = _now_iso()
    path = output_dir / "progress.json"
    temporary = output_dir / "progress.tmp.json"
    temporary.write_text(json.dumps(progress, indent=2), encoding="utf-8")
    temporary.replace(path)


def _progress_from_jobs(output_dir: Path, jobs: list[CandidateJob], *, started_at: str | None = None) -> dict[str, object]:
    job_rows = []
    completed = 0
    completed_weight = 0.0
    total_weight = float(sum(job.weight for job in jobs))
    for job in jobs:
        done = _checkpoint_done(output_dir, job.job_id)
        completed += int(done)
        completed_weight += job.weight if done else 0.0
        job_rows.append(
            {
                "job_id": job.job_id,
                "label": job.label,
                "kind": job.kind,
                "eval_resolution": job.eval_resolution,
                "weight": job.weight,
                "status": "completed" if done else "pending",
            }
        )
    return {
        "experiment": 6,
        "run_signature": _jobs_signature(jobs),
        "started_at": started_at or _now_iso(),
        "total_jobs": len(jobs),
        "completed_jobs": completed,
        "total_weight": total_weight,
        "completed_weight": completed_weight,
        "jobs": job_rows,
    }


def _jobs_signature(jobs: list[CandidateJob]) -> str:
    payload = [
        {
            "job_id": job.job_id,
            "kind": job.kind,
            "eval_resolution": job.eval_resolution,
            "beam_width": job.beam_width,
            "training_feature_grid": job.training_feature_grid,
            "width": job.width,
            "chain": None if job.chain is None else job.chain.name,
        }
        for job in jobs
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def experiment6_status(output_dir: Path) -> str:
    progress = _read_progress(output_dir)
    if progress is None:
        return f"No Experiment 6 progress manifest found at {output_dir / 'progress.json'}"
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
    started_at = progress.get("started_at")
    elapsed_seconds = None
    if isinstance(started_at, str):
        try:
            elapsed_seconds = (
                datetime.now(timezone.utc).astimezone()
                - datetime.fromisoformat(started_at)
            ).total_seconds()
        except ValueError:
            elapsed_seconds = None
    count_fraction = completed_jobs / total_jobs if total_jobs else 0.0
    weight_fraction = completed_weight / total_weight if total_weight > 0 else 0.0
    eta_seconds = None
    if elapsed_seconds is not None and completed_weight > 0 and total_weight > completed_weight:
        eta_seconds = elapsed_seconds * (total_weight - completed_weight) / completed_weight
    running = [job for job in jobs if job.get("status") == "running"]
    pending = [job for job in jobs if job.get("status") != "completed"]
    lines = [
        f"Experiment 6 progress: {completed_jobs}/{total_jobs} jobs ({count_fraction:.1%})",
        f"Estimated workload: {completed_weight:.2f}/{total_weight:.2f} units ({weight_fraction:.1%})",
        f"Elapsed: {_format_duration(elapsed_seconds)}",
        f"Estimated time remaining: {_format_duration(eta_seconds)}",
        f"Completion marker: {'yes' if (output_dir / 'COMPLETED_EXCPERIMENT_6.txt').exists() else 'no'}",
    ]
    if running:
        lines.append("Running:")
        lines.extend(f"  - {job.get('label', job.get('job_id'))}" for job in running[:8])
    elif pending:
        lines.append("Next pending:")
        lines.extend(f"  - {job.get('label', job.get('job_id'))}" for job in pending[:8])
    return "\n".join(lines)


def _interp_periodic(curve: np.ndarray, phase: np.ndarray | float) -> np.ndarray:
    values = np.asarray(curve)
    width = values.shape[-1]
    phases = np.asarray(phase, dtype=np.float64) % 1.0
    position = phases * width
    left = np.floor(position).astype(np.int64) % width
    frac = (position - np.floor(position)).astype(np.float64)
    return values[..., left] * (1.0 - frac) + values[..., (left + 1) % width] * frac


def _resample_periodic(values: np.ndarray, resolution: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    was_vector = array.ndim == 1
    if was_vector:
        array = array[None, :]
    if array.shape[-1] == resolution:
        return array[0].copy() if was_vector else array.copy()
    phases = np.linspace(0.0, 1.0, resolution, endpoint=False)
    out = np.stack([_interp_periodic(row, phases) for row in array]).astype(np.float32)
    return out[0] if was_vector else out


def _resample_chain(chain: PhaseChain, resolution: int) -> PhaseChain:
    if chain.bases.shape[-1] == resolution:
        return chain
    stages = tuple(
        _resample_periodic(stage.reshape(-1, stage.shape[-1]), resolution).reshape(
            stage.shape[0], stage.shape[1], resolution
        )
        for stage in chain.stages
    )
    return replace(chain, bases=_resample_periodic(chain.bases, resolution), stages=stages)


def _truncate_chain(chain: PhaseChain, depth: int, name: str | None = None) -> PhaseChain:
    if depth >= len(chain.stages):
        return replace(chain, name=name or chain.name)
    return PhaseChain(
        name or f"{chain.name}_d{depth}",
        chain.bases,
        chain.stages[:depth],
        chain.base_sources,
        chain.stage_sources[:depth],
        chain.stage_labels[:depth],
        chain.topology_conditioned,
        chain.stage_layers[:depth],
        chain.stage_branches[:depth],
        chain.canonical_rotations[:depth],
    )


def _exact_align(targets: np.ndarray, codes: np.ndarray, *, fixed_gain: float | None = None) -> AlignmentResult:
    """Use XPU when available, with CPU as a correctness-safe fallback."""
    device_preference = os.environ.get("LFO_ALIGN_DEVICE", "auto").strip().lower()
    if device_preference in {"cpu", "numpy"}:
        return exact_align_cpu(targets, codes, fixed_gain=fixed_gain)
    try:
        import torch

        if device_preference in {"xpu", "gpu"}:
            if not torch.xpu.is_available():
                raise RuntimeError("LFO_ALIGN_DEVICE requested XPU, but torch.xpu is unavailable")
            return exact_align_xpu(targets, codes, fixed_gain=fixed_gain)
        if device_preference == "auto" and torch.xpu.is_available():
            return exact_align_xpu(targets, codes, fixed_gain=fixed_gain)
    except Exception:
        if device_preference in {"xpu", "gpu"}:
            raise
        pass
    return exact_align_cpu(targets, codes, fixed_gain=fixed_gain)


def _conditions_for(dataset: CurveDataset, chain: PhaseChain, indices: np.ndarray) -> np.ndarray:
    if chain.topology_conditioned:
        return dataset.topology[indices].astype(np.int32)
    return np.zeros(len(indices), np.int32)


def _output_cost(chain: PhaseChain) -> dict[str, float | int | bool | str]:
    categorical = len(chain.bases) + sum(chain.stage_widths)
    continuous = 1 + 2 * len(chain.stages)  # base phase + residual phase/gain per stage
    effective_bits = math.log2(len(chain.bases)) + sum(math.log2(width) for width in chain.stage_widths)
    if chain.topology_conditioned:
        categorical += len(TOPOLOGY_NAMES)
        effective_bits += math.log2(len(TOPOLOGY_NAMES))
    stored_codes = int(len(chain.bases) + sum(stage.shape[0] * stage.shape[1] for stage in chain.stages))
    return {
        "dense_outputs": int(categorical + continuous),
        "categorical_logits": int(categorical),
        "continuous_scalars": int(continuous),
        "effective_index_bits": float(effective_bits),
        "stored_codes": stored_codes,
        "stored_floats": int(chain.stored_floats),
        "stored_bytes_float32": int(chain.stored_floats * 4),
        "decoder_branches": int(3 if chain.topology_conditioned else 1),
        "topology_dependency": bool(chain.topology_conditioned),
        "stage_widths": ",".join(map(str, chain.stage_widths)),
    }


def _grid_cost(width: int) -> dict[str, float | int | bool | str]:
    return {
        "dense_outputs": int(width),
        "categorical_logits": 0,
        "continuous_scalars": int(width),
        "effective_index_bits": 0.0,
        "stored_codes": 0,
        "stored_floats": 0,
        "stored_bytes_float32": 0,
        "decoder_branches": 1,
        "topology_dependency": False,
        "stage_widths": "",
    }


def _shape_subsets(frame: pd.DataFrame, topology_names: np.ndarray) -> pd.DataFrame:
    names = frame["shape_name"].astype(str).str.lower()
    gateish = (
        (topology_names == "discontinuous")
        | names.str.contains("gate", regex=False)
        | names.str.contains("pulse", regex=False)
        | names.str.contains("square", regex=False)
        | names.str.contains("stair", regex=False)
        | names.str.contains("trance", regex=False)
        | names.str.contains("shuffle", regex=False)
    )
    return pd.DataFrame(
        {
            "subset_all": True,
            "subset_custom_ish": ~frame["stock_name_hint"].map(_bool).to_numpy(),
            "subset_gate_pulse_heavy": gateish.to_numpy(),
        }
    )


def _node_metrics(dataset: CurveDataset, indices: np.ndarray, reconstructed: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    target_curves = dataset.curves[indices]
    for local, dataset_index in enumerate(indices):
        record = dataset.frame.iloc[int(dataset_index)]
        points = np.asarray(json.loads(record.points), dtype=np.float64).reshape(-1, 2)
        phases = list(points[:, 0] % 1.0)
        target_values = list(points[:, 1])
        unique, counts = np.unique(points[:, 0], return_counts=True)
        duplicate_x = unique[counts > 1]
        for x_value in duplicate_x:
            probe = (float(x_value) - 0.5 / target_curves.shape[1]) % 1.0
            phases.append(probe)
            target_values.append(float(_interp_periodic(target_curves[local], probe)))
        if not phases:
            errors = np.asarray([0.0], dtype=np.float64)
        else:
            recon_values = _interp_periodic(reconstructed[local], np.asarray(phases, dtype=np.float64))
            errors = np.abs(recon_values - np.asarray(target_values, dtype=np.float64))
        rows.append(
            {
                "dataset_index": int(dataset_index),
                "node_probe_count": int(len(errors)),
                "duplicate_x_probe_count": int(len(duplicate_x)),
                "node_max_error": float(np.max(errors)),
                "node_mean_error": float(np.mean(errors)),
                "node_p95_error": float(np.quantile(errors, 0.95)),
            }
        )
    return pd.DataFrame(rows)


def _evaluate_reconstruction(
    dataset: CurveDataset,
    indices: np.ndarray,
    reconstructed: np.ndarray,
    *,
    configuration: str,
    family: str,
    candidate: str,
    depth: int,
    eval_resolution: int,
    training_feature_grid: int | str,
    complexity: dict[str, object],
    elapsed_seconds: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = dataset.frame.iloc[indices].reset_index(drop=True)
    topology_names = np.asarray([TOPOLOGY_NAMES[value] for value in dataset.topology[indices]])
    metrics = metric_arrays(dataset.curves[indices], reconstructed)
    node = _node_metrics(dataset, indices, reconstructed)
    result = pd.DataFrame(
        {
            "dataset_index": indices,
            "preset_id": frame.preset_id,
            "author_id": frame.author_id,
            "shape_signature": frame.shape_signature,
            "shape_name": frame.shape_name,
            "topology": topology_names,
            "stock_name_hint": frame.stock_name_hint.map(_bool),
            "configuration": configuration,
            "family": family,
            "candidate": candidate,
            "depth": depth,
            "eval_resolution": eval_resolution,
            "training_feature_grid": training_feature_grid,
            "elapsed_seconds_total": elapsed_seconds,
            **metrics,
        }
    )
    for key, value in complexity.items():
        result[key] = value
    result = result.merge(node, on="dataset_index", how="left")
    subsets = _shape_subsets(frame, topology_names)
    subset_frame = pd.concat([result[["dataset_index", "configuration"]], subsets], axis=1)
    return result, subset_frame


def evaluate_direct_grid(
    dataset: CurveDataset,
    indices: np.ndarray,
    width: int,
    *,
    eval_resolution: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    started = time.perf_counter()
    targets = dataset.curves[indices]
    grid_phase = np.arange(width, dtype=np.float64) / width
    values = np.stack([_interp_periodic(row, grid_phase) for row in targets]).astype(np.float32)
    recon_phase = np.linspace(0.0, 1.0, eval_resolution, endpoint=False)
    reconstructed = np.stack([_interp_periodic(row, recon_phase) for row in values]).astype(np.float32)
    elapsed = time.perf_counter() - started
    config = f"grid{width}_eval{eval_resolution}"
    result, subsets = _evaluate_reconstruction(
        dataset,
        indices,
        reconstructed,
        configuration=config,
        family="direct_grid",
        candidate=f"Grid{width}",
        depth=0,
        eval_resolution=eval_resolution,
        training_feature_grid=width,
        complexity=_grid_cost(width),
        elapsed_seconds=elapsed,
    )
    paths = result[
        ["dataset_index", "configuration", "family", "candidate", "eval_resolution"]
    ].copy()
    paths["grid_width"] = width
    return result, subsets, paths


def _top_beam(result: AlignmentResult, beam_width: int) -> tuple[np.ndarray, np.ndarray]:
    width = min(beam_width, result.error.shape[1])
    choices = np.argpartition(result.error, width - 1, axis=1)[:, :width]
    rows = np.arange(result.error.shape[0])[:, None]
    order = np.argsort(result.error[rows, choices], axis=1)
    choices = np.take_along_axis(choices, order, axis=1)
    return choices, result.phase[rows, choices]


def _align_stage_grouped(
    residual: np.ndarray,
    stage: np.ndarray,
    conditions: np.ndarray,
    beam_width: int,
) -> AlignmentResult:
    """Align residual rows without physically repeating dictionaries per beam path."""
    b = len(conditions)
    k = stage.shape[1]
    error = np.empty((b * beam_width, k), dtype=np.float64)
    phase = np.empty((b * beam_width, k), dtype=np.float64)
    gain = np.empty((b * beam_width, k), dtype=np.float64)
    for condition in np.unique(conditions):
        rows = np.flatnonzero(conditions == condition)
        flat = (rows[:, None] * beam_width + np.arange(beam_width)[None]).reshape(-1)
        aligned = _exact_align(residual[flat], stage[int(condition)])
        error[flat] = aligned.error
        phase[flat] = aligned.phase
        gain[flat] = aligned.gain
    return AlignmentResult(error, phase, gain)


def _snapshot_noop_rates(stage_indices: list[np.ndarray]) -> dict[str, float]:
    return {
        f"stage_{stage_i+1}_noop_rate": float(np.mean(values == 0))
        for stage_i, values in enumerate(stage_indices)
    }


def encode_exact_beam(
    targets: np.ndarray,
    chain: PhaseChain,
    conditions: np.ndarray,
    *,
    beam_width: int,
    batch_size: int,
) -> tuple[np.ndarray, PhaseEncoding, dict[str, float]]:
    snapshots = encode_exact_beam_snapshots(
        targets,
        chain,
        conditions,
        beam_width=beam_width,
        batch_size=batch_size,
        snapshot_depths=(len(chain.stages),),
    )
    return snapshots[len(chain.stages)]


def encode_exact_beam_snapshots(
    targets: np.ndarray,
    chain: PhaseChain,
    conditions: np.ndarray,
    *,
    beam_width: int,
    batch_size: int,
    snapshot_depths: Iterable[int],
) -> dict[int, tuple[np.ndarray, PhaseEncoding, dict[str, float]]]:
    """Beam search where every code gets its own exact phase/gain before choice."""
    n, resolution = targets.shape
    requested_depths = sorted({int(depth) for depth in snapshot_depths})
    if not requested_depths or requested_depths[-1] > len(chain.stages) or requested_depths[0] < 1:
        raise ValueError(f"snapshot depths must be in 1..{len(chain.stages)}")
    snapshots: dict[int, dict[str, object]] = {}
    for depth in requested_depths:
        snapshots[depth] = {
            "base_index": np.empty(n, dtype=np.int16),
            "base_phase": np.zeros(n, dtype=np.float32),
            "stage_indices": [np.empty(n, dtype=np.int16) for _ in range(depth)],
            "stage_phases": [np.zeros(n, dtype=np.float32) for _ in range(depth)],
            "stage_gains": [np.zeros(n, dtype=np.float32) for _ in range(depth)],
            "reconstructed": np.empty_like(targets),
        }

    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        batch = targets[start:stop]
        b = len(batch)
        base_align = _exact_align(batch, chain.bases, fixed_gain=1.0)
        base_choices, base_phases = _top_beam(base_align, beam_width)
        beam = base_choices.shape[1]
        prefix = np.stack(
            [circular_shift(chain.bases[base_choices[row]], base_phases[row]) for row in range(b)]
        ).astype(np.float32)
        base_paths = base_choices.astype(np.int16)
        base_phase_paths = base_phases.astype(np.float32)
        index_paths = np.empty((b, beam, 0), dtype=np.int16)
        phase_paths = np.empty((b, beam, 0), dtype=np.float32)
        gain_paths = np.empty((b, beam, 0), dtype=np.float32)

        for stage_i, stage in enumerate(chain.stages):
            dictionaries = stage[conditions[start:stop]]
            bw = prefix.shape[1]
            residual = (batch[:, None, :] - prefix).reshape(b * bw, resolution)
            aligned = _align_stage_grouped(residual, stage, conditions[start:stop], bw)
            k = stage.shape[1]
            repeated = np.repeat(dictionaries, bw, axis=0)
            shifted = np.empty((b * bw, k, resolution), dtype=np.float32)
            for item in range(b * bw):
                shifted[item] = circular_shift(repeated[item], aligned.phase[item])
            additions = (shifted * aligned.gain[:, :, None]).reshape(b, bw, k, resolution)
            candidates = np.clip(prefix[:, :, None, :] + additions, 0.0, 1.0)
            mse = np.mean((batch[:, None, None, :] - candidates) ** 2, axis=3)

            # Preserve previous-depth paths exactly.  Code 0 should normally be
            # the zero/no-op atom, but the explicit parent path keeps the
            # acceptance rule robust even if a candidate family is malformed.
            previous = np.mean((batch[:, None, :] - prefix) ** 2, axis=2)
            mse[:, :, 0] = np.minimum(mse[:, :, 0], previous)
            candidates[:, :, 0, :] = prefix
            aligned.phase = aligned.phase.reshape(b, bw, k)
            aligned.gain = aligned.gain.reshape(b, bw, k)
            aligned.phase[:, :, 0] = 0.0
            aligned.gain[:, :, 0] = 0.0

            flat = mse.reshape(b, -1)
            next_width = min(beam_width, flat.shape[1])
            choice = np.argpartition(flat, next_width - 1, axis=1)[:, :next_width]
            rows = np.arange(b)[:, None]
            choice = np.take_along_axis(choice, np.argsort(flat[rows, choice], axis=1), axis=1)
            parent = choice // k
            code = choice % k
            prefix = candidates[rows, parent, code]
            selected_phase = aligned.phase[rows, parent, code]
            selected_gain = aligned.gain[rows, parent, code]
            base_paths = base_paths[rows, parent]
            base_phase_paths = base_phase_paths[rows, parent]
            index_paths = np.concatenate([index_paths[rows, parent], code[..., None].astype(np.int16)], axis=2)
            phase_paths = np.concatenate([phase_paths[rows, parent], selected_phase[..., None].astype(np.float32)], axis=2)
            gain_paths = np.concatenate([gain_paths[rows, parent], selected_gain[..., None].astype(np.float32)], axis=2)

            depth = stage_i + 1
            if depth in snapshots:
                snapshot = snapshots[depth]
                snapshot["reconstructed"][start:stop] = prefix[:, 0]
                snapshot["base_index"][start:stop] = base_paths[:, 0]
                snapshot["base_phase"][start:stop] = base_phase_paths[:, 0]
                for prefix_stage_i in range(depth):
                    snapshot["stage_indices"][prefix_stage_i][start:stop] = index_paths[:, 0, prefix_stage_i]
                    snapshot["stage_phases"][prefix_stage_i][start:stop] = phase_paths[:, 0, prefix_stage_i]
                    snapshot["stage_gains"][prefix_stage_i][start:stop] = gain_paths[:, 0, prefix_stage_i]

    outputs: dict[int, tuple[np.ndarray, PhaseEncoding, dict[str, float]]] = {}
    for depth, snapshot in snapshots.items():
        stage_indices = snapshot["stage_indices"]
        stage_phases = snapshot["stage_phases"]
        stage_gains = snapshot["stage_gains"]
        outputs[depth] = (
            snapshot["reconstructed"],
            PhaseEncoding(
                snapshot["base_index"],
                snapshot["base_phase"],
                stage_indices,
                stage_phases,
                stage_gains,
            ),
            _snapshot_noop_rates(stage_indices),
        )
    return outputs


def evaluate_phase_chain(
    dataset: CurveDataset,
    indices: np.ndarray,
    chain: PhaseChain,
    *,
    eval_resolution: int,
    beam_width: int,
    batch_size: int,
    training_feature_grid: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    started = time.perf_counter()
    chain = _resample_chain(chain, eval_resolution)
    conditions = _conditions_for(dataset, chain, indices)
    reconstructed, encoding, noop = encode_exact_beam(
        dataset.curves[indices], chain, conditions, beam_width=beam_width, batch_size=batch_size
    )
    elapsed = time.perf_counter() - started
    return format_phase_chain_evaluation(
        dataset,
        indices,
        chain,
        reconstructed,
        encoding,
        noop,
        eval_resolution=eval_resolution,
        beam_width=beam_width,
        training_feature_grid=training_feature_grid,
        elapsed_seconds=elapsed,
    )


def format_phase_chain_evaluation(
    dataset: CurveDataset,
    indices: np.ndarray,
    chain: PhaseChain,
    reconstructed: np.ndarray,
    encoding: PhaseEncoding,
    noop: dict[str, float],
    *,
    eval_resolution: int,
    beam_width: int,
    training_feature_grid: int,
    elapsed_seconds: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    complexity = _output_cost(chain)
    config = f"{chain.name}_bw{beam_width}_eval{eval_resolution}"
    result, subsets = _evaluate_reconstruction(
        dataset,
        indices,
        reconstructed,
        configuration=config,
        family="phase_residual",
        candidate=chain.name,
        depth=len(chain.stages),
        eval_resolution=eval_resolution,
        training_feature_grid=training_feature_grid,
        complexity={**complexity, **noop},
        elapsed_seconds=elapsed_seconds,
    )
    paths = result[
        ["dataset_index", "configuration", "family", "candidate", "eval_resolution"]
    ].copy()
    paths["base_index"] = encoding.base_indices
    paths["base_phase"] = encoding.base_phases
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
                    "code": int(code),
                    "uses": int(count),
                    "is_noop": code == 0,
                }
            )
    return result, subsets, paths, pd.DataFrame(usage_rows)


def _load_candidate_chains(experiment4_dir: Path, *, quick: bool) -> list[PhaseChain]:
    codebook_dir = experiment4_dir / "codebooks"
    shared = PhaseChain.load(codebook_dir / "phase_shared")
    topology = PhaseChain.load(codebook_dir / "phase_topology")
    chains: list[PhaseChain] = []
    for depth in ((2, 4) if quick else (2, 3, 4)):
        chains.append(_truncate_chain(shared, depth, f"phase_shared_d{depth}"))
        chains.append(_truncate_chain(topology, depth, f"phase_topology_d{depth}"))
        chains.append(_truncate_chain(compose_additive(shared, topology, 4), depth * 2, f"phase_additive_k4_d{depth}"))
        chains.append(_truncate_chain(compose_additive(shared, topology, 8), depth * 2, f"phase_additive_k8_d{depth}"))
        if not quick:
            chains.append(_truncate_chain(compose_additive(shared, topology, 12), depth * 2, f"phase_additive_k12_d{depth}"))
            chains.append(_truncate_chain(compose_additive(shared, topology, 16), depth * 2, f"phase_additive_k16_d{depth}"))
            for switch in (1, 2, 3):
                chains.append(_truncate_chain(compose_switch(shared, topology, switch), depth, f"phase_switch_{switch}_d{depth}"))
            chains.append(_truncate_chain(compose_partitioned(shared, topology, [8, 8, 8, 8], name="phase_partition_s8"), depth, f"phase_partition_s8_d{depth}"))
            chains.append(_truncate_chain(compose_partitioned(shared, topology, [12, 9, 6, 3], name="phase_partition_taper_12_9_6_3"), depth, f"phase_partition_taper_12_9_6_3_d{depth}"))

    # Keep names unique while preserving deterministic order.
    seen: set[str] = set()
    unique = []
    for chain in chains:
        if chain.name not in seen:
            seen.add(chain.name)
            unique.append(chain)
    return unique


def _structured_weight(chain: PhaseChain, resolution: int, beam_width: int) -> float:
    resolution_factor = resolution / 1024.0
    beam_factor = max(1.0, beam_width / 64.0)
    # Weight is only for progress/ETA.  It roughly tracks the number of exact
    # alignment stages and dictionary width, not a formal operation count.
    stage_factor = sum(max(1, width) for width in chain.stage_widths) / 16.0
    return float(resolution_factor * beam_factor * max(1, len(chain.stages)) * max(1.0, stage_factor))


def _direct_weight(width: int, resolution: int) -> float:
    return float(0.08 * (width / 64.0) * (resolution / 1024.0))


def _make_jobs(
    experiment4_dir: Path,
    *,
    quick: bool,
    beam_width: int,
) -> tuple[list[CandidateJob], list[int], list[int]]:
    grids = (32, 48, 64, 96) if quick else GRID_BASELINES
    eval_resolutions = (1024,) if quick else EVAL_RESOLUTIONS
    chains = _load_candidate_chains(experiment4_dir, quick=quick)
    if quick:
        chains = chains[:4]
    jobs: list[CandidateJob] = []
    for resolution in eval_resolutions:
        for width in grids:
            jobs.append(
                CandidateJob(
                    job_id=f"grid{width}_eval{resolution}",
                    kind="direct_grid",
                    eval_resolution=resolution,
                    beam_width=beam_width,
                    batch_size=0,
                    training_feature_grid=width,
                    width=width,
                    weight=_direct_weight(width, resolution),
                )
            )
        chain_batch = 8
        if quick:
            chain_batch = 4
        effective_beam = min(beam_width, 16 if quick else beam_width)
        for chain in chains:
            jobs.append(
                CandidateJob(
                    job_id=f"{chain.name}_bw{effective_beam}_eval{resolution}",
                    kind="phase_residual",
                    eval_resolution=resolution,
                    beam_width=effective_beam,
                    batch_size=chain_batch,
                    training_feature_grid=128,
                    chain=chain,
                    weight=_structured_weight(chain, resolution, effective_beam),
                )
            )
    return jobs, list(grids), list(eval_resolutions)


def _run_candidate_job(
    catalog_path: Path,
    output_dir: Path,
    job: CandidateJob,
    *,
    max_shapes: int | None,
    quick: bool,
) -> str:
    if _checkpoint_done(output_dir, job.job_id):
        return job.job_id
    dataset = load_curve_dataset(catalog_path, resolution=job.eval_resolution)
    _assert_author_split(dataset)
    return _run_candidate_job_from_dataset(dataset, output_dir, job, max_shapes=max_shapes, quick=quick)


def _run_candidate_job_from_dataset(
    dataset: CurveDataset,
    output_dir: Path,
    job: CandidateJob,
    *,
    max_shapes: int | None,
    quick: bool,
) -> str:
    if _checkpoint_done(output_dir, job.job_id):
        return job.job_id
    indices = dataset.validation_indices.copy()
    if max_shapes is not None:
        indices = indices[:max_shapes]
    if quick:
        indices = indices[: min(len(indices), 96)]
    if job.kind == "direct_grid":
        assert job.width is not None
        result, subsets, paths = evaluate_direct_grid(
            dataset, indices, job.width, eval_resolution=job.eval_resolution
        )
        usage = pd.DataFrame()
    elif job.kind == "phase_residual":
        assert job.chain is not None
        result, subsets, paths, usage = evaluate_phase_chain(
            dataset,
            indices,
            job.chain,
            eval_resolution=job.eval_resolution,
            beam_width=job.beam_width,
            batch_size=job.batch_size,
            training_feature_grid=job.training_feature_grid,
        )
    else:
        raise ValueError(f"unsupported Experiment 6 job kind: {job.kind}")
    _write_checkpoint(output_dir, job, result, subsets, paths, usage)
    return job.job_id


def _phase_group_key(job: CandidateJob) -> tuple[int, int, str]:
    assert job.chain is not None
    prefix, _, suffix = job.chain.name.rpartition("_d")
    if not suffix.isdigit() or not prefix:
        prefix = job.chain.name
    return job.eval_resolution, job.beam_width, prefix


def _run_phase_job_group_from_dataset(
    dataset: CurveDataset,
    output_dir: Path,
    jobs: list[CandidateJob],
    *,
    max_shapes: int | None,
    quick: bool,
) -> list[str]:
    pending = [job for job in jobs if not _checkpoint_done(output_dir, job.job_id)]
    if not pending:
        return [job.job_id for job in jobs]
    jobs = sorted(jobs, key=lambda job: len(job.chain.stages))  # type: ignore[union-attr]
    max_job = jobs[-1]
    assert max_job.chain is not None
    indices = dataset.validation_indices.copy()
    if max_shapes is not None:
        indices = indices[:max_shapes]
    if quick:
        indices = indices[: min(len(indices), 96)]
    started = time.perf_counter()
    max_chain = _resample_chain(max_job.chain, max_job.eval_resolution)
    conditions = _conditions_for(dataset, max_chain, indices)
    depths = sorted({len(job.chain.stages) for job in jobs if job.chain is not None})
    snapshots = encode_exact_beam_snapshots(
        dataset.curves[indices],
        max_chain,
        conditions,
        beam_width=max_job.beam_width,
        batch_size=max_job.batch_size,
        snapshot_depths=depths,
    )
    elapsed = time.perf_counter() - started
    written: list[str] = []
    for job in jobs:
        if _checkpoint_done(output_dir, job.job_id):
            written.append(job.job_id)
            continue
        assert job.chain is not None
        chain = _resample_chain(job.chain, job.eval_resolution)
        reconstructed, encoding, noop = snapshots[len(chain.stages)]
        result, subsets, paths, usage = format_phase_chain_evaluation(
            dataset,
            indices,
            chain,
            reconstructed,
            encoding,
            noop,
            eval_resolution=job.eval_resolution,
            beam_width=job.beam_width,
            training_feature_grid=job.training_feature_grid,
            elapsed_seconds=elapsed,
        )
        _write_checkpoint(output_dir, job, result, subsets, paths, usage)
        written.append(job.job_id)
    return written


def summarize_results(results: pd.DataFrame, subsets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results = results.copy()
    if "stage_widths" in results:
        results["stage_widths"] = results["stage_widths"].fillna("")
    keys = [
        "configuration",
        "family",
        "candidate",
        "depth",
        "eval_resolution",
        "training_feature_grid",
        "dense_outputs",
        "categorical_logits",
        "continuous_scalars",
        "effective_index_bits",
        "stored_codes",
        "stored_floats",
        "stored_bytes_float32",
        "decoder_branches",
        "topology_dependency",
        "stage_widths",
    ]
    summary = (
        results.groupby(keys, as_index=False)
        .agg(
            shapes=("dataset_index", "size"),
            rmse_mean=("rmse", "mean"),
            rmse_median=("rmse", "median"),
            rmse_p90=("rmse", lambda x: x.quantile(0.90)),
            rmse_p95=("rmse", lambda x: x.quantile(0.95)),
            rmse_p99=("rmse", lambda x: x.quantile(0.99)),
            max_error_median=("max_abs_error", "median"),
            max_error_p95=("max_abs_error", lambda x: x.quantile(0.95)),
            max_error_p99=("max_abs_error", lambda x: x.quantile(0.99)),
            derivative_rmse_median=("derivative_rmse", "median"),
            derivative_rmse_p95=("derivative_rmse", lambda x: x.quantile(0.95)),
            node_max_error_median=("node_max_error", "median"),
            node_max_error_p95=("node_max_error", lambda x: x.quantile(0.95)),
            node_max_error_p99=("node_max_error", lambda x: x.quantile(0.99)),
            duplicate_x_probe_count=("duplicate_x_probe_count", "sum"),
            elapsed_seconds_total=("elapsed_seconds_total", "max"),
        )
        .sort_values(["eval_resolution", "dense_outputs", "rmse_p95", "rmse_median"])
        .reset_index(drop=True)
    )
    for threshold in RMSE_THRESHOLDS:
        coverage = results.groupby("configuration")["rmse"].apply(lambda x, t=threshold: float(np.mean(x <= t)))
        summary[f"rmse_under_{threshold:g}"] = summary["configuration"].map(coverage)
    for threshold in NODE_THRESHOLDS:
        coverage = results.groupby("configuration")["node_max_error"].apply(lambda x, t=threshold: float(np.mean(x <= t)))
        summary[f"all_nodes_under_{threshold:g}"] = summary["configuration"].map(coverage)
    for threshold in NODE_THRESHOLDS:
        coverage = results.groupby("configuration")["max_abs_error"].apply(lambda x, t=threshold: float(np.mean(x <= t)))
        summary[f"all_eval_points_under_{threshold:g}"] = summary["configuration"].map(coverage)

    threshold_rows = []
    merged = results.merge(subsets, on=["dataset_index", "configuration"], how="left")
    subset_columns = ["subset_all", "subset_custom_ish", "subset_gate_pulse_heavy"]
    for configuration, group in merged.groupby("configuration"):
        for subset_name in subset_columns:
            members = group[group[subset_name]]
            if members.empty:
                continue
            for threshold in RMSE_THRESHOLDS:
                threshold_rows.append(
                    {
                        "configuration": configuration,
                        "subset": subset_name.replace("subset_", ""),
                        "metric": "rmse",
                        "threshold": threshold,
                        "coverage": float(np.mean(members.rmse <= threshold)),
                        "shapes": int(len(members)),
                    }
                )
            for threshold in NODE_THRESHOLDS:
                threshold_rows.append(
                    {
                        "configuration": configuration,
                        "subset": subset_name.replace("subset_", ""),
                        "metric": "all_nodes",
                        "threshold": threshold,
                        "coverage": float(np.mean(members.node_max_error <= threshold)),
                        "shapes": int(len(members)),
                    }
                )
            for threshold in NODE_THRESHOLDS:
                threshold_rows.append(
                    {
                        "configuration": configuration,
                        "subset": subset_name.replace("subset_", ""),
                        "metric": "all_eval_points",
                        "threshold": threshold,
                        "coverage": float(np.mean(members.max_abs_error <= threshold)),
                        "shapes": int(len(members)),
                    }
                )
        for topology, members in group.groupby("topology"):
            for threshold in RMSE_THRESHOLDS:
                threshold_rows.append(
                    {
                        "configuration": configuration,
                        "subset": f"topology_{topology}",
                        "metric": "rmse",
                        "threshold": threshold,
                        "coverage": float(np.mean(members.rmse <= threshold)),
                        "shapes": int(len(members)),
                    }
                )
            for threshold in NODE_THRESHOLDS:
                threshold_rows.append(
                    {
                        "configuration": configuration,
                        "subset": f"topology_{topology}",
                        "metric": "all_nodes",
                        "threshold": threshold,
                        "coverage": float(np.mean(members.node_max_error <= threshold)),
                        "shapes": int(len(members)),
                    }
                )
            for threshold in NODE_THRESHOLDS:
                threshold_rows.append(
                    {
                        "configuration": configuration,
                        "subset": f"topology_{topology}",
                        "metric": "all_eval_points",
                        "threshold": threshold,
                        "coverage": float(np.mean(members.max_abs_error <= threshold)),
                        "shapes": int(len(members)),
                    }
                )
    thresholds = pd.DataFrame(threshold_rows)

    topology = (
        results.groupby(["configuration", "topology"], as_index=False)
        .agg(
            shapes=("dataset_index", "size"),
            rmse_median=("rmse", "median"),
            rmse_p95=("rmse", lambda x: x.quantile(0.95)),
            node_max_error_p95=("node_max_error", lambda x: x.quantile(0.95)),
        )
    )
    return summary, thresholds, topology


def pareto_frontier(summary: pd.DataFrame, *, metric: str) -> pd.DataFrame:
    selected = []
    frame = summary.sort_values(["dense_outputs", metric, "rmse_median"]).reset_index(drop=True)
    for index, row in frame.iterrows():
        dominated = (
            (frame.dense_outputs <= row.dense_outputs)
            & (frame[metric] <= row[metric])
            & ((frame.dense_outputs < row.dense_outputs) | (frame[metric] < row[metric]))
        ).any()
        if not dominated:
            selected.append(index)
    return frame.loc[selected].reset_index(drop=True)


def pseudo_information_criteria(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    eps = 1e-18
    for configuration, group in results.groupby("configuration"):
        first = group.iloc[0]
        n = int(len(group) * int(first.eval_resolution))
        sse = float(np.sum((group.rmse.to_numpy() ** 2) * int(first.eval_resolution)))
        log_term = n * math.log(max(sse / max(n, 1), eps))
        penalties = {
            "dense_outputs": float(first.dense_outputs),
            "effective_index_bits": float(first.effective_index_bits),
            "stored_floats": float(first.stored_floats),
            "stored_bytes_float32": float(first.stored_bytes_float32),
        }
        for penalty_name, k in penalties.items():
            rows.append(
                {
                    "configuration": configuration,
                    "penalty_basis": penalty_name,
                    "n_samples": n,
                    "sse": sse,
                    "k": k,
                    "pseudo_aic": log_term + 2.0 * k,
                    "pseudo_bic": log_term + math.log(max(n, 2)) * k,
                }
            )
    return pd.DataFrame(rows)


def _assert_threshold_monotonic(thresholds: pd.DataFrame) -> None:
    for (configuration, subset, metric), group in thresholds.groupby(["configuration", "subset", "metric"]):
        ordered = group.sort_values("threshold")
        values = ordered.coverage.to_numpy()
        if np.any(np.diff(values) < -1e-12):
            raise AssertionError(f"non-monotonic coverage for {configuration}/{subset}/{metric}")


def _assert_author_split(dataset: CurveDataset) -> None:
    train = set(dataset.frame.iloc[dataset.train_indices].author_id.astype(str))
    validation = set(dataset.frame.iloc[dataset.validation_indices].author_id.astype(str))
    overlap = train & validation
    if overlap:
        raise AssertionError(f"author split overlaps: {sorted(list(overlap))[:5]}")


def _write_plots(summary: pd.DataFrame, thresholds: pd.DataFrame, output_dir: Path) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for metric, label in (("rmse_median", "Median RMSE"), ("rmse_p95", "P95 RMSE"), ("node_max_error_p95", "P95 node max error")):
        fig, ax = plt.subplots(figsize=(10, 6))
        for family, group in summary.groupby("family"):
            ax.scatter(group.dense_outputs, group[metric], label=family, alpha=0.75)
        ax.set_xlabel("Dense model outputs")
        ax.set_ylabel(label)
        ax.set_title(f"{label} vs output budget")
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"{metric}_vs_dense_outputs.png", dpi=170)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    direct = summary[(summary.family == "direct_grid") & (summary.eval_resolution == summary.eval_resolution.max())]
    ax.plot(direct.continuous_scalars, direct.rmse_p95, marker="o", label="direct grid P95")
    ax.plot(direct.continuous_scalars, direct.node_max_error_p95, marker="o", label="direct grid node P95")
    for _, row in direct.iterrows():
        ax.annotate(str(row.candidate), (row.continuous_scalars, row.rmse_p95), fontsize=8)
    ax.set_xlabel("Grid samples / dense outputs")
    ax.set_ylabel("Error")
    ax.set_title("Direct-grid baselines, including factor-of-3 grids")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "direct_grid_factor3_comparison.png", dpi=170)
    plt.close(fig)

    main_threshold = thresholds[
        (thresholds.subset == "all") & (thresholds.metric == "rmse") & (thresholds.threshold.isin([0.01, 0.02, 0.05]))
    ].merge(summary[["configuration", "dense_outputs"]], on="configuration", how="left")
    fig, ax = plt.subplots(figsize=(10, 6))
    for threshold, group in main_threshold.groupby("threshold"):
        ax.scatter(group.dense_outputs, group.coverage, label=f"RMSE <= {threshold:g}", alpha=0.75)
    ax.set_xlabel("Dense model outputs")
    ax.set_ylabel("Coverage")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Threshold coverage vs output budget")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "threshold_coverage_vs_dense_outputs.png", dpi=170)
    plt.close(fig)


def _markdown_table(frame: pd.DataFrame, columns: list[str], limit: int = 12) -> str:
    if frame.empty:
        return "_No rows._"
    selected = frame[columns].head(limit).copy()
    for column in selected.select_dtypes(include=["float", "float64", "float32"]).columns:
        selected[column] = selected[column].map(lambda x: f"{x:.6g}")
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in selected.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(map(str, row)) + " |")
    return "\n".join(lines)


def write_findings(
    summary: pd.DataFrame,
    thresholds: pd.DataFrame,
    topology: pd.DataFrame,
    info: pd.DataFrame,
    output_dir: Path,
) -> None:
    frontier_p95 = pareto_frontier(summary, metric="rmse_p95")
    frontier_node = pareto_frontier(summary, metric="node_max_error_p95")
    direct = summary[summary.family == "direct_grid"].sort_values(["eval_resolution", "continuous_scalars"])
    structured = summary[summary.family == "phase_residual"].sort_values(["rmse_p95", "rmse_median"])
    coverage_02 = thresholds[
        (thresholds.subset == "all") & (thresholds.metric == "rmse") & (thresholds.threshold == 0.02)
    ][["configuration", "coverage"]].rename(columns={"coverage": "rmse_under_0.02"})
    node_02 = thresholds[
        (thresholds.subset == "all") & (thresholds.metric == "all_nodes") & (thresholds.threshold == 0.02)
    ][["configuration", "coverage"]].rename(columns={"coverage": "all_nodes_under_0.02"})
    coverage_packet = (
        summary[["configuration", "dense_outputs", "family", "rmse_median", "rmse_p95", "node_max_error_p95"]]
        .merge(coverage_02, on="configuration", how="left")
        .merge(node_02, on="configuration", how="left")
        .sort_values(["dense_outputs", "rmse_p95"])
    )
    content = f"""# Experiment 6 Findings: Codebook-Generation Approach Selection

## What this run answers

This is a decision packet, not a final codebook freeze.  It compares direct-grid baselines, phase-aware residual codebook families, factor-of-3 grids, threshold coverage, editor-node preservation, and complexity.  The stock 15 bases remain provisional until controlled canonical saves replace them.

## Efficient frontier by held-out P95 RMSE

{_markdown_table(frontier_p95, ['configuration', 'family', 'dense_outputs', 'stored_floats', 'rmse_median', 'rmse_p95', 'node_max_error_p95'])}

## Efficient frontier by editor-node preservation

{_markdown_table(frontier_node, ['configuration', 'family', 'dense_outputs', 'stored_floats', 'rmse_p95', 'node_max_error_p95', 'all_nodes_under_0.02']) if 'all_nodes_under_0.02' in frontier_node else _markdown_table(frontier_node.merge(node_02, on='configuration', how='left'), ['configuration', 'family', 'dense_outputs', 'stored_floats', 'rmse_p95', 'node_max_error_p95', 'all_nodes_under_0.02'])}

## Factor-of-3 grid sanity check

{_markdown_table(direct, ['candidate', 'eval_resolution', 'dense_outputs', 'rmse_median', 'rmse_p95', 'node_max_error_p95'])}

The factor-of-3 grids are included because the subdivision audit found enough triplet/non-power-of-two structure in custom-ish LFOs that powers of two alone would be a suspicious convenience.

## Coverage packet for discussion

{_markdown_table(coverage_packet, ['configuration', 'family', 'dense_outputs', 'rmse_median', 'rmse_p95', 'rmse_under_0.02', 'node_max_error_p95', 'all_nodes_under_0.02'], limit=20)}

## Best structured candidates by tail error

{_markdown_table(structured, ['configuration', 'dense_outputs', 'stored_floats', 'effective_index_bits', 'rmse_median', 'rmse_p95', 'node_max_error_p95'], limit=15)}

## Topology tail check

{_markdown_table(topology.sort_values(['configuration', 'topology']), ['configuration', 'topology', 'shapes', 'rmse_median', 'rmse_p95', 'node_max_error_p95'], limit=24)}

## Pseudo-AIC/BIC caveat

The information-criterion tables are diagnostics over held-out sampled curves, not literal likelihood claims.  They are useful mainly for seeing how sensitive rankings are to dense outputs, index bits, and dictionary storage penalties.

## What I would look at first

1. If Grid96 or Grid192 closes most of the node-preservation gap, that is evidence the later sparse editor refit needs factor-of-3 awareness even if the production representation is structured.
2. If additive shared+topology keeps dominating P95 at modest extra outputs, it remains the most plausible production-codebook recipe.
3. If node coverage disagrees with RMSE coverage, trust node coverage for editor-state plausibility. RMSE can forgive tiny but visually meaningful point mistakes.
4. If direct-grid baselines beat everything on node preservation by a lot, we should consider a hybrid: structured LFO code for search reduction plus a small direct residual/grid head.

## Files to inspect

- `summary.csv`: main quality/complexity table.
- `threshold_coverage.csv`: RMSE and all-node threshold coverage.
- `per_shape_results.csv`: held-out rows with reconstruction and node metrics.
- `selected_paths.csv`: oracle codes/phases/gains for structured candidates.
- `pseudo_information_criteria.csv`: AIC/BIC-style diagnostics.
- `plots/`: Pareto and threshold figures.
"""
    (output_dir / "EXPERIMENT_6_FINDINGS.md").write_text(content, encoding="utf-8")


def run_experiment6(
    catalog_path: Path,
    codebook_path: Path,
    experiment4_dir: Path,
    output_dir: Path,
    *,
    quick: bool = False,
    beam_width: int = 64,
    finalist_beam_width: int = 128,
    max_shapes: int | None = None,
    parallel: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    del codebook_path  # Kept in the signature so the runner records the same prerequisites.
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs, grids, eval_resolutions = _make_jobs(experiment4_dir, quick=quick, beam_width=beam_width)
    started_at = None
    previous_progress = _read_progress(output_dir)
    current_signature = _jobs_signature(jobs)
    if (
        previous_progress
        and previous_progress.get("started_at")
        and previous_progress.get("run_signature") == current_signature
    ):
        started_at = str(previous_progress["started_at"])
    progress = _progress_from_jobs(output_dir, jobs, started_at=started_at)
    progress["parallel"] = max(1, int(parallel))
    _write_progress(output_dir, progress)

    pending = [job for job in jobs if not _checkpoint_done(output_dir, job.job_id)]
    skipped = len(jobs) - len(pending)
    print(
        f"Experiment 6: {skipped}/{len(jobs)} candidate checkpoints already complete; "
        f"{len(pending)} remaining; parallel={max(1, int(parallel))}",
        flush=True,
    )

    if pending:
        if parallel <= 1:
            datasets: dict[int, CurveDataset] = {}
            processed: set[str] = set()
            phase_groups: dict[tuple[int, int, str], list[CandidateJob]] = {}
            for job in jobs:
                if job.kind == "phase_residual" and job.chain is not None:
                    phase_groups.setdefault(_phase_group_key(job), []).append(job)

            def dataset_for(resolution: int) -> CurveDataset:
                if resolution not in datasets:
                    datasets[resolution] = load_curve_dataset(catalog_path, resolution=resolution)
                    _assert_author_split(datasets[resolution])
                return datasets[resolution]

            for job in jobs:
                if job.job_id in processed or _checkpoint_done(output_dir, job.job_id):
                    continue
                if job.kind == "phase_residual" and job.chain is not None:
                    group = phase_groups[_phase_group_key(job)]
                    active_jobs = [member for member in group if not _checkpoint_done(output_dir, member.job_id)]
                    labels = ", ".join(member.chain.name for member in active_jobs if member.chain is not None)
                    print(f"Combined depth group @ eval {job.eval_resolution}: {labels}", flush=True)
                else:
                    group = [job]
                    active_jobs = group
                    print(job.label, flush=True)
                progress = _read_progress(output_dir) or progress
                for row in progress["jobs"]:
                    if row["job_id"] in {member.job_id for member in active_jobs}:
                        row["status"] = "running"
                        row["started_at"] = _now_iso()
                _write_progress(output_dir, progress)
                dataset = dataset_for(job.eval_resolution)
                if job.kind == "phase_residual" and job.chain is not None:
                    _run_phase_job_group_from_dataset(
                        dataset,
                        output_dir,
                        group,
                        max_shapes=max_shapes,
                        quick=quick,
                    )
                    processed.update(member.job_id for member in group)
                else:
                    _run_candidate_job_from_dataset(dataset, output_dir, job, max_shapes=max_shapes, quick=quick)
                    processed.add(job.job_id)
                progress = _progress_from_jobs(output_dir, jobs, started_at=str(progress.get("started_at")))
                progress["parallel"] = 1
                _write_progress(output_dir, progress)
                print(experiment6_status(output_dir), flush=True)
        else:
            with ProcessPoolExecutor(max_workers=max(1, int(parallel))) as executor:
                future_by_job = {}
                progress = _read_progress(output_dir) or progress
                for job in pending:
                    for row in progress["jobs"]:
                        if row["job_id"] == job.job_id:
                            row["status"] = "running"
                            row["started_at"] = _now_iso()
                    future = executor.submit(
                        _run_candidate_job,
                        catalog_path,
                        output_dir,
                        job,
                        max_shapes=max_shapes,
                        quick=quick,
                    )
                    future_by_job[future] = job
                progress["parallel"] = max(1, int(parallel))
                _write_progress(output_dir, progress)
                for future in as_completed(future_by_job):
                    job = future_by_job[future]
                    future.result()
                    print(f"Completed {job.label}", flush=True)
                    progress = _progress_from_jobs(output_dir, jobs, started_at=str(progress.get("started_at")))
                    progress["parallel"] = max(1, int(parallel))
                    _write_progress(output_dir, progress)
                    print(experiment6_status(output_dir), flush=True)

    all_results: list[pd.DataFrame] = []
    all_subsets: list[pd.DataFrame] = []
    all_paths: list[pd.DataFrame] = []
    all_usage: list[pd.DataFrame] = []
    manifest_candidates = []
    missing = [job.job_id for job in jobs if not _checkpoint_done(output_dir, job.job_id)]
    if missing:
        raise RuntimeError(f"Experiment 6 incomplete; missing checkpoints: {missing[:8]}")
    for job in jobs:
        result, subsets, paths, usage = _load_checkpoint(output_dir, job)
        all_results.append(result)
        all_subsets.append(subsets)
        all_paths.append(paths)
        if not usage.empty:
            all_usage.append(usage)
        manifest_candidates.append(
            {
                "job_id": job.job_id,
                "kind": job.kind,
                "width": job.width,
                "name": None if job.chain is None else job.chain.name,
                "depth": 0 if job.chain is None else len(job.chain.stages),
                "stage_widths": None if job.chain is None else job.chain.stage_widths,
                "eval_resolution": job.eval_resolution,
                "beam_width": job.beam_width,
                "training_feature_grid": job.training_feature_grid,
                "weight": job.weight,
            }
        )

    results = pd.concat(all_results, ignore_index=True)
    subsets = pd.concat(all_subsets, ignore_index=True)
    paths = pd.concat(all_paths, ignore_index=True, sort=False)
    usage = pd.concat(all_usage, ignore_index=True, sort=False) if all_usage else pd.DataFrame()
    required_numeric = [
        "rmse",
        "max_abs_error",
        "derivative_rmse",
        "node_max_error",
        "node_mean_error",
        "node_p95_error",
        "dense_outputs",
        "stored_floats",
        "stored_bytes_float32",
    ]
    if not np.isfinite(results[required_numeric].to_numpy()).all():
        raise AssertionError("Experiment 6 produced NaN or infinite required metrics")

    summary, thresholds, topology = summarize_results(results, subsets)
    _assert_threshold_monotonic(thresholds)
    info = pseudo_information_criteria(results)
    frontier_p95 = pareto_frontier(summary, metric="rmse_p95")
    frontier_node = pareto_frontier(summary, metric="node_max_error_p95")

    results.to_csv(output_dir / "per_shape_results.csv", index=False)
    paths.to_csv(output_dir / "selected_paths.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    thresholds.to_csv(output_dir / "threshold_coverage.csv", index=False)
    topology.to_csv(output_dir / "topology_summary.csv", index=False)
    info.to_csv(output_dir / "pseudo_information_criteria.csv", index=False)
    frontier_p95.to_csv(output_dir / "pareto_frontier_p95.csv", index=False)
    frontier_node.to_csv(output_dir / "pareto_frontier_node.csv", index=False)
    if not usage.empty:
        usage.to_csv(output_dir / "codeword_utilization.csv", index=False)

    manifest = {
        "experiment": 6,
        "scope": "codebook_generation_approach_selection",
        "quick": quick,
        "seed": SEED,
        "grid_baselines": list(grids),
        "factor_of_3_grids": [width for width in grids if width % 3 == 0],
        "eval_resolutions": list(eval_resolutions),
        "beam_width": beam_width,
        "finalist_beam_width": finalist_beam_width,
        "finalist_beam_note": "reserved for manual finalist reruns after reviewing this decision packet",
        "parallel": max(1, int(parallel)),
        "candidate_count": len(manifest_candidates),
        "candidates": manifest_candidates,
        "provisional_stock_bases": 15,
        "depth5_status": "not evaluated from frozen Experiment 4 artifacts; requires retraining deeper endpoint chains",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    _write_plots(summary, thresholds, output_dir)
    write_findings(summary, thresholds, topology, info, output_dir)
    from .experiment6_analysis import run_experiment6_analysis

    run_experiment6_analysis(output_dir)
    progress = _progress_from_jobs(output_dir, jobs, started_at=str((_read_progress(output_dir) or {}).get("started_at", _now_iso())))
    progress["status"] = "complete"
    progress["parallel"] = max(1, int(parallel))
    _write_progress(output_dir, progress)
    print(summary[["configuration", "dense_outputs", "rmse_median", "rmse_p95", "node_max_error_p95"]].to_string(index=False))
    return results, summary

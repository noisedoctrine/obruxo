"""Experiment 5: per-code phase-alignment oracle and search-gap analysis."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .alignment5 import (
    AlignmentResult,
    clipped_grid_reference,
    dense_align_cpu,
    exact_align_cpu,
    exact_align_xpu,
    fft128_local9_cpu,
    phase_distance,
    select_best_code,
)
from .phase4 import PhaseChain, PhaseEncoding, circular_shift, decode_phase_chain
from .stacked import TOPOLOGY_NAMES, _features, load_curve_dataset, metric_arrays


FROZEN_CODEBOOKS = (
    "phase_shared",
    "phase_switch_1",
    "phase_switch_2",
    "phase_additive_k8",
    "phase_additive_k16",
)


def _conditions(dataset, chain: PhaseChain, indices: np.ndarray) -> np.ndarray:
    return dataset.topology[indices].astype(np.int32) if chain.topology_conditioned else np.zeros(len(indices), np.int32)


def _align_conditions(
    residual: np.ndarray,
    stage: np.ndarray,
    conditions: np.ndarray,
    solver: str,
) -> AlignmentResult:
    width = stage.shape[1]
    error = np.empty((len(residual), width), np.float64)
    phase = np.empty_like(error)
    gain = np.empty_like(error)
    for condition in range(stage.shape[0]):
        members = np.flatnonzero(conditions == condition)
        if not len(members):
            continue
        if solver == "exact":
            result = exact_align_xpu(residual[members], stage[condition])
        elif solver == "baseline":
            result = fft128_local9_cpu(residual[members], stage[condition])
        else:
            raise ValueError(solver)
        error[members], phase[members], gain[members] = result.error, result.phase, result.gain
    return AlignmentResult(error, phase, gain)


def greedy_corpus_alignment(dataset, chain: PhaseChain, indices: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, PhaseEncoding]:
    targets = dataset.curves[indices]
    conditions = _conditions(dataset, chain, indices)
    exact_base = exact_align_xpu(targets, chain.bases, fixed_gain=1.0)
    baseline_base = fft128_local9_cpu(targets, chain.bases, fixed_gain=1.0)
    base_code, _, base_phase, _ = select_best_code(exact_base)
    baseline_base_code, *_ = select_best_code(baseline_base)
    prefix = circular_shift(chain.bases[base_code], base_phase)
    per_code_frames = []

    def alignment_frame(stage_name: str, exact: AlignmentResult, baseline: AlignmentResult, condition_values: np.ndarray) -> pd.DataFrame:
        rows, codes = exact.error.shape
        return pd.DataFrame({
            "instance": np.repeat(np.arange(rows), codes),
            "dataset_index": np.repeat(indices, codes),
            "stage": stage_name,
            "condition": np.repeat(condition_values, codes),
            "code": np.tile(np.arange(codes), rows),
            "exact_error": exact.error.ravel(),
            "exact_phase": exact.phase.ravel(),
            "exact_gain": exact.gain.ravel(),
            "baseline_error": baseline.error.ravel(),
            "baseline_phase": baseline.phase.ravel(),
            "baseline_gain": baseline.gain.ravel(),
        })

    per_code_frames.append(alignment_frame("base", exact_base, baseline_base, np.zeros(len(indices), np.int32)))
    stage_indices: list[np.ndarray] = []
    stage_phases: list[np.ndarray] = []
    stage_gains: list[np.ndarray] = []
    stage_change = []
    prefix_errors = [np.sqrt(np.mean((prefix - targets) ** 2, axis=1))]
    for stage_index, stage in enumerate(chain.stages):
        residual = targets - prefix
        exact = _align_conditions(residual, stage, conditions, "exact")
        baseline = _align_conditions(residual, stage, conditions, "baseline")
        code, _, phase, gain = select_best_code(exact)
        baseline_code, *_ = select_best_code(baseline)
        rows = np.arange(len(targets))
        addition = circular_shift(stage[conditions, code], phase) * gain[:, None]
        candidate = np.clip(prefix + addition, 0.0, 1.0)
        old_error = np.mean((prefix - targets) ** 2, axis=1)
        new_error = np.mean((candidate - targets) ** 2, axis=1)
        fallback = new_error >= old_error - 1e-12
        code[fallback], phase[fallback], gain[fallback] = 0, 0.0, 0.0
        candidate[fallback] = prefix[fallback]
        prefix = candidate
        stage_indices.append(code.astype(np.int16))
        stage_phases.append(phase.astype(np.float32))
        stage_gains.append(gain.astype(np.float32))
        prefix_errors.append(np.sqrt(np.mean((prefix - targets) ** 2, axis=1)))
        stage_change.append(np.mean(code != baseline_code))
        per_code_frames.append(alignment_frame(chain.stage_labels[stage_index], exact, baseline, conditions))

    encoding = PhaseEncoding(
        base_code.astype(np.int16), base_phase.astype(np.float32),
        stage_indices, stage_phases, stage_gains,
    )
    metrics = metric_arrays(targets, prefix)
    frame = dataset.frame.iloc[indices].reset_index(drop=True)
    selected = pd.DataFrame({
        "dataset_index": indices,
        "preset_id": frame.preset_id,
        "shape_signature": frame.shape_signature,
        "topology": [TOPOLOGY_NAMES[value] for value in dataset.topology[indices]],
        "configuration": chain.name,
        "base_index": base_code,
        "base_phase": base_phase,
        "base_code_changed": base_code != baseline_base_code,
        **metrics,
    })
    for stage_index, label in enumerate(chain.stage_labels):
        selected[f"stage_{stage_index+1}_label"] = label
        selected[f"stage_{stage_index+1}_index"] = stage_indices[stage_index]
        selected[f"stage_{stage_index+1}_phase"] = stage_phases[stage_index]
        selected[f"stage_{stage_index+1}_gain"] = stage_gains[stage_index]
        selected[f"stage_{stage_index+1}_code_change_rate"] = stage_change[stage_index]
        selected[f"prefix_{stage_index+1}_rmse"] = prefix_errors[stage_index + 1]
    return pd.concat(per_code_frames, ignore_index=True), selected, encoding


def solver_benchmark(dataset, chain: PhaseChain, indices: np.ndarray, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    import torch
    targets = dataset.curves[indices]
    conditions = _conditions(dataset, chain, indices)
    # Use observed residuals against stage-one codes, cycling code identities so
    # dense 65k references stay memory-bounded while covering every topology.
    prefix_result = exact_align_cpu(targets, chain.bases, fixed_gain=1.0)
    base, *_ = select_best_code(prefix_result)
    _, _, base_phase, _ = select_best_code(prefix_result)
    prefix = circular_shift(chain.bases[base], base_phase)
    residual = targets - prefix
    stage = chain.stages[0]
    pair_codes = np.stack([
        stage[conditions[row], 1 + row % (stage.shape[1] - 1)] for row in range(len(indices))
    ])[:, None]
    methods = []
    results = {}
    # Compile/initialize XPU before measuring its steady-state path.
    exact_align_xpu(residual[:1], pair_codes[:1])
    torch.xpu.synchronize()
    for name, function in (
        ("fft128_local9", lambda: fft128_local9_cpu(residual, pair_codes)),
        ("grid1024", lambda: dense_align_cpu(residual, pair_codes, 1024)),
        ("grid4096", lambda: dense_align_cpu(residual, pair_codes, 4096)),
        ("grid65536", lambda: dense_align_cpu(residual, pair_codes, 65536)),
        ("exact_cpu", lambda: exact_align_cpu(residual, pair_codes)),
        ("exact_xpu", lambda: exact_align_xpu(residual, pair_codes)),
    ):
        started = time.perf_counter()
        value = function()
        if torch.xpu.is_available():
            torch.xpu.synchronize()
        elapsed = time.perf_counter() - started
        results[name] = value
        methods.append({
            "solver": name, "benchmark": "one_code_per_target", "pairs": len(indices), "elapsed_seconds": elapsed,
            "pairs_per_second": len(indices) / elapsed,
            "error_median": float(np.median(value.error)),
            "error_p95": float(np.quantile(value.error, .95)),
        })
    # SIMD/SIMT throughput benchmark: many targets against the complete stage dictionary.
    batch_count = min(256, len(dataset.validation_indices))
    batch_indices = dataset.validation_indices[:batch_count]
    batch_targets = dataset.curves[batch_indices]
    batch_base = exact_align_cpu(batch_targets, chain.bases, fixed_gain=1.0)
    batch_code, _, batch_phase, _ = select_best_code(batch_base)
    batch_prefix = circular_shift(chain.bases[batch_code], batch_phase)
    batch_residual = batch_targets - batch_prefix
    batch_codes = chain.stages[0][0]
    for name, function in (
        ("exact_cpu_batch", lambda: exact_align_cpu(batch_residual, batch_codes)),
        ("exact_xpu_batch", lambda: exact_align_xpu(batch_residual, batch_codes)),
    ):
        started = time.perf_counter(); value = function()
        if torch.xpu.is_available(): torch.xpu.synchronize()
        elapsed = time.perf_counter() - started
        pair_count = batch_count * len(batch_codes)
        methods.append({
            "solver": name, "benchmark": "full_dictionary", "pairs": pair_count,
            "elapsed_seconds": elapsed, "pairs_per_second": pair_count / elapsed,
            "error_median": float(np.median(value.error)), "error_p95": float(np.quantile(value.error, .95)),
        })
    exact = results["exact_cpu"]
    pair_rows = []
    for name, value in results.items():
        pair_rows.append(pd.DataFrame({
            "solver": name, "pair": np.arange(len(indices)),
            "error": value.error[:, 0], "phase": value.phase[:, 0], "gain": value.gain[:, 0],
            "error_gap_to_exact": value.error[:, 0] - exact.error[:, 0],
            "phase_distance_to_exact": phase_distance(value.phase[:, 0], exact.phase[:, 0]),
        }))
    return pd.DataFrame(methods), pd.concat(pair_rows, ignore_index=True)


def clipped_benchmark(dataset, chain: PhaseChain, indices: np.ndarray) -> pd.DataFrame:
    targets = dataset.curves[indices]
    base_result = exact_align_cpu(targets, chain.bases, fixed_gain=1.0)
    base, _, phase, _ = select_best_code(base_result)
    prefix = circular_shift(chain.bases[base], phase)
    conditions = _conditions(dataset, chain, indices)
    codes = np.stack([chain.stages[0][conditions[row], 1 + row % (chain.stages[0].shape[1] - 1)] for row in range(len(indices))])[:, None]
    started = time.perf_counter()
    reference = clipped_grid_reference(targets, prefix, codes, positions=4096)
    elapsed = time.perf_counter() - started
    dense_subset = min(4, len(indices))
    dense = clipped_grid_reference(
        targets[:dense_subset], prefix[:dense_subset], codes[:dense_subset],
        positions=65536, top_peaks=8, refine_rounds=2,
    )
    rows = pd.DataFrame({
        "pair": np.arange(len(indices)), "grid4096_refined_error": reference.error[:, 0],
        "phase": reference.phase[:, 0], "gain": reference.gain[:, 0],
        "elapsed_seconds_total": elapsed,
    })
    rows["grid65536_refined_error"] = np.nan
    rows.loc[: dense_subset - 1, "grid65536_refined_error"] = dense.error[:, 0]
    return rows


def exact_feature_beam(dataset, chain: PhaseChain, indices: np.ndarray, beam_width: int) -> np.ndarray:
    targets_full = dataset.curves[indices]
    targets = _features(targets_full).astype(np.float32)
    bases = _features(chain.bases).astype(np.float32)
    conditions = _conditions(dataset, chain, indices)
    final = np.empty_like(targets)
    for start in range(0, len(targets), 8):
        stop = min(start + 8, len(targets))
        target = targets[start:stop]
        result = exact_align_xpu(target, bases, fixed_gain=1.0)
        width = min(beam_width, len(bases))
        codes = np.argpartition(result.error, width - 1, axis=1)[:, :width]
        rows = np.arange(len(target))[:, None]
        order = np.argsort(result.error[rows, codes], axis=1)
        codes = np.take_along_axis(codes, order, axis=1)
        phases = result.phase[rows, codes]
        prefix = np.stack([circular_shift(bases[codes[row]], phases[row]) for row in range(len(target))])
        for stage_index, stage in enumerate(chain.stages):
            stage_features = _features(stage.reshape(-1, stage.shape[-1])).reshape(stage.shape[0], stage.shape[1], -1)
            dictionaries = stage_features[conditions[start:stop]]
            b, w, f = prefix.shape
            remaining = (target[:, None] - prefix).reshape(b * w, f)
            repeated = np.repeat(dictionaries, w, axis=0)
            aligned = exact_align_xpu(remaining, repeated)
            k = repeated.shape[1]
            shifted = []
            for item in range(b * w):
                shifted.append(circular_shift(repeated[item], aligned.phase[item]) * aligned.gain[item, :, None])
            additions = np.asarray(shifted).reshape(b, w, k, f)
            candidates = np.clip(prefix[:, :, None] + additions, 0.0, 1.0)
            error = np.mean((target[:, None, None] - candidates) ** 2, axis=3).reshape(b, -1)
            next_width = min(beam_width, error.shape[1])
            choice = np.argpartition(error, next_width - 1, axis=1)[:, :next_width]
            choice = np.take_along_axis(choice, np.argsort(np.take_along_axis(error, choice, axis=1), axis=1), axis=1)
            parent, code = choice // k, choice % k
            prefix = candidates[np.arange(b)[:, None], parent, code]
        final[start:stop] = prefix[:, 0]
    return np.sqrt(np.mean((final - targets) ** 2, axis=1))


def beam_benchmark(dataset, chains: list[PhaseChain], full_indices: np.ndarray, quick: bool) -> pd.DataFrame:
    rows = []
    audit = full_indices[: max(8, len(full_indices) // 10)]
    for chain in chains:
        widths = (8, 32) if quick else (64,)
        for width in widths:
            started = time.perf_counter()
            rmse = exact_feature_beam(dataset, chain, full_indices, width)
            rows.append({
                "configuration": chain.name, "scope": "full", "beam_width": width,
                "shapes": len(rmse), "rmse_median": np.median(rmse),
                "rmse_p95": np.quantile(rmse, .95), "elapsed_seconds": time.perf_counter() - started,
            })
        audit_values = {}
        for width in ((8, 32) if quick else (8, 32, 64, 128, 256)):
            started = time.perf_counter()
            rmse = exact_feature_beam(dataset, chain, audit, width)
            audit_values[width] = rmse
            rows.append({
                "configuration": chain.name, "scope": "audit10", "beam_width": width,
                "shapes": len(rmse), "rmse_median": np.median(rmse),
                "rmse_p95": np.quantile(rmse, .95), "elapsed_seconds": time.perf_counter() - started,
            })
        if not quick:
            improvement_share = float(np.mean(audit_values[256] < audit_values[64] - 1e-5))
            if improvement_share > .001:
                started = time.perf_counter()
                rmse = exact_feature_beam(dataset, chain, full_indices, 128)
                rows.append({
                    "configuration": chain.name, "scope": "full_rerun", "beam_width": 128,
                    "shapes": len(rmse), "rmse_median": np.median(rmse),
                    "rmse_p95": np.quantile(rmse, .95), "elapsed_seconds": time.perf_counter() - started,
                    "audit_improvement_share": improvement_share,
                })
    return pd.DataFrame(rows)


def _table(frame: pd.DataFrame, columns: list[str]) -> str:
    selected = frame[columns].copy()
    for column in selected.select_dtypes(include=["float"]).columns:
        selected[column] = selected[column].map(lambda x: f"{x:.6g}")
    return "\n".join([
        "| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|",
        *["| " + " | ".join(map(str, row)) + " |" for row in selected.itertuples(index=False, name=None)],
    ])


def write_findings(solver: pd.DataFrame, selected_summary: pd.DataFrame,
                   changes: pd.DataFrame, beam: pd.DataFrame, output: Path) -> None:
    fastest = solver.sort_values("pairs_per_second", ascending=False).iloc[0]
    content = f"""# Experiment 5 Findings: Per-Code Phase-Alignment Oracle

## What was separated?

Every code was independently optimized for phase and gain before categorical selection. This separates alignment approximation, code changes caused by alignment, and layered beam pruning.

## Solver accuracy and speed

{_table(solver, ['solver', 'benchmark', 'pairs_per_second', 'error_median', 'error_p95'])}

The fastest measured solver was `{fastest.solver}`. Production selection should still be based on the error-gap tests, not speed alone.

## What changed on the corpus?

{_table(selected_summary, ['configuration', 'shapes', 'rmse_median', 'rmse_p95'])}

{_table(changes, ['configuration', 'stage', 'code_change_rate', 'median_error_gain'])}

## Layered beam-search gap

{_table(beam, ['configuration', 'scope', 'beam_width', 'rmse_median', 'rmse_p95', 'elapsed_seconds'])}

## Error accounting

1. `solver_pair_results.csv` measures phase/gain approximation for fixed codes.
2. `code_change_summary.csv` measures categorical changes after each code receives its own optimum.
3. `beam_summary.csv` measures layered pruning separately.

## Recommendation

Use the XPU analytic interval solver as the reference-alignment implementation when its CPU agreement and dense-grid checks pass. Retain FFT-128 only if its P95 error gap and code-change rate are negligible relative to that reference. Clipped-prefix results remain explicitly numerical and are reported separately from the exact residual-space solution.
"""
    output.write_text(content, encoding="utf-8")


def run_experiment5(catalog_path: Path, experiment4_dir: Path, output_dir: Path, *, quick: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    import torch
    output_dir.mkdir(parents=True, exist_ok=True)
    if not torch.xpu.is_available():
        raise RuntimeError("Experiment 5 full run requires the verified XPU device")
    dataset = load_curve_dataset(catalog_path)
    available = FROZEN_CODEBOOKS[:1] if quick else FROZEN_CODEBOOKS
    chains = [PhaseChain.load(experiment4_dir / "codebooks" / name) for name in available]
    validation = dataset.validation_indices[:128] if quick else dataset.validation_indices
    benchmark_indices = validation[:8 if quick else 24]
    clipped_indices = validation[:2 if quick else 12]

    per_code_parts = []
    selected_parts = []
    for chain in chains:
        print(f"Exact greedy alignment: {chain.name}", flush=True)
        per_code, selected, _ = greedy_corpus_alignment(dataset, chain, validation)
        per_code["configuration"] = chain.name
        per_code_parts.append(per_code)
        selected_parts.append(selected)
    per_code = pd.concat(per_code_parts, ignore_index=True)
    selected = pd.concat(selected_parts, ignore_index=True)
    selected_summary = selected.groupby("configuration", as_index=False).agg(
        shapes=("dataset_index", "size"), rmse_median=("rmse", "median"),
        rmse_p95=("rmse", lambda x: x.quantile(.95)), max_error_p95=("max_abs_error", lambda x: x.quantile(.95)),
    )
    change_rows = []
    for (configuration, stage), group in per_code.groupby(["configuration", "stage"]):
        exact_choice = group.loc[group.groupby("instance").exact_error.idxmin()][["instance", "code", "exact_error"]]
        baseline_choice = group.loc[group.groupby("instance").baseline_error.idxmin()][["instance", "code", "exact_error"]]
        merged = exact_choice.merge(baseline_choice, on="instance", suffixes=("_exact", "_baseline"))
        change_rows.append({
            "configuration": configuration, "stage": stage,
            "code_change_rate": np.mean(merged.code_exact != merged.code_baseline),
            "median_error_gain": np.median(merged.exact_error_baseline - merged.exact_error_exact),
        })
    changes = pd.DataFrame(change_rows)

    solver, solver_pairs = solver_benchmark(dataset, chains[0], benchmark_indices, output_dir)
    clipped = clipped_benchmark(dataset, chains[0], clipped_indices)
    beam = beam_benchmark(dataset, chains[:1] if quick else chains, validation, quick)

    per_code.to_csv(output_dir / "per_code_alignment.csv", index=False)
    selected.to_csv(output_dir / "selected_paths.csv", index=False)
    selected_summary.to_csv(output_dir / "selected_summary.csv", index=False)
    changes.to_csv(output_dir / "code_change_summary.csv", index=False)
    solver.to_csv(output_dir / "solver_summary.csv", index=False)
    solver_pairs.to_csv(output_dir / "solver_pair_results.csv", index=False)
    clipped.to_csv(output_dir / "clipped_reference.csv", index=False)
    beam.to_csv(output_dir / "beam_summary.csv", index=False)
    manifest = {
        "frozen_codebooks": list(available), "gain_bounds": [-2.0, 2.0],
        "train_instances": len(dataset.train_indices), "validation_instances": len(validation),
        "xpu_device": torch.xpu.get_device_name(0), "seed": 20260621,
        "scope": "alignment_only",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for name, group in solver_pairs.groupby("solver"):
        axes[0].plot(np.sort(group.error_gap_to_exact), np.linspace(0, 1, len(group)), label=name)
    axes[0].set(xlabel="MSE gap to exact", ylabel="CDF", title="Fixed-code alignment gap")
    axes[0].legend(fontsize=7)
    axes[1].bar(solver.solver, solver.pairs_per_second)
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].set(title="Alignment throughput", ylabel="pairs / second")
    change_plot = changes.groupby("stage").code_change_rate.mean()
    axes[2].bar(change_plot.index, change_plot.values)
    axes[2].tick_params(axis="x", rotation=45)
    axes[2].set(title="Code changes after exact alignment", ylabel="share")
    for axis in axes:
        axis.grid(alpha=.2)
    figure.tight_layout()
    figure.savefig(output_dir / "alignment_summary.png", dpi=170)
    plt.close(figure)

    # Representative phase landscape for the first benchmark pair.
    phases = np.linspace(0, 1, 4096, endpoint=False)
    target = dataset.curves[benchmark_indices[0]]
    code = chains[0].stages[0][0, 1]
    shifted = np.stack([circular_shift(code, value) for value in phases])
    denominator = np.sum(shifted * shifted, axis=1)
    gain = np.clip(np.divide(shifted @ target, denominator, out=np.zeros(len(phases)), where=denominator > 1e-12), -2, 2)
    landscape = np.mean((gain[:, None] * shifted - target) ** 2, axis=1)
    plt.figure(figsize=(11, 4))
    plt.plot(phases * 360, landscape)
    plt.xlabel("Phase (degrees)"); plt.ylabel("Residual MSE"); plt.title("Representative per-code phase landscape")
    plt.grid(alpha=.2); plt.tight_layout(); plt.savefig(output_dir / "phase_landscape.png", dpi=170); plt.close()

    write_findings(solver, selected_summary, changes, beam, output_dir / "EXPERIMENT_5_FINDINGS.md")
    print(selected_summary.to_string(index=False))
    return selected, selected_summary

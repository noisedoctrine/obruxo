#!/usr/bin/env python
"""Standalone Experiment 10 subdivision and direct-grid LFO audit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np

from lfo_era2.dataset import LfoShape, power_scale
from lfo_era2.metrics import rmse_per_curve
from lfo_era2.processed_corpus import (
    DEFAULT_CORPUS_DIR,
    DEFAULT_DENSE_RESOLUTION,
    DEFAULT_METADATA,
    build_lfo_corpus,
    load_processed_shape_corpus,
)


DEFAULT_POINT_BUDGETS = (24, 36, 48, 60, 72, 96, 100)
DEFAULT_SUBDIVISIONS = (24, 25, 32, 36, 37, 40, 48, 49, 60, 61, 64, 72, 73, 80, 96, 97, 100)
DEFAULT_FACTOR3_COMPARISONS = (
    (24, 25),
    (24, 32),
    (36, 37),
    (36, 40),
    (48, 49),
    (48, 64),
    (60, 61),
    (60, 64),
    (72, 73),
    (72, 80),
    (96, 97),
    (96, 100),
)
VITAL_MAX_POINTS = 100
BOUNDARY_EXACT_TOLERANCE = 1e-6
ERA2_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ERA2_ROOT / "artifacts" / "experiment_10" / "subdivision_grid"


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    root.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    root.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    root.add_argument(
        "--point-budgets",
        "--point-counts",
        dest="point_budgets",
        default=",".join(str(value) for value in DEFAULT_POINT_BUDGETS),
        help="comma-separated raw point-count budgets",
    )
    root.add_argument(
        "--subdivisions",
        "--atom-grid-points",
        dest="subdivisions",
        default=",".join(str(value) for value in DEFAULT_SUBDIVISIONS),
        help="comma-separated x-grid subdivision counts / direct-grid widths",
    )
    root.add_argument("--dense-resolution", "--dense-points", dest="dense_resolution", type=int, default=DEFAULT_DENSE_RESOLUTION)
    root.add_argument("--include-inactive", action="store_true")
    root.add_argument("--no-build-corpus", action="store_true")
    root.add_argument("--force-rebuild-corpus", action="store_true")
    return root


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    result = run_experiment10_grid_audit(
        metadata_path=args.metadata,
        corpus_dir=args.corpus_dir,
        output_dir=args.output_dir,
        point_budgets=parse_counts(args.point_budgets),
        subdivisions=parse_counts(args.subdivisions),
        dense_resolution=args.dense_resolution,
        active_only=not args.include_inactive,
        build_corpus_if_missing=not args.no_build_corpus,
        force_rebuild_corpus=args.force_rebuild_corpus,
        progress=lambda message: print(message, flush=True),
    )
    print(f"Wrote Experiment 10 grid-audit results to {result['output_dir']}", flush=True)
    print(f"summary={result['summary']}", flush=True)
    print(f"report={result['report']}", flush=True)


def run_experiment10_grid_audit(
    *,
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    metadata_path: Path = DEFAULT_METADATA,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    point_budgets: tuple[int, ...] = DEFAULT_POINT_BUDGETS,
    subdivisions: tuple[int, ...] = DEFAULT_SUBDIVISIONS,
    dense_resolution: int = DEFAULT_DENSE_RESOLUTION,
    active_only: bool = True,
    build_corpus_if_missing: bool = True,
    force_rebuild_corpus: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    corpus_dir = Path(corpus_dir)
    if force_rebuild_corpus or not (corpus_dir / "manifest.json").exists():
        if not build_corpus_if_missing and not force_rebuild_corpus:
            raise FileNotFoundError(f"missing processed LFO corpus: {corpus_dir}")
        if progress:
            progress(f"experiment10: building processed LFO corpus at {corpus_dir}")
        build_lfo_corpus(
            metadata_path=metadata_path,
            output_dir=corpus_dir,
            dense_resolution=dense_resolution,
            force=force_rebuild_corpus,
            progress=progress,
        )

    if progress:
        progress(f"experiment10: loading processed LFO shape corpus resolution={dense_resolution}")
    corpus = load_processed_shape_corpus(
        corpus_dir,
        resolution=dense_resolution,
        active_only=active_only,
        mmap=True,
    )
    if not corpus.shapes:
        raise ValueError("Experiment 10 requires at least one LFO shape")

    reference = np.asarray(corpus.curves, dtype=np.float32)
    weights = np.asarray(corpus.active_occurrence_count if active_only else corpus.occurrence_count, dtype=np.float64)
    point_lengths = np.asarray([len(shape.points) for shape in corpus.shapes], dtype=np.int32)
    point_rows = _point_budget_rows(point_lengths, weights, point_budgets=point_budgets)
    subdivision_rows = []
    direct_rows = []
    for index, subdivisions_count in enumerate(subdivisions, start=1):
        if progress:
            progress(f"experiment10: [{index}/{len(subdivisions)}] subdivisions={subdivisions_count} starting")
        subdivision_rows.append(
            _subdivision_row(
                corpus.shapes,
                weights,
                point_lengths,
                subdivisions_count=subdivisions_count,
            )
        )
        direct_rows.append(
            _direct_grid_row(
                corpus.shapes,
                reference,
                weights,
                width=subdivisions_count,
                dense_resolution=dense_resolution,
                progress=progress,
            )
        )
        if progress:
            progress(
                "experiment10: "
                f"[{index}/{len(subdivisions)}] subdivisions={subdivisions_count} done "
                f"boundary_exact={subdivision_rows[-1]['boundary_exact_rate_weighted']:.4f} "
                f"direct_p95={direct_rows[-1]['direct_grid_p95_rmse_weighted']:.8f}"
            )

    tables = {
        "point_budget": point_rows,
        "subdivision": subdivision_rows,
        "direct_grid": direct_rows,
        "factor3": _factor3_rows(subdivision_rows, direct_rows, comparisons=DEFAULT_FACTOR3_COMPARISONS),
        "summary": _summary_rows(subdivision_rows, direct_rows),
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "point_budget_summary.csv", tables["point_budget"])
    _write_csv(output_dir / "subdivision_summary.csv", tables["subdivision"])
    _write_csv(output_dir / "direct_grid_summary.csv", tables["direct_grid"])
    _write_csv(output_dir / "factor3_comparisons.csv", tables["factor3"])
    _write_csv(output_dir / "summary.csv", tables["summary"])
    (output_dir / "EXPERIMENT_10_SUBDIVISION_GRID_REPORT.md").write_text(_report(tables), encoding="utf-8")
    manifest = {
        "experiment_id": "experiment_10",
        "experiment_name": "subdivision_grid_and_direct_grid_audit",
        "corpus_dir": str(corpus_dir),
        "corpus_manifest": corpus.manifest,
        "point_budgets": list(point_budgets),
        "subdivisions": list(subdivisions),
        "factor3_comparisons": [list(pair) for pair in DEFAULT_FACTOR3_COMPARISONS],
        "dense_resolution": int(dense_resolution),
        "active_only": bool(active_only),
        "method": "point_count_coverage_plus_subdivision_boundary_coverage_plus_direct_grid_reproduction",
        "renderer_contract": "Raw shape references and x-boundary checks use Vital-ish power/smooth rendering; direct sampled grids store only y-values and decode with periodic linear interpolation.",
        "standalone_note": "Experiment 10 is intentionally outside the shared Era 2 model-runner CLI.",
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if progress:
        progress(f"experiment10: wrote summary/report to {output_dir}")
    return {
        "output_dir": str(output_dir),
        "summary": str(output_dir / "summary.csv"),
        "point_budget_summary": str(output_dir / "point_budget_summary.csv"),
        "subdivision_summary": str(output_dir / "subdivision_summary.csv"),
        "direct_grid_summary": str(output_dir / "direct_grid_summary.csv"),
        "factor3_comparisons": str(output_dir / "factor3_comparisons.csv"),
        "report": str(output_dir / "EXPERIMENT_10_SUBDIVISION_GRID_REPORT.md"),
        "manifest": manifest,
        "rows": tables["summary"],
    }


def _point_budget_rows(
    point_lengths: np.ndarray,
    weights: np.ndarray,
    *,
    point_budgets: tuple[int, ...],
) -> list[dict[str, Any]]:
    rows = []
    total_weight = float(np.sum(weights))
    for budget in point_budgets:
        within = point_lengths <= budget
        within_weight = float(np.sum(weights[within]))
        rows.append(
            {
                "point_budget": int(budget),
                "unique_shape_count": int(len(point_lengths)),
                "occurrence_weight_total": total_weight,
                "within_point_budget_unique": int(np.sum(within)),
                "within_point_budget_weighted": within_weight,
                "over_point_budget_unique": int(np.sum(~within)),
                "over_point_budget_weighted": float(np.sum(weights[~within])),
                "point_budget_coverage_unique": float(np.mean(within)) if len(within) else 0.0,
                "point_budget_coverage_weighted": within_weight / total_weight if total_weight else 0.0,
                "max_source_point_count": int(np.max(point_lengths)) if len(point_lengths) else 0,
            }
        )
    return rows


def _subdivision_row(
    shapes: tuple[LfoShape, ...],
    weights: np.ndarray,
    point_lengths: np.ndarray,
    *,
    subdivisions_count: int,
) -> dict[str, Any]:
    distances = []
    distance_weights = []
    exact_weights = []
    for shape, weight in zip(shapes, weights):
        interior = shape.points[:, 0]
        interior = interior[(interior > 0.0) & (interior < 1.0)]
        if not len(interior):
            continue
        nearest_index = np.rint(interior * subdivisions_count)
        nearest = np.clip(nearest_index / subdivisions_count, 0.0, 1.0)
        distance = np.abs(interior - nearest)
        distances.append(distance)
        distance_weights.append(np.full(len(distance), float(weight), dtype=np.float64))
        exact_weights.append(np.where(distance <= BOUNDARY_EXACT_TOLERANCE, float(weight), 0.0))
    if distances:
        all_distances = np.concatenate(distances)
        all_weights = np.concatenate(distance_weights)
        all_exact_weights = np.concatenate(exact_weights)
    else:
        all_distances = np.asarray([], dtype=np.float64)
        all_weights = np.asarray([], dtype=np.float64)
        all_exact_weights = np.asarray([], dtype=np.float64)
    total_boundary_weight = float(np.sum(all_weights))
    vital_eligible = point_lengths <= VITAL_MAX_POINTS
    vital_weight = float(np.sum(weights[vital_eligible]))
    return {
        "subdivisions": int(subdivisions_count),
        "inclusive_grid_slots": int(subdivisions_count + 1),
        "direct_grid_nodes": int(subdivisions_count),
        "divisible_by_3": bool(subdivisions_count % 3 == 0),
        "divisible_by_5": bool(subdivisions_count % 5 == 0),
        "fits_100_direct_grid_nodes": bool(subdivisions_count <= VITAL_MAX_POINTS),
        "fits_100_inclusive_slots": bool(subdivisions_count + 1 <= VITAL_MAX_POINTS),
        "vital_point_coverage_unique": float(np.mean(vital_eligible)) if len(vital_eligible) else 0.0,
        "vital_point_coverage_weighted": vital_weight / float(np.sum(weights)) if np.sum(weights) else 0.0,
        "boundary_occurrence_weight": total_boundary_weight,
        "boundary_exact_weighted": float(np.sum(all_exact_weights)),
        "boundary_exact_rate_weighted": float(np.sum(all_exact_weights) / total_boundary_weight) if total_boundary_weight else 1.0,
        "boundary_nearest_dx_mean_weighted": _weighted_mean(all_distances, all_weights),
        "boundary_nearest_dx_median_weighted": _weighted_quantile(all_distances, all_weights, 0.5),
        "boundary_nearest_dx_p95_weighted": _weighted_quantile(all_distances, all_weights, 0.95),
        "boundary_nearest_dx_p99_weighted": _weighted_quantile(all_distances, all_weights, 0.99),
        "boundary_nearest_dx_max": float(np.max(all_distances)) if len(all_distances) else 0.0,
    }


def _direct_grid_row(
    shapes: tuple[LfoShape, ...],
    reference: np.ndarray,
    weights: np.ndarray,
    *,
    width: int,
    dense_resolution: int,
    progress: Callable[[str], None] | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    grid_phase = np.arange(width, dtype=np.float64) / float(width)
    eval_phase = np.arange(dense_resolution, dtype=np.float64) / float(dense_resolution)
    reconstructed = np.empty_like(reference, dtype=np.float32)
    step = max(1, len(shapes) // 10)
    for index, shape in enumerate(shapes):
        if progress and (index == 0 or (index + 1) % step == 0 or index + 1 == len(shapes)):
            progress(f"experiment10: subdivisions={width} direct grid {_percent(index + 1, len(shapes))} shapes={index + 1}/{len(shapes)}")
        values = _sample_shape_at_phase(shape, grid_phase)
        reconstructed[index] = _interp_periodic_values(values, eval_phase)
    rmse = rmse_per_curve(reference, reconstructed)
    return {
        "subdivisions": int(width),
        "direct_grid_nodes": int(width),
        "direct_grid_elapsed_seconds": time.perf_counter() - started,
        "direct_grid_median_rmse_unique": float(np.median(rmse)),
        "direct_grid_p95_rmse_unique": float(np.quantile(rmse, 0.95)),
        "direct_grid_p99_rmse_unique": float(np.quantile(rmse, 0.99)),
        "direct_grid_max_rmse_unique": float(np.max(rmse)),
        "direct_grid_mean_rmse_weighted": _weighted_mean(rmse, weights),
        "direct_grid_median_rmse_weighted": _weighted_quantile(rmse, weights, 0.5),
        "direct_grid_p95_rmse_weighted": _weighted_quantile(rmse, weights, 0.95),
        "direct_grid_p99_rmse_weighted": _weighted_quantile(rmse, weights, 0.99),
    }


def _sample_shape_at_phase(shape: LfoShape, phase: np.ndarray) -> np.ndarray:
    phase = np.asarray(phase, dtype=np.float64)
    x = shape.points[:, 0]
    y = shape.points[:, 1]
    right = np.searchsorted(x, phase, side="left")
    right = np.clip(right, 1, len(x) - 1)
    left = right - 1
    width = x[right] - x[left]
    local = np.divide(phase - x[left], width, out=np.ones_like(phase), where=width > 0.0)
    local = np.clip(local, 0.0, 1.0)
    if shape.smooth:
        local = local * local * (3.0 - 2.0 * local)
    local = power_scale(local, shape.powers[left])
    return (y[left] + local * (y[right] - y[left])).astype(np.float32)


def _interp_periodic_values(values: np.ndarray, phase: np.ndarray) -> np.ndarray:
    width = len(values)
    position = (np.asarray(phase, dtype=np.float64) % 1.0) * float(width)
    left = np.floor(position).astype(np.int64) % width
    right = (left + 1) % width
    frac = position - np.floor(position)
    return (values[left] * (1.0 - frac) + values[right] * frac).astype(np.float32)


def _factor3_rows(
    subdivision_rows: list[dict[str, Any]],
    direct_rows: list[dict[str, Any]],
    *,
    comparisons: tuple[tuple[int, int], ...],
) -> list[dict[str, Any]]:
    subdivision_by_count = {int(row["subdivisions"]): row for row in subdivision_rows}
    direct_by_count = {int(row["subdivisions"]): row for row in direct_rows}
    rows = []
    for factor3, higher in comparisons:
        if factor3 not in subdivision_by_count or higher not in subdivision_by_count:
            continue
        a_sub = subdivision_by_count[factor3]
        b_sub = subdivision_by_count[higher]
        a_direct = direct_by_count[factor3]
        b_direct = direct_by_count[higher]
        rows.append(
            {
                "factor3_subdivisions": int(factor3),
                "higher_nonfactor3_subdivisions": int(higher),
                "extra_direct_grid_nodes_for_higher": int(higher - factor3),
                "factor3_boundary_exact_rate_weighted": a_sub["boundary_exact_rate_weighted"],
                "higher_boundary_exact_rate_weighted": b_sub["boundary_exact_rate_weighted"],
                "boundary_exact_rate_delta_factor3_minus_higher": float(a_sub["boundary_exact_rate_weighted"] - b_sub["boundary_exact_rate_weighted"]),
                "factor3_boundary_p95_dx_weighted": a_sub["boundary_nearest_dx_p95_weighted"],
                "higher_boundary_p95_dx_weighted": b_sub["boundary_nearest_dx_p95_weighted"],
                "boundary_p95_dx_delta_factor3_minus_higher": float(a_sub["boundary_nearest_dx_p95_weighted"] - b_sub["boundary_nearest_dx_p95_weighted"]),
                "factor3_direct_p95_rmse_weighted": a_direct["direct_grid_p95_rmse_weighted"],
                "higher_direct_p95_rmse_weighted": b_direct["direct_grid_p95_rmse_weighted"],
                "direct_p95_rmse_delta_factor3_minus_higher": float(a_direct["direct_grid_p95_rmse_weighted"] - b_direct["direct_grid_p95_rmse_weighted"]),
                "factor3_direct_p95_beats_or_matches_higher": bool(a_direct["direct_grid_p95_rmse_weighted"] <= b_direct["direct_grid_p95_rmse_weighted"]),
            }
        )
    return rows


def _summary_rows(
    subdivision_rows: list[dict[str, Any]],
    direct_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    direct_by_count = {int(row["subdivisions"]): row for row in direct_rows}
    return [{**row, **direct_by_count[int(row["subdivisions"])]} for row in subdivision_rows]


def parse_counts(value: str) -> tuple[int, ...]:
    counts = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not counts:
        raise ValueError("at least one count is required")
    return counts


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    if not len(values):
        return 0.0
    total = float(np.sum(weights))
    return float(np.sum(values * weights) / total) if total else 0.0


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    if not len(values):
        return 0.0
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    total = float(np.sum(sorted_weights))
    if total <= 0.0:
        return float(np.quantile(values, quantile))
    cutoff = quantile * total
    cumulative = np.cumsum(sorted_weights)
    index = int(np.searchsorted(cumulative, cutoff, side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _report(tables: dict[str, list[dict[str, Any]]]) -> str:
    lines = [
        "# Experiment 10: Subdivision And Direct Grid Audit",
        "",
        "Experiment 10 is a standalone corpus/grid audit, not an Era 2 model-runner experiment. The point-count table is only corpus accounting. The main tests are subdivision boundary coverage and Era-1-style direct sampled-grid reproduction.",
        "",
        "Renderer contract:",
        "",
        "- Raw LFO references are rendered from the original points, powers, and smooth flag.",
        "- Boundary coverage measures how close original x positions are to a subdivision grid.",
        "- Direct grids sample the true raw curve at `i / W`, store only y-values, then decode by periodic linear interpolation.",
        "",
        "## Factor-of-3 comparisons",
        "",
        "| factor-3 subdivisions | higher non-factor-3 | direct p95 delta | factor-3 beats/matches higher | boundary exact delta | boundary p95 dx delta |",
        "| ---: | ---: | ---: | :---: | ---: | ---: |",
    ]
    for row in tables["factor3"]:
        lines.append(
            "| {factor3_subdivisions} | {higher_nonfactor3_subdivisions} | {direct_p95_rmse_delta_factor3_minus_higher:.8f} | {factor3_direct_p95_beats_or_matches_higher} | {boundary_exact_rate_delta_factor3_minus_higher:.8f} | {boundary_p95_dx_delta_factor3_minus_higher:.8f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Point-count coverage",
            "",
            "| point budget | weighted coverage | over-budget unique |",
            "| ---: | ---: | ---: |",
        ]
    )
    for row in tables["point_budget"]:
        lines.append(
            "| {point_budget} | {point_budget_coverage_weighted:.6f} | {over_point_budget_unique} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "Interpretation notes:",
            "",
            "- A negative direct p95 delta means the factor-of-3 grid beat the higher non-factor-of-3 grid.",
            "- The direct-grid comparison is not a model prediction head claim. It is a dense y-node reproduction baseline.",
            "- Topology remains analysis-only in the processed corpus.",
        ]
    )
    return "\n".join(lines) + "\n"


def _percent(done: int, total: int) -> str:
    if total <= 0:
        return "0.0%"
    return f"{100.0 * min(done, total) / total:.1f}%"


if __name__ == "__main__":
    main()

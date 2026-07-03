#!/usr/bin/env python
"""Standalone Experiment 10 control-point x-grid audit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np

from lfo_era2.dataset import LfoShape
from lfo_era2.processed_corpus import (
    DEFAULT_CORPUS_DIR,
    DEFAULT_METADATA,
    build_lfo_corpus,
    load_processed_shape_corpus,
)


DEFAULT_GRID_POINT_COUNTS = (
    24,
    25,
    26,
    32,
    33,
    36,
    37,
    38,
    40,
    41,
    48,
    49,
    50,
    60,
    61,
    62,
    64,
    65,
    72,
    73,
    74,
    80,
    81,
    96,
    97,
    98,
    100,
)
DEFAULT_FACTOR3_GRID_POINT_COMPARISONS = (
    (25, 26),
    (25, 33),
    (37, 38),
    (37, 41),
    (49, 50),
    (49, 65),
    (61, 62),
    (61, 65),
    (73, 74),
    (73, 81),
    (97, 98),
)
VITAL_MAX_POINTS = 100
EXACT_TOLERANCE = 1e-6
CONTROL_POINT_X_PASS_TOLERANCE = 0.01
ERA2_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ERA2_ROOT / "artifacts" / "experiment_10" / "control_point_x_grid"


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    root.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    root.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    root.add_argument(
        "--grid-point-counts",
        dest="grid_point_counts",
        default=",".join(str(value) for value in DEFAULT_GRID_POINT_COUNTS),
        help="comma-separated inclusive x-grid point counts",
    )
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
        grid_point_counts=parse_counts(args.grid_point_counts),
        active_only=not args.include_inactive,
        build_corpus_if_missing=not args.no_build_corpus,
        force_rebuild_corpus=args.force_rebuild_corpus,
        progress=lambda message: print(message, flush=True),
    )
    print(f"Wrote Experiment 10 x-grid audit results to {result['output_dir']}", flush=True)
    print(f"summary={result['summary']}", flush=True)
    print(f"report={result['report']}", flush=True)


def run_experiment10_grid_audit(
    *,
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    metadata_path: Path = DEFAULT_METADATA,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    grid_point_counts: tuple[int, ...] = DEFAULT_GRID_POINT_COUNTS,
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
            force=force_rebuild_corpus,
            progress=progress,
        )

    if progress:
        progress("experiment10: loading processed LFO shape corpus")
    corpus = load_processed_shape_corpus(corpus_dir, active_only=active_only, mmap=True)
    if not corpus.shapes:
        raise ValueError("Experiment 10 requires at least one LFO shape")

    weights = np.asarray(corpus.active_occurrence_count if active_only else corpus.occurrence_count, dtype=np.float64)
    source_point_counts = np.asarray([len(shape.points) for shape in corpus.shapes], dtype=np.int32)
    point_frequency_rows = _point_count_frequency_rows(source_point_counts, weights)
    control_rows = []
    learned_grid_records = []
    for index, grid_point_count in enumerate(grid_point_counts, start=1):
        if progress:
            progress(f"experiment10: [{index}/{len(grid_point_counts)}] grid_point_count={grid_point_count} starting")
        uniform_grid = _uniform_grid(grid_point_count)
        rows_for_count = [
            _control_point_x_row(
                corpus.shapes,
                weights,
                grid_point_count=grid_point_count,
                grid_kind="uniform",
                grid_learning_weighting="none",
                grid_points=uniform_grid,
            )
        ]
        for weighting in ("deduplicated", "occurrence_weighted"):
            learned_grid = _global_quantile_grid(
                corpus.shapes,
                weights,
                grid_point_count=grid_point_count,
                weighting=weighting,
            )
            learned_grid_records.append(
                {
                    "grid_kind": "global_quantile",
                    "grid_learning_weighting": weighting,
                    "grid_point_count": int(grid_point_count),
                    "grid_points": [float(value) for value in learned_grid],
                }
            )
            rows_for_count.append(
                _control_point_x_row(
                    corpus.shapes,
                    weights,
                    grid_point_count=grid_point_count,
                    grid_kind="global_quantile",
                    grid_learning_weighting=weighting,
                    grid_points=learned_grid,
                )
            )
        control_rows.extend(rows_for_count)
        uniform_row = rows_for_count[0]
        if progress:
            progress(
                "experiment10: "
                f"[{index}/{len(grid_point_counts)}] grid_point_count={grid_point_count} "
                f"subdivision_count={uniform_row['subdivision_count']} "
                f"uniform_interior_p95={uniform_row['control_point_x_p95_abs_error_interior_occurrence_weighted']:.8f}"
            )
    factor3_rows = _factor3_grid_point_rows(control_rows, comparisons=DEFAULT_FACTOR3_GRID_POINT_COMPARISONS)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "point_count_frequency.csv", point_frequency_rows)
    _write_csv(output_dir / "control_point_x_summary.csv", control_rows)
    _write_csv(output_dir / "factor3_grid_point_comparisons.csv", factor3_rows)
    _write_csv(output_dir / "summary.csv", control_rows)
    (output_dir / "global_nonuniform_grids.json").write_text(
        json.dumps(learned_grid_records, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plot_paths = _write_plots(
        output_dir / "plots",
        point_frequency_rows=point_frequency_rows,
        control_rows=control_rows,
    )
    report = _report(
        point_frequency_rows=point_frequency_rows,
        control_rows=control_rows,
        factor3_rows=factor3_rows,
        plot_paths=plot_paths,
    )
    report_path = output_dir / "EXPERIMENT_10_CONTROL_POINT_X_GRID_REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    manifest = {
        "experiment_id": "experiment_10",
        "experiment_name": "control_point_x_grid_audit",
        "corpus_dir": str(corpus_dir),
        "corpus_manifest": corpus.manifest,
        "grid_point_counts": list(grid_point_counts),
        "factor3_grid_point_comparisons": [list(pair) for pair in DEFAULT_FACTOR3_GRID_POINT_COMPARISONS],
        "active_only": bool(active_only),
        "method": "point_count_frequency_plus_control_point_x_grid_error",
        "grid_count_contract": "Experiment 10 varies grid_point_count. subdivision_count is inferred as grid_point_count - 1. W is reserved for residual-layer atom choices.",
        "control_point_x_contract": "Control-point placement is evaluated on x only. For each true ordered control point, the predicted x is the nearest point in the fixed grid; y is not scored and no curve is rendered.",
        "pass_rate_contract": "lfo_all_points_within_0p01_* is the fraction of LFOs whose maximum control-point x error is <= 0.01.",
        "nonuniform_grid_contract": "global_quantile grids are fixed offline-learned decoder grids. They do not require the deployed model to predict grid locations.",
        "report_plots": plot_paths,
        "standalone_note": "Experiment 10 is intentionally outside the shared Era 2 model-runner CLI.",
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if progress:
        progress(f"experiment10: wrote summary/report to {output_dir}")
    return {
        "output_dir": str(output_dir),
        "summary": str(output_dir / "summary.csv"),
        "point_count_frequency": str(output_dir / "point_count_frequency.csv"),
        "control_point_x_summary": str(output_dir / "control_point_x_summary.csv"),
        "factor3_grid_point_comparisons": str(output_dir / "factor3_grid_point_comparisons.csv"),
        "global_nonuniform_grids": str(output_dir / "global_nonuniform_grids.json"),
        "plots": {key: str(output_dir / relative_path) for key, relative_path in plot_paths.items()},
        "report": str(report_path),
        "manifest": manifest,
        "rows": control_rows,
    }


def _point_count_frequency_rows(point_counts: np.ndarray, weights: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    unique_total = int(len(point_counts))
    occurrence_total = float(np.sum(weights))
    cumulative_unique = 0
    cumulative_occurrence = 0.0
    for source_point_count in sorted(int(value) for value in np.unique(point_counts)):
        mask = point_counts == source_point_count
        deduplicated_count = int(np.sum(mask))
        occurrence_count = float(np.sum(weights[mask]))
        cumulative_unique += deduplicated_count
        cumulative_occurrence += occurrence_count
        rows.append(
            {
                "source_point_count": source_point_count,
                "deduplicated_lfo_count": deduplicated_count,
                "deduplicated_lfo_fraction": deduplicated_count / unique_total if unique_total else 0.0,
                "deduplicated_lfo_cumulative_fraction": cumulative_unique / unique_total if unique_total else 0.0,
                "lfo_corpus_occurrence_count": occurrence_count,
                "lfo_corpus_occurrence_fraction": occurrence_count / occurrence_total if occurrence_total else 0.0,
                "lfo_corpus_cumulative_fraction": cumulative_occurrence / occurrence_total if occurrence_total else 0.0,
            }
        )
    return rows


def _uniform_grid(grid_point_count: int) -> np.ndarray:
    if grid_point_count < 2:
        raise ValueError("grid_point_count must be at least 2")
    return np.linspace(0.0, 1.0, int(grid_point_count), dtype=np.float64)


def _global_quantile_grid(
    shapes: tuple[LfoShape, ...],
    weights: np.ndarray,
    *,
    grid_point_count: int,
    weighting: str,
) -> np.ndarray:
    if grid_point_count < 2:
        raise ValueError("grid_point_count must be at least 2")
    if grid_point_count == 2:
        return np.asarray([0.0, 1.0], dtype=np.float64)
    values = []
    value_weights = []
    for shape, occurrence_weight in zip(shapes, weights):
        interior = shape.points[:, 0]
        interior = interior[(interior > 0.0) & (interior < 1.0)]
        if not len(interior):
            continue
        values.append(interior)
        if weighting == "deduplicated":
            value_weights.append(np.ones(len(interior), dtype=np.float64))
        elif weighting == "occurrence_weighted":
            value_weights.append(np.full(len(interior), float(occurrence_weight), dtype=np.float64))
        else:
            raise ValueError(f"unsupported nonuniform grid weighting: {weighting}")
    if not values:
        return _uniform_grid(grid_point_count)
    all_values = np.concatenate(values)
    all_weights = np.concatenate(value_weights)
    interior_count = grid_point_count - 2
    quantiles = np.arange(1, interior_count + 1, dtype=np.float64) / float(interior_count + 1)
    grid = np.concatenate(
        [
            np.asarray([0.0], dtype=np.float64),
            np.asarray([_weighted_quantile(all_values, all_weights, float(q)) for q in quantiles], dtype=np.float64),
            np.asarray([1.0], dtype=np.float64),
        ]
    )
    return _expand_to_unique_grid_points(grid, grid_point_count)


def _expand_to_unique_grid_points(grid: np.ndarray, grid_point_count: int) -> np.ndarray:
    rounded = np.round(np.clip(np.asarray(grid, dtype=np.float64), 0.0, 1.0), 12)
    unique = sorted(set(float(value) for value in rounded) | {0.0, 1.0})
    while len(unique) < grid_point_count:
        gaps = [(unique[index + 1] - unique[index], index) for index in range(len(unique) - 1)]
        _, gap_index = max(gaps, key=lambda item: item[0])
        midpoint = 0.5 * (unique[gap_index] + unique[gap_index + 1])
        unique.insert(gap_index + 1, midpoint)
    if len(unique) > grid_point_count:
        interior = [value for value in unique if value not in {0.0, 1.0}]
        keep = max(0, grid_point_count - 2)
        quantile_indices = np.linspace(0, max(0, len(interior) - 1), keep).round().astype(int) if keep else []
        unique = [0.0] + [interior[int(index)] for index in quantile_indices] + [1.0]
    return np.asarray(sorted(unique), dtype=np.float64)


def _nearest_grid_distances(values: np.ndarray, grid_points: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    grid_points = np.asarray(grid_points, dtype=np.float64)
    right = np.searchsorted(grid_points, values, side="left")
    right = np.clip(right, 0, len(grid_points) - 1)
    left = np.clip(right - 1, 0, len(grid_points) - 1)
    return np.minimum(np.abs(values - grid_points[left]), np.abs(values - grid_points[right]))


def _control_point_x_row(
    shapes: tuple[LfoShape, ...],
    weights: np.ndarray,
    *,
    grid_point_count: int,
    grid_kind: str,
    grid_learning_weighting: str,
    grid_points: np.ndarray,
) -> dict[str, Any]:
    if grid_point_count < 2:
        raise ValueError("grid_point_count must be at least 2")
    grid_points = np.asarray(grid_points, dtype=np.float64)
    if len(grid_points) != grid_point_count:
        raise ValueError("grid_points length must match grid_point_count")
    subdivision_count = int(grid_point_count) - 1 if grid_kind == "uniform" else ""
    all_distances = []
    all_occurrence_weights = []
    all_deduplicated_weights = []
    interior_distances = []
    interior_occurrence_weights = []
    interior_deduplicated_weights = []
    lfo_all_pass = []
    lfo_interior_pass = []
    for shape, weight in zip(shapes, weights):
        x = shape.points[:, 0]
        distance = _nearest_grid_distances(x, grid_points)
        occurrence_point_weights = np.full(len(distance), float(weight), dtype=np.float64)
        deduplicated_point_weights = np.ones(len(distance), dtype=np.float64)
        all_distances.append(distance)
        all_occurrence_weights.append(occurrence_point_weights)
        all_deduplicated_weights.append(deduplicated_point_weights)
        lfo_all_pass.append(bool(np.max(distance) <= CONTROL_POINT_X_PASS_TOLERANCE))

        interior = (x > 0.0) & (x < 1.0)
        if np.any(interior):
            interior_distances.append(distance[interior])
            interior_occurrence_weights.append(occurrence_point_weights[interior])
            interior_deduplicated_weights.append(deduplicated_point_weights[interior])
            lfo_interior_pass.append(bool(np.max(distance[interior]) <= CONTROL_POINT_X_PASS_TOLERANCE))
        else:
            lfo_interior_pass.append(True)

    all_dx = np.concatenate(all_distances) if all_distances else np.asarray([], dtype=np.float64)
    all_occurrence_w = np.concatenate(all_occurrence_weights) if all_occurrence_weights else np.asarray([], dtype=np.float64)
    all_deduplicated_w = np.concatenate(all_deduplicated_weights) if all_deduplicated_weights else np.asarray([], dtype=np.float64)
    interior_dx = np.concatenate(interior_distances) if interior_distances else np.asarray([], dtype=np.float64)
    interior_occurrence_w = np.concatenate(interior_occurrence_weights) if interior_occurrence_weights else np.asarray([], dtype=np.float64)
    interior_deduplicated_w = np.concatenate(interior_deduplicated_weights) if interior_deduplicated_weights else np.asarray([], dtype=np.float64)
    lfo_all_pass_array = np.asarray(lfo_all_pass, dtype=np.bool_)
    lfo_interior_pass_array = np.asarray(lfo_interior_pass, dtype=np.bool_)
    return {
        "grid_kind": grid_kind,
        "grid_learning_weighting": grid_learning_weighting,
        "grid_point_count": int(grid_point_count),
        "subdivision_count": subdivision_count,
        "subdivision_divisible_by_2": bool(subdivision_count % 2 == 0) if grid_kind == "uniform" else "",
        "subdivision_divisible_by_3": bool(subdivision_count % 3 == 0) if grid_kind == "uniform" else "",
        "subdivision_divisible_by_5": bool(subdivision_count % 5 == 0) if grid_kind == "uniform" else "",
        "fits_vital_100_points": bool(grid_point_count <= VITAL_MAX_POINTS),
        "unique_grid_point_count": int(len(np.unique(grid_points))),
        "control_point_x_grid_step": float(1.0 / subdivision_count) if grid_kind == "uniform" else "",
        "control_point_x_max_rounding_error": float(0.5 / subdivision_count) if grid_kind == "uniform" else "",
        "control_point_x_occurrence_weight_all": float(np.sum(all_occurrence_w)),
        "control_point_x_exact_rate_all_occurrence_weighted": _exact_rate(all_dx, all_occurrence_w),
        "control_point_x_mean_abs_error_all_occurrence_weighted": _weighted_mean(all_dx, all_occurrence_w),
        "control_point_x_median_abs_error_all_occurrence_weighted": _weighted_quantile(all_dx, all_occurrence_w, 0.5),
        "control_point_x_p95_abs_error_all_occurrence_weighted": _weighted_quantile(all_dx, all_occurrence_w, 0.95),
        "control_point_x_p99_abs_error_all_occurrence_weighted": _weighted_quantile(all_dx, all_occurrence_w, 0.99),
        "control_point_x_exact_rate_all_deduplicated": _exact_rate(all_dx, all_deduplicated_w),
        "control_point_x_mean_abs_error_all_deduplicated": _weighted_mean(all_dx, all_deduplicated_w),
        "control_point_x_median_abs_error_all_deduplicated": _weighted_quantile(all_dx, all_deduplicated_w, 0.5),
        "control_point_x_p95_abs_error_all_deduplicated": _weighted_quantile(all_dx, all_deduplicated_w, 0.95),
        "control_point_x_p99_abs_error_all_deduplicated": _weighted_quantile(all_dx, all_deduplicated_w, 0.99),
        "control_point_x_max_abs_error_all": float(np.max(all_dx)) if len(all_dx) else 0.0,
        "control_point_x_occurrence_weight_interior": float(np.sum(interior_occurrence_w)),
        "control_point_x_exact_rate_interior_occurrence_weighted": _exact_rate(interior_dx, interior_occurrence_w),
        "control_point_x_mean_abs_error_interior_occurrence_weighted": _weighted_mean(interior_dx, interior_occurrence_w),
        "control_point_x_median_abs_error_interior_occurrence_weighted": _weighted_quantile(interior_dx, interior_occurrence_w, 0.5),
        "control_point_x_p95_abs_error_interior_occurrence_weighted": _weighted_quantile(interior_dx, interior_occurrence_w, 0.95),
        "control_point_x_p99_abs_error_interior_occurrence_weighted": _weighted_quantile(interior_dx, interior_occurrence_w, 0.99),
        "control_point_x_exact_rate_interior_deduplicated": _exact_rate(interior_dx, interior_deduplicated_w),
        "control_point_x_mean_abs_error_interior_deduplicated": _weighted_mean(interior_dx, interior_deduplicated_w),
        "control_point_x_median_abs_error_interior_deduplicated": _weighted_quantile(interior_dx, interior_deduplicated_w, 0.5),
        "control_point_x_p95_abs_error_interior_deduplicated": _weighted_quantile(interior_dx, interior_deduplicated_w, 0.95),
        "control_point_x_p99_abs_error_interior_deduplicated": _weighted_quantile(interior_dx, interior_deduplicated_w, 0.99),
        "control_point_x_max_abs_error_interior": float(np.max(interior_dx)) if len(interior_dx) else 0.0,
        "lfo_all_points_within_0p01_deduplicated_fraction": float(np.mean(lfo_all_pass_array)) if len(lfo_all_pass_array) else 0.0,
        "lfo_all_points_within_0p01_occurrence_fraction": _weighted_bool_fraction(lfo_all_pass_array, weights),
        "lfo_interior_points_within_0p01_deduplicated_fraction": float(np.mean(lfo_interior_pass_array)) if len(lfo_interior_pass_array) else 0.0,
        "lfo_interior_points_within_0p01_occurrence_fraction": _weighted_bool_fraction(lfo_interior_pass_array, weights),
    }


def _factor3_grid_point_rows(
    control_rows: list[dict[str, Any]],
    *,
    comparisons: tuple[tuple[int, int], ...],
) -> list[dict[str, Any]]:
    rows_by_grid_point_count = {
        int(row["grid_point_count"]): row
        for row in control_rows
        if row["grid_kind"] == "uniform"
    }
    rows = []
    for factor3_grid_point_count, higher_grid_point_count in comparisons:
        if factor3_grid_point_count not in rows_by_grid_point_count or higher_grid_point_count not in rows_by_grid_point_count:
            continue
        factor3 = rows_by_grid_point_count[factor3_grid_point_count]
        higher = rows_by_grid_point_count[higher_grid_point_count]
        rows.append(
            {
                "factor3_grid_point_count": int(factor3_grid_point_count),
                "factor3_subdivision_count": int(factor3["subdivision_count"]),
                "higher_nonfactor3_grid_point_count": int(higher_grid_point_count),
                "higher_nonfactor3_subdivision_count": int(higher["subdivision_count"]),
                "extra_grid_points_for_higher": int(higher_grid_point_count - factor3_grid_point_count),
                "factor3_interior_p95_abs_error": factor3["control_point_x_p95_abs_error_interior_occurrence_weighted"],
                "higher_interior_p95_abs_error": higher["control_point_x_p95_abs_error_interior_occurrence_weighted"],
                "interior_p95_abs_error_delta_factor3_minus_higher": float(
                    factor3["control_point_x_p95_abs_error_interior_occurrence_weighted"]
                    - higher["control_point_x_p95_abs_error_interior_occurrence_weighted"]
                ),
                "factor3_beats_or_matches_higher_on_interior_p95": bool(
                    factor3["control_point_x_p95_abs_error_interior_occurrence_weighted"]
                    <= higher["control_point_x_p95_abs_error_interior_occurrence_weighted"]
                ),
                "factor3_interior_mean_abs_error": factor3["control_point_x_mean_abs_error_interior_occurrence_weighted"],
                "higher_interior_mean_abs_error": higher["control_point_x_mean_abs_error_interior_occurrence_weighted"],
                "interior_mean_abs_error_delta_factor3_minus_higher": float(
                    factor3["control_point_x_mean_abs_error_interior_occurrence_weighted"]
                    - higher["control_point_x_mean_abs_error_interior_occurrence_weighted"]
                ),
            }
        )
    return rows


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


def _exact_rate(values: np.ndarray, weights: np.ndarray) -> float:
    total = float(np.sum(weights))
    if total <= 0.0:
        return 1.0
    return float(np.sum(weights[values <= EXACT_TOLERANCE]) / total)


def _weighted_bool_fraction(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.bool_)
    weights = np.asarray(weights, dtype=np.float64)
    total = float(np.sum(weights))
    if total <= 0.0 or not len(values):
        return 0.0
    return float(np.sum(weights[values]) / total)


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


def _write_plots(
    plot_dir: Path,
    *,
    point_frequency_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "point_count_frequency": plot_dir / "experiment10_point_count_frequency.png",
        "x_error_p95": plot_dir / "experiment10_control_point_x_p95.png",
        "lfo_pass_rate_0p01": plot_dir / "experiment10_lfo_pass_rate_0p01.png",
        "nonuniform_delta": plot_dir / "experiment10_nonuniform_delta.png",
    }
    _plot_point_count_frequency(plt, paths["point_count_frequency"], point_frequency_rows)
    _plot_control_metric(
        plt,
        paths["x_error_p95"],
        control_rows,
        metric="control_point_x_p95_abs_error_interior_occurrence_weighted",
        ylabel="Interior P95 abs x error",
        title="Control-point x placement error by grid type",
    )
    _plot_control_metric(
        plt,
        paths["lfo_pass_rate_0p01"],
        control_rows,
        metric="lfo_all_points_within_0p01_occurrence_fraction",
        ylabel="Occurrence-weighted LFO fraction",
        title="LFOs with every control point within 0.01 of grid",
        y_limits=(0.0, 1.02),
    )
    _plot_nonuniform_delta(plt, paths["nonuniform_delta"], control_rows)
    return {key: path.relative_to(plot_dir.parent).as_posix() for key, path in paths.items()}


def _plot_point_count_frequency(plt: Any, path: Path, rows: list[dict[str, Any]]) -> None:
    x = np.asarray([int(row["source_point_count"]) for row in rows], dtype=np.float64)
    dedup = np.asarray([float(row["deduplicated_lfo_fraction"]) for row in rows], dtype=np.float64)
    occurrence = np.asarray([float(row["lfo_corpus_occurrence_fraction"]) for row in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(11.5, 5.8), constrained_layout=True)
    width = 0.42
    ax.bar(x - width / 2.0, dedup, width=width, label="Deduplicated LFO corpus", color="#4C78A8")
    ax.bar(x + width / 2.0, occurrence, width=width, label="Occurrence-weighted LFO corpus", color="#F58518")
    ax.set_title("Source control-point count frequency")
    ax.set_xlabel("Source point count")
    ax.set_ylabel("Fraction of corpus")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_control_metric(
    plt: Any,
    path: Path,
    rows: list[dict[str, Any]],
    *,
    metric: str,
    ylabel: str,
    title: str,
    y_limits: tuple[float, float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.8), constrained_layout=True)
    for label, group in _group_control_rows(rows).items():
        ordered = sorted(group, key=lambda row: int(row["grid_point_count"]))
        x = [int(row["grid_point_count"]) for row in ordered]
        y = [float(row[metric]) for row in ordered]
        ax.plot(x, y, marker="o", linewidth=2.0, markersize=4.5, label=label)
    ax.set_title(title)
    ax.set_xlabel("Grid point count")
    ax.set_ylabel(ylabel)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_nonuniform_delta(plt: Any, path: Path, rows: list[dict[str, Any]]) -> None:
    uniform_by_points = {
        int(row["grid_point_count"]): row
        for row in rows
        if row["grid_kind"] == "uniform"
    }
    fig, ax = plt.subplots(figsize=(11.5, 5.8), constrained_layout=True)
    for weighting in ("deduplicated", "occurrence_weighted"):
        ordered = sorted(
            [
                row
                for row in rows
                if row["grid_kind"] == "global_quantile" and row["grid_learning_weighting"] == weighting
            ],
            key=lambda row: int(row["grid_point_count"]),
        )
        x = []
        y = []
        for row in ordered:
            grid_point_count = int(row["grid_point_count"])
            baseline = uniform_by_points.get(grid_point_count)
            if baseline is None:
                continue
            x.append(grid_point_count)
            y.append(
                float(row["control_point_x_p95_abs_error_interior_occurrence_weighted"])
                - float(baseline["control_point_x_p95_abs_error_interior_occurrence_weighted"])
            )
        ax.plot(x, y, marker="o", linewidth=2.0, markersize=4.5, label=f"global quantile ({weighting})")
    ax.axhline(0.0, color="#222222", linewidth=1.0)
    ax.set_title("Global non-uniform grid P95 delta vs uniform")
    ax.set_xlabel("Grid point count")
    ax.set_ylabel("Interior P95 abs x error delta")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _group_control_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        label = _grid_label(row)
        groups.setdefault(label, []).append(row)
    return groups


def _grid_label(row: dict[str, Any]) -> str:
    if row["grid_kind"] == "uniform":
        return "uniform"
    if row["grid_kind"] == "global_quantile":
        return f"global quantile ({row['grid_learning_weighting']})"
    return f"{row['grid_kind']} ({row['grid_learning_weighting']})"


def _report(
    *,
    point_frequency_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
    factor3_rows: list[dict[str, Any]],
    plot_paths: dict[str, str],
) -> str:
    lines = [
        "# Experiment 10: Control-Point X Grid Audit",
        "",
        "Experiment 10 is a standalone corpus/grid audit, not an Era 2 model-runner experiment. It asks how many source control points real LFOs use, then tests how well inclusive x-grid point counts place those ordered control-point x positions.",
        "",
        "Naming contract:",
        "",
        "- `W` is reserved for residual-layer atom choices in Era 2 model experiments.",
        "- Experiment 10 varies `grid_point_count`.",
        "- `subdivision_count = grid_point_count - 1`.",
        "- Factor language applies to the inferred `subdivision_count`, not to `grid_point_count`.",
        "- Example: `grid_point_count = 97` means `subdivision_count = 96`, which is divisible by 2 and 3.",
        "",
        "Control-point x contract:",
        "",
        "- For each true ordered control point, predicted x is the nearest point in the fixed grid.",
        "- For `uniform`, grid points are `k / subdivision_count`.",
        "- For `global_quantile`, grid points are fixed offline-learned non-uniform positions.",
        "- Y is ignored, and no line, Bezier, power curve, or other segment is rendered.",
        "- Repeated grid points are allowed because discontinuous LFOs can contain repeated x positions.",
        "- `lfo_all_points_within_0p01_*` reports the fraction of LFOs whose maximum x error is at most 0.01.",
        "",
        "Global non-uniform grids:",
        "",
        "- `global_quantile` grids are learned once offline from corpus control-point x positions.",
        "- The deployed model would still predict a grid slot; it would not predict the grid positions.",
        "- Both deduplicated and occurrence-weighted learned grids are reported.",
        "",
        "## Analytics Read",
        "",
        "- Point-count frequency shows how often real LFOs use each raw control-point count.",
        "- P95 x-error shows the tail cost of grid placement among interior control points.",
        "- The `<=0.01` plot is the easiest operational read: it asks whether every control point in an LFO is close enough to the grid.",
        "- Non-uniform deltas compare fixed learned grids against uniform grids at the same `grid_point_count`; negative is better.",
        "",
        "## Source Point-Count Frequency",
        "",
        f"![Source point-count frequency]({plot_paths['point_count_frequency']})",
        "",
        "| source point count | deduplicated LFOs | dedup fraction | LFO corpus occurrences | corpus fraction |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in point_frequency_rows:
        lines.append(
            "| {source_point_count} | {deduplicated_lfo_count} | {deduplicated_lfo_fraction:.6f} | {lfo_corpus_occurrence_count:.0f} | {lfo_corpus_occurrence_fraction:.6f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Control-Point X Placement",
            "",
            f"![Interior P95 x error by grid type]({plot_paths['x_error_p95']})",
            "",
            f"![LFO pass rate at 0.01 tolerance]({plot_paths['lfo_pass_rate_0p01']})",
            "",
            f"![Non-uniform grid delta versus uniform]({plot_paths['nonuniform_delta']})",
            "",
            "| grid kind | learning weight | grid point count | subdivisions | div by 3 | LFOs <=0.01 dedup | LFOs <=0.01 corpus | interior mean x error | interior p95 x error |",
            "| --- | --- | ---: | ---: | :---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in control_rows:
        lines.append(
            "| {grid_kind} | {grid_learning_weighting} | {grid_point_count} | {subdivision_count} | {subdivision_divisible_by_3} | {lfo_all_points_within_0p01_deduplicated_fraction:.6f} | {lfo_all_points_within_0p01_occurrence_fraction:.6f} | {control_point_x_mean_abs_error_interior_occurrence_weighted:.8f} | {control_point_x_p95_abs_error_interior_occurrence_weighted:.8f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Factor-3 Subdivision Checks",
            "",
            "| factor-3 grid points | factor-3 subdivisions | higher grid points | higher subdivisions | p95 delta | factor-3 beats/matches higher |",
            "| ---: | ---: | ---: | ---: | ---: | :---: |",
        ]
    )
    for row in factor3_rows:
        lines.append(
            "| {factor3_grid_point_count} | {factor3_subdivision_count} | {higher_nonfactor3_grid_point_count} | {higher_nonfactor3_subdivision_count} | {interior_p95_abs_error_delta_factor3_minus_higher:.8f} | {factor3_beats_or_matches_higher_on_interior_p95} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "Interpretation notes:",
            "",
            "- A negative p95 delta means the factor-3 subdivision grid beat the higher point-count comparator.",
            "- This is not a reconstruction metric; it ignores y and all segment connection rules.",
            "- This is not a model prediction head budget claim.",
            "- Topology remains analysis-only in the processed corpus.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()

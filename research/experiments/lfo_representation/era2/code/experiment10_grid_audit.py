#!/usr/bin/env python
"""Standalone Experiment 10 control-point x-grid audit."""

from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import json
from pathlib import Path
import shutil
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


DEFAULT_SUBDIVISION_COUNTS = tuple(range(8, 101, 2))
DEFAULT_GRID_POINT_COUNTS = tuple(subdivision_count + 1 for subdivision_count in DEFAULT_SUBDIVISION_COUNTS)
DEFAULT_FACTOR3_SUBDIVISION_COMPARISONS = tuple(
    (subdivision_count, subdivision_count + 2)
    for subdivision_count in DEFAULT_SUBDIVISION_COUNTS
    if subdivision_count % 3 == 0 and subdivision_count + 2 in DEFAULT_SUBDIVISION_COUNTS
)
DEFAULT_FACTOR3_GRID_POINT_COMPARISONS = tuple(
    (factor3_subdivision_count + 1, higher_subdivision_count + 1)
    for factor3_subdivision_count, higher_subdivision_count in DEFAULT_FACTOR3_SUBDIVISION_COMPARISONS
)
VITAL_MAX_POINTS = 100
EXACT_TOLERANCE = 1e-6
CONTROL_POINT_X_PASS_TOLERANCE = 0.001
ERA2_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ERA2_ROOT / "artifacts" / "experiment_10" / "control_point_x_grid"
DEFAULT_REPORT_PATH = ERA2_ROOT / "reports" / "EXPERIMENT_10_CONTROL_POINT_X_GRID_REPORT.md"


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    root.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    root.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    root.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    root.add_argument(
        "--subdivision-counts",
        dest="subdivision_counts",
        default=",".join(str(value) for value in DEFAULT_SUBDIVISION_COUNTS),
        help="comma-separated x-grid subdivision counts; control point count is subdivisions + 1",
    )
    root.add_argument(
        "--grid-point-counts",
        dest="grid_point_counts",
        default=None,
        help="optional comma-separated inclusive x-grid point counts; overrides --subdivision-counts",
    )
    root.add_argument("--include-inactive", action="store_true")
    root.add_argument("--no-build-corpus", action="store_true")
    root.add_argument("--force-rebuild-corpus", action="store_true")
    return root


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    grid_point_counts = (
        parse_counts(args.grid_point_counts)
        if args.grid_point_counts is not None
        else tuple(subdivision_count + 1 for subdivision_count in parse_counts(args.subdivision_counts))
    )
    result = run_experiment10_grid_audit(
        metadata_path=args.metadata,
        corpus_dir=args.corpus_dir,
        output_dir=args.output_dir,
        report_path=args.report_path,
        grid_point_counts=grid_point_counts,
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
    report_path: Path = DEFAULT_REPORT_PATH,
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
    lattice_frequency_rows = _control_point_x_lattice_frequency_rows(corpus.shapes, weights)
    control_rows = []
    learned_grid_records = []
    for index, grid_point_count in enumerate(grid_point_counts, start=1):
        subdivision_count = int(grid_point_count) - 1
        if progress:
            progress(
                "experiment10: "
                f"[{index}/{len(grid_point_counts)}] subdivision_count={subdivision_count} "
                f"control_point_count={grid_point_count} starting"
            )
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
                    "subdivision_count": int(subdivision_count),
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
                f"[{index}/{len(grid_point_counts)}] subdivision_count={uniform_row['subdivision_count']} "
                f"control_point_count={grid_point_count} "
                f"uniform_interior_p95={uniform_row['control_point_x_p95_abs_error_interior_occurrence_weighted']:.8f}"
            )
    factor3_rows = _factor3_grid_point_rows(control_rows, comparisons=DEFAULT_FACTOR3_GRID_POINT_COMPARISONS)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "point_count_frequency.csv", point_frequency_rows)
    _write_csv(output_dir / "control_point_x_lattice_frequency.csv", lattice_frequency_rows)
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
        lattice_frequency_rows=lattice_frequency_rows,
        control_rows=control_rows,
    )
    report_path = Path(report_path)
    report_plot_paths = _copy_report_plots(plot_paths, output_dir=output_dir, report_path=report_path)
    report = _report(
        point_frequency_rows=point_frequency_rows,
        lattice_frequency_rows=lattice_frequency_rows,
        control_rows=control_rows,
        factor3_rows=factor3_rows,
        plot_paths=report_plot_paths,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    manifest = {
        "experiment_id": "experiment_10",
        "experiment_name": "control_point_x_grid_audit",
        "corpus_dir": str(corpus_dir),
        "report_path": str(report_path),
        "corpus_manifest": corpus.manifest,
        "grid_point_counts": list(grid_point_counts),
        "subdivision_counts": [int(grid_point_count) - 1 for grid_point_count in grid_point_counts],
        "control_point_count_contract": "control_point_count is the number of grid points and equals subdivision_count + 1.",
        "factor3_grid_point_comparisons": [list(pair) for pair in DEFAULT_FACTOR3_GRID_POINT_COMPARISONS],
        "factor3_subdivision_comparisons": [list(pair) for pair in DEFAULT_FACTOR3_SUBDIVISION_COMPARISONS],
        "active_only": bool(active_only),
        "method": "point_count_frequency_plus_control_point_x_grid_error",
        "grid_count_contract": "Experiment 10 varies subdivision_count. control_point_count/grid_point_count is inferred as subdivision_count + 1. W is reserved for residual-layer atom choices.",
        "control_point_x_contract": "Control-point placement is evaluated on x only. For each true ordered control point, the predicted x is the nearest point in the fixed grid; y is not scored and no curve is rendered.",
        "pass_rate_contract": "lfo_all_points_within_0p001_* is the fraction of LFOs whose maximum control-point x error is <= 0.001.",
        "nonuniform_grid_contract": "global_quantile grids are fixed offline-learned decoder grids. They do not require the deployed model to predict grid locations.",
        "report_plots": plot_paths,
        "report_image_paths": report_plot_paths,
        "standalone_note": "Experiment 10 is intentionally outside the shared Era 2 model-runner CLI.",
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if progress:
        progress(f"experiment10: wrote summary artifacts to {output_dir}")
        progress(f"experiment10: wrote report to {report_path}")
    return {
        "output_dir": str(output_dir),
        "summary": str(output_dir / "summary.csv"),
        "point_count_frequency": str(output_dir / "point_count_frequency.csv"),
        "control_point_x_lattice_frequency": str(output_dir / "control_point_x_lattice_frequency.csv"),
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


def _control_point_x_lattice_frequency_rows(
    shapes: tuple[LfoShape, ...],
    weights: np.ndarray,
) -> list[dict[str, Any]]:
    occurrence_counts: dict[float, float] = {}
    deduplicated_counts: dict[float, float] = {}
    for shape, occurrence_weight in zip(shapes, weights):
        interior_x = shape.points[:, 0]
        interior_x = interior_x[(interior_x > 0.0) & (interior_x < 1.0)]
        for x_value in interior_x:
            key = round(float(x_value), 12)
            occurrence_counts[key] = occurrence_counts.get(key, 0.0) + float(occurrence_weight)
            deduplicated_counts[key] = deduplicated_counts.get(key, 0.0) + 1.0

    occurrence_total = float(sum(occurrence_counts.values()))
    deduplicated_total = float(sum(deduplicated_counts.values()))
    rows = []
    for x_value, occurrence_count in sorted(occurrence_counts.items()):
        fraction = Fraction(x_value).limit_denominator(128)
        fraction_value = float(fraction)
        is_simple_rational = abs(fraction_value - x_value) <= EXACT_TOLERANCE
        rows.append(
            {
                "x_value": float(x_value),
                "fraction": _format_fraction(fraction) if is_simple_rational else "",
                "numerator": int(fraction.numerator) if is_simple_rational else "",
                "denominator": int(fraction.denominator) if is_simple_rational else "",
                "is_simple_rational": bool(is_simple_rational),
                "occurrence_point_count": float(occurrence_count),
                "occurrence_point_fraction": occurrence_count / occurrence_total if occurrence_total else 0.0,
                "deduplicated_point_count": float(deduplicated_counts.get(x_value, 0.0)),
                "deduplicated_point_fraction": (
                    deduplicated_counts.get(x_value, 0.0) / deduplicated_total if deduplicated_total else 0.0
                ),
            }
        )
    rows.sort(key=lambda row: (-float(row["occurrence_point_count"]), float(row["x_value"])))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
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
    subdivision_count = int(grid_point_count) - 1
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
        "lfo_all_points_within_0p001_deduplicated_fraction": float(np.mean(lfo_all_pass_array)) if len(lfo_all_pass_array) else 0.0,
        "lfo_all_points_within_0p001_occurrence_fraction": _weighted_bool_fraction(lfo_all_pass_array, weights),
        "lfo_interior_points_within_0p001_deduplicated_fraction": float(np.mean(lfo_interior_pass_array)) if len(lfo_interior_pass_array) else 0.0,
        "lfo_interior_points_within_0p001_occurrence_fraction": _weighted_bool_fraction(lfo_interior_pass_array, weights),
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
    lattice_frequency_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "point_count_frequency": plot_dir / "experiment10_point_count_frequency.png",
        "x_lattice_frequency": plot_dir / "experiment10_control_point_x_lattice_frequency.png",
        "x_error_median": plot_dir / "experiment10_control_point_x_median.png",
        "x_error_p95": plot_dir / "experiment10_control_point_x_p95.png",
        "lfo_pass_rate_0p001": plot_dir / "experiment10_lfo_pass_rate_0p001.png",
        "nonuniform_delta": plot_dir / "experiment10_nonuniform_delta.png",
        "factor3_checks": plot_dir / "experiment10_factor3_checks.png",
    }
    _plot_point_count_frequency(plt, paths["point_count_frequency"], point_frequency_rows)
    _plot_x_lattice_frequency(plt, paths["x_lattice_frequency"], lattice_frequency_rows)
    _plot_control_metric(
        plt,
        paths["x_error_median"],
        control_rows,
        metric="control_point_x_median_abs_error_interior_occurrence_weighted",
        ylabel="Interior median abs x error",
        title="Typical control-point x placement error",
        direction_note="Lower is better",
    )
    _plot_control_metric(
        plt,
        paths["x_error_p95"],
        control_rows,
        metric="control_point_x_p95_abs_error_interior_occurrence_weighted",
        ylabel="Interior P95 abs x error",
        title="Tail control-point x placement error",
        direction_note="Lower is better",
    )
    _plot_control_metric(
        plt,
        paths["lfo_pass_rate_0p001"],
        control_rows,
        metric="lfo_all_points_within_0p001_occurrence_fraction",
        ylabel="Occurrence-weighted LFO fraction",
        title="LFOs with every control point within 0.001 of grid",
        direction_note="Higher is better",
        y_limits=(0.0, 1.02),
    )
    _plot_nonuniform_delta(plt, paths["nonuniform_delta"], control_rows)
    _plot_factor3_checks(
        plt,
        paths["factor3_checks"],
        _factor3_grid_point_rows(control_rows, comparisons=DEFAULT_FACTOR3_GRID_POINT_COMPARISONS),
    )
    return {key: path.relative_to(plot_dir.parent).as_posix() for key, path in paths.items()}


def _copy_report_plots(plot_paths: dict[str, str], *, output_dir: Path, report_path: Path) -> dict[str, str]:
    report_dir = Path(report_path).resolve().parent
    image_dir = report_dir / "images" / "experiment_10"
    image_dir.mkdir(parents=True, exist_ok=True)
    report_paths = {}
    for key, relative_path in plot_paths.items():
        source = Path(output_dir, relative_path).resolve()
        destination = image_dir / source.name
        shutil.copy2(source, destination)
        report_paths[key] = destination.relative_to(report_dir).as_posix()
    return report_paths


def _plot_point_count_frequency(plt: Any, path: Path, rows: list[dict[str, Any]]) -> None:
    top_rows = sorted(rows, key=lambda row: float(row["lfo_corpus_occurrence_fraction"]), reverse=True)[:14]
    top_rows = sorted(top_rows, key=lambda row: int(row["source_point_count"]))
    x = np.arange(len(top_rows), dtype=np.float64)
    labels = [str(int(row["source_point_count"])) for row in top_rows]
    dedup = np.asarray([float(row["deduplicated_lfo_fraction"]) for row in top_rows], dtype=np.float64)
    occurrence = np.asarray([float(row["lfo_corpus_occurrence_fraction"]) for row in top_rows], dtype=np.float64)
    all_counts = [int(row["source_point_count"]) for row in rows]
    cumulative_dedup = [float(row["deduplicated_lfo_cumulative_fraction"]) for row in rows]
    cumulative_occurrence = [float(row["lfo_corpus_cumulative_fraction"]) for row in rows]

    fig, (ax, cdf_ax) = plt.subplots(
        2,
        1,
        figsize=(12.0, 8.0),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2.0, 1.05]},
    )
    width = 0.42
    ax.bar(x - width / 2.0, dedup, width=width, label="Deduplicated LFO corpus", color="#7E57C2")
    ax.bar(x + width / 2.0, occurrence, width=width, label="Occurrence-weighted LFO corpus", color="#4C78A8")
    ax.set_title("Source control-point count frequency: dominant counts plus cumulative tail")
    ax.set_xlabel("Source point count (top counts by occurrence)")
    ax.set_ylabel("Fraction of corpus")
    ax.set_xticks(x, labels)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)

    cdf_ax.plot(all_counts, cumulative_dedup, marker="o", markersize=3.0, linewidth=1.8, label="Deduplicated cumulative", color="#7E57C2")
    cdf_ax.plot(all_counts, cumulative_occurrence, marker="o", markersize=3.0, linewidth=1.8, label="Occurrence cumulative", color="#4C78A8")
    cdf_ax.axhline(0.95, color="#222222", linewidth=1.0, linestyle="--", alpha=0.65)
    cdf_ax.set_xlabel("Source point count")
    cdf_ax.set_ylabel("Cumulative fraction")
    cdf_ax.set_ylim(0.0, 1.02)
    cdf_ax.grid(alpha=0.25)
    cdf_ax.legend(frameon=False, loc="lower right")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_x_lattice_frequency(plt: Any, path: Path, rows: list[dict[str, Any]]) -> None:
    top_rows = sorted(rows, key=lambda row: float(row["occurrence_point_fraction"]), reverse=True)[:18]
    top_rows = sorted(top_rows, key=lambda row: float(row["x_value"]))
    denominator_rows = _denominator_family_rows(rows)
    denominator_rows = sorted(denominator_rows, key=lambda row: float(row["occurrence_point_fraction"]), reverse=True)[:12]

    fig, (position_ax, denominator_ax) = plt.subplots(
        2,
        1,
        figsize=(13.0, 8.0),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.35, 1.0]},
    )

    x = np.arange(len(top_rows), dtype=np.float64)
    labels = [_lattice_row_label(row) for row in top_rows]
    dedup = np.asarray([float(row["deduplicated_point_fraction"]) for row in top_rows], dtype=np.float64)
    occurrence = np.asarray([float(row["occurrence_point_fraction"]) for row in top_rows], dtype=np.float64)
    width = 0.42
    position_ax.bar(x - width / 2.0, dedup, width=width, label="Deduplicated interior points", color="#7E57C2")
    position_ax.bar(x + width / 2.0, occurrence, width=width, label="Occurrence-weighted interior points", color="#4C78A8")
    position_ax.set_title("Top interior control-point x positions")
    position_ax.set_xlabel("Interior x position")
    position_ax.set_ylabel("Fraction of interior points")
    position_ax.set_xticks(x, labels, rotation=35, ha="right")
    position_ax.grid(axis="y", alpha=0.25)
    position_ax.legend(frameon=False)

    d_x = np.arange(len(denominator_rows), dtype=np.float64)
    d_labels = [str(row["denominator_label"]) for row in denominator_rows]
    d_dedup = np.asarray([float(row["deduplicated_point_fraction"]) for row in denominator_rows], dtype=np.float64)
    d_occurrence = np.asarray([float(row["occurrence_point_fraction"]) for row in denominator_rows], dtype=np.float64)
    denominator_ax.bar(d_x - width / 2.0, d_dedup, width=width, label="Deduplicated denominator-family mass", color="#7E57C2")
    denominator_ax.bar(d_x + width / 2.0, d_occurrence, width=width, label="Occurrence-weighted denominator-family mass", color="#4C78A8")
    denominator_ax.set_title("Reduced-denominator family mass")
    denominator_ax.set_xlabel("Reduced denominator of simple rational x")
    denominator_ax.set_ylabel("Fraction of interior points")
    denominator_ax.set_xticks(d_x, d_labels)
    denominator_ax.grid(axis="y", alpha=0.25)
    denominator_ax.legend(frameon=False)

    fig.suptitle("Interior x-position lattice structure", fontsize=14)
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
    direction_note: str,
    y_limits: tuple[float, float] | None = None,
) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14.0, 5.8),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.08, 1.0]},
    )
    zoom_threshold = 54
    for ax, zoom in zip(axes, (False, True)):
        _draw_control_metric_axis(ax, rows, metric=metric, zoom=zoom)
        ax.set_xlabel("Subdivision count (control point count = subdivisions + 1)")
        ax.set_ylabel(ylabel)
        ax.axvline(96, color="#222222", linewidth=1.0, linestyle=":", alpha=0.45)
        if y_limits is not None and not zoom:
            ax.set_ylim(*y_limits)
        if zoom:
            _apply_zoom_limits(ax, rows, metric=metric, min_subdivision=zoom_threshold, y_limits=y_limits)
            ax.set_title(f"Zoom: subdivisions >= {zoom_threshold}")
        else:
            ax.set_title("Full sweep")
        ax.grid(alpha=0.25)
    fig.suptitle(f"{title} ({direction_note})", fontsize=14)
    axes[0].legend(frameon=False, loc="best")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _draw_control_metric_axis(ax: Any, rows: list[dict[str, Any]], *, metric: str, zoom: bool) -> None:
    for label, group in _group_control_rows(rows).items():
        ordered = sorted(group, key=lambda row: int(row["grid_point_count"]))
        if zoom:
            ordered = [row for row in ordered if int(row["subdivision_count"]) >= 54]
        x = [int(row["subdivision_count"]) for row in ordered]
        y = [float(row[metric]) for row in ordered]
        if label == "uniform":
            ax.plot(x, y, linewidth=2.0, color="#E45756", label=label)
            div3_x = [int(row["subdivision_count"]) for row in ordered if row["subdivision_divisible_by_3"] is True]
            div3_y = [float(row[metric]) for row in ordered if row["subdivision_divisible_by_3"] is True]
            non_div3_x = [int(row["subdivision_count"]) for row in ordered if row["subdivision_divisible_by_3"] is not True]
            non_div3_y = [float(row[metric]) for row in ordered if row["subdivision_divisible_by_3"] is not True]
            ax.scatter(non_div3_x, non_div3_y, s=35, color="#E45756", edgecolors="white", linewidths=0.6)
            ax.scatter(div3_x, div3_y, s=70, marker="D", color="#54A24B", edgecolors="#222222", linewidths=0.45, label="uniform, subdivision divisible by 3")
        else:
            ax.plot(x, y, marker="o", linewidth=1.8, markersize=4.0, color=_grid_plot_color(label), label=label)


def _apply_zoom_limits(
    ax: Any,
    rows: list[dict[str, Any]],
    *,
    metric: str,
    min_subdivision: int,
    y_limits: tuple[float, float] | None,
) -> None:
    ax.set_xlim(min_subdivision - 1, max(int(row["subdivision_count"]) for row in rows) + 1)
    values = [
        float(row[metric])
        for row in rows
        if int(row["subdivision_count"]) >= min_subdivision
    ]
    if not values:
        return
    if y_limits is not None:
        lower = max(y_limits[0], min(values) - 0.05 * max(1e-12, max(values) - min(values)))
        upper = min(y_limits[1], max(values) + 0.05 * max(1e-12, max(values) - min(values)))
    else:
        margin = 0.08 * max(1e-12, max(values) - min(values))
        lower = max(0.0, min(values) - margin)
        upper = max(values) + margin
    if upper > lower:
        ax.set_ylim(lower, upper)


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
            x.append(int(row["subdivision_count"]))
            y.append(
                float(row["control_point_x_p95_abs_error_interior_occurrence_weighted"])
                - float(baseline["control_point_x_p95_abs_error_interior_occurrence_weighted"])
            )
        label = f"global quantile ({weighting})"
        ax.plot(x, y, marker="o", linewidth=2.0, markersize=4.5, color=_grid_plot_color(label), label=label)
    ax.axhline(0.0, color="#222222", linewidth=1.0)
    ax.set_title("Global non-uniform grid P95 delta vs uniform")
    ax.set_xlabel("Subdivision count (same control point count as uniform comparator)")
    ax.set_ylabel("Interior P95 abs x error delta")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_factor3_checks(plt: Any, path: Path, rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.8), constrained_layout=True)
    labels = [
        str(row["factor3_subdivision_count"])
        for row in rows
    ]
    x = np.arange(len(rows), dtype=np.float64)
    factor3 = np.asarray([float(row["factor3_interior_p95_abs_error"]) for row in rows], dtype=np.float64)
    higher = np.asarray([float(row["higher_interior_p95_abs_error"]) for row in rows], dtype=np.float64)
    width = 0.42
    ax.bar(x - width / 2.0, factor3, width=width, color="#E45756", label="factor-3 subdivision grid")
    ax.bar(x + width / 2.0, higher, width=width, color="#6B7280", label="higher non-factor-3 comparator")
    ax.set_title("Factor-3 subdivision checks against next higher even non-factor-3 subdivision")
    ax.set_xlabel("Factor-3 subdivision count")
    ax.set_ylabel("Interior P95 abs x error")
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _grid_plot_color(label: str) -> str:
    if label == "uniform":
        return "#E45756"
    if "deduplicated" in label:
        return "#7E57C2"
    if "occurrence_weighted" in label:
        return "#4C78A8"
    return "#6B7280"


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


def _format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _lattice_row_label(row: dict[str, Any]) -> str:
    fraction = str(row.get("fraction", ""))
    if fraction:
        return fraction
    return f"{float(row['x_value']):.4f}"


def _denominator_family_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals = {
        "occurrence": sum(float(row["occurrence_point_count"]) for row in rows),
        "deduplicated": sum(float(row["deduplicated_point_count"]) for row in rows),
    }
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        denominator = row.get("denominator", "")
        label = str(int(denominator)) if denominator != "" else "other"
        group = grouped.setdefault(
            label,
            {
                "denominator_label": label,
                "denominator": int(denominator) if denominator != "" else "",
                "occurrence_point_count": 0.0,
                "deduplicated_point_count": 0.0,
            },
        )
        group["occurrence_point_count"] += float(row["occurrence_point_count"])
        group["deduplicated_point_count"] += float(row["deduplicated_point_count"])
    denominator_rows = []
    for group in grouped.values():
        occurrence_count = float(group["occurrence_point_count"])
        deduplicated_count = float(group["deduplicated_point_count"])
        denominator_rows.append(
            {
                **group,
                "occurrence_point_fraction": occurrence_count / totals["occurrence"] if totals["occurrence"] else 0.0,
                "deduplicated_point_fraction": (
                    deduplicated_count / totals["deduplicated"] if totals["deduplicated"] else 0.0
                ),
            }
        )
    return sorted(
        denominator_rows,
        key=lambda row: (-float(row["occurrence_point_fraction"]), _denominator_sort_value(row["denominator"])),
    )


def _denominator_sort_value(value: Any) -> int:
    return int(value) if value != "" else 10_000


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _sum_denominator_fraction(rows: list[dict[str, Any]], predicate: Callable[[int], bool]) -> float:
    total = 0.0
    for row in rows:
        denominator = row.get("denominator", "")
        if denominator == "":
            continue
        if predicate(int(denominator)):
            total += float(row["occurrence_point_fraction"])
    return total


def _nonuniform_band_summary(rows: list[dict[str, Any]], *, low: int, high: int) -> dict[str, Any]:
    band_rows = [row for row in rows if low <= int(row["subdivision_count"]) <= high]
    if not band_rows:
        return {
            "low": low,
            "high": high,
            "count": 0,
            "negative_count": 0,
            "mean_delta": 0.0,
        }
    deltas = np.asarray([float(row["p95_delta"]) for row in band_rows], dtype=np.float64)
    return {
        "low": low,
        "high": high,
        "count": len(band_rows),
        "negative_count": int(np.sum(deltas < 0.0)),
        "mean_delta": float(np.mean(deltas)),
    }


def _report_findings(
    point_frequency_rows: list[dict[str, Any]],
    lattice_frequency_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
    factor3_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    uniform_rows = sorted(
        [row for row in control_rows if row["grid_kind"] == "uniform"],
        key=lambda row: int(row["grid_point_count"]),
    )
    occurrence_top = max(point_frequency_rows, key=lambda row: float(row["lfo_corpus_occurrence_fraction"]))
    dedup_top = max(point_frequency_rows, key=lambda row: float(row["deduplicated_lfo_fraction"]))
    cumulative_5 = _point_frequency_at_or_before(point_frequency_rows, 5)
    cumulative_18 = _point_frequency_at_or_before(point_frequency_rows, 18)
    first_uniform_pass_95 = _first_row_at_or_above(
        uniform_rows,
        "lfo_all_points_within_0p001_occurrence_fraction",
        0.95,
    )
    first_uniform_p95_le_001 = _first_row_at_or_below(
        uniform_rows,
        "control_point_x_p95_abs_error_interior_occurrence_weighted",
        0.001,
    )
    best_uniform_p95 = min(uniform_rows, key=lambda row: float(row["control_point_x_p95_abs_error_interior_occurrence_weighted"]))
    valid_uniform_rows = [row for row in uniform_rows if int(row["grid_point_count"]) <= VITAL_MAX_POINTS]
    best_valid_uniform_p95 = min(
        valid_uniform_rows,
        key=lambda row: float(row["control_point_x_p95_abs_error_interior_occurrence_weighted"]),
    )
    best_overall_p95 = min(control_rows, key=lambda row: float(row["control_point_x_p95_abs_error_interior_occurrence_weighted"]))
    nonuniform_delta_rows = _nonuniform_delta_rows(control_rows)
    best_nonuniform_delta = min(nonuniform_delta_rows, key=lambda row: row["p95_delta"]) if nonuniform_delta_rows else None
    worst_nonuniform_delta = max(nonuniform_delta_rows, key=lambda row: row["p95_delta"]) if nonuniform_delta_rows else None
    factor3_wins = sum(1 for row in factor3_rows if row["factor3_beats_or_matches_higher_on_interior_p95"])
    denominator_rows = _denominator_family_rows(lattice_frequency_rows)
    uniform_median_zero_subdivisions = [
        int(row["subdivision_count"])
        for row in uniform_rows
        if float(row["control_point_x_median_abs_error_interior_occurrence_weighted"]) <= EXACT_TOLERANCE
    ]
    factor3_win_rows = [row for row in factor3_rows if row["factor3_beats_or_matches_higher_on_interior_p95"]]
    return {
        "occurrence_top": occurrence_top,
        "dedup_top": dedup_top,
        "cumulative_5": cumulative_5,
        "cumulative_18": cumulative_18,
        "top_lattice_rows": lattice_frequency_rows[:8],
        "denominator_rows": denominator_rows,
        "dyadic_occurrence_fraction": _sum_denominator_fraction(
            denominator_rows,
            lambda denominator: _is_power_of_two(denominator),
        ),
        "third_family_occurrence_fraction": _sum_denominator_fraction(
            denominator_rows,
            lambda denominator: denominator % 3 == 0,
        ),
        "first_uniform_pass_95": first_uniform_pass_95,
        "first_uniform_p95_le_001": first_uniform_p95_le_001,
        "best_uniform_p95": best_uniform_p95,
        "best_valid_uniform_p95": best_valid_uniform_p95,
        "best_overall_p95": best_overall_p95,
        "best_nonuniform_delta": best_nonuniform_delta,
        "worst_nonuniform_delta": worst_nonuniform_delta,
        "nonuniform_low_band": _nonuniform_band_summary(nonuniform_delta_rows, low=8, high=30),
        "nonuniform_mid_band": _nonuniform_band_summary(nonuniform_delta_rows, low=32, high=52),
        "nonuniform_high_band": _nonuniform_band_summary(nonuniform_delta_rows, low=54, high=100),
        "uniform_median_zero_subdivisions": uniform_median_zero_subdivisions,
        "factor3_win_rows": factor3_win_rows,
        "factor3_wins": factor3_wins,
        "factor3_total": len(factor3_rows),
    }


def _point_frequency_at_or_before(rows: list[dict[str, Any]], source_point_count: int) -> dict[str, Any]:
    eligible = [row for row in rows if int(row["source_point_count"]) <= source_point_count]
    return eligible[-1] if eligible else rows[0]


def _first_row_at_or_above(rows: list[dict[str, Any]], metric: str, threshold: float) -> dict[str, Any] | None:
    for row in rows:
        if float(row[metric]) >= threshold:
            return row
    return None


def _first_row_at_or_below(rows: list[dict[str, Any]], metric: str, threshold: float) -> dict[str, Any] | None:
    for row in rows:
        if float(row[metric]) <= threshold:
            return row
    return None


def _nonuniform_delta_rows(control_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    uniform_by_points = {
        int(row["grid_point_count"]): row
        for row in control_rows
        if row["grid_kind"] == "uniform"
    }
    rows = []
    for row in control_rows:
        if row["grid_kind"] != "global_quantile":
            continue
        grid_point_count = int(row["grid_point_count"])
        uniform = uniform_by_points.get(grid_point_count)
        if uniform is None:
            continue
        rows.append(
            {
                "grid_kind": row["grid_kind"],
                "grid_learning_weighting": row["grid_learning_weighting"],
                "grid_point_count": grid_point_count,
                "subdivision_count": row["subdivision_count"],
                "uniform_subdivision_count": uniform["subdivision_count"],
                "p95_delta": float(row["control_point_x_p95_abs_error_interior_occurrence_weighted"])
                - float(uniform["control_point_x_p95_abs_error_interior_occurrence_weighted"]),
                "row_p95": float(row["control_point_x_p95_abs_error_interior_occurrence_weighted"]),
                "uniform_p95": float(uniform["control_point_x_p95_abs_error_interior_occurrence_weighted"]),
            }
        )
    return rows


def _selected_uniform_rows(control_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected_subdivisions = {8, 10, 12, 24, 32, 48, 60, 64, 72, 80, 96, 98, 100}
    return [
        row
        for row in sorted(control_rows, key=lambda item: int(item["grid_point_count"]))
        if row["grid_kind"] == "uniform" and int(row["subdivision_count"]) in selected_subdivisions
    ]


def _format_grid_point_row(row: dict[str, Any] | None) -> str:
    if row is None:
        return "not reached by the tested subdivision counts"
    return (
        f"`subdivision_count={int(row['subdivision_count'])}` "
        f"(`control_point_count={int(row['grid_point_count'])}`)"
    )


def _format_subdivision(row: dict[str, Any]) -> str:
    return str(row.get("subdivision_count", ""))


def _format_bool(value: Any) -> str:
    if value == "":
        return "n/a"
    return "yes" if bool(value) else "no"


def _report(
    *,
    point_frequency_rows: list[dict[str, Any]],
    lattice_frequency_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
    factor3_rows: list[dict[str, Any]],
    plot_paths: dict[str, str],
) -> str:
    findings = _report_findings(point_frequency_rows, lattice_frequency_rows, control_rows, factor3_rows)
    occurrence_top = findings["occurrence_top"]
    dedup_top = findings["dedup_top"]
    cumulative_5 = findings["cumulative_5"]
    cumulative_18 = findings["cumulative_18"]
    first_uniform_pass_95 = findings["first_uniform_pass_95"]
    first_uniform_p95_le_001 = findings["first_uniform_p95_le_001"]
    best_uniform_p95 = findings["best_uniform_p95"]
    best_valid_uniform_p95 = findings["best_valid_uniform_p95"]
    best_overall_p95 = findings["best_overall_p95"]
    best_nonuniform_delta = findings["best_nonuniform_delta"]
    worst_nonuniform_delta = findings["worst_nonuniform_delta"]
    top_lattice = ", ".join(
        f"`{_lattice_row_label(row)}` ({float(row['occurrence_point_fraction']):.2%})"
        for row in findings["top_lattice_rows"][:6]
    )
    denominator_mass = ", ".join(
        f"`{row['denominator_label']}` ({float(row['occurrence_point_fraction']):.2%})"
        for row in findings["denominator_rows"][:6]
    )
    median_zero = findings["uniform_median_zero_subdivisions"]
    median_zero_summary = ", ".join(str(value) for value in median_zero[:12])
    if len(median_zero) > 12:
        median_zero_summary += ", ..."
    factor3_win_rows = findings["factor3_win_rows"]
    if factor3_win_rows:
        factor3_win_summary = ", ".join(
            f"`{int(row['factor3_subdivision_count'])}` vs `{int(row['higher_nonfactor3_subdivision_count'])}`"
            for row in factor3_win_rows
        )
    else:
        factor3_win_summary = "none"
    low_band = findings["nonuniform_low_band"]
    mid_band = findings["nonuniform_mid_band"]
    high_band = findings["nonuniform_high_band"]
    lines = [
        "# Experiment 10: Control-Point X Grid Audit",
        "",
        "## Main Findings",
        "",
        "The main result is not that factor-3 grids are broadly special. The stronger pattern is that this corpus is heavily dyadic. Interior control-point x positions pile up on `1/2`, quarters, eighths, sixteenths, and thirty-seconds. Factor-3 helps only when it lands on top of that dyadic structure instead of competing with it.",
        "",
        f"The source corpus is also simple in point count. The most common occurrence-weighted source count is {int(occurrence_top['source_point_count'])} points ({float(occurrence_top['lfo_corpus_occurrence_fraction']):.3%} of LFO occurrences). The most common deduplicated source count is {int(dedup_top['source_point_count'])} points ({float(dedup_top['deduplicated_lfo_fraction']):.3%} of unique LFO shapes). Source LFOs with at most 5 points cover {float(cumulative_5['lfo_corpus_cumulative_fraction']):.3%} of occurrence-weighted usage and {float(cumulative_5['deduplicated_lfo_cumulative_fraction']):.3%} of deduplicated shapes.",
        "",
        f"The top occurrence-weighted interior x positions are {top_lattice}. By reduced denominator, the largest families are {denominator_mass}. Summing denominator families, dyadic positions account for {float(findings['dyadic_occurrence_fraction']):.3%} of occurrence-weighted interior point mass, while denominator families divisible by 3 account for {float(findings['third_family_occurrence_fraction']):.3%}. This is why the plots have a sawtooth/periodic look.",
        "",
        f"Uniform median error hits zero at subdivision counts `{median_zero_summary}`. That is not an accident: those rows align with enough of the dominant dyadic lattice that more than half of occurrence-weighted interior points are exact hits.",
        "",
        f"The P95 curve mostly rewards more subdivisions, but it also has sharp alignment drops at dyadic-friendly rows. The best valid high-end uniform row is {_format_grid_point_row(best_valid_uniform_p95)} with interior P95 x error {float(best_valid_uniform_p95['control_point_x_p95_abs_error_interior_occurrence_weighted']):.8f}. That row matters because `subdivision_count=96` gives 97 control points, stays inside Vital's 100-point limit, and combines dyadic alignment with factor-3 alignment. The absolute best uniform row in the sweep is {_format_grid_point_row(best_uniform_p95)} with P95 {float(best_uniform_p95['control_point_x_p95_abs_error_interior_occurrence_weighted']):.8f}, but rows above 99 control points need to be read against the Vital limit.",
        "",
        f"The `0.001` whole-LFO pass-rate check is intentionally strict. Uniform spacing alone would only guarantee every in-range x position is within `0.001` at `subdivision_count >= 500`, because the worst in-range rounding error is `1 / (2 * subdivision_count)`. So in this sweep the `0.001` plot should be read as a lattice-alignment diagnostic, not as a dense-grid acceptability threshold. The first uniform grid where at least 95% of occurrence-weighted LFOs pass that strict whole-LFO check is {_format_grid_point_row(first_uniform_pass_95)}.",
        "",
        f"Fixed global non-uniform grids are mixed. At very low subdivision counts they often over-focus the most frequent x positions and leave the tail exposed. In the low band (`8` through `30`), {low_band['negative_count']} of {low_band['count']} non-uniform P95 deltas are negative, with mean delta {float(low_band['mean_delta']):.8f}. In the mid band (`32` through `52`), {mid_band['negative_count']} of {mid_band['count']} are negative, mean delta {float(mid_band['mean_delta']):.8f}. In the high band (`54` through `100`), {high_band['negative_count']} of {high_band['count']} are negative, mean delta {float(high_band['mean_delta']):.8f}. Negative is good, but the high-subdivision margins are small.",
        "",
        f"Factor-3 alone is not a general win. It beats or matches the next higher even non-factor-3 comparator in {findings['factor3_wins']} of {findings['factor3_total']} tested pairings. Winning pairings: {factor3_win_summary}. The important case is `96`, because it is also dyadic-friendly.",
        "",
        "## Why The Curves Look Periodic",
        "",
        f"![Interior x-position lattice structure]({plot_paths['x_lattice_frequency']})",
        "",
        "The top subplot shows the actual interior x positions that dominate the corpus. The bottom subplot groups simple rational x positions by reduced denominator. If a uniform grid has `subdivision_count` divisible by one of these denominators, those points become exact hits. If it misses the denominator, the same points produce a visible jump in median, P95, or whole-LFO pass rate.",
        "",
        "This is why the uniform median curve has clean zero drops at multiples of 8, and why `subdivision_count=96` is unusually strong: it is divisible by 32 and by 3. A pure factor-3 interpretation would miss the larger dyadic story.",
        "",
        "## Plot Notes",
        "",
        "### Source Point-Count Frequency",
        "",
        f"![Source point-count frequency]({plot_paths['point_count_frequency']})",
        "",
        "Higher bars mean more corpus mass at that source control-point count. The occurrence-weighted corpus is even more concentrated than the deduplicated corpus: repeated preset usage strongly favors 3-point shapes. The lower cumulative plot shows that the long tail exists, but most practical coverage is already inside low point counts.",
        "",
        "### Median Interior X Error",
        "",
        f"![Interior median x error by grid type]({plot_paths['x_error_median']})",
        "",
        "Lower is better. The uniform median plot is the clearest periodicity signal: it repeatedly drops to zero at multiples of 8. That means the typical interior control point is not merely close to the grid; it is exactly on the grid for those subdivision counts. Global quantile grids reach near-zero median early because they place grid points directly on frequent corpus positions, but that does not mean the tail is solved.",
        "",
        "### P95 Interior X Error",
        "",
        f"![Interior P95 x error by grid type]({plot_paths['x_error_p95']})",
        "",
        "Lower is better. P95 mostly follows capacity: more subdivisions reduce the worst common rounding errors. The interesting deviations are the dyadic drops. `32`, `64`, and `96` are better than their immediate neighbors because they align with high-mass denominator families. The zoom panel is the useful decision region: it shows why `96` is the clean high-end uniform default under the 100-control-point constraint.",
        "",
        "### Whole-LFO Pass Rate At 0.001",
        "",
        f"![LFO pass rate at 0.001 tolerance]({plot_paths['lfo_pass_rate_0p001']})",
        "",
        "Higher is better. This plot asks a harder question than point-level P95: does every control point in the LFO land within `0.001` of the grid? At this threshold the curve should not saturate inside the tested sweep just because the grid is dense. Peaks are therefore telling us about exact or near-exact rational alignment, especially with the dyadic lattice.",
        "",
        "### Global Non-Uniform Delta",
        "",
        f"![Non-uniform grid delta versus uniform]({plot_paths['nonuniform_delta']})",
        "",
        "Negative is good. This subtracts same-count uniform P95 from global non-uniform P95. The large positive spikes at low subdivision counts are the cost of spending scarce grid slots on the most frequent positions while leaving less common positions exposed. After roughly the low 30s, non-uniform grids are usually slightly better on P95, but the high-subdivision advantage is small enough that it should be treated as a candidate, not a conclusion.",
        "",
        "### Factor-3 Subdivision Checks",
        "",
        f"![Factor-3 subdivision checks]({plot_paths['factor3_checks']})",
        "",
        "Lower is better. Each red bar is a subdivision count divisible by 3; each grey bar is the next higher even subdivision count that is not divisible by 3. Most factor-3 rows lose because the comparator simply has more subdivisions. The exception at `96` is the meaningful one because `96` is also aligned with the dominant dyadic families.",
        "",
        "## Practical Takeaways",
        "",
        "- Carry forward fixed uniform `subdivision_count=96` as the Era 2 default: `control_point_count=97`, inside Vital's 100-point limit, and aligned with both dyadic and factor-3 structure.",
        "- Do not describe factor-3 as a general rule. The better rule is: match the corpus x-position lattice, and note that this corpus is mostly dyadic.",
        "- Use the `0.001` pass-rate plot as a strict lattice diagnostic. It is not an acceptability-style read because the density guarantee is far outside the current sweep.",
        "- Future model experiments should not spend model prediction head budget on x-coordinate prediction, grid selection, or variable grid spacing.",
        "- This audit still does not choose atoms, score y values, render segments, or make model prediction head budget claims. It only measures x-position damage from fixed control-point grids.",
        "",
        "## Method Notes",
        "",
        "Experiment 10 is a standalone corpus/grid audit, not an Era 2 model-runner experiment. It asks how many source control points real LFOs use, then tests how well inclusive x-grid subdivision counts place those ordered control-point x positions.",
        "",
        "Naming contract:",
        "",
        "- `W` is reserved for residual-layer atom choices in Era 2 model experiments.",
        "- Experiment 10 varies `subdivision_count`.",
        "- `control_point_count = subdivision_count + 1`.",
        "- The CSV keeps `grid_point_count` as the implementation field name for the same value as `control_point_count`.",
        "- Factor language applies to `subdivision_count`, not to `control_point_count`.",
        "- Example: `subdivision_count = 96` means `control_point_count = 97`, and 96 is divisible by 2 and 3.",
        "",
        "Control-point x contract:",
        "",
        "- For each true ordered control point, predicted x is the nearest point in the fixed grid.",
        "- For `uniform`, grid points are `k / subdivision_count` for integer `k` from 0 through `subdivision_count`.",
        "- For `global_quantile`, grid points are fixed offline-learned non-uniform positions.",
        "- Y is ignored, and no line, Bezier, power curve, or other segment is rendered.",
        "- Repeated grid points are allowed because discontinuous LFOs can contain repeated x positions.",
        "- `lfo_all_points_within_0p001_*` reports the fraction of LFOs whose maximum x error is at most 0.001.",
        "",
        "Global non-uniform grids:",
        "",
        "- `global_quantile` grids are learned once offline from corpus control-point x positions.",
        "- The deployed model would still predict a grid slot; it would not predict the grid positions.",
        "- Both deduplicated and occurrence-weighted learned grids are reported.",
        "",
        "Generated artifacts:",
        "",
        "- `point_count_frequency.csv`",
        "- `control_point_x_lattice_frequency.csv`",
        "- `control_point_x_summary.csv`",
        "- `factor3_grid_point_comparisons.csv`",
        "- `global_nonuniform_grids.json`",
        "- `summary.csv`",
    ]
    if best_nonuniform_delta is not None and worst_nonuniform_delta is not None:
        lines.extend(
            [
                "",
                "Numerical anchors:",
                "",
                f"- The first uniform grid where occurrence-weighted interior P95 x error is at most `0.001` is {_format_grid_point_row(first_uniform_p95_le_001)}.",
                f"- The best overall P95 row in this sweep is `{best_overall_p95['grid_kind']}` / `{best_overall_p95['grid_learning_weighting']}` at {_format_grid_point_row(best_overall_p95)} with interior P95 x error {float(best_overall_p95['control_point_x_p95_abs_error_interior_occurrence_weighted']):.8f}.",
                f"- The strongest fixed global non-uniform P95 improvement versus same-count uniform is `{best_nonuniform_delta['grid_learning_weighting']}` at `subdivision_count={best_nonuniform_delta['uniform_subdivision_count']}` (`control_point_count={best_nonuniform_delta['grid_point_count']}`), delta {best_nonuniform_delta['p95_delta']:.8f}.",
                f"- The worst fixed global non-uniform P95 regression versus same-count uniform is `{worst_nonuniform_delta['grid_learning_weighting']}` at `subdivision_count={worst_nonuniform_delta['uniform_subdivision_count']}` (`control_point_count={worst_nonuniform_delta['grid_point_count']}`), delta {worst_nonuniform_delta['p95_delta']:.8f}.",
            ]
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()

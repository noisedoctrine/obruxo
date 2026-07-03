"""Best-case fixed atom-grid reconstruction ceiling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json
import time
from typing import Any, Callable

import numpy as np

from .dataset import TOPOLOGY_NAMES, LfoShape, load_presetshare_curve_dataset, sample_shape
from .manifest import write_json
from .metrics import reconstruction_summary, rmse_per_curve
from .runner import DEFAULT_METADATA, ERA2_ROOT


DEFAULT_ATOM_GRID_POINTS = (24, 36, 48, 60, 72, 96, 100)
DEFAULT_DENSE_POINTS = 1920
DEFAULT_OUTPUT_DIR = ERA2_ROOT / "artifacts" / "grid_ceiling"


@dataclass(frozen=True)
class GridCeilingResult:
    atom_grid_points: int
    dense_points: int
    row_count: int
    elapsed_seconds: float
    metrics: dict[str, Any]

    def as_row(self) -> dict[str, Any]:
        return {
            "atom_grid_points": self.atom_grid_points,
            "dense_points": self.dense_points,
            "row_count": self.row_count,
            "elapsed_seconds": self.elapsed_seconds,
            **self.metrics,
        }


def run_grid_ceiling_audit(
    *,
    metadata_path: Path = DEFAULT_METADATA,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    atom_grid_points: tuple[int, ...] = DEFAULT_ATOM_GRID_POINTS,
    dense_points: int = DEFAULT_DENSE_POINTS,
    metadata_limit: int | None = None,
    active_only: bool = True,
    chunk_size: int = 512,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    if progress:
        progress(
            "grid-ceiling: loading metadata "
            f"dense_points={dense_points} active_only={active_only} metadata_limit={metadata_limit}"
        )
    dataset = load_presetshare_curve_dataset(
        metadata_path,
        resolution=dense_points,
        active_only=active_only,
        metadata_limit=metadata_limit,
        progress=progress,
    )
    if not dataset.shapes:
        raise ValueError("grid ceiling audit requires source LFO shapes")
    if progress:
        progress(f"grid-ceiling: loaded {len(dataset.shapes)} LFO shapes")
    reference_dense = dataset.curves.astype(np.float32)
    results = []
    for index, points in enumerate(atom_grid_points, start=1):
        if progress:
            progress(f"grid-ceiling: [{index}/{len(atom_grid_points)}] N={points} starting")
        row_started = time.perf_counter()
        reconstructed = best_fixed_grid_reconstruction(
            dataset.shapes,
            atom_grid_points=points,
            dense_points=dense_points,
            reference_dense=reference_dense,
            chunk_size=chunk_size,
            progress=progress,
        )
        metrics = reconstruction_summary(reference_dense, reconstructed)
        metrics.update(_topology_metrics(reference_dense, reconstructed, dataset.topology))
        results.append(
            GridCeilingResult(
                atom_grid_points=points,
                dense_points=dense_points,
                row_count=len(reference_dense),
                elapsed_seconds=time.perf_counter() - row_started,
                metrics=metrics,
            )
        )
        if progress:
            progress(
                "grid-ceiling: "
                f"[{index}/{len(atom_grid_points)}] N={points} done "
                f"elapsed={results[-1].elapsed_seconds:.1f}s p95_rmse={metrics['p95_rmse']:.8f}"
            )
    rows = [result.as_row() for result in results]
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "summary.csv", rows)
    report = _report(rows)
    (output_dir / "GRID_CEILING_REPORT.md").write_text(report, encoding="utf-8")
    manifest = {
        "audit_id": "grid_ceiling",
        "atom_grid_points": list(atom_grid_points),
        "dense_points": dense_points,
        "metadata_path": str(metadata_path),
        "metadata_limit": metadata_limit,
        "active_only": active_only,
        "dataset": dataset.manifest_fields(),
        "elapsed_seconds": time.perf_counter() - started,
        "method": "bounded_least_squares_projection_onto_inclusive_fixed_linear_point_grid",
    }
    write_json(output_dir / "manifest.json", manifest)
    if progress:
        progress(f"grid-ceiling: wrote summary/report to {output_dir}")
    return {
        "output_dir": str(output_dir),
        "summary": str(output_dir / "summary.csv"),
        "report": str(output_dir / "GRID_CEILING_REPORT.md"),
        "manifest": manifest,
        "rows": rows,
    }


def best_fixed_grid_reconstruction(
    shapes: tuple[LfoShape, ...] | list[LfoShape],
    *,
    atom_grid_points: int,
    dense_points: int,
    reference_dense: np.ndarray | None = None,
    chunk_size: int = 512,
    progress: Callable[[str], None] | None = None,
) -> np.ndarray:
    if atom_grid_points < 2:
        raise ValueError("atom_grid_points must be at least 2")
    if dense_points < 2:
        raise ValueError("dense_points must be at least 2")
    basis = fixed_grid_basis(atom_grid_points, dense_points)
    gram = basis.T @ basis
    pinv = np.linalg.pinv(gram) @ basis.T
    if reference_dense is None:
        reference_dense = np.stack([sample_shape(shape, dense_points) for shape in shapes]).astype(np.float32)
    dense = np.asarray(reference_dense, dtype=np.float32)
    output = np.empty_like(dense)
    chunk_size = max(1, int(chunk_size))
    if progress:
        progress(
            f"grid-ceiling: N={atom_grid_points} projecting {len(dense)} curves "
            f"in chunks of {chunk_size}"
        )
    for start in range(0, len(dense), chunk_size):
        stop = min(start + chunk_size, len(dense))
        values = dense[start:stop].astype(np.float64)
        grid_values = values @ pinv.T
        invalid_rows = np.flatnonzero((np.min(grid_values, axis=1) < 0.0) | (np.max(grid_values, axis=1) > 1.0))
        if len(invalid_rows):
            if progress:
                progress(
                    f"grid-ceiling: N={atom_grid_points} rows {start}-{stop} "
                    f"bounded least squares for {len(invalid_rows)} curves"
                )
            grid_values[invalid_rows] = _bounded_grid_values(
                basis,
                values[invalid_rows],
                initial=np.clip(grid_values[invalid_rows], 0.0, 1.0),
            )
        reconstructed = grid_values @ basis.T
        output[start:stop] = reconstructed.astype(np.float32)
    return output


def fixed_grid_basis(atom_grid_points: int, dense_points: int) -> np.ndarray:
    grid_x = np.linspace(0.0, 1.0, atom_grid_points, dtype=np.float64)
    dense_x = np.arange(dense_points, dtype=np.float64) / float(dense_points)
    basis = np.empty((dense_points, atom_grid_points), dtype=np.float64)
    basis.fill(0.0)
    right = np.searchsorted(grid_x, dense_x, side="right")
    right = np.clip(right, 1, atom_grid_points - 1)
    left = right - 1
    width = grid_x[right] - grid_x[left]
    frac = np.divide(dense_x - grid_x[left], width, out=np.zeros_like(dense_x), where=width > 0.0)
    rows = np.arange(dense_points)
    basis[rows, left] += 1.0 - frac
    basis[rows, right] += frac
    return basis


def _topology_metrics(reference: np.ndarray, reconstructed: np.ndarray, topology: np.ndarray) -> dict[str, float]:
    rmse = rmse_per_curve(reference, reconstructed)
    result: dict[str, float] = {}
    for index, name in enumerate(TOPOLOGY_NAMES):
        values = rmse[topology == index]
        if len(values):
            result[f"analysis_only_topology_{name}_median_rmse"] = float(np.median(values))
            result[f"analysis_only_topology_{name}_p95_rmse"] = float(np.quantile(values, 0.95))
    p95s = [value for key, value in result.items() if key.endswith("_p95_rmse")]
    if p95s:
        result["analysis_only_topology_p95_gap"] = float(max(p95s) - min(p95s))
    return result


def _bounded_grid_values(basis: np.ndarray, values: np.ndarray, *, initial: np.ndarray) -> np.ndarray:
    try:
        from scipy.optimize import lsq_linear
    except Exception:
        return initial
    solved = np.empty((len(values), basis.shape[1]), dtype=np.float64)
    for row, target in enumerate(values):
        result = lsq_linear(basis, target, bounds=(0.0, 1.0), method="trf", lsmr_tol="auto")
        solved[row] = result.x if result.success else initial[row]
    return solved


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


def _report(rows: list[dict[str, Any]]) -> str:
    ordered = sorted(rows, key=lambda row: int(row["atom_grid_points"]))
    lines = [
        "# Atom Grid Ceiling Audit",
        "",
        "This estimates the best dense reconstruction possible if each atom is an inclusive fixed-grid LFO point vector, final point values stay in `[0, 1]`, and the decoder renders that grid with linear interpolation.",
        "",
        "| atom grid points | dense points | median RMSE | p95 RMSE | p99 RMSE | max RMSE |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ordered:
        lines.append(
            "| {atom_grid_points} | {dense_points} | {median_rmse:.8f} | {p95_rmse:.8f} | {p99_rmse:.8f} | {max_rmse:.8f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- This is a ceiling audit, not an Experiment 10 model-budget row.",
            "- `atom_grid_points` changes atom dimensionality and Vital point-grid fidelity, not model prediction head budget.",
            "- If 96 is close to 100, 96 is the cleaner default because it stays under Vital's 100-point limit and has better subdivision factors.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_grid_points(value: str) -> tuple[int, ...]:
    points = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not points:
        raise ValueError("at least one atom grid point count is required")
    return points

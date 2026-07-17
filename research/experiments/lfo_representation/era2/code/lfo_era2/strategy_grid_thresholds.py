"""Validation-only strict-perfect threshold replay for Experiment 13 reports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from . import component_ladder as x12
from .assets import ReconstructionAssets
from .dataset import Era2CurveDataset
from .metrics import max_abs_error_per_curve, rmse_per_curve
from .strategy_grid_execution import load_dataset_cached
from .strategy_grid_runtime import component_spec, read_csv, write_csv


STRICT_THRESHOLD_SCHEMA = "experiment13_strict_threshold_sweep_v1"
STRICT_TOLERANCES = (1e-2, 1e-3, 1e-4, 1e-5)


def replay_strict_perfect_thresholds(
    *,
    run_dir: Path,
    output_dir: Path,
    metadata_path: Path,
    cache_dir: Path | None,
    backend: str = "auto",
    chunk_size: int = 256,
    tolerances: Sequence[float] = STRICT_TOLERANCES,
    progress: Callable[[str], None] | None = None,
) -> dict[str, str]:
    """Replay saved codebooks on validation only and persist exact threshold rates.

    This never reconstructs dictionaries or writes inside the completed source run.
    The one-parameter tolerance preserves the original strict-perfect definition:
    ``rmse <= tolerance / 10`` and ``max_abs <= tolerance``.
    """
    from .strategy_grid import experiment13a_specs

    run_dir = Path(run_dir).resolve()
    output_dir = Path(output_dir).resolve()
    metadata_path = Path(metadata_path).resolve()
    if not run_dir.is_dir():
        raise ValueError(f"threshold replay source run does not exist: {run_dir}")
    _require_outside_source(run_dir, output_dir)
    tolerances = tuple(sorted({float(value) for value in tolerances}, reverse=True))
    if not tolerances or any(not np.isfinite(value) or value <= 0 for value in tolerances):
        raise ValueError("strict-perfect tolerances must be finite positive values")

    summaries = {str(row["row_id"]): row for row in read_csv(run_dir / "summary.csv") if row.get("experiment_phase") == "13A"}
    specs = {spec.row_id: spec for spec in experiment13a_specs()}
    if len(summaries) != 90 or set(summaries) != set(specs):
        raise ValueError(f"threshold replay requires the complete 90-row 13A grid; found {len(summaries)} rows")

    first = next(iter(summaries.values()))
    dataset, dataset_cache = load_dataset_cached(
        metadata_path,
        cache_dir=Path(cache_dir).resolve() if cache_dir is not None else None,
        resolution=int(float(first["resolution"])),
        x_grid_mode=str(first["x_grid_mode"]),
        progress=progress,
    )
    sampled = _sampled_dataset(run_dir, dataset)
    validation = sampled.validation_curves
    rows: list[dict[str, Any]] = []
    replay_checks: list[dict[str, Any]] = []
    for index, row_id in enumerate(sorted(specs), start=1):
        row_dir = run_dir / "rows" / row_id
        archive_path = row_dir / "codebooks.npz"
        if not archive_path.is_file():
            raise ValueError(f"saved codebook is missing for {row_id}: {archive_path}")
        with np.load(archive_path, allow_pickle=False) as archive:
            base = np.asarray(archive["base_dictionary"], dtype=np.float32)
            dictionaries = [np.asarray(archive[f"residual_layer_{layer}"], dtype=np.float32) for layer in range(1, 17)]
        assets = ReconstructionAssets(base, dictionaries, metadata={"experiment_id": "experiment_13", "row_id": row_id})
        runtime_spec = component_spec(specs[row_id])
        _, reconstructed, _, _ = x12._encode_decode(
            runtime_spec,
            validation,
            assets,
            backend=backend,
            chunk_size=chunk_size,
            progress=None,
            progress_label="strict_threshold_validation_replay",
        )
        rmse = rmse_per_curve(validation, reconstructed).astype(np.float64)
        max_abs = max_abs_error_per_curve(validation, reconstructed).astype(np.float64)
        summary = summaries[row_id]
        replay_checks.append(_validate_replay(row_id, summary, rmse, max_abs))
        for tolerance in tolerances:
            rmse_tolerance = tolerance / 10.0
            qualifying = int(np.sum((rmse <= rmse_tolerance) & (max_abs <= tolerance)))
            rows.append({
                "schema_version": STRICT_THRESHOLD_SCHEMA,
                "row_id": row_id,
                "dataset_split": "validation",
                "max_abs_tolerance": tolerance,
                "rmse_tolerance": rmse_tolerance,
                "strict_perfect_lfo_count": qualifying,
                "row_count": len(rmse),
                "strict_perfect_lfo_rate": qualifying / len(rmse),
            })
        if progress:
            progress(f"strict-threshold replay {index}/90 {row_id}")

    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "strict_perfect_threshold_sweep.csv"
    write_csv(table_path, rows)
    manifest = {
        "schema_version": STRICT_THRESHOLD_SCHEMA,
        "source_run": str(run_dir),
        "source_summary_sha256": hashlib.sha256((run_dir / "summary.csv").read_bytes()).hexdigest(),
        "sample_manifest_sha256": hashlib.sha256((run_dir / "sample_manifest.json").read_bytes()).hexdigest(),
        "validation_indices_sha256": _array_sha256(sampled.validation_indices),
        "row_count": len(summaries),
        "validation_curve_count": len(validation),
        "tolerances": list(tolerances),
        "definition": "rmse <= max_abs_tolerance / 10 and max_abs_error <= max_abs_tolerance",
        "default_tolerance": 1e-5,
        "backend": backend,
        "chunk_size": int(chunk_size),
        "dataset_cache": dataset_cache,
        "replay_checks": replay_checks,
        "table_path": str(table_path),
        "table_sha256": hashlib.sha256(table_path.read_bytes()).hexdigest(),
    }
    manifest_path = output_dir / "strict_perfect_threshold_sweep_manifest.json"
    _atomic_json(manifest_path, manifest)
    return {"strict_thresholds": str(table_path), "strict_threshold_manifest": str(manifest_path)}


def load_strict_threshold_sweep(path: Path, *, expected_row_ids: Sequence[str]) -> dict[str, Any]:
    """Validate and compact a threshold sweep for the interactive report."""
    path = Path(path).resolve()
    rows = read_csv(path)
    expected = set(expected_row_ids)
    if not rows or {str(row.get("row_id", "")) for row in rows} != expected:
        raise ValueError("strict threshold sweep row ids do not match the report rows")
    tolerances = sorted({float(row["max_abs_tolerance"]) for row in rows}, reverse=True)
    if tolerances != list(STRICT_TOLERANCES):
        raise ValueError(f"strict threshold sweep must contain {STRICT_TOLERANCES}")
    grouped: dict[str, dict[str, float]] = {row_id: {} for row_id in expected}
    counts: dict[str, dict[str, int]] = {row_id: {} for row_id in expected}
    for row in rows:
        if row.get("schema_version") != STRICT_THRESHOLD_SCHEMA or row.get("dataset_split") != "validation":
            raise ValueError("strict threshold sweep schema or dataset split is invalid")
        key = _tolerance_key(float(row["max_abs_tolerance"]))
        row_id = str(row["row_id"])
        grouped[row_id][key] = float(row["strict_perfect_lfo_rate"])
        counts[row_id][key] = int(float(row["strict_perfect_lfo_count"]))
    if any(len(values) != len(STRICT_TOLERANCES) for values in grouped.values()):
        raise ValueError("strict threshold sweep is incomplete")
    return {
        "schema_version": STRICT_THRESHOLD_SCHEMA,
        "default_tolerance": "1e-5",
        "tolerances": [_tolerance_key(value) for value in STRICT_TOLERANCES],
        "definition": "RMSE <= tolerance / 10 and maximum absolute point error <= tolerance",
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rates_by_row": grouped,
        "counts_by_row": counts,
    }


def _sampled_dataset(run_dir: Path, dataset: Era2CurveDataset) -> Era2CurveDataset:
    sample_path = run_dir / "sample_indices.npz"
    manifest_path = run_dir / "sample_manifest.json"
    if not sample_path.is_file() or not manifest_path.is_file():
        raise ValueError("threshold replay requires sample_indices.npz and sample_manifest.json")
    with np.load(sample_path, allow_pickle=False) as archive:
        train = np.asarray(archive["train_indices"], dtype=np.int32)
        validation = np.asarray(archive["validation_indices"], dtype=np.int32)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, values in (("train", train), ("validation", validation)):
        expected = str(manifest[f"{name}_indices_sha256"])
        if _array_sha256(values) != expected:
            raise ValueError(f"saved {name} indices do not match sample manifest")
    if str(manifest["source_fingerprint"]) != dataset.source_fingerprint:
        raise ValueError("dataset source fingerprint does not match the completed run")
    return Era2CurveDataset(
        curves=dataset.curves,
        topology=dataset.topology,
        train_indices=train,
        validation_indices=validation,
        row_metadata=dataset.row_metadata,
        source_fingerprint=dataset.source_fingerprint,
        resolution=dataset.resolution,
        x_grid_mode=dataset.x_grid_mode,
        shapes=dataset.shapes,
    )


def _validate_replay(row_id: str, summary: dict[str, str], rmse: np.ndarray, max_abs: np.ndarray) -> dict[str, Any]:
    median_delta = abs(float(np.median(rmse)) - float(summary["validation_median_rmse"]))
    p95_delta = abs(float(np.quantile(rmse, 0.95)) - float(summary["validation_p95_rmse"]))
    current_rate = float(np.mean((rmse <= 1e-6) & (max_abs <= 1e-5)))
    rate_delta = abs(current_rate - float(summary["validation_strict_perfect_lfo_rate"]))
    if median_delta > 2e-7 or p95_delta > 2e-7 or rate_delta > 1e-12:
        raise ValueError(
            f"validation replay for {row_id} does not match saved metrics: "
            f"median_delta={median_delta} p95_delta={p95_delta} strict_rate_delta={rate_delta}"
        )
    return {"row_id": row_id, "median_delta": median_delta, "p95_delta": p95_delta, "strict_rate_delta": rate_delta}


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _tolerance_key(value: float) -> str:
    return f"1e-{int(round(-np.log10(value)))}"


def _require_outside_source(source: Path, output: Path) -> None:
    try:
        output.relative_to(source)
    except ValueError:
        return
    raise ValueError("strict threshold replay output directory must be outside the source run")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)

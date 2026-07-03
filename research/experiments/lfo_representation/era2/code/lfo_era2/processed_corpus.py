"""Processed PresetShare LFO corpus cache for Era 2."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Any, Callable
import uuid

import numpy as np

from .dataset import (
    Era2CurveDataset,
    LfoShape,
    _author_is_validation,
    _author_key,
    _bool,
    _routes_by_lfo,
    _topology,
    sample_shape,
)


ERA2_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ERA2_ROOT.parents[3]
DEFAULT_METADATA = REPO_ROOT / "datasets" / "presetshare" / "raw" / "presetshare_vital_metadata.csv"
DEFAULT_CORPUS_DIR = REPO_ROOT / "datasets" / "presetshare" / "processed" / "lfo_corpus_v1"
DEFAULT_DENSE_RESOLUTION = 1920


@dataclass(frozen=True)
class ProcessedShapeCorpus:
    shapes: tuple[LfoShape, ...]
    curves: np.ndarray
    occurrence_count: np.ndarray
    active_occurrence_count: np.ndarray
    topology: np.ndarray
    manifest: dict[str, Any]
    resolution: int

    @property
    def weights(self) -> np.ndarray:
        return self.active_occurrence_count


def build_lfo_corpus(
    *,
    metadata_path: Path = DEFAULT_METADATA,
    output_dir: Path = DEFAULT_CORPUS_DIR,
    dense_resolution: int = DEFAULT_DENSE_RESOLUTION,
    force: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    metadata_path = Path(metadata_path)
    output_dir = Path(output_dir)
    if output_dir.exists() and not force:
        raise FileExistsError(f"processed LFO corpus already exists; pass force=True to rebuild: {output_dir}")
    temp_dir = output_dir.with_name(f"{output_dir.name}.tmp_{uuid.uuid4().hex[:8]}")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)

    rows: list[dict[str, Any]] = []
    shape_records: dict[str, dict[str, Any]] = {}
    shape_order: list[str] = []
    errors = 0
    total_metadata_rows = _metadata_row_total(metadata_path)
    if progress:
        progress(f"build-lfo-corpus: scanning metadata rows={total_metadata_rows}")

    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for ordinal, record in enumerate(reader, start=1):
            if progress and (ordinal == 1 or ordinal % 500 == 0):
                progress(
                    "build-lfo-corpus: "
                    f"{_percent(ordinal, total_metadata_rows)} metadata rows={ordinal}/{total_metadata_rows} "
                    f"lfo_rows={len(rows)} unique_shapes={len(shape_order)} errors={errors}"
                )
            preset_file = record.get("preset_file", "")
            if not preset_file:
                continue
            preset_path = metadata_path.parent / Path(preset_file.replace("\\", "/"))
            try:
                preset = json.loads(preset_path.read_text(encoding="utf-8"))
                settings = preset["settings"]
                lfos = settings.get("lfos", [])
            except Exception:
                errors += 1
                continue

            routes_by_lfo = _routes_by_lfo(settings)
            for lfo_index, routes in sorted(routes_by_lfo.items()):
                if not (1 <= lfo_index <= len(lfos)) or not isinstance(lfos[lfo_index - 1], dict):
                    continue
                try:
                    shape = LfoShape.from_json(lfos[lfo_index - 1])
                except Exception:
                    errors += 1
                    continue
                signature = shape.signature()
                materially_active = any(route["materially_active"] for route in routes)
                if signature not in shape_records:
                    shape_order.append(signature)
                    shape_records[signature] = _shape_record(
                        shape_id=len(shape_order) - 1,
                        signature=signature,
                        shape=shape,
                    )
                shape_records[signature]["occurrence_count"] += 1
                if materially_active:
                    shape_records[signature]["active_occurrence_count"] += 1
                rows.append(
                    {
                        "row_id": len(rows),
                        "preset_id": record.get("preset_id", ""),
                        "author_id": record.get("author_id", ""),
                        "author": record.get("author", ""),
                        "title": record.get("title", ""),
                        "genre": record.get("genre", ""),
                        "type": record.get("type", ""),
                        "lfo_index": int(lfo_index),
                        "shape_id": shape_records[signature]["shape_id"],
                        "shape_signature": signature,
                        "shape_name": shape.name,
                        "is_materially_active": bool(materially_active),
                        "active_route_count": int(sum(route["materially_active"] for route in routes)),
                        "route_count": int(len(routes)),
                        "is_validation": bool(_author_is_validation(_author_key(record))),
                    }
                )

    if not rows:
        raise ValueError(f"no routed LFO rows found in {metadata_path}")
    shapes = [_shape_from_record(shape_records[signature]) for signature in shape_order]
    if progress:
        progress(f"build-lfo-corpus: sampling {len(shapes)} unique shapes at resolution={dense_resolution}")
    curves = np.stack([sample_shape(shape, resolution=dense_resolution) for shape in shapes]).astype(np.float32)

    shape_rows = [shape_records[signature] for signature in shape_order]
    row_shape_ids = np.asarray([row["shape_id"] for row in rows], dtype=np.int32)
    row_topology = np.asarray([shape_rows[row["shape_id"]]["topology_analysis_id"] for row in rows], dtype=np.int8)
    row_is_active = np.asarray([row["is_materially_active"] for row in rows], dtype=np.bool_)
    row_is_validation = np.asarray([row["is_validation"] for row in rows], dtype=np.bool_)
    active_indices = np.flatnonzero(row_is_active)
    active_validation_mask = row_is_validation[active_indices]
    train_indices = np.flatnonzero(~active_validation_mask).astype(np.int32)
    validation_indices = np.flatnonzero(active_validation_mask).astype(np.int32)

    _write_jsonl(temp_dir / "rows.jsonl", rows)
    _write_jsonl(temp_dir / "shapes.jsonl", shape_rows)
    np.save(temp_dir / "row_shape_ids.npy", row_shape_ids)
    np.save(temp_dir / "row_topology_analysis.npy", row_topology)
    np.save(temp_dir / "row_is_active.npy", row_is_active)
    np.save(temp_dir / "row_is_validation.npy", row_is_validation)
    np.save(temp_dir / "train_indices.npy", train_indices)
    np.save(temp_dir / "validation_indices.npy", validation_indices)
    np.save(temp_dir / f"curves_r{dense_resolution}_f32.npy", curves)

    manifest = {
        "corpus_id": "presetshare_lfo_corpus_v1",
        "builder_version": 1,
        "built_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source_metadata_path": str(metadata_path),
        "source_fingerprint": _source_fingerprint(metadata_path),
        "dense_resolution": int(dense_resolution),
        "available_resolutions": [int(dense_resolution)],
        "row_count": int(len(rows)),
        "active_row_count": int(np.sum(row_is_active)),
        "unique_shape_count": int(len(shape_rows)),
        "errors": int(errors),
        "topology_labels": ["smooth", "continuous", "discontinuous"],
        "contract_note": "Topology labels are analysis-only and must not enter runtime targets, losses, decoder lookup, or model prediction head accounting.",
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(temp_dir / "manifest.json", manifest)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir.rename(output_dir)
    if progress:
        progress(
            "build-lfo-corpus: done "
            f"rows={manifest['row_count']} active_rows={manifest['active_row_count']} "
            f"unique_shapes={manifest['unique_shape_count']} output={output_dir}"
        )
    return {"output_dir": str(output_dir), "manifest": manifest}


def load_processed_lfo_corpus(
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    *,
    resolution: int = DEFAULT_DENSE_RESOLUTION,
    active_only: bool = True,
    mmap: bool = True,
) -> Era2CurveDataset:
    corpus_dir = Path(corpus_dir)
    manifest = _read_json(corpus_dir / "manifest.json")
    rows = _read_jsonl(corpus_dir / "rows.jsonl")
    shape_records = _read_jsonl(corpus_dir / "shapes.jsonl")
    row_shape_ids = np.load(corpus_dir / "row_shape_ids.npy")
    row_topology = np.load(corpus_dir / "row_topology_analysis.npy")
    row_is_active = np.load(corpus_dir / "row_is_active.npy")
    row_is_validation = np.load(corpus_dir / "row_is_validation.npy")
    mask = row_is_active if active_only else np.ones(len(row_shape_ids), dtype=np.bool_)
    original_indices = np.flatnonzero(mask)
    curves_by_shape = _load_or_sample_curves(corpus_dir, shape_records, resolution=resolution, mmap=mmap)
    filtered_shape_ids = row_shape_ids[original_indices]
    curves = np.asarray(curves_by_shape[filtered_shape_ids], dtype=np.float32)
    validation_mask = row_is_validation[original_indices]
    shapes = tuple(_shape_from_record(shape_records[int(shape_id)]) for shape_id in filtered_shape_ids)
    metadata = [rows[int(index)] for index in original_indices]
    fingerprint = f"{manifest.get('source_fingerprint', '')}:processed:{resolution}:active={active_only}"
    return Era2CurveDataset(
        curves=curves,
        topology=np.asarray(row_topology[original_indices], dtype=np.int8),
        train_indices=np.flatnonzero(~validation_mask).astype(np.int32),
        validation_indices=np.flatnonzero(validation_mask).astype(np.int32),
        row_metadata=metadata,
        source_fingerprint=fingerprint,
        resolution=int(resolution),
        shapes=shapes,
    )


def load_processed_shape_corpus(
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    *,
    resolution: int = DEFAULT_DENSE_RESOLUTION,
    active_only: bool = True,
    mmap: bool = True,
) -> ProcessedShapeCorpus:
    corpus_dir = Path(corpus_dir)
    manifest = _read_json(corpus_dir / "manifest.json")
    records = _read_jsonl(corpus_dir / "shapes.jsonl")
    curves = _load_or_sample_curves(corpus_dir, records, resolution=resolution, mmap=mmap)
    occurrence = np.asarray([record["occurrence_count"] for record in records], dtype=np.int32)
    active_occurrence = np.asarray([record["active_occurrence_count"] for record in records], dtype=np.int32)
    topology = np.asarray([record["topology_analysis_id"] for record in records], dtype=np.int8)
    mask = active_occurrence > 0 if active_only else np.ones(len(records), dtype=np.bool_)
    shapes = tuple(_shape_from_record(record) for record, keep in zip(records, mask) if keep)
    return ProcessedShapeCorpus(
        shapes=shapes,
        curves=np.asarray(curves[mask], dtype=np.float32),
        occurrence_count=occurrence[mask],
        active_occurrence_count=active_occurrence[mask],
        topology=topology[mask],
        manifest=manifest,
        resolution=int(resolution),
    )


def _shape_record(*, shape_id: int, signature: str, shape: LfoShape) -> dict[str, Any]:
    topology_id = _topology(shape)
    return {
        "shape_id": int(shape_id),
        "shape_signature": signature,
        "name": shape.name,
        "num_points": int(len(shape.points)),
        "points": shape.points.tolist(),
        "powers": shape.powers.tolist(),
        "smooth": bool(shape.smooth),
        "topology_analysis_id": int(topology_id),
        "topology_analysis_label": ["smooth", "continuous", "discontinuous"][topology_id],
        "occurrence_count": 0,
        "active_occurrence_count": 0,
    }


def _shape_from_record(record: dict[str, Any]) -> LfoShape:
    return LfoShape(
        name=str(record.get("name", "")),
        points=np.asarray(record["points"], dtype=np.float64),
        powers=np.asarray(record["powers"], dtype=np.float64),
        smooth=_bool(record.get("smooth", False)),
    )


def _load_or_sample_curves(
    corpus_dir: Path,
    shape_records: list[dict[str, Any]],
    *,
    resolution: int,
    mmap: bool,
) -> np.ndarray:
    path = corpus_dir / f"curves_r{resolution}_f32.npy"
    if path.exists():
        return np.load(path, mmap_mode="r" if mmap else None)
    shapes = [_shape_from_record(record) for record in shape_records]
    return np.stack([sample_shape(shape, resolution=resolution) for shape in shapes]).astype(np.float32)


def _metadata_row_total(metadata_path: Path) -> int:
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def _source_fingerprint(metadata_path: Path) -> str:
    stat = metadata_path.stat()
    payload = {
        "metadata_name": metadata_path.name,
        "metadata_size": stat.st_size,
        "metadata_mtime_ns": stat.st_mtime_ns,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _percent(done: int, total: int) -> str:
    if total <= 0:
        return "0.0%"
    return f"{100.0 * min(done, total) / total:.1f}%"

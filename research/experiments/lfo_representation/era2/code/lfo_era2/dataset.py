"""Era 2 LFO dataset ingestion.

This module intentionally ports only the small amount of Vital LFO parsing and
sampling needed for Era 2 experiments. It does not depend on Era 1 experiment
code.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


TOPOLOGY_NAMES = ("smooth", "continuous", "discontinuous")


@dataclass(frozen=True)
class LfoShape:
    name: str
    points: np.ndarray
    powers: np.ndarray
    smooth: bool

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "LfoShape":
        raw_points = np.asarray(value.get("points", []), dtype=np.float64)
        if raw_points.ndim != 1 or raw_points.size % 2:
            raise ValueError("LFO points must be a flat sequence of x/y pairs")
        points = raw_points.reshape(-1, 2)
        declared = int(value.get("num_points", len(points)))
        if declared != len(points):
            raise ValueError("num_points does not match coordinate pair count")
        if len(points) < 2:
            raise ValueError("an LFO shape needs at least two points")
        if np.any(np.diff(points[:, 0]) < 0.0):
            raise ValueError("LFO x coordinates must be non-decreasing")

        powers = np.asarray(value.get("powers", []), dtype=np.float64)
        if powers.shape != (len(points),):
            raise ValueError("LFO powers must contain one value per point")
        return cls(
            name=str(value.get("name", "")),
            points=points,
            powers=powers,
            smooth=_bool(value.get("smooth", False)),
        )

    def signature(self) -> str:
        payload = json.dumps(
            {
                "num_points": len(self.points),
                "points": self.points.reshape(-1).tolist(),
                "powers": self.powers.tolist(),
                "smooth": self.smooth,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class Era2CurveDataset:
    curves: np.ndarray
    topology: np.ndarray
    train_indices: np.ndarray
    validation_indices: np.ndarray
    row_metadata: list[dict[str, Any]]
    source_fingerprint: str
    resolution: int
    shapes: tuple[LfoShape, ...] = ()

    @property
    def train_curves(self) -> np.ndarray:
        return self.curves[self.train_indices]

    @property
    def validation_curves(self) -> np.ndarray:
        return self.curves[self.validation_indices]

    def subset(self, *, train_count: int | None = None, validation_count: int | None = None) -> "Era2CurveDataset":
        train = self.train_indices
        validation = self.validation_indices
        if train_count is not None:
            train = train[: max(0, int(train_count))]
        if validation_count is not None:
            validation = validation[: max(0, int(validation_count))]
        return Era2CurveDataset(
            curves=self.curves,
            topology=self.topology,
            train_indices=train.copy(),
            validation_indices=validation.copy(),
            row_metadata=self.row_metadata,
            source_fingerprint=self.source_fingerprint,
            resolution=self.resolution,
            shapes=self.shapes,
        )

    def manifest_fields(self) -> dict[str, Any]:
        return {
            "dataset_fingerprint": self.source_fingerprint,
            "dataset_row_count": int(len(self.curves)),
            "train_count": int(len(self.train_indices)),
            "validation_count": int(len(self.validation_indices)),
            "resolution": int(self.resolution),
        }


def load_presetshare_curve_dataset(
    metadata_path: Path,
    *,
    resolution: int = 128,
    active_only: bool = True,
    metadata_limit: int | None = None,
) -> Era2CurveDataset:
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing metadata CSV: {metadata_path}")
    rows: list[dict[str, Any]] = []
    curves: list[np.ndarray] = []
    topology: list[int] = []
    shapes: list[LfoShape] = []
    errors = 0
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for ordinal, record in enumerate(reader, start=1):
            if metadata_limit is not None and ordinal > metadata_limit:
                break
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
            for lfo_index, routes in routes_by_lfo.items():
                if not (1 <= lfo_index <= len(lfos)) or not isinstance(lfos[lfo_index - 1], dict):
                    continue
                if active_only and not any(route["materially_active"] for route in routes):
                    continue
                try:
                    shape = LfoShape.from_json(lfos[lfo_index - 1])
                    curve = sample_shape(shape, resolution=resolution).astype(np.float32)
                except Exception:
                    errors += 1
                    continue
                rows.append(
                    {
                        "preset_id": record.get("preset_id", ""),
                        "author_id": record.get("author_id", ""),
                        "author": record.get("author", ""),
                        "lfo_index": int(lfo_index),
                        "shape_name": shape.name,
                        "shape_signature": shape.signature(),
                        "active_route_count": sum(route["materially_active"] for route in routes),
                    }
                )
                curves.append(curve)
                topology.append(_topology(shape))
                shapes.append(shape)
    if not curves:
        raise ValueError(f"no usable LFO curves found in {metadata_path}")

    curve_array = np.stack(curves).astype(np.float32)
    topology_array = np.asarray(topology, dtype=np.int8)
    validation_mask = np.asarray([_author_is_validation(_author_key(row)) for row in rows])
    if not np.any(validation_mask) or np.all(validation_mask):
        raise ValueError("deterministic author split produced an empty partition")
    fingerprint = _dataset_fingerprint(metadata_path, rows, resolution=resolution, errors=errors)
    return Era2CurveDataset(
        curves=curve_array,
        topology=topology_array,
        train_indices=np.flatnonzero(~validation_mask).astype(np.int32),
        validation_indices=np.flatnonzero(validation_mask).astype(np.int32),
        row_metadata=rows,
        source_fingerprint=fingerprint,
        resolution=int(resolution),
        shapes=tuple(shapes),
    )


def sample_shape(shape: LfoShape, resolution: int = 128) -> np.ndarray:
    if resolution < 2:
        raise ValueError("resolution must be at least 2")
    phase = np.arange(resolution, dtype=np.float64) / resolution
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


def power_scale(t: np.ndarray, power: np.ndarray) -> np.ndarray:
    t, power = np.broadcast_arrays(np.asarray(t, dtype=np.float64), np.asarray(power, dtype=np.float64))
    result = t.copy()
    curved = np.abs(power) >= 0.01
    if np.any(curved):
        numerator = np.expm1(power[curved] * t[curved])
        denominator = np.expm1(power[curved])
        result[curved] = numerator / denominator
    return np.clip(result, 0.0, 1.0)


def make_tiny_curve_dataset(*, resolution: int = 32, row_count: int = 24) -> Era2CurveDataset:
    x = np.arange(resolution, dtype=np.float32) / float(resolution)
    curves = []
    topology = []
    rows = []
    for index in range(row_count):
        if index % 3 == 0:
            curve = 0.5 + 0.45 * np.sin(2.0 * np.pi * x + index * 0.17)
            topo = 0
        elif index % 3 == 1:
            curve = np.where(x < ((index % 7) + 1) / 8.0, x, 1.0 - x)
            topo = 1
        else:
            curve = (x > ((index % 5) + 1) / 6.0).astype(np.float32)
            topo = 2
        curves.append(np.clip(curve, 0.0, 1.0).astype(np.float32))
        topology.append(topo)
        rows.append({"author_id": f"author_{index % 5}", "preset_id": f"tiny_{index}", "lfo_index": 1})
    train = np.asarray([i for i in range(row_count) if i % 5 != 0], dtype=np.int32)
    validation = np.asarray([i for i in range(row_count) if i % 5 == 0], dtype=np.int32)
    return Era2CurveDataset(
        curves=np.stack(curves).astype(np.float32),
        topology=np.asarray(topology, dtype=np.int8),
        train_indices=train,
        validation_indices=validation,
        row_metadata=rows,
        source_fingerprint=f"tiny_{row_count}_{resolution}",
        resolution=resolution,
        shapes=(),
    )


def _routes_by_lfo(settings: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    routes: dict[int, list[dict[str, Any]]] = {}
    for slot, route in enumerate(settings.get("modulations", []), start=1):
        if not isinstance(route, dict):
            continue
        source = str(route.get("source", ""))
        destination = str(route.get("destination", ""))
        if not destination or not source.startswith("lfo_"):
            continue
        suffix = source[4:]
        if not suffix.isdigit():
            continue
        amount = float(settings.get(f"modulation_{slot}_amount", 0.0) or 0.0)
        bypass = bool(float(settings.get(f"modulation_{slot}_bypass", 0.0) or 0.0))
        routes.setdefault(int(suffix), []).append(
            {
                "slot": slot,
                "destination": destination,
                "amount": amount,
                "bypass": bypass,
                "materially_active": (not bypass and abs(amount) > 1e-8),
            }
        )
    return routes


def _topology(shape: LfoShape) -> int:
    if shape.smooth:
        return 0
    x = shape.points[:, 0]
    return 2 if np.any(np.diff(x) == 0.0) else 1


def _author_key(row: dict[str, Any]) -> str:
    return str(row.get("author_id") or row.get("author") or row.get("preset_id") or "")


def _author_is_validation(key: str) -> bool:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 100 < 20


def _dataset_fingerprint(metadata_path: Path, rows: list[dict[str, Any]], *, resolution: int, errors: int) -> str:
    stat = metadata_path.stat()
    payload = {
        "metadata_name": metadata_path.name,
        "metadata_size": stat.st_size,
        "metadata_mtime_ns": stat.st_mtime_ns,
        "row_count": len(rows),
        "resolution": resolution,
        "errors": errors,
        "first_signatures": [row.get("shape_signature", "") for row in rows[:64]],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)

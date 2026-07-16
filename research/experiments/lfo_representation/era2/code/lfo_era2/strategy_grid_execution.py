"""Execution safety, deterministic sampling, and reusable caches for Experiment 13."""

from __future__ import annotations

from contextlib import contextmanager
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, Iterator

import numpy as np

from . import component_ladder as x12
from .alignment import AlignmentChoice, best_alignment
from .dataset import Era2CurveDataset, load_presetshare_curve_dataset
from .manifest import write_json


SAMPLE_POLICY_VERSION = "experiment13_stratified_hash_v1"
CACHE_SCHEMA_VERSION = "experiment13_execution_cache_v1"
OPTIMIZATION_VERSION = "experiment13_exact_execution_v1"
KEEP_AWAKE_FLAGS = 0x80000001  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
KEEP_AWAKE_RESET = 0x80000000  # ES_CONTINUOUS


class KeepAwakeError(RuntimeError):
    pass


@dataclass(frozen=True)
class SampleProvenance:
    source_fingerprint: str
    sample_fingerprint: str
    policy_version: str
    seed: int
    train_fraction: float
    validation_fraction: float
    train_count: int
    validation_count: int
    train_indices_sha256: str
    validation_indices_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_fingerprint": self.source_fingerprint,
            "sample_fingerprint": self.sample_fingerprint,
            "sample_policy_version": self.policy_version,
            "sample_seed": self.seed,
            "train_sample_fraction": self.train_fraction,
            "validation_sample_fraction": self.validation_fraction,
            "train_count": self.train_count,
            "validation_count": self.validation_count,
            "train_indices_sha256": self.train_indices_sha256,
            "validation_indices_sha256": self.validation_indices_sha256,
        }


@dataclass(frozen=True)
class BaseStage:
    base_dictionary: np.ndarray
    train_alignment: AlignmentChoice
    validation_alignment: AlignmentChoice
    cache_key: str
    cache_hit: bool


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def implementation_fingerprint() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for name in ("strategy_grid_execution.py", "strategy_grid_runtime.py", "alignment.py", "component_ladder.py"):
        digest.update(name.encode("utf-8"))
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


@contextmanager
def scoped_system_required(*, strict: bool = True) -> Iterator[dict[str, Any]]:
    """Keep Windows awake for the scoped worker without controlling PowerToys."""
    state = {
        "platform": sys.platform,
        "enabled": False,
        "flags": KEEP_AWAKE_FLAGS if sys.platform == "win32" else 0,
        "acquired_at_utc": now_utc(),
        "external_power_toys_untouched": True,
    }
    if sys.platform != "win32":
        yield state
        return
    import ctypes

    result = int(ctypes.windll.kernel32.SetThreadExecutionState(KEEP_AWAKE_FLAGS))
    if result == 0:
        if strict:
            raise KeepAwakeError("SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) failed")
        yield state
        return
    state["enabled"] = True
    try:
        yield state
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(KEEP_AWAKE_RESET)


def source_inventory_fingerprint(metadata_path: Path) -> str:
    """Hash metadata contents and its ordered preset-path inventory."""
    metadata_path = Path(metadata_path)
    digest = hashlib.sha256()
    digest.update(metadata_path.read_bytes())
    paths: list[str] = []
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            raw = str(record.get("preset_file", "")).replace("\\", "/")
            if raw:
                paths.append(raw)
    for raw in paths:
        digest.update(b"\0")
        digest.update(raw.encode("utf-8"))
    return digest.hexdigest()


def load_dataset_cached(
    metadata_path: Path,
    *,
    cache_dir: Path | None,
    resolution: int,
    x_grid_mode: str,
    rebuild: bool = False,
    progress: Callable[[str], None] | None = None,
) -> tuple[Era2CurveDataset, dict[str, Any]]:
    metadata_path = Path(metadata_path)
    inventory = source_inventory_fingerprint(metadata_path)
    key_payload = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "inventory_fingerprint": inventory,
        "resolution": int(resolution),
        "x_grid_mode": x_grid_mode,
        "active_only": True,
    }
    key = hashlib.sha256(_canonical_json(key_payload).encode()).hexdigest()
    if cache_dir is None:
        dataset = load_presetshare_curve_dataset(
            metadata_path, resolution=resolution, x_grid_mode=x_grid_mode, progress=progress
        )
        return dataset, {"cache_key": key, "cache_hit": False, "cache_path": None}

    root = Path(cache_dir) / "datasets" / key
    arrays_path = root / "dataset.npz"
    metadata_cache_path = root / "dataset.json"
    if not rebuild and arrays_path.exists() and metadata_cache_path.exists():
        try:
            payload = json.loads(metadata_cache_path.read_text(encoding="utf-8"))
            if payload.get("cache_key") != key or payload.get("inventory_fingerprint") != inventory:
                raise ValueError("dataset cache identity mismatch")
            with np.load(arrays_path, allow_pickle=False) as archive:
                curves = np.asarray(archive["curves"], dtype=np.float32)
                topology = np.asarray(archive["topology"], dtype=np.int8)
                train = np.asarray(archive["train_indices"], dtype=np.int32)
                validation = np.asarray(archive["validation_indices"], dtype=np.int32)
            _validate_dataset_arrays(curves, topology, train, validation, payload)
            dataset = Era2CurveDataset(
                curves=curves,
                topology=topology,
                train_indices=train,
                validation_indices=validation,
                row_metadata=list(payload["row_metadata"]),
                source_fingerprint=str(payload["source_fingerprint"]),
                resolution=int(payload["resolution"]),
                x_grid_mode=str(payload["x_grid_mode"]),
                shapes=(),
            )
            if progress:
                progress(f"dataset cache hit: {key[:12]}")
            return dataset, {"cache_key": key, "cache_hit": True, "cache_path": str(root)}
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            if progress:
                progress(f"dataset cache invalid; rebuilding: {key[:12]}")

    dataset = load_presetshare_curve_dataset(
        metadata_path, resolution=resolution, x_grid_mode=x_grid_mode, progress=progress
    )
    root.mkdir(parents=True, exist_ok=True)
    tmp_arrays = root / f".dataset.{os.getpid()}.npz"
    np.savez_compressed(
        tmp_arrays,
        curves=dataset.curves,
        topology=dataset.topology,
        train_indices=dataset.train_indices,
        validation_indices=dataset.validation_indices,
    )
    tmp_arrays.replace(arrays_path)
    payload = {
        **key_payload,
        "cache_key": key,
        "source_fingerprint": dataset.source_fingerprint,
        "resolution": dataset.resolution,
        "x_grid_mode": dataset.x_grid_mode,
        "row_metadata": dataset.row_metadata,
        "curves_shape": list(dataset.curves.shape),
        "curves_dtype": str(dataset.curves.dtype),
        "topology_sha256": _array_sha256(dataset.topology),
        "train_indices_sha256": _array_sha256(dataset.train_indices),
        "validation_indices_sha256": _array_sha256(dataset.validation_indices),
        "created_at_utc": now_utc(),
    }
    write_json(metadata_cache_path, payload)
    return dataset, {"cache_key": key, "cache_hit": False, "cache_path": str(root)}


def deterministic_sample(
    dataset: Era2CurveDataset,
    *,
    train_fraction: float,
    validation_fraction: float,
    seed: int,
) -> tuple[Era2CurveDataset, SampleProvenance]:
    train_fraction = _fraction(train_fraction, "train_sample_fraction")
    validation_fraction = _fraction(validation_fraction, "validation_sample_fraction")
    train = _stratified_indices(dataset, dataset.train_indices, train_fraction, seed, "training")
    validation = _stratified_indices(dataset, dataset.validation_indices, validation_fraction, seed, "validation")
    sampled = Era2CurveDataset(
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
    train_hash = _array_sha256(train)
    validation_hash = _array_sha256(validation)
    fingerprint_payload = {
        "source_fingerprint": dataset.source_fingerprint,
        "policy_version": SAMPLE_POLICY_VERSION,
        "seed": int(seed),
        "train_fraction": train_fraction,
        "validation_fraction": validation_fraction,
        "train_indices_sha256": train_hash,
        "validation_indices_sha256": validation_hash,
    }
    provenance = SampleProvenance(
        source_fingerprint=dataset.source_fingerprint,
        sample_fingerprint=hashlib.sha256(_canonical_json(fingerprint_payload).encode()).hexdigest(),
        policy_version=SAMPLE_POLICY_VERSION,
        seed=int(seed),
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        train_count=len(train),
        validation_count=len(validation),
        train_indices_sha256=train_hash,
        validation_indices_sha256=validation_hash,
    )
    return sampled, provenance


def write_sample_artifacts(output_dir: Path, dataset: Era2CurveDataset, provenance: SampleProvenance) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "sample_manifest.json", provenance.as_dict())
    target = output_dir / "sample_indices.npz"
    tmp = output_dir / f".sample_indices.{os.getpid()}.npz"
    np.savez_compressed(tmp, train_indices=dataset.train_indices, validation_indices=dataset.validation_indices)
    tmp.replace(target)


def load_or_build_base_stage(
    dataset: Era2CurveDataset,
    provenance: SampleProvenance,
    *,
    backend: str,
    chunk_size: int,
    cache_dir: Path | None,
    rebuild: bool = False,
    progress: Callable[[str], None] | None = None,
) -> BaseStage:
    phase_count = dataset.curves.shape[1]
    key_payload = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "optimization_version": OPTIMIZATION_VERSION,
        "implementation_fingerprint": implementation_fingerprint(),
        "sample_fingerprint": provenance.sample_fingerprint,
        "backend": backend,
        "chunk_size": int(chunk_size),
        "base_dictionary_size": 32,
        "resolution": phase_count,
        "phase_policy": "fft_lattice",
        "gain_policy": "fixed",
    }
    key = hashlib.sha256(_canonical_json(key_payload).encode()).hexdigest()
    root = None if cache_dir is None else Path(cache_dir) / "base_stages" / key
    if root is not None and not rebuild and (root / "base_stage.npz").exists() and (root / "manifest.json").exists():
        try:
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            if manifest.get("cache_key") != key:
                raise ValueError("base-stage cache identity mismatch")
            with np.load(root / "base_stage.npz", allow_pickle=False) as archive:
                stage = _base_stage_from_archive(archive, key, True)
            _validate_base_stage(stage, len(dataset.train_indices), len(dataset.validation_indices), phase_count, manifest)
            if progress:
                progress(f"base-stage cache hit: {key[:12]}")
            return stage
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            if progress:
                progress(f"base-stage cache invalid; rebuilding: {key[:12]}")

    if progress:
        progress("base-stage: selecting base dictionary")
    base = x12._select_farthest_atoms(dataset.train_curves, width=32, include_zero=False, topology=None)
    if progress:
        progress("base-stage: training alignment")
    train = best_alignment(
        dataset.train_curves, base, phase_policy="fft_lattice", gain_policy="fixed", backend=backend,
        chunk_size=chunk_size, phase_candidate_count=phase_count,
    )
    if progress:
        progress("base-stage: validation alignment")
    validation = best_alignment(
        dataset.validation_curves, base, phase_policy="fft_lattice", gain_policy="fixed", backend=backend,
        chunk_size=chunk_size, phase_candidate_count=phase_count,
    )
    stage = BaseStage(_readonly(base), _readonly_choice(train), _readonly_choice(validation), key, False)
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
        target = root / "base_stage.npz"
        tmp = root / f".base_stage.{os.getpid()}.npz"
        np.savez_compressed(
            tmp,
            base_dictionary=stage.base_dictionary,
            train_indices=stage.train_alignment.indices,
            train_phases=stage.train_alignment.phases,
            train_gains=stage.train_alignment.gains,
            train_values=stage.train_alignment.values,
            train_losses=stage.train_alignment.losses,
            validation_indices=stage.validation_alignment.indices,
            validation_phases=stage.validation_alignment.phases,
            validation_gains=stage.validation_alignment.gains,
            validation_values=stage.validation_alignment.values,
            validation_losses=stage.validation_alignment.losses,
        )
        tmp.replace(target)
        arrays = {
            "base_dictionary": stage.base_dictionary,
            "train_values": stage.train_alignment.values,
            "validation_values": stage.validation_alignment.values,
        }
        write_json(root / "manifest.json", {
            **key_payload,
            "cache_key": key,
            "array_sha256": {name: _array_sha256(value) for name, value in arrays.items()},
            "created_at_utc": now_utc(),
        })
    return stage


def sample_manifest_matches(path: Path, provenance: SampleProvenance) -> bool:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return all(payload.get(key) == value for key, value in provenance.as_dict().items())


def _stratified_indices(
    dataset: Era2CurveDataset, indices: np.ndarray, fraction: float, seed: int, split: str
) -> np.ndarray:
    source = np.asarray(indices, dtype=np.int32)
    if fraction >= 1.0:
        return source.copy()
    target = max(1, int(len(source) * fraction))
    strata = {int(value): source[dataset.topology[source] == value] for value in np.unique(dataset.topology[source])}
    raw = {key: len(values) * fraction for key, values in strata.items()}
    counts = {key: int(np.floor(value)) for key, value in raw.items()}
    remaining = target - sum(counts.values())
    order = sorted(strata, key=lambda key: (-(raw[key] - counts[key]), key))
    for key in order[:remaining]:
        counts[key] += 1
    selected: list[int] = []
    for key in sorted(strata):
        ranked = sorted(
            (int(index) for index in strata[key]),
            key=lambda index: (_sample_digest(dataset, index, split, seed), index),
        )
        selected.extend(ranked[: counts[key]])
    return np.asarray(sorted(selected), dtype=np.int32)


def _sample_digest(dataset: Era2CurveDataset, index: int, split: str, seed: int) -> bytes:
    metadata = dataset.row_metadata[index]
    stable = {
        "policy": SAMPLE_POLICY_VERSION,
        "seed": int(seed),
        "split": split,
        "dataset_index": int(index),
        "preset_id": metadata.get("preset_id", ""),
        "author_id": metadata.get("author_id", ""),
        "lfo_index": metadata.get("lfo_index", ""),
        "shape_signature": metadata.get("shape_signature", ""),
    }
    return hashlib.sha256(_canonical_json(stable).encode()).digest()


def _base_stage_from_archive(archive: Any, key: str, hit: bool) -> BaseStage:
    base = _readonly(np.asarray(archive["base_dictionary"], dtype=np.float32))
    train = _readonly_choice(AlignmentChoice(
        np.asarray(archive["train_indices"], dtype=np.int32),
        np.asarray(archive["train_phases"], dtype=np.float32),
        np.asarray(archive["train_gains"], dtype=np.float32),
        np.asarray(archive["train_values"], dtype=np.float32),
        np.asarray(archive["train_losses"], dtype=np.float32),
        "cache",
    ))
    validation = _readonly_choice(AlignmentChoice(
        np.asarray(archive["validation_indices"], dtype=np.int32),
        np.asarray(archive["validation_phases"], dtype=np.float32),
        np.asarray(archive["validation_gains"], dtype=np.float32),
        np.asarray(archive["validation_values"], dtype=np.float32),
        np.asarray(archive["validation_losses"], dtype=np.float32),
        "cache",
    ))
    return BaseStage(base, train, validation, key, hit)


def _validate_dataset_arrays(
    curves: np.ndarray, topology: np.ndarray, train: np.ndarray, validation: np.ndarray, payload: dict[str, Any]
) -> None:
    if list(curves.shape) != list(payload.get("curves_shape", [])) or str(curves.dtype) != payload.get("curves_dtype"):
        raise ValueError("dataset cache shape or dtype mismatch")
    if len(topology) != len(curves) or not len(train) or not len(validation):
        raise ValueError("dataset cache split is invalid")
    for name, value in (("topology", topology), ("train_indices", train), ("validation_indices", validation)):
        if _array_sha256(value) != payload.get(f"{name}_sha256"):
            raise ValueError(f"dataset cache hash mismatch: {name}")


def _validate_base_stage(stage: BaseStage, train_count: int, validation_count: int, resolution: int, manifest: dict[str, Any]) -> None:
    if stage.base_dictionary.shape != (32, resolution):
        raise ValueError("base-stage dictionary shape mismatch")
    if stage.train_alignment.values.shape != (train_count, resolution):
        raise ValueError("base-stage training shape mismatch")
    if stage.validation_alignment.values.shape != (validation_count, resolution):
        raise ValueError("base-stage validation shape mismatch")
    hashes = manifest.get("array_sha256", {})
    for name, value in (
        ("base_dictionary", stage.base_dictionary),
        ("train_values", stage.train_alignment.values),
        ("validation_values", stage.validation_alignment.values),
    ):
        if _array_sha256(value) != hashes.get(name):
            raise ValueError(f"base-stage cache hash mismatch: {name}")


def _readonly(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    array.setflags(write=False)
    return array


def _readonly_choice(value: AlignmentChoice) -> AlignmentChoice:
    return AlignmentChoice(
        _readonly(value.indices), _readonly(value.phases), _readonly(value.gains),
        _readonly(value.values), _readonly(value.losses), value.backend_used,
    )


def _fraction(value: float, name: str) -> float:
    result = float(value)
    if not 0.0 < result <= 1.0:
        raise ValueError(f"{name} must be in (0, 1]")
    return result


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(_canonical_json(list(array.shape)).encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

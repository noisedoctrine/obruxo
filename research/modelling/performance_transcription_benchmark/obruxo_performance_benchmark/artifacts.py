"""Immutable model/checkpoint identity and safe path validation."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REQUIRED_MODEL_IDS = (
    "basic_pitch",
    "timbre_trap_base",
    "ymt3_plus",
    "yptf_multi",
    "yptf_moe_multi",
    "muscriptor_small",
    "muscriptor_medium",
    "muscriptor_large",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_PLACEHOLDER = re.compile(r"<[^>]+>|\b(?:TODO|TBD|PLACEHOLDER|LATEST)\b", re.IGNORECASE)
CHECKPOINT_IDENTITY_STATUSES = {
    "locked",
    "gated_digest_not_exposed_without_access",
}


class ArtifactError(ValueError):
    """Invalid release metadata or a failed identity check."""


class ArtifactUnavailable(RuntimeError):
    """A permitted model path is unavailable without environment changes."""


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    family: str
    publication_year: int
    output_contract: str
    source_repository: str
    source_revision: str
    checkpoint_repository: str
    checkpoint_revision: str
    checkpoint_path: str
    checkpoint_sha256: str | None
    checkpoint_size_bytes: int | None
    code_license: str
    weight_license: str
    benchmark_dtype: str
    native_sample_rate: int
    environment: str
    stock_inference: Mapping[str, object]
    availability: str = "available"
    unavailability_reason: str | None = None
    native_output_type: str = "unknown"
    native_batch_sizes: tuple[int, ...] = (1,)
    differentiable_boundary: str | None = None
    source_url: str | None = None
    checkpoint_url: str | None = None
    representation: Mapping[str, object] = field(default_factory=dict)
    checkpoint_identity_status: str = "locked"

    @property
    def is_available(self) -> bool:
        return self.availability == "available"

    @property
    def is_fully_locked(self) -> bool:
        return self.checkpoint_identity_status == "locked" and bool(self.checkpoint_sha256) and self.checkpoint_size_bytes is not None

    def public_identity(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "family": self.family,
            "publication_year": self.publication_year,
            "output_contract": self.output_contract,
            "source_repository": self.source_repository,
            "source_revision": self.source_revision,
            "checkpoint_repository": self.checkpoint_repository,
            "checkpoint_revision": self.checkpoint_revision,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_size_bytes": self.checkpoint_size_bytes,
            "checkpoint_identity_status": self.checkpoint_identity_status,
            "code_license": self.code_license,
            "weight_license": self.weight_license,
            "benchmark_dtype": self.benchmark_dtype,
            "native_sample_rate": self.native_sample_rate,
            "native_output_type": self.native_output_type,
            "native_batch_sizes": list(self.native_batch_sizes),
            "environment": self.environment,
            "stock_inference": dict(self.stock_inference),
            "availability": self.availability,
            "unavailability_reason": self.unavailability_reason,
            "differentiable_boundary": self.differentiable_boundary,
            "source_url": self.source_url,
            "checkpoint_url": self.checkpoint_url,
            "representation": dict(self.representation),
        }

    def identity_digest(self) -> str:
        payload = json.dumps(self.public_identity(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _required(mapping: Mapping[str, Any], key: str, label: str) -> Any:
    if key not in mapping or mapping[key] is None:
        raise ArtifactError(f"{label} is missing {key}")
    return mapping[key]


def _text(value: Any, label: str, *, allow_unverified: bool = False) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArtifactError(f"{label} must be a non-empty string")
    result = value.strip()
    if _PLACEHOLDER.search(result):
        raise ArtifactError(f"{label} contains an unresolved placeholder")
    if not allow_unverified and result.casefold() in {"unverified", "not-used", "unknown"}:
        raise ArtifactError(f"{label} is not locked")
    return result


def _spec(model_id: str, raw: Mapping[str, Any]) -> ModelSpec:
    source = _required(raw, "source", model_id)
    checkpoint = _required(raw, "checkpoint", model_id)
    if not isinstance(source, Mapping) or not isinstance(checkpoint, Mapping):
        raise ArtifactError(f"{model_id} source/checkpoint must be mappings")
    availability = str(raw.get("availability", "available"))
    if availability not in {"available", "unavailable"}:
        raise ArtifactError(f"{model_id}.availability must be available or unavailable")
    sizes = raw.get("native_batch_sizes", [1])
    if not isinstance(sizes, list) or any(type(item) is not int or item < 1 for item in sizes):
        raise ArtifactError(f"{model_id} native_batch_sizes is invalid")
    stock = raw.get("stock_inference")
    if not isinstance(stock, Mapping):
        raise ArtifactError(f"{model_id} stock_inference must be a mapping")
    identity_status = _text(
        _required(checkpoint, "identity_status", model_id),
        f"{model_id}.checkpoint.identity_status",
        allow_unverified=True,
    )
    if identity_status not in CHECKPOINT_IDENTITY_STATUSES:
        raise ArtifactError(f"{model_id}.checkpoint.identity_status is invalid")
    sha_value = checkpoint.get("sha256")
    size_value = checkpoint.get("size_bytes")
    if identity_status == "locked":
        if not isinstance(sha_value, str) or not _SHA256.fullmatch(sha_value):
            raise ArtifactError(f"{model_id}.checkpoint.sha256 is not a SHA-256 digest")
        if type(size_value) is not int or size_value <= 0:
            raise ArtifactError(f"{model_id}.checkpoint.size_bytes is not a positive integer")
    else:
        if sha_value is not None:
            raise ArtifactError(f"{model_id}.checkpoint.sha256 must be null when its digest is not exposed")
        if type(size_value) is not int or size_value <= 0:
            raise ArtifactError(f"{model_id}.checkpoint.size_bytes must be a positive public size")
    values = {
        "model_id": model_id,
        "family": _text(_required(raw, "family", model_id), f"{model_id}.family"),
        "publication_year": int(_required(raw, "publication_year", model_id)),
        "output_contract": _text(_required(raw, "output_contract", model_id), f"{model_id}.output_contract"),
        "source_repository": _text(_required(source, "repository", model_id), f"{model_id}.source.repository"),
        "source_revision": _text(_required(source, "revision", model_id), f"{model_id}.source.revision"),
        "checkpoint_repository": _text(_required(checkpoint, "repository", model_id), f"{model_id}.checkpoint.repository"),
        "checkpoint_revision": _text(_required(checkpoint, "revision", model_id), f"{model_id}.checkpoint.revision"),
        "checkpoint_path": _text(_required(checkpoint, "path", model_id), f"{model_id}.checkpoint.path"),
        "checkpoint_sha256": sha_value,
        "checkpoint_size_bytes": size_value,
        "checkpoint_identity_status": identity_status,
        "code_license": _text(_required(raw, "code_license", model_id), f"{model_id}.code_license"),
        "weight_license": _text(_required(raw, "weight_license", model_id), f"{model_id}.weight_license"),
        "benchmark_dtype": _text(_required(raw, "benchmark_dtype", model_id), f"{model_id}.benchmark_dtype"),
        "native_sample_rate": int(_required(raw, "native_sample_rate", model_id)),
        "environment": _text(_required(raw, "environment", model_id), f"{model_id}.environment"),
        "stock_inference": dict(stock),
        "availability": availability,
        "unavailability_reason": raw.get("unavailability_reason"),
        "native_output_type": str(raw.get("native_output_type", raw.get("output_contract", "unknown"))),
        "native_batch_sizes": tuple(sizes),
        "differentiable_boundary": raw.get("differentiable_boundary"),
        "source_url": raw.get("source_url"),
        "checkpoint_url": raw.get("checkpoint_url"),
        "representation": dict(raw.get("representation", {})),
    }
    if not _REVISION.fullmatch(values["source_revision"]) or not _REVISION.fullmatch(values["checkpoint_revision"]):
        raise ArtifactError(f"{model_id} source/checkpoint revisions must be immutable commit IDs")
    if values["native_sample_rate"] <= 0:
        raise ArtifactError(f"{model_id} contains an invalid size or sample rate")
    if availability != "available" and not isinstance(values["unavailability_reason"], str):
        raise ArtifactError(f"{model_id}.unavailability_reason is required when unavailable")
    if availability == "available" and not values["checkpoint_identity_status"] == "locked":
        raise ArtifactError(f"{model_id}.available checkpoint identity must be locked")
    return ModelSpec(**values)


def load_model_specs(path: Path) -> dict[str, ModelSpec]:
    config_path = Path(path).resolve(strict=True)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ArtifactError("model config could not be read") from exc
    if not isinstance(raw, Mapping) or raw.get("version") != 1 or not isinstance(raw.get("models"), Mapping):
        raise ArtifactError("model config must be format version 1")
    specs = {str(model_id): _spec(str(model_id), value) for model_id, value in raw["models"].items()}
    if tuple(specs) != REQUIRED_MODEL_IDS:
        raise ArtifactError("model config does not contain exactly the required model IDs")
    return specs


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checkpoint(spec: ModelSpec, checkpoint: Path) -> None:
    if not spec.is_available:
        raise ArtifactUnavailable(spec.unavailability_reason or "checkpoint is unavailable")
    candidate = Path(checkpoint).resolve(strict=False)
    if not candidate.is_file():
        raise ArtifactUnavailable("checkpoint_missing")
    try:
        size = candidate.stat().st_size
        digest = _sha256(candidate)
    except OSError as exc:
        raise ArtifactUnavailable("checkpoint_missing") from exc
    if size != spec.checkpoint_size_bytes or digest != spec.checkpoint_sha256:
        raise ArtifactError("checkpoint_hash_mismatch")


def verify_checkout(spec: ModelSpec, source_root: Path) -> None:
    if not spec.is_available:
        raise ArtifactUnavailable(spec.unavailability_reason or "source checkout is unavailable")
    root = Path(source_root).resolve(strict=True)
    if not root.is_dir():
        raise ArtifactUnavailable("source checkout is unavailable")
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise ArtifactUnavailable("source checkout metadata is unavailable") from exc
    revision = completed.stdout.strip()
    if completed.returncode != 0 or not revision or not revision.startswith(spec.source_revision):
        raise ArtifactError("source_revision_mismatch")


def public_specs(specs: Mapping[str, ModelSpec]) -> list[dict[str, Any]]:
    return [spec.public_identity() for spec in (specs[model_id] for model_id in REQUIRED_MODEL_IDS)]

"""Accessors for the landed #23/#24/#25 Basic Pitch seams."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from ..artifacts import (
    ArtifactError,
    ArtifactUnavailable,
    ModelSpec,
    verify_checkpoint,
)


def _basic_pitch_root() -> Path:
    root = Path(__file__).resolve().parents[3] / "basic_pitch"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def fixed_basic_pitch_contract() -> dict[str, Any]:
    _basic_pitch_root()
    from obruxo_basic_pitch.evaluation.runner import backend_contract

    return backend_contract()


def read_landed_baseline(manifest_path: Path) -> dict[str, Any]:
    """Read #25's stored baseline without re-running or re-scoring it."""
    manifest = Path(manifest_path).resolve(strict=True)
    output = manifest.parent
    run_path = output / "run.json"
    aggregate_path = output / "aggregates.json"
    if not run_path.is_file() or not aggregate_path.is_file():
        return {"status": "unavailable", "failure_code": "baseline_results_unavailable"}
    try:
        run = json.loads(run_path.read_text(encoding="utf-8"))
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactUnavailable("landed Basic Pitch baseline is unreadable") from exc
    return {
        "status": str(run.get("status", "unavailable")),
        "failure_code": run.get("failure_code"),
        "pair_count": int(run.get("pair_count", 0)),
        "aggregate": aggregate,
        "backend": run.get("backend"),
        "run_identity": run.get("run_identity"),
    }


class BasicPitchAdapter:
    """A deliberate no-op adapter: quality consumes #25's canonical result."""

    def __init__(self, spec: ModelSpec, source_root: Path | None, checkpoint: Path | None) -> None:
        self.spec = spec
        self.source_root = None if source_root is None else Path(source_root)
        self.checkpoint = None if checkpoint is None else Path(checkpoint)

    def preflight(self) -> None:
        root = _basic_pitch_root() if self.source_root is None else self.source_root.resolve(strict=True)
        metadata_path = root / "artifacts" / "basic_pitch_icassp_2022.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            source = metadata["source"]
        except (KeyError, OSError, TypeError, json.JSONDecodeError) as exc:
            raise ArtifactUnavailable("landed Basic Pitch metadata is unavailable") from exc
        if source.get("revision") != self.spec.source_revision or source.get("repository") != self.spec.source_repository:
            raise ArtifactError("Basic Pitch landed metadata does not match models.yaml")
        if self.checkpoint is not None:
            verify_checkpoint(self.spec, self.checkpoint)

    def load(self) -> None:
        self.preflight()

    def transcribe(self, _audio: Path) -> Any:
        raise ArtifactUnavailable("Basic Pitch quality is consumed from the landed #25 result")

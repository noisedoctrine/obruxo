"""Corpus evaluation orchestration over the fixed #24 PyTorch CPU seam."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ..benchmark import load_config
from ..constants import (
    FRAME_THRESHOLD,
    MIN_NOTE_LENGTH_FRAMES,
    MODEL_ID,
    ONSET_THRESHOLD,
    SPOTIFY_ONNX_GIT_BLOB_SHA1,
)
from ..inference import prepare_wav, unwrap_window_outputs
from ..model import BasicPitchICASSP2022
from ..postprocess import posteriorgrams_to_note_events
from .aggregate import aggregate_results
from .corpus import EvaluationPair, compare_source_snapshot, load_evaluation_manifest
from .labels import performance_labels
from .metrics import evaluate_notes_and_frames

EXPECTED_BACKEND_ID = "pytorch_cpu"
BACKEND_CONTRACT_VERSION = 1
PAIR_FAILURE_CODES = (
    "audio_decode_failed",
    "reference_failed",
    "inference_failed",
    "decoder_failed",
    "source_changed",
)


class EvaluationInputError(ValueError):
    """The evaluation input or destination violates the fixed contract."""


class BackendUnavailable(RuntimeError):
    """The exact selected #24 backend cannot execute in this runtime."""


def _approved_output_root() -> Path:
    return (Path(__file__).resolve().parents[2] / "outputs").resolve()


def _output_dir(path: Path | str) -> Path:
    root = _approved_output_root()
    resolved = Path(path).resolve(strict=False)
    if resolved == root or not resolved.is_relative_to(root):
        raise EvaluationInputError("evaluation output must be inside Basic Pitch outputs")
    return resolved


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _code_identity() -> dict[str, str]:
    package_root = Path(__file__).resolve().parent
    names = ("corpus.py", "labels.py", "metrics.py", "aggregate.py", "runner.py", "report.py")
    return {name: _digest(package_root / name) for name in names}


def _runtime_identity(torch: Any) -> dict[str, Any]:
    import numpy as np
    import scipy

    return {
        "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
    }


def backend_contract() -> dict[str, Any]:
    """Return the concrete, non-tunable corpus backend contract inherited from #24."""
    config_path = Path(__file__).resolve().parents[2] / "configs" / "backend_benchmark.yaml"
    try:
        config = load_config(config_path)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise BackendUnavailable("#24 backend configuration is unavailable") from exc
    if config.precision != "float32" or config.end_to_end_batch_size != 1:
        raise BackendUnavailable("#24 corpus boundary is not the fixed float32 batch-1 contract")
    return {
        "contract_version": BACKEND_CONTRACT_VERSION,
        "backend_id": EXPECTED_BACKEND_ID,
        "benchmark_spec_version": config.version,
        "boundary": "#24_end_to_end_audio_to_note_events_batch_1",
        "precision": config.precision,
        "config": config.as_dict(),
    }


def validate_backend_id(backend_id: str) -> None:
    """Reject every backend except the fixed #24 route; there is no fallback."""
    if backend_id != EXPECTED_BACKEND_ID:
        raise BackendUnavailable(f"selected backend {backend_id!r} is unavailable; no fallback is permitted")


def _load_model(checkpoint: Path) -> tuple[Any, dict[str, Any]]:
    try:
        import torch

        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = BasicPitchICASSP2022()
        model.load_state_dict(state, strict=True)
        model.eval()
    except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise BackendUnavailable("the canonical Basic Pitch CPU model could not be loaded") from exc
    return model, _runtime_identity(torch)


def _predict_pair(model: Any, pair: EvaluationPair) -> dict[str, Any]:
    try:
        import numpy as np
        import torch

        prepared = prepare_wav(pair.audio_path)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        return {"status": "failed", "failure_code": "audio_decode_failed", "error_type": type(exc).__name__}
    try:
        outputs: dict[str, list[Any]] = {name: [] for name in ("note", "onset", "contour")}
        with torch.inference_mode():
            for start in range(0, prepared.windows.shape[0], 1):
                host_batch = torch.from_numpy(prepared.windows[start : start + 1])
                prediction = model(host_batch)
                for name, values in outputs.items():
                    values.append(prediction[name].detach().cpu().numpy())
        windowed = {name: np.concatenate(values, axis=0) for name, values in outputs.items()}
        unwrapped = unwrap_window_outputs(windowed, original_sample_count=prepared.original_sample_count)
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        return {"status": "failed", "failure_code": "inference_failed", "error_type": type(exc).__name__}
    try:
        reference, _ = performance_labels(pair.midi_path)
        if unwrapped["note"].shape[0] == 0:
            predicted = []
        else:
            predicted = posteriorgrams_to_note_events(unwrapped)
        metrics = evaluate_notes_and_frames(reference, predicted, unwrapped["note"])
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        return {"status": "failed", "failure_code": "decoder_failed", "error_type": type(exc).__name__}
    return {
        "status": "ok",
        "failure_code": None,
        "audio_seconds": prepared.audio_seconds,
        "predicted_note_count": len(predicted),
        "metrics": metrics,
    }


def _pair_result(pair: EvaluationPair, identity: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "pair_id": pair.pair_id,
        "preset_id": pair.preset_id,
        "labels": pair.labels,
        "pairing_method": pair.pairing_method,
        "provenance_status": pair.provenance_status,
        "qa_warning_codes": list(pair.qa_warning_codes),
        "run_identity": dict(identity),
        **dict(result),
    }


def _same_identity(path: Path, pair_id: str, identity: Mapping[str, Any]) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if value.get("pair_id") != pair_id or value.get("run_identity") != dict(identity):
        return None
    return value


def evaluate_corpus(
    manifest: Path | str,
    output: Path | str,
    *,
    checkpoint: Path | str | None = None,
    force: bool = False,
    predictor: Callable[[EvaluationPair], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate every manifest row, resuming only rows with the current identity."""
    output_dir = _output_dir(output)
    pairs = load_evaluation_manifest(manifest)
    for pair in pairs:
        for source in (pair.audio_path, pair.midi_path, pair.preset_path):
            if source is not None and (output_dir == source.parent or output_dir.is_relative_to(source.parent)):
                raise EvaluationInputError("evaluation output overlaps a source directory")
    backend = backend_contract()
    validate_backend_id(str(backend["backend_id"]))
    checkpoint_path = (
        Path(checkpoint).resolve(strict=True)
        if checkpoint is not None
        else Path(__file__).resolve().parents[2] / "artifacts" / "basic_pitch_icassp_2022.pt"
    )
    if not checkpoint_path.is_file():
        raise BackendUnavailable("canonical Basic Pitch checkpoint is unavailable")
    manifest_path = Path(manifest).resolve(strict=True)
    identity = {
        "format_version": 1,
        "manifest_sha256": _digest(manifest_path),
        "checkpoint_sha256": _digest(checkpoint_path),
        "model_id": MODEL_ID,
        "source_git_blob_sha1": SPOTIFY_ONNX_GIT_BLOB_SHA1,
        "evaluation_code": _code_identity(),
        "backend": backend,
        "decoder": {
            "onset_threshold": ONSET_THRESHOLD,
            "frame_threshold": FRAME_THRESHOLD,
            "minimum_note_length_frames": MIN_NOTE_LENGTH_FRAMES,
            "infer_onsets": True,
            "melodia_trick": True,
            "minimum_frequency": None,
            "maximum_frequency": None,
        },
    }
    pairs_dir = output_dir / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, Any]] = []
    pending: list[EvaluationPair] = []
    for pair in pairs:
        path = pairs_dir / f"{pair.pair_id}.json"
        cached = _same_identity(path, pair.pair_id, identity)
        if cached is not None and not force:
            existing.append(cached)
        else:
            pending.append(pair)
    model = None
    runtime = None
    if pending and predictor is None:
        model, runtime = _load_model(checkpoint_path)
    results = list(existing)
    for pair in pending:
        result = predictor(pair) if predictor is not None else _predict_pair(model, pair)
        row = _pair_result(pair, identity, result)
        _atomic_json(pairs_dir / f"{pair.pair_id}.json", row)
        results.append(row)
    results.sort(key=lambda row: str(row["pair_id"]))
    aggregate = aggregate_results(results)
    failures = [
        {"pair_id": row["pair_id"], "failure_code": row.get("failure_code"), "error_type": row.get("error_type")}
        for row in results
        if row.get("status") != "ok"
    ]
    source_check = None
    snapshot_path = manifest_path.with_name("source_snapshot.json")
    if snapshot_path.is_file():
        try:
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            source_check = compare_source_snapshot(list(snapshot.get("before", [])))
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise EvaluationInputError("source snapshot is invalid") from exc
        if source_check["source_stat_mismatches"]:
            raise EvaluationInputError("a source artifact changed during evaluation")
    status = "ok" if pairs else "unavailable"
    run_value = {
        "format_version": 1,
        "status": status,
        "failure_code": None if pairs else "no_eligible_pairs",
        "model": {"model_id": MODEL_ID, "source_git_blob_sha1": SPOTIFY_ONNX_GIT_BLOB_SHA1},
        "runtime": runtime,
        "backend": backend,
        "decoder": identity["decoder"],
        "pair_count": len(pairs),
        "successful_pair_count": sum(row.get("status") == "ok" for row in results),
        "failed_pair_count": len(failures),
        "source_check": None if source_check is None else {
            "source_stat_records": source_check["source_stat_records"],
            "source_stat_mismatches": len(source_check["source_stat_mismatches"]),
        },
        "run_identity": identity,
    }
    for name, value in (("run.json", run_value), ("aggregates.json", aggregate), ("failure_cases.json", {"failures": failures})):
        _atomic_json(output_dir / name, value)
    return {"status": status, "pair_count": len(pairs), "successful_pair_count": run_value["successful_pair_count"], "failed_pair_count": len(failures), "aggregate": aggregate}

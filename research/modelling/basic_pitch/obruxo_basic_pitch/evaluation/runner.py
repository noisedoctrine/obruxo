"""Corpus evaluation orchestration over the exact #24-selected inference seam."""

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

BACKEND_CONTRACT_VERSION = 1
SUPPORTED_TORCH_BACKENDS = {"pytorch_cpu", "pytorch_xpu"}
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
        raise EvaluationInputError(
            "evaluation output must be inside Basic Pitch outputs"
        )
    return resolved


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
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
    evaluation_root = Path(__file__).resolve().parent
    basic_pitch_root = evaluation_root.parent
    files = {
        "evaluation/aggregate.py": evaluation_root / "aggregate.py",
        "evaluation/corpus.py": evaluation_root / "corpus.py",
        "evaluation/labels.py": evaluation_root / "labels.py",
        "evaluation/metrics.py": evaluation_root / "metrics.py",
        "evaluation/report.py": evaluation_root / "report.py",
        "evaluation/runner.py": evaluation_root / "runner.py",
        "inference.py": basic_pitch_root / "inference.py",
        "model.py": basic_pitch_root / "model.py",
        "postprocess.py": basic_pitch_root / "postprocess.py",
    }
    return {name: _digest(path) for name, path in files.items()}


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
    """Return the exact machine-readable corpus route selected by #24."""
    config_path = (
        Path(__file__).resolve().parents[2] / "configs" / "backend_benchmark.yaml"
    )
    try:
        config = load_config(config_path)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise BackendUnavailable("#24 backend configuration is unavailable") from exc
    if config.precision != "float32" or config.end_to_end_batch_size != 1:
        raise BackendUnavailable(
            "#24 corpus boundary is not the fixed float32 batch-1 contract"
        )
    report_path = (
        Path(__file__).resolve().parents[2] / "reports" / "backend_benchmark.json"
    )
    try:
        benchmark = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BackendUnavailable("#24 benchmark report is unavailable") from exc
    if not isinstance(benchmark, Mapping):
        raise BackendUnavailable("#24 benchmark report is invalid")
    decision = benchmark.get("corpus_inference_decision")
    report_identity = benchmark.get("run_identity")
    if not isinstance(decision, Mapping) or decision.get("status") != "selected":
        raise BackendUnavailable("#24 corpus inference decision is unavailable")
    supporting_identity = decision.get("supporting_run_identity")
    if (
        not isinstance(report_identity, Mapping)
        or supporting_identity != report_identity
    ):
        raise BackendUnavailable(
            "#24 corpus inference decision identity does not match its report"
        )
    backend_id = str(decision.get("backend_id", ""))
    if backend_id not in SUPPORTED_TORCH_BACKENDS:
        raise BackendUnavailable(
            f"selected #24 backend {backend_id!r} is not executable by this evaluator"
        )
    if (
        decision.get("precision") != config.precision
        or decision.get("boundary") != "end_to_end_audio_to_note_event"
    ):
        raise BackendUnavailable(
            "#24 corpus inference decision does not match the fixed evaluation contract"
        )
    device = str(decision.get("device", ""))
    if backend_id == "pytorch_cpu" and device != "cpu":
        raise BackendUnavailable("#24 CPU route has an invalid selected device")
    if backend_id == "pytorch_xpu" and not device.startswith("xpu:"):
        raise BackendUnavailable("#24 XPU route has an invalid selected device")
    return {
        "contract_version": BACKEND_CONTRACT_VERSION,
        "backend_id": backend_id,
        "device": device,
        "benchmark_spec_version": config.version,
        "boundary": decision["boundary"],
        "precision": config.precision,
        "config": config.as_dict(),
        "selection_source": "#24 corpus_inference_decision",
        "selection_rule": decision.get("selection_rule"),
        "supporting_run_identity": dict(supporting_identity),
        "supporting_measurement": dict(decision.get("supporting_measurement", {})),
    }


def validate_backend_id(backend_id: str) -> None:
    """Reject routes this evaluator cannot execute; there is no fallback."""
    if backend_id not in SUPPORTED_TORCH_BACKENDS:
        raise BackendUnavailable(
            f"selected backend {backend_id!r} is unavailable; no fallback is permitted"
        )


def _select_device(torch: Any, backend: Mapping[str, Any]) -> Any:
    backend_id = str(backend["backend_id"])
    device_name = str(backend["device"])
    if backend_id == "pytorch_cpu":
        return torch.device("cpu")
    if backend_id != "pytorch_xpu":
        raise BackendUnavailable(
            f"selected backend {backend_id!r} is unavailable; no fallback is permitted"
        )
    try:
        if not torch.xpu.is_available():
            raise BackendUnavailable(
                "selected PyTorch XPU route is unavailable; no CPU fallback is permitted"
            )
        device = torch.device(device_name)
        if device.type != "xpu":
            raise BackendUnavailable("selected PyTorch XPU route has an invalid device")
        return device
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise BackendUnavailable(
            "selected PyTorch XPU route is unavailable; no CPU fallback is permitted"
        ) from exc


def _synchronize(torch: Any, device: Any) -> None:
    if device.type == "xpu":
        torch.xpu.synchronize(device)


def _load_model(
    checkpoint: Path, backend: Mapping[str, Any], torch: Any
) -> tuple[Any, Any]:
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = BasicPitchICASSP2022()
        model.load_state_dict(state, strict=True)
        model.eval()
        device = _select_device(torch, backend)
        _synchronize(torch, device)
        model.to(device)
        _synchronize(torch, device)
    except BackendUnavailable:
        raise
    except (
        AttributeError,
        ImportError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise BackendUnavailable(
            "the canonical Basic Pitch selected-route model could not be loaded"
        ) from exc
    return model, device


def _predict_pair(
    model: Any, pair: EvaluationPair, torch: Any, device: Any
) -> dict[str, Any]:
    try:
        import numpy as np

        prepared = prepare_wav(pair.audio_path)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        return {
            "status": "failed",
            "failure_code": "audio_decode_failed",
            "error_type": type(exc).__name__,
        }
    try:
        outputs: dict[str, list[Any]] = {
            name: [] for name in ("note", "onset", "contour")
        }
        with torch.inference_mode():
            for start in range(0, prepared.windows.shape[0], 1):
                host_batch = torch.from_numpy(prepared.windows[start : start + 1]).to(
                    device
                )
                _synchronize(torch, device)
                prediction = model(host_batch)
                _synchronize(torch, device)
                for name, values in outputs.items():
                    values.append(prediction[name].detach().cpu().numpy())
        windowed = {
            name: np.concatenate(values, axis=0) for name, values in outputs.items()
        }
        unwrapped = unwrap_window_outputs(
            windowed, original_sample_count=prepared.original_sample_count
        )
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        return {
            "status": "failed",
            "failure_code": "inference_failed",
            "error_type": type(exc).__name__,
        }
    try:
        reference, _ = performance_labels(pair.midi_path)
        if unwrapped["note"].shape[0] == 0:
            predicted = []
        else:
            predicted = posteriorgrams_to_note_events(unwrapped)
        metrics = evaluate_notes_and_frames(reference, predicted, unwrapped["note"])
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        return {
            "status": "failed",
            "failure_code": "decoder_failed",
            "error_type": type(exc).__name__,
        }
    return {
        "status": "ok",
        "failure_code": None,
        "audio_seconds": prepared.audio_seconds,
        "predicted_note_count": len(predicted),
        "metrics": metrics,
    }


def _pair_result(
    pair: EvaluationPair, identity: Mapping[str, Any], result: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "pair_id": pair.pair_id,
        "preset_id": pair.preset_id,
        "labels": pair.labels,
        "pairing_method": pair.pairing_method,
        "audio_source": pair.audio_source,
        "provenance_status": pair.provenance_status,
        "qa_warning_codes": list(pair.qa_warning_codes),
        "run_identity": dict(identity),
        **dict(result),
    }


def _same_identity(
    path: Path, pair_id: str, identity: Mapping[str, Any]
) -> dict[str, Any] | None:
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
            if source is not None and (
                output_dir == source.parent or output_dir.is_relative_to(source.parent)
            ):
                raise EvaluationInputError(
                    "evaluation output overlaps a source directory"
                )
    backend = backend_contract()
    validate_backend_id(str(backend["backend_id"]))
    try:
        import torch

        runtime = _runtime_identity(torch)
    except (
        AttributeError,
        ImportError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise BackendUnavailable(
            "the required PyTorch runtime could not be initialized"
        ) from exc
    checkpoint_path = (
        Path(checkpoint).resolve(strict=True)
        if checkpoint is not None
        else Path(__file__).resolve().parents[2]
        / "artifacts"
        / "basic_pitch_icassp_2022.pt"
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
        "runtime": runtime,
        "device": backend["device"],
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
    device = None
    if pending and predictor is None:
        model, device = _load_model(checkpoint_path, backend, torch)
    results = list(existing)
    for pair in pending:
        result = (
            predictor(pair)
            if predictor is not None
            else _predict_pair(model, pair, torch, device)
        )
        row = _pair_result(pair, identity, result)
        _atomic_json(pairs_dir / f"{pair.pair_id}.json", row)
        results.append(row)
    results.sort(key=lambda row: str(row["pair_id"]))
    aggregate = aggregate_results(results)
    failures = [
        {
            "pair_id": row["pair_id"],
            "failure_code": row.get("failure_code"),
            "error_type": row.get("error_type"),
        }
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
        "model": {
            "model_id": MODEL_ID,
            "source_git_blob_sha1": SPOTIFY_ONNX_GIT_BLOB_SHA1,
        },
        "runtime": runtime,
        "backend": backend,
        "decoder": identity["decoder"],
        "pair_count": len(pairs),
        "successful_pair_count": sum(row.get("status") == "ok" for row in results),
        "failed_pair_count": len(failures),
        "source_check": None
        if source_check is None
        else {
            "source_stat_records": source_check["source_stat_records"],
            "source_stat_mismatches": len(source_check["source_stat_mismatches"]),
        },
        "run_identity": identity,
    }
    for name, value in (
        ("run.json", run_value),
        ("aggregates.json", aggregate),
        ("failure_cases.json", {"failures": failures}),
    ):
        _atomic_json(output_dir / name, value)
    return {
        "status": status,
        "pair_count": len(pairs),
        "successful_pair_count": run_value["successful_pair_count"],
        "failed_pair_count": len(failures),
        "aggregate": aggregate,
    }

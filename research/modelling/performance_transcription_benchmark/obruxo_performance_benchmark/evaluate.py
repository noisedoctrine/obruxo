"""Evaluation orchestration that imports the landed #25 metric stack."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import ArtifactError, ArtifactUnavailable, ModelSpec
from .types import (
    NormalizedNote,
    TranscriptionOutput,
    empty_frame_prediction,
    validate_output,
)


def _basic_pitch_root() -> Path:
    return Path(__file__).resolve().parents[2] / "basic_pitch"


def _metric_modules() -> tuple[Any, Any, Any, Any]:
    import sys

    root = _basic_pitch_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from obruxo_basic_pitch.evaluation.aggregate import aggregate_results
    from obruxo_basic_pitch.evaluation.corpus import load_evaluation_manifest
    from obruxo_basic_pitch.evaluation.labels import performance_labels
    from obruxo_basic_pitch.evaluation.metrics import frame_metrics, note_metrics

    return aggregate_results, load_evaluation_manifest, performance_labels, (note_metrics, frame_metrics)


@dataclass(frozen=True)
class EvaluationRun:
    model_id: str
    variant_id: str
    manifest_identity: str
    model_identity: str
    successful_pairs: int
    failed_pairs: int
    status: str = "ok"
    failure_code: str | None = None


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _approved_output(path: Path | str) -> Path:
    root = (Path(__file__).resolve().parents[1] / "outputs").resolve()
    candidate = Path(path).resolve(strict=False)
    if candidate == root or not candidate.is_relative_to(root):
        raise ValueError("evaluation output must be inside the ignored benchmark output area")
    return candidate


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


def _note_events(notes: Sequence[NormalizedNote]) -> list[Any]:
    from obruxo_basic_pitch.postprocess import NoteEvent

    return [
        NoteEvent(
            start_time_s=note.onset_seconds,
            end_time_s=note.offset_seconds,
            pitch_midi=note.midi_pitch,
            amplitude=(note.velocity_midi / 127.0) if note.velocity_midi is not None else 0.0,
            pitch_bend=None,
        )
        for note in notes
    ]


def _null_velocity() -> dict[str, Any]:
    return {
        "support": 0,
        "signed_mean": None,
        "mae": None,
        "median_abs": None,
        "p90_abs": None,
        "p95_abs": None,
        "pearson": None,
        "pearson_status": "not_applicable_no_native_velocity",
    }


def score_output(reference: Sequence[Any], output: TranscriptionOutput) -> dict[str, Any]:
    """Score only already-decoded output through #25's exact functions."""
    _, _, _, (note_metrics, frame_metrics) = _metric_modules()
    result: dict[str, Any] = {}
    if output.notes is not None:
        predicted = _note_events(output.notes)
        result["notes"] = note_metrics(reference, predicted)
        if not any(note.velocity_midi is not None for note in output.notes):
            result["notes"]["velocity"] = _null_velocity()
    if output.frame_pitch is not None:
        from .types import common_frame_times

        expected = common_frame_times(output.frame_pitch.active_midi.shape[0])
        if not np.array_equal(output.frame_pitch.times_seconds, expected):
            raise ValueError("frame prediction is not on #25's exact common grid")
        result["frames"] = frame_metrics(reference, output.frame_pitch.active_midi.astype(np.float64))
    return result


def failure_metrics(reference: Sequence[Any], *, frame_model: bool, frame_count: int = 0) -> dict[str, Any]:
    """Create the required empty note/all-false frame penalty via #25."""
    _, _, _, (note_metrics, frame_metrics) = _metric_modules()
    result: dict[str, Any] = {}
    if not frame_model:
        result["notes"] = note_metrics(reference, [])
        result["notes"]["velocity"] = _null_velocity()
    else:
        result["frames"] = frame_metrics(reference, empty_frame_prediction(frame_count).active_midi.astype(np.float64))
    return result


def _coverage(eligible: int, successful: int, failed: int, aggregate: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(aggregate)
    value.update(
        {
            "pair_count": eligible,
            "successful_pair_count": successful,
            "failed_pair_count": failed,
            "coverage": float(successful / eligible) if eligible else None,
        }
    )
    return value


def _annotate_category_coverage(aggregate: dict[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
    for field in tuple(aggregate.get("groups", {})):
        values = {str(row.get("labels", {}).get(field, "unknown") or "unknown") for row in rows}
        groups = aggregate.setdefault("groups", {}).setdefault(field, {})
        for value in sorted(values):
            matching = [row for row in rows if str(row.get("labels", {}).get(field, "unknown") or "unknown") == value]
            successful = sum(row.get("status") == "ok" for row in matching)
            summary = groups.setdefault(value, {})
            summary.update(
                {
                    "eligible_pairs": len(matching),
                    "successful_pairs": successful,
                    "failed_pairs": len(matching) - successful,
                    "coverage": float(successful / len(matching)) if matching else None,
                }
            )


def build_quality_views(
    rows: Sequence[Mapping[str, Any]],
    references: Mapping[str, Sequence[Any]],
    *,
    frame_model: bool,
    bootstrap_replicates: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Produce success-only and failure-penalized #25 aggregations."""
    aggregate_results, _, _, _ = _metric_modules()
    materialized = [dict(row) for row in rows]
    successful = sum(row.get("status") == "ok" for row in materialized)
    failed = len(materialized) - successful
    successful_rows = [row for row in materialized if row.get("status") == "ok"]
    success_only = aggregate_results(successful_rows, bootstrap_replicates=bootstrap_replicates, seed=seed)
    _annotate_category_coverage(success_only, materialized)
    penalized_rows: list[dict[str, Any]] = []
    for row in materialized:
        if row.get("status") == "ok":
            penalized_rows.append(dict(row))
            continue
        pair_id = str(row.get("pair_id", ""))
        reference = references.get(pair_id)
        if reference is None:
            raise ValueError("failure penalty requires the corresponding reference notes")
        penalized = dict(row)
        penalized["source_status"] = row.get("status")
        penalized["source_failure_code"] = row.get("failure_code")
        penalized["status"] = "ok"
        penalized["metrics"] = failure_metrics(reference, frame_model=frame_model, frame_count=int(row.get("frame_count", 0)))
        penalized_rows.append(penalized)
    failure_penalized = aggregate_results(penalized_rows, bootstrap_replicates=bootstrap_replicates, seed=seed)
    _annotate_category_coverage(failure_penalized, materialized)
    return {
        "success_only": {
            "eligible_pairs": len(materialized),
            "successful_pairs": successful,
            "failed_pairs": failed,
            "coverage": float(successful / len(materialized)) if materialized else None,
            "aggregate": _coverage(len(materialized), successful, failed, success_only),
        },
        "failure_penalized": {
            "eligible_pairs": len(materialized),
            "successful_pairs": successful,
            "failed_pairs": failed,
            "coverage": float(successful / len(materialized)) if materialized else None,
            "aggregate": _coverage(len(materialized), successful, failed, failure_penalized),
        },
    }


def _frame_count(audio_path: Path) -> int:
    try:
        root = _basic_pitch_root()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from obruxo_basic_pitch.constants import ANNOTATIONS_FPS, AUDIO_SAMPLE_RATE
        from obruxo_basic_pitch.inference import prepare_wav

        prepared = prepare_wav(audio_path)
        return max(0, int(prepared.original_sample_count * ANNOTATIONS_FPS // AUDIO_SAMPLE_RATE))
    except (ImportError, OSError, ValueError, TypeError):
        return 0


def _runtime_identity() -> dict[str, Any]:
    """Resolve the imported runtime before any pair cache can be reused."""
    import numpy
    import scipy
    import torch

    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "platform": platform.platform(),
    }


def _code_identity() -> dict[str, str]:
    benchmark_root = Path(__file__).resolve().parent
    basic_pitch_root = _basic_pitch_root()
    files = {
        "artifacts.py": benchmark_root / "artifacts.py",
        "types.py": benchmark_root / "types.py",
        "evaluate.py": benchmark_root / "evaluate.py",
        "quantization.py": benchmark_root / "quantization.py",
        "adapters/__init__.py": benchmark_root / "adapters" / "__init__.py",
        "adapters/basic_pitch.py": benchmark_root / "adapters" / "basic_pitch.py",
        "adapters/muscriptor.py": benchmark_root / "adapters" / "muscriptor.py",
        "adapters/timbre_trap.py": benchmark_root / "adapters" / "timbre_trap.py",
        "adapters/yourmt3.py": benchmark_root / "adapters" / "yourmt3.py",
        "basic_pitch/constants.py": basic_pitch_root / "obruxo_basic_pitch" / "constants.py",
        "basic_pitch/inference.py": basic_pitch_root / "obruxo_basic_pitch" / "inference.py",
        "basic_pitch/model.py": basic_pitch_root / "obruxo_basic_pitch" / "model.py",
        "basic_pitch/postprocess.py": basic_pitch_root / "obruxo_basic_pitch" / "postprocess.py",
        "basic_pitch/evaluation/aggregate.py": basic_pitch_root / "obruxo_basic_pitch" / "evaluation" / "aggregate.py",
        "basic_pitch/evaluation/corpus.py": basic_pitch_root / "obruxo_basic_pitch" / "evaluation" / "corpus.py",
        "basic_pitch/evaluation/labels.py": basic_pitch_root / "obruxo_basic_pitch" / "evaluation" / "labels.py",
        "basic_pitch/evaluation/metrics.py": basic_pitch_root / "obruxo_basic_pitch" / "evaluation" / "metrics.py",
    }
    return {name: _digest(path) for name, path in files.items()}


def _adapter_identity(adapter: object) -> str:
    return f"{type(adapter).__module__}.{type(adapter).__qualname__}"


def _pair_resume_identity(
    spec: ModelSpec,
    variant_id: str,
    adapter: object,
    manifest_identity: str,
    runtime_identity: Mapping[str, Any],
    pair_id: str,
    code_identity: Mapping[str, str] | None = None,
) -> tuple[str, dict[str, Any]]:
    identity = {
        "pair_id": pair_id,
        "manifest_sha256": manifest_identity,
        "model_identity": spec.identity_digest(),
        "variant_id": variant_id,
        "adapter_identity": _adapter_identity(adapter),
        "runtime_identity": dict(runtime_identity),
        "code_identity": dict(code_identity or _code_identity()),
        "backend_contract": {
            "route": "pytorch_cpu",
            "device": "cpu",
            "precision": spec.benchmark_dtype,
            "boundary": "full_clip_native_transcription",
        },
        "stock_inference": dict(spec.stock_inference),
        "metric_contract": "issue_25_note_frame_metrics_v1",
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), identity


def _load_cached_pair(
    path: Path,
    expected_identity: str,
    pair_id: str,
    expected_details: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or value.get("pair_id") != pair_id:
        return None
    stored_identity = value.get("resume_identity_digest", value.get("resume_identity"))
    if stored_identity != expected_identity:
        return None
    if expected_details is not None and value.get("resume_identity") != dict(expected_details):
        return None
    if value.get("status") not in {"ok", "runtime_failed", "out_of_memory", "invalid_native_output"}:
        return None
    return value


def _source_stat_snapshot(pairs: Sequence[Any]) -> dict[str, tuple[int, int]]:
    records: dict[str, tuple[int, int]] = {}
    for pair in pairs:
        for path in (pair.audio_path, pair.midi_path, pair.preset_path):
            if path is None:
                continue
            candidate = Path(path).resolve(strict=True)
            stat = candidate.stat()
            records[str(candidate)] = (int(stat.st_size), int(stat.st_mtime_ns))
    return records


def _model_run(
    spec: ModelSpec,
    variant_id: str,
    manifest: Path,
    status: str,
    *,
    failure_code: str | None = None,
    successful_pairs: int = 0,
    failed_pairs: int = 0,
) -> EvaluationRun:
    return EvaluationRun(
        model_id=spec.model_id,
        variant_id=variant_id,
        manifest_identity=_digest(manifest),
        model_identity=spec.identity_digest(),
        successful_pairs=successful_pairs,
        failed_pairs=failed_pairs,
        status=status,
        failure_code=failure_code,
    )


def _write_unavailable(output: Path, run: EvaluationRun, *, reason: str | None = None) -> None:
    _atomic_json(
        output / "run.json",
        {
            "format_version": 1,
            "status": run.status,
            "failure_code": run.failure_code,
            "reason": reason,
            "model_id": run.model_id,
            "variant_id": run.variant_id,
            "manifest_sha256": run.manifest_identity,
            "model_identity": run.model_identity,
            "successful_pairs": run.successful_pairs,
            "failed_pairs": run.failed_pairs,
        },
    )
    _atomic_json(output / "aggregates.json", {"status": run.status, "failure_code": run.failure_code, "quality": None})


def _landed_quality_views(aggregate: Mapping[str, Any], *, eligible_pairs: int, successful_pairs: int, failed_pairs: int) -> dict[str, Any]:
    view = {
        "eligible_pairs": eligible_pairs,
        "successful_pairs": successful_pairs,
        "failed_pairs": failed_pairs,
        "coverage": float(successful_pairs / eligible_pairs) if eligible_pairs else None,
        "aggregate": dict(aggregate),
    }
    return {"success_only": view, "failure_penalized": dict(view)}


def _consume_landed_basic_pitch(
    spec: ModelSpec,
    adapter: object,
    manifest: Path,
    pairs: Sequence[Any],
    output: Path,
    runtime_identity: Mapping[str, Any],
    code_identity: Mapping[str, str],
) -> EvaluationRun:
    from .adapters.basic_pitch import read_landed_baseline

    baseline = read_landed_baseline(manifest)
    if baseline.get("status") != "ok":
        raise ArtifactError("landed Basic Pitch baseline is unavailable")
    identity = baseline.get("run_identity")
    if not isinstance(identity, Mapping) or identity.get("manifest_sha256") != _digest(manifest):
        raise ArtifactError("landed Basic Pitch baseline does not match the supplied #25 manifest")
    if identity.get("checkpoint_sha256") != spec.checkpoint_sha256:
        raise ArtifactError("landed Basic Pitch baseline checkpoint does not match models.yaml")
    pair_count = int(baseline.get("pair_count", 0))
    successful_pairs = int(baseline.get("successful_pair_count", 0))
    failed_pairs = int(baseline.get("failed_pair_count", 0))
    if pair_count != len(pairs) or successful_pairs + failed_pairs != pair_count:
        raise ArtifactError("landed Basic Pitch baseline pair population does not match the supplied #25 manifest")
    aggregate = baseline.get("aggregate")
    if not isinstance(aggregate, Mapping):
        raise ArtifactError("landed Basic Pitch aggregate is unavailable")
    variant_id = "full_precision"
    adapter_identity = f"{type(adapter).__module__}.{type(adapter).__qualname__}"
    _atomic_json(
        output / "model_lock.json",
        {
            "format_version": 1,
            "model_id": spec.model_id,
            "model_identity": spec.identity_digest(),
            "variant_id": variant_id,
            "adapter_identity": adapter_identity,
            "source": "landed_issue_25_result",
            "source_run_identity": dict(identity),
            "runtime_identity": runtime_identity,
            "code_identity": code_identity,
        },
    )
    run = _model_run(spec, variant_id, manifest, "ok", successful_pairs=successful_pairs, failed_pairs=failed_pairs)
    _atomic_json(
        output / "run.json",
        {
            "format_version": 1,
            "status": run.status,
            "failure_code": None,
            "model_id": run.model_id,
            "variant_id": run.variant_id,
            "manifest_sha256": run.manifest_identity,
            "model_identity": run.model_identity,
            "successful_pairs": successful_pairs,
            "failed_pairs": failed_pairs,
            "source": "landed_issue_25_result",
            "source_run_identity": dict(identity),
            "runtime_identity": runtime_identity,
            "code_identity": code_identity,
        },
    )
    _atomic_json(
        output / "aggregates.json",
        {"status": "ok", "quality": _landed_quality_views(aggregate, eligible_pairs=pair_count, successful_pairs=successful_pairs, failed_pairs=failed_pairs)},
    )
    return run


def evaluate_model(
    spec: ModelSpec,
    adapter: object,
    manifest_path: Path,
    output_dir: Path,
    *,
    quantized: bool = False,
    force: bool = False,
) -> EvaluationRun:
    """Run fixed per-pair evaluation with exact identity-based resume."""
    output = _approved_output(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = Path(manifest_path).resolve(strict=True)
    _, load_manifest, performance_labels, _ = _metric_modules()
    pairs = tuple(load_manifest(manifest))
    source_before = _source_stat_snapshot(pairs)
    variant_id = "dynamic_int8_linear" if quantized else "full_precision"
    runtime_identity = _runtime_identity()
    code_identity = _code_identity()
    if spec.family == "basic_pitch" and quantized:
        load = getattr(adapter, "load", None)
        if callable(load):
            load()
    elif spec.family == "basic_pitch" and not quantized:
        load = getattr(adapter, "load", None)
        if callable(load):
            load()
        return _consume_landed_basic_pitch(spec, adapter, manifest, pairs, output, runtime_identity, code_identity)
    if not spec.is_available:
        run = _model_run(spec, variant_id, manifest, "unavailable", failure_code="dependency_unavailable")
        _write_unavailable(output, run, reason=spec.unavailability_reason)
        return run
    if spec.family != "basic_pitch":
        load = getattr(adapter, "load", None)
        try:
            if callable(load):
                load()
        except ArtifactUnavailable as exc:
            run = _model_run(spec, variant_id, manifest, "failed", failure_code="dependency_unavailable")
            _write_unavailable(output, run, reason=str(exc))
            return run
        except (ArtifactError, OSError, RuntimeError, TypeError, ValueError) as exc:
            run = _model_run(spec, variant_id, manifest, "failed", failure_code="model_load_failed")
            _write_unavailable(output, run, reason=str(exc))
            return run
    quantization_info: dict[str, Any] | None = None
    if quantized:
        from .quantization import quantize_dynamic_linear_int8

        try:
            quantization = getattr(adapter, "quantization_result", None)
            if callable(quantization):
                result = quantization()
            else:
                model = getattr(adapter, "model", None)
                if model is None:
                    raise RuntimeError("quantization source model is unavailable")
                result = quantize_dynamic_linear_int8(model)
            quantization_info = {
                "status": str(result.status),
                "original_linear_modules": int(result.original_linear_modules),
                "quantized_linear_modules": int(result.quantized_linear_modules),
                "engine": result.engine,
            }
            if result.status != "ok" or result.model is None or result.quantized_linear_modules < 1:
                raise RuntimeError(str(result.status))
            bind = getattr(adapter, "bind_model", None)
            if not callable(bind):
                raise TypeError("adapter cannot bind the quantized model")
            bind(result.model)
            active_model = getattr(adapter, "active_model", None)
            if active_model is not result.model:
                raise RuntimeError("adapter did not bind the quantized model")
        except (ArtifactUnavailable, RuntimeError, TypeError, ValueError) as exc:
            failure_code = "quantization_unsupported" if quantization_info is None or quantization_info.get("status") != "ok" else "quantized_runtime_failed"
            run = _model_run(spec, variant_id, manifest, "failed", failure_code=failure_code)
            _write_unavailable(output, run, reason=str(exc))
            return run
    adapter_identity = _adapter_identity(adapter)
    _atomic_json(
        output / "model_lock.json",
        {
            "format_version": 1,
            "model_id": spec.model_id,
            "model_identity": spec.identity_digest(),
            "variant_id": variant_id,
            "adapter_identity": adapter_identity,
            "stock_inference": dict(spec.stock_inference),
            "runtime_identity": runtime_identity,
            "code_identity": code_identity,
            "checkpoint_identity_status": spec.checkpoint_identity_status,
            "quantization": quantization_info,
        },
    )
    if not pairs:
        run = _model_run(spec, variant_id, manifest, "unavailable", failure_code="no_eligible_pairs")
        _write_unavailable(output, run, reason="the exact landed #25 manifest contains no eligible pairs")
        return run
    rows: list[dict[str, Any]] = []
    references: dict[str, Sequence[Any]] = {}
    frame_model = spec.output_contract == "frame_pitch"
    pairs_dir = output / "pairs"
    for pair in pairs:
        reference, _ = performance_labels(pair.midi_path)
        references[pair.pair_id] = reference
        resume_identity, resume_details = _pair_resume_identity(
            spec,
            variant_id,
            adapter,
            _digest(manifest),
            runtime_identity,
            pair.pair_id,
            code_identity,
        )
        cached = None if force else _load_cached_pair(pairs_dir / f"{pair.pair_id}.json", resume_identity, pair.pair_id, resume_details)
        if cached is not None:
            row = cached
            rows.append(row)
            continue
        try:
            value = adapter.transcribe(pair.audio_path)
            output_value = validate_output(value)
            metrics = score_output(reference, output_value)
            row = {
                "pair_id": pair.pair_id,
                "preset_id": pair.preset_id,
                "labels": pair.labels,
                "status": "ok",
                "failure_code": None,
                "metrics": metrics,
                "frame_count": output_value.frame_pitch.active_midi.shape[0] if output_value.frame_pitch is not None else 0,
            }
        except MemoryError:
            row = {"pair_id": pair.pair_id, "preset_id": pair.preset_id, "labels": pair.labels, "status": "out_of_memory", "failure_code": "out_of_memory", "frame_count": _frame_count(pair.audio_path) if frame_model else 0}
        except ValueError as exc:
            row = {"pair_id": pair.pair_id, "preset_id": pair.preset_id, "labels": pair.labels, "status": "invalid_native_output", "failure_code": "invalid_native_output", "error_type": type(exc).__name__, "frame_count": _frame_count(pair.audio_path) if frame_model else 0}
        except Exception as exc:  # noqa: BLE001 - pair failure must not discard later pairs
            row = {"pair_id": pair.pair_id, "preset_id": pair.preset_id, "labels": pair.labels, "status": "runtime_failed", "failure_code": "transcription_runtime_error", "error_type": type(exc).__name__, "frame_count": _frame_count(pair.audio_path) if frame_model else 0}
        row["resume_identity"] = resume_details
        row["resume_identity_digest"] = resume_identity
        rows.append(row)
        _atomic_json(pairs_dir / f"{row['pair_id']}.json", row)
    if _source_stat_snapshot(pairs) != source_before:
        raise ArtifactError("a #25 source artifact changed during #26 evaluation")
    views = build_quality_views(rows, references, frame_model=frame_model)
    successful = sum(row["status"] == "ok" for row in rows)
    run = _model_run(spec, variant_id, manifest, "ok", successful_pairs=successful, failed_pairs=len(rows) - successful)
    _atomic_json(
        output / "run.json",
        {
            "format_version": 1,
            "status": run.status,
            "failure_code": None,
            "model_id": run.model_id,
            "variant_id": run.variant_id,
            "manifest_sha256": run.manifest_identity,
            "model_identity": run.model_identity,
            "successful_pairs": successful,
            "failed_pairs": len(rows) - successful,
            "frame_model": frame_model,
            "adapter_identity": adapter_identity,
            "runtime_identity": runtime_identity,
            "code_identity": code_identity,
            "metric_contract": "issue_25_note_frame_metrics_v1",
            "quantization": quantization_info,
        },
    )
    _atomic_json(output / "aggregates.json", {"status": "ok", "quality": views})
    return run

"""Evaluation orchestration that imports the landed #25 metric stack."""

from __future__ import annotations

import hashlib
import json
import math
import os
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
        from obruxo_basic_pitch.constants import FFT_HOP
        from scipy.io import wavfile

        _, audio = wavfile.read(audio_path)
        return max(0, math.ceil(np.asarray(audio).shape[0] / FFT_HOP))
    except (ImportError, OSError, ValueError, TypeError):
        return 0


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


def evaluate_model(
    spec: ModelSpec,
    adapter: object,
    manifest_path: Path,
    output_dir: Path,
    *,
    quantized: bool = False,
    force: bool = False,
) -> EvaluationRun:
    """Run fixed per-pair evaluation, or persist a truthful unavailable state."""
    output = _approved_output(output_dir)
    if output.exists() and not force and any(output.iterdir()):
        raise FileExistsError("refusing to overwrite evaluation output without force=True")
    output.mkdir(parents=True, exist_ok=True)
    manifest = Path(manifest_path).resolve(strict=True)
    _, load_manifest, performance_labels, _ = _metric_modules()
    pairs = tuple(load_manifest(manifest))
    if not spec.is_available:
        run = _model_run(spec, "dynamic_int8_linear" if quantized else "full_precision", manifest, "unavailable", failure_code="dependency_unavailable")
        _write_unavailable(output, run, reason=spec.unavailability_reason)
        return run
    variant_id = "dynamic_int8_linear" if quantized else "full_precision"
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
    adapter_identity = f"{type(adapter).__module__}.{type(adapter).__qualname__}"
    _atomic_json(
        output / "model_lock.json",
        {
            "format_version": 1,
            "model_id": spec.model_id,
            "model_identity": spec.identity_digest(),
            "variant_id": variant_id,
            "adapter_identity": adapter_identity,
            "stock_inference": dict(spec.stock_inference),
        },
    )
    if not pairs:
        run = _model_run(spec, variant_id, manifest, "unavailable", failure_code="no_eligible_pairs")
        _write_unavailable(output, run, reason="the exact landed #25 manifest contains no eligible pairs")
        return run
    try:
        if quantized:
            from .quantization import quantize_dynamic_linear_int8

            model = getattr(adapter, "model", None)
            if model is None:
                raise ArtifactUnavailable("quantization source model is unavailable")
            quantized_result = quantize_dynamic_linear_int8(model)
            if quantized_result.status != "ok":
                raise ArtifactUnavailable(quantized_result.status)
    except ArtifactUnavailable as exc:
        run = _model_run(spec, variant_id, manifest, "failed", failure_code="dependency_unavailable")
        _write_unavailable(output, run, reason=str(exc))
        return run
    rows: list[dict[str, Any]] = []
    references: dict[str, Sequence[Any]] = {}
    frame_model = spec.output_contract == "frame_pitch"
    pairs_dir = output / "pairs"
    for pair in pairs:
        try:
            reference, _ = performance_labels(pair.midi_path)
            references[pair.pair_id] = reference
            value = adapter.transcribe(pair.audio_path)
            output_value = validate_output(value)
            metrics = score_output(reference, output_value)
            rows.append(
                {
                    "pair_id": pair.pair_id,
                    "preset_id": pair.preset_id,
                    "labels": pair.labels,
                    "status": "ok",
                    "failure_code": None,
                    "metrics": metrics,
                    "frame_count": output_value.frame_pitch.active_midi.shape[0] if output_value.frame_pitch is not None else 0,
                }
            )
        except MemoryError:
            rows.append({"pair_id": pair.pair_id, "preset_id": pair.preset_id, "labels": pair.labels, "status": "out_of_memory", "failure_code": "out_of_memory", "frame_count": _frame_count(pair.audio_path) if frame_model else 0})
        except ValueError as exc:
            rows.append({"pair_id": pair.pair_id, "preset_id": pair.preset_id, "labels": pair.labels, "status": "invalid_native_output", "failure_code": "invalid_native_output", "error_type": type(exc).__name__, "frame_count": _frame_count(pair.audio_path) if frame_model else 0})
        except Exception as exc:  # noqa: BLE001 - pair failure must not discard later pairs
            rows.append({"pair_id": pair.pair_id, "preset_id": pair.preset_id, "labels": pair.labels, "status": "runtime_failed", "failure_code": "transcription_runtime_error", "error_type": type(exc).__name__, "frame_count": _frame_count(pair.audio_path) if frame_model else 0})
        row = rows[-1]
        row["resume_identity"] = hashlib.sha256(
            json.dumps(
                {
                    "pair_id": pair.pair_id,
                    "model_identity": spec.identity_digest(),
                    "variant_id": variant_id,
                    "adapter_identity": adapter_identity,
                    "stock_inference": dict(spec.stock_inference),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        _atomic_json(pairs_dir / f"{row['pair_id']}.json", row)
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
        },
    )
    _atomic_json(output / "aggregates.json", {"status": "ok", "quality": views})
    return run

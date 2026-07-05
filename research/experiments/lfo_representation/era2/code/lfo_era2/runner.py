"""Experiment 11 runner, status, and artifact orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import platform
import re
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable

import numpy as np

from .accelerator import BackendPreference, xpu_available
from .analytics import analyze_run
from .assets import DecoderPolicy
from .accounting import RuntimeInterfaceSpec
from .contracts import TopologyFlags, find_stage_keys, validate_topology_contract
from .dataset import Era2CurveDataset, TOPOLOGY_NAMES, load_presetshare_curve_dataset
from .flat import PhaseSearchSpec, construct_flat_assets_from_curves, decode_flat, encode_flat
from .manifest import ExperimentRowManifest, write_json, write_summary_csv
from .metrics import flat_atom_usage, reconstruction_summary


@dataclass(frozen=True)
class ExperimentRowSpec:
    row_id: str
    D: int
    W: int
    budget_band: str
    base_dictionary_size: int = 32
    oracle_phase_search_policy: str = "fft_lattice"
    oracle_phase_candidate_count: int | None = None
    resolution: int = 97
    train_count: int | None = None
    validation_count: int | None = None
    seed: int = 20260704
    backend: BackendPreference = "auto"
    chunk_size: int = 256


@dataclass(frozen=True)
class ExperimentRunManifest:
    run_id: str
    screen: str
    smoke: bool
    corpus_sample_fraction_requested: float
    row_count: int
    started_at_utc: str
    dataset: dict[str, Any]
    environment: dict[str, Any]
    rows: list[dict[str, Any]] = field(default_factory=list)


ERA2_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ERA2_ROOT.parents[3]
DEFAULT_METADATA = REPO_ROOT / "datasets" / "presetshare" / "raw" / "presetshare_vital_metadata.csv"
DEFAULT_RUN_ROOT = ERA2_ROOT / "artifacts" / "experiment_11" / "runs"
SMOKE_TRAIN_COUNT = 48
SMOKE_VALIDATION_COUNT = 32


def experiment11_row_specs(
    *,
    backend: BackendPreference = "auto",
    oracle_phase_search_policy: str = "fft_lattice",
    oracle_phase_candidate_count: int | None = None,
) -> list[ExperimentRowSpec]:
    rows = [
        ("w4_d48", 48, 4, "small"),
        ("w6_d32", 32, 6, "small"),
        ("w8_d28", 28, 8, "small"),
        ("w4_d120", 120, 4, "medium"),
        ("w6_d80", 80, 6, "medium"),
        ("w8_d72", 72, 8, "medium"),
    ]
    return [
        ExperimentRowSpec(
            row_id=row_id,
            D=D,
            W=W,
            budget_band=band,
            backend=backend,
            oracle_phase_search_policy=oracle_phase_search_policy,
            oracle_phase_candidate_count=oracle_phase_candidate_count,
        )
        for row_id, D, W, band in rows
    ]


def run_experiment11_screen(
    *,
    backend: BackendPreference = "auto",
    smoke: bool = False,
    corpus_sample_fraction: float = 1.0,
    run_dir: Path | None = None,
    resume: bool = False,
    rerun_failed: bool = False,
    metadata_path: Path = DEFAULT_METADATA,
    oracle_phase_search_policy: str = "fft_lattice",
    oracle_phase_candidate_count: int | None = None,
    dataset: Era2CurveDataset | None = None,
    row_specs: list[ExperimentRowSpec] | None = None,
    analyze: bool = True,
    monitor: Callable[[Path], None] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    _validate_corpus_size_args(smoke=smoke, corpus_sample_fraction=corpus_sample_fraction)
    specs = row_specs or experiment11_row_specs(
        backend=backend,
        oracle_phase_search_policy=oracle_phase_search_policy,
        oracle_phase_candidate_count=oracle_phase_candidate_count,
    )
    if not specs:
        raise ValueError("at least one row spec is required")
    run_dir = _resolve_run_dir(run_dir)
    if run_dir.exists() and not resume:
        raise FileExistsError(f"run directory already exists; pass resume=True to continue: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    rows_dir = run_dir / "rows"
    rows_dir.mkdir(exist_ok=True)

    resolution = specs[0].resolution
    _log(
        progress,
        "run_prepare "
        f"smoke={smoke} corpus_sample_fraction={corpus_sample_fraction} backend={backend} rows={len(specs)} "
        f"run_dir={run_dir} control_point_count={resolution}",
    )
    _log(progress, f"dataset_load_start metadata={metadata_path} resolution={resolution}")
    if dataset is None:
        dataset = load_presetshare_curve_dataset(
            metadata_path,
            resolution=resolution,
            progress=lambda message: _log(progress, message),
        )
    train_count, validation_count = _requested_dataset_counts(
        dataset,
        smoke=smoke,
        corpus_sample_fraction=corpus_sample_fraction,
    )
    dataset = dataset.subset(train_count=train_count, validation_count=validation_count)
    _log(progress, "dataset_ready " + _format_dataset_fields(dataset.manifest_fields()))

    run_id = run_dir.name
    status = _load_status(run_dir) or _initial_status(run_id, smoke, corpus_sample_fraction, specs)
    manifest = ExperimentRunManifest(
        run_id=run_id,
        screen="experiment11",
        smoke=smoke,
        corpus_sample_fraction_requested=float(corpus_sample_fraction),
        row_count=len(specs),
        started_at_utc=status["started_at_utc"],
        dataset=dataset.manifest_fields(),
        environment=_environment_fields(),
        rows=[asdict(spec) for spec in specs],
    )
    write_json(run_dir / "run_manifest.json", asdict(manifest))
    _event(
        run_dir,
        status,
        "run_start",
        message=f"smoke={smoke} corpus_sample_fraction={corpus_sample_fraction} rows={len(specs)} backend={backend}",
        monitor=monitor,
        progress=progress,
    )

    for row_index, spec in enumerate(specs, start=1):
        row_dir = rows_dir / spec.row_id
        existing = status["rows"].get(spec.row_id, {})
        if existing.get("status") == "completed":
            _event(
                run_dir,
                status,
                "row_skipped",
                row_id=spec.row_id,
                message=f"already completed {row_index}/{len(specs)}",
                monitor=monitor,
                progress=progress,
            )
            continue
        if existing.get("status") == "failed" and not rerun_failed:
            _event(
                run_dir,
                status,
                "row_skipped",
                row_id=spec.row_id,
                message=f"previously failed {row_index}/{len(specs)}",
                monitor=monitor,
                progress=progress,
            )
            continue
        try:
            status["current_row_id"] = spec.row_id
            status["current_phase"] = "running"
            status["current_row_number"] = row_index
            _set_current_task(status, spec.row_id, row_index, 0.0, "starting")
            status["rows"][spec.row_id] = {"status": "running", "started_at_utc": _now()}
            _set_overall_progress(status)
            _write_status(run_dir, status)
            _event(
                run_dir,
                status,
                "row_start",
                row_id=spec.row_id,
                message=_row_start_message(spec, row_index, len(specs)),
                monitor=monitor,
                progress=progress,
            )

            def row_progress(message: str) -> None:
                status["current_phase"] = message
                progress_info = _row_progress_from_message(
                    message,
                    residual_layer_count=spec.D,
                    previous_percent=status.get("current_task_percent"),
                    previous_phase=status.get("current_task_phase"),
                )
                status["current_task_percent"] = progress_info["percent"]
                status["current_task_phase"] = progress_info["phase"]
                _event(
                    run_dir,
                    status,
                    "row_phase",
                    row_id=spec.row_id,
                    message=message,
                    monitor=monitor,
                    progress=progress,
                )

            summary = _run_row(
                spec,
                dataset.subset(train_count=spec.train_count, validation_count=spec.validation_count),
                row_dir,
                smoke=smoke,
                corpus_sample_fraction=corpus_sample_fraction,
                progress=row_progress,
            )
            status["rows"][spec.row_id] = {
                "status": "completed",
                "completed_at_utc": _now(),
                "head_outputs_actual": summary["head_outputs_actual"],
                "validation_p95_rmse": summary["validation_p95_rmse"],
                "row_elapsed_seconds": summary["row_elapsed_seconds"],
            }
            _set_overall_progress(status)
            _set_current_task(status, spec.row_id, row_index, 100.0, "complete")
            _event(
                run_dir,
                status,
                "row_complete",
                row_id=spec.row_id,
                backend_used=summary.get("backend_used"),
                message=_row_complete_message(summary, row_index, len(specs)),
                monitor=monitor,
                progress=progress,
            )
        except Exception as exc:
            status["rows"][spec.row_id] = {"status": "failed", "failed_at_utc": _now(), "error": str(exc)}
            _set_overall_progress(status)
            status["current_task_phase"] = f"failed: {status.get('current_task_phase') or status.get('current_phase') or 'unknown'}"
            _event(
                run_dir,
                status,
                "row_failed",
                row_id=spec.row_id,
                message=str(exc),
                monitor=monitor,
                progress=progress,
            )
            _write_status(run_dir, status)
            raise
        _write_status(run_dir, status)

    status["current_row_id"] = ""
    status["current_phase"] = "complete"
    status["current_row_number"] = ""
    status["current_task_id"] = ""
    status["current_task_number"] = ""
    status["current_task_percent"] = 100.0
    status["current_task_phase"] = "complete"
    status["completed_at_utc"] = _now()
    _set_overall_progress(status)
    _write_status(run_dir, status)
    _log(progress, "analytics_start" if analyze else "analytics_skipped")
    analytics = analyze_run(run_dir) if analyze else {}
    _event(
        run_dir,
        status,
        "run_complete",
        message=str(analytics.get("summary", "")),
        monitor=monitor,
        progress=progress,
    )
    return {"run_dir": str(run_dir), "status": status, "analytics": analytics}


def status_text(run_dir: Path) -> str:
    status_path = Path(run_dir) / "run_status.json"
    if not status_path.exists():
        raise FileNotFoundError(f"missing status file: {status_path}")
    status = _read_json_with_retry(status_path)
    rows = status.get("rows", {})
    total = int(status.get("row_count", len(rows)))
    completed = sum(1 for row in rows.values() if row.get("status") == "completed")
    failed = sum(1 for row in rows.values() if row.get("status") == "failed")
    skipped = sum(1 for row in rows.values() if row.get("status") == "skipped")
    last = _last_event(Path(run_dir))
    elapsed = _elapsed_since(status.get("started_at_utc", ""))
    row_order = status.get("row_order", [])
    pending = [
        row_id
        for row_id in row_order
        if rows.get(row_id, {}).get("status") not in {"completed", "skipped"}
    ]
    mode = "smoke" if status.get("smoke") else "full"
    lines = [
        f"run_id={status.get('run_id', Path(run_dir).name)} mode={mode} corpus_sample_fraction={status.get('corpus_sample_fraction_requested', '')} elapsed={_format_duration(elapsed)}",
        f"Overall: {completed}/{total} rows complete ({_percent(completed, total)}) failed={failed} skipped={skipped}",
        "Current: " + _current_task_status_line(status),
        f"started_at_utc={status.get('started_at_utc', '')}",
    ]
    if status.get("completed_at_utc"):
        lines.append(f"completed_at_utc={status['completed_at_utc']}")
    if last:
        lines.append(f"Last event: {last.get('event', '')} - {last.get('message', '')}")
    if pending:
        lines.append("next_pending=" + ", ".join(pending[:5]))
    return "\n".join(lines)


def _run_row(
    spec: ExperimentRowSpec,
    dataset: Era2CurveDataset,
    row_dir: Path,
    *,
    smoke: bool = False,
    corpus_sample_fraction: float = 1.0,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    row_started = time.perf_counter()
    row_dir.mkdir(parents=True, exist_ok=True)
    flags = TopologyFlags()
    contract = validate_topology_contract(flags)
    if not contract.passed:
        raise ValueError(f"topology contract failed: {contract.violations}")
    phase_search = PhaseSearchSpec(
        policy=spec.oracle_phase_search_policy,
        candidate_count=spec.oracle_phase_candidate_count,
    )
    phase_candidate_count = phase_search.resolved_candidate_count(spec.resolution)
    if phase_candidate_count <= 1 and not smoke:
        raise ValueError(
            "Experiment 11 canonical rows count phase scalar outputs; "
            "oracle phase search must have more than one candidate"
        )

    construction_started = time.perf_counter()
    _log(progress, "construction_start")
    assets = construct_flat_assets_from_curves(
        dataset.train_curves,
        base_dictionary_size=spec.base_dictionary_size,
        residual_layer_count=spec.D,
        width=spec.W,
        backend=spec.backend,
        chunk_size=spec.chunk_size,
        phase_search=phase_search,
        progress=progress,
    )
    construction_time = time.perf_counter() - construction_started
    _log(progress, f"construction_complete elapsed={construction_time:.2f}s")

    train_started = time.perf_counter()
    _log(progress, "train_encoding_start")
    train_encoded = encode_flat(
        dataset.train_curves,
        assets,
        phase_search=phase_search,
        backend=spec.backend,
        chunk_size=spec.chunk_size,
        progress=progress,
        progress_label="train encoding",
    )
    train_encoding_time = time.perf_counter() - train_started
    _log(progress, f"train_encoding_complete elapsed={train_encoding_time:.2f}s backend={train_encoded.backend_used}")
    train_reconstructed = decode_flat(assets, train_encoded.encoding, decoder_policy=DecoderPolicy())

    validation_started = time.perf_counter()
    _log(progress, "validation_encoding_start")
    validation_encoded = encode_flat(
        dataset.validation_curves,
        assets,
        phase_search=phase_search,
        backend=spec.backend,
        chunk_size=spec.chunk_size,
        progress=progress,
        progress_label="validation encoding",
    )
    validation_encoding_time = time.perf_counter() - validation_started
    _log(progress, f"validation_encoding_complete elapsed={validation_encoding_time:.2f}s backend={validation_encoded.backend_used}")
    validation_reconstructed = decode_flat(assets, validation_encoded.encoding, decoder_policy=DecoderPolicy())

    runtime_spec = RuntimeInterfaceSpec(
        addressing_scheme="flat_categorical",
        residual_layer_count=spec.D,
        dictionary_scope="per_residual_layer",
        parameters={"width": spec.W},
    )
    budget = runtime_spec.budget(base_dictionary_size=spec.base_dictionary_size)
    schema = validation_encoded.encoding.target_schema()
    schema_stage_keys = find_stage_keys(schema)
    if schema_stage_keys:
        raise ValueError(f"target schema uses old stage terminology: {schema_stage_keys}")

    _log(progress, "metrics_start")
    manifest = ExperimentRowManifest(
        experiment_id="experiment_11",
        oracle_construction_id="topology_blind_observed_residual_stack_v1",
        runtime_interface_id="flat_categorical_per_residual_layer",
        decoder_policy_id="final_clip",
        base_dictionary_size=spec.base_dictionary_size,
        residual_layer_count=spec.D,
        scalar_families=["phase"],
        dictionary_scope=assets.dictionary_scope,
        codebook_storage_count=assets.codebook_storage_count,
        budget=budget,
        topology_flags=flags,
        lfo_control_point_count=spec.resolution,
        oracle_construction_time=construction_time,
        oracle_encoding_time=validation_encoding_time,
        method_parameters={
            "row_id": spec.row_id,
            "budget_band": spec.budget_band,
            "W": spec.W,
            "W_by_residual_layer": assets.residual_widths(),
            **phase_search.as_manifest_fields(spec.resolution),
            "resolution": spec.resolution,
            "fixed_x_grid_note": "LFO x-grid geometry is decoder-owned and adds zero model prediction head outputs.",
            "smoke": smoke,
            "corpus_sample_fraction_requested": float(corpus_sample_fraction),
            "train_count": len(dataset.train_indices),
            "validation_count": len(dataset.validation_indices),
            "seed": spec.seed,
            "backend_preference": spec.backend,
            "backend_used": sorted(set(train_encoded.backend_used + validation_encoded.backend_used)),
            "schema_stage_key_violations": schema_stage_keys,
            **assets.metadata,
            **dataset.manifest_fields(),
        },
        notes="Experiment 11 topology-free flat-categorical row.",
    )
    train_metrics = _prefix("train", reconstruction_summary(dataset.train_curves, train_reconstructed))
    validation_metrics = _prefix("validation", reconstruction_summary(dataset.validation_curves, validation_reconstructed))
    usage = _prefix(
        "validation",
        flat_atom_usage(
            validation_encoded.encoding.as_arrays(),
            residual_layer_count=spec.D,
            widths_by_residual_layer=assets.residual_widths(),
        ),
    )
    summary = {
        **manifest.as_dict(),
        **train_metrics,
        **validation_metrics,
        **usage,
        **_topology_bucket_metrics(dataset, validation_reconstructed),
        "topology_contract_pass": contract.passed,
        "train_encoding_time": train_encoding_time,
        "validation_encoding_time": validation_encoding_time,
        "row_elapsed_seconds": time.perf_counter() - row_started,
    }
    write_json(row_dir / "manifest.json", manifest.as_dict())
    write_json(row_dir / "targets_schema.json", schema)
    write_json(row_dir / "topology_contract.json", contract.as_dict())
    write_summary_csv(row_dir / "summary.csv", summary)
    _log(progress, "metrics_complete")
    return summary


def _topology_bucket_metrics(dataset: Era2CurveDataset, reconstructed: np.ndarray) -> dict[str, Any]:
    from .metrics import rmse_per_curve

    rmse = rmse_per_curve(dataset.validation_curves, reconstructed)
    labels = dataset.topology[dataset.validation_indices]
    result: dict[str, Any] = {}
    medians = []
    p95s = []
    for index, name in enumerate(TOPOLOGY_NAMES):
        values = rmse[labels == index]
        if len(values):
            median = float(np.median(values))
            p95 = float(np.quantile(values, 0.95))
            medians.append(median)
            p95s.append(p95)
            result[f"analysis_only_topology_{name}_median_rmse"] = median
            result[f"analysis_only_topology_{name}_p95_rmse"] = p95
    if p95s:
        result["analysis_only_topology_p95_gap"] = float(max(p95s) - min(p95s))
    return result


def _initial_status(run_id: str, smoke: bool, corpus_sample_fraction: float, specs: list[ExperimentRowSpec]) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "smoke": smoke,
        "corpus_sample_fraction_requested": float(corpus_sample_fraction),
        "row_count": len(specs),
        "overall_tasks_completed": 0,
        "overall_tasks_total": len(specs),
        "overall_tasks_percent": 0.0,
        "current_task_id": "",
        "current_task_number": "",
        "current_task_percent": 0.0,
        "current_task_phase": "created",
        "started_at_utc": _now(),
        "completed_at_utc": "",
        "current_row_id": "",
        "current_row_number": "",
        "current_phase": "created",
        "row_order": [spec.row_id for spec in specs],
        "rows": {},
    }


def _set_overall_progress(status: dict[str, Any]) -> None:
    rows = status.get("rows", {})
    total = int(status.get("row_count", len(rows)))
    completed = sum(1 for row in rows.values() if row.get("status") == "completed")
    status["overall_tasks_completed"] = completed
    status["overall_tasks_total"] = total
    status["overall_tasks_percent"] = _numeric_percent(completed, total)


def _set_current_task(
    status: dict[str, Any],
    row_id: str,
    row_index: int | str,
    percent: float,
    phase: str,
) -> None:
    status["current_task_id"] = row_id
    status["current_task_number"] = row_index
    status["current_task_percent"] = round(float(percent), 1)
    status["current_task_phase"] = phase


def _row_progress_from_message(
    message: str,
    *,
    residual_layer_count: int,
    previous_percent: Any = None,
    previous_phase: Any = None,
) -> dict[str, Any]:
    message = str(message)
    percent = _float_or_none(previous_percent)
    phase = _readable_phase(message) or str(previous_phase or "working")

    residual = re.search(r"^(construction|train encoding|validation encoding): residual layer (\d+)/(\d+)", message)
    if residual:
        family, index_text, total_text = residual.groups()
        index = int(index_text)
        total = max(1, int(total_text))
        phase_fraction = max(0.0, min(1.0, index / total))
        start, end = _phase_bounds(family)
        percent = start + (end - start) * phase_fraction
        return {"percent": round(percent, 1), "phase": f"{family} residual layer {index}/{total}"}

    if message.startswith("construction_start"):
        return {"percent": 0.0, "phase": "construction"}
    if message.startswith("construction: base"):
        return {"percent": 1.0, "phase": "construction base choice"}
    if message.startswith("construction_complete"):
        return {"percent": 80.0, "phase": "construction complete"}
    if message.startswith("train_encoding_start"):
        return {"percent": 80.0, "phase": "train encoding"}
    if message.startswith("train encoding: base"):
        return {"percent": 80.5, "phase": "train encoding base choice"}
    if message.startswith("train_encoding_complete"):
        return {"percent": 94.0, "phase": "train encoding complete"}
    if message.startswith("validation_encoding_start"):
        return {"percent": 94.0, "phase": "validation encoding"}
    if message.startswith("validation encoding: base"):
        return {"percent": 94.2, "phase": "validation encoding base choice"}
    if message.startswith("validation_encoding_complete"):
        return {"percent": 99.0, "phase": "validation encoding complete"}
    if message.startswith("metrics_start"):
        return {"percent": 99.0, "phase": "metrics"}
    if message.startswith("metrics_complete"):
        return {"percent": 100.0, "phase": "metrics complete"}

    return {
        "percent": round(percent if percent is not None else 0.0, 1),
        "phase": phase,
    }


def _phase_bounds(family: str) -> tuple[float, float]:
    if family == "construction":
        return 0.0, 80.0
    if family == "train encoding":
        return 80.0, 94.0
    if family == "validation encoding":
        return 94.0, 99.0
    return 0.0, 100.0


def _readable_phase(message: str) -> str:
    cleaned = message.split(" elapsed=", 1)[0].split(" backend=", 1)[0]
    cleaned = cleaned.replace("_", " ").replace(":", "")
    return " ".join(cleaned.split())


def _current_task_status_line(status: dict[str, Any]) -> str:
    task_id = str(status.get("current_task_id") or status.get("current_row_id") or "")
    phase = str(status.get("current_task_phase") or status.get("current_phase") or "")
    if not task_id:
        if status.get("current_phase") == "complete" or phase == "complete":
            return "complete"
        return phase or "waiting"

    task_number = status.get("current_task_number") or status.get("current_row_number") or ""
    total = status.get("row_count", "")
    task_status = status.get("rows", {}).get(task_id, {}).get("status", "")
    percent = _format_numeric_percent(status.get("current_task_percent"))
    prefix = f"{task_id}"
    if task_number != "":
        prefix += f" row {task_number}/{total}"
    if task_status == "failed":
        if phase.startswith("failed: "):
            phase = phase[len("failed: ") :]
        return f"{prefix} failed at {percent} - {phase}"
    return f"{prefix} {percent} - {phase}"


def _numeric_percent(done: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(100.0 * done / total, 1)


def _format_numeric_percent(value: Any) -> str:
    number = _float_or_none(value)
    if number is None:
        return "n/a"
    return f"{number:.1f}%"


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _event(
    run_dir: Path,
    status: dict[str, Any],
    event: str,
    *,
    row_id: str = "",
    backend_used: Any = "",
    message: str = "",
    monitor: Callable[[Path], None] | None = None,
    progress: Callable[[str], None] | None = None,
) -> None:
    payload = {
        "timestamp_utc": _now(),
        "run_id": status.get("run_id", run_dir.name),
        "row_id": row_id,
        "event": event,
        "phase": status.get("current_phase", ""),
        "backend_used": backend_used,
        "message": message,
    }
    with (run_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    _write_status(run_dir, status)
    _log(progress, _event_line(payload))
    if monitor is not None:
        monitor(run_dir)


def _load_status(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "run_status.json"
    if not path.exists():
        return None
    return _read_json_with_retry(path)


def _write_status(run_dir: Path, status: dict[str, Any]) -> None:
    write_json(run_dir / "run_status.json", status)


def _last_event(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "events.jsonl"
    if not path.exists():
        return None
    last = ""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                last = line
    return json.loads(last) if last else None


def _read_json_with_retry(path: Path, *, attempts: int = 5, delay_seconds: float = 0.05) -> dict[str, Any]:
    last_error: Exception | None = None
    for _ in range(max(1, attempts)):
        try:
            text = path.read_text(encoding="utf-8")
            if text.strip():
                return json.loads(text)
            last_error = ValueError(f"empty JSON file: {path}")
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
        time.sleep(delay_seconds)
    raise ValueError(f"could not read stable JSON from {path}: {last_error}")


def _resolve_run_dir(run_dir: Path | None) -> Path:
    if run_dir is not None:
        return Path(run_dir)
    return DEFAULT_RUN_ROOT / f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"


def _validate_corpus_size_args(*, smoke: bool, corpus_sample_fraction: float) -> None:
    if not (0.0 < float(corpus_sample_fraction) <= 1.0):
        raise ValueError("corpus_sample_fraction must be in (0, 1]")
    if smoke and float(corpus_sample_fraction) != 1.0:
        raise ValueError("smoke cannot be combined with corpus_sample_fraction other than 1.0")


def _requested_dataset_counts(
    dataset: Era2CurveDataset,
    *,
    smoke: bool,
    corpus_sample_fraction: float,
) -> tuple[int | None, int | None]:
    _validate_corpus_size_args(smoke=smoke, corpus_sample_fraction=corpus_sample_fraction)
    if smoke:
        return SMOKE_TRAIN_COUNT, SMOKE_VALIDATION_COUNT
    if float(corpus_sample_fraction) == 1.0:
        return None, None
    return (
        max(1, int(len(dataset.train_indices) * float(corpus_sample_fraction))),
        max(1, int(len(dataset.validation_indices) * float(corpus_sample_fraction))),
    )


def _environment_fields() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "xpu_available": xpu_available(),
        "git_status": _git_status_short(),
    }


def _git_status_short() -> str:
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return ""
    return result.stdout.strip()


def _prefix(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _log(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _event_line(payload: dict[str, Any]) -> str:
    row = f" row={payload['row_id']}" if payload.get("row_id") else ""
    backend = f" backend={payload['backend_used']}" if payload.get("backend_used") else ""
    message = f" {payload['message']}" if payload.get("message") else ""
    return f"{payload['event']}{row}{backend}{message}".strip()


def _row_start_message(spec: ExperimentRowSpec, row_index: int, total_rows: int) -> str:
    budget = RuntimeInterfaceSpec(
        addressing_scheme="flat_categorical",
        residual_layer_count=spec.D,
        dictionary_scope="per_residual_layer",
        parameters={"width": spec.W},
    ).budget(base_dictionary_size=spec.base_dictionary_size)
    return (
        f"{row_index}/{total_rows} D={spec.D} W={spec.W} "
        f"head_outputs={budget.head_outputs_actual} "
        f"train={spec.train_count if spec.train_count is not None else 'run'} "
        f"validation={spec.validation_count if spec.validation_count is not None else 'run'}"
    )


def _row_complete_message(summary: dict[str, Any], row_index: int, total_rows: int) -> str:
    return (
        f"{row_index}/{total_rows} "
        f"validation_p95_rmse={summary.get('validation_p95_rmse', '')} "
        f"elapsed={float(summary.get('row_elapsed_seconds', 0.0)):.2f}s"
    )


def _format_dataset_fields(fields: dict[str, Any]) -> str:
    return " ".join(f"{key}={value}" for key, value in fields.items())


def _elapsed_since(started_at: Any) -> float | None:
    if not isinstance(started_at, str) or not started_at:
        return None
    try:
        return (datetime.now(timezone.utc) - datetime.fromisoformat(started_at)).total_seconds()
    except ValueError:
        return None


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minute = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minute}m {sec}s"
    if minute:
        return f"{minute}m {sec}s"
    return f"{sec}s"


def _percent(done: int, total: int) -> str:
    if total <= 0:
        return "0.0%"
    return f"{100.0 * done / total:.1f}%"

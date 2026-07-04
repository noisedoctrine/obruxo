"""Experiment 11 runner, status, and artifact orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import platform
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
from .flat import construct_flat_assets_from_curves, decode_flat, encode_flat
from .manifest import ExperimentRowManifest, write_json, write_summary_csv
from .metrics import flat_atom_usage, reconstruction_summary


@dataclass(frozen=True)
class ExperimentRowSpec:
    row_id: str
    D: int
    W: int
    budget_band: str
    base_dictionary_size: int = 32
    phase_bins: int = 1
    resolution: int = 97
    train_count: int = 96
    validation_count: int = 64
    seed: int = 20260704
    backend: BackendPreference = "auto"
    chunk_size: int = 256


@dataclass(frozen=True)
class ExperimentRunManifest:
    run_id: str
    screen: str
    profile: str
    row_count: int
    started_at_utc: str
    dataset: dict[str, Any]
    environment: dict[str, Any]
    rows: list[dict[str, Any]] = field(default_factory=list)


ERA2_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ERA2_ROOT.parents[3]
DEFAULT_METADATA = REPO_ROOT / "datasets" / "presetshare" / "raw" / "presetshare_vital_metadata.csv"
DEFAULT_RUN_ROOT = ERA2_ROOT / "artifacts" / "experiment_11" / "runs"


def experiment11_row_specs(profile: str, *, backend: BackendPreference = "auto") -> list[ExperimentRowSpec]:
    profile = profile.lower()
    if profile == "quick":
        rows = [
            ("q_w4_d48", 48, 4, "small", 48, 32),
            ("q_w6_d32", 32, 6, "small", 48, 32),
            ("q_w8_d28", 28, 8, "small", 48, 32),
            ("q_w4_d120", 120, 4, "medium", 48, 32),
            ("q_w6_d80", 80, 6, "medium", 48, 32),
            ("q_w8_d72", 72, 8, "medium", 48, 32),
        ]
    elif profile == "screen":
        rows = [
            ("s_w4_d48", 48, 4, "small", 192, 128),
            ("s_w6_d32", 32, 6, "small", 192, 128),
            ("s_w8_d28", 28, 8, "small", 192, 128),
            ("s_w4_d120", 120, 4, "medium", 192, 128),
            ("s_w6_d80", 80, 6, "medium", 192, 128),
            ("s_w8_d72", 72, 8, "medium", 192, 128),
        ]
    elif profile == "extended":
        rows = [
            ("x_w4_d48", 48, 4, "small", 384, 256),
            ("x_w6_d32", 32, 6, "small", 384, 256),
            ("x_w8_d28", 28, 8, "small", 384, 256),
            ("x_w4_d120", 120, 4, "medium", 384, 256),
            ("x_w6_d80", 80, 6, "medium", 384, 256),
            ("x_w8_d72", 72, 8, "medium", 384, 256),
            ("x_w8_d128", 128, 8, "large", 384, 256),
            ("x_w12_d80", 80, 12, "large", 384, 256),
            ("x_w16_d60", 60, 16, "large", 384, 256),
        ]
    else:
        raise ValueError("profile must be one of: quick, screen, extended")
    return [
        ExperimentRowSpec(row_id=row_id, D=D, W=W, budget_band=band, train_count=train, validation_count=validation, backend=backend)
        for row_id, D, W, band, train, validation in rows
    ]


def run_experiment11_screen(
    *,
    profile: str = "quick",
    backend: BackendPreference = "auto",
    run_dir: Path | None = None,
    resume: bool = False,
    rerun_failed: bool = False,
    metadata_path: Path = DEFAULT_METADATA,
    dataset: Era2CurveDataset | None = None,
    row_specs: list[ExperimentRowSpec] | None = None,
    analyze: bool = True,
    monitor: Callable[[Path], None] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    specs = row_specs or experiment11_row_specs(profile, backend=backend)
    if not specs:
        raise ValueError("at least one row spec is required")
    run_dir = _resolve_run_dir(run_dir)
    if run_dir.exists() and not resume:
        raise FileExistsError(f"run directory already exists; pass resume=True to continue: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    rows_dir = run_dir / "rows"
    rows_dir.mkdir(exist_ok=True)

    max_train = max(spec.train_count for spec in specs)
    max_validation = max(spec.validation_count for spec in specs)
    resolution = specs[0].resolution
    _log(
        progress,
        "run_prepare "
        f"profile={profile} backend={backend} rows={len(specs)} "
        f"run_dir={run_dir} control_point_count={resolution}",
    )
    _log(progress, f"dataset_load_start metadata={metadata_path} resolution={resolution}")
    if dataset is None:
        dataset = load_presetshare_curve_dataset(
            metadata_path,
            resolution=resolution,
            progress=lambda message: _log(progress, message),
        )
    dataset = dataset.subset(train_count=max_train, validation_count=max_validation)
    _log(progress, "dataset_ready " + _format_dataset_fields(dataset.manifest_fields()))

    run_id = run_dir.name
    status = _load_status(run_dir) or _initial_status(run_id, profile, specs)
    manifest = ExperimentRunManifest(
        run_id=run_id,
        screen="experiment11",
        profile=profile,
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
        message=f"profile={profile} rows={len(specs)} backend={backend}",
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
            status["rows"][spec.row_id] = {"status": "running", "started_at_utc": _now()}
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
                progress=row_progress,
            )
            status["rows"][spec.row_id] = {
                "status": "completed",
                "completed_at_utc": _now(),
                "head_outputs_actual": summary["head_outputs_actual"],
                "validation_p95_rmse": summary["validation_p95_rmse"],
                "row_elapsed_seconds": summary["row_elapsed_seconds"],
            }
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
    status["completed_at_utc"] = _now()
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
    status = json.loads(status_path.read_text(encoding="utf-8"))
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
    lines = [
        f"run_id={status.get('run_id', Path(run_dir).name)} profile={status.get('profile', '')}",
        f"rows completed={completed}/{total} ({_percent(completed, total)}) failed={failed} skipped={skipped}",
        f"elapsed={_format_duration(elapsed)}",
        f"current_row={status.get('current_row_id', '')} row_number={status.get('current_row_number', '')} phase={status.get('current_phase', '')}",
        f"started_at_utc={status.get('started_at_utc', '')}",
    ]
    if status.get("completed_at_utc"):
        lines.append(f"completed_at_utc={status['completed_at_utc']}")
    if last:
        lines.append(f"last_event={last.get('event', '')} row={last.get('row_id', '')} message={last.get('message', '')}")
    if pending:
        lines.append("next_pending=" + ", ".join(pending[:5]))
    return "\n".join(lines)


def _run_row(
    spec: ExperimentRowSpec,
    dataset: Era2CurveDataset,
    row_dir: Path,
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    row_started = time.perf_counter()
    row_dir.mkdir(parents=True, exist_ok=True)
    flags = TopologyFlags()
    contract = validate_topology_contract(flags)
    if not contract.passed:
        raise ValueError(f"topology contract failed: {contract.violations}")

    construction_started = time.perf_counter()
    _log(progress, "construction_start")
    assets = construct_flat_assets_from_curves(
        dataset.train_curves,
        base_dictionary_size=spec.base_dictionary_size,
        residual_layer_count=spec.D,
        width=spec.W,
        backend=spec.backend,
        chunk_size=spec.chunk_size,
        progress=progress,
    )
    construction_time = time.perf_counter() - construction_started
    _log(progress, f"construction_complete elapsed={construction_time:.2f}s")

    train_started = time.perf_counter()
    _log(progress, "train_encoding_start")
    train_encoded = encode_flat(
        dataset.train_curves,
        assets,
        phase_bins=spec.phase_bins,
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
        phase_bins=spec.phase_bins,
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
            "phase_bins": spec.phase_bins,
            "resolution": spec.resolution,
            "fixed_x_grid_note": "LFO x-grid geometry is decoder-owned and adds zero model prediction head outputs.",
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


def _initial_status(run_id: str, profile: str, specs: list[ExperimentRowSpec]) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "profile": profile,
        "row_count": len(specs),
        "started_at_utc": _now(),
        "completed_at_utc": "",
        "current_row_id": "",
        "current_row_number": "",
        "current_phase": "created",
        "row_order": [spec.row_id for spec in specs],
        "rows": {},
    }


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
    return json.loads(path.read_text(encoding="utf-8"))


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


def _resolve_run_dir(run_dir: Path | None) -> Path:
    if run_dir is not None:
        return Path(run_dir)
    return DEFAULT_RUN_ROOT / f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"


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
        f"train={spec.train_count} validation={spec.validation_count}"
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

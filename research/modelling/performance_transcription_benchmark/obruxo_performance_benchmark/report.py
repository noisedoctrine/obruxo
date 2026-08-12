"""Sanitized aggregate report generation; local pair results never cross this boundary."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .artifacts import ModelSpec, load_model_specs

_PRIVATE_KEY_PARTS = ("pair_id", "preset_id", "request_id", "audio_path", "midi_path", "source_root", "local_path", "hostname")
_PRIVATE_TEXT = re.compile(r"(?:^[A-Za-z]:[\\/]|/Users/|/home/|\\Users\\|datasets[\\/]|pair-[0-9a-f]{8,}|preset-[0-9a-f]{8,}|request-[0-9a-f]{8,})", re.IGNORECASE)


class ReportPrivacyError(ValueError):
    """A report attempted to cross the public/private boundary."""


def _assert_public(value: Any, key: str = "") -> None:
    if any(part in key.casefold() for part in _PRIVATE_KEY_PARTS):
        raise ReportPrivacyError(f"private field is not allowed in public report: {key}")
    if isinstance(value, Mapping):
        for child_key, child_value in value.items():
            _assert_public(child_value, str(child_key))
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_public(child, key)
    elif isinstance(value, str) and (_PRIVATE_TEXT.search(value) or "private_" in value.casefold() or "\\tmp\\" in value.casefold()):
        raise ReportPrivacyError("private identifier or machine path is not allowed in public report")


def sanitize_public_report(value: Mapping[str, Any]) -> dict[str, Any]:
    _assert_public(value)
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _approved_report(path: Path | str) -> Path:
    root = (Path(__file__).resolve().parents[1] / "reports").resolve()
    candidate = Path(path).resolve(strict=False)
    if candidate == root or not candidate.is_relative_to(root):
        raise ValueError("public report must be inside the approved reports directory")
    return candidate


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _stored_results(root: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    if not root.is_dir():
        return results
    for run_path in sorted(root.rglob("run.json"), key=lambda path: path.as_posix()):
        run = _read_json(run_path)
        if not run or not isinstance(run.get("model_id"), str):
            continue
        model_id = str(run["model_id"])
        current = results.setdefault(model_id, {})
        if run.get("variant_id") == "dynamic_int8_linear":
            current["quantization"] = run.get("quantization")
            continue
        current["run"] = run
        current["runtime"] = _read_json(run_path.with_name("runtime.json"))
        current["aggregates"] = _read_json(run_path.with_name("aggregates.json"))
    return results


def _landed_basic_pitch_reports() -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    reports = Path(__file__).resolve().parents[2] / "basic_pitch" / "reports"
    return _read_json(reports / "presetshare_baseline.json"), _read_json(reports / "backend_benchmark.json")


def _landed_basic_pitch_quality(report: Mapping[str, Any]) -> dict[str, Any]:
    pairing = report.get("pairing", {})
    aggregate = report.get("aggregate", {})
    eligible = int(aggregate.get("pair_count", pairing.get("eligible_count", 0)))
    successful = int(aggregate.get("successful_pair_count", eligible))
    failed = int(aggregate.get("failed_pair_count", max(0, eligible - successful)))
    view = {
        "eligible_pairs": eligible,
        "successful_pairs": successful,
        "failed_pairs": failed,
        "coverage": float(successful / eligible) if eligible else None,
        "aggregate": dict(aggregate),
    }
    return {"success_only": view, "failure_penalized": dict(view)}


def _landed_basic_pitch_runtime(report: Mapping[str, Any]) -> dict[str, Any]:
    routes = list(report.get("inference", [])) + list(report.get("training", []))
    successful = [route for route in routes if route.get("status") == "ok"]
    failures = [
        {"route": route.get("route"), "status": route.get("status")}
        for route in routes
        if route.get("status") != "ok"
    ]
    value = dict(report)
    value.update(
        {
            "source": "landed_issue_24_report",
            "status": "measured" if successful else "unavailable",
            "failure_code": None,
            "routes": routes,
            "timing_contract": report.get("config"),
            "route_failures": failures,
            "measurement_status": report.get("measurement_status"),
            "openvino_parity_history": report.get("parity_diagnostics"),
            "openvino_precision_diagnostic": report.get("openvino_precision_diagnostic"),
            "reporting_note": "Routes and findings are consumed from the landed #24 report; #26 does not rerun Basic Pitch cost measurements.",
        }
    )
    return value


def build_public_report(specs: Mapping[str, ModelSpec], input_root: Path) -> dict[str, Any]:
    stored = _stored_results(Path(input_root).resolve(strict=False))
    landed_quality, landed_runtime = _landed_basic_pitch_reports()
    models: list[dict[str, Any]] = []
    for model_id, spec in specs.items():
        current = stored.get(model_id, {})
        run = current.get("run") or {}
        runtime = current.get("runtime") or {}
        aggregates = current.get("aggregates") or {}
        stored_quantization = current.get("quantization") or run.get("quantization")
        if model_id == "basic_pitch" and (landed_quality or landed_runtime):
            run = {
                "model_id": model_id,
                "status": (landed_quality or {}).get("status", spec.availability),
                "failure_code": (landed_quality or {}).get("failure_code"),
            }
            runtime = _landed_basic_pitch_runtime(landed_runtime) if landed_runtime else {}
            aggregates = {"quality": _landed_basic_pitch_quality(landed_quality)} if landed_quality else {}
        status = str(run.get("status", spec.availability))
        failure_code = run.get("failure_code") or (spec.unavailability_reason if status != "ok" else None)
        routes = runtime.get("routes")
        if routes is None:
            routes = list(runtime.get("inference", [])) + list(runtime.get("training", []))
        item: dict[str, Any] = {
            "model_id": model_id,
            "family": spec.family,
            "publication_year": spec.publication_year,
            "output_contract": spec.output_contract,
            "identity": spec.public_identity(),
            "status": status,
            "failure_code": failure_code,
            "availability_reason": spec.unavailability_reason,
            "quality": aggregates.get("quality") if status == "ok" else None,
            "quality_provenance": {
                "source": "landed_issue_25_report",
                "backend": (landed_quality or {}).get("backend"),
                "runtime_provenance": (landed_quality or {}).get("runtime_provenance"),
                "category_findings": (landed_quality or {}).get("category_findings"),
            }
            if model_id == "basic_pitch" and landed_quality
            else None,
            "execution": {
                "source": runtime.get("source"),
                "status": runtime.get("status", "unavailable"),
                "failure_code": runtime.get("failure_code"),
                "routes": routes,
                "timing_contract": runtime.get("timing_contract") or runtime.get("config"),
                "phases": runtime.get("phases"),
                "route_failures": runtime.get("route_failures", []),
                "measurement_status": runtime.get("measurement_status"),
                "openvino_parity_history": runtime.get("openvino_parity_history"),
                "openvino_precision_diagnostic": runtime.get("openvino_precision_diagnostic"),
                "reporting_note": runtime.get("reporting_note"),
                "resources": runtime.get("resources") or run.get("resources"),
                "native_batch_sizes": runtime.get("native_batch_sizes", list(spec.native_batch_sizes)),
                "backward": runtime.get("backward"),
            },
            "resources": run.get("resources") or runtime.get("resources"),
            "representation": spec.representation,
            "quantization": stored_quantization,
        }
        if model_id == "basic_pitch" and (landed_quality or landed_runtime):
            item["landed_baseline"] = {
                "quality_status": (landed_quality or {}).get("status"),
                "quality_failure_code": (landed_quality or {}).get("failure_code"),
                "eligible_pairs": ((landed_quality or {}).get("pairing") or {}).get("eligible_count"),
                "quality_coverage": ((landed_quality or {}).get("aggregate") or {}).get("coverage"),
                "cost_status": runtime.get("status"),
                "cost_failure_code": runtime.get("failure_code"),
                "cost_route_failures": runtime.get("route_failures", []),
                "cost_measurement_status": runtime.get("measurement_status"),
                "openvino_gpu_diagnostic_status": ((runtime.get("openvino_precision_diagnostic") or {}).get("status")),
            }
        models.append(item)
    measured_models = [model["model_id"] for model in models if model.get("status") == "ok"]
    unavailable_models = [model["model_id"] for model in models if model.get("status") != "ok"]
    return sanitize_public_report(
        {
            "format_version": 1,
            "comparison": {
                "quality_contract": "landed_issue_25_metrics_and_10000_replicate_seed_0_bootstrap",
                "cost_contract": "landed_issue_24_end_to_end_boundary",
                "quantization_contract": "cpu_dynamic_qint8_ordinary_linear_only",
                "no_composite_winner": True,
                "status": "incomplete_alternatives_unavailable",
            },
            "models": models,
            "evidence": {
                "measured_models": measured_models,
                "metadata_only_models": unavailable_models,
                "measured_scope": "Only Basic Pitch produced executable #24/#25 evidence in the permitted existing runtime and storage. Its quality and cost evidence are inherited, not rerun by #26; the current #24 cost rows include the corrected OpenVINO GPU FP32 + PERFORMANCE route when present.",
                "bounded_diagnostic_scope": "The corrected OpenVINO GPU route also retains a bounded five-window FP32 + PERFORMANCE parity diagnostic; that diagnostic is separate from the corrected route's smoke-benchmark timing and resource measurements.",
                "not_measured_scope": "All alternative-model quality/cost comparisons remain not measured or blocked; no #26 candidate comparison was rerun for this reporting update.",
                "sourced_scope": "Candidate source, checkpoint, representation, architecture boundary, native sample rate, batch semantics, and license fields are verified inventory facts; they are not performance measurements.",
                "unresolved_scope": "No comparative quality, execution cost, resource, backward-cost, or quantization result exists for the unavailable alternatives. The intended comparative benchmark remains incomplete.",
            },
            "conclusion": {
                "measured": "Directly measured evidence exists only for Basic Pitch: #25 quality on 1,769 eligible pairs and #24 route/cost evidence. The current #24 inference comparison includes corrected OpenVINO GPU FP32 + PERFORMANCE startup, throughput, end-to-end, parity, and resource measurements when that route is present.",
                "bounded_diagnostic": "A separate bounded #24 diagnostic compiled OpenVINO GPU with INFERENCE_PRECISION_HINT=float32 while retaining PERFORMANCE and passed parity on five public synthetic windows. This is a parity result, not a performance/resource result.",
                "not_measured": "The alternative-model comparison remains incomplete because the required candidate executions were unavailable; this reporting update did not rerun #25 evaluation or #26 candidate inference.",
                "sourced": "The candidate inventory establishes model identity, representation, architecture boundary, native rate/batch semantics, and licensing where verified, but none of these facts ranks execution quality or cost.",
                "unresolved": "Alternative-model quality, latency, throughput, memory, backward cost, quantization response, and quality-versus-cost trade-offs remain unanswered because those candidates were not executable in the permitted local state.",
                "quality": "No quality score is published for unavailable models or an empty eligible population.",
                "cost": "Cost rows remain separate by route and are unavailable when the fixed smoke input or candidate runtime is unavailable.",
                "representation": "Representation and licensing are inventories, not an integration decision.",
                "later_integration": "No candidate is promoted without executable common-corpus evidence.",
            },
        }
    )


def _legacy_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Performance transcription benchmark",
        "",
        "The JSON report is authoritative. Quality, execution cost, resources, licensing, representation, and quantization are reported separately; no composite winner is computed.",
        "",
        "## Candidate status",
        "",
        "| Model | Family | Status | Quality |",
        "| --- | --- | --- | --- |",
    ]
    for model in report.get("models", []):
        quality = "reported" if model.get("quality") is not None else "unavailable"
        lines.append(f"| `{model.get('model_id', 'unknown')}` | `{model.get('family', 'unknown')}` | `{model.get('status', 'unknown')}` | `{quality}` |")
    lines.extend(
        [
            "",
            "## Contract",
            "",
            f"- Quality: `{report.get('comparison', {}).get('quality_contract', 'unknown')}`",
            f"- Cost: `{report.get('comparison', {}).get('cost_contract', 'unknown')}`",
            f"- Quantization: `{report.get('comparison', {}).get('quantization_contract', 'unknown')}`",
            "- Timbre-Trap remains frame-only; no synthetic note-event decoder is included.",
            "- Private paths, pair identifiers, source filenames, and row predictions are excluded.",
            "",
        ]
    )
    def _metric(value: Any) -> str:
        return "n/a" if value is None else f"{float(value):.6f}"

    lines.extend(["## Quality, cost, and quantization details", ""])
    for model in report.get("models", []):
        model_id = model.get("model_id", "unknown")
        lines.extend([f"### `{model_id}`", ""])
        if model.get("status") != "ok":
            reason = model.get("availability_reason") or model.get("failure_code") or "not recorded"
            lines.append(f"- Availability: `{model.get('status', 'unknown')}` — {reason}.")
        quality = model.get("quality") or {}
        if quality:
            lines.extend(
                [
                    "",
                    "| Quality view | Eligible | Succeeded | Failed | Coverage | Onset+pitch F1 | Onset+pitch+offset F1 | Frame F1 |",
                    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for view_name in ("success_only", "failure_penalized"):
                view = quality.get(view_name) or {}
                micro = (view.get("aggregate") or {}).get("micro") or {}
                lines.append(
                    f"| `{view_name}` | {view.get('eligible_pairs', 'n/a')} | {view.get('successful_pairs', 'n/a')} | {view.get('failed_pairs', 'n/a')} | {_metric(view.get('coverage'))} | {_metric((micro.get('onset_pitch') or {}).get('f1'))} | {_metric((micro.get('onset_pitch_offset') or {}).get('f1'))} | {_metric((micro.get('frames') or {}).get('f1'))} |"
                )
        else:
            lines.append("- Quality: unavailable; no score is synthesized.")
        execution = model.get("execution") or {}
        routes = execution.get("routes") or []
        if routes:
            lines.extend(["", f"- Execution status: `{execution.get('status', 'unknown')}`.", "", "| Route | Status |", "| --- | --- |"])
            for route in routes:
                lines.append(f"| `{route.get('route', 'unknown')}` | `{route.get('status', 'unknown')}` |")
        else:
            lines.append(f"- Execution status: `{execution.get('status', 'unavailable')}`.")
        quantization = model.get("quantization")
        if quantization:
            lines.append(
                f"- Quantization: `{quantization.get('status', 'unknown')}`; ordinary Linear modules {quantization.get('original_linear_modules', 'n/a')} → {quantization.get('quantized_linear_modules', 'n/a')}; engine `{quantization.get('engine', 'n/a')}`."
            )
        lines.append("")
    lines.extend(["## Interpretation", ""])
    conclusion = report.get("conclusion", {})
    for key in ("quality", "cost", "representation", "later_integration"):
        lines.append(f"- {conclusion.get(key, 'not recorded')}")
    return "\n".join(lines) + "\n"


def _report_number(value: Any, digits: int = 3) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _report_mib(value: Any) -> str:
    return "n/a" if value is None else f"{float(value) / (1024 * 1024):.1f}"


def _route_median(route: Mapping[str, Any], batch_size: int, field: str) -> Any:
    value = route.get("batch_results", {}).get(str(batch_size), {}).get(field, {})
    return value.get("median") if isinstance(value, Mapping) else None


def _startup_median(route: Mapping[str, Any]) -> Any:
    value = route.get("startup", {}).get("total_seconds", {})
    if not isinstance(value, Mapping):
        return None
    nested = value.get("value")
    return nested.get("median") if isinstance(nested, Mapping) else value.get("median")


def _startup_component_median(route: Mapping[str, Any], component: str) -> Any:
    value = route.get("startup", {}).get(component, {})
    if not isinstance(value, Mapping):
        return None
    nested = value.get("value")
    return nested.get("median") if isinstance(nested, Mapping) else value.get("median")


def _report_count(value: Any) -> str:
    return "n/a" if value is None else str(int(value))


def _historical_route(history: Mapping[str, Any], route_id: str) -> Mapping[str, Any] | None:
    for route in history.get("routes", []):
        if isinstance(route, Mapping) and route.get("route") == route_id:
            return route
    return None


def _quality_f1(quality: Mapping[str, Any], view_name: str, metric: str) -> Any:
    view = quality.get(view_name, {})
    return view.get("aggregate", {}).get("micro", {}).get(metric, {}).get("f1")


def _identity_representation(identity: Mapping[str, Any]) -> str:
    representation = identity.get("representation") or {}
    if isinstance(representation, Mapping) and representation:
        return "; ".join(f"{key}={value}" for key, value in representation.items())
    return str(identity.get("native_output_type", identity.get("output_contract", "unknown")))


def _markdown(report: Mapping[str, Any]) -> str:
    models = list(report.get("models", []))
    evidence = report.get("evidence", {})
    conclusion = report.get("conclusion", {})
    measured = [model for model in models if model.get("status") == "ok"]
    unavailable = [model for model in models if model.get("status") != "ok"]
    lines = [
        "# Performance transcription benchmark",
        "",
        "## Research status",
        "",
        f"**Comparative status: `{report.get('comparison', {}).get('status', 'unknown')}`.** Exactly `{len(measured)}` of `{len(models)}` configured candidates produced executable benchmark evidence in the permitted local state. The result is a Basic Pitch baseline plus explicit alternative-model blockers, not a completed comparative benchmark.",
        "",
        "The JSON is authoritative, but this Markdown is intended to stand alone as the research finding. Quality, execution/resource cost, backward cost, representation, licensing, and quantization remain separate evidence classes; no composite winner is computed.",
        "",
        "## What was successfully established",
        "",
        f"- Measured candidates: `{', '.join(evidence.get('measured_models', [])) or 'none'}`.",
        f"- Metadata-only or unavailable candidates: `{', '.join(evidence.get('metadata_only_models', [])) or 'none'}`.",
        f"- Directly measured scope: {evidence.get('measured_scope', 'not recorded')}",
        f"- Sourced/model-level scope: {evidence.get('sourced_scope', 'not recorded')}",
        f"- Unresolved comparative scope: {evidence.get('unresolved_scope', 'not recorded')}",
        "",
        "## Candidate identity and known properties",
        "",
        "These are verified inventory facts, separated from observations produced by executing a model. A known source, representation, or license does not imply that the candidate was runnable here.",
        "",
        "| Candidate | Family | Status | Output / representation | Native rate | Native batch | Code / weight license | Differentiable boundary |",
        "| --- | --- | --- | --- | ---: | --- | --- | --- |",
    ]
    for model in models:
        identity = model.get("identity", {})
        native_batch = ", ".join(str(value) for value in identity.get("native_batch_sizes", [])) or "n/a"
        licenses = f"{identity.get('code_license', 'n/a')} / {identity.get('weight_license', 'n/a')}"
        lines.append(
            f"| `{model.get('model_id', 'unknown')}` | `{model.get('family', 'unknown')}` | `{model.get('status', 'unknown')}` | `{model.get('output_contract', identity.get('output_contract', 'unknown'))}`; { _identity_representation(identity) } | {identity.get('native_sample_rate', 'n/a')} | `{native_batch}` | `{licenses}` | `{identity.get('differentiable_boundary', 'n/a')}` |"
        )
    lines.extend(
        [
            "",
            "### Identity/source inventory",
            "",
            "| Candidate | Source identity | Checkpoint identity | Availability reason |",
            "| --- | --- | --- | --- |",
        ]
    )
    for model in models:
        identity = model.get("identity", {})
        source = f"{identity.get('source_repository', 'n/a')} @ {identity.get('source_revision', 'n/a')}"
        checkpoint = f"{identity.get('checkpoint_repository', 'n/a')} @ {identity.get('checkpoint_revision', 'n/a')}"
        reason = model.get("availability_reason") or "available and verified"
        lines.append(f"| `{model.get('model_id', 'unknown')}` | `{source}` | `{checkpoint}` | {reason} |")
    lines.extend(
        [
            "",
            "Sourced representation notes: Timbre-Trap is retained as a native frame/pitch output and is not given a fabricated note-event decoder; YourMT3 variants expose stock note-event output; MuScriptor exposes timing-corrected MIDI note events with stock prelude forcing. These facts describe upstream interfaces, not measured OBRUXO performance.",
            "",
            "## What was actually executed",
            "",
            "Only Basic Pitch produced executable evidence. The following sections consume the landed #24 and #25 reports; #26 did not rerun inference, evaluation, rendering, or quantization for this reporting revision.",
            "",
        ]
    )
    basic_pitch = next((model for model in models if model.get("model_id") == "basic_pitch"), None)
    corrected_openvino_gpu_measured = False
    model_call_winners: dict[int, str] = {}
    model_call_winner_rates: dict[int, Any] = {}
    e2e_winner: str | None = None
    e2e_winner_rate: Any = None
    if basic_pitch:
        quality = basic_pitch.get("quality") or {}
        quality_provenance = basic_pitch.get("quality_provenance") or {}
        execution = basic_pitch.get("execution") or {}
        openvino_diagnostic = execution.get("openvino_precision_diagnostic") or {}
        if not isinstance(openvino_diagnostic, Mapping):
            openvino_diagnostic = {}
        openvino_history = execution.get("openvino_parity_history") or {}
        if not isinstance(openvino_history, Mapping):
            openvino_history = {}
        inference_routes = [
            route
            for route in execution.get("routes", [])
            if route.get("mode") == "inference" and route.get("status") == "ok"
        ]
        openvino_gpu = next(
            (route for route in inference_routes if route.get("route") == "openvino_gpu"),
            None,
        )
        corrected_openvino_gpu_measured = bool(
            openvino_gpu
            and isinstance(openvino_gpu.get("batch_results"), Mapping)
            and openvino_gpu.get("batch_results")
        )
        for batch_size in (1, 2, 4, 8):
            candidates = [
                (route, _route_median(route, batch_size, "audio_seconds_per_second"))
                for route in inference_routes
            ]
            candidates = [(route, value) for route, value in candidates if value is not None]
            if candidates:
                winner_route, winner_rate = max(candidates, key=lambda item: item[1])
                model_call_winners[batch_size] = winner_route.get("route", "unknown")
                model_call_winner_rates[batch_size] = winner_rate
        e2e_candidates = [
            (route, (route.get("end_to_end") or {}).get("audio_seconds_per_wall_second", {}).get("median"))
            for route in inference_routes
        ]
        e2e_candidates = [(route, value) for route, value in e2e_candidates if value is not None]
        if e2e_candidates:
            winner_route, e2e_winner_rate = max(e2e_candidates, key=lambda item: item[1])
            e2e_winner = winner_route.get("route", "unknown")
        quality_backend = quality_provenance.get("backend") or {}
        runtime_provenance = quality_provenance.get("runtime_provenance") or {}
        lines.extend(
            [
                "### Basic Pitch quality evidence inherited from #25",
                "",
                f"- Source: `{quality_provenance.get('source', 'not recorded')}`; eligible population: `{quality.get('success_only', {}).get('eligible_pairs', 'n/a')}`; coverage: `{_report_number(quality.get('success_only', {}).get('coverage'))}`.",
                f"- Recorded #25 backend: `{quality_backend.get('backend_id', 'unknown')}`; boundary: `{quality_backend.get('boundary', 'unknown')}`; precision: `{quality_backend.get('precision', 'unknown')}`.",
                f"- #25 route provenance assessment: `{runtime_provenance.get('consistency', 'not recorded')}`. {runtime_provenance.get('interpretation', '')}",
                "",
                "| Quality view | Eligible | Succeeded | Failed | Coverage | Onset+pitch F1 | Onset+pitch+offset F1 | Frame F1 |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for view_name in ("success_only", "failure_penalized"):
            view = quality.get(view_name, {})
            lines.append(
                f"| `{view_name}` | {view.get('eligible_pairs', 'n/a')} | {view.get('successful_pairs', 'n/a')} | {view.get('failed_pairs', 'n/a')} | {_report_number(view.get('coverage'))} | {_report_number(_quality_f1(quality, view_name, 'onset_pitch'))} | {_report_number(_quality_f1(quality, view_name, 'onset_pitch_offset'))} | {_report_number(_quality_f1(quality, view_name, 'frames'))} |"
            )
        bootstrap = quality.get("success_only", {}).get("aggregate", {}).get("bootstrap", {})
        lines.extend(
            [
                "",
                f"- Uncertainty: `{bootstrap.get('replicates', 'n/a')}` seed-`{bootstrap.get('seed', 'n/a')}` preset-cluster replicates over `{bootstrap.get('cluster_count', 'n/a')}` clusters. These are Basic Pitch baseline intervals, not alternative-model comparisons.",
                "",
                "### Basic Pitch execution and resource evidence inherited from #24",
                "",
                f"- Source: `{execution.get('source', 'not recorded')}`; {execution.get('reporting_note', '')}",
                "- Cost evidence is route-specific; a route failure is not converted into a score or a fallback result.",
                "- The table below consumes the current #24 route rows. The historical default OpenVINO GPU failure and the bounded corrected parity diagnostic remain separate from the corrected timed route row.",
                "",
                "| Mode | Route | Evidence state | Batch-1 throughput | Batch-8 throughput | E2E rate | Startup (s) | Host RSS (MiB) | XPU allocated/reserved (MiB) | OpenVINO GPU memory (MiB) |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
            ]
        )
        for route in execution.get("routes", []):
            end_to_end = route.get("end_to_end") or {}
            memory = route.get("memory") or {}
            e2e_rate = end_to_end.get("audio_seconds_per_wall_second", {}).get("median")
            allocated = _report_mib(memory.get("pytorch_xpu_peak_allocated_bytes"))
            reserved = _report_mib(memory.get("pytorch_xpu_peak_reserved_bytes"))
            gpu_memory = _report_mib(memory.get("openvino_gpu_memory_bytes"))
            gpu_total_memory = _report_mib(memory.get("openvino_gpu_total_memory_bytes"))
            if gpu_memory != "n/a" and gpu_total_memory != "n/a":
                gpu_memory = f"{gpu_memory} / {gpu_total_memory}"
            evidence_state = route.get("status", "unknown")
            if route.get("route") == "openvino_gpu":
                evidence_state = "measured_corrected_fp32_performance" if corrected_openvino_gpu_measured else "pre_fix_parity_failed"
            lines.append(
                f"| `{route.get('mode', 'unknown')}` | `{route.get('route', 'unknown')}` | `{evidence_state}` | {_report_number(_route_median(route, 1, 'audio_seconds_per_second'))} | {_report_number(_route_median(route, 8, 'audio_seconds_per_second'))} | {_report_number(e2e_rate)} | {_report_number(_startup_median(route))} | {_report_mib(memory.get('host_peak_rss_bytes'))} | {allocated} / {reserved} | {gpu_memory} |"
            )
        failures = execution.get("route_failures") or []
        if failures:
            lines.extend(["", "Historical route records:"])
            for failure in failures:
                if failure.get("route") == "openvino_gpu" and openvino_diagnostic:
                    lines.append(f"- `{failure.get('route')}`: historical pre-fix/default `float16` + `PERFORMANCE` -> `{failure.get('status', 'unknown')}` before timing; the current corrected route is reported separately.")
                else:
                    lines.append(f"- `{failure.get('route', 'unknown')}`: `{failure.get('status', 'unknown')}`. The landed #24 report suppresses timing for this route.")

        historical_gpu = _historical_route(openvino_history, "openvino_gpu")
        if openvino_diagnostic or historical_gpu or corrected_openvino_gpu_measured:
            configuration = openvino_diagnostic.get("configuration") or {}
            parity = openvino_diagnostic.get("parity") or {}
            lines.extend(["", "### OpenVINO GPU evidence state", ""])
            if historical_gpu:
                historical_max = historical_gpu.get("max_across_repetitions") or {}
                historical_repetitions = historical_gpu.get("repetitions") or []
                historical_first = historical_repetitions[0] if historical_repetitions else {}
                historical_parity = historical_first.get("parity") or {}
                historical_configuration = openvino_history.get("configuration") or {}
                lines.append(
                    f"- Historical pre-fix/default result: requested `{historical_configuration.get('inference_precision_hint', 'plugin default')}`, observed `{historical_configuration.get('execution_mode_hint', 'plugin default')}`; status `{historical_gpu.get('status', 'not_recorded')}` before timing. Non-finite contour/note/onset values were `{_report_count(historical_max.get('contour_non_finite_count'))}` / `{_report_count(historical_max.get('note_non_finite_count'))}` / `{_report_count(historical_max.get('onset_non_finite_count'))}`; candidate/reference event counts were `{_report_count(historical_parity.get('candidate_event_count'))}` / `{_report_count(historical_parity.get('reference_event_count'))}`. No performance or resource result is inferred from this failed route."
                )
            if openvino_diagnostic:
                lines.extend(
                    [
                        f"- Bounded diagnostic result (corrected): requested `INFERENCE_PRECISION_HINT={configuration.get('inference_precision_hint_requested', 'not_recorded')}`, compiled `{configuration.get('inference_precision_hint_compiled', 'not_recorded')}` + `{configuration.get('execution_mode_hint_compiled', 'not_recorded')}`; status `{openvino_diagnostic.get('status', 'not_recorded')}` on `{openvino_diagnostic.get('windows', 'not_recorded')}` public synthetic windows.",
                        f"- Bounded parity metrics: non-finite values and threshold/event disagreements were `0`; maximum contour/note/onset errors were `{_report_number(parity.get('contour_max_abs_error'), 9)}`, `{_report_number(parity.get('note_max_abs_error'), 9)}`, and `{_report_number(parity.get('onset_max_abs_error'), 9)}`.",
                    ]
                )
            if corrected_openvino_gpu_measured and openvino_gpu:
                backend = openvino_gpu.get("backend") or {}
                gpu_memory = openvino_gpu.get("memory") or {}
                timed_parity = openvino_gpu.get("parity") or {}
                lines.extend(
                    [
                        f"- Corrected measured result: the actual #24 smoke route compiled `{backend.get('inference_precision_hint_compiled', 'not_recorded')}` on `{backend.get('full_device_name', backend.get('selected_device', 'not_recorded'))}` with `{backend.get('execution_mode_hint_compiled', 'not_recorded')}` execution; timed-route parity is `{openvino_gpu.get('parity_status', openvino_gpu.get('status', 'not_recorded'))}`.",
                        f"- Corrected startup medians: backend import `{_report_number(_startup_component_median(openvino_gpu, 'backend_import_seconds'))}` s, model conversion `{_report_number(_startup_component_median(openvino_gpu, 'openvino_conversion_seconds'))}` s, GPU compilation `{_report_number(_startup_component_median(openvino_gpu, 'openvino_compile_seconds'))}` s, total startup `{_report_number(_startup_median(openvino_gpu))}` s; first-call / warmup at batch 1 `{_report_number(_route_median(openvino_gpu, 1, 'first_inference_seconds'))}` / `{_report_number(_route_median(openvino_gpu, 1, 'inference_warmup_seconds'))}` s.",
                        f"- Corrected steady-state audio-equivalent throughput (audio-s/s): batch 1 `{_report_number(_route_median(openvino_gpu, 1, 'audio_seconds_per_second'))}`, batch 2 `{_report_number(_route_median(openvino_gpu, 2, 'audio_seconds_per_second'))}`, batch 4 `{_report_number(_route_median(openvino_gpu, 4, 'audio_seconds_per_second'))}`, batch 8 `{_report_number(_route_median(openvino_gpu, 8, 'audio_seconds_per_second'))}`.",
                        f"- Corrected end-to-end throughput: `{_report_number((openvino_gpu.get('end_to_end') or {}).get('audio_seconds_per_wall_second', {}).get('median'))}` audio-s/s; host peak RSS `{_report_mib(gpu_memory.get('host_peak_rss_bytes'))}` MiB; OpenVINO GPU memory `{_report_mib(gpu_memory.get('openvino_gpu_memory_bytes'))}` / `{_report_mib(gpu_memory.get('openvino_gpu_total_memory_bytes'))}` MiB where reported.",
                        f"- Timed-route parity errors: non-finite values and threshold/event disagreements were `0`; maximum contour/note/onset errors were `{_report_number(timed_parity.get('contour_max_abs_error'), 9)}`, `{_report_number(timed_parity.get('note_max_abs_error'), 9)}`, and `{_report_number(timed_parity.get('onset_max_abs_error'), 9)}`.",
                    ]
                )
            elif not corrected_openvino_gpu_measured:
                lines.append("- Corrected FP32 performance/resource measurements are not present in the landed #24 route rows; no corrected OpenVINO GPU performance claim is made.")
        else:
            lines.extend(
                [
                    "",
                    "The OpenVINO GPU row is a parity failure before timing; it supports no GPU speed, memory, or quality conclusion. The #24 report retains the detailed route findings, including startup/throughput crossover and CPU/XPU backward measurements.",
                ]
            )
        quantization = basic_pitch.get("quantization") or {}
        lines.extend(
            [
                "",
                "### Basic Pitch quantization evidence",
                "",
                f"- Status: `{quantization.get('status', 'not recorded')}`; ordinary Linear modules `{quantization.get('original_linear_modules', 'n/a')}` -> `{quantization.get('quantized_linear_modules', 'n/a')}`; engine `{quantization.get('engine', 'n/a')}`.",
                "- No quantized artifact was produced, and no quantized XPU/OpenVINO/backward/batch-sweep result exists.",
            ]
        )
    if corrected_openvino_gpu_measured:
        model_call_summary = ", ".join(
            f"batch {batch_size}: `{route}` ({_report_number(model_call_winner_rates.get(batch_size))} audio-s/s)"
            for batch_size, route in model_call_winners.items()
        ) or "not recorded"
        e2e_summary = f"`{e2e_winner}` ({_report_number(e2e_winner_rate)} audio-s/s)" if e2e_winner else "not recorded"
        measured_route_note = f"- The measured #24 model-call throughput winners were {model_call_summary}; the end-to-end winner was {e2e_summary}. These are Basic Pitch route findings, not alternative-model results."
        bounded_route_note = "- The bounded corrected parity diagnostic remains a correctness result; the corrected GPU timing/resource rows above are the separate measured result."
    else:
        measured_route_note = "- The observed Basic Pitch route trade-offs and the historical pre-fix OpenVINO GPU failure are findings of #24; corrected GPU performance is not present in the consumed route rows."
        bounded_route_note = "- This bounded result validates numerical parity only; it does not add a corrected OpenVINO GPU speed, startup, end-to-end, memory, or resource result."
    lines.extend(["", "## What could not be executed", "", "No alternative candidate produced a quality, execution-cost, resource, backward-cost, or quantization measurement. The unavailability reasons are concrete local prerequisites, not claims that the models are intrinsically impossible to run.", "", "| Candidate | Status | Concrete blocker | What this prevents |", "| --- | --- | --- | --- |"])
    for model in unavailable:
        reason = model.get("availability_reason") or model.get("failure_code") or "not recorded"
        lines.append(f"| `{model.get('model_id', 'unknown')}` | `{model.get('status', 'unknown')}` | {reason} | no comparative quality/cost result |")
    lines.extend(
        [
            "",
            "- `timbre_trap_base`: the pinned checkpoint size is not locally verifiable and no approved Timbre-Trap checkout is present in the existing runtime/storage.",
            "- `ymt3_plus`, `yptf_multi`, `yptf_moe_multi`: the official source/checkpoint material is not present in permitted local storage.",
            "- `muscriptor_small`, `muscriptor_medium`, `muscriptor_large`: checkpoints are gated and no approved credential or local copy is available; no login, terms acceptance, or acquisition was attempted.",
            "",
            "## Conclusions by evidence class",
            "",
            "### Directly supported by measured results",
            "",
            f"- {conclusion.get('measured', 'not recorded')}",
            "- The Basic Pitch evidence is a complete baseline for the landed #24/#25 contracts, not a comparison against the unavailable alternatives.",
            measured_route_note,
            "- The #25 quality result remains CPU-provenanced until route provenance is resolved.",
            "",
            "### Bounded diagnostic results",
            "",
            f"- {conclusion.get('bounded_diagnostic', 'not recorded')}",
            bounded_route_note,
            "",
            "### Supported only by verified model characteristics",
            "",
            f"- {conclusion.get('sourced', 'not recorded')}",
            "- Representation and licensing facts can inform later integration design, but they do not establish transcription quality, runtime cost, memory, or suitability.",
            "",
            "### Comparative questions that remain unanswered",
            "",
            f"- {conclusion.get('unresolved', 'not recorded')}",
            f"- {conclusion.get('not_measured', 'not recorded')}",
            "- No ranking, quality estimate, cost estimate, quantization effect, or integration recommendation is assigned to any unavailable candidate.",
            "",
            "## What is required before the intended comparison can be completed",
            "",
            "The following prerequisites must become available before a legitimate comparative run can be attempted; none is acquired or changed by this report revision:",
            "",
            "- An approved, immutable Timbre-Trap source checkout plus a verifiable pinned `tt-orig.pt` checkpoint and the already-permitted runtime prerequisites.",
            "- The approved YourMT3 source checkout and all three exact immutable checkpoints selected in `models.yaml`.",
            "- Approved access and local copies of the three gated MuScriptor checkpoints, without exposing credentials or taking account actions in this task.",
            "- A permitted existing-runtime preflight for each candidate, followed by the fixed common #25 quality population and applicable #24 cost routes. Missing dependencies must be recorded as blockers rather than installed or substituted.",
            "- After executable results exist, report both success-only and failure-penalized quality views, route/resource/backward applicability, and the fixed CPU dynamic-Linear quantization scope separately; do not synthesize a composite winner.",
            "",
            "## Contract and privacy limits",
            "",
            f"- Quality: `{report.get('comparison', {}).get('quality_contract', 'unknown')}`; cost: `{report.get('comparison', {}).get('cost_contract', 'unknown')}`; quantization: `{report.get('comparison', {}).get('quantization_contract', 'unknown')}`.",
            "- #26 reused the exact #25 eligible population/audio provenance and the landed #24 results; it did not render, rebuild audio, rerun Basic Pitch, or use alternative fallbacks.",
            "- Private paths, pair identifiers, source filenames, row predictions, gated weights, and local run state are excluded from this public report.",
        ]
    )
    return "\n".join(lines) + "\n"


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False, newline="\n") as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_public_report(
    input_root: Path,
    json_path: Path,
    markdown_path: Path,
    *,
    config_path: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    input_root = Path(input_root).resolve(strict=False)
    if not input_root.is_relative_to((Path(__file__).resolve().parents[1] / "outputs").resolve()):
        raise ValueError("report input must be inside the ignored benchmark output area")
    json_file = _approved_report(json_path)
    markdown_file = _approved_report(markdown_path)
    if not force and (json_file.exists() or markdown_file.exists()):
        raise FileExistsError("refusing to overwrite public report without force=True")
    config = config_path or Path(__file__).resolve().parents[1] / "config" / "models.yaml"
    specs = load_model_specs(config)
    report = build_public_report(specs, input_root)
    _atomic_text(json_file, json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    _atomic_text(markdown_file, _markdown(report))
    return report

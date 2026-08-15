"""Sanitized aggregate report generation; local pair results never cross this boundary."""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .artifacts import ModelSpec, load_model_specs

_PRIVATE_KEY_PARTS = (
    "pair_id",
    "preset_id",
    "request_id",
    "audio_path",
    "midi_path",
    "source_root",
    "local_path",
    "hostname",
)
_PRIVATE_TEXT = re.compile(
    r"(?:^[A-Za-z]:[\\/]|/Users/|/home/|\\Users\\|datasets[\\/]|pair-[0-9a-f]{8,}|preset-[0-9a-f]{8,}|request-[0-9a-f]{8,})",
    re.IGNORECASE,
)


class ReportPrivacyError(ValueError):
    """A report attempted to cross the public/private boundary."""


def _strip_private_aggregate_fields(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _strip_private_aggregate_fields(child)
            for key, child in value.items()
            if key != "per_preset"
        }
    if isinstance(value, list):
        return [_strip_private_aggregate_fields(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_strip_private_aggregate_fields(child) for child in value)
    return value


def _assert_public(value: Any, key: str = "") -> None:
    if any(part in key.casefold() for part in _PRIVATE_KEY_PARTS):
        raise ReportPrivacyError(
            f"private field is not allowed in public report: {key}"
        )
    if isinstance(value, Mapping):
        for child_key, child_value in value.items():
            _assert_public(child_value, str(child_key))
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_public(child, key)
    elif isinstance(value, str) and (
        _PRIVATE_TEXT.search(value)
        or "private_" in value.casefold()
        or "\\tmp\\" in value.casefold()
    ):
        raise ReportPrivacyError(
            "private identifier or machine path is not allowed in public report"
        )


def sanitize_public_report(value: Mapping[str, Any]) -> dict[str, Any]:
    sanitized = _strip_private_aggregate_fields(value)
    _assert_public(sanitized)
    return json.loads(json.dumps(sanitized, sort_keys=True, allow_nan=False))


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


def _stored_results(
    root: Path, specs: Mapping[str, ModelSpec]
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    if not root.is_dir():
        return results
    for run_path in sorted(root.rglob("run.json"), key=lambda path: path.as_posix()):
        run = _read_json(run_path)
        if not run or not isinstance(run.get("model_id"), str):
            continue
        model_id = str(run["model_id"])
        spec = specs.get(model_id)
        if spec is not None and run.get("model_identity") != spec.identity_digest():
            continue
        current = results.setdefault(model_id, {})
        if run.get("variant_id") == "dynamic_int8_linear":
            current["quantization"] = {
                "run": run,
                "runtime": _read_json(run_path.with_name("runtime.json")),
                "aggregates": _read_json(run_path.with_name("aggregates.json")),
            }
            continue
        current["run"] = run
        current["runtime"] = _read_json(run_path.with_name("runtime.json"))
        current["aggregates"] = _read_json(run_path.with_name("aggregates.json"))
    return results


def _stored_pair_rows(
    root: Path, specs: Mapping[str, ModelSpec]
) -> dict[str, list[dict[str, Any]]]:
    """Read pair rows from the selected full-precision run for each model."""
    selected_runs: dict[str, Path] = {}
    if not root.is_dir():
        return {}
    for run_path in sorted(root.rglob("run.json"), key=lambda path: path.as_posix()):
        run = _read_json(run_path)
        if not run or run.get("variant_id") == "dynamic_int8_linear":
            continue
        model_id = run.get("model_id")
        spec = specs.get(model_id) if isinstance(model_id, str) else None
        if spec is None or run.get("model_identity") != spec.identity_digest():
            continue
        selected_runs[model_id] = run_path

    rows_by_model: dict[str, list[dict[str, Any]]] = {}
    for model_id, run_path in selected_runs.items():
        rows: list[dict[str, Any]] = []
        for pair_path in sorted((run_path.parent / "pairs").glob("*.json")):
            row = _read_json(pair_path)
            if row is not None and row.get("status") in {
                "ok",
                "runtime_failed",
                "out_of_memory",
                "invalid_native_output",
            }:
                rows.append(row)
        if rows:
            rows_by_model[model_id] = rows
    return rows_by_model


def _aggregate_pair_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    basic_pitch_root = Path(__file__).resolve().parents[2] / "basic_pitch"
    basic_pitch_root_text = str(basic_pitch_root)
    if basic_pitch_root_text not in sys.path:
        sys.path.insert(0, basic_pitch_root_text)
    from obruxo_basic_pitch.evaluation.aggregate import aggregate_results

    return aggregate_results(rows, bootstrap_replicates=10_000, seed=0)


def _compact_common_aggregate(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: aggregate[key]
        for key in (
            "pair_count",
            "successful_pair_count",
            "failed_pair_count",
            "coverage",
            "micro",
            "pair_macro",
            "failure_analysis",
            "bootstrap",
        )
        if key in aggregate
    }


def _same_successful_population(
    root: Path, specs: Mapping[str, ModelSpec]
) -> dict[str, Any]:
    """Build a diagnostic comparison on rows all selected candidates completed."""
    rows_by_model = _stored_pair_rows(root, specs)
    alternative_ids = [
        model_id
        for model_id, spec in specs.items()
        if model_id != "basic_pitch" and model_id in rows_by_model
    ]
    frame_ids = [
        model_id
        for model_id in alternative_ids
        if specs[model_id].output_contract in {"frame_pitch", "note_events"}
    ]
    note_event_ids = [
        model_id
        for model_id in alternative_ids
        if specs[model_id].output_contract == "note_events"
    ]
    aggregate_cache: dict[tuple[str, frozenset[str]], dict[str, Any]] = {}

    def build_population(model_ids: list[str], scope: str) -> dict[str, Any]:
        success_sets = [
            {
                str(row["pair_id"])
                for row in rows_by_model[model_id]
                if row.get("status") == "ok" and row.get("pair_id") is not None
            }
            for model_id in model_ids
        ]
        common_pair_ids = set.intersection(*success_sets) if success_sets else set()
        models = []
        for model_id in model_ids:
            rows = [
                row
                for row in rows_by_model[model_id]
                if row.get("status") == "ok"
                and str(row.get("pair_id")) in common_pair_ids
            ]
            cache_key = (model_id, frozenset(common_pair_ids))
            if cache_key not in aggregate_cache:
                aggregate_cache[cache_key] = _compact_common_aggregate(
                    _aggregate_pair_rows(rows)
                )
            models.append(
                {
                    "model_id": model_id,
                    "successful_pairs": len(rows),
                    "aggregate": aggregate_cache[cache_key],
                }
            )
        return {
            "scope": scope,
            "eligible_pairs": len(common_pair_ids),
            "model_ids": model_ids,
            "models": models,
            "interpretation": "Diagnostic success-only comparison on the exact pair intersection completed by every listed candidate; it does not replace full-population coverage or failure-penalized views.",
        }

    return {
        "frame_comparable": build_population(
            frame_ids, "alternative_candidates_with_frame_or_note_event_outputs"
        ),
        "note_event": build_population(
            note_event_ids, "alternative_candidates_with_native_note_events"
        ),
        "basic_pitch_note": "Basic Pitch is excluded because its inherited #25 public aggregate does not expose row-level results for constructing this intersection; its full-population baseline is reported separately.",
    }


def _landed_basic_pitch_reports() -> tuple[
    dict[str, Any] | None, dict[str, Any] | None
]:
    reports = Path(__file__).resolve().parents[2] / "basic_pitch" / "reports"
    return _read_json(reports / "presetshare_baseline.json"), _read_json(
        reports / "backend_benchmark.json"
    )


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
    historical_evidence = report.get("historical_evidence") or {}
    pre_fix = (
        historical_evidence.get("pre_fix_default_openvino_gpu", {})
        if isinstance(historical_evidence, Mapping)
        else {}
    )
    bounded = (
        historical_evidence.get("bounded_corrected_openvino_gpu", {})
        if isinstance(historical_evidence, Mapping)
        else {}
    )
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
            "openvino_parity_history": pre_fix.get("data")
            if isinstance(pre_fix, Mapping)
            else None,
            "openvino_precision_diagnostic": bounded.get("data")
            if isinstance(bounded, Mapping)
            else None,
            "reporting_note": "Routes and findings are consumed from the landed #24 report; #26 does not rerun Basic Pitch cost measurements.",
        }
    )
    return value


def _public_candidate_routes(runtime: Mapping[str, Any]) -> list[dict[str, Any]]:
    routes = runtime.get("routes")
    if routes is None:
        routes = list(runtime.get("inference", [])) + list(runtime.get("training", []))
    result: list[dict[str, Any]] = []
    for route in routes or []:
        if not isinstance(route, Mapping):
            continue
        clean = {key: value for key, value in route.items() if key != "repetitions"}
        end_to_end = clean.get("end_to_end")
        if isinstance(end_to_end, Mapping):
            clean["end_to_end"] = {
                key: value for key, value in end_to_end.items() if key != "cases"
            }
        result.append(clean)
    return result


def _public_quantization(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    run = value.get("run") if isinstance(value.get("run"), Mapping) else {}
    runtime = value.get("runtime") if isinstance(value.get("runtime"), Mapping) else {}
    aggregates = (
        value.get("aggregates") if isinstance(value.get("aggregates"), Mapping) else {}
    )
    return {
        "status": run.get("status", runtime.get("status", "unavailable")),
        "failure_code": run.get("failure_code", runtime.get("failure_code")),
        "quality": aggregates.get("quality"),
        "execution": {
            "status": runtime.get("status", "unavailable"),
            "failure_code": runtime.get("failure_code"),
            "routes": _public_candidate_routes(runtime),
            "timing_contract": runtime.get("timing_contract"),
            "route_failures": runtime.get("route_failures", []),
            "quantization": runtime.get("quantization") or run.get("quantization"),
        },
        "measurement": run.get("quantization") or runtime.get("quantization"),
    }


def _quality_measurement_status(quality: Mapping[str, Any] | None) -> str:
    if not isinstance(quality, Mapping):
        return "not_measured"
    view = quality.get("success_only")
    if not isinstance(view, Mapping):
        return "not_measured"
    eligible = int(view.get("eligible_pairs", 0) or 0)
    successful = int(view.get("successful_pairs", 0) or 0)
    if eligible <= 0:
        return "not_measured"
    return "complete" if successful == eligible else "partial_pair_coverage"


def build_public_report(
    specs: Mapping[str, ModelSpec], input_root: Path
) -> dict[str, Any]:
    stored = _stored_results(Path(input_root).resolve(strict=False), specs)
    landed_quality, landed_runtime = _landed_basic_pitch_reports()
    models: list[dict[str, Any]] = []
    for model_id, spec in specs.items():
        current = stored.get(model_id, {})
        run = current.get("run") or {}
        runtime = current.get("runtime") or {}
        aggregates = current.get("aggregates") or {}
        stored_quantization = _public_quantization(
            current.get("quantization")
        ) or run.get("quantization")
        if model_id == "basic_pitch" and (landed_quality or landed_runtime):
            run = {
                "model_id": model_id,
                "status": (landed_quality or {}).get("status", spec.availability),
                "failure_code": (landed_quality or {}).get("failure_code"),
            }
            runtime = (
                _landed_basic_pitch_runtime(landed_runtime) if landed_runtime else {}
            )
            aggregates = (
                {"quality": _landed_basic_pitch_quality(landed_quality)}
                if landed_quality
                else {}
            )
        status = str(run.get("status", spec.availability))
        failure_code = run.get("failure_code") or (
            spec.unavailability_reason if status != "ok" else None
        )
        duration_views = _read_json(Path(input_root) / model_id / "duration_views.json")
        quality_value = aggregates.get("quality") if status == "ok" else None
        if quality_value is not None and duration_views is not None:
            quality_value = dict(quality_value)
            quality_value["duration_views"] = duration_views
        quality_measurement_status = _quality_measurement_status(
            aggregates.get("quality")
        )
        routes = runtime.get("routes")
        if model_id != "basic_pitch":
            routes = _public_candidate_routes(runtime)
        elif routes is None:
            routes = list(runtime.get("inference", [])) + list(
                runtime.get("training", [])
            )
        item: dict[str, Any] = {
            "model_id": model_id,
            "family": spec.family,
            "publication_year": spec.publication_year,
            "output_contract": spec.output_contract,
            "identity": spec.public_identity(),
            "status": status,
            "measurement_status": quality_measurement_status,
            "failure_code": failure_code,
            "availability_reason": spec.unavailability_reason,
            "quality": quality_value,
            "quality_provenance": (
                {
                    "source": "landed_issue_25_report",
                    "backend": (landed_quality or {}).get("backend"),
                    "runtime_provenance": (landed_quality or {}).get(
                        "runtime_provenance"
                    ),
                    "category_findings": (landed_quality or {}).get(
                        "category_findings"
                    ),
                }
                if model_id == "basic_pitch" and landed_quality
                else {
                    "source": "local_issue_26_evaluation",
                    "metric_contract": "issue_25_note_frame_metrics_v1",
                }
                if status == "ok"
                else None
            ),
            "execution": {
                "source": runtime.get("source"),
                "status": runtime.get("status", "unavailable"),
                "failure_code": runtime.get("failure_code"),
                "routes": routes,
                "timing_contract": runtime.get("timing_contract")
                or runtime.get("config"),
                "phases": runtime.get("phases"),
                "route_failures": runtime.get("route_failures", []),
                "measurement_status": runtime.get("measurement_status")
                or quality_measurement_status,
                "execution_note": runtime.get("execution_note")
                or run.get("execution_note"),
                "adapter_status": "landed_baseline"
                if model_id == "basic_pitch"
                else "implemented_official_path",
                "openvino_parity_history": runtime.get("openvino_parity_history"),
                "openvino_precision_diagnostic": runtime.get(
                    "openvino_precision_diagnostic"
                ),
                "reporting_note": runtime.get("reporting_note"),
                "adapter_configuration": runtime.get("adapter_configuration")
                or run.get("adapter_configuration"),
                "segment_batch_sizes_observed": runtime.get(
                    "segment_batch_sizes_observed"
                )
                or run.get("segment_batch_sizes_observed"),
                "cache_reuse_policy": runtime.get("cache_reuse_policy")
                or run.get("cache_reuse_policy"),
                "resources": runtime.get("resources") or run.get("resources"),
                "native_batch_sizes": runtime.get(
                    "native_batch_sizes", list(spec.native_batch_sizes)
                ),
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
                "eligible_pairs": ((landed_quality or {}).get("pairing") or {}).get(
                    "eligible_count"
                ),
                "quality_coverage": ((landed_quality or {}).get("aggregate") or {}).get(
                    "coverage"
                ),
                "cost_status": runtime.get("status"),
                "cost_failure_code": runtime.get("failure_code"),
                "cost_route_failures": runtime.get("route_failures", []),
                "cost_measurement_status": runtime.get("measurement_status"),
                "openvino_gpu_diagnostic_status": (
                    (runtime.get("openvino_precision_diagnostic") or {}).get("status")
                ),
            }
        models.append(item)
    measured_models = [
        model["model_id"] for model in models if model.get("status") == "ok"
    ]
    partial_models = [
        model["model_id"]
        for model in models
        if model.get("measurement_status") == "partial_pair_coverage"
    ]
    unavailable_models = [
        model["model_id"] for model in models if model.get("status") != "ok"
    ]
    comparison_status = (
        "complete"
        if not unavailable_models and not partial_models
        else "partial_executable_candidates"
    )
    measured_alternatives = [
        model_id for model_id in measured_models if model_id != "basic_pitch"
    ]
    blocked_text = ", ".join(unavailable_models) or "none"
    measured_text = ", ".join(measured_models) or "none"
    return sanitize_public_report(
        {
            "format_version": 1,
            "comparison": {
                "quality_contract": "landed_issue_25_metrics_and_10000_replicate_seed_0_bootstrap",
                "cost_contract": "landed_issue_24_end_to_end_boundary",
                "quantization_contract": "cpu_dynamic_qint8_ordinary_linear_only",
                "no_composite_winner": True,
                "status": comparison_status,
                "same_successful_population": _same_successful_population(
                    Path(input_root).resolve(strict=False), specs
                ),
            },
            "models": models,
            "evidence": {
                "measured_models": measured_models,
                "partial_measurement_models": partial_models,
                "metadata_only_models": unavailable_models,
                "measured_scope": f"Executable evidence is present for {measured_text}. Basic Pitch quality and cost evidence are inherited from #25/#24; {', '.join(measured_alternatives) or 'no alternative candidate'} produced new #26 evidence in the unchanged py312 runtime. Partial pair coverage is explicitly identified for {', '.join(partial_models) or 'none'}.",
                "bounded_diagnostic_scope": "The corrected OpenVINO GPU route also retains a bounded five-window FP32 + PERFORMANCE parity diagnostic; that diagnostic is separate from the corrected route's smoke-benchmark timing and resource measurements.",
                "not_measured_scope": f"No quality/cost conclusion is available for {blocked_text}; these candidates were either externally blocked or failed before producing executable evidence.",
                "sourced_scope": "Candidate source, checkpoint, representation, architecture boundary, native sample rate, batch semantics, and license fields are verified inventory facts; they are not performance measurements.",
                "adapter_scope": "The repository contains an implemented pinned-official adapter path for every required candidate family. An implemented adapter is not treated as executed when its source, checkpoint, dependency, or credential prerequisite is unavailable.",
                "unresolved_scope": f"The intended comparative benchmark remains incomplete for unavailable candidates `{blocked_text}` and partial-coverage candidates `{', '.join(partial_models) or 'none'}`; measured candidates are reported separately and no unavailable candidate receives a fabricated score.",
            },
            "conclusion": {
                "measured": f"Measured evidence exists for {measured_text}. Basic Pitch contributes the inherited #25 quality and #24 route/cost baseline; executable alternatives contribute their own #26 corpus quality and applicable CPU/XPU cost measurements. Partial pair-coverage candidates are not complete correctness results.",
                "bounded_diagnostic": "A separate bounded #24 diagnostic compiled OpenVINO GPU with INFERENCE_PRECISION_HINT=float32 while retaining PERFORMANCE and passed parity on five public synthetic windows. This is a parity result, not a performance/resource result.",
                "not_measured": f"The remaining candidate execution states are {blocked_text}; their quality, cost, memory, backward, and quantization results are not inferred from metadata or from other models. Partial-coverage candidates still require full-population correctness confirmation.",
                "sourced": "The candidate inventory establishes model identity, representation, architecture boundary, native rate/batch semantics, and licensing where verified, but none of these facts ranks execution quality or cost.",
                "unresolved": f"Comparative questions remain for unavailable candidates `{blocked_text}` and partial-coverage candidates `{', '.join(partial_models) or 'none'}`; measured candidates must be compared by separate quality and cost evidence rather than a composite winner.",
                "quality": "Quality is published only for candidates with an executed #25-compatible population; globally unavailable models receive no invented F1.",
                "cost": "Cost rows remain separate by candidate route, with unsupported or failed routes explicitly reported rather than replaced by fallback measurements.",
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
        "| Model | Family | Status | Measurement | Quality |",
        "| --- | --- | --- | --- | --- |",
    ]
    for model in report.get("models", []):
        quality = "reported" if model.get("quality") is not None else "unavailable"
        lines.append(
            f"| `{model.get('model_id', 'unknown')}` | `{model.get('family', 'unknown')}` | `{model.get('status', 'unknown')}` | `{model.get('measurement_status', 'unknown')}` | `{quality}` |"
        )
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
            reason = (
                model.get("availability_reason")
                or model.get("failure_code")
                or "not recorded"
            )
            lines.append(
                f"- Availability: `{model.get('status', 'unknown')}` — {reason}."
            )
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
            lines.extend(
                [
                    "",
                    f"- Execution status: `{execution.get('status', 'unknown')}`.",
                    "",
                    "| Route | Status |",
                    "| --- | --- |",
                ]
            )
            for route in routes:
                lines.append(
                    f"| `{route.get('route', 'unknown')}` | `{route.get('status', 'unknown')}` |"
                )
            detailed_routes = [
                route
                for route in routes
                if isinstance(route.get("startup"), Mapping)
                or isinstance(route.get("steady_state"), Mapping)
                or isinstance(route.get("end_to_end"), Mapping)
            ]
            if detailed_routes:
                lines.extend(
                    [
                        "",
                        "| Route | Startup/load (s) | First call (s) | Steady calls/s | E2E audio-s/s | Host RSS (MiB) |",
                        "| --- | ---: | ---: | ---: | ---: | ---: |",
                    ]
                )
                for route in detailed_routes:
                    startup = route.get("startup") or {}
                    steady = route.get("steady_state") or {}
                    e2e = route.get("end_to_end") or {}
                    first = _median_value(startup.get("first_call_seconds"))
                    if first is None:
                        first = _route_median(route, 1, "first_inference_seconds")
                    startup_value = _median_value(startup.get("model_load_seconds"))
                    if startup_value is None:
                        startup_value = _startup_median(route)
                    lines.append(
                        f"| `{route.get('route', 'unknown')}` | {_report_number(startup_value)} | {_report_number(first)} | {_report_number(_median_value(steady.get('calls_per_second')))} | {_report_number(_median_value(e2e.get('audio_seconds_per_wall_second')))} | {_report_mib((route.get('resources') or {}).get('host_peak_rss_bytes'))} |"
                    )
        else:
            lines.append(
                f"- Execution status: `{execution.get('status', 'unavailable')}`."
            )
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
    if isinstance(value, Mapping):
        value = value.get("value", value)
        if isinstance(value, Mapping):
            value = value.get("median")
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


def _median_value(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    nested = value.get("value", value)
    return nested.get("median") if isinstance(nested, Mapping) else nested


def _report_count(value: Any) -> str:
    return "n/a" if value is None else str(int(value))


def _historical_route(
    history: Mapping[str, Any], route_id: str
) -> Mapping[str, Any] | None:
    for route in history.get("routes", []):
        if isinstance(route, Mapping) and route.get("route") == route_id:
            return route
    return None


def _quality_f1(quality: Mapping[str, Any], view_name: str, metric: str) -> Any:
    view = quality.get(view_name, {})
    return view.get("aggregate", {}).get("micro", {}).get(metric, {}).get("f1")


def _aggregate_f1(aggregate: Mapping[str, Any], metric: str) -> Any:
    return aggregate.get("micro", {}).get(metric, {}).get("f1")


def _identity_representation(identity: Mapping[str, Any]) -> str:
    representation = identity.get("representation") or {}
    if isinstance(representation, Mapping) and representation:
        return "; ".join(f"{key}={value}" for key, value in representation.items())
    return str(
        identity.get("native_output_type", identity.get("output_contract", "unknown"))
    )


def _markdown(report: Mapping[str, Any]) -> str:
    models = list(report.get("models", []))
    evidence = report.get("evidence", {})
    conclusion = report.get("conclusion", {})
    measured = [model for model in models if model.get("status") == "ok"]
    unavailable = [model for model in models if model.get("status") != "ok"]
    comparison_status = report.get("comparison", {}).get("status", "unknown")
    same_population = report.get("comparison", {}).get("same_successful_population", {})
    status_message = (
        f"All `{len(models)}` configured candidates produced executable evidence."
        if not unavailable
        else f"`{len(measured)}` of `{len(models)}` configured candidates produced executable evidence; `{len(unavailable)}` remain externally blocked or failed before measurement."
    )
    lines = [
        "# Performance transcription benchmark",
        "",
        "## Research status",
        "",
        f"**Comparative status: `{comparison_status}`.** {status_message} The report separates measured candidates from genuine blockers and does not infer a composite winner.",
        "",
        "The JSON is authoritative, but this Markdown is intended to stand alone as the research finding. Quality, execution/resource cost, backward cost, representation, licensing, and quantization remain separate evidence classes; no composite winner is computed.",
        "",
        "## What was successfully established",
        "",
        f"- Measured candidates: `{', '.join(evidence.get('measured_models', [])) or 'none'}`.",
        f"- Partial pair-coverage candidates: `{', '.join(evidence.get('partial_measurement_models', [])) or 'none'}`; these are not treated as completed correctness evaluations.",
        f"- Metadata-only or unavailable candidates: `{', '.join(evidence.get('metadata_only_models', [])) or 'none'}`.",
        f"- Directly measured scope: {evidence.get('measured_scope', 'not recorded')}",
        f"- Sourced/model-level scope: {evidence.get('sourced_scope', 'not recorded')}",
        f"- Adapter implementation scope: {evidence.get('adapter_scope', 'not recorded')}",
        f"- Unresolved comparative scope: {evidence.get('unresolved_scope', 'not recorded')}",
        "",
        "## Candidate identity and known properties",
        "",
        "These are verified inventory facts, separated from observations produced by executing a model. A known source, representation, or license does not imply that the candidate was runnable here.",
        "",
        "| Candidate | Family | Status | Measurement | Output / representation | Native rate | Native batch | Code / weight license | Differentiable boundary |",
        "| --- | --- | --- | --- | --- | ---: | --- | --- | --- |",
    ]
    for model in models:
        identity = model.get("identity", {})
        native_batch = (
            ", ".join(str(value) for value in identity.get("native_batch_sizes", []))
            or "n/a"
        )
        licenses = f"{identity.get('code_license', 'n/a')} / {identity.get('weight_license', 'n/a')}"
        lines.append(
            f"| `{model.get('model_id', 'unknown')}` | `{model.get('family', 'unknown')}` | `{model.get('status', 'unknown')}` | `{model.get('measurement_status', 'unknown')}` | `{model.get('output_contract', identity.get('output_contract', 'unknown'))}`; {_identity_representation(identity)} | {identity.get('native_sample_rate', 'n/a')} | `{native_batch}` | `{licenses}` | `{identity.get('differentiable_boundary', 'n/a')}` |"
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
        lines.append(
            f"| `{model.get('model_id', 'unknown')}` | `{source}` | `{checkpoint}` | {reason} |"
        )
    lines.extend(
        [
            "",
            "Checkpoint lock status is explicit in the JSON source of truth: `locked` means the public digest and byte size are fixed; `gated_digest_not_exposed_without_access` means the immutable model revision and public size are recorded but the upstream gated service did not expose a digest without access. Neither state implies local executability.",
        ]
    )
    lines.extend(
        [
            "",
            "Sourced representation notes: Timbre-Trap is retained as a native frame/pitch output and is not given a fabricated note-event decoder; YourMT3 variants expose stock note-event output; MuScriptor exposes timing-corrected MIDI note events with stock prelude forcing. These facts describe upstream interfaces, not measured OBRUXO performance.",
            "",
            "## What was actually executed",
            "",
            f"Basic Pitch quality/cost evidence is consumed from the landed #25/#24 reports. Executed #26 alternatives are `{', '.join(model.get('model_id', 'unknown') for model in measured if model.get('model_id') != 'basic_pitch') or 'none'}`; partial-coverage alternatives are identified separately and are not treated as completed correctness evaluations. Blocked candidates are reported separately. No new rendering or Basic Pitch rerun is implied.",
            "",
        ]
    )
    population_sections = (
        ("note_event", "Note-event candidates"),
        ("frame_comparable", "Frame-comparable alternatives"),
    )
    if any(
        isinstance(same_population.get(key), Mapping)
        and same_population[key].get("models")
        for key, _ in population_sections
    ):
        lines.extend(
            [
                "## Same-population correctness comparison",
                "",
                "This is the apples-to-apples comparison requested for the already completed candidate rows. Each table uses only the exact pair intersection on which every listed candidate returned `ok`; it is a diagnostic success-only view, not a replacement for the full #25 population, coverage, or failure-penalized results.",
                "",
            ]
        )
        for population_key, heading in population_sections:
            population = same_population.get(population_key) or {}
            population_models = population.get("models") or []
            if not population_models:
                continue
            lines.extend(
                [
                    f"### {heading} (`{population.get('eligible_pairs', 'n/a')}` common successful pairs)",
                    "",
                    "| Candidate | Common successful pairs | Onset+pitch F1 | Onset+pitch+offset F1 | Frame F1 |",
                    "| --- | ---: | ---: | ---: | ---: |",
                ]
            )
            for population_model in population_models:
                aggregate = population_model.get("aggregate") or {}
                lines.append(
                    f"| `{population_model.get('model_id', 'unknown')}` | {population_model.get('successful_pairs', 'n/a')} | {_report_number(_aggregate_f1(aggregate, 'onset_pitch'))} | {_report_number(_aggregate_f1(aggregate, 'onset_pitch_offset'))} | {_report_number(_aggregate_f1(aggregate, 'frames'))} |"
                )
            lines.extend(
                [
                    "",
                    f"- Interpretation: {population.get('interpretation', 'This conditional comparison does not replace full-population evidence.')}",
                ]
            )
        lines.extend(
            [
                "",
                f"- {same_population.get('basic_pitch_note', 'Basic Pitch is not included in this row-level intersection.')}",
                "- The shared-successful subset can show relative behavior when all candidates ran, but it must not be read as a general model ranking because candidate-specific failures are excluded by construction.",
                "",
            ]
        )
    duration_models = [
        model
        for model in measured
        if (model.get("quality") or {}).get("duration_views")
    ]
    if duration_models:
        population_order = (
            "full_population",
            "under_5_seconds",
            "under_10_seconds",
            "under_15_seconds",
            "shared_population",
            "shared_under_5_seconds",
            "shared_under_10_seconds",
            "shared_under_15_seconds",
        )

        def _duration_f1(population: Mapping[str, Any], basis: str, metric: str) -> Any:
            return (
                ((population.get(basis) or {}).get("aggregate") or {})
                .get("micro", {})
                .get(metric, {})
                .get("f1")
            )

        lines.extend(
            [
                "### Cached quality by duration and comparison population",
                "",
                "These views are recomputed from already-cached per-pair observations joined to the canonical manifest; no model inference was rerun. `Full` includes the entire eligible population with missing rows failure-penalized. `Shared` is the common TT/small cached population (and the same IDs are used for the other models where available). The JSON retains the complete aggregate diagnostics, including precision, recall, counts, pair-macro summaries, category groups, and bootstrap intervals. The landed full-population #25 baseline retains its 10,000-replicate bootstrap; these additional duration-stratum summaries use 1,000 replicates to keep this reporting-only pass bounded.",
                "",
                "| Model | Population | Eligible | Successful | Failed | Coverage | Success onset+pitch F1 | Penalized onset+pitch F1 | Success onset+pitch+offset F1 | Penalized onset+pitch+offset F1 | Success frame F1 | Penalized frame F1 |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for model in duration_models:
            quality = model.get("quality") or {}
            duration_views = quality.get("duration_views") or {}
            for population_name in population_order:
                population = duration_views.get(population_name)
                if not isinstance(population, Mapping):
                    continue
                failure_penalized = population.get("failure_penalized") or {}
                lines.append(
                    f"| `{model.get('model_id', 'unknown')}` | `{population_name}` | {failure_penalized.get('eligible_pairs', 'n/a')} | {failure_penalized.get('successful_pairs', 'n/a')} | {failure_penalized.get('failed_pairs', 'n/a')} | {_report_number(failure_penalized.get('coverage'))} | {_report_number(_duration_f1(population, 'success_only', 'onset_pitch'))} | {_report_number(_duration_f1(population, 'failure_penalized', 'onset_pitch'))} | {_report_number(_duration_f1(population, 'success_only', 'onset_pitch_offset'))} | {_report_number(_duration_f1(population, 'failure_penalized', 'onset_pitch_offset'))} | {_report_number(_duration_f1(population, 'success_only', 'frames'))} | {_report_number(_duration_f1(population, 'failure_penalized', 'frames'))} |"
                )
        lines.append(
            "- Timbre-Trap is frame/pitch output only in this benchmark; its onset+pitch and onset+pitch+offset event columns are `n/a` by design because no MIDI/event decoder was used."
        )
    basic_pitch = next(
        (model for model in models if model.get("model_id") == "basic_pitch"), None
    )
    corrected_openvino_gpu_measured = False
    model_call_winners: dict[int, str] = {}
    model_call_winner_rates: dict[int, Any] = {}
    e2e_winner: str | None = None
    e2e_winner_rate: Any = None
    quality_backend: Mapping[str, Any] = {}
    runtime_provenance: Mapping[str, Any] = {}
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
            (
                route
                for route in inference_routes
                if route.get("route") == "openvino_gpu"
            ),
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
            candidates = [
                (route, value) for route, value in candidates if value is not None
            ]
            if candidates:
                winner_route, winner_rate = max(candidates, key=lambda item: item[1])
                model_call_winners[batch_size] = winner_route.get("route", "unknown")
                model_call_winner_rates[batch_size] = winner_rate
        e2e_candidates = [
            (
                route,
                (route.get("end_to_end") or {})
                .get("audio_seconds_per_wall_second", {})
                .get("median"),
            )
            for route in inference_routes
        ]
        e2e_candidates = [
            (route, value) for route, value in e2e_candidates if value is not None
        ]
        if e2e_candidates:
            winner_route, e2e_winner_rate = max(
                e2e_candidates, key=lambda item: item[1]
            )
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
        bootstrap = (
            quality.get("success_only", {}).get("aggregate", {}).get("bootstrap", {})
        )
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
            gpu_total_memory = _report_mib(
                memory.get("openvino_gpu_total_memory_bytes")
            )
            if gpu_memory != "n/a" and gpu_total_memory != "n/a":
                gpu_memory = f"{gpu_memory} / {gpu_total_memory}"
            evidence_state = route.get("status", "unknown")
            if route.get("route") == "openvino_gpu":
                evidence_state = (
                    "measured_corrected_fp32_performance"
                    if corrected_openvino_gpu_measured
                    else "pre_fix_parity_failed"
                )
            lines.append(
                f"| `{route.get('mode', 'unknown')}` | `{route.get('route', 'unknown')}` | `{evidence_state}` | {_report_number(_route_median(route, 1, 'audio_seconds_per_second'))} | {_report_number(_route_median(route, 8, 'audio_seconds_per_second'))} | {_report_number(e2e_rate)} | {_report_number(_startup_median(route))} | {_report_mib(memory.get('host_peak_rss_bytes'))} | {allocated} / {reserved} | {gpu_memory} |"
            )
        failures = execution.get("route_failures") or []
        if failures:
            lines.extend(["", "Historical route records:"])
            for failure in failures:
                if failure.get("route") == "openvino_gpu" and openvino_diagnostic:
                    lines.append(
                        f"- `{failure.get('route')}`: historical pre-fix/default `float16` + `PERFORMANCE` -> `{failure.get('status', 'unknown')}` before timing; the current corrected route is reported separately."
                    )
                else:
                    lines.append(
                        f"- `{failure.get('route', 'unknown')}`: `{failure.get('status', 'unknown')}`. The landed #24 report suppresses timing for this route."
                    )

        historical_gpu = _historical_route(openvino_history, "openvino_gpu")
        if openvino_diagnostic or historical_gpu or corrected_openvino_gpu_measured:
            configuration = openvino_diagnostic.get("configuration") or {}
            parity = openvino_diagnostic.get("parity") or {}
            lines.extend(["", "### OpenVINO GPU evidence state", ""])
            if historical_gpu:
                historical_max = historical_gpu.get("max_across_repetitions") or {}
                historical_repetitions = historical_gpu.get("repetitions") or []
                historical_first = (
                    historical_repetitions[0] if historical_repetitions else {}
                )
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
                lines.append(
                    "- Corrected FP32 performance/resource measurements are not present in the landed #24 route rows; no corrected OpenVINO GPU performance claim is made."
                )
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
    alternative_models = [
        model for model in measured if model.get("model_id") != "basic_pitch"
    ]
    if alternative_models:
        lines.extend(
            [
                "",
                "## Executed alternative-candidate evidence",
                "",
                "These rows are measured #26 results from the exact #25 eligible population. Timbre-Trap contributes frame quality only; native note-event metrics are shown only for event-output candidates. `n/a` means the metric is not applicable, not zero.",
                "",
                "| Candidate | Success coverage | Onset+pitch F1 | Frame F1 | CPU E2E audio-s/s | XPU E2E audio-s/s | CPU host RSS MiB | Quantization |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )

        def candidate_route(
            model: Mapping[str, Any], route_id: str
        ) -> Mapping[str, Any]:
            return next(
                (
                    route
                    for route in model.get("execution", {}).get("routes", [])
                    if route.get("route") == route_id
                ),
                {},
            )

        def candidate_e2e(route: Mapping[str, Any]) -> Any:
            value = (route.get("end_to_end") or {}).get("audio_seconds_per_wall_second")
            return _median_value(value)

        def candidate_rss(route: Mapping[str, Any]) -> Any:
            return _median_value(
                (route.get("resources") or {}).get("host_peak_rss_bytes")
            )

        for model in alternative_models:
            quality = model.get("quality") or {}
            cpu = candidate_route(model, "pytorch_cpu")
            xpu = candidate_route(model, "pytorch_xpu")
            quantization = model.get("quantization") or {}
            lines.append(
                f"| `{model.get('model_id', 'unknown')}` | {_report_number((quality.get('success_only') or {}).get('coverage'))} | {_report_number(_quality_f1(quality, 'success_only', 'onset_pitch'))} | {_report_number(_quality_f1(quality, 'success_only', 'frames'))} | {_report_number(candidate_e2e(cpu))} | {_report_number(candidate_e2e(xpu))} | {_report_mib(candidate_rss(cpu))} | `{quantization.get('status', 'not_run')}` |"
            )
        lines.extend(["", "### Alternative quality and route details", ""])
        for model in alternative_models:
            model_id = model.get("model_id", "unknown")
            quality = model.get("quality") or {}
            lines.extend([f"#### `{model_id}`", ""])
            if model.get("measurement_status") != "complete":
                lines.append(
                    f"- Measurement status: `{model.get('measurement_status', 'unknown')}`; a later apples-to-apples correctness run is still required before this candidate can be treated as complete."
                )
                execution_note = (model.get("execution") or {}).get("execution_note")
                if execution_note:
                    lines.append(f"- Execution note: {execution_note}")
            for view_name in ("success_only", "failure_penalized"):
                view = quality.get(view_name) or {}
                lines.append(
                    f"- `{view_name}`: `{view.get('successful_pairs', 'n/a')}` / `{view.get('eligible_pairs', 'n/a')}` successful, coverage `{_report_number(view.get('coverage'))}`, onset+pitch F1 `{_report_number(_quality_f1(quality, view_name, 'onset_pitch'))}`, onset+pitch+offset F1 `{_report_number(_quality_f1(quality, view_name, 'onset_pitch_offset'))}`, frame F1 `{_report_number(_quality_f1(quality, view_name, 'frames'))}`."
                )
            aggregate = (quality.get("success_only") or {}).get("aggregate") or {}
            groups = aggregate.get("groups") or {}
            category_rows: list[tuple[str, str, int, Any, Any]] = []
            for field in (
                "type",
                "note_density_class",
                "pitch_register_class",
                "duration_class",
                "polyphony_class",
                "role",
                "envelope",
                "performance",
            ):
                values = groups.get(field) or {}
                if not isinstance(values, Mapping):
                    continue
                for category, value in values.items():
                    if not isinstance(value, Mapping):
                        continue
                    support = int(
                        value.get("pair_count", value.get("eligible_pairs", 0)) or 0
                    )
                    micro = value.get("micro") or {}
                    onset = (micro.get("onset_pitch") or {}).get("f1")
                    frame = (micro.get("frames") or {}).get("f1")
                    if support:
                        category_rows.append(
                            (
                                f"{field}={category}",
                                str(category),
                                support,
                                onset,
                                frame,
                            )
                        )
            if category_rows:
                best = max(
                    category_rows,
                    key=lambda row: float(row[3]) if row[3] is not None else -1.0,
                )
                worst = min(
                    category_rows,
                    key=lambda row: float(row[3]) if row[3] is not None else 2.0,
                )
                lines.append(
                    f"- Category range (success-only onset+pitch F1): highest `{best[0]}` = `{_report_number(best[3])}` over `{best[2]}` pairs; lowest `{worst[0]}` = `{_report_number(worst[3])}` over `{worst[2]}` pairs. Small supports should not be treated as robust rankings."
                )
            route_rows = []
            for route_id in ("pytorch_cpu", "pytorch_xpu"):
                route = candidate_route(model, route_id)
                if not route:
                    continue
                startup = _median_value(
                    (route.get("startup") or {}).get("model_load_seconds")
                )
                calls = _median_value(
                    (route.get("steady_state") or {}).get("calls_per_second")
                )
                route_rows.append(
                    f"`{route_id}` status `{route.get('status', 'unknown')}`, load `{_report_number(startup)}` s, steady calls/s `{_report_number(calls)}`, E2E audio-s/s `{_report_number(candidate_e2e(route))}`"
                )
            if route_rows:
                lines.append("- Execution: " + "; ".join(route_rows) + ".")
            observed_batches = execution.get("segment_batch_sizes_observed")
            cache_policy = execution.get("cache_reuse_policy")
            adapter_configuration = execution.get("adapter_configuration")
            if observed_batches is not None or cache_policy or adapter_configuration:
                lines.append(
                    f"- Adapter execution configuration: `{adapter_configuration or 'none'}`; observed segment batch sizes `{observed_batches or 'not recorded'}`; cache reuse policy `{cache_policy or 'not recorded'}`."
                )
            quantization = model.get("quantization") or {}
            if quantization:
                q_quality = quantization.get("quality") or {}
                q_execution = quantization.get("execution") or {}
                q_cpu = next(
                    (
                        route
                        for route in q_execution.get("routes", [])
                        if route.get("route") == "pytorch_cpu"
                    ),
                    {},
                )
                lines.append(
                    f"- Quantization: status `{quantization.get('status', 'unknown')}`, Linear measurement `{(quantization.get('measurement') or {}).get('original_linear_modules', 'n/a')} -> {(quantization.get('measurement') or {}).get('quantized_linear_modules', 'n/a')}`, quantized success-only onset+pitch F1 `{_report_number(_quality_f1(q_quality, 'success_only', 'onset_pitch'))}`, CPU E2E audio-s/s `{_report_number(candidate_e2e(q_cpu))}`."
                )

    if corrected_openvino_gpu_measured:
        model_call_summary = (
            ", ".join(
                f"batch {batch_size}: `{route}` ({_report_number(model_call_winner_rates.get(batch_size))} audio-s/s)"
                for batch_size, route in model_call_winners.items()
            )
            or "not recorded"
        )
        e2e_summary = (
            f"`{e2e_winner}` ({_report_number(e2e_winner_rate)} audio-s/s)"
            if e2e_winner
            else "not recorded"
        )
        measured_route_note = f"- The measured #24 model-call throughput winners were {model_call_summary}; the end-to-end winner was {e2e_summary}. These are Basic Pitch route findings, not alternative-model results."
        bounded_route_note = "- The bounded corrected parity diagnostic remains a correctness result; the corrected GPU timing/resource rows above are the separate measured result."
    else:
        measured_route_note = "- The observed Basic Pitch route trade-offs and the historical pre-fix OpenVINO GPU failure are findings of #24; corrected GPU performance is not present in the consumed route rows."
        bounded_route_note = "- This bounded result validates numerical parity only; it does not add a corrected OpenVINO GPU speed, startup, end-to-end, memory, or resource result."
    if runtime_provenance.get("consistency") == "exact_issue_24_decision_consumed":
        quality_route_note = f"- The #25 quality result is explicitly provenanced to the exact #24-selected `{quality_backend.get('backend_id', 'unknown')}` route on `{quality_backend.get('device', 'unknown')}`; it does not establish quality equivalence for any other backend."
    else:
        quality_route_note = "- The #25 quality route provenance is not an exact #24 decision match, so it is not treated as the canonical corpus baseline."
    lines.extend(
        [
            "",
            "## What could not be executed",
            "",
            "The table distinguishes a genuine candidate-level blocker or load failure from a measured candidate. No unavailable candidate receives an invented quality or cost result.",
            "",
            "| Candidate | Status | Concrete blocker/failure | What this prevents |",
            "| --- | --- | --- | --- |",
        ]
    )
    for model in unavailable:
        reason = (
            model.get("availability_reason")
            or model.get("failure_code")
            or "not recorded"
        )
        lines.append(
            f"| `{model.get('model_id', 'unknown')}` | `{model.get('status', 'unknown')}` | {reason} | no comparative quality/cost result |"
        )
    if not unavailable:
        lines.append("| none | — | all configured candidates executed | — |")
    partial_models = [
        model
        for model in models
        if model.get("measurement_status") == "partial_pair_coverage"
    ]
    if partial_models:
        lines.extend(
            [
                "",
                "## Partial or incomplete candidate execution",
                "",
                "These candidates produced some pair-level evidence but did not complete the exact-population correctness gate. Their failure-penalized view is reported as observed runtime behavior, not as a substitute for a completed correctness evaluation.",
                "",
                "| Candidate | Successful pairs | Eligible pairs | Remaining requirement |",
                "| --- | ---: | ---: | --- |",
            ]
        )
        for model in partial_models:
            view = (model.get("quality") or {}).get("success_only") or {}
            lines.append(
                f"| `{model.get('model_id', 'unknown')}` | {view.get('successful_pairs', 'n/a')} | {view.get('eligible_pairs', 'n/a')} | apples-to-apples correctness rerun on the full #25 population |"
            )
    lines.extend(
        [
            "",
            "## Conclusions by evidence class",
            "",
            "### Directly supported by measured results",
            "",
            f"- {conclusion.get('measured', 'not recorded')}",
            "- Basic Pitch remains the inherited baseline for the landed #24/#25 contracts; alternative rows are separate #26 measurements and do not replace that provenance.",
            measured_route_note,
            quality_route_note,
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
            f"- {conclusion.get('quality', 'No unavailable model receives a fabricated quality score.')}",
            "",
            "## What is required before the intended comparison can be completed",
            "",
        ]
    )
    if unavailable:
        lines.append(
            "The remaining blocked candidates require the following concrete external prerequisites:"
        )
        lines.append("")
        for model in unavailable:
            reason = (
                model.get("availability_reason")
                or model.get("failure_code")
                or "not recorded"
            )
            lines.append(f"- `{model.get('model_id', 'unknown')}`: {reason}.")
        lines.append(
            "- After those prerequisites become available, run only the fixed common #25 population and applicable #24 cost routes; do not infer their results from measured candidates."
        )
    else:
        lines.append(
            "All configured candidates have executable evidence under the fixed contract; no additional acquisition blocker is recorded."
        )
    lines.extend(
        [
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
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
            newline="\n",
        ) as handle:
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
    if not input_root.is_relative_to(
        (Path(__file__).resolve().parents[1] / "outputs").resolve()
    ):
        raise ValueError(
            "report input must be inside the ignored benchmark output area"
        )
    json_file = _approved_report(json_path)
    markdown_file = _approved_report(markdown_path)
    if not force and (json_file.exists() or markdown_file.exists()):
        raise FileExistsError("refusing to overwrite public report without force=True")
    config = (
        config_path or Path(__file__).resolve().parents[1] / "config" / "models.yaml"
    )
    specs = load_model_specs(config)
    report = build_public_report(specs, input_root)
    _atomic_text(
        json_file, json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    _atomic_text(markdown_file, _markdown(report))
    return report

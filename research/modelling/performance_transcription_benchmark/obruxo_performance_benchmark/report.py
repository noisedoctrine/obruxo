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
            "status": "measured" if successful else "unavailable",
            "failure_code": None,
            "routes": routes,
            "timing_contract": report.get("config"),
            "route_failures": failures,
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
            "execution": {
                "status": runtime.get("status", "unavailable"),
                "failure_code": runtime.get("failure_code"),
                "routes": routes,
                "timing_contract": runtime.get("timing_contract") or runtime.get("config"),
                "phases": runtime.get("phases"),
                "route_failures": runtime.get("route_failures", []),
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
            }
        models.append(item)
    return sanitize_public_report(
        {
            "format_version": 1,
            "comparison": {
                "quality_contract": "landed_issue_25_metrics_and_10000_replicate_seed_0_bootstrap",
                "cost_contract": "landed_issue_24_end_to_end_boundary",
                "quantization_contract": "cpu_dynamic_qint8_ordinary_linear_only",
                "no_composite_winner": True,
            },
            "models": models,
            "conclusion": {
                "quality": "No quality score is published for unavailable models or an empty eligible population.",
                "cost": "Cost rows remain separate by route and are unavailable when the fixed smoke input or candidate runtime is unavailable.",
                "representation": "Representation and licensing are inventories, not an integration decision.",
                "later_integration": "No candidate is promoted without executable common-corpus evidence.",
            },
        }
    )


def _markdown(report: Mapping[str, Any]) -> str:
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

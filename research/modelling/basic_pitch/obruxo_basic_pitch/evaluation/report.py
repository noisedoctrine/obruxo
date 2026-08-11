"""Private-result loading and sanitized PresetShare baseline reports."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .aggregate import HEADLINE_METRICS


class ReportInputError(ValueError):
    """A report input or destination violates the public-report contract."""


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _reports_root() -> Path:
    return (Path(__file__).resolve().parents[2] / "reports").resolve()


def _report_path(path: Path | str) -> Path:
    root = _reports_root()
    resolved = Path(path).resolve(strict=False)
    if resolved == root or not resolved.is_relative_to(root):
        raise ReportInputError("public report must be inside the Basic Pitch reports directory")
    return resolved


def _read_json(path: Path | str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).resolve(strict=True).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportInputError("report input could not be read") from exc
    if not isinstance(value, dict):
        raise ReportInputError("report input must be an object")
    return value


def _public_metric(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value.get(key)
        for key in ("reference_count", "prediction_count", "count_bias", "tp", "fp", "fn", "precision", "recall", "f1", "false_negative_rate", "false_positive_fraction")
    }


def sanitize_aggregate(value: Mapping[str, Any]) -> dict[str, Any]:
    """Keep aggregate evidence while dropping private per-pair/preset material."""
    bootstrap = dict(value.get("bootstrap", {}))
    if "cluster_rule" in bootstrap:
        bootstrap["cluster_rule"] = "known preset grouping; unknown presets are singleton clusters"
    result: dict[str, Any] = {
        "pair_count": value.get("pair_count", 0),
        "successful_pair_count": value.get("successful_pair_count", 0),
        "failed_pair_count": value.get("failed_pair_count", 0),
        "coverage": value.get("coverage"),
        "micro": {},
        "pair_macro": value.get("pair_macro", {}),
        "bootstrap": bootstrap,
        "failure_analysis": value.get("failure_analysis", {}),
        "groups": {},
        "polyphony_by_source_role": value.get("polyphony_by_source_role", {}),
    }
    micro = value.get("micro", {})
    for name in HEADLINE_METRICS:
        if isinstance(micro.get(name), Mapping):
            result["micro"][name] = _public_metric(micro[name])
    groups = value.get("groups", {})
    if isinstance(groups, Mapping):
        for field, entries in groups.items():
            if not isinstance(entries, Mapping):
                continue
            result["groups"][str(field)] = {
                str(category): {
                    key: summary.get(key)
                    for key in ("pair_count", "successful_pair_count", "failed_pair_count", "coverage", "micro", "pair_macro")
                }
                for category, summary in entries.items()
                if isinstance(summary, Mapping)
            }
    return result


def _pairing_summary(audit: Mapping[str, Any]) -> dict[str, Any]:
    source_snapshot = audit.get("source_snapshot", {})
    return {
        "corpus_layout": audit.get("corpus_layout"),
        "candidate_count": audit.get("candidate_count", 0),
        "eligible_count": audit.get("eligible_count", 0),
        "excluded_count": audit.get("excluded_count", 0),
        "ambiguous_count": audit.get("ambiguous_count", 0),
        "excluded_by_reason": dict(audit.get("excluded_by_reason", {})),
        "selected_pairing_methods": list(audit.get("selected_pairing_methods", [])),
        "derived_render_opt_in": bool(audit.get("derived_render_opt_in", False)),
        "spot_check": {
            "count": audit.get("spot_check", {}).get("count", 0),
            "discrepancies": audit.get("spot_check", {}).get("discrepancies"),
            "status": audit.get("spot_check", {}).get("status", "unknown"),
        },
        "source_stat_records": source_snapshot.get("source_stat_records"),
        "source_stat_mismatches": source_snapshot.get("source_stat_mismatches"),
    }


def build_sanitized_report(audit: Mapping[str, Any], run: Mapping[str, Any], aggregate: Mapping[str, Any]) -> dict[str, Any]:
    """Construct the committed report from private inputs without copying row data."""
    return {
        "format_version": 1,
        "status": run.get("status", "unknown"),
        "failure_code": run.get("failure_code"),
        "pairing": _pairing_summary(audit),
        "model": run.get("model", {}),
        "backend": {
            "backend_id": run.get("backend", {}).get("backend_id"),
            "contract_version": run.get("backend", {}).get("contract_version"),
            "boundary": run.get("backend", {}).get("boundary"),
            "precision": run.get("backend", {}).get("precision"),
        },
        "decoder": run.get("decoder", {}),
        "aggregate": sanitize_aggregate(aggregate),
        "interpretation": {
            "measured": "Aggregate values are computed from successful pair results using fixed stock Basic Pitch decoding and no threshold tuning.",
            "meaning": "Quality describes the frozen Basic Pitch performance prior; it is not a claim about source audio reconstruction or an OBRUXO training objective.",
        },
    }


def _metric_row(name: str, value: Mapping[str, Any]) -> str:
    label = {"onset_pitch": "Onset + pitch", "onset_pitch_offset": "Onset + pitch + offset", "frames": "Frame"}.get(name, name)
    f1 = value.get("f1")
    return f"| {label} | {value.get('reference_count', 0)} | {value.get('prediction_count', 0)} | {value.get('tp', 0)} | {value.get('fp', 0)} | {value.get('fn', 0)} | {'n/a' if f1 is None else f'{f1:.6f}'} |"


def _markdown(report: Mapping[str, Any]) -> str:
    pairing = report["pairing"]
    aggregate = report["aggregate"]
    lines = [
        "# PresetShare Basic Pitch baseline",
        "",
        "## Corpus pairing",
        "",
        f"- Layout: `{pairing.get('corpus_layout')}`.",
        f"- Candidate directories: `{pairing.get('candidate_count', 0)}`.",
        f"- Eligible pairs: `{pairing.get('eligible_count', 0)}`.",
        f"- Excluded candidates: `{pairing.get('excluded_count', 0)}`; ambiguous: `{pairing.get('ambiguous_count', 0)}`.",
        f"- Exclusions by reason: {'; '.join(f'`{reason}`={count}' for reason, count in pairing.get('excluded_by_reason', {}).items()) or 'none'}.",
        f"- Pairing methods: `{', '.join(pairing.get('selected_pairing_methods', [])) or 'none'}`.",
        "- Pair identity uses the observed direct-directory relationship with exactly one MIDI and one audio file; no fuzzy matching is used.",
        f"- Derived-render opt-in: `{pairing.get('derived_render_opt_in', False)}`. Any derived audio is labeled separately and remains ignored local output.",
        "- Existing audio remains read-only; derived audio uses the parent-approved Vital/Pedalboard path and is never described as historical source audio.",
        f"- Private source-stat records: `{pairing.get('source_stat_records')}`; mismatches: `{pairing.get('source_stat_mismatches')}`.",
        "",
        "## Evaluation status",
        "",
        f"- Status: `{report.get('status')}`.",
        f"- Failure code: `{report.get('failure_code') or 'none'}`.",
        "- The report distinguishes unavailable execution from measured quality; unavailable rows are not treated as zero-quality predictions.",
        "",
        "## Overall quality",
        "",
        "| Metric | Reference | Predicted | TP | FP | FN | F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    micro = aggregate.get("micro", {})
    for name in HEADLINE_METRICS:
        if isinstance(micro.get(name), Mapping):
            lines.append(_metric_row(name, micro[name]))
    lines.extend([
        "",
        f"- Pair coverage: `{aggregate.get('successful_pair_count', 0)}/{aggregate.get('pair_count', 0)}`.",
        "- Micro metrics are derived from total counts. Pair-macro values and preset-cluster bootstrap intervals remain separate.",
        "",
        "## Category summaries",
        "",
        "The committed report retains counts and support for objective MIDI categories and explicit source metadata categories. Unknown metadata remains unknown; no style labels are inferred from filenames.",
        "",
    ])
    for field, entries in aggregate.get("groups", {}).items():
        if not entries:
            continue
        lines.extend([f"### {field}", "", "| Category | Pairs | Successful | Coverage | Onset + pitch F1 |", "| --- | ---: | ---: | ---: | ---: |"])
        for category, summary in entries.items():
            f1 = summary.get("micro", {}).get("onset_pitch", {}).get("f1")
            lines.append(f"| `{category}` | {summary.get('pair_count', 0)} | {summary.get('successful_pair_count', 0)} | {'n/a' if summary.get('coverage') is None else f'{summary.get('coverage'):.3f}'} | {'n/a' if f1 is None else f'{f1:.6f}'} |")
        lines.append("")
    lines.extend([
        "## Failure analysis",
        "",
        f"- Onset + pitch false negatives: `{aggregate.get('failure_analysis', {}).get('note', {}).get('onset_pitch_false_negatives', 0)}`; false positives: `{aggregate.get('failure_analysis', {}).get('note', {}).get('onset_pitch_false_positives', 0)}`.",
        f"- Additional offset false negatives: `{aggregate.get('failure_analysis', {}).get('note', {}).get('additional_offset_false_negatives', 0)}`.",
        f"- Assigned near-onset pitch errors: `{aggregate.get('failure_analysis', {}).get('pitch', {}).get('assigned_near_onset_errors', 0)}`; octave errors: `{aggregate.get('failure_analysis', {}).get('pitch', {}).get('octave_error_count', 0)}`; ambiguous/unassigned: `{aggregate.get('failure_analysis', {}).get('pitch', {}).get('unassigned_near_onset_errors', 0)}`.",
        "- Pair-level failures and private best/worst rows remain under ignored local outputs. This committed view contains only aggregate coverage and stable exclusion counts.",
        "",
        "## Interpretation",
        "",
        report["interpretation"]["measured"],
        "",
        report["interpretation"]["meaning"],
        "",
        "## Provenance and limits",
        "",
        f"- Model: `{report.get('model', {}).get('model_id', 'unknown')}`.",
        f"- Backend: `{report.get('backend', {}).get('backend_id', 'unknown')}`; precision: `{report.get('backend', {}).get('precision', 'unknown')}`.",
        "- Stock settings are onset threshold 0.5, frame threshold 0.3, minimum note length 11 frames, inferred onsets enabled, Melodia fallback enabled, and no frequency limits.",
        "- No composite score is used and no upstream model/runtime setting is tuned from corpus results.",
        "",
    ])
    return "\n".join(lines)


def _assert_public(value: Any) -> None:
    serialized = json.dumps(value, sort_keys=True)
    forbidden = ("pair-", ".mid", ".wav", "preset_id", "preset_path", "audio_path", "midi_path", "author", "request_id")
    if any(token in serialized.lower() for token in forbidden):
        raise ReportInputError("sanitized report contains a private identifier or source reference")


def write_sanitized_reports(
    audit_path: Path | str,
    run_path: Path | str,
    aggregate_path: Path | str,
    json_path: Path | str,
    markdown_path: Path | str,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Write only sanitized aggregate JSON/Markdown reports under tracked reports."""
    report = build_sanitized_report(_read_json(audit_path), _read_json(run_path), _read_json(aggregate_path))
    _assert_public(report)
    json_file = _report_path(json_path)
    markdown_file = _report_path(markdown_path)
    if not force and (json_file.exists() or markdown_file.exists()):
        raise FileExistsError("refusing to overwrite public reports without force=True")
    _atomic_write(json_file, json.dumps(report, indent=2, sort_keys=True) + "\n")
    _atomic_write(markdown_file, _markdown(report))
    return report

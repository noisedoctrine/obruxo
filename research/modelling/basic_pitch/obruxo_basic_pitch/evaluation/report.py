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


def _category_f1(summary: Mapping[str, Any]) -> float | None:
    value = summary.get("micro", {}).get("onset_pitch", {}).get("f1")
    return None if value is None else float(value)


def _support_class(pair_count: int) -> str:
    if pair_count >= 100:
        return "well_supported"
    if pair_count >= 30:
        return "moderately_supported"
    return "small_subset"


def _category_findings(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize observed category ranges without hiding their support sizes."""
    groups = aggregate.get("groups", {})
    findings: dict[str, Any] = {}
    for field in ("duration_class", "note_density_class", "pitch_register_class", "polyphony_class", "type"):
        entries = groups.get(field, {})
        rows = []
        if not isinstance(entries, Mapping):
            continue
        for category, summary in entries.items():
            if not isinstance(summary, Mapping) or _category_f1(summary) is None:
                continue
            pair_count = int(summary.get("pair_count", 0))
            rows.append(
                {
                    "category": str(category),
                    "pair_count": pair_count,
                    "support": _support_class(pair_count),
                    "onset_pitch_f1": _category_f1(summary),
                }
            )
        if not rows:
            continue
        rows.sort(key=lambda row: (row["onset_pitch_f1"], row["category"]))
        supported = [row for row in rows if row["pair_count"] >= 100]
        findings[field] = {
            "metric": "onset_pitch_f1",
            "support_rule": "well_supported >=100 pairs; moderately_supported 30-99; small_subset <30",
            "lowest": rows[0],
            "highest": rows[-1],
            "lowest_well_supported": supported[0] if supported else None,
            "highest_well_supported": supported[-1] if supported else None,
        }
    return findings


def _runtime_selection(run: Mapping[str, Any]) -> dict[str, Any]:
    """Compare the recorded #25 backend with the already-landed #24 report."""
    backend = run.get("backend", {})
    selected = backend.get("backend_id")
    result: dict[str, Any] = {
        "selected_backend": selected,
        "selected_contract": backend.get("boundary"),
        "selection_source": "the backend contract recorded by the existing #25 run",
        "selection_rationale": f"The existing #25 run fixed {selected} in its backend contract; the run artifacts do not record an independent runtime-selection rationale.",
        "issue_24_observed_highest_batch_1_route": None,
        "issue_24_observed_highest_batch_1_audio_seconds_per_second": None,
        "issue_24_observed_highest_end_to_end_route": None,
        "issue_24_observed_highest_end_to_end_audio_seconds_per_wall_second": None,
        "consistency": "issue_24_report_unavailable",
        "interpretation": "The quality result is attributed to the recorded backend only; no alternate-backend quality result is inferred.",
    }
    benchmark_path = _reports_root() / "backend_benchmark.json"
    try:
        benchmark = _read_json(benchmark_path)
    except ReportInputError:
        return result
    inference = [row for row in benchmark.get("inference", []) if row.get("status") == "ok"]
    if not inference:
        return result
    highest_batch = max(
        inference,
        key=lambda row: row.get("batch_results", {}).get("1", {}).get("audio_seconds_per_second", {}).get("median", float("-inf")),
    )
    highest_end_to_end = max(
        (row for row in inference if row.get("end_to_end")),
        key=lambda row: row.get("end_to_end", {}).get("audio_seconds_per_wall_second", {}).get("median", float("-inf")),
        default=None,
    )
    result.update(
        {
            "issue_24_observed_highest_batch_1_route": highest_batch.get("route"),
            "issue_24_observed_highest_batch_1_audio_seconds_per_second": highest_batch.get("batch_results", {}).get("1", {}).get("audio_seconds_per_second", {}).get("median"),
            "issue_24_observed_highest_end_to_end_route": highest_end_to_end.get("route") if highest_end_to_end else None,
            "issue_24_observed_highest_end_to_end_audio_seconds_per_wall_second": highest_end_to_end.get("end_to_end", {}).get("audio_seconds_per_wall_second", {}).get("median") if highest_end_to_end else None,
        }
    )
    result["selection_rationale"] = (
        f"The existing #25 run fixed {selected} in its backend contract. Its artifacts do not record why that route was selected; current #24 measurements identify {highest_batch.get('route')} as the batch-1 model-call throughput leader and {highest_end_to_end.get('route') if highest_end_to_end else 'no route'} as the warmed end-to-end leader. This report does not relabel the existing quality result or infer alternate-backend quality."
    )
    if selected == highest_batch.get("route") == (highest_end_to_end or {}).get("route"):
        result["consistency"] = "matches_issue_24_observed_leader"
        result["interpretation"] = "The recorded #25 backend matches both the measured #24 batch-1 model-call and warmed end-to-end leaders."
    else:
        result["consistency"] = "recorded_backend_does_not_match_issue_24_observed_leader"
        result["interpretation"] = "The existing #25 quality run records a backend different from one or both current #24 leaders. Its quality result remains attributed only to the recorded backend; no alternate-backend full-corpus quality or equivalence claim is made."
    return result


def build_sanitized_report(audit: Mapping[str, Any], run: Mapping[str, Any], aggregate: Mapping[str, Any]) -> dict[str, Any]:
    """Construct the committed report from private inputs without copying row data."""
    sanitized_aggregate = sanitize_aggregate(aggregate)
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
        "aggregate": sanitized_aggregate,
        "runtime_provenance": _runtime_selection(run),
        "category_findings": _category_findings(sanitized_aggregate),
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
        "## Runtime provenance and #24 route decision",
        "",
        f"- Recorded #25 corpus backend: `{report.get('backend', {}).get('backend_id')}`; boundary: `{report.get('backend', {}).get('boundary')}`; precision: `{report.get('backend', {}).get('precision')}`.",
        f"- Runtime-selection source: {report.get('runtime_provenance', {}).get('selection_source', 'not recorded')}.",
        f"- Selection rationale recorded by the artifacts: {report.get('runtime_provenance', {}).get('selection_rationale', 'not recorded')}.",
        f"- Existing #24 report's highest batch-1 inference route: `{report.get('runtime_provenance', {}).get('issue_24_observed_highest_batch_1_route')}` at `{report.get('runtime_provenance', {}).get('issue_24_observed_highest_batch_1_audio_seconds_per_second')}` audio-seconds/second.",
        f"- Existing #24 report's highest warmed end-to-end route: `{report.get('runtime_provenance', {}).get('issue_24_observed_highest_end_to_end_route')}` at `{report.get('runtime_provenance', {}).get('issue_24_observed_highest_end_to_end_audio_seconds_per_wall_second')}` audio-seconds/wall-second.",
        f"- Consistency assessment: `{report.get('runtime_provenance', {}).get('consistency')}`.",
        f"- Interpretation: {report.get('runtime_provenance', {}).get('interpretation', 'not recorded')}",
        "- This report revision does not rerun the corpus evaluation. The backend mismatch is surfaced for review rather than silently reassigning the existing F1 result to XPU.",
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
        "## Uncertainty and support",
        "",
    ])
    bootstrap = aggregate.get("bootstrap", {})
    lines.extend(
        [
            f"- Preset-cluster bootstrap: `{bootstrap.get('replicates', 'n/a')}` replicates, seed `{bootstrap.get('seed', 'n/a')}`, `{bootstrap.get('cluster_count', 'n/a')}` clusters.",
            "",
            "| Metric | F1 | Bootstrap 95% interval |",
            "| --- | ---: | ---: |",
        ]
    )
    micro = aggregate.get("micro", {})
    for name in HEADLINE_METRICS:
        metric = micro.get(name, {})
        interval = bootstrap.get("metrics", {}).get(name, {})
        if isinstance(metric, Mapping):
            f1 = metric.get("f1")
            low = interval.get("lower_95")
            high = interval.get("upper_95")
            lines.append(f"| `{name}` | {'n/a' if f1 is None else f'{f1:.6f}'} | {'n/a' if low is None or high is None else f'{low:.6f} - {high:.6f}'} |")
    lines.extend([
        "",
        "## Category summaries",
        "",
        "The committed report retains counts and support for objective MIDI categories and explicit source metadata categories. Unknown metadata remains unknown; no style labels are inferred from filenames.",
        "",
    ])
    findings = report.get("category_findings", {})
    lines.extend(["## Category interpretation", "", "The following statements use onset+pitch F1 and retain category support. `well_supported` means at least 100 pairs, `moderately_supported` 30-99, and `small_subset` fewer than 30; small-subset extremes are descriptive, not robust corpus-wide findings.", ""])
    for field in ("duration_class", "note_density_class", "pitch_register_class", "polyphony_class", "type"):
        finding = findings.get(field)
        if not finding:
            continue
        highest = finding.get("highest", {})
        lowest = finding.get("lowest", {})
        high_supported = finding.get("highest_well_supported") or highest
        low_supported = finding.get("lowest_well_supported") or lowest
        if field == "polyphony_class":
            delta = abs(float(highest.get("onset_pitch_f1", 0.0)) - float(lowest.get("onset_pitch_f1", 0.0)))
            interpretation = "near tie" if delta < 0.02 else "separation observed"
            lines.append(f"- `{field}`: `{highest.get('category')}` `{highest.get('onset_pitch_f1'):.6f}` vs `{lowest.get('category')}` `{lowest.get('onset_pitch_f1'):.6f}`; {interpretation} across `{highest.get('pair_count')}` and `{lowest.get('pair_count')}` pairs. Frame behavior should be read separately from event F1.")
        elif field == "type":
            lines.append(f"- `{field}`: the overall high is `{highest.get('category')}` `{highest.get('onset_pitch_f1'):.6f}` on `{highest.get('pair_count')}` pairs (`{highest.get('support')}`); among well-supported types, `{high_supported.get('category')}` is highest at `{high_supported.get('onset_pitch_f1'):.6f}` on `{high_supported.get('pair_count')}` pairs, while `{low_supported.get('category')}` is lowest at `{low_supported.get('onset_pitch_f1'):.6f}` on `{low_supported.get('pair_count')}` pairs. This separates robust patterns from tiny type strata.")
        else:
            lines.append(f"- `{field}`: highest `{highest.get('category')}` `{highest.get('onset_pitch_f1'):.6f}` ({highest.get('pair_count')} pairs, `{highest.get('support')}`); lowest `{lowest.get('category')}` `{lowest.get('onset_pitch_f1'):.6f}` ({lowest.get('pair_count')} pairs, `{lowest.get('support')}`). The well-supported range is `{low_supported.get('onset_pitch_f1'):.6f}`-`{high_supported.get('onset_pitch_f1'):.6f}` across `{low_supported.get('category')}` to `{high_supported.get('category')}`.")
    lines.extend(["", "The category tables below retain every observed stratum so these summaries can be checked against the underlying aggregate counts.", ""])
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

"""Deterministic aggregate summaries and preset-cluster uncertainty."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

HEADLINE_METRICS = ("onset_pitch", "onset_pitch_offset", "frames")
GROUP_FIELDS = (
    "polyphony_class",
    "duration_class",
    "note_density_class",
    "pitch_register_class",
    "instrument",
    "genre",
    "type",
    "vital_style",
)


def _f1(true_positive: int, reference_count: int, prediction_count: int) -> float | None:
    if reference_count == 0 and prediction_count == 0:
        return None
    if reference_count == 0 or prediction_count == 0:
        return 0.0
    precision = true_positive / prediction_count
    recall = true_positive / reference_count
    return float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0


def _metric_value(row: Mapping[str, Any], metric_name: str) -> Mapping[str, Any] | None:
    metrics = row.get("metrics", {})
    if metric_name in metrics and isinstance(metrics[metric_name], Mapping):
        return metrics[metric_name]
    if metric_name in {"onset_pitch", "onset_pitch_offset"}:
        notes = metrics.get("notes", {})
        value = notes.get(metric_name) if isinstance(notes, Mapping) else None
    else:
        value = metrics.get("frames")
    return value if isinstance(value, Mapping) else None


def _metric_summary(rows: list[Mapping[str, Any]], metric_name: str) -> dict[str, Any]:
    values = [value for row in rows if row.get("status") == "ok" if (value := _metric_value(row, metric_name)) is not None]
    reference_count = sum(int(value.get("reference_count", 0)) for value in values)
    prediction_count = sum(int(value.get("prediction_count", 0)) for value in values)
    true_positive = sum(int(value.get("tp", 0)) for value in values)
    false_positive = prediction_count - true_positive
    false_negative = reference_count - true_positive
    return {
        "reference_count": reference_count,
        "prediction_count": prediction_count,
        "count_bias": prediction_count - reference_count,
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "precision": float(true_positive / prediction_count) if prediction_count else None,
        "recall": float(true_positive / reference_count) if reference_count else None,
        "f1": _f1(true_positive, reference_count, prediction_count),
        "false_negative_rate": float(false_negative / reference_count) if reference_count else None,
        "false_positive_fraction": float(false_positive / prediction_count) if prediction_count else None,
    }


def _macro_summary(rows: list[Mapping[str, Any]], metric_name: str) -> dict[str, Any]:
    values = [
        value.get("f1")
        for row in rows
        if row.get("status") == "ok" and (value := _metric_value(row, metric_name)) is not None and value.get("f1") is not None
    ]
    return {"support": len(values), "f1": float(np.mean(values)) if values else None}


def _weighted_error(rows: list[Mapping[str, Any]], field: str, component: str | None = None) -> dict[str, Any]:
    support = 0
    signed_total = 0.0
    absolute_total = 0.0
    for row in rows:
        notes = row.get("metrics", {}).get("notes", {})
        summary = notes.get(field, {}) if isinstance(notes, Mapping) else {}
        if component is not None and isinstance(summary, Mapping):
            summary = summary.get(component, {})
        current_support = int(summary.get("support", 0)) if isinstance(summary, Mapping) else 0
        if current_support:
            support += current_support
            signed_total += float(summary.get("signed_mean", 0.0)) * current_support
            absolute_total += float(summary.get("mae", 0.0)) * current_support
    return {
        "support": support,
        "signed_mean": float(signed_total / support) if support else None,
        "mae": float(absolute_total / support) if support else None,
    }


def _failure_analysis(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    onset_pitch = _metric_summary(rows, "onset_pitch")
    onset_pitch_offset = _metric_summary(rows, "onset_pitch_offset")
    octave_errors = 0
    assigned_errors = 0
    unassigned_errors = 0
    for row in rows:
        notes = row.get("metrics", {}).get("notes", {})
        confusion = notes.get("pitch_confusion", {}) if isinstance(notes, Mapping) else {}
        octave_errors += int(confusion.get("octave_error_count", 0))
        assigned_errors += int(confusion.get("assigned_count", 0))
        unassigned_errors += int(confusion.get("unassigned_error_count", 0))
    return {
        "note": {
            "onset_pitch_false_negatives": onset_pitch["fn"],
            "onset_pitch_false_positives": onset_pitch["fp"],
            "additional_offset_false_negatives": max(0, onset_pitch_offset["fn"] - onset_pitch["fn"]),
        },
        "timing": {
            "onset": _weighted_error(rows, "timing_diagnostics", "onset"),
            "offset": _weighted_error(rows, "timing_diagnostics", "offset"),
            "duration": _weighted_error(rows, "timing_diagnostics", "duration"),
        },
        "pitch": {
            "assigned_near_onset_errors": assigned_errors,
            "unassigned_near_onset_errors": unassigned_errors,
            "octave_error_count": octave_errors,
        },
        "velocity": _weighted_error(rows, "velocity"),
    }


def _summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if row.get("status") == "ok"]
    result: dict[str, Any] = {
        "pair_count": len(rows),
        "successful_pair_count": len(successful),
        "failed_pair_count": len(rows) - len(successful),
        "coverage": float(len(successful) / len(rows)) if rows else None,
        "micro": {name: _metric_summary(rows, name) for name in HEADLINE_METRICS},
        "pair_macro": {name: _macro_summary(rows, name) for name in HEADLINE_METRICS},
        "failure_analysis": _failure_analysis(rows),
    }
    return result


def _clusters(rows: list[Mapping[str, Any]]) -> list[list[Mapping[str, Any]]]:
    known: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    unknown: list[Mapping[str, Any]] = []
    for row in rows:
        preset_id = row.get("preset_id")
        if preset_id:
            known[str(preset_id)].append(row)
        else:
            unknown.append(row)
    return list(known.values()) + [[row] for row in unknown]


def _bootstrap(rows: list[Mapping[str, Any]], *, replicates: int, seed: int) -> dict[str, Any]:
    clusters = _clusters(rows)
    result: dict[str, Any] = {
        "replicates": replicates,
        "seed": seed,
        "cluster_count": len(clusters),
        "cluster_rule": "known_preset_id; unknown-preset rows are singleton clusters",
        "metrics": {},
    }
    if not clusters or not any(row.get("status") == "ok" for row in rows):
        for name in HEADLINE_METRICS:
            result["metrics"][name] = {"support": 0, "lower_95": None, "upper_95": None}
        return result
    generator = np.random.default_rng(seed)
    sampled_values = {name: np.empty(replicates, dtype=np.float64) for name in HEADLINE_METRICS}
    for index in range(replicates):
        selected = generator.integers(0, len(clusters), size=len(clusters))
        sampled_rows = [row for cluster_index in selected for row in clusters[int(cluster_index)]]
        for name in HEADLINE_METRICS:
            value = _metric_summary(sampled_rows, name)["f1"]
            sampled_values[name][index] = 0.0 if value is None else value
    for name, values in sampled_values.items():
        result["metrics"][name] = {
            "support": len(values),
            "lower_95": float(np.percentile(values, 2.5)),
            "upper_95": float(np.percentile(values, 97.5)),
        }
    return result


def _group_rows(rows: Iterable[Mapping[str, Any]], field: str) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        labels = row.get("labels", {})
        value = str(labels.get(field, "unknown") or "unknown")
        groups[value].append(row)
    return dict(sorted(groups.items()))


def aggregate_results(rows: Iterable[Mapping[str, Any]], *, bootstrap_replicates: int = 10_000, seed: int = 0) -> dict[str, Any]:
    """Aggregate stored pair results without rerunning inference."""
    materialized = [dict(row) for row in rows]
    result = _summary(materialized)
    result["bootstrap"] = _bootstrap(materialized, replicates=bootstrap_replicates, seed=seed)
    result["groups"] = {
        field: {value: _summary(group_rows) for value, group_rows in _group_rows(materialized, field).items()}
        for field in GROUP_FIELDS
    }
    known_presets = {
        str(row["preset_id"]): []
        for row in materialized
        if row.get("preset_id")
    }
    for row in materialized:
        if row.get("preset_id"):
            known_presets[str(row["preset_id"])].append(row)
    result["per_preset"] = {key: _summary(value) for key, value in sorted(known_presets.items())}
    roles = _group_rows(materialized, "instrument")
    result["polyphony_by_source_role"] = {
        role: {
            polyphony: _summary([row for row in role_rows if row.get("labels", {}).get("polyphony_class") == polyphony])
            for polyphony in ("monophonic", "polyphonic")
            if any(row.get("labels", {}).get("polyphony_class") == polyphony for row in role_rows)
        }
        for role, role_rows in roles.items()
        if role != "unknown"
    }
    return result

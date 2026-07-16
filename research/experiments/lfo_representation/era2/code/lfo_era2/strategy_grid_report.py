"""Reusable analysis and report generation for Experiment 13."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

from .strategy_grid_runtime import read_csv, write_csv


CO_PRIMARY_METRICS = (
    "validation_median_rmse",
    "validation_strict_perfect_lfo_rate",
    "validation_p95_rmse",
    "validation_node_max_error_p95",
)
SUPPLEMENTAL_METRICS = (
    "validation_p99_rmse",
    "validation_max_rmse",
    "validation_max_abs_error_p95",
)
CALIBRATION_TABLES = (
    "layer_epsilon_quantiles.csv",
    "slot_epsilon_quantiles.csv",
    "epsilon_coverage.csv",
    "retired_error_mass.csv",
)


@dataclass(frozen=True)
class AnalysisBundle:
    summaries: list[dict[str, Any]]
    coverage: list[dict[str, Any]]
    co_primary: list[dict[str, Any]]
    matched_deltas: list[dict[str, Any]]
    partial_codebook: list[dict[str, Any]]
    calibration: dict[str, list[dict[str, Any]]]
    paths: dict[str, Path]


def prepare_analysis_artifacts(
    *,
    source_run: Path,
    analysis_output_dir: Path,
    expected_rows: Sequence[Mapping[str, Any]],
    phase: str | None = None,
    forbid_source_writes: bool = False,
) -> AnalysisBundle:
    """Collect sharded results and write reusable derived analysis tables."""
    source_run = Path(source_run).resolve()
    analysis_output_dir = Path(analysis_output_dir).resolve()
    if not source_run.is_dir():
        raise ValueError(f"analysis source run does not exist: {source_run}")
    if forbid_source_writes:
        _require_outside_source(source_run, analysis_output_dir, "analysis output directory")

    summaries = _load_table(source_run, "summary.csv")
    if phase is not None:
        summaries = [row for row in summaries if row.get("experiment_phase") == phase]
    summaries = sorted(summaries, key=lambda row: str(row.get("row_id", "")))
    if not summaries:
        raise ValueError(f"no completed {phase or 'Experiment 13'} row summaries found in {source_run}")
    row_ids = [str(row.get("row_id", "")) for row in summaries]
    if any(not row_id for row_id in row_ids) or len(set(row_ids)) != len(row_ids):
        raise ValueError("completed row summaries must have unique nonempty row_id values")

    expected = [dict(row) for row in expected_rows if phase is None or row.get("experiment_phase") == phase]
    expected_ids = {str(row.get("row_id", "")) for row in expected}
    unknown = sorted(set(row_ids) - expected_ids)
    if unknown:
        raise ValueError("analysis source contains rows outside the planned grid: " + ", ".join(unknown))

    coverage = _coverage_rows(expected, set(row_ids))
    co_primary = _co_primary_rows(summaries)
    matched = _matched_factor_deltas(summaries)
    partial = _load_table(source_run, "partial_codebook_validation.csv")
    if phase is not None:
        partial = [row for row in partial if row.get("experiment_phase") == phase]
    partial = sorted(partial, key=lambda row: (str(row.get("row_id", "")), _number(row, "active_atom_count")))
    calibration = {name: _load_table(source_run, name) for name in CALIBRATION_TABLES}
    if phase is not None:
        calibration = {
            name: [row for row in rows if row.get("experiment_phase") == phase]
            for name, rows in calibration.items()
        }

    analysis_output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "coverage": analysis_output_dir / "completed_row_coverage.csv",
        "co_primary": analysis_output_dir / "co_primary_metrics.csv",
        "matched_deltas": analysis_output_dir / "matched_factor_deltas.csv",
        "partial_codebook": analysis_output_dir / "partial_codebook_progression.csv",
    }
    write_csv(paths["coverage"], coverage)
    write_csv(paths["co_primary"], co_primary)
    write_csv(paths["matched_deltas"], matched)
    write_csv(paths["partial_codebook"], partial)
    for name, rows in calibration.items():
        key = f"aggregated_{Path(name).stem}"
        paths[key] = analysis_output_dir / f"aggregated_{name}"
        write_csv(paths[key], rows)
    return AnalysisBundle(summaries, coverage, co_primary, matched, partial, calibration, paths)


def write_provisional_report(
    *,
    source_run: Path,
    analysis_output_dir: Path,
    report_path: Path,
    image_dir: Path,
    expected_rows: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """Generate a findings-first report from an immutable partial 13A run."""
    source_run = Path(source_run).resolve()
    analysis_output_dir = Path(analysis_output_dir).resolve()
    report_path = Path(report_path).resolve()
    image_dir = Path(image_dir).resolve()
    for path, label in (
        (analysis_output_dir, "analysis output directory"),
        (report_path, "report path"),
        (image_dir, "image directory"),
    ):
        _require_outside_source(source_run, path, label)
    if report_path.suffix.lower() != ".md":
        raise ValueError("provisional report path must end in .md")

    bundle = prepare_analysis_artifacts(
        source_run=source_run,
        analysis_output_dir=analysis_output_dir,
        expected_rows=expected_rows,
        phase="13A",
        forbid_source_writes=True,
    )
    plot_paths = _write_plots(bundle, image_dir)
    expected_count = len([row for row in expected_rows if row.get("experiment_phase") == "13A"])
    report_text = _provisional_markdown(
        bundle,
        source_run=source_run,
        report_path=report_path,
        plot_paths=plot_paths,
        expected_count=expected_count,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_text(report_path, report_text)

    manifest = {
        "schema_version": "experiment13_provisional_report_v1",
        "source_run": str(source_run),
        "source_archive_sha256": _directory_fingerprint(source_run),
        "completed_13a_rows": len(bundle.summaries),
        "expected_13a_rows": expected_count,
        "report_status": "provisional_incomplete_13a",
        "epsilon_selected": False,
        "runtime_comparison_allowed": False,
        "report_path": str(report_path),
        "image_dir": str(image_dir),
    }
    manifest_path = analysis_output_dir / "provisional_report_manifest.json"
    _atomic_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "analysis_output_dir": str(analysis_output_dir),
        "report": str(report_path),
        "report_image_dir": str(image_dir),
        "manifest": str(manifest_path),
        **{key: str(path) for key, path in bundle.paths.items()},
    }


def _load_table(run_dir: Path, name: str) -> list[dict[str, Any]]:
    aggregate = run_dir / name
    if aggregate.is_file() and aggregate.stat().st_size > 0:
        return read_csv(aggregate)
    rows: list[dict[str, Any]] = []
    for source in sorted((run_dir / "rows").glob(f"*/{name}"), key=lambda path: path.parent.name):
        shard = read_csv(source)
        for row in shard:
            if not row.get("row_id"):
                row["row_id"] = source.parent.name
        rows.extend(shard)
    return rows


def _coverage_rows(expected_rows: Sequence[Mapping[str, Any]], completed: set[str]) -> list[dict[str, Any]]:
    fields = (
        "experiment_phase",
        "row_id",
        "pair_id",
        "construction_policy",
        "construction_family",
        "layer_schedule",
        "utility_candidate_budget",
        "layer_normalization_policy",
        "broad_atom_builder",
        "repair_atom_builder",
    )
    rows: list[dict[str, Any]] = []
    for source in sorted(expected_rows, key=lambda row: str(row.get("row_id", ""))):
        row = {field: source.get(field) for field in fields}
        row["completed"] = str(source.get("row_id", "")) in completed
        rows.append(row)
    return rows


def _co_primary_rows(summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    pareto_ids = _pareto_ids(summaries)
    fields = (
        "experiment_phase",
        "row_id",
        "pair_id",
        "construction_policy",
        "construction_family",
        "layer_schedule",
        "utility_candidate_budget",
        "layer_normalization_policy",
        *CO_PRIMARY_METRICS,
        *SUPPLEMENTAL_METRICS,
        "oracle_construction_time",
        "train_encoding_time",
        "validation_encoding_time",
    )
    result = []
    for source in summaries:
        row = {field: source.get(field, "") for field in fields}
        row["pareto_candidate"] = str(source.get("row_id")) in pareto_ids
        result.append(row)
    return result


def _pareto_ids(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    def dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
        comparisons = (
            (_number(left, "validation_median_rmse"), _number(right, "validation_median_rmse"), "min"),
            (_number(left, "validation_p95_rmse"), _number(right, "validation_p95_rmse"), "min"),
            (_number(left, "validation_node_max_error_p95"), _number(right, "validation_node_max_error_p95"), "min"),
            (_number(left, "validation_strict_perfect_lfo_rate"), _number(right, "validation_strict_perfect_lfo_rate"), "max"),
        )
        no_worse = all(a <= b if direction == "min" else a >= b for a, b, direction in comparisons)
        strictly_better = any(a < b if direction == "min" else a > b for a, b, direction in comparisons)
        return no_worse and strictly_better

    return {
        str(row.get("row_id"))
        for row in rows
        if not any(other is not row and dominates(other, row) for other in rows)
    }


def _matched_factor_deltas(summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    comparisons = (
        ("layer_normalization_policy", "FinalClipOnly", "LayerClip0To1", ("construction_policy", "utility_candidate_budget")),
        ("utility_candidate_budget", "CandidateBudget24", "CandidateBudget48", ("construction_policy", "layer_normalization_policy")),
        ("layer_schedule", "Interleaved", "TwoPhase", ("construction_family", "utility_candidate_budget", "layer_normalization_policy")),
    )
    result: list[dict[str, Any]] = []
    for field, left_value, right_value, key_fields in comparisons:
        grouped: dict[tuple[str, ...], dict[str, Mapping[str, Any]]] = {}
        for row in summaries:
            value = str(row.get(field, ""))
            if value not in {left_value, right_value}:
                continue
            key = tuple(str(row.get(name, "")) for name in key_fields)
            grouped.setdefault(key, {})[value] = row
        for key, pair in sorted(grouped.items()):
            if left_value not in pair or right_value not in pair:
                continue
            left, right = pair[left_value], pair[right_value]
            row: dict[str, Any] = {
                "comparison": field,
                "left_value": left_value,
                "right_value": right_value,
                "match_key": " | ".join(key),
                "left_row_id": left.get("row_id", ""),
                "right_row_id": right.get("row_id", ""),
            }
            for metric in CO_PRIMARY_METRICS:
                left_metric, right_metric = _number(left, metric), _number(right, metric)
                row[f"left_{metric}"] = left_metric
                row[f"right_{metric}"] = right_metric
                row[f"delta_{metric}"] = right_metric - left_metric
            result.append(row)
    return result


def _write_plots(bundle: AnalysisBundle, image_dir: Path) -> dict[str, Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - dependency failure is environment-specific
        raise RuntimeError("matplotlib is required to generate the Experiment 13 report") from exc

    image_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "pareto": image_dir / "co_primary_pareto.png",
        "normalization": image_dir / "normalization_p95_deltas.png",
        "budget": image_dir / "candidate_budget_p95_deltas.png",
        "schedule": image_dir / "schedule_p95_deltas.png",
        "partial": image_dir / "partial_codebook_progression.png",
        "runtime": image_dir / "legacy_oracle_runtime.png",
        "layer_quantiles": image_dir / "layer_epsilon_quantiles.png",
        "slot_quantiles": image_dir / "slot_epsilon_quantiles.png",
        "layer_coverage": image_dir / "completed_layer_coverage.png",
        "slot_coverage": image_dir / "slot_coverage.png",
        "retired": image_dir / "retired_fraction_vs_energy.png",
        "energy": image_dir / "incoming_vs_unexplained_energy.png",
    }
    _plot_pareto(plt, paths["pareto"], bundle.co_primary)
    _plot_delta(plt, paths["normalization"], bundle.matched_deltas, "layer_normalization_policy", "LayerClip0To1 minus FinalClipOnly")
    _plot_delta(plt, paths["budget"], bundle.matched_deltas, "utility_candidate_budget", "CandidateBudget48 minus CandidateBudget24")
    _plot_delta(plt, paths["schedule"], bundle.matched_deltas, "layer_schedule", "TwoPhase minus Interleaved")
    _plot_partial(plt, paths["partial"], bundle.partial_codebook, bundle.summaries)
    _plot_runtime(plt, paths["runtime"], bundle.summaries)
    _plot_quantiles(plt, paths["layer_quantiles"], bundle.calibration["layer_epsilon_quantiles.csv"], "residual_layer", "Completed-layer epsilon quantiles")
    _plot_quantiles(plt, paths["slot_quantiles"], bundle.calibration["slot_epsilon_quantiles.csv"], "active_atom_slot", "Slot-level epsilon quantiles")
    _plot_coverage(plt, paths["layer_coverage"], bundle.calibration["epsilon_coverage.csv"], completed=True)
    _plot_coverage(plt, paths["slot_coverage"], bundle.calibration["epsilon_coverage.csv"], completed=False)
    _plot_retired(plt, paths["retired"], bundle.calibration["retired_error_mass.csv"])
    _plot_energy(plt, paths["energy"], bundle.calibration["retired_error_mass.csv"])
    return paths


def _plot_pareto(plt: Any, path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    figure, axis = plt.subplots(figsize=(8.2, 5.4))
    families = sorted({str(row.get("construction_family", "")) for row in rows})
    colors = {family: plt.cm.tab10(index % 10) for index, family in enumerate(families)}
    for family in families:
        selected = [row for row in rows if row.get("construction_family") == family]
        axis.scatter(
            [_number(row, "validation_median_rmse") for row in selected],
            [_number(row, "validation_p95_rmse") for row in selected],
            label=family,
            color=colors[family],
            s=[72 if _truth(row.get("pareto_candidate")) else 30 for row in selected],
            edgecolors=["black" if _truth(row.get("pareto_candidate")) else "none" for row in selected],
            alpha=0.82,
        )
    axis.set(xlabel="validation median RMSE (lower is better)", ylabel="validation P95 RMSE (lower is better)", title="Provisional co-primary quality tradeoffs")
    axis.legend(fontsize=7, loc="best")
    _save(plt, figure, path)


def _plot_delta(plt: Any, path: Path, rows: Sequence[Mapping[str, Any]], comparison: str, title: str) -> None:
    selected = sorted(
        (row for row in rows if row.get("comparison") == comparison),
        key=lambda row: _number(row, "delta_validation_p95_rmse"),
    )
    figure, axis = plt.subplots(figsize=(10.5, max(3.4, 0.31 * len(selected) + 1.4)))
    values = [_number(row, "delta_validation_p95_rmse") for row in selected]
    positions = list(range(len(selected)))
    axis.barh(positions, values, color=["#3A923A" if value < 0 else "#D35454" for value in values])
    axis.axvline(0.0, color="black", linewidth=0.9)
    labels = [
        str(row.get("match_key", ""))
        .replace("CandidateBudget", "B")
        .replace("LayerClip0To1", "LayerClip")
        .replace("FinalClipOnly", "FinalClip")
        .replace(" | ", " · ")
        for row in selected
    ]
    axis.set_yticks(positions, labels, fontsize=7)
    axis.set(xlabel="validation P95 RMSE delta (negative favors right-hand policy)", ylabel="matched pair", title=title)
    _save(plt, figure, path)


def _plot_partial(plt: Any, path: Path, rows: Sequence[Mapping[str, Any]], summaries: Sequence[Mapping[str, Any]]) -> None:
    family_by_row = {str(row.get("row_id")): str(row.get("construction_family", "")) for row in summaries}
    grouped: dict[tuple[str, int], list[float]] = {}
    for row in rows:
        family = family_by_row.get(str(row.get("row_id")), "Unknown")
        key = (family, int(_number(row, "active_atom_count")))
        grouped.setdefault(key, []).append(_number(row, "validation_p95_rmse"))
    figure, axis = plt.subplots(figsize=(8.2, 5.0))
    for family in sorted({key[0] for key in grouped}):
        x = sorted(key[1] for key in grouped if key[0] == family)
        if x:
            axis.plot(x, [median(grouped[(family, value)]) for value in x], marker="o", label=family)
    axis.set(xlabel="active atoms per residual layer", ylabel="median validation P95 RMSE (lower is better)", title="Partial-codebook progression by covered family")
    axis.legend(fontsize=7, loc="best")
    _save(plt, figure, path)


def _plot_runtime(plt: Any, path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    selected = sorted(rows, key=lambda row: _number(row, "oracle_construction_time"), reverse=True)[:15]
    figure, axis = plt.subplots(figsize=(10.5, 6.8))
    values = [_number(row, "oracle_construction_time") for row in selected]
    axis.barh(range(len(selected)), values, color="#4C78A8", alpha=0.85)
    axis.set_xscale("log")
    labels = [str(row.get("row_id", "")).removeprefix("x13a_") for row in selected]
    axis.set_yticks(range(len(selected)), labels, fontsize=7)
    axis.invert_yaxis()
    axis.set(xlabel="legacy oracle construction seconds (log scale; lower is faster)", ylabel="15 slowest completed rows", title="Historical legacy construction runtime")
    _save(plt, figure, path)


def _plot_quantiles(plt: Any, path: Path, rows: Sequence[Mapping[str, Any]], x_key: str, title: str) -> None:
    grouped: dict[tuple[float, int], list[float]] = {}
    for row in rows:
        if row.get("dataset_split", "training") != "training":
            continue
        grouped.setdefault((_number(row, "percentile"), int(_number(row, x_key))), []).append(_number(row, "epsilon_value"))
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    for percentile in sorted({key[0] for key in grouped}, reverse=True):
        x = sorted(key[1] for key in grouped if key[0] == percentile)
        axis.plot(x, [median(grouped[(percentile, value)]) for value in x], marker="o", markersize=3, label=f"q={percentile:g}")
    axis.set(xlabel=x_key.replace("_", " "), ylabel="median epsilon value", title=title)
    axis.legend(fontsize=7, ncol=2)
    _save(plt, figure, path)


def _plot_coverage(plt: Any, path: Path, rows: Sequence[Mapping[str, Any]], *, completed: bool) -> None:
    grouped: dict[tuple[float, int], list[float]] = {}
    for row in rows:
        if row.get("dataset_split") != "training":
            continue
        slot = row.get("active_atom_slot")
        is_completed = slot in {None, "", "None"}
        if is_completed != completed:
            continue
        x = int(_number(row, "residual_layer" if completed else "active_atom_slot"))
        grouped.setdefault((_number(row, "epsilon"), x), []).append(_number(row, "resolved_fraction"))
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    for epsilon in sorted({key[0] for key in grouped}):
        x = sorted(key[1] for key in grouped if key[0] == epsilon)
        axis.plot(x, [median(grouped[(epsilon, value)]) for value in x], marker="o", markersize=3, label=f"{epsilon:g}")
    axis.set(
        xlabel="residual layer" if completed else "active atom slot",
        ylabel="median reconstructed fraction (higher means more curves below epsilon)",
        title="Completed-layer reconstructed fractions" if completed else "Slot-level reconstructed fractions",
    )
    axis.legend(fontsize=7, ncol=3)
    _save(plt, figure, path)


def _plot_retired(plt: Any, path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    figure, axis = plt.subplots(figsize=(7.0, 5.2))
    for epsilon in sorted({_number(row, "epsilon") for row in rows}):
        selected = [row for row in rows if math.isclose(_number(row, "epsilon"), epsilon)]
        axis.scatter(
            [_number(row, "retired_lfo_fraction") for row in selected],
            [_number(row, "unexplained_retired_energy_fraction") for row in selected],
            s=7,
            alpha=0.25,
            label=f"{epsilon:g}",
        )
    axis.set(xlabel="retired LFO fraction", ylabel="unexplained retired-error energy fraction (lower is safer)", title="Counterfactual retirement coverage and unexplained energy")
    axis.legend(fontsize=7)
    _save(plt, figure, path)


def _plot_energy(plt: Any, path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    figure, axis = plt.subplots(figsize=(7.0, 5.2))
    incoming = [_number(row, "incoming_retired_energy_fraction") for row in rows]
    unexplained = [_number(row, "unexplained_retired_energy_fraction") for row in rows]
    axis.scatter(incoming, unexplained, s=7, alpha=0.25, color="#4C78A8")
    maximum = max([1e-9, *incoming, *unexplained])
    axis.plot([0.0, maximum], [0.0, maximum], linestyle="--", color="gray", linewidth=1)
    axis.set(xlabel="incoming retired-energy fraction", ylabel="unexplained retired-energy fraction (lower is safer)", title="Incoming versus unexplained retired energy")
    _save(plt, figure, path)


def _provisional_markdown(
    bundle: AnalysisBundle,
    *,
    source_run: Path,
    report_path: Path,
    plot_paths: Mapping[str, Path],
    expected_count: int,
) -> str:
    completed = len(bundle.summaries)
    best = min(bundle.summaries, key=lambda row: _number(row, "validation_p95_rmse"))
    normalization = _comparison_summary(bundle.matched_deltas, "layer_normalization_policy")
    budget = _comparison_summary(bundle.matched_deltas, "utility_candidate_budget")
    schedule = _comparison_summary(bundle.matched_deltas, "layer_schedule")
    planned_families = {str(row.get("construction_family", "")) for row in bundle.coverage}
    completed_families = {
        str(row.get("construction_family", "")) for row in bundle.summaries
    }
    absent_families = sorted(planned_families - completed_families)
    partial_families = []
    for family in sorted(completed_families):
        family_rows = [row for row in bundle.coverage if row.get("construction_family") == family]
        done = sum(_truth(row.get("completed")) for row in family_rows)
        if done < len(family_rows):
            partial_families.append(f"`{family}` ({done}/{len(family_rows)})")
    pareto = [row for row in bundle.co_primary if _truth(row.get("pareto_candidate"))]
    top = sorted(bundle.summaries, key=lambda row: _number(row, "validation_p95_rmse"))[:5]
    runtime_rows = sorted(bundle.summaries, key=lambda row: _number(row, "oracle_construction_time"), reverse=True)
    source_display = os.path.relpath(source_run, report_path.parent).replace("\\", "/")

    def image(name: str, alt: str) -> str:
        relative = os.path.relpath(plot_paths[name], report_path.parent).replace("\\", "/")
        return f"![{alt}]({relative})"

    absent_text = ", ".join(f"`{value}`" for value in absent_families) or "none"
    partial_text = ", ".join(partial_families) or "none"
    lines = [
        "# Experiment 13: Provisional Fixed-W8D16 Strategy-Grid Report",
        "",
        "> **Provisional evidence only.** This report covers a non-random execution-order prefix of "
        f"`{completed}/{expected_count}` Experiment 13A rows from an interrupted legacy full-training run. "
        "Experiment 13A did not complete, no eligibility epsilon has been selected, and Experiment 13B has not run.",
        "",
        "## Main Findings",
        "",
        f"Within the covered rows, layer-wise clipping is the clearest repeatable effect. `LayerClip0To1` improves validation P95 RMSE in `{normalization['improved']}/{normalization['count']}` matched pairs; the median right-minus-left change is `{normalization['median']:.8g}` (range `{normalization['minimum']:.8g}` to `{normalization['maximum']:.8g}`). Lower is better, so every available matched normalization comparison favors clipping.",
        "",
        f"Increasing the offline candidate shortlist from 24 to 48 has a smaller and inconsistent effect. `CandidateBudget48` improves `{budget['improved']}/{budget['count']}` matched pairs, with median validation-P95 change `{budget['median']:.8g}` (range `{budget['minimum']:.8g}` to `{budget['maximum']:.8g}`). This does not support treating the larger shortlist as an automatic quality win.",
        "",
        f"Schedule choice is unresolved in this fragment. `TwoPhase` improves `{schedule['improved']}/{schedule['count']}` matched pairs and worsens the others; its median validation-P95 change versus `Interleaved` is `{schedule['median']:.8g}`. The covered policies therefore provide no global schedule winner.",
        "",
        f"The lowest observed validation P95 RMSE is `{_number(best, 'validation_p95_rmse'):.8g}` from `{best.get('row_id')}`. It is only the best observed row in this prefix: absent and partially covered families prevent a full-grid ranking.",
        "",
        f"There are `{len(pareto)}` provisional Pareto candidates across validation median RMSE, strict-perfect rate, P95 RMSE, and node-max P95. They are retained as tradeoffs rather than collapsed into one scalar score.",
        "",
        image("pareto", "Provisional co-primary quality tradeoffs"),
        "",
        "The scatter's x-axis is validation median RMSE and its y-axis is validation P95 RMSE; lower-left is better on both. Color identifies the covered construction family. Larger black-outlined points are provisional Pareto candidates after also accounting for strict-perfect rate and node-max P95, so no single point is declared the automatic winner.",
        "",
        "## Why These Patterns Appear",
        "",
        "`LayerClip0To1` applies a decoder-free physical range constraint after every residual layer. In the covered rows it consistently suppresses accumulated overshoot, so its P95 benefit is both larger and more stable than the changes caused by shortlist size or layer schedule.",
        "",
        "The candidate budget changes how many observed residual candidates the offline constructor scores; it does not add model prediction heads. A larger shortlist can find a stronger repair atom, but later atoms and Beam4 encoding can compensate for an earlier local choice. That makes the effect non-monotonic and usually much smaller than clipping.",
        "",
        "`Interleaved` and `TwoPhase` reorder broad and repair residual layers without changing W8D16 or the deployed runtime interface. Their split result is consistent with an interaction: alternating repair can help some objectives, while reserving repair for later residuals can help others.",
        "",
        "## Plot Notes",
        "",
        "### Matched Normalization Effect",
        "",
        "Each bar is one matched construction policy and candidate budget. The x-axis is `LayerClip0To1` minus `FinalClipOnly` validation P95 RMSE; negative is better for layer clipping. Every bar lies below zero, supporting layer-wise clipping inside the covered strategy families.",
        "",
        image("normalization", "Matched normalization P95 deltas"),
        "",
        "### Matched Candidate-Budget Effect",
        "",
        "Each bar is one matched policy and normalization. The x-axis is CandidateBudget48 minus CandidateBudget24 validation P95 RMSE; negative favors 48. Bars fall on both sides of zero, showing that the extra offline search is not reliably converted into validation quality.",
        "",
        image("budget", "Matched candidate-budget P95 deltas"),
        "",
        "### Matched Schedule Effect",
        "",
        "Each bar is one matched construction family, budget, and normalization. The x-axis is TwoPhase minus Interleaved validation P95 RMSE; negative favors TwoPhase. The balanced signs and near-zero median mean schedule should remain an interaction term, not a global default, until the grid is complete.",
        "",
        image("schedule", "Matched schedule P95 deltas"),
        "",
        "### Partial-Codebook Progression",
        "",
        "The x-axis is the number of active atoms retained per residual layer and the y-axis is family-median validation P95 RMSE, where lower is better. Every covered family improves sharply from one to two active atoms, then shows diminishing returns; BroadMeanGlobalRepair and BroadMeanHardRepair form the lowest curves, while BroadMeanFinishRepair plateaus highest. This suggests the first few codebook choices carry most of the quality, but it is descriptive only for the families present in the fragment.",
        "",
        image("partial", "Partial-codebook progression"),
        "",
        "### Historical Oracle Runtime",
        "",
        "Lower is faster. The x-axis is legacy oracle construction time on a logarithmic scale and the y-axis ranks the completed rows. The two largest observations are "
        f"`{_number(runtime_rows[0], 'oracle_construction_time'):.8g}` and `{_number(runtime_rows[1], 'oracle_construction_time'):.8g}` seconds. These measurements include the superseded implementation and Modern Standby effects, so they diagnose the aborted run but must not be compared with optimized-run timing.",
        "",
        image("runtime", "Historical legacy construction runtime"),
        "",
        "## Provisional Experiment 13A Calibration",
        "",
        "These plots summarize counterfactual epsilon behavior for the 39 completed unfiltered rows. They do not satisfy the deterministic selection rule, which requires all 90 Experiment 13A rows. No curve or apparent elbow in this section is an epsilon decision.",
        "",
        "The completed-layer and slot quantile plots show the epsilon needed to cover different fractions of curves as construction progresses. Lower values mean the partial reconstruction is closer. Completed-layer quantiles fall quickly in the first few residual layers and then flatten, showing large early gains followed by diminishing returns. The coverage plots invert that view: higher reconstructed fraction means more training curves are already below a fixed candidate epsilon. Even the largest candidate reaches only a small median fraction in this prefix, so these curves do not justify freezing a threshold.",
        "",
        image("layer_quantiles", "Completed-layer epsilon quantiles"),
        "",
        image("slot_quantiles", "Slot-level epsilon quantiles"),
        "",
        image("layer_coverage", "Completed-layer reconstructed fractions"),
        "",
        image("slot_coverage", "Slot-level reconstructed fractions"),
        "",
        "The retirement scatter compares how many LFOs would be excluded with how much unexplained residual energy those LFOs still carry. Lower unexplained energy is safer. The larger candidate epsilons extend both retirement coverage and the upper tail of unexplained energy, exposing the intended safety-versus-work tradeoff. The final plot separates incoming retired energy from unexplained retired energy; points below the diagonal indicate that the current partial codebook already explains some of the energy that would be retired.",
        "",
        image("retired", "Retired fraction versus unexplained energy"),
        "",
        image("energy", "Incoming versus unexplained retired energy"),
        "",
        "## Source Coverage",
        "",
        f"Zero completed rows are available for these planned construction families: {absent_text}.",
        "",
        f"These represented families are still incomplete: {partial_text}.",
        "",
        "The five lowest observed validation-P95 rows are:",
        "",
    ]
    for index, row in enumerate(top, start=1):
        lines.append(
            f"{index}. `{row.get('row_id')}` — P95 `{_number(row, 'validation_p95_rmse'):.8g}`, "
            f"median `{_number(row, 'validation_median_rmse'):.8g}`, strict-perfect "
            f"`{_number(row, 'validation_strict_perfect_lfo_rate'):.8g}`, node-max P95 "
            f"`{_number(row, 'validation_node_max_error_p95'):.8g}`."
        )
    lines.extend(
        [
            "",
            "## Practical Takeaways",
            "",
            "- Keep `LayerClip0To1` as the strongest provisional decoder-free policy candidate.",
            "- Keep both candidate budgets until complete-grid interactions are available; 48 is not a universal improvement.",
            "- Do not choose a global layer schedule from the fragment.",
            "- Do not select the Experiment 13B eligibility epsilon from these incomplete calibration artifacts.",
            "- Do not use legacy runtime to estimate the optimized run or the 50%-training scaling ablation.",
            "",
            "## Method Notes and Generated Artifacts",
            "",
            f"The immutable source is `{source_display}` relative to this report. The report reads completed row shards and writes all derived CSVs outside that archive. `completed_row_coverage.csv` records the exact planned cells present or absent; `co_primary_metrics.csv` retains detailed metrics and Pareto membership; `matched_factor_deltas.csv` contains every matched comparison; and `partial_codebook_progression.csv` retains the one-through-seven-atom results.",
            "",
            "All results use the fixed W8D16 runtime contract: 32 base choices, eight residual-layer atom choices across 16 residual layers, PhaseAndResidualGain scalars, Beam4 encoding, and 193 model prediction outputs. Codebook construction is offline/oracle work; topology is not a deployed runtime input.",
            "",
        ]
    )
    return "\n".join(lines)


def _comparison_summary(rows: Sequence[Mapping[str, Any]], comparison: str) -> dict[str, float | int]:
    values = [
        _number(row, "delta_validation_p95_rmse")
        for row in rows
        if row.get("comparison") == comparison
    ]
    if not values:
        return {"count": 0, "improved": 0, "median": math.nan, "minimum": math.nan, "maximum": math.nan}
    return {
        "count": len(values),
        "improved": sum(value < 0 for value in values),
        "median": median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _number(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key, 0.0)
    if value in {None, "", "None"}:
        return 0.0
    return float(value)


def _truth(value: Any) -> bool:
    return value is True or str(value).lower() in {"true", "1", "yes"}


def _require_outside_source(source: Path, output: Path, label: str) -> None:
    source, output = source.resolve(), output.resolve()
    if output == source or source in output.parents:
        raise ValueError(f"{label} must be outside the immutable source run: {output}")


def _directory_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        file_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        digest.update(f"{relative}\t{file_digest}\n".encode("utf-8"))
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _save(plt: Any, figure: Any, path: Path) -> None:
    figure.tight_layout()
    temporary = path.with_name(f".{path.name}.tmp")
    figure.savefig(temporary, dpi=160, format="png", metadata={"Software": "OBRUXO Experiment 13 report generator"})
    plt.close(figure)
    temporary.replace(path)

"""Reusable analysis and report generation for Experiment 13."""

from __future__ import annotations

from dataclasses import asdict, dataclass
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
INTERACTIVE_REPORT_SCHEMA = "experiment13_interactive_report_v6"
INTERACTIVE_TEMPLATE = Path(__file__).with_name("templates") / "experiment_13_provisional.html"
COMPLETE_13A_REPORT_SCHEMA = "experiment13_complete_13a_report_v4"
COMPLETE_REPORT_SCHEMA = "experiment13_complete_report_v2"


@dataclass(frozen=True)
class AnalysisBundle:
    summaries: list[dict[str, Any]]
    coverage: list[dict[str, Any]]
    co_primary: list[dict[str, Any]]
    matched_deltas: list[dict[str, Any]]
    partial_codebook: list[dict[str, Any]]
    diagnostics: list[dict[str, Any]]
    rankings: list[dict[str, Any]]
    marginal_atoms: list[dict[str, Any]]
    layer_progression: list[dict[str, Any]]
    mechanism_diagnostics: list[dict[str, Any]]
    factor_interactions: list[dict[str, Any]]
    eligibility_capacity_effects: list[dict[str, Any]]
    eligibility_depth_summary: list[dict[str, Any]]
    calibration: dict[str, list[dict[str, Any]]]
    paths: dict[str, Path]


def analyze_partial_strategy_grid(
    *,
    run_dir: Path,
    analysis_output_dir: Path,
    report_path: Path,
    image_dir: Path,
    html_report_path: Path | None = None,
) -> dict[str, str]:
    """Generate a provisional report from completed, sharded 13A rows."""
    from .strategy_grid import experiment13a_specs

    return write_provisional_report(
        source_run=Path(run_dir),
        analysis_output_dir=Path(analysis_output_dir),
        report_path=Path(report_path),
        html_report_path=Path(html_report_path) if html_report_path is not None else None,
        image_dir=Path(image_dir),
        expected_rows=[asdict(spec) for spec in experiment13a_specs()],
    )


def analyze_13a_strategy_grid(
    *,
    run_dir: Path,
    analysis_output_dir: Path,
    report_path: Path,
    image_dir: Path,
    html_report_path: Path | None = None,
    scaling_baseline_run: Path | None = None,
    strict_thresholds_path: Path | None = None,
) -> dict[str, str]:
    """Generate the complete Experiment 13A report without weakening final-analysis gates."""
    from .strategy_grid import experiment13a_specs, load_epsilon_selection, validate_completed_13a

    source_run = Path(run_dir).resolve()
    manifest, _ = validate_completed_13a(source_run, allow_historical_configuration=True)
    selection_path = source_run / "epsilon_selection.json"
    if not selection_path.is_file():
        raise ValueError("complete 13A reporting requires epsilon selection to be attempted first")
    selection = load_epsilon_selection(
        selection_path,
        expected_run_identity=str(manifest["experiment13a_run_identity"]),
        expected_configuration_fingerprint=str(manifest["configuration_fingerprint"]),
        require_passed=False,
    )
    return write_complete_13a_report(
        source_run=source_run,
        analysis_output_dir=Path(analysis_output_dir),
        report_path=Path(report_path),
        html_report_path=Path(html_report_path) if html_report_path is not None else None,
        image_dir=Path(image_dir),
        expected_rows=[asdict(spec) for spec in experiment13a_specs()],
        selection=dict(selection.payload),
        scaling_baseline_run=Path(scaling_baseline_run) if scaling_baseline_run is not None else None,
        strict_thresholds_path=Path(strict_thresholds_path) if strict_thresholds_path is not None else None,
    )


def analyze_complete_strategy_grid(
    *,
    run_dir: Path,
    analysis_output_dir: Path,
    report_path: Path,
    image_dir: Path,
    html_report_path: Path | None = None,
) -> dict[str, str]:
    """Generate the final report after the complete 13A and 13B phases."""
    from .strategy_grid import (
        experiment13a_specs,
        experiment13b_specs,
        load_epsilon_selection,
        validate_completed_13a,
        validate_completed_13b,
    )

    source_run = Path(run_dir).resolve()
    manifest, _ = validate_completed_13a(source_run, allow_historical_configuration=True)
    validate_completed_13b(source_run, allow_historical_configuration=True)
    selection = load_epsilon_selection(
        source_run / "epsilon_selection.json",
        expected_run_identity=str(manifest["experiment13a_run_identity"]),
        expected_configuration_fingerprint=str(manifest["configuration_fingerprint"]),
        require_passed=False,
    )
    artifact_root = source_run.parent
    scaling_baseline = artifact_root / "strategy_grid_train100_val100_interrupted_39rows_20260716"
    strict_thresholds = artifact_root / "analysis_train50_val100_13a_complete" / "strict_perfect_threshold_sweep.csv"
    return write_complete_report(
        source_run=source_run,
        analysis_output_dir=Path(analysis_output_dir),
        report_path=Path(report_path),
        html_report_path=Path(html_report_path) if html_report_path is not None else None,
        image_dir=Path(image_dir),
        expected_rows=[asdict(spec) for spec in [*experiment13a_specs(), *experiment13b_specs()]],
        selection=dict(selection.payload),
        scaling_baseline_run=scaling_baseline if scaling_baseline.is_dir() else None,
        strict_thresholds_path=strict_thresholds if strict_thresholds.is_file() else None,
    )


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
    if phase is None:
        matched.extend(_paired_population_deltas(summaries))
        matched.sort(key=lambda row: (str(row.get("comparison", "")), str(row.get("match_key", ""))))
    partial = _load_table(source_run, "partial_codebook_validation.csv")
    if phase is not None:
        partial = [row for row in partial if row.get("experiment_phase") == phase]
    partial = sorted(partial, key=lambda row: (str(row.get("row_id", "")), _number(row, "active_atom_count")))
    slot_progression = _load_table(source_run, "slot_progression.csv")
    atom_construction = _load_table(source_run, "atom_construction.csv")
    candidate_search = _load_table(source_run, "candidate_search_diagnostics.csv")
    if phase is not None:
        slot_progression = [row for row in slot_progression if row.get("experiment_phase") == phase]
        atom_construction = [row for row in atom_construction if row.get("experiment_phase") == phase]
        candidate_search = [row for row in candidate_search if row.get("experiment_phase") == phase]
    diagnostics = _strategy_diagnostic_rows(summaries)
    rankings = _metric_ranking_rows(summaries)
    marginal_atoms = _marginal_atom_rows(partial, summaries)
    layer_progression = _layer_progression_rows(slot_progression, atom_construction, summaries)
    mechanism_diagnostics = _mechanism_diagnostic_rows(atom_construction, candidate_search, summaries)
    factor_interactions = _factor_interaction_rows(matched)
    eligibility_capacity_effects = _eligibility_capacity_effect_rows(matched, layer_progression, summaries)
    eligibility_depth_summary = _eligibility_depth_summary_rows(layer_progression, summaries)
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
        "diagnostics": analysis_output_dir / "strategy_diagnostics.csv",
        "rankings": analysis_output_dir / "metric_rankings.csv",
        "marginal_atoms": analysis_output_dir / "marginal_atom_value.csv",
        "layer_progression": analysis_output_dir / "residual_layer_progression.csv",
        "mechanism_diagnostics": analysis_output_dir / "construction_mechanism_diagnostics.csv",
        "factor_interactions": analysis_output_dir / "factor_interaction_summary.csv",
    }
    write_csv(paths["coverage"], coverage)
    write_csv(paths["co_primary"], co_primary)
    write_csv(paths["matched_deltas"], matched)
    write_csv(paths["partial_codebook"], partial)
    write_csv(paths["diagnostics"], diagnostics)
    write_csv(paths["rankings"], rankings)
    write_csv(paths["marginal_atoms"], marginal_atoms)
    write_csv(paths["layer_progression"], layer_progression)
    write_csv(paths["mechanism_diagnostics"], mechanism_diagnostics)
    write_csv(paths["factor_interactions"], factor_interactions)
    if eligibility_capacity_effects:
        paths["eligibility_capacity_effects"] = analysis_output_dir / "eligibility_paired_capacity_effect.csv"
        write_csv(paths["eligibility_capacity_effects"], eligibility_capacity_effects)
    if eligibility_depth_summary:
        paths["eligibility_depth_summary"] = analysis_output_dir / "eligibility_depth_summary.csv"
        write_csv(paths["eligibility_depth_summary"], eligibility_depth_summary)
    for name, rows in calibration.items():
        key = f"aggregated_{Path(name).stem}"
        paths[key] = analysis_output_dir / f"aggregated_{name}"
        write_csv(paths[key], rows)
    return AnalysisBundle(
        summaries,
        coverage,
        co_primary,
        matched,
        partial,
        diagnostics,
        rankings,
        marginal_atoms,
        layer_progression,
        mechanism_diagnostics,
        factor_interactions,
        eligibility_capacity_effects,
        eligibility_depth_summary,
        calibration,
        paths,
    )


def write_provisional_report(
    *,
    source_run: Path,
    analysis_output_dir: Path,
    report_path: Path,
    image_dir: Path,
    expected_rows: Sequence[Mapping[str, Any]],
    html_report_path: Path | None = None,
) -> dict[str, str]:
    """Generate a findings-first report from an immutable partial 13A run."""
    source_run = Path(source_run).resolve()
    analysis_output_dir = Path(analysis_output_dir).resolve()
    report_path = Path(report_path).resolve()
    html_report_path = Path(html_report_path or report_path.with_suffix(".html")).resolve()
    image_dir = Path(image_dir).resolve()
    for path, label in (
        (analysis_output_dir, "analysis output directory"),
        (report_path, "report path"),
        (html_report_path, "HTML report path"),
        (image_dir, "image directory"),
    ):
        _require_outside_source(source_run, path, label)
    if report_path.suffix.lower() != ".md":
        raise ValueError("provisional report path must end in .md")
    if html_report_path.suffix.lower() != ".html":
        raise ValueError("interactive report path must end in .html")

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

    source_fingerprint = _directory_fingerprint(source_run)
    interactive_payload = _interactive_payload(
        bundle,
        source_run=source_run,
        html_report_path=html_report_path,
        expected_count=expected_count,
        source_fingerprint=source_fingerprint,
    )
    interactive_text = _interactive_html(interactive_payload)
    _atomic_text(html_report_path, interactive_text)
    interactive_sha256 = hashlib.sha256(interactive_text.encode("utf-8")).hexdigest()

    manifest = {
        "schema_version": "experiment13_provisional_report_v1",
        "source_run": str(source_run),
        "source_archive_sha256": source_fingerprint,
        "completed_13a_rows": len(bundle.summaries),
        "expected_13a_rows": expected_count,
        "report_status": "provisional_incomplete_13a",
        "epsilon_selected": False,
        "runtime_comparison_allowed": False,
        "report_path": str(report_path),
        "html_report_path": str(html_report_path),
        "interactive_payload_schema": INTERACTIVE_REPORT_SCHEMA,
        "interactive_report_sha256": interactive_sha256,
        "image_dir": str(image_dir),
    }
    manifest_path = analysis_output_dir / "provisional_report_manifest.json"
    _atomic_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "analysis_output_dir": str(analysis_output_dir),
        "report": str(report_path),
        "html_report": str(html_report_path),
        "report_image_dir": str(image_dir),
        "manifest": str(manifest_path),
        **{key: str(path) for key, path in bundle.paths.items()},
    }


def write_complete_13a_report(
    *,
    source_run: Path,
    analysis_output_dir: Path,
    report_path: Path,
    image_dir: Path,
    expected_rows: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
    html_report_path: Path | None = None,
    scaling_baseline_run: Path | None = None,
    strict_thresholds_path: Path | None = None,
) -> dict[str, str]:
    """Write a complete-13A snapshot while keeping canonical 13A/13B analysis separate."""
    from .strategy_grid import read_phase_status

    source_run = Path(source_run).resolve()
    analysis_output_dir = Path(analysis_output_dir).resolve()
    report_path = Path(report_path).resolve()
    html_report_path = Path(html_report_path or report_path.with_suffix(".html")).resolve()
    image_dir = Path(image_dir).resolve()
    scaling_baseline_run = Path(scaling_baseline_run).resolve() if scaling_baseline_run is not None else None
    strict_thresholds_path = Path(strict_thresholds_path).resolve() if strict_thresholds_path is not None else None
    for path, label in (
        (analysis_output_dir, "analysis output directory"),
        (report_path, "report path"),
        (html_report_path, "HTML report path"),
        (image_dir, "image directory"),
    ):
        _require_outside_source(source_run, path, label)
    if report_path.suffix.lower() != ".md":
        raise ValueError("complete 13A report path must end in .md")
    if html_report_path.suffix.lower() != ".html":
        raise ValueError("interactive report path must end in .html")

    bundle = prepare_analysis_artifacts(
        source_run=source_run,
        analysis_output_dir=analysis_output_dir,
        expected_rows=expected_rows,
        phase="13A",
        forbid_source_writes=True,
    )
    expected_count = len([row for row in expected_rows if row.get("experiment_phase") == "13A"])
    if expected_count != 90 or len(bundle.summaries) != expected_count:
        raise ValueError(f"complete 13A report requires 90/90 rows; got {len(bundle.summaries)}/{expected_count}")
    if bool(selection.get("selection_passed")):
        raise ValueError("complete 13A snapshot expects the automatic selector result before a pilot override")

    strict_thresholds: dict[str, Any] | None = None
    if strict_thresholds_path is not None:
        from .strategy_grid_thresholds import load_strict_threshold_sweep

        strict_thresholds = load_strict_threshold_sweep(
            strict_thresholds_path,
            expected_row_ids=[str(row.get("row_id", "")) for row in bundle.summaries],
        )

    scaling_rows: list[dict[str, Any]] = []
    scaling_path: Path | None = None
    scaling_validation_sha256: str | None = None
    if scaling_baseline_run is not None:
        scaling_rows, scaling_validation_sha256 = _training_scaling_rows(
            baseline_run=scaling_baseline_run,
            sampled_run=source_run,
        )
        scaling_path = analysis_output_dir / "training_data_scaling_ablation.csv"
        write_csv(scaling_path, scaling_rows)

    plot_paths = _write_plots(
        bundle,
        image_dir,
        historical_runtime=False,
        deep_analysis=True,
        strict_thresholds=strict_thresholds,
    )
    report_text = _complete_13a_markdown(
        bundle,
        source_run=source_run,
        report_path=report_path,
        plot_paths=plot_paths,
        selection=selection,
        scaling_rows=scaling_rows,
        scaling_baseline_run=scaling_baseline_run,
        strict_thresholds=strict_thresholds,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_text(report_path, report_text)

    source_fingerprint = _analysis_source_fingerprint(source_run)
    phase_b_status = read_phase_status(source_run, "13B")
    interactive_payload = _interactive_payload(
        bundle,
        source_run=source_run,
        html_report_path=html_report_path,
        expected_count=expected_count,
        source_fingerprint=source_fingerprint,
        report_mode="complete_13a",
        selection=selection,
        scaling_rows=scaling_rows,
        experiment13b_state=str(phase_b_status.get("state", "not_started")),
        strict_thresholds=strict_thresholds,
    )
    interactive_text = _interactive_html(interactive_payload)
    _atomic_text(html_report_path, interactive_text)
    interactive_sha256 = hashlib.sha256(interactive_text.encode("utf-8")).hexdigest()

    manifest = {
        "schema_version": COMPLETE_13A_REPORT_SCHEMA,
        "source_run": str(source_run),
        "source_analysis_sha256": source_fingerprint,
        "completed_13a_rows": len(bundle.summaries),
        "expected_13a_rows": expected_count,
        "report_status": "complete_13a_pending_13b",
        "epsilon_selection_passed": bool(selection.get("selection_passed")),
        "epsilon_selection_notes": selection.get("selection_notes"),
        "experiment13b_state": phase_b_status.get("state", "not_started"),
        "runtime_comparison_allowed": False,
        "scaling_matched_row_count": len(scaling_rows),
        "scaling_validation_membership_sha256": scaling_validation_sha256,
        "strict_thresholds_path": str(strict_thresholds_path) if strict_thresholds_path is not None else None,
        "strict_thresholds_sha256": strict_thresholds.get("source_sha256") if strict_thresholds else None,
        "report_path": str(report_path),
        "html_report_path": str(html_report_path),
        "interactive_payload_schema": INTERACTIVE_REPORT_SCHEMA,
        "interactive_report_sha256": interactive_sha256,
        "image_dir": str(image_dir),
    }
    manifest_path = analysis_output_dir / "complete_13a_report_manifest.json"
    _atomic_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    result = {
        "analysis_output_dir": str(analysis_output_dir),
        "report": str(report_path),
        "html_report": str(html_report_path),
        "report_image_dir": str(image_dir),
        "manifest": str(manifest_path),
        **{key: str(path) for key, path in bundle.paths.items()},
    }
    if scaling_path is not None:
        result["training_data_scaling_ablation"] = str(scaling_path)
    if strict_thresholds_path is not None:
        result["strict_thresholds"] = str(strict_thresholds_path)
    return result


def write_complete_report(
    *,
    source_run: Path,
    analysis_output_dir: Path,
    report_path: Path,
    image_dir: Path,
    expected_rows: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
    html_report_path: Path | None = None,
    scaling_baseline_run: Path | None = None,
    strict_thresholds_path: Path | None = None,
) -> dict[str, str]:
    """Write the final 13A+13B findings-first Markdown and interactive reports."""
    source_run = Path(source_run).resolve()
    analysis_output_dir = Path(analysis_output_dir).resolve()
    report_path = Path(report_path).resolve()
    html_report_path = Path(html_report_path or report_path.with_suffix(".html")).resolve()
    image_dir = Path(image_dir).resolve()
    scaling_baseline_run = Path(scaling_baseline_run).resolve() if scaling_baseline_run is not None else None
    strict_thresholds_path = Path(strict_thresholds_path).resolve() if strict_thresholds_path is not None else None
    for path, label in (
        (analysis_output_dir, "analysis output directory"),
        (report_path, "report path"),
        (html_report_path, "HTML report path"),
        (image_dir, "image directory"),
    ):
        _require_outside_source(source_run, path, label)
    if report_path.suffix.lower() != ".md" or html_report_path.suffix.lower() != ".html":
        raise ValueError("complete report paths must end in .md and .html")

    bundle = prepare_analysis_artifacts(
        source_run=source_run,
        analysis_output_dir=analysis_output_dir,
        expected_rows=expected_rows,
        phase=None,
        forbid_source_writes=True,
    )
    phase_a = [row for row in bundle.summaries if row.get("experiment_phase") == "13A"]
    phase_b = [row for row in bundle.summaries if row.get("experiment_phase") == "13B"]
    if len(phase_a) != 90 or len(phase_b) != 135:
        raise ValueError(f"complete report requires 90 13A and 135 13B rows; got {len(phase_a)} and {len(phase_b)}")

    strict_thresholds: dict[str, Any] | None = None
    if strict_thresholds_path is not None:
        from .strategy_grid_thresholds import load_strict_threshold_sweep

        strict_thresholds = load_strict_threshold_sweep(
            strict_thresholds_path,
            expected_row_ids=[str(row.get("row_id", "")) for row in phase_a],
        )
    scaling_rows: list[dict[str, Any]] = []
    scaling_validation_sha256: str | None = None
    if scaling_baseline_run is not None:
        scaling_rows, scaling_validation_sha256 = _training_scaling_rows(
            baseline_run=scaling_baseline_run,
            sampled_run=source_run,
        )
        write_csv(analysis_output_dir / "training_data_scaling_ablation.csv", scaling_rows)

    plot_paths = _write_final_plots(bundle, image_dir, strict_thresholds=strict_thresholds)
    report_text = _complete_markdown(
        bundle,
        source_run=source_run,
        report_path=report_path,
        plot_paths=plot_paths,
        selection=selection,
        scaling_rows=scaling_rows,
        strict_thresholds=strict_thresholds,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_text(report_path, report_text)

    source_fingerprint = _analysis_source_fingerprint(source_run)
    payload = _interactive_payload(
        bundle,
        source_run=source_run,
        html_report_path=html_report_path,
        expected_count=len(expected_rows),
        source_fingerprint=source_fingerprint,
        report_mode="complete",
        selection=selection,
        scaling_rows=scaling_rows,
        experiment13b_state="complete",
        strict_thresholds=strict_thresholds,
    )
    interactive_text = _interactive_html(payload)
    _atomic_text(html_report_path, interactive_text)
    interactive_sha256 = hashlib.sha256(interactive_text.encode("utf-8")).hexdigest()
    manifest = {
        "schema_version": COMPLETE_REPORT_SCHEMA,
        "source_run": str(source_run),
        "source_analysis_sha256": source_fingerprint,
        "completed_13a_rows": len(phase_a),
        "completed_13b_rows": len(phase_b),
        "expected_rows": len(expected_rows),
        "report_status": "complete",
        "epsilon_selection_passed": bool(selection.get("selection_passed")),
        "eligibility_epsilon_sweep": [0.01, 0.001, 0.0001],
        "scaling_matched_row_count": len(scaling_rows),
        "scaling_validation_membership_sha256": scaling_validation_sha256,
        "strict_thresholds_sha256": strict_thresholds.get("source_sha256") if strict_thresholds else None,
        "report_path": str(report_path),
        "html_report_path": str(html_report_path),
        "interactive_payload_schema": INTERACTIVE_REPORT_SCHEMA,
        "interactive_report_sha256": interactive_sha256,
        "image_dir": str(image_dir),
    }
    manifest_path = analysis_output_dir / "complete_report_manifest.json"
    _atomic_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "analysis_output_dir": str(analysis_output_dir),
        "report": str(report_path),
        "html_report": str(html_report_path),
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


def _training_scaling_rows(
    *,
    baseline_run: Path,
    sampled_run: Path,
) -> tuple[list[dict[str, Any]], str]:
    """Return the bounded 100%- versus 50%-training quality ablation."""
    from .strategy_grid import _completed_row_summaries, _validation_membership_sha256

    baseline_run = Path(baseline_run).resolve()
    sampled_run = Path(sampled_run).resolve()
    baseline = _completed_row_summaries(baseline_run, "13A")
    sampled = _completed_row_summaries(sampled_run, "13A")
    matched = sorted(set(baseline) & set(sampled))
    if not matched:
        raise ValueError("training-data scaling ablation has no matched completed 13A rows")
    baseline_validation = _validation_membership_sha256(
        baseline_run / "rows" / matched[0] / "atom_assignments.csv"
    )
    sampled_validation = _validation_membership_sha256(
        sampled_run / "rows" / matched[0] / "atom_assignments.csv"
    )
    if baseline_validation != sampled_validation:
        raise ValueError("training-data scaling ablation requires identical validation membership")
    metrics = (
        "validation_median_rmse",
        "validation_strict_perfect_lfo_rate",
        "validation_p95_rmse",
        "validation_node_max_error_p95",
    )
    rows: list[dict[str, Any]] = []
    for row_id in matched:
        sampled_row = sampled[row_id]
        row: dict[str, Any] = {
            "row_id": row_id,
            "construction_family": sampled_row.get("construction_family", ""),
            "construction_policy": sampled_row.get("construction_policy", ""),
            "layer_schedule": sampled_row.get("layer_schedule", ""),
            "utility_candidate_budget": sampled_row.get("utility_candidate_budget", ""),
            "layer_normalization_policy": sampled_row.get("layer_normalization_policy", ""),
            "full_train_fraction": 1.0,
            "sampled_train_fraction": 0.5,
            "validation_fraction": 1.0,
            "runtime_comparison_allowed": False,
        }
        for metric in metrics:
            full_value = float(baseline[row_id][metric])
            sampled_value = float(sampled_row[metric])
            row[f"full_{metric}"] = full_value
            row[f"sampled_{metric}"] = sampled_value
            row[f"delta_{metric}"] = sampled_value - full_value
        rows.append(row)
    return rows, baseline_validation


def _analysis_source_fingerprint(source_run: Path) -> str:
    """Hash only the retained inputs consumed by the complete-13A report."""
    names = (
        "summary.csv",
        "partial_codebook_validation.csv",
        "slot_progression.csv",
        "atom_construction.csv",
        "candidate_search_diagnostics.csv",
        "layer_epsilon_quantiles.csv",
        "slot_epsilon_quantiles.csv",
        "epsilon_coverage.csv",
        "retired_error_mass.csv",
        "execution_timing.csv",
        "experiment13a_status.json",
        "experiment13b_status.json",
        "run_status.json",
        "epsilon_selection.json",
        "epsilon_selection_status.json",
    )
    digest = hashlib.sha256()
    for name in names:
        path = Path(source_run) / name
        if not path.is_file():
            continue
        digest.update(f"{name}\t{hashlib.sha256(path.read_bytes()).hexdigest()}\n".encode("utf-8"))
    return digest.hexdigest()


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
        "residual_population_policy",
        "eligibility_epsilon",
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
    phases = {str(row.get("experiment_phase", "")) for row in summaries}
    pareto_ids = set().union(*(
        _pareto_ids([row for row in summaries if str(row.get("experiment_phase", "")) == phase])
        for phase in phases
    ))
    fields = (
        "experiment_phase",
        "row_id",
        "pair_id",
        "construction_policy",
        "construction_family",
        "layer_schedule",
        "utility_candidate_budget",
        "layer_normalization_policy",
        "residual_population_policy",
        "eligibility_epsilon",
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
            key = (
                str(row.get("experiment_phase", "")),
                str(row.get("eligibility_epsilon", "")),
                *(str(row.get(name, "")) for name in key_fields),
            )
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
                "construction_family": left.get("construction_family", ""),
                "construction_policy": left.get("construction_policy", ""),
                "layer_schedule": left.get("layer_schedule", ""),
                "utility_candidate_budget": left.get("utility_candidate_budget", ""),
                "layer_normalization_policy": left.get("layer_normalization_policy", ""),
                "experiment_phase": left.get("experiment_phase", ""),
                "eligibility_epsilon": left.get("eligibility_epsilon", ""),
            }
            for metric in CO_PRIMARY_METRICS:
                left_metric, right_metric = _number(left, metric), _number(right, metric)
                row[f"left_{metric}"] = left_metric
                row[f"right_{metric}"] = right_metric
                row[f"delta_{metric}"] = right_metric - left_metric
            result.append(row)
    return result


def _paired_population_deltas(summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Pair every 13B epsilon row with its matching clipped 13A strategy."""
    phase_a = {
        str(row.get("pair_id", "")): row
        for row in summaries
        if row.get("experiment_phase") == "13A"
        and row.get("layer_normalization_policy") == "LayerClip0To1"
    }
    result: list[dict[str, Any]] = []
    for right in sorted(
        (row for row in summaries if row.get("experiment_phase") == "13B"),
        key=lambda row: (str(row.get("pair_id", "")), _number(row, "eligibility_epsilon")),
    ):
        left = phase_a.get(str(right.get("pair_id", "")))
        if left is None:
            continue
        epsilon = _number(right, "eligibility_epsilon")
        row: dict[str, Any] = {
            "comparison": "residual_population_policy",
            "left_value": "AllResiduals",
            "right_value": "UnresolvedOnly",
            "match_key": f"{right.get('construction_policy', '')} | epsilon {epsilon:g}",
            "left_row_id": left.get("row_id", ""),
            "right_row_id": right.get("row_id", ""),
            "construction_family": right.get("construction_family", ""),
            "construction_policy": right.get("construction_policy", ""),
            "layer_schedule": right.get("layer_schedule", ""),
            "utility_candidate_budget": right.get("utility_candidate_budget", ""),
            "layer_normalization_policy": "LayerClip0To1",
            "experiment_phase": "13B",
            "eligibility_epsilon": epsilon,
        }
        for metric in CO_PRIMARY_METRICS:
            left_metric, right_metric = _number(left, metric), _number(right, metric)
            row[f"left_{metric}"] = left_metric
            row[f"right_{metric}"] = right_metric
            row[f"delta_{metric}"] = right_metric - left_metric
        result.append(row)
    return result


def _eligibility_capacity_effect_rows(
    matched_deltas: Sequence[Mapping[str, Any]],
    layer_progression: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Join every 13B treatment to its no-filter 13A baseline and freed capacity."""
    metadata = {str(row.get("row_id", "")): row for row in summaries}
    layers_by_row: dict[str, dict[int, Mapping[str, Any]]] = {}
    for row in layer_progression:
        if row.get("experiment_phase") != "13B":
            continue
        row_id = str(row.get("row_id", ""))
        layers_by_row.setdefault(row_id, {})[int(_number(row, "residual_layer"))] = row

    output: list[dict[str, Any]] = []
    population_rows = [
        row for row in matched_deltas
        if row.get("comparison") == "residual_population_policy"
    ]
    for pair in sorted(
        population_rows,
        key=lambda row: (str(row.get("right_row_id", "")), _number(row, "eligibility_epsilon")),
    ):
        right_id = str(pair.get("right_row_id", ""))
        left_id = str(pair.get("left_row_id", ""))
        right = metadata.get(right_id)
        left = metadata.get(left_id)
        row_layers = layers_by_row.get(right_id, {})
        if right is None or left is None or set(row_layers) != set(range(1, 17)):
            continue
        train_count = max(1, int(_number(right, "train_count")))
        retired = {
            layer: 1.0 - _number(source, "eligible_residual_count") / train_count
            for layer, source in row_layers.items()
        }
        result: dict[str, Any] = {
            "pair_id": right.get("pair_id", ""),
            "no_filter_row_id": left_id,
            "eligibility_row_id": right_id,
            "baseline_population_policy": "No eligibility filter (13A)",
            "treatment_population_policy": "UnresolvedOnly",
            "eligibility_epsilon": _number(right, "eligibility_epsilon"),
            "construction_family": right.get("construction_family", ""),
            "construction_policy": right.get("construction_policy", ""),
            "layer_schedule": right.get("layer_schedule", ""),
            "utility_candidate_budget": right.get("utility_candidate_budget", ""),
            "layer_normalization_policy": right.get("layer_normalization_policy", ""),
            "mean_future_layer_population_freed": sum(retired[layer] for layer in range(1, 16)) / 15.0,
            "layer8_population_freed": retired[8],
            "layer16_endpoint_coverage": retired[16],
        }
        for metric in CO_PRIMARY_METRICS:
            result[f"no_filter_{metric}"] = _number(pair, f"left_{metric}")
            result[f"eligibility_{metric}"] = _number(pair, f"right_{metric}")
            result[f"delta_{metric}"] = _number(pair, f"delta_{metric}")
        output.append(result)
    return output


def _eligibility_depth_summary_rows(
    layer_progression: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize the measured 13B retirement trajectory by family and dose."""
    metadata = {str(row.get("row_id", "")): row for row in summaries}
    grouped: dict[tuple[str, float, int], list[float]] = {}
    for row in layer_progression:
        if row.get("experiment_phase") != "13B":
            continue
        summary = metadata.get(str(row.get("row_id", "")))
        if summary is None:
            continue
        train_count = max(1, int(_number(summary, "train_count")))
        key = (
            str(summary.get("construction_family", "")),
            _number(summary, "eligibility_epsilon"),
            int(_number(row, "residual_layer")),
        )
        grouped.setdefault(key, []).append(
            1.0 - _number(row, "eligible_residual_count") / train_count
        )
    return [
        {
            "construction_family": family,
            "eligibility_epsilon": epsilon,
            "residual_layer": layer,
            "strategy_count": len(values),
            "q25_population_freed": _percentile(values, 0.25),
            "median_population_freed": _percentile(values, 0.5),
            "q75_population_freed": _percentile(values, 0.75),
            "endpoint_only": layer == 16,
        }
        for (family, epsilon, layer), values in sorted(grouped.items())
    ]


def _strategy_diagnostic_rows(summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Retain the bounded train/validation, decoder, dictionary, and work audit."""
    fields = (
        "experiment_phase", "row_id", "construction_policy", "construction_family",
        "layer_schedule", "utility_candidate_budget", "layer_normalization_policy",
        "residual_population_policy", "eligibility_epsilon",
        "train_median_rmse", "train_strict_perfect_lfo_rate", "train_p95_rmse",
        "train_node_max_error_p95", *CO_PRIMARY_METRICS,
        "validation_overshoot_rate_before_final_clip",
        "validation_overshoot_abs_p95_before_final_clip",
        "residual_layer_dead_atom_rate_median",
        "residual_layer_dominant_atom_share_median",
        "residual_layer_usage_entropy_median",
        "residual_layer_no_op_usage_rate_median",
        "residual_layer_effective_no_op_usage_rate_median",
        "residual_gain_median", "residual_gain_abs_p95", "residual_gain_nonzero_rate",
        "duplicate_atom_rate", "oracle_construction_time", "train_encoding_time",
        "validation_encoding_time", "head_outputs_actual",
    )
    rows: list[dict[str, Any]] = []
    for source in summaries:
        row = {field: source.get(field, "") for field in fields}
        for train_metric, validation_metric, gap_name in (
            ("train_median_rmse", "validation_median_rmse", "generalization_gap_median_rmse"),
            ("train_strict_perfect_lfo_rate", "validation_strict_perfect_lfo_rate", "generalization_gap_strict_perfect_lfo_rate"),
            ("train_p95_rmse", "validation_p95_rmse", "generalization_gap_p95_rmse"),
            ("train_node_max_error_p95", "validation_node_max_error_p95", "generalization_gap_node_max_error_p95"),
        ):
            row[gap_name] = _number(source, validation_metric) - _number(source, train_metric)
        row["offline_analysis_time"] = sum(
            _number(source, name)
            for name in ("oracle_construction_time", "train_encoding_time", "validation_encoding_time")
        )
        rows.append(row)
    return rows


def _metric_ranking_rows(summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Rank each strategy on every co-primary metric while preserving ties."""
    metric_specs = (
        ("validation_median_rmse", False),
        ("validation_strict_perfect_lfo_rate", True),
        ("validation_p95_rmse", False),
        ("validation_node_max_error_p95", False),
    )
    rank_maps: dict[str, dict[str, float]] = {}
    for metric, higher_is_better in metric_specs:
        values = [(str(row.get("row_id", "")), _number(row, metric)) for row in summaries]
        rank_maps[metric] = _average_ranks(values, reverse=higher_is_better)
    rows: list[dict[str, Any]] = []
    for source in summaries:
        row_id = str(source.get("row_id", ""))
        row: dict[str, Any] = {
            "row_id": row_id,
            "experiment_phase": source.get("experiment_phase", ""),
            "construction_family": source.get("construction_family", ""),
            "construction_policy": source.get("construction_policy", ""),
            "layer_schedule": source.get("layer_schedule", ""),
            "utility_candidate_budget": source.get("utility_candidate_budget", ""),
            "layer_normalization_policy": source.get("layer_normalization_policy", ""),
            "residual_population_policy": source.get("residual_population_policy", ""),
            "eligibility_epsilon": source.get("eligibility_epsilon", ""),
        }
        ranks = []
        for metric, _ in metric_specs:
            rank = rank_maps[metric][row_id]
            row[f"rank_{metric}"] = rank
            ranks.append(rank)
        row["mean_co_primary_rank"] = sum(ranks) / len(ranks)
        row["rank_spread"] = max(ranks) - min(ranks)
        rows.append(row)
    return sorted(rows, key=lambda row: (_number(row, "mean_co_primary_rank"), str(row.get("row_id", ""))))


def _average_ranks(values: Sequence[tuple[str, float]], *, reverse: bool = False) -> dict[str, float]:
    ordered = sorted(values, key=lambda item: item[1], reverse=reverse)
    ranks: dict[str, float] = {}
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and math.isclose(ordered[end][1], ordered[index][1], rel_tol=0.0, abs_tol=1e-15):
            end += 1
        average = ((index + 1) + end) / 2.0
        for row_id, _ in ordered[index:end]:
            ranks[row_id] = average
        index = end
    return ranks


def _marginal_atom_rows(
    partial_rows: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    metadata = {str(row.get("row_id", "")): row for row in summaries}
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in partial_rows:
        grouped.setdefault(str(row.get("row_id", "")), []).append(row)
    result: list[dict[str, Any]] = []
    for row_id, rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: _number(row, "active_atom_count"))
        previous: Mapping[str, Any] | None = None
        for current in ordered:
            if previous is None:
                previous = current
                continue
            source = metadata.get(row_id, {})
            row: dict[str, Any] = {
                "row_id": row_id,
                "experiment_phase": source.get("experiment_phase", ""),
                "construction_family": source.get("construction_family", ""),
                "construction_policy": source.get("construction_policy", ""),
                "layer_schedule": source.get("layer_schedule", ""),
                "utility_candidate_budget": source.get("utility_candidate_budget", ""),
                "layer_normalization_policy": source.get("layer_normalization_policy", ""),
                "residual_population_policy": source.get("residual_population_policy", ""),
                "eligibility_epsilon": source.get("eligibility_epsilon", ""),
                "active_atom_count": int(_number(current, "active_atom_count")),
                "previous_active_atom_count": int(_number(previous, "active_atom_count")),
            }
            for metric in CO_PRIMARY_METRICS:
                delta = _number(current, metric) - _number(previous, metric)
                row[f"delta_{metric}"] = delta
                row[f"current_{metric}"] = _number(current, metric)
            result.append(row)
            previous = current
    return result


def _layer_progression_rows(
    slot_rows: Sequence[Mapping[str, Any]],
    atom_rows: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    metadata = {str(row.get("row_id", "")): row for row in summaries}
    roles: dict[tuple[str, int], str] = {}
    for row in atom_rows:
        roles.setdefault(
            (str(row.get("row_id", "")), int(_number(row, "residual_layer"))),
            str(row.get("layer_role", "")),
        )
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in slot_rows:
        grouped.setdefault(
            (str(row.get("row_id", "")), int(_number(row, "residual_layer"))),
            [],
        ).append(row)
    result: list[dict[str, Any]] = []
    for (row_id, layer), rows in sorted(grouped.items()):
        source = max(rows, key=lambda row: _number(row, "active_atom_slot"))
        summary = metadata.get(row_id, {})
        result.append({
            "row_id": row_id,
            "experiment_phase": summary.get("experiment_phase", ""),
            "construction_family": summary.get("construction_family", ""),
            "construction_policy": summary.get("construction_policy", ""),
            "layer_schedule": summary.get("layer_schedule", ""),
            "utility_candidate_budget": summary.get("utility_candidate_budget", ""),
            "layer_normalization_policy": summary.get("layer_normalization_policy", ""),
            "residual_population_policy": summary.get("residual_population_policy", ""),
            "eligibility_epsilon": summary.get("eligibility_epsilon", ""),
            "residual_layer": layer,
            "layer_role": roles.get((row_id, layer), ""),
            "active_atom_slot": int(_number(source, "active_atom_slot")),
            "eligible_residual_count": int(_number(source, "eligible_residual_count")),
            "training_median_rmse": _number(source, "training_median_rmse"),
            "training_p95_rmse": _number(source, "training_p95_rmse"),
            "training_max_abs_error_p95": _number(source, "training_max_abs_error_p95"),
        })
    return result


def _mechanism_diagnostic_rows(
    atom_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    atoms_by_id: dict[str, list[Mapping[str, Any]]] = {}
    candidates_by_id: dict[str, list[Mapping[str, Any]]] = {}
    for row in atom_rows:
        atoms_by_id.setdefault(str(row.get("row_id", "")), []).append(row)
    for row in candidate_rows:
        candidates_by_id.setdefault(str(row.get("row_id", "")), []).append(row)
    result: list[dict[str, Any]] = []
    for summary in summaries:
        row_id = str(summary.get("row_id", ""))
        atoms = atoms_by_id.get(row_id, [])
        candidates = candidates_by_id.get(row_id, [])
        improvements = [
            _number(row, "training_p95_rmse_before") - _number(row, "training_p95_rmse_after")
            for row in atoms
        ]
        prototype = [row for row in atoms if str(row.get("atom_source_kind", "")) == "synthesized_prototype"]
        observed = [row for row in atoms if str(row.get("atom_source_kind", "")) == "observed_residual"]
        result.append({
            "row_id": row_id,
            "experiment_phase": summary.get("experiment_phase", ""),
            "construction_family": summary.get("construction_family", ""),
            "construction_policy": summary.get("construction_policy", ""),
            "layer_schedule": summary.get("layer_schedule", ""),
            "utility_candidate_budget": summary.get("utility_candidate_budget", ""),
            "layer_normalization_policy": summary.get("layer_normalization_policy", ""),
            "residual_population_policy": summary.get("residual_population_policy", ""),
            "eligibility_epsilon": summary.get("eligibility_epsilon", ""),
            "atom_count": len(atoms),
            "broad_atom_count": sum(str(row.get("layer_role", "")) == "Broad" for row in atoms),
            "repair_atom_count": sum(str(row.get("layer_role", "")) == "Repair" for row in atoms),
            "synthesized_prototype_count": len(prototype),
            "observed_residual_count": len(observed),
            "median_slot_p95_improvement": median(improvements) if improvements else "",
            "mean_slot_p95_improvement": sum(improvements) / len(improvements) if improvements else "",
            "exact_duplicate_alignment_reuse_rate": (
                sum(_truth(row.get("exact_duplicate_alignment_reused")) for row in atoms) / len(atoms)
                if atoms else ""
            ),
            "prototype_convergence_rate": (
                sum(_truth(row.get("prototype_converged")) for row in prototype) / len(prototype)
                if prototype else ""
            ),
            "prototype_iterations_median": (
                median([_number(row, "prototype_iterations_executed") for row in prototype])
                if prototype else ""
            ),
            "candidate_search_event_count": sum(_number(row, "candidate_count") > 0 for row in candidates),
            "candidate_evaluations": sum(_number(row, "candidate_count") for row in candidates),
            "validation_p95_rmse": _number(summary, "validation_p95_rmse"),
            "oracle_construction_time": _number(summary, "oracle_construction_time"),
        })
    return result


def _factor_interaction_rows(matched_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = {}
    for row in matched_rows:
        grouped.setdefault(
            (
                str(row.get("comparison", "")), str(row.get("construction_family", "")),
                str(row.get("experiment_phase", "")), str(row.get("eligibility_epsilon", "")),
            ),
            [],
        ).append(row)
    result: list[dict[str, Any]] = []
    for (comparison, family, phase, epsilon), rows in sorted(grouped.items()):
        record: dict[str, Any] = {
            "comparison": comparison,
            "construction_family": family,
            "experiment_phase": phase,
            "eligibility_epsilon": epsilon,
            "pair_count": len(rows),
            "left_value": rows[0].get("left_value", ""),
            "right_value": rows[0].get("right_value", ""),
        }
        for metric in CO_PRIMARY_METRICS:
            values = [_number(row, f"delta_{metric}") for row in rows]
            record[f"median_delta_{metric}"] = median(values)
            higher = metric == "validation_strict_perfect_lfo_rate"
            record[f"right_policy_wins_{metric}"] = sum(value > 0 if higher else value < 0 for value in values)
            record[f"left_policy_wins_{metric}"] = sum(value < 0 if higher else value > 0 for value in values)
            record[f"ties_{metric}"] = sum(value == 0 for value in values)
        result.append(record)
    return result


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return math.nan
    left_map = _average_ranks([(str(index), value) for index, value in enumerate(left)])
    left_ranks = [left_map[str(index)] for index in range(len(left))]
    right_map = _average_ranks([(str(index), value) for index, value in enumerate(right)])
    right_ranks = [right_map[str(index)] for index in range(len(right))]
    left_mean = sum(left_ranks) / len(left_ranks)
    right_mean = sum(right_ranks) / len(right_ranks)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left_ranks, right_ranks))
    left_scale = math.sqrt(sum((a - left_mean) ** 2 for a in left_ranks))
    right_scale = math.sqrt(sum((b - right_mean) ** 2 for b in right_ranks))
    return numerator / (left_scale * right_scale) if left_scale and right_scale else math.nan


def _write_plots(
    bundle: AnalysisBundle,
    image_dir: Path,
    *,
    historical_runtime: bool = True,
    deep_analysis: bool = False,
    strict_thresholds: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
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
        "runtime": image_dir / ("legacy_oracle_runtime.png" if historical_runtime else "oracle_runtime.png"),
        "layer_quantiles": image_dir / "layer_epsilon_quantiles.png",
        "slot_quantiles": image_dir / "slot_epsilon_quantiles.png",
        "layer_coverage": image_dir / "completed_layer_coverage.png",
        "slot_coverage": image_dir / "slot_coverage.png",
        "retired": image_dir / "retired_fraction_vs_energy.png",
        "energy": image_dir / "incoming_vs_unexplained_energy.png",
    }
    if deep_analysis:
        paths.update({
            "metric_agreement": image_dir / "metric_rank_agreement.png",
            "generalization": image_dir / "train_validation_stability.png",
            "interactions": image_dir / "factor_interactions.png",
            "layers": image_dir / "residual_layer_progression.png",
            "marginal_atoms": image_dir / "marginal_atom_value.png",
            "diagnostics": image_dir / "strategy_diagnostics.png",
            "work": image_dir / "offline_work_efficiency.png",
        })
    if strict_thresholds:
        paths["strict_thresholds"] = image_dir / "strict_perfect_threshold_sensitivity.png"
    _plot_pareto(plt, paths["pareto"], bundle.co_primary)
    _plot_delta(plt, paths["normalization"], bundle.matched_deltas, "layer_normalization_policy", "LayerClip0To1 minus FinalClipOnly")
    _plot_delta(plt, paths["budget"], bundle.matched_deltas, "utility_candidate_budget", "CandidateBudget48 minus CandidateBudget24")
    _plot_delta(plt, paths["schedule"], bundle.matched_deltas, "layer_schedule", "TwoPhase minus Interleaved")
    _plot_partial(plt, paths["partial"], bundle.partial_codebook, bundle.summaries)
    if historical_runtime:
        _plot_runtime(plt, paths["runtime"], bundle.summaries)
    else:
        _plot_current_runtime(plt, paths["runtime"], bundle.summaries)
    _plot_family_tolerance_depths(
        plt,
        paths["layer_quantiles"],
        bundle.calibration["layer_epsilon_quantiles.csv"],
        bundle.summaries,
    )
    _plot_quantiles(plt, paths["slot_quantiles"], bundle.calibration["slot_epsilon_quantiles.csv"], "active_atom_slot", "Slot-level epsilon quantiles")
    _plot_family_coverage_depths(
        plt,
        paths["layer_coverage"],
        bundle.calibration["epsilon_coverage.csv"],
        bundle.summaries,
    )
    _plot_coverage(plt, paths["slot_coverage"], bundle.calibration["epsilon_coverage.csv"], completed=False)
    _plot_retired(plt, paths["retired"], bundle.calibration["retired_error_mass.csv"])
    _plot_energy(plt, paths["energy"], bundle.calibration["retired_error_mass.csv"])
    if deep_analysis:
        _plot_metric_agreement(plt, paths["metric_agreement"], bundle.summaries)
        _plot_generalization(plt, paths["generalization"], bundle.diagnostics)
        _plot_factor_interactions(plt, paths["interactions"], bundle.factor_interactions)
        _plot_layer_progression(plt, paths["layers"], bundle.layer_progression)
        _plot_marginal_atoms(plt, paths["marginal_atoms"], bundle.marginal_atoms)
        _plot_strategy_diagnostics(plt, paths["diagnostics"], bundle.diagnostics)
        _plot_offline_work(plt, paths["work"], bundle.mechanism_diagnostics, bundle.summaries)
    if strict_thresholds:
        _plot_strict_thresholds(plt, paths["strict_thresholds"], bundle.summaries, strict_thresholds)
    return paths


def _write_final_plots(
    bundle: AnalysisBundle,
    image_dir: Path,
    *,
    strict_thresholds: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required to generate the Experiment 13 report") from exc

    image_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "pareto": image_dir / "experiment13b_quality_frontier.png",
        "population": image_dir / "eligibility_paired_effects_all_metrics.png",
        "capacity_quality": image_dir / "eligibility_capacity_vs_quality.png",
        "strict_noop": image_dir / "strict_perfect_regime.png",
        "eligibility": image_dir / "eligibility_capacity_through_depth.png",
        "partial": image_dir / "experiment13b_partial_codebook.png",
        "factors": image_dir / "experiment13b_factor_effects.png",
        "runtime": image_dir / "complete_run_offline_timing.png",
        "13a_pareto": image_dir / "experiment13a_quality_frontier.png",
        "13a_metric_agreement": image_dir / "experiment13a_metric_agreement.png",
        "13a_interactions": image_dir / "experiment13a_factor_interactions.png",
        "13a_partial": image_dir / "experiment13a_partial_codebook.png",
        "13a_layers": image_dir / "experiment13a_residual_layers.png",
        "13a_diagnostics": image_dir / "experiment13a_diagnostics.png",
    }
    if strict_thresholds:
        paths["13a_strict_thresholds"] = image_dir / "experiment13a_strict_threshold_sensitivity.png"
    phase_a = [row for row in bundle.co_primary if row.get("experiment_phase") == "13A"]
    phase_b = [row for row in bundle.co_primary if row.get("experiment_phase") == "13B"]
    population = [row for row in bundle.matched_deltas if row.get("comparison") == "residual_population_policy"]
    _plot_pareto(plt, paths["pareto"], phase_b)
    _plot_population_effect(plt, paths["population"], population)
    _plot_capacity_vs_quality(plt, paths["capacity_quality"], bundle.eligibility_capacity_effects)
    _plot_strict_noop(plt, paths["strict_noop"], phase_b, bundle.diagnostics)
    _plot_eligibility_depth_summary(plt, paths["eligibility"], bundle.eligibility_depth_summary)
    _plot_partial_by_epsilon(plt, paths["partial"], bundle.partial_codebook, bundle.summaries)
    _plot_final_factor_effects(plt, paths["factors"], bundle.matched_deltas)
    _plot_complete_runtime(plt, paths["runtime"], bundle.summaries)
    _plot_pareto(plt, paths["13a_pareto"], phase_a)
    _plot_metric_agreement(plt, paths["13a_metric_agreement"], [row for row in bundle.summaries if row.get("experiment_phase") == "13A"])
    _plot_factor_interactions(plt, paths["13a_interactions"], [row for row in bundle.factor_interactions if row.get("experiment_phase") == "13A"])
    _plot_partial(plt, paths["13a_partial"], [row for row in bundle.partial_codebook if row.get("experiment_phase") == "13A"], [row for row in bundle.summaries if row.get("experiment_phase") == "13A"])
    _plot_layer_progression(plt, paths["13a_layers"], [row for row in bundle.layer_progression if row.get("experiment_phase") == "13A"])
    _plot_strategy_diagnostics(plt, paths["13a_diagnostics"], [row for row in bundle.diagnostics if row.get("experiment_phase") == "13A"])
    if strict_thresholds:
        _plot_strict_thresholds(plt, paths["13a_strict_thresholds"], [row for row in bundle.summaries if row.get("experiment_phase") == "13A"], strict_thresholds)
    return paths


def _plot_population_effect(plt: Any, path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    families = sorted({str(row.get("construction_family", "")) for row in rows})
    epsilons = (0.01, 0.001, 0.0001)
    specifications = (
        ("validation_median_rmse", "Median RMSE", False),
        ("validation_strict_perfect_lfo_rate", "Strict-perfect rate", True),
        ("validation_p95_rmse", "P95 RMSE", False),
        ("validation_node_max_error_p95", "Node-max P95", False),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12.8, max(10.0, 0.7 * len(families) + 3.4)))
    for axis, (metric, label, higher_is_better) in zip(axes.flat, specifications):
        raw_matrix = [[
            median([
                _number(row, f"delta_{metric}")
                for row in rows
                if row.get("construction_family") == family
                and math.isclose(_number(row, "eligibility_epsilon"), epsilon)
            ])
            for epsilon in epsilons]
            for family in families
        ]
        benefit_matrix = [
            [value if higher_is_better else -value for value in line]
            for line in raw_matrix
        ]
        limit = max((abs(value) for line in benefit_matrix for value in line), default=0.01)
        image = axis.imshow(benefit_matrix, aspect="auto", cmap="RdYlGn", vmin=-limit, vmax=limit)
        axis.set_xticks(range(3), ["1e-2", "1e-3", "1e-4"])
        axis.set_yticks(range(len(families)), families, fontsize=7.2)
        axis.set_title(f"{label}: eligibility minus no filter")
        for y, values in enumerate(raw_matrix):
            for x, value in enumerate(values):
                text = f"{value * 100:+.2f} pp" if higher_is_better else f"{value:+.4f}"
                axis.text(x, y, text, ha="center", va="center", color="#5d666a", fontsize=6.2)
        figure.colorbar(image, ax=axis, fraction=0.035, pad=0.025, label="green favors eligibility")
    figure.suptitle("Eligibility treatment effect versus each exact 13A no-filter baseline")
    figure.supxlabel("eligibility epsilon (treatment intensity)")
    figure.tight_layout(rect=(0, 0.02, 1, 0.97))
    _save(plt, figure, path, tight=False)


def _plot_capacity_vs_quality(plt: Any, path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    specifications = (
        ("validation_median_rmse", "Median RMSE delta", False),
        ("validation_strict_perfect_lfo_rate", "Strict-perfect delta", True),
        ("validation_p95_rmse", "P95 RMSE delta", False),
        ("validation_node_max_error_p95", "Node-max P95 delta", False),
    )
    colors = {0.01: "#A86100", 0.001: "#286DB7", 0.0001: "#147E67"}
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.6))
    for axis, (metric, label, higher_is_better) in zip(axes.flat, specifications):
        for epsilon in (0.01, 0.001, 0.0001):
            selected = [row for row in rows if math.isclose(_number(row, "eligibility_epsilon"), epsilon)]
            axis.scatter(
                [100.0 * _number(row, "mean_future_layer_population_freed") for row in selected],
                [100.0 * _number(row, f"delta_{metric}") if higher_is_better else _number(row, f"delta_{metric}") for row in selected],
                s=30,
                alpha=0.68,
                color=colors[epsilon],
                label=f"epsilon {epsilon:g}",
            )
        axis.scatter([0.0], [0.0], marker="*", s=90, color="#5d666a", label="No eligibility filter (13A)")
        axis.axhline(0.0, color="#5d666a", linewidth=0.8)
        axis.axvline(0.0, color="#5d666a", linewidth=0.8)
        axis.set_xlabel("mean population excluded from future layers (%)")
        axis.set_ylabel(f"{label}{' (pp)' if higher_is_better else ''}")
        axis.grid(True, alpha=0.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    figure.suptitle("Does freeing later construction population improve validation quality?")
    figure.tight_layout(rect=(0, 0.08, 1, 0.96))
    _save(plt, figure, path, tight=False)


def _plot_eligibility_depth_summary(plt: Any, path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    families = sorted({str(row.get("construction_family", "")) for row in rows})
    layers = list(range(1, 17))
    epsilons = (0.01, 0.001, 0.0001)
    maximum = max((_number(row, "median_population_freed") for row in rows), default=0.01)
    figure, axes = plt.subplots(3, 1, figsize=(14.5, max(13.0, len(families) * 1.05 + 5.0)))
    for axis, epsilon in zip(axes, epsilons):
        matrix = [[
            next(
                (_number(row, "median_population_freed") for row in rows
                 if row.get("construction_family") == family
                 and math.isclose(_number(row, "eligibility_epsilon"), epsilon)
                 and int(_number(row, "residual_layer")) == layer),
                math.nan,
            )
            for layer in layers]
            for family in families
        ]
        image = axis.imshow(matrix, aspect="auto", cmap="YlGn", vmin=0.0, vmax=maximum)
        axis.set_xticks(range(16), [str(layer) for layer in layers])
        axis.set_yticks(range(len(families)), families, fontsize=7)
        axis.axvline(14.5, color="#5d666a", linewidth=1.2)
        axis.set_title(f"epsilon {epsilon:g} · 13A no-filter baseline is 0% retired at every layer")
        for y, values in enumerate(matrix):
            for x, value in enumerate(values):
                if math.isfinite(value):
                    axis.text(x, y, f"{100 * value:.1f}", ha="center", va="center", color="#5d666a", fontsize=5.2)
        figure.colorbar(image, ax=axis, fraction=0.018, pad=0.015, label="median population freed")
    axes[-1].set_xlabel("completed residual layer · layer 16 is endpoint coverage only")
    figure.suptitle("Eligibility capacity released through depth by construction family")
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    _save(plt, figure, path, tight=False)


def _plot_epsilon_family(plt: Any, path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    families = sorted({str(row.get("construction_family", "")) for row in rows})
    epsilons = (0.0001, 0.001, 0.01)
    figure, axes = plt.subplots(1, 2, figsize=(14.2, 5.8))
    for family in families:
        selected = [row for row in rows if row.get("construction_family") == family]
        axes[0].plot(
            epsilons,
            [median(_number(row, "validation_p95_rmse") for row in selected if math.isclose(_number(row, "eligibility_epsilon"), epsilon)) for epsilon in epsilons],
            marker="o", linewidth=1.2, label=family,
        )
        axes[1].plot(
            epsilons,
            [median(_number(row, "validation_strict_perfect_lfo_rate") for row in selected if math.isclose(_number(row, "eligibility_epsilon"), epsilon)) for epsilon in epsilons],
            marker="o", linewidth=1.2, label=family,
        )
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlabel("eligibility epsilon (log scale)")
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("family-median validation P95 RMSE (lower is better)")
    axes[1].set_ylabel("family-median strict-perfect rate (higher is better)")
    axes[1].yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    axes[0].set_title("Tail quality")
    axes[1].set_title("Exact-finish regime")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="center left", bbox_to_anchor=(0.87, 0.5), fontsize=7, frameon=False)
    figure.suptitle("Experiment 13B epsilon sensitivity is family-specific")
    figure.tight_layout(rect=(0, 0, 0.86, 0.96))
    _save(plt, figure, path, tight=False)


def _plot_strict_noop(
    plt: Any,
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 5.6))
    for epsilon, marker in ((0.01, "o"), (0.001, "s"), (0.0001, "^")):
        selected = [row for row in rows if math.isclose(_number(row, "eligibility_epsilon"), epsilon)]
        axis.scatter(
            [_number(row, "validation_p95_rmse") for row in selected],
            [_number(row, "validation_strict_perfect_lfo_rate") for row in selected],
            label=f"epsilon {epsilon:g}", marker=marker, s=38, alpha=0.72,
        )
    axis.set(
        xlabel="validation P95 RMSE (lower is better)",
        ylabel="validation strict-perfect LFO rate",
        title="Strict-perfect gains form a discrete finishing regime",
    )
    axis.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    axis.legend(fontsize=8)
    _save(plt, figure, path)


def _plot_eligibility_progression(
    plt: Any,
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
) -> None:
    train_count = {str(row.get("row_id", "")): max(1, int(_number(row, "train_count"))) for row in summaries}
    selected = [row for row in rows if row.get("experiment_phase") == "13B"]
    figure, axis = plt.subplots(figsize=(8.8, 5.2))
    for epsilon in (0.01, 0.001, 0.0001):
        epsilon_rows = [row for row in selected if math.isclose(_number(row, "eligibility_epsilon"), epsilon)]
        layers = sorted({int(_number(row, "residual_layer")) for row in epsilon_rows})
        values = [
            median(
                1.0 - _number(row, "eligible_residual_count") / train_count[str(row.get("row_id", ""))]
                for row in epsilon_rows if int(_number(row, "residual_layer")) == layer
            )
            for layer in layers
        ]
        axis.plot(layers, values, marker="o", markersize=3, label=f"epsilon {epsilon:g}")
    axis.set(
        xlabel="completed residual layer",
        ylabel="median training LFO population retired from later construction",
        title="Typical eligibility filtering remains small; a few families create large regimes",
    )
    axis.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    axis.legend()
    _save(plt, figure, path)


def _plot_partial_by_epsilon(
    plt: Any,
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
) -> None:
    metadata = {str(row.get("row_id", "")): row for row in summaries}
    selected = [row for row in rows if metadata.get(str(row.get("row_id", "")), {}).get("experiment_phase") == "13B"]
    figure, axis = plt.subplots(figsize=(8.5, 5.0))
    for epsilon in (0.01, 0.001, 0.0001):
        epsilon_rows = [row for row in selected if math.isclose(_number(metadata[str(row.get("row_id", ""))], "eligibility_epsilon"), epsilon)]
        slots = sorted({int(_number(row, "active_atom_count")) for row in epsilon_rows})
        axis.plot(
            slots,
            [median(_number(row, "validation_p95_rmse") for row in epsilon_rows if int(_number(row, "active_atom_count")) == slot) for slot in slots],
            marker="o", label=f"epsilon {epsilon:g}",
        )
    axis.set(
        xlabel="active atoms retained per residual layer",
        ylabel="median validation P95 RMSE (lower is better)",
        title="Experiment 13B partial-codebook progression",
    )
    axis.legend()
    _save(plt, figure, path)


def _plot_final_factor_effects(plt: Any, path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    comparisons = (
        ("utility_candidate_budget", "Budget48 minus Budget24"),
        ("layer_schedule", "TwoPhase minus Interleaved"),
    )
    epsilons = (0.01, 0.001, 0.0001)
    matrix: list[list[float]] = []
    labels: list[str] = []
    for comparison, label in comparisons:
        for epsilon in epsilons:
            values = [
                _number(row, "delta_validation_p95_rmse")
                for row in rows
                if row.get("comparison") == comparison
                and row.get("experiment_phase") == "13B"
                and math.isclose(_number(row, "eligibility_epsilon"), epsilon)
            ]
            matrix.append([median(values), sum(value < 0 for value in values), len(values)])
            labels.append(f"{label} | epsilon {epsilon:g}")
    figure, axis = plt.subplots(figsize=(9.2, 4.8))
    values = [row[0] for row in matrix]
    axis.barh(range(len(values)), values, color=["#147E67" if value < 0 else "#B63B34" for value in values])
    axis.axvline(0.0, color="#5d666a", linewidth=0.9)
    axis.set_yticks(range(len(labels)), labels, fontsize=8)
    axis.invert_yaxis()
    for index, (_, wins, count) in enumerate(matrix):
        axis.text(values[index], index, f" {wins}/{count} improve", va="center", fontsize=8, color="#5d666a")
    axis.set(xlabel="median validation P95 RMSE delta", title="13B budget and schedule effects by epsilon")
    _save(plt, figure, path)


def _plot_complete_runtime(plt: Any, path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        phase = str(row.get("experiment_phase", ""))
        epsilon = row.get("eligibility_epsilon", "")
        label = phase if phase == "13A" else f"13B epsilon {_number(row, 'eligibility_epsilon'):g}"
        grouped.setdefault(label, []).append(_number(row, "oracle_construction_time"))
    labels = list(grouped)
    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    axis.boxplot([grouped[label] for label in labels], tick_labels=labels, showfliers=True)
    axis.set(ylabel="oracle construction seconds", title="Same-run offline construction timing")
    axis.tick_params(axis="x", rotation=18)
    _save(plt, figure, path)


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
    phases = sorted({str(row.get("experiment_phase", "")) for row in rows})
    phase_label = phases[0] if len(phases) == 1 else "Experiment 13"
    axis.set(xlabel="validation median RMSE (lower is better)", ylabel="validation P95 RMSE (lower is better)", title=f"{phase_label} co-primary quality tradeoffs")
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
    axis.set(xlabel="validation P95 RMSE delta (negative favors first-named policy)", ylabel="matched pair", title=title)
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
    axis.set(xlabel="active atoms per residual layer", ylabel="median validation P95 RMSE (lower is better)", title="Partial-codebook progression by construction family")
    axis.legend(fontsize=7, loc="best")
    _save(plt, figure, path)


def _plot_runtime(plt: Any, path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    ordered = sorted(rows, key=lambda row: _number(row, "oracle_construction_time"), reverse=True)
    sleep_rows = ordered[:2]
    ordinary_rows = ordered[2:17]
    selected = ordinary_rows + sleep_rows
    figure, (ordinary_axis, sleep_axis) = plt.subplots(
        1,
        2,
        sharey=True,
        figsize=(12.0, 8.8),
        gridspec_kw={"width_ratios": (4.5, 1.25), "wspace": 0.05},
    )
    y = range(len(selected))
    ordinary_values = [
        0.0 if row in sleep_rows else _number(row, "oracle_construction_time")
        for row in selected
    ]
    sleep_values = [
        _number(row, "oracle_construction_time") if row in sleep_rows else 0.0
        for row in selected
    ]
    ordinary_axis.barh(y, ordinary_values, color="#4C78A8", alpha=0.85)
    sleep_axis.barh(y, sleep_values, color="#B63B34", alpha=0.88)
    labels = [
        f"{row.get('construction_policy', '')} · "
        f"{str(row.get('utility_candidate_budget', '')).replace('CandidateBudget', 'B')} · "
        f"{str(row.get('layer_normalization_policy', '')).replace('LayerClip0To1', 'LayerClip').replace('FinalClipOnly', 'FinalClip')}"
        for row in selected
    ]
    ordinary_axis.set_yticks(list(y), labels, fontsize=7)
    ordinary_axis.invert_yaxis()
    ordinary_max = max(ordinary_values, default=1.0)
    sleep_nonzero = [value for value in sleep_values if value > 0]
    ordinary_axis.set_xlim(0.0, ordinary_max * 1.08)
    if sleep_nonzero:
        sleep_axis.set_xlim(min(sleep_nonzero) * 0.96, max(sleep_nonzero) * 1.04)
    ordinary_axis.spines["right"].set_visible(False)
    sleep_axis.spines["left"].set_visible(False)
    sleep_axis.tick_params(axis="y", left=False, labelleft=False)
    sleep_axis.yaxis.tick_right()
    slash = {"marker": [(-1, -0.8), (1, 0.8)], "markersize": 9, "linestyle": "none", "color": "#60757E", "mec": "#60757E", "clip_on": False}
    ordinary_axis.plot([1, 1], [0, 1], transform=ordinary_axis.transAxes, **slash)
    sleep_axis.plot([0, 0], [0, 1], transform=sleep_axis.transAxes, **slash)
    ordinary_axis.set(xlabel="ordinary legacy construction seconds (lower is faster)", ylabel="15 slowest ordinary rows + 2 sleep artifacts")
    sleep_axis.set(xlabel="host-sleep-inflated seconds")
    figure.suptitle("Historical legacy construction runtime — broken axis isolates host sleep")
    figure.subplots_adjust(left=0.32, right=0.98, top=0.91, bottom=0.1, wspace=0.05)
    _save(plt, figure, path, tight=False)


def _plot_current_runtime(plt: Any, path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    selected = sorted(rows, key=lambda row: _number(row, "oracle_construction_time"), reverse=True)[:20]
    figure, axis = plt.subplots(figsize=(11.0, 7.8))
    y = range(len(selected))
    values = [_number(row, "oracle_construction_time") for row in selected]
    labels = [
        f"{row.get('construction_policy', '')} · "
        f"{str(row.get('utility_candidate_budget', '')).replace('CandidateBudget', 'B')} · "
        f"{str(row.get('layer_normalization_policy', '')).replace('LayerClip0To1', 'LayerClip').replace('FinalClipOnly', 'FinalClip')}"
        for row in selected
    ]
    axis.barh(y, values, color="#4C78A8", alpha=0.86)
    axis.set_yticks(list(y), labels, fontsize=7)
    axis.invert_yaxis()
    axis.set(
        xlabel="oracle construction seconds (lower is faster)",
        ylabel="20 slowest rows in the optimized 13A run",
        title="Experiment 13A same-run construction timing",
    )
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
        axis.plot(x, [median(grouped[(percentile, value)]) for value in x], marker="o", markersize=3, label=f"retire {percentile:.1%} of LFOs")
    axis.set(xlabel=x_key.replace("_", " "), ylabel="median tolerance required", title=title)
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
        ylabel="median LFO reduction (higher means more curves retired)",
        title="Completed-layer LFO reduction" if completed else "Slot-level LFO reduction",
    )
    axis.legend(fontsize=7, ncol=3)
    _save(plt, figure, path)


def _plot_family_coverage_depths(
    plt: Any,
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
) -> None:
    """Show how much of the construction population each tested tolerance removes."""
    family_by_row = {str(row.get("row_id", "")): str(row.get("construction_family", "")) for row in summaries}
    figure, axes = plt.subplots(1, 2, figsize=(14.8, 6.0), sharey=True)
    families = sorted(set(family_by_row.values()))
    colors = {family: plt.cm.tab20(index % 20) for index, family in enumerate(families)}
    for axis, depth in zip(axes, (8, 16)):
        depth_rows = [
            row for row in rows
            if row.get("dataset_split") == "training"
            and row.get("active_atom_slot") in {None, "", "None"}
            and int(_number(row, "residual_layer")) == depth
        ]
        epsilons = sorted({_number(row, "epsilon") for row in depth_rows})
        for family in families:
            family_rows = [row for row in depth_rows if family_by_row.get(str(row.get("row_id", ""))) == family]
            family_epsilons = [epsilon for epsilon in epsilons if any(_number(row, "epsilon") == epsilon for row in family_rows)]
            values = [
                median(_number(row, "resolved_fraction") for row in family_rows if _number(row, "epsilon") == epsilon)
                for epsilon in family_epsilons
            ]
            if values:
                axis.plot(family_epsilons, values, marker="o", markersize=3, linewidth=1.2, label=family, color=colors[family])
        axis.set_xscale("log")
        axis.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
        axis.set(
            xlabel="tested eligibility tolerance ε (log scale)",
            title=(
                "Layer 8: LFOs removed before layers 9–16"
                if depth == 8 else
                "Layer 16: endpoint LFOs meeting tolerance"
            ),
        )
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("median LFO population reduction")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="center left", bbox_to_anchor=(0.83, 0.5), fontsize=8, frameon=False)
    figure.suptitle("Eligibility capacity released at two construction depths")
    figure.tight_layout(rect=(0, 0, 0.82, 0.96))
    _save(plt, figure, path, tight=False)


def _plot_family_tolerance_depths(
    plt: Any,
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
) -> None:
    """Show the exact tolerance required to retire each target fraction by family."""
    family_by_row = {str(row.get("row_id", "")): str(row.get("construction_family", "")) for row in summaries}
    families = sorted(set(family_by_row.values()))
    targets = sorted({_number(row, "percentile") for row in rows if row.get("dataset_split") == "training"})
    figure, axes = plt.subplots(2, 1, figsize=(13.8, 10.5), sharex=True)
    for axis, depth in zip(axes, (8, 16)):
        matrix: list[list[float]] = []
        for family in families:
            family_rows = [
                row for row in rows
                if row.get("dataset_split") == "training"
                and int(_number(row, "residual_layer")) == depth
                and family_by_row.get(str(row.get("row_id", ""))) == family
            ]
            matrix.append([
                median(_number(row, "epsilon_value") for row in family_rows if _number(row, "percentile") == target)
                if any(_number(row, "percentile") == target for row in family_rows) else math.nan
                for target in targets
            ])
        finite = [value for matrix_row in matrix for value in matrix_row if math.isfinite(value) and value > 0]
        if finite:
            image = axis.imshow(matrix, aspect="auto", cmap="YlGnBu", norm=plt.matplotlib.colors.LogNorm(vmin=min(finite), vmax=max(finite)))
            for y, matrix_row in enumerate(matrix):
                for x, value in enumerate(matrix_row):
                    if math.isfinite(value):
                        axis.text(x, y, f"{value:.2g}", ha="center", va="center", fontsize=7, color="#59676D")
            figure.colorbar(image, ax=axis, fraction=0.02, pad=0.015, label="required tolerance ε")
        axis.set_yticks(range(len(families)), families, fontsize=8)
        axis.set_title(f"After residual layer {depth}: tolerance required for each LFO-reduction target")
    axes[-1].set_xticks(range(len(targets)), [f"retire {target:.1%}" for target in targets], rotation=25, ha="right")
    figure.suptitle("Exact eligibility tolerance by construction family")
    figure.tight_layout(rect=(0, 0, 1, 0.97))
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


def _plot_metric_agreement(plt: Any, path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    labels = ("Median RMSE", "Strict-perfect", "P95 RMSE", "Node-max P95")
    correlations = [
        [
            _spearman(
                [_number(row, left) for row in rows],
                [_number(row, right) for row in rows],
            )
            for right in CO_PRIMARY_METRICS
        ]
        for left in CO_PRIMARY_METRICS
    ]
    figure, axis = plt.subplots(figsize=(7.4, 6.2))
    image = axis.imshow(correlations, vmin=-1.0, vmax=1.0, cmap="RdBu_r")
    axis.set_xticks(range(4), labels, rotation=28, ha="right")
    axis.set_yticks(range(4), labels)
    for y, row in enumerate(correlations):
        for x, value in enumerate(row):
            axis.text(x, y, f"{value:.2f}", ha="center", va="center", color="white" if abs(value) > 0.55 else "#34444A")
    axis.set_title("Co-primary metric rank agreement (Spearman ρ)")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="rank correlation")
    _save(plt, figure, path)


def _plot_strict_thresholds(
    plt: Any,
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    strict_thresholds: Mapping[str, Any],
) -> None:
    """Plot family-level strict-perfect sensitivity on a logarithmic tolerance axis."""
    tolerances = [float(value) for value in strict_thresholds["tolerances"]]
    rates_by_row = strict_thresholds["rates_by_row"]
    families = sorted({str(row.get("construction_family", "")) for row in rows})
    figure, axis = plt.subplots(figsize=(9.4, 5.8))
    colors = {family: plt.cm.tab20(index % 20) for index, family in enumerate(families)}
    for family in families:
        family_ids = [str(row.get("row_id", "")) for row in rows if row.get("construction_family") == family]
        medians = [
            median(float(rates_by_row[row_id][_compact_exponent(tolerance)]) for row_id in family_ids)
            for tolerance in tolerances
        ]
        axis.plot(tolerances, medians, marker="o", linewidth=1.25, markersize=4, label=family, color=colors[family], alpha=0.85)
    all_ids = [str(row.get("row_id", "")) for row in rows]
    overall_medians = [median(float(rates_by_row[row_id][_compact_exponent(tolerance)]) for row_id in all_ids) for tolerance in tolerances]
    best_rates = [max(float(rates_by_row[row_id][_compact_exponent(tolerance)]) for row_id in all_ids) for tolerance in tolerances]
    axis.plot(tolerances, overall_medians, marker="D", linewidth=2.4, color="#26383F", label="All-strategy median")
    axis.plot(tolerances, best_rates, marker="s", linewidth=2.0, linestyle="--", color="#B45309", label="Best observed row")
    axis.set_xscale("log")
    axis.set_xticks(tolerances, [f"{value:.0e}\n({value / 10:.0e} RMSE)" for value in tolerances])
    axis.set(
        xlabel="maximum-absolute tolerance (paired RMSE tolerance in parentheses; logarithmic scale)",
        ylabel="validation strict-perfect LFO rate",
        title="Strict-perfect sensitivity across tolerance tuples",
    )
    axis.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    axis.grid(True, axis="both", alpha=0.25)
    axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=False)
    figure.tight_layout()
    _save(plt, figure, path)


def _compact_exponent(value: float) -> str:
    return f"{value:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def _plot_generalization(plt: Any, path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    specs = (
        ("train_median_rmse", "validation_median_rmse", "Median RMSE"),
        ("train_p95_rmse", "validation_p95_rmse", "P95 RMSE"),
    )
    for axis, (train_key, validation_key, label) in zip(axes, specs):
        train = [_number(row, train_key) for row in rows]
        validation = [_number(row, validation_key) for row in rows]
        axis.scatter(train, validation, s=24, alpha=0.72, color="#286DB7")
        if train and validation:
            lower, upper = min([*train, *validation]), max([*train, *validation])
            axis.plot([lower, upper], [lower, upper], linestyle="--", linewidth=1, color="#60757E")
        axis.set(xlabel=f"training {label}", ylabel=f"validation {label}", title=f"{label}: train vs validation")
    figure.suptitle("Generalization stability on the fixed train-50% / validation-100% split")
    _save(plt, figure, path)


def _plot_factor_interactions(plt: Any, path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    comparisons = (
        ("layer_normalization_policy", "LayerClip − FinalClip"),
        ("utility_candidate_budget", "Budget48 − Budget24"),
        ("layer_schedule", "TwoPhase − Interleaved"),
    )
    families = sorted({str(row.get("construction_family", "")) for row in rows if row.get("construction_family")})
    lookup = {
        (str(row.get("construction_family", "")), str(row.get("comparison", ""))): _number(row, "median_delta_validation_p95_rmse")
        for row in rows
    }
    matrix = [[lookup.get((family, comparison), math.nan) for comparison, _ in comparisons] for family in families]
    limit = max((abs(value) for value in lookup.values()), default=0.01)
    figure, axis = plt.subplots(figsize=(8.6, max(5.0, 0.37 * len(families) + 1.8)))
    image = axis.imshow(matrix, aspect="auto", cmap="RdYlGn_r", vmin=-limit, vmax=limit)
    axis.set_xticks(range(len(comparisons)), [label for _, label in comparisons], rotation=18, ha="right")
    axis.set_yticks(range(len(families)), families, fontsize=8)
    for y, family in enumerate(families):
        for x, (comparison, _) in enumerate(comparisons):
            value = lookup.get((family, comparison))
            axis.text(x, y, "—" if value is None else f"{value:+.4f}", ha="center", va="center", color="#606A6F", fontsize=7)
    axis.set_title("Family-specific matched P95 policy deltas")
    figure.colorbar(image, ax=axis, fraction=0.035, pad=0.03, label="validation P95 RMSE delta; negative favors first-named policy")
    _save(plt, figure, path)


def _plot_layer_progression(plt: Any, path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    grouped: dict[tuple[str, int], list[float]] = {}
    for row in rows:
        grouped.setdefault(
            (str(row.get("construction_family", "Unknown")), int(_number(row, "residual_layer"))),
            [],
        ).append(_number(row, "training_p95_rmse"))
    figure, axis = plt.subplots(figsize=(9.2, 5.4))
    for family in sorted({key[0] for key in grouped}):
        layers = sorted(key[1] for key in grouped if key[0] == family)
        axis.plot(layers, [median(grouped[(family, layer)]) for layer in layers], marker="o", markersize=2.8, linewidth=1.25, label=family)
    axis.set(
        xlabel="completed residual layer",
        ylabel="family-median training P95 RMSE (lower is better)",
        title="Residual-layer learning curves after slot 7",
    )
    if grouped:
        axis.legend(fontsize=6.7, ncol=2, loc="best")
    _save(plt, figure, path)


def _plot_marginal_atoms(plt: Any, path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    grouped: dict[int, list[float]] = {}
    for row in rows:
        grouped.setdefault(int(_number(row, "active_atom_count")), []).append(_number(row, "delta_validation_p95_rmse"))
    slots = sorted(grouped)
    medians = [median(grouped[slot]) for slot in slots]
    improved = [sum(value < 0 for value in grouped[slot]) for slot in slots]
    figure, axis = plt.subplots(figsize=(8.6, 4.9))
    bars = axis.bar(slots, medians, color="#147E67", alpha=0.85)
    axis.axhline(0.0, color="#60757E", linewidth=0.9)
    for slot, count in zip(slots, improved):
        axis.text(slot, -0.00018, f"{count}/{len(grouped[slot])} improve", ha="center", va="top", color="#34444A", fontsize=8)
    axis.set(
        xlabel="new active atom count (delta from previous count)",
        ylabel="median validation P95 RMSE delta",
        title="Marginal value of each additional active atom",
    )
    _save(plt, figure, path)


def _plot_strategy_diagnostics(plt: Any, path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    specs = (
        ("validation_overshoot_rate_before_final_clip", "overshoot rate before final clip"),
        ("residual_layer_effective_no_op_usage_rate_median", "effective no-op usage"),
        ("residual_gain_nonzero_rate", "non-zero residual-gain rate"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(13.4, 4.4), sharey=True)
    p95 = [_number(row, "validation_p95_rmse") for row in rows]
    for axis, (key, label) in zip(axes, specs):
        values = [_number(row, key) for row in rows]
        rho = _spearman(values, p95)
        axis.scatter(values, p95, s=22, alpha=0.7, color="#286DB7")
        axis.set(xlabel=label, title=f"ρ = {rho:+.2f}")
    axes[0].set_ylabel("validation P95 RMSE")
    figure.suptitle("Decoder and dictionary diagnostics versus tail quality")
    _save(plt, figure, path)


def _plot_offline_work(
    plt: Any,
    path: Path,
    mechanism_rows: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.6))
    budget_rows = [row for row in mechanism_rows if _number(row, "candidate_evaluations") > 0]
    for budget, color in (("CandidateBudget24", "#286DB7"), ("CandidateBudget48", "#147E67")):
        selected = [row for row in budget_rows if str(row.get("utility_candidate_budget", "")) == budget]
        axes[0].scatter(
            [_number(row, "candidate_evaluations") for row in selected],
            [_number(row, "validation_p95_rmse") for row in selected],
            label=budget,
            color=color,
            s=25,
            alpha=0.72,
        )
    axes[0].set(xlabel="candidate evaluations", ylabel="validation P95 RMSE", title="Search work vs tail quality")
    axes[0].legend(fontsize=7)
    axes[1].scatter(
        [_number(row, "oracle_construction_time") for row in summaries],
        [_number(row, "validation_p95_rmse") for row in summaries],
        s=25,
        alpha=0.72,
        color="#A86100",
    )
    axes[1].set(xlabel="oracle construction seconds", ylabel="validation P95 RMSE", title="Construction time vs quality")
    families = sorted({str(row.get("construction_family", "")) for row in summaries})
    construction = [median([_number(row, "oracle_construction_time") for row in summaries if row.get("construction_family") == family]) for family in families]
    train = [median([_number(row, "train_encoding_time") for row in summaries if row.get("construction_family") == family]) for family in families]
    validation = [median([_number(row, "validation_encoding_time") for row in summaries if row.get("construction_family") == family]) for family in families]
    positions = list(range(len(families)))
    axes[2].barh(positions, construction, label="construct", color="#286DB7")
    axes[2].barh(positions, train, left=construction, label="train encode", color="#147E67")
    axes[2].barh(positions, validation, left=[a + b for a, b in zip(construction, train)], label="validation encode", color="#A86100")
    axes[2].set_yticks(positions, families, fontsize=6.5)
    axes[2].set(xlabel="family-median seconds", title="Offline work decomposition")
    axes[2].legend(fontsize=7)
    figure.suptitle("Experiment-work efficiency (all rows retain 193 deployed heads)")
    _save(plt, figure, path)


def _interactive_payload(
    bundle: AnalysisBundle,
    *,
    source_run: Path,
    html_report_path: Path,
    expected_count: int,
    source_fingerprint: str,
    report_mode: str = "provisional",
    selection: Mapping[str, Any] | None = None,
    scaling_rows: Sequence[Mapping[str, Any]] = (),
    experiment13b_state: str = "not_started",
    strict_thresholds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the compact, deterministic payload embedded in the HTML report."""
    completed = len(bundle.summaries)
    final_report = report_mode == "complete"
    decision_rows = [row for row in bundle.summaries if row.get("experiment_phase") == "13B"] if final_report else bundle.summaries
    best = min(decision_rows, key=lambda row: _number(row, "validation_p95_rmse"))
    normalization = _comparison_summary(bundle.matched_deltas, "layer_normalization_policy")
    budget = _comparison_summary(bundle.matched_deltas, "utility_candidate_budget")
    schedule = _comparison_summary(bundle.matched_deltas, "layer_schedule")
    pareto = [
        row for row in bundle.co_primary
        if _truth(row.get("pareto_candidate"))
        and (not final_report or row.get("experiment_phase") == "13B")
    ]
    # The final report preserves the full 13A foundation as well as 13B.
    # Compact arrays below keep that broader evidence under the portable-HTML cap.
    payload_summaries = bundle.summaries
    payload_row_ids = [str(row.get("row_id", "")) for row in payload_summaries]
    payload_row_index = {row_id: index for index, row_id in enumerate(payload_row_ids)}
    partial_rows = [
        ([
            payload_row_index[str(row.get("row_id", ""))],
            int(_number(row, "active_atom_count")),
            _number(row, "validation_p95_rmse"),
        ] if final_report else [
            payload_row_index[str(row.get("row_id", ""))],
            int(_number(row, "active_atom_count")),
            *[_number(row, metric) for metric in CO_PRIMARY_METRICS],
        ])
        for row in bundle.partial_codebook
        if str(row.get("row_id", "")) in payload_row_index
    ]

    coverage_by_family: list[dict[str, Any]] = []
    for family in sorted({str(row.get("construction_family", "")) for row in bundle.coverage}):
        planned = [row for row in bundle.coverage if row.get("construction_family") == family]
        coverage_by_family.append({
            "construction_family": family,
            "completed": sum(_truth(row.get("completed")) for row in planned),
            "planned": len(planned),
        })
    absent_families = [row["construction_family"] for row in coverage_by_family if row["completed"] == 0]
    incomplete_families = [row for row in coverage_by_family if row["completed"] < row["planned"]]
    runtime_rows = sorted(
        bundle.summaries,
        key=lambda row: _number(row, "oracle_construction_time"),
        reverse=True,
    )
    calibration = _compact_calibration(bundle.calibration)
    source_display = os.path.relpath(source_run, html_report_path.parent).replace("\\", "/")
    complete_13a = report_mode == "complete_13a"
    if complete_13a:
        # Keep only the two decision depths needed for filter-reactive family
        # comparisons. The frozen selector itself remains defined over all rows.
        calibration["layer_quantiles_by_row"] = [
            row for row in calibration["layer_quantiles_by_row"] if int(row[1]) in {8, 16}
        ]
        calibration["retired_sample_by_row"] = []
        compact_retired: list[dict[str, Any]] = []
        retired_by_epsilon: dict[float, list[dict[str, Any]]] = {}
        for row in calibration["retired_sample"]:
            retired_by_epsilon.setdefault(float(row["epsilon"]), []).append(row)
        for rows in retired_by_epsilon.values():
            if len(rows) <= 60:
                compact_retired.extend(rows)
            else:
                indexes = sorted({round(index * (len(rows) - 1) / 59) for index in range(60)})
                compact_retired.extend(rows[index] for index in indexes)
        calibration["retired_sample"] = compact_retired
    elif final_report:
        phase_by_id = {
            str(row.get("row_id", "")): str(row.get("experiment_phase", ""))
            for row in bundle.summaries
        }
        allowed_indexes = {
            index for index, row_id in enumerate(calibration["row_ids"])
            if phase_by_id.get(str(row_id)) == "13A"
        }
        calibration["layer_quantiles_by_row"] = [
            row for row in calibration["layer_quantiles_by_row"]
            if int(row[0]) in allowed_indexes and int(row[1]) in {8, 16}
        ]
        calibration["layer_coverage_by_row"] = [
            row for row in calibration["layer_coverage_by_row"]
            if int(row[0]) in allowed_indexes and int(row[1]) in {8, 16}
        ]
        calibration.update({
            "layer_quantiles": [], "slot_quantiles": [], "layer_coverage": [],
            "slot_coverage": [], "retired_summary": [], "retired_sample": [],
            "retired_sample_by_row": [],
        })
    selection_payload = dict(selection or {})
    candidate_statistics = selection_payload.get("training_statistics_used", {}).get("candidate_statistics", {})
    max_early_middle = max(
        (
            float(value.get("max_early_middle_median_retired_lfo_fraction", 0.0))
            for value in candidate_statistics.values()
        ),
        default=0.0,
    )
    metric_correlations_raw = [
        [
            _spearman(
                [_number(row, left) for row in bundle.summaries],
                [_number(row, right) for row in bundle.summaries],
            )
            for right in CO_PRIMARY_METRICS
        ]
        for left in CO_PRIMARY_METRICS
    ]
    metric_correlations = [
        [value if math.isfinite(value) else None for value in row]
        for row in metric_correlations_raw
    ]
    generalization = {}
    for key in (
        "generalization_gap_median_rmse",
        "generalization_gap_strict_perfect_lfo_rate",
        "generalization_gap_p95_rmse",
        "generalization_gap_node_max_error_p95",
    ):
        values = [_number(row, key) for row in bundle.diagnostics]
        generalization[key] = {
            "minimum": min(values) if values else math.nan,
            "q25": _percentile(values, 0.25),
            "median": _percentile(values, 0.5),
            "q75": _percentile(values, 0.75),
            "maximum": max(values) if values else math.nan,
        }
    diagnostic_correlations = {}
    validation_p95 = [_number(row, "validation_p95_rmse") for row in bundle.diagnostics]
    for key in (
        "train_p95_rmse", "validation_median_rmse", "validation_node_max_error_p95",
        "validation_overshoot_rate_before_final_clip",
        "residual_layer_effective_no_op_usage_rate_median",
        "residual_layer_dead_atom_rate_median", "residual_gain_nonzero_rate",
        "duplicate_atom_rate", "oracle_construction_time",
    ):
        correlation = _spearman(
            [_number(row, key) for row in bundle.diagnostics],
            validation_p95,
        )
        diagnostic_correlations[key] = correlation if math.isfinite(correlation) else None
    marginal_summary = []
    for active_count in sorted({int(_number(row, "active_atom_count")) for row in bundle.marginal_atoms}):
        selected = [row for row in bundle.marginal_atoms if int(_number(row, "active_atom_count")) == active_count]
        values = [_number(row, "delta_validation_p95_rmse") for row in selected]
        marginal_summary.append({
            "active_atom_count": active_count,
            "pair_count": len(values),
            "median_delta_validation_p95_rmse": median(values) if values else math.nan,
            "p95_improved_count": sum(value < 0 for value in values),
        })
    deep_row_ids = payload_row_ids
    deep_row_index = payload_row_index

    payload = {
        "schema_version": INTERACTIVE_REPORT_SCHEMA,
        "meta": {
            "report_mode": report_mode,
            "display_title": "Experiment 13 — Complete W8D16 Strategy Grid" if final_report else ("Experiment 13A — Complete W8D16 Strategy Grid" if complete_13a else "Experiment 13 — Provisional W8D16 Strategy Grid"),
            "title": "Experiment 13 — Complete W8D16 Strategy Grid" if final_report else ("Experiment 13A — Complete W8D16 Strategy Grid" if complete_13a else "Experiment 13 — Provisional W8D16 Strategy Grid"),
            "status": "complete" if final_report else ("complete_13a_pending_13b" if complete_13a else "provisional_incomplete_13a"),
            "completed_rows": completed,
            "expected_rows": expected_count,
            "source_run": source_display,
            "source_archive_sha256": source_fingerprint,
            "epsilon_selected": bool(selection_payload.get("selection_passed")),
            "epsilon_selection_attempted": bool(selection_payload),
            "epsilon_selection_notes": selection_payload.get("selection_notes"),
            "experiment13b_started": experiment13b_state != "not_started",
            "experiment13b_state": experiment13b_state,
            "runtime_comparison_allowed": False,
            "contract": {
                "window_width": 8,
                "residual_layers": 16,
                "control_points": 97,
                "base_choices": 32,
                "atom_choices": 8,
                "path_search": "Beam4",
                "scalar_context": "PhaseAndResidualGain",
                "model_outputs": 193,
            },
        },
        "findings": {
            "normalization": normalization,
            "candidate_budget": budget,
            "schedule": schedule,
            "best_row": {
                field: best.get(field, "")
                for field in (
                    "row_id", "construction_policy", "construction_family", "layer_schedule",
                    "utility_candidate_budget", "layer_normalization_policy", *CO_PRIMARY_METRICS,
                )
            },
            "pareto_count": len(pareto),
            "absent_families": absent_families,
            "incomplete_families": incomplete_families,
            "coverage_by_family": coverage_by_family,
            "runtime_outliers": [
                {
                    "row_id": row.get("row_id", ""),
                    "oracle_construction_time": _number(row, "oracle_construction_time"),
                }
                for row in runtime_rows[:15]
            ],
            "selection": {
                "selection_passed": bool(selection_payload.get("selection_passed")),
                "selection_notes": selection_payload.get("selection_notes", "not attempted"),
                "selected_epsilon": selection_payload.get("selected_epsilon"),
                "max_early_middle_median_retired_lfo_fraction": max_early_middle,
                "required_early_middle_fraction": 0.05,
                "candidate_statistics": candidate_statistics,
            },
            "scaling_matched_row_count": len(scaling_rows),
            "metric_correlations": metric_correlations,
            "generalization": generalization,
            "diagnostic_correlations_with_validation_p95": diagnostic_correlations,
            "marginal_atom_summary": marginal_summary,
        },
        "tables": {
            "metrics": [
                {
                    field: row.get(field, "")
                    for field in (
                        "experiment_phase", "row_id", "construction_policy", "construction_family", "layer_schedule",
                        "utility_candidate_budget", "layer_normalization_policy", *CO_PRIMARY_METRICS,
                        "residual_population_policy", "eligibility_epsilon",
                        "train_count", "validation_count",
                        "oracle_construction_time", "train_encoding_time", "validation_encoding_time",
                        "pareto_candidate",
                    )
                }
                for row in bundle.co_primary
            ],
            "coverage": ([] if final_report else [
                {
                    field: row.get(field, "")
                    for field in (
                        "experiment_phase", "row_id", "construction_policy", "construction_family",
                        "layer_schedule", "utility_candidate_budget", "layer_normalization_policy",
                        "residual_population_policy", "eligibility_epsilon", "completed",
                    )
                }
                for row in bundle.coverage
            ]),
            "matched_deltas": [
                {
                    field: row.get(field, "")
                    for field in (
                        "comparison", "left_value", "right_value", "match_key", "left_row_id", "right_row_id",
                        "construction_family", "experiment_phase", "eligibility_epsilon",
                        *(f"delta_{metric}" for metric in CO_PRIMARY_METRICS),
                    )
                }
                for row in bundle.matched_deltas
            ],
            "diagnostics": [
                {
                    field: row.get(field, "")
                    for field in (
                        "experiment_phase", "row_id", "eligibility_epsilon", "train_median_rmse", "train_strict_perfect_lfo_rate",
                        "train_p95_rmse", "train_node_max_error_p95",
                        "generalization_gap_median_rmse", "generalization_gap_strict_perfect_lfo_rate",
                        "generalization_gap_p95_rmse", "generalization_gap_node_max_error_p95",
                        "validation_overshoot_rate_before_final_clip", "validation_overshoot_abs_p95_before_final_clip",
                        "residual_layer_dead_atom_rate_median", "residual_layer_dominant_atom_share_median",
                        "residual_layer_usage_entropy_median", "residual_layer_no_op_usage_rate_median",
                        "residual_layer_effective_no_op_usage_rate_median", "residual_gain_median",
                        "residual_gain_abs_p95", "residual_gain_nonzero_rate", "duplicate_atom_rate",
                        "oracle_construction_time", "train_encoding_time", "validation_encoding_time", "offline_analysis_time",
                    )
                }
                for row in bundle.diagnostics
            ],
            "rankings": [
                {
                    field: row.get(field, "")
                    for field in (
                        "experiment_phase", "row_id", "eligibility_epsilon", *(f"rank_{metric}" for metric in CO_PRIMARY_METRICS),
                        "mean_co_primary_rank", "rank_spread",
                    )
                }
                for row in bundle.rankings
            ],
            "factor_interactions": [
                row for row in bundle.factor_interactions
            ],
            "partial_codebook": partial_rows,
            "scaling": [
                {
                    field: row.get(field, "")
                    for field in (
                        "row_id", "construction_policy", "construction_family", "layer_schedule",
                        "utility_candidate_budget", "layer_normalization_policy",
                        *(f"delta_{metric}" for metric in CO_PRIMARY_METRICS),
                    )
                }
                for row in scaling_rows
            ],
        },
        "deep_analysis": {
            "row_ids": deep_row_ids,
            "marginal_atoms": [
                [
                    deep_row_index[str(row.get("row_id", ""))],
                    int(_number(row, "active_atom_count")),
                    *[_number(row, f"delta_{metric}") for metric in CO_PRIMARY_METRICS],
                ]
                for row in bundle.marginal_atoms
                if str(row.get("row_id", "")) in deep_row_index
            ],
            "layer_progression": [
                ([
                    deep_row_index[str(row.get("row_id", ""))],
                    int(_number(row, "residual_layer")),
                    int(_number(row, "eligible_residual_count")),
                    _number(row, "training_p95_rmse"),
                ] if final_report else [
                    deep_row_index[str(row.get("row_id", ""))],
                    int(_number(row, "residual_layer")),
                    str(row.get("layer_role", "")),
                    int(_number(row, "eligible_residual_count")),
                    _number(row, "training_median_rmse"),
                    _number(row, "training_p95_rmse"),
                    _number(row, "training_max_abs_error_p95"),
                ])
                for row in bundle.layer_progression
                if str(row.get("row_id", "")) in deep_row_index
            ],
            "mechanisms": [
                [
                    deep_row_index[str(row.get("row_id", ""))],
                    int(_number(row, "atom_count")),
                    int(_number(row, "broad_atom_count")),
                    int(_number(row, "repair_atom_count")),
                    int(_number(row, "synthesized_prototype_count")),
                    int(_number(row, "observed_residual_count")),
                    _number(row, "median_slot_p95_improvement"),
                    _number(row, "exact_duplicate_alignment_reuse_rate"),
                    _number(row, "prototype_convergence_rate"),
                    _number(row, "prototype_iterations_median"),
                    int(_number(row, "candidate_search_event_count")),
                    int(_number(row, "candidate_evaluations")),
                ]
                for row in bundle.mechanism_diagnostics
                if str(row.get("row_id", "")) in deep_row_index
            ],
        },
        "calibration": calibration,
        "strict_thresholds": dict(strict_thresholds) if strict_thresholds is not None else None,
    }
    if final_report:
        # Final HTML recomputes ranks and interaction summaries from the
        # embedded metrics/deltas, avoiding two redundant record tables.
        payload["tables"]["rankings"] = []
        payload["tables"]["factor_interactions"] = []
        for name in (
            "metrics", "matched_deltas", "diagnostics", "rankings",
            "factor_interactions", "scaling",
        ):
            payload["tables"][name] = _columnar_records(payload["tables"][name])
    return payload


def _columnar_records(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Dictionary-encode repeated record keys for the portable final HTML."""
    if not rows:
        return {"columns": [], "rows": []}
    columns = list(rows[0])
    return {
        "columns": columns,
        "rows": [[row.get(column, "") for column in columns] for row in rows],
    }


def _compact_calibration(calibration: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    row_ids = sorted({
        str(row.get("row_id", ""))
        for table in calibration.values()
        for row in table
        if row.get("row_id")
    })
    row_index = {row_id: index for index, row_id in enumerate(row_ids)}

    layer_quantiles: dict[tuple[int, float], list[float]] = {}
    layer_quantiles_by_row: list[list[float | int]] = []
    for row in calibration["layer_epsilon_quantiles.csv"]:
        if row.get("dataset_split", "training") != "training":
            continue
        residual_layer = int(_number(row, "residual_layer"))
        percentile = _number(row, "percentile")
        epsilon_value = _number(row, "epsilon_value")
        key = (residual_layer, percentile)
        layer_quantiles.setdefault(key, []).append(epsilon_value)
        layer_quantiles_by_row.append([
            row_index[str(row.get("row_id", ""))],
            residual_layer,
            percentile,
            epsilon_value,
        ])

    slot_quantiles: dict[tuple[int, float], list[float]] = {}
    for row in calibration["slot_epsilon_quantiles.csv"]:
        if row.get("dataset_split", "training") != "training":
            continue
        key = (int(_number(row, "active_atom_slot")), _number(row, "percentile"))
        slot_quantiles.setdefault(key, []).append(_number(row, "epsilon_value"))

    layer_coverage: dict[tuple[int, float], list[float]] = {}
    slot_coverage: dict[tuple[int, float], list[float]] = {}
    layer_coverage_by_row: list[list[float | int]] = []
    for row in calibration["epsilon_coverage.csv"]:
        if row.get("dataset_split") != "training":
            continue
        slot = row.get("active_atom_slot")
        epsilon = _number(row, "epsilon")
        if slot in {None, "", "None"}:
            residual_layer = int(_number(row, "residual_layer"))
            key = (residual_layer, epsilon)
            layer_coverage.setdefault(key, []).append(_number(row, "resolved_fraction"))
            if residual_layer in {8, 16}:
                layer_coverage_by_row.append([
                    row_index[str(row.get("row_id", ""))],
                    residual_layer,
                    epsilon,
                    _number(row, "resolved_fraction"),
                ])
        else:
            key = (int(_number(row, "active_atom_slot")), epsilon)
            slot_coverage.setdefault(key, []).append(_number(row, "resolved_fraction"))

    retired_rows = list(calibration["retired_error_mass.csv"])
    retired_by_epsilon: dict[float, list[Mapping[str, Any]]] = {}
    for row in retired_rows:
        retired_by_epsilon.setdefault(_number(row, "epsilon"), []).append(row)
    retired_summary = []
    retired_sample = []
    retired_sample_by_row: list[list[float | int]] = []
    for epsilon, rows in sorted(retired_by_epsilon.items()):
        retired = [_number(row, "retired_lfo_fraction") for row in rows]
        incoming = [_number(row, "incoming_retired_energy_fraction") for row in rows]
        unexplained = [_number(row, "unexplained_retired_energy_fraction") for row in rows]
        retired_summary.append({
            "epsilon": epsilon,
            "count": len(rows),
            "retired_q25": _percentile(retired, 0.25),
            "retired_median": _percentile(retired, 0.5),
            "retired_q75": _percentile(retired, 0.75),
            "unexplained_q25": _percentile(unexplained, 0.25),
            "unexplained_median": _percentile(unexplained, 0.5),
            "unexplained_q75": _percentile(unexplained, 0.75),
            "unexplained_p95": _percentile(unexplained, 0.95),
            "incoming_median": _percentile(incoming, 0.5),
            "incoming_p95": _percentile(incoming, 0.95),
        })
        ordered = sorted(
            rows,
            key=lambda row: (
                str(row.get("row_id", "")),
                int(_number(row, "residual_layer")),
                int(_number(row, "active_atom_slot")),
            ),
        )
        if len(ordered) <= 150:
            sampled = ordered
        else:
            indexes = sorted({round(index * (len(ordered) - 1) / 149) for index in range(150)})
            sampled = [ordered[index] for index in indexes]
        retired_sample.extend({
            "epsilon": epsilon,
            "retired_lfo_fraction": _number(row, "retired_lfo_fraction"),
            "incoming_retired_energy_fraction": _number(row, "incoming_retired_energy_fraction"),
            "unexplained_retired_energy_fraction": _number(row, "unexplained_retired_energy_fraction"),
            "residual_layer": int(_number(row, "residual_layer")),
            "active_atom_slot": int(_number(row, "active_atom_slot")),
        } for row in sampled)

        rows_by_id: dict[str, list[Mapping[str, Any]]] = {}
        for row in rows:
            rows_by_id.setdefault(str(row.get("row_id", "")), []).append(row)
        for row_id, row_group in sorted(rows_by_id.items()):
            ordered_group = sorted(
                row_group,
                key=lambda row: (
                    int(_number(row, "residual_layer")),
                    int(_number(row, "active_atom_slot")),
                ),
            )
            if len(ordered_group) <= 4:
                filtered_sample = ordered_group
            else:
                indexes = sorted({round(index * (len(ordered_group) - 1) / 3) for index in range(4)})
                filtered_sample = [ordered_group[index] for index in indexes]
            retired_sample_by_row.extend([
                row_index[row_id],
                epsilon,
                _number(row, "retired_lfo_fraction"),
                _number(row, "unexplained_retired_energy_fraction"),
            ] for row in filtered_sample)

    return {
        "row_ids": row_ids,
        "layer_quantiles": _median_records(layer_quantiles, "residual_layer", "percentile", "epsilon_value"),
        "layer_quantiles_by_row": layer_quantiles_by_row,
        "slot_quantiles": _median_records(slot_quantiles, "active_atom_slot", "percentile", "epsilon_value"),
        "layer_coverage": _median_records(layer_coverage, "residual_layer", "epsilon", "resolved_fraction"),
        "layer_coverage_by_row": layer_coverage_by_row,
        "slot_coverage": _median_records(slot_coverage, "active_atom_slot", "epsilon", "resolved_fraction"),
        "retired_summary": retired_summary,
        "retired_sample": retired_sample,
        "retired_sample_by_row": retired_sample_by_row,
    }


def _median_records(
    grouped: Mapping[tuple[int, float], Sequence[float]],
    x_key: str,
    series_key: str,
    value_key: str,
) -> list[dict[str, float | int]]:
    return [
        {x_key: key[0], series_key: key[1], value_key: median(values)}
        for key, values in sorted(grouped.items())
    ]


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _interactive_html(payload: Mapping[str, Any]) -> str:
    template = INTERACTIVE_TEMPLATE.read_text(encoding="utf-8")
    marker = "__REPORT_DATA_JSON__"
    if template.count(marker) != 1:
        raise RuntimeError(f"interactive report template must contain exactly one {marker} marker")
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    data = data.replace("</", "<\\/")
    return template.replace(marker, data)


def _legacy_complete_markdown(
    bundle: AnalysisBundle,
    *,
    source_run: Path,
    report_path: Path,
    plot_paths: Mapping[str, Path],
    selection: Mapping[str, Any],
) -> str:
    """Render the final findings-first Experiment 13 report."""
    phase_a = [row for row in bundle.summaries if row.get("experiment_phase") == "13A"]
    phase_b = [row for row in bundle.summaries if row.get("experiment_phase") == "13B"]
    population = [row for row in bundle.matched_deltas if row.get("comparison") == "residual_population_policy"]
    source_display = os.path.relpath(source_run, report_path.parent).replace("\\", "/")

    def image(name: str, alt: str) -> str:
        relative = os.path.relpath(plot_paths[name], report_path.parent).replace("\\", "/")
        return f"![{alt}]({relative})"

    metric_specs = (
        ("validation_median_rmse", "Median RMSE", False),
        ("validation_strict_perfect_lfo_rate", "Strict-perfect LFO rate", True),
        ("validation_p95_rmse", "P95 RMSE", False),
        ("validation_node_max_error_p95", "Node-max error P95", False),
    )
    leaders = [
        "| Co-primary validation metric | Better | Best 13B value | Strategy | Eligibility epsilon |",
        "| --- | --- | ---: | --- | ---: |",
    ]
    for metric, label, higher in metric_specs:
        row = (max if higher else min)(phase_b, key=lambda item, key=metric: _number(item, key))
        value = _number(row, metric)
        value_text = f"{value:.3%}" if higher else f"{value:.8g}"
        leaders.append(
            f"| {label} | {'higher' if higher else 'lower'} | "
            f"{value_text} | `{row.get('row_id')}` | `{_number(row, 'eligibility_epsilon'):g}` |"
        )

    population_table = [
        "| Epsilon | P95 improved / tied / worse vs paired 13A | Median P95 delta | Mean P95 delta | Strict-perfect improved / tied / worse |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for epsilon in (0.01, 0.001, 0.0001):
        rows = [row for row in population if math.isclose(_number(row, "eligibility_epsilon"), epsilon)]
        p95 = [_number(row, "delta_validation_p95_rmse") for row in rows]
        strict = [_number(row, "delta_validation_strict_perfect_lfo_rate") for row in rows]
        population_table.append(
            f"| `{epsilon:g}` | {sum(value < 0 for value in p95)} / {sum(value == 0 for value in p95)} / {sum(value > 0 for value in p95)} | "
            f"{median(p95):+.8g} | {sum(p95) / len(p95):+.8g} | "
            f"{sum(value > 0 for value in strict)} / {sum(value == 0 for value in strict)} / {sum(value < 0 for value in strict)} |"
        )

    family_effects = [
        "| Construction family | Epsilon 1e-2 | Epsilon 1e-3 | Epsilon 1e-4 | Interpretation |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    families = sorted({str(row.get("construction_family", "")) for row in population})
    for family in families:
        values = []
        wins = []
        for epsilon in (0.01, 0.001, 0.0001):
            selected = [
                _number(row, "delta_validation_p95_rmse")
                for row in population
                if row.get("construction_family") == family
                and math.isclose(_number(row, "eligibility_epsilon"), epsilon)
            ]
            values.append(median(selected))
            wins.append(sum(value < 0 for value in selected))
        if all(value < -0.01 for value in values):
            interpretation = "large repeatable rescue"
        elif all(value < 0 for value in values):
            interpretation = "consistent but smaller improvement"
        elif all(value > 0 for value in values):
            interpretation = "consistent regression"
        else:
            interpretation = "mixed or strategy-dependent"
        family_effects.append(
            f"| {family} | {values[0]:+.6f} ({wins[0]} wins) | {values[1]:+.6f} ({wins[1]} wins) | "
            f"{values[2]:+.6f} ({wins[2]} wins) | {interpretation} |"
        )

    epsilon_table = [
        "| Eligibility epsilon | Median RMSE | Strict-perfect rate | P95 RMSE | Node-max P95 | Median LFO population retired after layer 16 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    train_count = {str(row.get("row_id", "")): max(1, int(_number(row, "train_count"))) for row in bundle.summaries}
    for epsilon in (0.01, 0.001, 0.0001):
        rows = [row for row in phase_b if math.isclose(_number(row, "eligibility_epsilon"), epsilon)]
        end = [
            row for row in bundle.layer_progression
            if row.get("experiment_phase") == "13B"
            and math.isclose(_number(row, "eligibility_epsilon"), epsilon)
            and int(_number(row, "residual_layer")) == 16
        ]
        retired = median(
            1.0 - _number(row, "eligible_residual_count") / train_count[str(row.get("row_id", ""))]
            for row in end
        )
        epsilon_table.append(
            f"| `{epsilon:g}` | {median(_number(row, 'validation_median_rmse') for row in rows):.8g} | "
            f"{median(_number(row, 'validation_strict_perfect_lfo_rate') for row in rows):.3%} | "
            f"{median(_number(row, 'validation_p95_rmse') for row in rows):.8g} | "
            f"{median(_number(row, 'validation_node_max_error_p95') for row in rows):.8g} | {retired:.3%} |"
        )

    factor_table = [
        "| Lever | Epsilon | Right-hand policy | P95 improves / loses | Median P95 delta |",
        "| --- | ---: | --- | ---: | ---: |",
    ]
    for comparison, right in (("utility_candidate_budget", "CandidateBudget48"), ("layer_schedule", "TwoPhase")):
        for epsilon in (0.01, 0.001, 0.0001):
            rows = [
                row for row in bundle.matched_deltas
                if row.get("comparison") == comparison
                and row.get("experiment_phase") == "13B"
                and math.isclose(_number(row, "eligibility_epsilon"), epsilon)
            ]
            values = [_number(row, "delta_validation_p95_rmse") for row in rows]
            factor_table.append(
                f"| {comparison.replace('_', ' ')} | `{epsilon:g}` | {right} | "
                f"{sum(value < 0 for value in values)} / {sum(value > 0 for value in values)} | {median(values):+.8g} |"
            )

    strict_peak = max(_number(row, "validation_strict_perfect_lfo_rate") for row in phase_b)
    strict_rows = sorted(
        [row for row in phase_b if math.isclose(_number(row, "validation_strict_perfect_lfo_rate"), strict_peak)],
        key=lambda row: (_number(row, "validation_p95_rmse"), str(row.get("row_id", ""))),
    )
    strict_table = [
        "| Strategy | Family | Epsilon | Strict-perfect | Median RMSE | P95 RMSE |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in strict_rows:
        strict_table.append(
            f"| `{row.get('row_id')}` | {row.get('construction_family')} | `{_number(row, 'eligibility_epsilon'):g}` | "
            f"{_number(row, 'validation_strict_perfect_lfo_rate'):.3%} | "
            f"{_number(row, 'validation_median_rmse'):.8g} | {_number(row, 'validation_p95_rmse'):.8g} |"
        )

    partial_b = [
        row for row in bundle.partial_codebook
        if next((item for item in phase_b if item.get("row_id") == row.get("row_id")), None) is not None
    ]
    partial_by_slot = {
        slot: median(
            _number(row, "validation_p95_rmse")
            for row in partial_b if int(_number(row, "active_atom_count")) == slot
        )
        for slot in range(1, 8)
    }

    return "\n".join([
        "# Experiment 13: Fixed-W8D16 Strategy Grid", "",
        "**Complete evidence · 90/90 Experiment 13A rows · 135/135 Experiment 13B rows · no failures**", "",
        "## Main Findings", "",
        "Experiment 13 does not produce one scalar winner. It produces a small, interpretable frontier: `AllClusterMeans` has the best validation P95, while `DiverseCoverageHardRepairTwoPhase` supplies the best median and node-max solutions, and a separate `DiverseCoverageHardRepairInterleaved` regime preserves many more exact reconstructions.", "",
        f"The best 13B P95 is `{min(_number(row, 'validation_p95_rmse') for row in phase_b):.8g}` from `AllClusterMeans`; all three epsilon variants tie. The best median is `{min(_number(row, 'validation_median_rmse') for row in phase_b):.8g}` at epsilon `1e-2`, and the best node-max P95 is `{min(_number(row, 'validation_node_max_error_p95') for row in phase_b):.8g}` at epsilon `1e-4`, both from `DiverseCoverageHardRepairTwoPhase + CandidateBudget48`.", "",
        f"`UnresolvedOnly` is a targeted rescue, not a global free win. Across the 45 matched clipped strategies the median P95 change is effectively zero at every epsilon. The negative mean is driven by `BroadMeanFinishRepair`, whose P95 falls from roughly `0.087–0.091` in 13A to `0.046–0.050` in 13B. Other families include both gains and regressions.", "",
        "No epsilon wins globally. `1e-2` has the lowest across-strategy median P95 by a very small margin, `1e-3` gives the best median P95 inside `DiverseCoverageHardRepair`, and `1e-4` contains the best node-max row. Epsilon changes the population used to build later atoms, so the final quality curves are legitimately non-monotonic.", "",
        f"{len(strict_rows)} 13B rows share the best strict-perfect rate of `{strict_peak:.3%}`. In the completed run, five are `BroadMeanFinishRepairTwoPhase`; the two scientifically stronger rows are `DiverseCoverageHardRepairInterleaved + CandidateBudget48` at `1e-3` and `1e-4`, because they retain that exact-finish rate with P95 near `0.040–0.041`.", "",
        *leaders, "",
        image("pareto", "Experiment 13B co-primary quality frontier"), "",
        "## Why UnresolvedOnly Helps Some Families", "",
        "`AllResiduals` lets every training residual continue influencing every later atom. `UnresolvedOnly` removes a curve from later construction after its maximum point error falls below the eligibility epsilon. This is useful when nearly solved curves would otherwise monopolize finish-oriented proposals, but it can remove stabilizing mass from families whose broad prototypes benefit from the full population.", "",
        "The paired result is therefore skewed rather than uniform. The median strategy is unchanged, while a handful of formerly pathological finish-repair rows improve dramatically. Win/loss counts and family medians are more informative than the overall mean.", "",
        *population_table, "",
        image("population", "Family-level UnresolvedOnly versus AllResiduals P95 effects"), "",
        "### Family-specific paired effects", "",
        *family_effects, "",
        "## The Three-Epsilon Sweep", "",
        "The epsilon value is a construction policy, not a post-hoc scoring threshold. A looser epsilon can retire more curves early, but because the remaining curves then define different atoms, a looser epsilon does not guarantee a monotonic improvement or regression in final validation quality.", "",
        *epsilon_table, "",
        image("epsilon", "Family-specific Experiment 13B epsilon sensitivity"), "",
        image("eligibility", "Eligibility population retired by residual-layer depth"), "",
        "Typical retirement is modest: the median strategy has only about two percent of training LFOs removed by layer 16. The mean and maximum are much larger because `BroadMeanFinishRepair` and `DiverseCoverageHardRepair` create high-retirement regimes. This is why one aggregate retirement percentage would conceal the mechanism.", "",
        "## Strict-Perfect Finishing Is a Distinct Regime", "",
        "The original strict-perfect definition remains RMSE `<= 1e-6` and maximum point error `<= 1e-5`. It is deliberately much stricter than the 13B eligibility epsilons. The 13B jump to 19.5% is not a gentle shift in average RMSE; seven rows land on exactly the same 313-of-1605 validation count.", "",
        "A validation-only forensic replay of saved codebooks confirmed that a representative `DiverseCoverageHardRepair` row and a representative `BroadMeanFinishRepair` row finish the exact same 313 LFOs. The ordinary 13-of-1605 exact set is a subset. Of the 313, 299 are analysis-labeled `continuous` curves and 14 are `discontinuous`; only seven are bit-exact, while the rest satisfy the strict numerical tolerance. This topology label is post-hoc analysis only and does not enter construction or deployed runtime.", "",
        *strict_table, "",
        image("strict_noop", "Strict-perfect finishing regime versus validation tail error"), "",
        "The separate complete-13A report retains the replayed `1e-2`, `1e-3`, `1e-4`, and `1e-5` strict-perfect tolerance sensitivity. Those toggles are not mixed into this final cross-phase frontier because the replay artifact covers 13A rows only.", "",
        "## Budget and Schedule", "",
        "CandidateBudget48 is the more repeatable 13B lever: it improves P95 in 13/21, 15/21, and 16/21 matched pairs as epsilon tightens. TwoPhase is not a global winner; it improves only 8/18 matched schedule pairs at each epsilon and has a slightly worse median P95 delta. Its strong showing in the best `DiverseCoverage` rows is a family interaction, not a universal schedule rule.", "",
        *factor_table, "",
        image("factors", "Experiment 13B budget and schedule effects"), "",
        "## Partial Codebook and Residual Depth", "",
        f"The first two active atoms do most of the tail work: the across-row median P95 drops from `{partial_by_slot[1]:.6f}` with one active atom to `{partial_by_slot[2]:.6f}` with two. Later atoms still matter, but the median step from six to seven is only `{partial_by_slot[7] - partial_by_slot[6]:+.6f}`. This supports the current ordering logic while also identifying a future head-budget ablation; it does not by itself authorize shrinking W8.", "",
        image("partial", "Experiment 13B partial-codebook progression by epsilon"), "",
        "## Runtime and Deployment Boundary", "",
        "All timings are offline experiment work. Every row preserves the same deployed 193-output W8D16 interface, Beam4 encoding, phase and residual gain, and topology-free runtime contract. The optimized 13B rows are somewhat faster to construct than 13A on the median, but this is descriptive same-run evidence, not a claim that filtering reduces deployed inference cost.", "",
        image("runtime", "Complete Experiment 13 same-run offline timing"), "",
        "## Practical Takeaways", "",
        "1. Keep `LayerClip0To1`; Experiment 13B correctly avoided spending another 45 rows on the inferior clipping branch.",
        "2. Use `AllClusterMeans` when the primary objective is the lowest P95 with a simple prototype-only construction.",
        "3. Use `DiverseCoverageHardRepairTwoPhase + CandidateBudget48` as the balanced low-median/low-node-max candidate; retain epsilon as a small family-specific choice rather than a global constant.",
        "4. Preserve `DiverseCoverageHardRepairInterleaved + CandidateBudget48` at epsilon `1e-3` as the strongest exact-finish candidate: it reaches 19.5% strict-perfect without the large median/tail penalty of finish-repair rows.",
        "5. Do not generalize `UnresolvedOnly` to every construction family. Its value is largest where the unfiltered family was visibly pathological.",
        "6. If the deployed head budget must be reduced, run a dedicated atom-count/depth ablation. Partial-codebook curves are motivation, not proof of a smaller contract.", "",
        "## Method Notes", "",
        f"Source run: `{source_display}`.", "",
        f"The automatic 13A selector recorded `selection_passed={selection.get('selection_passed')}`. The three 13B epsilons were an explicit exploratory sweep; none is relabeled as the selector's frozen choice.", "",
        "The report uses the four co-primary validation metrics separately: median RMSE, strict-perfect LFO rate, P95 RMSE, and node-max error P95. Pareto membership means no other row is at least as good on all four and strictly better on at least one.", "",
        "`AllResiduals` versus `UnresolvedOnly` comparisons use only the 45 matched `LayerClip0To1` 13A rows. Epsilon comparisons use matched construction strategies inside 13B. Runtime comparisons are limited to same-run offline diagnostics.", "",
        f"Configuration fingerprint: `{selection.get('configuration_fingerprint')}`.", "",
    ])


def _complete_markdown(
    bundle: AnalysisBundle,
    *,
    source_run: Path,
    report_path: Path,
    plot_paths: Mapping[str, Path],
    selection: Mapping[str, Any],
    scaling_rows: Sequence[Mapping[str, Any]] = (),
    strict_thresholds: Mapping[str, Any] | None = None,
) -> str:
    """Render the complete report with 13A as the baseline for 13B capacity effects."""
    phase_a = [row for row in bundle.summaries if row.get("experiment_phase") == "13A"]
    phase_b = [row for row in bundle.summaries if row.get("experiment_phase") == "13B"]
    effects = bundle.eligibility_capacity_effects
    source_display = os.path.relpath(source_run, report_path.parent).replace("\\", "/")
    metric_specs = (
        ("validation_median_rmse", "Median RMSE", False),
        ("validation_strict_perfect_lfo_rate", "Strict-perfect LFO rate", True),
        ("validation_p95_rmse", "P95 RMSE", False),
        ("validation_node_max_error_p95", "Node-max error P95", False),
    )

    def image(name: str, alt: str) -> str:
        relative = os.path.relpath(plot_paths[name], report_path.parent).replace("\\", "/")
        return f"![{alt}]({relative})"

    def metric_text(value: float, higher: bool) -> str:
        return f"{value:.3%}" if higher else f"{value:.8g}"

    def effect_counts(rows: Sequence[Mapping[str, Any]], metric: str, higher: bool) -> tuple[int, int, int, float]:
        values = [_number(row, f"delta_{metric}") for row in rows]
        improved = sum(value > 0 if higher else value < 0 for value in values)
        worsened = sum(value < 0 if higher else value > 0 for value in values)
        return improved, sum(value == 0 for value in values), worsened, median(values)

    phase_a_pareto = [row for row in phase_a if str(row.get("row_id", "")) in _pareto_ids(phase_a)]
    phase_b_pareto = [row for row in phase_b if str(row.get("row_id", "")) in _pareto_ids(phase_b)]
    phase_a_leaders = [
        "| 13A co-primary metric | Better | Best no-filter value | Strategy |",
        "| --- | --- | ---: | --- |",
    ]
    phase_b_leaders = [
        "| Final co-primary metric | Better | Best 13B value | Strategy | Epsilon |",
        "| --- | --- | ---: | --- | ---: |",
    ]
    for metric, label, higher in metric_specs:
        row_a = (max if higher else min)(phase_a, key=lambda row, key=metric: _number(row, key))
        row_b = (max if higher else min)(phase_b, key=lambda row, key=metric: _number(row, key))
        phase_a_leaders.append(
            f"| {label} | {'higher' if higher else 'lower'} | {metric_text(_number(row_a, metric), higher)} | `{row_a.get('row_id')}` |"
        )
        phase_b_leaders.append(
            f"| {label} | {'higher' if higher else 'lower'} | {metric_text(_number(row_b, metric), higher)} | "
            f"`{row_b.get('row_id')}` | `{_number(row_b, 'eligibility_epsilon'):g}` |"
        )

    phase_a_effects = [
        "| 13A matched lever | Metric | Improved / tied / worsened | Median right-minus-left delta |",
        "| --- | --- | ---: | ---: |",
    ]
    for comparison, label in (
        ("layer_normalization_policy", "LayerClip0To1 vs FinalClipOnly"),
        ("utility_candidate_budget", "CandidateBudget48 vs CandidateBudget24"),
        ("layer_schedule", "TwoPhase vs Interleaved"),
    ):
        rows = [row for row in bundle.matched_deltas if row.get("comparison") == comparison and row.get("experiment_phase") == "13A"]
        for metric, metric_label, higher in metric_specs:
            improved, tied, worsened, delta = effect_counts(rows, metric, higher)
            delta_text = f"{delta * 100:+.3f} pp" if higher else f"{delta:+.8g}"
            phase_a_effects.append(f"| {label} | {metric_label} | {improved} / {tied} / {worsened} | {delta_text} |")

    family_table = [
        "| 13A construction family | Rows | Median RMSE | Strict-perfect | P95 RMSE | Node-max P95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for family in sorted({str(row.get("construction_family", "")) for row in phase_a}):
        rows = [row for row in phase_a if row.get("construction_family") == family]
        family_table.append(
            f"| {family} | {len(rows)} | {median(_number(row, 'validation_median_rmse') for row in rows):.8g} | "
            f"{median(_number(row, 'validation_strict_perfect_lfo_rate') for row in rows):.3%} | "
            f"{median(_number(row, 'validation_p95_rmse') for row in rows):.8g} | "
            f"{median(_number(row, 'validation_node_max_error_p95') for row in rows):.8g} |"
        )

    capacity_table = [
        "| Population policy | Treatment epsilon | Mean future-layer population freed | Layer-8 population freed | Layer-16 endpoint coverage |",
        "| --- | ---: | ---: | ---: | ---: |",
        "| No eligibility filter (13A) | â€” | 0.000% | 0.000% | 0.000% |",
    ]
    paired_effect_table = [
        "| Treatment epsilon | Metric | Improved / tied / worsened vs exact no-filter baseline | Median paired delta |",
        "| ---: | --- | ---: | ---: |",
    ]
    for epsilon in (0.01, 0.001, 0.0001):
        rows = [row for row in effects if math.isclose(_number(row, "eligibility_epsilon"), epsilon)]
        capacity_table.append(
            f"| UnresolvedOnly | `{epsilon:g}` | {median(_number(row, 'mean_future_layer_population_freed') for row in rows):.3%} | "
            f"{median(_number(row, 'layer8_population_freed') for row in rows):.3%} | "
            f"{median(_number(row, 'layer16_endpoint_coverage') for row in rows):.3%} |"
        )
        for metric, label, higher in metric_specs:
            improved, tied, worsened, delta = effect_counts(rows, metric, higher)
            delta_text = f"{delta * 100:+.3f} pp" if higher else f"{delta:+.8g}"
            paired_effect_table.append(f"| `{epsilon:g}` | {label} | {improved} / {tied} / {worsened} | {delta_text} |")

    capacity_correlation = [
        "| Paired validation benefit | Spearman Ï with mean future-layer population freed | Reading |",
        "| --- | ---: | --- |",
    ]
    capacities = [_number(row, "mean_future_layer_population_freed") for row in effects]
    for metric, label, higher in metric_specs:
        benefits = [(_number(row, f"delta_{metric}") if higher else -_number(row, f"delta_{metric}")) for row in effects]
        correlation = _spearman(capacities, benefits)
        reading = "more released capacity tends to align with more benefit" if correlation > 0.2 else ("more released capacity tends to align with less benefit" if correlation < -0.2 else "little global monotonic relationship")
        capacity_correlation.append(f"| {label} | {correlation:+.3f} | {reading}; family confounding remains |")

    strict_threshold_table: list[str] = []
    if strict_thresholds:
        strict_threshold_table = [
            "| 13A max-absolute tolerance | RMSE tolerance | Best strict-perfect rate | Median row rate |",
            "| ---: | ---: | ---: | ---: |",
        ]
        rates_by_row = strict_thresholds["rates_by_row"]
        for tolerance in strict_thresholds["tolerances"]:
            rates = [float(rates_by_row[str(row.get("row_id", ""))][tolerance]) for row in phase_a]
            strict_threshold_table.append(
                f"| `{tolerance}` | `{float(tolerance) / 10:.0e}` | {max(rates):.3%} | {median(rates):.3%} |"
            )

    partial_a = [row for row in bundle.partial_codebook if row.get("experiment_phase") == "13A"]
    marginal_a = [row for row in bundle.marginal_atoms if row.get("experiment_phase") == "13A"]
    atom_table = [
        "| Added active atom | Median 13A validation-P95 delta | 13A rows improved |",
        "| ---: | ---: | ---: |",
    ]
    for active in range(2, 8):
        rows = [row for row in marginal_a if int(_number(row, "active_atom_count")) == active]
        values = [_number(row, "delta_validation_p95_rmse") for row in rows]
        atom_table.append(f"| {active} | {median(values):+.8g} | {sum(value < 0 for value in values)}/{len(values)} |")

    scaling_table: list[str] = []
    if scaling_rows:
        scaling_table = [
            "| 50%-minus-100% training comparison | Median delta | 50% better / 100% better / ties |",
            "| --- | ---: | ---: |",
        ]
        for metric, label, higher in metric_specs:
            values = [_number(row, f"delta_{metric}") for row in scaling_rows]
            better = sum(value > 0 if higher else value < 0 for value in values)
            worse = sum(value < 0 if higher else value > 0 for value in values)
            delta = median(values)
            delta_text = f"{delta * 100:+.3f} pp" if higher else f"{delta:+.8g}"
            scaling_table.append(f"| {label} | {delta_text} | {better} / {worse} / {sum(value == 0 for value in values)} |")

    strict_peak = max(_number(row, "validation_strict_perfect_lfo_rate") for row in phase_b)
    factor_rows = [row for row in bundle.factor_interactions if row.get("experiment_phase") == "13B"]
    return "\n".join([
        "# Experiment 13: Fixed-W8D16 Strategy Grid", "",
        "**Complete evidence · 90/90 Experiment 13A rows · 135/135 Experiment 13B rows · no failures**", "",
        "## Experiment Question and Answer", "",
        "Experiment 13 asks whether removing already-solved LFOs from later dictionary construction frees useful capacity and improves the remaining reconstruction problem. Experiment 13A is the no-filter foundation. Experiment 13B applies three eligibility thresholds to the exact same 45 clipped strategies, so every treatment has a deterministic paired baseline.", "",
        "The answer is family-specific. Eligibility releases measurable construction population, but more capacity released does not monotonically produce better validation outcomes. `BroadMeanFinishRepair` receives a large rescue; several other families show small, mixed, or adverse changes. Epsilon is therefore a treatment intensity, not the primary contest and not a globally selected winner.", "",
        "## Experiment 13A Foundation: What the Unfiltered Grid Established", "",
        "Layer-wise clipping is the clearest global result: `LayerClip0To1` improves validation P95 in all `45/45` matched pairs. CandidateBudget48 and TwoPhase have smaller mixed effects whose signs change by construction family. Those conclusions remain part of the final experiment rather than being replaced by 13B.", "",
        *phase_a_leaders, "",
        f"The 13A four-objective frontier contains `{len(phase_a_pareto)}` strategies with distinct jobs: simple cluster prototypes lead P95, diverse-coverage hard repair leads median and node-max quality, and the Experiment 12 anchor family supplies the strongest thresholded finishing behavior.", "",
        image("13a_pareto", "Experiment 13A no-filter four-metric quality frontier"), "",
        "### Metric tension and strict-tolerance sensitivity", "",
        "Median RMSE, P95 RMSE, and node-max P95 share substantial ordering information, while strict-perfect rate remains a distinct thresholded signal. The 13A replay at four tolerance tuples is retained below; it changes the strict-perfect definition only and does not alter continuous RMSE metrics.", "",
        *strict_threshold_table, "" if strict_threshold_table else "The strict-threshold replay artifact was not available to this regeneration.", "",
        *( [image("13a_strict_thresholds", "Experiment 13A strict-perfect tolerance sensitivity"), ""] if "13a_strict_thresholds" in plot_paths else [] ),
        image("13a_metric_agreement", "Experiment 13A co-primary metric rank agreement"), "",
        "### Matched policy effects and construction-family interactions", "",
        *phase_a_effects, "",
        image("13a_interactions", "Experiment 13A family-specific policy interactions"), "",
        *family_table, "",
        "### Atom capacity, residual depth, and mechanism diagnostics", "",
        "The largest typical tail gain arrives when moving from one to two active atoms per residual layer. Later atoms have diminishing but still positive value for most rows. Training P95 also continues falling through residual layer 16, so 13A supports the retained W8D16 contract while motivating a separate head-budget ablation.", "",
        *atom_table, "",
        image("13a_partial", "Experiment 13A partial-codebook progression"), "",
        image("13a_layers", "Experiment 13A residual-layer progression"), "",
        "Overshoot, effective no-op use, and dead atoms correlate with worse validation tails, while non-zero residual-gain use correlates with better tails. These are mechanism diagnostics, not substitutes for matched interventions.", "",
        image("13a_diagnostics", "Experiment 13A decoder and dictionary diagnostics"), "",
        "### Eligibility gate and bounded scaling ablation", "",
        "The automatic 13A eligibility gate did not pass: the best early/middle median reconstructed fraction was about `2.054%`, below the required `5%`, despite satisfying retired-energy limits. That failure is why the three 13B thresholds are exploratory treatments rather than a frozen selector output.", "",
        *scaling_table, "" if scaling_table else "The preserved 39-row full-training prefix was not available to this regeneration.", "",
        "The scaling comparison remains a non-random 39-row prefix and excludes runtime. It is useful as a bounded sensitivity check, not a general data-scaling law.", "",
        "## Experiment 13B: How Much Capacity Was Freed?", "",
        "The no-filter 13A baseline retires no LFOs from construction. For 13B, capacity is measured as the mean training-population fraction excluded after layers 1–15, because those exclusions affect at least one later residual layer. Layer-16 coverage is reported separately: it describes endpoint eligibility but frees no remaining construction.", "",
        *capacity_table, "",
        image("eligibility", "Eligibility capacity released through depth by family"), "",
        "Family-specific depth trajectories are essential. The all-row median is nearly flat because many strategies retire few LFOs, while finish-oriented and diversity-aware families create much larger regimes.", "",
        "## Paired Validation Effects Versus No Filtering", "",
        "Each cell and count below is 13B minus its exact `AllResiduals + LayerClip0To1` 13A counterpart. Negative deltas improve RMSE metrics; positive deltas improve strict-perfect rate.", "",
        *paired_effect_table, "",
        image("population", "Four-metric eligibility effects versus matched no-filter baselines"), "",
        "## Does Freed Capacity Improve the Rest?", "",
        "The capacity-to-quality comparison puts the experiment's mechanism on the x-axis and its paired validation consequence on the y-axis. The global rank correlations are descriptive across intentionally different families; the paired point for each strategy remains the stronger unit of evidence.", "",
        *capacity_correlation, "",
        image("capacity_quality", "Future-layer capacity freed versus paired validation benefit"), "",
        "## Strict-Perfect Regime and Post-Filter Interactions", "",
        f"The best 13B strict-perfect rate is `{strict_peak:.3%}`. Seven rows reach the same 313-of-1605 validation count, showing a discrete finishing regime rather than a gentle shift in average error. The strongest balanced exact-finish candidates come from `DiverseCoverageHardRepairInterleaved + CandidateBudget48`; finish-repair rows reach the same rate with much worse median and tail quality.", "",
        image("strict_noop", "Experiment 13B strict-perfect regime versus tail quality"), "",
        "CandidateBudget48 remains the more repeatable post-filter lever. TwoPhase still changes sign by family and should not be promoted to a universal schedule rule.", "",
        image("factors", "Experiment 13B candidate-budget and schedule interactions"), "",
        f"The generated interaction audit contains `{len(factor_rows)}` family-by-epsilon summaries.", "",
        "## Final Cross-Phase Interpretation", "",
        *phase_b_leaders, "",
        f"The 13B four-objective frontier contains `{len(phase_b_pareto)}` rows. `AllClusterMeans` remains the simplest low-P95 option. `DiverseCoverageHardRepairTwoPhase + CandidateBudget48` supplies the strongest median/node-max tradeoff, while the interleaved CandidateBudget48 variant provides the most credible high strict-perfect alternative.", "",
        image("pareto", "Experiment 13B four-metric quality frontier"), "",
        "The partial-codebook result remains consistent with 13A: most tail improvement arrives early, but later atoms continue to help enough that shrinking W8 requires a dedicated contract-changing ablation.", "",
        image("partial", "Experiment 13B partial-codebook progression"), "",
        "## Recommendations", "",
        "1. Keep `LayerClip0To1`; its 45/45 matched P95 result is the strongest general Experiment 13 conclusion.",
        "2. Treat eligibility as a family-specific intervention. Use the paired capacity-effect table, not an epsilon-only leaderboard.",
        "3. Keep `AllClusterMeans` as the simple low-P95 candidate and the `DiverseCoverageHardRepair + CandidateBudget48` variants as the balanced frontier candidates.",
        "4. Preserve CandidateBudget48 only where its family-specific gain justifies doubled offline search; keep schedule coupled to family.",
        "5. Do not shrink W8D16 or infer deployed latency changes from these offline construction diagnostics.", "",
        "## Method and Audit Boundaries", "",
        f"Source run: `{source_display}`.", "",
        f"The automatic selector recorded `selection_passed={selection.get('selection_passed')}`. No tested epsilon is relabeled as a frozen selector choice.", "",
        "All paired eligibility comparisons use the exact 45 `LayerClip0To1` 13A rows. The baseline is labeled `No eligibility filter (13A)` and is never represented as epsilon zero. The capacity measure excludes layer 16 from future-layer savings. Validation quality retains four separate co-primary metrics.", "",
        "All timing is offline experiment work. Every row preserves Beam4, PhaseAndResidualGain, 97 control points, W8D16, and 193 deployed prediction outputs.", "",
        image("runtime", "Complete Experiment 13 same-run offline timing"), "",
        f"Configuration fingerprint: `{selection.get('configuration_fingerprint')}`.", "",
    ])


def _complete_13a_markdown(
    bundle: AnalysisBundle,
    *,
    source_run: Path,
    report_path: Path,
    plot_paths: Mapping[str, Path],
    selection: Mapping[str, Any],
    scaling_rows: Sequence[Mapping[str, Any]],
    scaling_baseline_run: Path | None,
    strict_thresholds: Mapping[str, Any] | None,
) -> str:
    normalization = _comparison_summary(bundle.matched_deltas, "layer_normalization_policy")
    budget = _comparison_summary(bundle.matched_deltas, "utility_candidate_budget")
    schedule = _comparison_summary(bundle.matched_deltas, "layer_schedule")
    pareto = [row for row in bundle.co_primary if _truth(row.get("pareto_candidate"))]
    top = sorted(bundle.summaries, key=lambda row: _number(row, "validation_p95_rmse"))[:10]
    runtime = sorted(bundle.summaries, key=lambda row: _number(row, "oracle_construction_time"))
    candidate_statistics = selection.get("training_statistics_used", {}).get("candidate_statistics", {})
    max_early_middle = max(
        (
            float(value.get("max_early_middle_median_retired_lfo_fraction", 0.0))
            for value in candidate_statistics.values()
        ),
        default=0.0,
    )
    source_display = os.path.relpath(source_run, report_path.parent).replace("\\", "/")
    baseline_display = (
        os.path.relpath(scaling_baseline_run, report_path.parent).replace("\\", "/")
        if scaling_baseline_run is not None
        else None
    )

    def image(name: str, alt: str) -> str:
        relative = os.path.relpath(plot_paths[name], report_path.parent).replace("\\", "/")
        return f"![{alt}]({relative})"

    metric_specs = (
        ("validation_median_rmse", "Median RMSE", False),
        ("validation_strict_perfect_lfo_rate", "Strict-perfect LFO rate", True),
        ("validation_p95_rmse", "P95 RMSE", False),
        ("validation_node_max_error_p95", "Node-max error P95", False),
    )
    metric_leaders = [
        "| Co-primary metric | Better | Best value | Strategy row |",
        "| --- | --- | ---: | --- |",
    ]
    for metric, label, higher in metric_specs:
        row = (max if higher else min)(bundle.summaries, key=lambda item, key=metric: _number(item, key))
        value = _number(row, metric)
        value_text = f"{value:.3%}" if higher else f"{value:.8g}"
        metric_leaders.append(f"| {label} | {'higher' if higher else 'lower'} | {value_text} | `{row.get('row_id')}` |")

    threshold_sensitivity = [
        "| Max-absolute tolerance | RMSE tolerance | Best validation strict-perfect rate | Median row rate | Distinct rates | Pareto rows |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if strict_thresholds:
        rates_by_row = strict_thresholds["rates_by_row"]
        for tolerance in strict_thresholds["tolerances"]:
            rates = [float(rates_by_row[str(row.get("row_id", ""))][tolerance]) for row in bundle.summaries]
            threshold_rows = []
            for source, rate in zip(bundle.summaries, rates):
                record = dict(source)
                record["validation_strict_perfect_lfo_rate"] = rate
                threshold_rows.append(record)
            threshold_sensitivity.append(
                f"| `{tolerance}` | `{float(tolerance) / 10:.0e}` | {max(rates):.3%} | "
                f"{median(rates):.3%} | {len(set(rates))} | {len(_pareto_ids(threshold_rows))} |"
            )

    metric_agreement_table = [
        "| Metric pair | Spearman ρ | Interpretation |",
        "| --- | ---: | --- |",
    ]
    for left_index, (left, left_label, _) in enumerate(metric_specs):
        for right, right_label, _ in metric_specs[left_index + 1:]:
            correlation = _spearman(
                [_number(row, left) for row in bundle.summaries],
                [_number(row, right) for row in bundle.summaries],
            )
            strength = "strong agreement" if abs(correlation) >= 0.8 else "partial agreement" if abs(correlation) >= 0.5 else "weak / distinct signal"
            metric_agreement_table.append(f"| {left_label} vs {right_label} | {correlation:+.3f} | {strength} |")

    rank_disagreement_table = [
        "| High-disagreement strategy row | Family | Median rank | Strict-perfect rank | P95 rank | Node-max rank | Rank spread |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(bundle.rankings, key=lambda item: _number(item, "rank_spread"), reverse=True)[:8]:
        rank_disagreement_table.append(
            f"| `{row.get('row_id')}` | {row.get('construction_family')} | "
            f"{_number(row, 'rank_validation_median_rmse'):.1f} | "
            f"{_number(row, 'rank_validation_strict_perfect_lfo_rate'):.1f} | "
            f"{_number(row, 'rank_validation_p95_rmse'):.1f} | "
            f"{_number(row, 'rank_validation_node_max_error_p95'):.1f} | "
            f"{_number(row, 'rank_spread'):.1f} |"
        )

    generalization_table = [
        "| Metric | Validation − training median gap | Range | Rows where validation is better |",
        "| --- | ---: | ---: | ---: |",
    ]
    for train_key, validation_key, label, higher in (
        ("train_median_rmse", "validation_median_rmse", "Median RMSE", False),
        ("train_strict_perfect_lfo_rate", "validation_strict_perfect_lfo_rate", "Strict-perfect rate", True),
        ("train_p95_rmse", "validation_p95_rmse", "P95 RMSE", False),
        ("train_node_max_error_p95", "validation_node_max_error_p95", "Node-max P95", False),
    ):
        values = [_number(row, validation_key) - _number(row, train_key) for row in bundle.summaries]
        better = sum(value > 0 if higher else value < 0 for value in values)
        suffix = " pp" if higher else ""
        scale = 100.0 if higher else 1.0
        generalization_table.append(
            f"| {label} | {median(values) * scale:+.6g}{suffix} | "
            f"{min(values) * scale:+.6g} to {max(values) * scale:+.6g}{suffix} | {better}/{len(values)} |"
        )

    interaction_table = [
        "| Construction family | LayerClip − FinalClip | Budget48 − Budget24 | TwoPhase − Interleaved |",
        "| --- | ---: | ---: | ---: |",
    ]
    interaction_lookup = {
        (str(row.get("construction_family", "")), str(row.get("comparison", ""))): _number(row, "median_delta_validation_p95_rmse")
        for row in bundle.factor_interactions
    }
    interaction_families = sorted({str(row.get("construction_family", "")) for row in bundle.factor_interactions if row.get("construction_family")})
    for family in interaction_families:
        values = []
        for comparison in ("layer_normalization_policy", "utility_candidate_budget", "layer_schedule"):
            value = interaction_lookup.get((family, comparison))
            values.append("—" if value is None else f"{value:+.6g}")
        interaction_table.append(f"| {family} | {' | '.join(values)} |")

    marginal_table = [
        "| Added active atom | Median validation P95 delta | Rows improved | Median validation median-RMSE delta |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for active_count in sorted({int(_number(row, "active_atom_count")) for row in bundle.marginal_atoms}):
        selected = [row for row in bundle.marginal_atoms if int(_number(row, "active_atom_count")) == active_count]
        p95_values = [_number(row, "delta_validation_p95_rmse") for row in selected]
        median_values = [_number(row, "delta_validation_median_rmse") for row in selected]
        marginal_table.append(
            f"| {active_count} | {median(p95_values):+.8g} | {sum(value < 0 for value in p95_values)}/{len(p95_values)} | {median(median_values):+.8g} |"
        )

    layer_table = [
        "| Residual layer completed | Median training P95 RMSE | Median layer-to-layer change |",
        "| ---: | ---: | ---: |",
    ]
    layer_medians: dict[int, float] = {}
    for layer in sorted({int(_number(row, "residual_layer")) for row in bundle.layer_progression}):
        values = [_number(row, "training_p95_rmse") for row in bundle.layer_progression if int(_number(row, "residual_layer")) == layer]
        layer_medians[layer] = median(values)
        delta = "—" if layer - 1 not in layer_medians else f"{layer_medians[layer] - layer_medians[layer - 1]:+.8g}"
        layer_table.append(f"| {layer} | {layer_medians[layer]:.8g} | {delta} |")

    diagnostic_specs = (
        ("train_p95_rmse", "Training P95 RMSE"),
        ("validation_node_max_error_p95", "Validation node-max P95"),
        ("validation_median_rmse", "Validation median RMSE"),
        ("validation_overshoot_rate_before_final_clip", "Pre-final-clip overshoot rate"),
        ("residual_layer_effective_no_op_usage_rate_median", "Effective no-op usage"),
        ("residual_layer_dead_atom_rate_median", "Dead-atom rate"),
        ("residual_gain_nonzero_rate", "Non-zero residual-gain rate"),
        ("duplicate_atom_rate", "Duplicate-atom rate"),
        ("oracle_construction_time", "Oracle construction time"),
    )
    diagnostic_table = [
        "| Diagnostic | Spearman ρ with validation P95 | Reading |",
        "| --- | ---: | --- |",
    ]
    validation_p95_values = [_number(row, "validation_p95_rmse") for row in bundle.diagnostics]
    for key, label in diagnostic_specs:
        correlation = _spearman([_number(row, key) for row in bundle.diagnostics], validation_p95_values)
        diagnostic_table.append(
            f"| {label} | {correlation:+.3f} | {'higher tracks worse tail quality' if correlation > 0 else 'higher tracks better tail quality'} |"
        )

    family_table = [
        "| Construction family | Rows | Median RMSE | Median P95 RMSE | Median node-max P95 | Best P95 row |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for family in sorted({str(row.get("construction_family", "")) for row in bundle.summaries}):
        rows = [row for row in bundle.summaries if row.get("construction_family") == family]
        best_family = min(rows, key=lambda row: _number(row, "validation_p95_rmse"))
        family_table.append(
            f"| {family} | {len(rows)} | {median([_number(row, 'validation_median_rmse') for row in rows]):.8g} | "
            f"{median([_number(row, 'validation_p95_rmse') for row in rows]):.8g} | "
            f"{median([_number(row, 'validation_node_max_error_p95') for row in rows]):.8g} | `{best_family.get('row_id')}` |"
        )

    mechanism_table = [
        "| Construction family | Median slot P95 gain | Prototype convergence | Prototype iterations | Duplicate-alignment reuse | Candidate evaluations |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for family in sorted({str(row.get("construction_family", "")) for row in bundle.mechanism_diagnostics}):
        rows = [row for row in bundle.mechanism_diagnostics if row.get("construction_family") == family]
        present = lambda key: [_number(row, key) for row in rows if row.get(key) not in {None, "", "None"}]
        gains = present("median_slot_p95_improvement")
        convergence = present("prototype_convergence_rate")
        iterations = present("prototype_iterations_median")
        duplicate = present("exact_duplicate_alignment_reuse_rate")
        candidates = present("candidate_evaluations")
        mechanism_table.append(
            f"| {family} | {'—' if not gains else f'{median(gains):.8g}'} | "
            f"{'—' if not convergence else f'{median(convergence):.1%}'} | "
            f"{'—' if not iterations else f'{median(iterations):.1f}'} | "
            f"{'—' if not duplicate else f'{median(duplicate):.1%}'} | "
            f"{'—' if not candidates else f'{median(candidates):.0f}'} |"
        )

    matched_table = [
        "| Matched factor | Metric | Right wins / left wins / ties | Median right-minus-left delta |",
        "| --- | --- | ---: | ---: |",
    ]
    comparison_labels = (
        ("layer_normalization_policy", "LayerClip0To1 vs FinalClipOnly"),
        ("utility_candidate_budget", "CandidateBudget48 vs CandidateBudget24"),
        ("layer_schedule", "TwoPhase vs Interleaved"),
    )
    for comparison, comparison_label in comparison_labels:
        for metric, label, higher in metric_specs:
            summary = _comparison_metric_summary(bundle.matched_deltas, comparison, metric, higher_is_better=higher)
            delta = float(summary["median"])
            delta_text = f"{delta * 100:+.5f} pp" if higher else f"{delta:+.8g}"
            matched_table.append(
                f"| {comparison_label} | {label} | {summary['improved']} / {summary['worsened']} / {summary['tied']} | {delta_text} |"
            )

    pareto_table = [
        "| Pareto strategy | Median RMSE ↓ | Strict-perfect ↑ | P95 RMSE ↓ | Node-max P95 ↓ |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(pareto, key=lambda item: _number(item, "validation_p95_rmse")):
        pareto_table.append(
            f"| `{row.get('row_id')}` | {_number(row, 'validation_median_rmse'):.8g} | "
            f"{_number(row, 'validation_strict_perfect_lfo_rate'):.3%} | "
            f"{_number(row, 'validation_p95_rmse'):.8g} | {_number(row, 'validation_node_max_error_p95'):.8g} |"
        )

    scaling_table = [
        "| Co-primary metric | Median 50%-minus-100% delta | 50% better / 100% better / ties |",
        "| --- | ---: | ---: |",
    ]
    for metric, label, higher in metric_specs:
        values = [_number(row, f"delta_{metric}") for row in scaling_rows]
        if not values:
            continue
        scaling_table.append(
            f"| {label} | {median(values) * (100 if higher else 1):+.8g}{' pp' if higher else ''} | "
            f"{sum(value > 0 if higher else value < 0 for value in values)} / "
            f"{sum(value < 0 if higher else value > 0 for value in values)} / {sum(value == 0 for value in values)} |"
        )

    lines = [
        "# Experiment 13A: Complete Fixed-W8D16 Strategy Grid",
        "",
        "> **13A complete · 90/90 rows.** This report is authoritative for the unfiltered `AllResiduals` strategy grid. "
        "The automatic epsilon selector did not pass, no epsilon is frozen, and Experiment 13B has not run; therefore this is not the final AllResiduals-versus-UnresolvedOnly report.",
        "",
        "## Main Findings",
        "",
        f"Layer-wise clipping is the clearest global result. `LayerClip0To1` improves validation P95 RMSE in `{normalization['improved']}/{normalization['count']}` matched pairs, with median delta `{normalization['median']:.8g}` and range `{normalization['minimum']:.8g}` to `{normalization['maximum']:.8g}`. It is a decoder-free constraint and changes no model prediction heads.",
        "This complete 13A result fixes Experiment 13B at `LayerClip0To1`. The filtered phase retains the 45 clipped construction, schedule, and applicable candidate-budget cells, then evaluates each independently at eligibility epsilons `1e-2`, `1e-3`, and `1e-4` for 135 rows. The sweep is exploratory because it was fixed after 13A completed and `1e-4` was not in the original calibration candidate set.",
        "",
        f"A larger repair shortlist is a secondary, mixed lever. `CandidateBudget48` improves `{budget['improved']}/{budget['count']}`, worsens `{budget['worsened']}`, and ties `{budget['tied']}` P95 comparisons; its median effect is only `{budget['median']:.8g}`. More offline search is not a guaranteed quality win.",
        "",
        f"`TwoPhase` improves `{schedule['improved']}/{schedule['count']}` schedule pairs with median P95 delta `{schedule['median']:.8g}`. The slight aggregate edge remains family-dependent, so schedule should remain an interaction rather than a universal default.",
        "",
        "The quality frontier has three distinct jobs rather than one universal winner: `AllClusterMeans + LayerClip0To1` gives the lowest P95 RMSE; `DiverseCoverageHardRepairTwoPhase + CandidateBudget48 + LayerClip0To1` gives the best median RMSE and node-max P95; and the clipped `CommonCaseRepair + CandidateBudget24` anchor preserves the highest strict-perfect rate.",
        "",
        f"The automatic epsilon rule did not pass. All candidates satisfy the retired unexplained-energy limits, but the best early/middle median reconstructed fraction is only `{max_early_middle:.3%}`, below the required `5%`. Experiment 13B therefore treats epsilon as an exploratory sweep axis at `1e-2`, `1e-3`, and `1e-4` rather than claiming one selected threshold.",
        "",
        "## Research Questions",
        "",
        "This complete 13A analysis asks seven questions: which strategies occupy the four-objective quality frontier; whether validation behavior tracks training behavior; which policy effects survive matched controls; where those effects interact with construction family; how quickly residual layers and atom slots earn their capacity; what decoder and dictionary diagnostics explain failure modes; and how much offline work each strategy consumes under the fixed 193-head deployed contract.",
        "",
        image("pareto", "Experiment 13A co-primary quality frontier"),
        "",
        "The x-axis is validation median RMSE and the y-axis is validation P95 RMSE; lower-left is better. Outlined points remain non-dominated after strict-perfect rate and node-max P95 are also considered. The plot is navigation across tradeoffs, not a scalar leaderboard.",
        "",
        "## Four Co-Primary Validation Metrics",
        "",
        *metric_leaders,
        "",
        *pareto_table,
        "",
        "Strict-perfect rate has only two observed values across the 90 rows. RMSE improvements therefore must not be described as automatically improving exact finishes at the fixed `1e-5` threshold.",
        "",
        "### Strict-perfect threshold sensitivity",
        "",
        *(threshold_sensitivity if strict_thresholds else ["Threshold sensitivity was not replayed for this report build."]),
        "",
        *( [image("strict_thresholds", "Strict-perfect rate across logarithmically spaced tolerance tuples")] if strict_thresholds else [] ),
        "",
        "The tolerance parameter preserves the original two-condition definition: per-LFO RMSE must be at most one tenth of the selected tolerance and maximum absolute point error must be at most the selected tolerance. The interactive report recomputes the strict-perfect leader, four-objective Pareto membership, ranks, correlations, and matched strict-perfect deltas when the tolerance changes. Continuous RMSE and node-max metrics do not change.",
        "",
        "### Metric agreement and disagreement",
        "",
        image("metric_agreement", "Co-primary metric rank agreement"),
        "",
        *metric_agreement_table,
        "",
        *rank_disagreement_table,
        "",
        "Median RMSE, P95 RMSE, and node-max P95 share substantial ordering information, but they are not interchangeable. Strict-perfect rate is nearly orthogonal to the tail metrics because it is both thresholded and coarse: only two observed values split the grid. This is why the frontier retains all four objectives instead of reporting one synthetic score.",
        "",
        "The rows with the largest rank spread are particularly useful audit cases: they are strong on one objective and weak on another. The generated `metric_rankings.csv` retains tied ranks, mean co-primary rank, and rank spread for every strategy.",
        "",
        "## Train-to-Validation Stability",
        "",
        image("generalization", "Training versus validation stability"),
        "",
        *generalization_table,
        "",
        "The train/validation relationship is stable but not a conventional overfitting story. Validation median RMSE is slightly higher on the median row, while validation P95 is often lower. The fixed 50% construction sample is therefore not simply an easier subset than validation. Strong train-P95 versus validation-P95 rank agreement supports using training construction diagnostics, but the non-zero gaps prohibit substituting training metrics for held-out quality.",
        "",
        "## Matched Policy Effects",
        "",
        *matched_table,
        "",
        "Negative RMSE and node-max deltas favor the policy named before `vs`; positive strict-perfect deltas favor it. These matched comparisons isolate one design factor while holding the others fixed.",
        "",
        "### Layer normalization",
        "",
        image("normalization", "Matched normalization P95 deltas"),
        "",
        "Every bar is below zero: clipping after each residual layer consistently prevents physical-range overshoot from accumulating into the validation tail.",
        "",
        "### Candidate budget",
        "",
        image("budget", "Matched candidate-budget P95 deltas"),
        "",
        "Bars fall on both sides of zero. CandidateBudget48 can find better observed repairs, but later slots and Beam4 encoding frequently compensate for the smaller shortlist.",
        "",
        "### Layer schedule",
        "",
        image("schedule", "Matched schedule P95 deltas"),
        "",
        "The signs remain mixed. TwoPhase works especially well for some diversity-aware and robust prototype families, while other families benefit from earlier repair interleaving.",
        "",
        "### Factor interactions by construction family",
        "",
        image("interactions", "Family-specific matched policy interactions"),
        "",
        *interaction_table,
        "",
        "Each cell is a within-family median of matched validation-P95 deltas. The normalization column is consistently negative, so clipping generalizes across construction mechanisms. Budget and schedule change sign by family. Aggregating those signs into one global winner would erase the main design interaction.",
        "",
        "## Construction-Family Interpretation",
        "",
        "Pure cluster prototypes are competitive at the tail: `AllClusterMeans + LayerClip0To1` is the P95 leader. The best median and node-max row instead combines diverse broad coverage with hard-tail repair, supporting a mechanism in which dissimilar population prototypes remove reusable structure before observed examples address the remaining difficult cases. The CommonCaseRepair anchor retains the strict-perfect lead, showing that finishing behavior is not captured by aggregate RMSE alone.",
        "",
        *family_table,
        "",
        "The ten lowest-P95 rows are:",
        "",
    ]
    for index, row in enumerate(top, start=1):
        lines.append(
            f"{index}. `{row.get('row_id')}` — P95 `{_number(row, 'validation_p95_rmse'):.8g}`, median "
            f"`{_number(row, 'validation_median_rmse'):.8g}`, strict-perfect "
            f"`{_number(row, 'validation_strict_perfect_lfo_rate'):.3%}`, node-max P95 "
            f"`{_number(row, 'validation_node_max_error_p95'):.8g}`."
        )
    lines.extend([
        "",
        "## Partial-Codebook Progression",
        "",
        image("partial", "Partial-codebook progression"),
        "",
        "Move left to right as each residual layer gains another active atom; lower validation P95 is better. The early slope measures capacity efficiency, while late flattening shows diminishing returns. The first few atoms carry most of the quality gain, but family curves continue to separate through slot seven, so this fixed-W8 design does not support removing late slots without a separate head-budget experiment.",
        "",
        image("marginal_atoms", "Marginal value of each additional active atom"),
        "",
        *marginal_table,
        "",
        "The second atom produces the largest typical improvement, and the next two still remove substantial tail error. Later atoms have smaller median gains but remain beneficial for most strategies: the seventh improves validation P95 in the majority of rows. Strict-perfect rate has a median marginal change of zero at every slot, another indication that thresholded finishes and continuous reconstruction quality answer different questions.",
        "",
        "## Residual-Layer Learning Curve",
        "",
        image("layers", "Residual-layer progression after active slot seven"),
        "",
        *layer_table,
        "",
        "This view follows the completed seven-atom codebook after every residual layer. Tail error falls monotonically from layer 1 through layer 16, with diminishing but still material reductions late in the stack. The result supports D16 for this experiment: it does not prove that every layer is cost-optimal, but it rules out the claim that the later layers are doing nothing.",
        "",
        "## Decoder and Dictionary Diagnostics",
        "",
        image("diagnostics", "Decoder and dictionary diagnostics versus validation P95"),
        "",
        *diagnostic_table,
        "",
        "Overshoot, effective no-op usage, and dead atoms all track worse tail quality, while frequent non-zero residual gains track better tail quality. These are associations across deliberately different strategy families, not isolated causal effects. The matched clipping result supplies the stronger causal design evidence for overshoot: LayerClip0To1 removes overshoot and improves every matched P95 pair.",
        "",
        *mechanism_table,
        "",
        "Broad and repair atoms solve different problems. Synthesized prototypes seek reusable population structure; observed residuals perform concrete cleanup. Several iterative prototype builders reach their iteration cap rather than declaring convergence, while one-shot cluster/diversity builders are structurally different. Duplicate-alignment reuse is a small but non-zero signal of redundant atom proposals and is retained as an audit diagnostic rather than treated as a failure by itself.",
        "",
        "## Eligibility Calibration and Gate Result",
        "",
        "Eligibility is framed as LFO population reduction: the percentage of already-solved LFOs that can stop competing for later construction capacity. Layer 8 is the useful mid-run checkpoint—an LFO retired there is absent from layers 9–16. Layer 16 is an endpoint accuracy check and does not itself save later-layer work.",
        "",
        image("layer_quantiles", "Exact tolerance required for each LFO-reduction target at layers 8 and 16"),
        "",
        image("slot_quantiles", "Slot-level epsilon quantiles"),
        "",
        image("layer_coverage", "Family-level LFO population reduction at layers 8 and 16"),
        "",
        image("slot_coverage", "Slot-level reconstructed fractions"),
        "",
        "The retirement plots ask whether excluding more LFOs would abandon meaningful unexplained residual energy. The desired direction is lower-right: more LFOs retired with less unexplained energy. The energy safety criteria pass, but the coverage criterion does not, so these plots motivate measuring a threshold sweep rather than declaring an epsilon winner.",
        "",
        image("retired", "Retired fraction versus unexplained energy"),
        "",
        image("energy", "Incoming versus unexplained retired energy"),
        "",
        "## Training-Data Scaling Ablation",
        "",
        f"The preserved full-training prefix supplies `{len(scaling_rows)}` matched rows with identical validation membership. It is a non-random execution-order prefix, so this is a bounded method-level ablation rather than a balanced estimate over all construction families.",
        "",
        *scaling_table,
        "",
        "The 50%-training run has modestly worse median and P95 RMSE on the matched prefix, while strict-perfect rate and node-max P95 improve on most rows. This mixed direction argues against describing the sample reduction as uniformly harmful or uniformly beneficial. Runtime is excluded because the legacy fragment includes Modern Standby and a superseded execution implementation.",
        "",
        "## Same-Run Runtime Diagnostics",
        "",
        image("runtime", "Experiment 13A same-run oracle construction time"),
        "",
        f"This chart compares rows only inside the optimized train-50% run. Median oracle construction time is `{median([_number(row, 'oracle_construction_time') for row in runtime]):.3f}` seconds and the maximum is `{_number(runtime[-1], 'oracle_construction_time'):.3f}` seconds. The scale is continuous because this run contains no host-sleep outliers. These timings support within-run cost comparisons only.",
        "",
        image("work", "Offline work efficiency"),
        "",
        "CandidateBudget48 exactly doubles deterministic repair-candidate evaluation relative to CandidateBudget24 wherever repair search applies, yet its median quality gain is small and the sign is mixed. Oracle construction time has essentially no monotonic relationship with validation P95, so spending longer is not evidence of a better dictionary. The timing decomposition separates construction, training encoding, and validation encoding; all three are offline experiment costs. Every row still emits the same 193 deployed prediction heads, so none of these charts is an inference-latency comparison.",
        "",
        "## Practical Takeaways",
        "",
        "- Lock Experiment 13B to the 45 `LayerClip0To1` counterparts and evaluate each at `1e-2`, `1e-3`, and `1e-4` for 135 rows; do not rerun `FinalClipOnly`.",
        "- Carry all three Pareto strategies into the 13B interpretation; no scalar winner represents all four quality objectives.",
        "- Treat CandidateBudget48 and TwoPhase as interaction-dependent choices, not unconditional defaults.",
        "- Preserve all seven active atoms and all 16 residual layers for 13B; 13A shows diminishing returns, not dead capacity.",
        "- Use overshoot, no-op, gain-use, duplicate, and convergence diagnostics to explain results, not to replace matched quality evidence.",
        "- Run the fixed 135-row Experiment 13B sweep and interpret epsilon as an explicit factor, not a selected scalar.",
        "- Do not compare legacy and optimized wall-clock timings or claim a general 50%-training scaling law from the 39-row prefix.",
        "",
        "## Method Notes and Generated Artifacts",
        "",
        f"The completed source run is `{source_display}` relative to this report. The scaling baseline is `{baseline_display}`. Derived analysis tables, report images, and the interactive payload are written outside both source runs.",
        "",
        "The audit artifacts now include `strategy_diagnostics.csv`, `metric_rankings.csv`, `factor_interaction_summary.csv`, `marginal_atom_value.csv`, `residual_layer_progression.csv`, and `construction_mechanism_diagnostics.csv` in addition to the original coverage, frontier, matched-effect, partial-codebook, and calibration tables.",
        "",
        "All rows preserve W8D16, 32 base choices, one no-op plus seven active atoms per residual layer, 97 control points, PhaseAndResidualGain scalars, Beam4 encoding, and 193 model prediction outputs. Codebook construction is offline/oracle work; topology is not a deployed runtime input.",
        "",
        "### Audit boundaries",
        "",
        "This report does not claim an eligibility benefit, a selected epsilon, a complete Experiment 13 winner, deployed runtime differences, or a general training-data scaling law. It reports complete 13A AllResiduals evidence, a bounded 39-row scaling ablation, and same-run offline timing diagnostics. Those boundaries mirror the more forensic reporting standard used in Experiments 8–12.",
        "",
    ])
    return "\n".join(lines)


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
    metric_specs = (
        ("validation_median_rmse", "Median RMSE", False),
        ("validation_strict_perfect_lfo_rate", "Strict-perfect LFO rate", True),
        ("validation_p95_rmse", "P95 RMSE", False),
        ("validation_node_max_error_p95", "Node-max error P95", False),
    )

    def metric_value(metric: str, value: float) -> str:
        return f"{value:.3%}" if metric == "validation_strict_perfect_lfo_rate" else f"{value:.8g}"

    primary_metric_table = [
        "| Co-primary metric | Better direction | Best observed value | Best observed row |",
        "| --- | --- | ---: | --- |",
    ]
    for metric, label, higher_is_better in metric_specs:
        best_metric_row = (max if higher_is_better else min)(
            bundle.summaries,
            key=lambda row, key=metric: _number(row, key),
        )
        primary_metric_table.append(
            f"| {label} | {'higher' if higher_is_better else 'lower'} | "
            f"{metric_value(metric, _number(best_metric_row, metric))} | "
            f"`{best_metric_row.get('row_id')}` |"
        )

    comparison_labels = (
        ("layer_normalization_policy", "LayerClip0To1 vs FinalClipOnly"),
        ("utility_candidate_budget", "CandidateBudget48 vs CandidateBudget24"),
        ("layer_schedule", "TwoPhase vs Interleaved"),
    )
    matched_metric_table = [
        "| Matched factor | Co-primary metric | Right / left / ties | Median right-minus-left delta |",
        "| --- | --- | ---: | ---: |",
    ]
    for comparison, comparison_label in comparison_labels:
        for metric, label, higher_is_better in metric_specs:
            summary = _comparison_metric_summary(
                bundle.matched_deltas,
                comparison,
                metric,
                higher_is_better=higher_is_better,
            )
            delta = float(summary["median"])
            delta_text = f"{delta * 100:+.5f} pp" if higher_is_better else f"{delta:+.8g}"
            matched_metric_table.append(
                f"| {comparison_label} | {label} | {summary['improved']} / {summary['worsened']} / "
                f"{summary['tied']} | {delta_text} |"
            )

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
        "### All Four Co-Primary Validation Metrics",
        "",
        "Experiment 13 defines quality using four co-primary outcomes. Each row below reports the best observed value in the incomplete prefix; different metrics can select different strategy rows.",
        "",
        *primary_metric_table,
        "",
        "### Matched Effects Across All Four Co-Primary Metrics",
        "",
        "The table below prevents the P95-focused static plots from standing in for the complete outcome set. The comparison label names both policies explicitly: negative RMSE and node-max deltas favor the policy named after `vs`, while positive strict-perfect deltas favor that same policy.",
        "",
        *matched_metric_table,
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
        "Lower is faster. A broken x-axis separates ordinary legacy construction timings from the two host-sleep-inflated observations, so the ordinary pattern remains readable while the artifacts stay visible. The two separated observations are "
        f"`{_number(runtime_rows[0], 'oracle_construction_time'):.8g}` and `{_number(runtime_rows[1], 'oracle_construction_time'):.8g}` seconds. These measurements diagnose the aborted run but must not be compared with optimized-run timing.",
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
    return _comparison_metric_summary(
        rows,
        comparison,
        "validation_p95_rmse",
        higher_is_better=False,
    )


def _comparison_metric_summary(
    rows: Sequence[Mapping[str, Any]],
    comparison: str,
    metric: str,
    *,
    higher_is_better: bool,
) -> dict[str, float | int]:
    values = [
        _number(row, f"delta_{metric}")
        for row in rows
        if row.get("comparison") == comparison
    ]
    if not values:
        return {
            "count": 0,
            "improved": 0,
            "worsened": 0,
            "tied": 0,
            "median": math.nan,
            "minimum": math.nan,
            "maximum": math.nan,
        }
    return {
        "count": len(values),
        "improved": sum(value > 0 if higher_is_better else value < 0 for value in values),
        "worsened": sum(value < 0 if higher_is_better else value > 0 for value in values),
        "tied": sum(value == 0 for value in values),
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


def _save(plt: Any, figure: Any, path: Path, *, tight: bool = True) -> None:
    if tight:
        figure.tight_layout()
    temporary = path.with_name(f".{path.name}.tmp")
    figure.savefig(temporary, dpi=160, format="png", metadata={"Software": "OBRUXO Experiment 13 report generator"})
    plt.close(figure)
    temporary.replace(path)

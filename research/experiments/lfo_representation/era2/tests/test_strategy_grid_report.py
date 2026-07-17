from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest


ERA2_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ERA2_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from lfo_era2 import strategy_grid as grid  # noqa: E402
from lfo_era2 import strategy_grid_report as strategy_report  # noqa: E402
from lfo_era2 import strategy_grid_runtime as runtime  # noqa: E402


class StrategyGridPartialReportTests(unittest.TestCase):
    def test_partial_report_uses_shards_and_preserves_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "legacy_fragment"
            _write_partial_fixture(source)
            before = _fingerprint(source)
            analysis = root / "analysis"
            report = root / "reports" / "EXPERIMENT_13_PROVISIONAL.md"
            html_report = root / "reports" / "EXPERIMENT_13_PROVISIONAL.html"
            images = root / "reports" / "images" / "experiment_13" / "provisional"

            result = strategy_report.analyze_partial_strategy_grid(
                run_dir=source,
                analysis_output_dir=analysis,
                report_path=report,
                html_report_path=html_report,
                image_dir=images,
            )

            self.assertEqual(_fingerprint(source), before)
            self.assertEqual(Path(result["report"]), report.resolve())
            self.assertEqual(Path(result["html_report"]), html_report.resolve())
            self.assertFalse((source / "summary.csv").exists())
            text = report.read_text(encoding="utf-8")
            self.assertIn("Provisional evidence only", text)
            self.assertIn("`8/90`", text)
            self.assertIn("no eligibility epsilon has been selected", text)
            self.assertIn("Zero completed rows are available", text)
            self.assertNotIn("The frozen eligibility epsilon is", text)
            self.assertIn("All Four Co-Primary Validation Metrics", text)
            self.assertIn("Matched Effects Across All Four Co-Primary Metrics", text)
            self.assertIn("Strict-perfect LFO rate", text)
            self.assertIn("Node-max error P95", text)

            html = html_report.read_text(encoding="utf-8")
            self.assertIn('qs("#status-label").textContent = isComplete13A ? "13A COMPLETE" : "PROVISIONAL"', html)
            self.assertIn("Source coverage", html)
            self.assertIn("Layer normalization", html)
            self.assertIn("Candidate budget", html)
            self.assertIn("Layer schedule", html)
            self.assertIn("Construction families", html)
            self.assertIn("Partial codebook", html)
            self.assertIn("Eligibility calibration", html)
            self.assertIn("Legacy runtime", html)
            self.assertIn("historical only", html)
            self.assertIn("Four metrics define quality", html)
            self.assertIn("Median RMSE", html)
            self.assertIn("Strict-perfect LFO rate", html)
            self.assertIn("P95 RMSE", html)
            self.assertIn("Node-max error P95", html)
            self.assertIn("four co-primary metrics", html)
            self.assertIn("Global filters", html)
            self.assertIn("Multi-select · all report views", html)
            self.assertIn("family → construction variant → run settings", html)
            self.assertIn("completed</b>/planned rows", html)
            self.assertIn("Apply to every chart", html)
            self.assertIn('data-global-field="${esc(id)}"', html)
            self.assertIn('["family","construction_family","All families"]', html)
            self.assertIn('["policy","construction_policy","All policies"]', html)
            self.assertIn('["schedule","layer_schedule","All schedules"]', html)
            self.assertIn('["budget","utility_candidate_budget","All budgets"]', html)
            self.assertIn('["normalization","layer_normalization_policy","All normalization"]', html)
            self.assertIn('id="global-collapse"', html)
            self.assertIn('class="filter-pill${', html)
            self.assertIn('aria-pressed="${selected}"', html)
            self.assertIn("globalState[id].size", html)
            self.assertIn("renderReactiveFilterGroups", html)
            self.assertIn("facetStats", html)
            self.assertIn("facetRowMatches", html)
            self.assertIn("prunePolicySelections", html)
            self.assertIn("Select at least one family", html)
            self.assertIn("Pareto membership is undefined for uncompleted rows", html)
            self.assertIn("scheduleChartResize", html)
            self.assertNotIn('class="filter-field"><span>Construction family</span><select', html)
            self.assertIn("position: sticky", html)
            self.assertIn("Green favors LayerClip0To1", html)
            self.assertIn("Green favors CandidateBudget48", html)
            self.assertIn("Green favors TwoPhase", html)
            self.assertIn("`${rightPolicy} favored`", html)
            self.assertNotIn("right policy favored", html)
            self.assertNotIn("left policy favored", html)
            self.assertNotIn('id="theme-toggle"', html)
            self.assertNotIn('data-theme="dark"', html)
            self.assertIn("deltaCellLabel", html)
            self.assertIn('color:"#66757b"', html)
            self.assertIn("CONSTRUCTION_GUIDE", html)
            self.assertIn("AllDominantDirections", html)
            self.assertIn("Why this matters", html)
            self.assertIn("lower-right", html)
            self.assertIn("Host sleep artifact", html)
            self.assertIn("broken axis", html)
            self.assertIn("ResizeObserver", html)
            self.assertIn("grid-template-columns: minmax(0, 1fr)", html)
            self.assertIn("chart.delta-chart", html)
            self.assertIn("delta_validation_strict_perfect_lfo_rate", html)
            self.assertIn("https://cdn.jsdelivr.net/npm/echarts@6.1.0/dist/echarts.min.js", html)
            self.assertNotIn(str(root.resolve()), html)
            self.assertLess(html_report.stat().st_size, 1_000_000)
            payload_match = re.search(r'<script id="report-data" type="application/json">(.*?)</script>', html, re.DOTALL)
            self.assertIsNotNone(payload_match)
            payload = json.loads(payload_match.group(1))
            self.assertEqual(payload["schema_version"], "experiment13_interactive_report_v5")
            self.assertEqual(payload["meta"]["completed_rows"], 8)
            self.assertEqual(payload["meta"]["expected_rows"], 90)
            self.assertFalse(payload["meta"]["epsilon_selected"])
            self.assertFalse(payload["meta"]["experiment13b_started"])
            self.assertEqual(len(payload["tables"]["metrics"]), 8)
            self.assertEqual(len(payload["tables"]["coverage"]), 90)
            self.assertEqual(len(payload["tables"]["matched_deltas"]), 12)
            self.assertEqual(len(payload["tables"]["partial_codebook"]), 56)
            self.assertEqual(len(payload["calibration"]["row_ids"]), 8)
            self.assertTrue(payload["calibration"]["layer_quantiles_by_row"])
            self.assertTrue(payload["calibration"]["retired_sample_by_row"])
            element_ids = re.findall(r'\bid="([^"]+)"', html)
            self.assertEqual(len(element_ids), len(set(element_ids)))

            self.assertEqual(len(runtime.read_csv(analysis / "completed_row_coverage.csv")), 90)
            self.assertEqual(len(runtime.read_csv(analysis / "co_primary_metrics.csv")), 8)
            deltas = runtime.read_csv(analysis / "matched_factor_deltas.csv")
            self.assertEqual(len(deltas), 12)
            normalization = [row for row in deltas if row["comparison"] == "layer_normalization_policy"]
            self.assertEqual(len(normalization), 4)
            self.assertTrue(all(float(row["delta_validation_p95_rmse"]) < 0 for row in normalization))
            self.assertEqual(len(runtime.read_csv(analysis / "partial_codebook_progression.csv")), 56)

            image_links = re.findall(r"!\[[^]]*\]\(([^)]+)\)", text)
            self.assertEqual(len(image_links), 12)
            for relative in image_links:
                self.assertTrue((report.parent / relative).is_file(), relative)

            first_text = text
            first_html = html_report.read_bytes()
            first_deltas = (analysis / "matched_factor_deltas.csv").read_bytes()
            strategy_report.analyze_partial_strategy_grid(
                run_dir=source,
                analysis_output_dir=analysis,
                report_path=report,
                html_report_path=html_report,
                image_dir=images,
            )
            self.assertEqual(report.read_text(encoding="utf-8"), first_text)
            self.assertEqual(html_report.read_bytes(), first_html)
            self.assertEqual((analysis / "matched_factor_deltas.csv").read_bytes(), first_deltas)
            self.assertEqual(_fingerprint(source), before)

    def test_partial_report_rejects_source_run_destinations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "legacy_fragment"
            _write_partial_fixture(source)
            with self.assertRaisesRegex(ValueError, "outside the immutable source run"):
                strategy_report.analyze_partial_strategy_grid(
                    run_dir=source,
                    analysis_output_dir=source / "analysis",
                    report_path=Path(tmp) / "report.md",
                    image_dir=Path(tmp) / "images",
                )
            with self.assertRaisesRegex(ValueError, "outside the immutable source run"):
                strategy_report.analyze_partial_strategy_grid(
                    run_dir=source,
                    analysis_output_dir=Path(tmp) / "analysis",
                    report_path=Path(tmp) / "report.md",
                    html_report_path=source / "report.html",
                    image_dir=Path(tmp) / "images",
                )

    def test_complete_13a_report_keeps_final_analysis_gated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "complete_13a"
            baseline = root / "legacy_prefix"
            expected, selection = _write_complete_13a_report_fixture(source, baseline)
            thresholds = root / "strict_perfect_threshold_sweep.csv"
            _write_threshold_fixture(thresholds, expected)
            before = _fingerprint(source)
            analysis = root / "analysis"
            report = root / "reports" / "EXPERIMENT_13A.md"
            html_report = root / "reports" / "EXPERIMENT_13A.html"
            images = root / "reports" / "images" / "13a"

            result = strategy_report.write_complete_13a_report(
                source_run=source,
                analysis_output_dir=analysis,
                report_path=report,
                html_report_path=html_report,
                image_dir=images,
                expected_rows=expected,
                selection=selection,
                scaling_baseline_run=baseline,
                strict_thresholds_path=thresholds,
            )

            self.assertEqual(_fingerprint(source), before)
            self.assertEqual(Path(result["report"]), report.resolve())
            text = report.read_text(encoding="utf-8")
            self.assertIn("13A complete · 90/90 rows", text)
            self.assertIn("automatic epsilon selector did not pass", text)
            self.assertIn("restricted pilot", text)
            self.assertIn("reducing 13B from 90 to 45 rows", text)
            self.assertIn("Lock Experiment 13B to the 45 `LayerClip0To1` counterparts", text)
            self.assertIn("Training-Data Scaling Ablation", text)
            self.assertIn("Metric agreement and disagreement", text)
            self.assertIn("Train-to-Validation Stability", text)
            self.assertIn("Factor interactions by construction family", text)
            self.assertIn("Residual-Layer Learning Curve", text)
            self.assertIn("Decoder and Dictionary Diagnostics", text)
            self.assertIn("Offline work efficiency", text)
            self.assertIn("Audit boundaries", text)
            self.assertIn("Strict-perfect threshold sensitivity", text)
            self.assertIn("`1e-3`", text)
            self.assertNotIn("The frozen eligibility epsilon is", text)

            html = html_report.read_text(encoding="utf-8")
            self.assertIn("13A COMPLETE", html)
            self.assertIn("Complete grid coverage", html)
            self.assertIn("Training-data scaling", html)
            self.assertIn("pilot required", html)
            self.assertIn("45 LayerClip0To1-only rows", html)
            self.assertIn("chartScaling", html)
            self.assertIn("Metric agreement and tension", html)
            self.assertIn("Train-to-validation stability", html)
            self.assertIn("Family-specific interactions", html)
            self.assertIn("Residual-layer progression", html)
            self.assertIn("Decoder and dictionary behavior", html)
            self.assertIn("Offline work efficiency", html)
            self.assertIn("chartMetricMap", html)
            self.assertIn("chartGeneralization", html)
            self.assertIn("chartInteractions", html)
            self.assertIn("chartMarginal", html)
            self.assertIn("chartLayers", html)
            self.assertIn("chartDiagnostics", html)
            self.assertIn("chartWork", html)
            self.assertIn("Strict-perfect tolerance", html)
            self.assertIn("data-strict-tolerance", html)
            self.assertIn("applyStrictTolerance", html)
            self.assertNotIn(str(root.resolve()), html)
            self.assertLess(html_report.stat().st_size, 1_000_000)
            marker = 'application/json">'
            start = html.index(marker) + len(marker)
            payload = json.loads(html[start:html.index("</script>", start)])
            self.assertEqual(payload["meta"]["status"], "complete_13a_pending_13b")
            self.assertEqual(payload["meta"]["completed_rows"], 90)
            self.assertTrue(payload["meta"]["epsilon_selection_attempted"])
            self.assertFalse(payload["meta"]["epsilon_selected"])
            self.assertEqual(len(payload["tables"]["metrics"]), 90)
            self.assertEqual(len(payload["tables"]["coverage"]), 90)
            self.assertEqual(len(payload["tables"]["partial_codebook"]), 630)
            self.assertEqual(len(payload["tables"]["scaling"]), 4)
            self.assertEqual(len(payload["tables"]["diagnostics"]), 90)
            self.assertEqual(len(payload["tables"]["rankings"]), 90)
            self.assertEqual(len(payload["deep_analysis"]["marginal_atoms"]), 540)
            self.assertEqual(len(payload["deep_analysis"]["layer_progression"]), 1440)
            self.assertEqual(len(payload["deep_analysis"]["mechanisms"]), 90)
            self.assertEqual(payload["strict_thresholds"]["tolerances"], ["1e-2", "1e-3", "1e-4", "1e-5"])
            self.assertEqual(len(payload["strict_thresholds"]["rates_by_row"]), 90)
            self.assertIn("chartStrictThresholds", html)
            self.assertIn("chartCapacityDepth", html)
            self.assertIn("chartCapacityMatrix", html)
            self.assertIn("chartToleranceMatrix", html)
            self.assertIn("layer_coverage_by_row", payload["calibration"])
            self.assertIn("strict_perfect_threshold_sensitivity.png", text)
            self.assertEqual(len(runtime.read_csv(analysis / "training_data_scaling_ablation.csv")), 4)
            self.assertEqual(len(runtime.read_csv(analysis / "strategy_diagnostics.csv")), 90)
            self.assertEqual(len(runtime.read_csv(analysis / "metric_rankings.csv")), 90)
            self.assertEqual(len(runtime.read_csv(analysis / "marginal_atom_value.csv")), 540)
            self.assertEqual(len(runtime.read_csv(analysis / "residual_layer_progression.csv")), 1440)
            self.assertEqual(len(runtime.read_csv(analysis / "construction_mechanism_diagnostics.csv")), 90)
            self.assertGreater(len(runtime.read_csv(analysis / "factor_interaction_summary.csv")), 0)
            with self.assertRaisesRegex(grid.AnalysisNotReadyError, "complete 13A and 13B"):
                grid.analyze_strategy_grid(run_dir=source)

            first_html = html_report.read_bytes()
            strategy_report.write_complete_13a_report(
                source_run=source,
                analysis_output_dir=analysis,
                report_path=report,
                html_report_path=html_report,
                image_dir=images,
                expected_rows=expected,
                selection=selection,
                scaling_baseline_run=baseline,
                strict_thresholds_path=thresholds,
            )
            self.assertEqual(html_report.read_bytes(), first_html)
            self.assertEqual(_fingerprint(source), before)


def _write_complete_13a_report_fixture(
    source: Path,
    baseline: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    specs = grid.experiment13a_specs()
    summaries: list[dict[str, object]] = []
    partial: list[dict[str, object]] = []
    layer_quantiles: list[dict[str, object]] = []
    slot_quantiles: list[dict[str, object]] = []
    coverage: list[dict[str, object]] = []
    retired: list[dict[str, object]] = []
    slot_progression: list[dict[str, object]] = []
    atom_construction: list[dict[str, object]] = []
    candidate_search: list[dict[str, object]] = []
    for index, spec in enumerate(specs):
        clip = -0.01 if spec.layer_normalization_policy == "LayerClip0To1" else 0.0
        budget = -0.001 if spec.utility_candidate_budget == "CandidateBudget48" else 0.0
        schedule = -0.002 if spec.layer_schedule == "TwoPhase" else 0.0
        p95 = 0.08 + (index % 9) * 0.0005 + clip + budget + schedule
        summary = {
            **spec.manifest_dict("fixture", "fixture"),
            "validation_median_rmse": p95 / 2.0,
            "validation_strict_perfect_lfo_rate": 0.02 if spec.construction_policy == "CommonCaseRepair" else 0.01,
            "validation_p95_rmse": p95,
            "validation_node_max_error_p95": p95 * 2.0,
            "validation_p99_rmse": p95 * 1.1,
            "validation_max_rmse": p95 * 1.2,
            "validation_max_abs_error_p95": p95 * 2.0,
            "train_median_rmse": p95 / 2.0 - 0.0005,
            "train_strict_perfect_lfo_rate": 0.02,
            "train_p95_rmse": p95 + 0.001,
            "train_node_max_error_p95": p95 * 2.0 + 0.002,
            "validation_overshoot_rate_before_final_clip": 0.0 if spec.layer_normalization_policy == "LayerClip0To1" else 0.1,
            "validation_overshoot_abs_p95_before_final_clip": 0.0 if spec.layer_normalization_policy == "LayerClip0To1" else 0.02,
            "residual_layer_dead_atom_rate_median": 0.01,
            "residual_layer_dominant_atom_share_median": 0.3,
            "residual_layer_usage_entropy_median": 2.0,
            "residual_layer_no_op_usage_rate_median": 0.01,
            "residual_layer_effective_no_op_usage_rate_median": 0.02,
            "residual_gain_median": 0.05,
            "residual_gain_abs_p95": 0.3,
            "residual_gain_nonzero_rate": 0.98,
            "duplicate_atom_rate": 0.0,
            "head_outputs_actual": 193,
            "oracle_construction_time": 20.0 + index,
            "train_encoding_time": 2.0,
            "validation_encoding_time": 1.0,
        }
        summaries.append(summary)
        partial.extend(
            {
                "experiment_phase": "13A",
                "row_id": spec.row_id,
                "pair_id": spec.pair_id,
                "active_atom_count": active,
                "validation_median_rmse": p95 / 2.0 + (7 - active) * 0.001,
                "validation_strict_perfect_lfo_rate": summary["validation_strict_perfect_lfo_rate"],
                "validation_p95_rmse": p95 + (7 - active) * 0.002,
                "validation_node_max_error_p95": p95 * 2.0,
            }
            for active in range(1, 8)
        )
        common = {"experiment_phase": "13A", "row_id": spec.row_id, "pair_id": spec.pair_id, "dataset_split": "training"}
        for depth in (8, 16):
            for target in (0.1, 0.5):
                layer_quantiles.append({**common, "residual_layer": depth, "percentile": target, "epsilon_value": 0.01 / depth + target * 0.001})
        slot_quantiles.append({**common, "residual_layer": 1, "active_atom_slot": 1, "percentile": 0.5, "epsilon_value": 0.01})
        coverage.extend([
            {**common, "residual_layer": 1, "active_atom_slot": "", "epsilon": 0.001, "resolved_fraction": 0.02},
            {**common, "residual_layer": 1, "active_atom_slot": 1, "epsilon": 0.001, "resolved_fraction": 0.02},
        ])
        for depth in (8, 16):
            for epsilon in (0.001, 0.0025):
                coverage.append({**common, "residual_layer": depth, "active_atom_slot": "", "epsilon": epsilon, "resolved_fraction": 0.02 + depth * epsilon})
        retired.append({
            **common,
            "residual_layer": 1,
            "active_atom_slot": 1,
            "epsilon": 0.001,
            "retired_lfo_fraction": 0.02,
            "incoming_retired_energy_fraction": 0.001,
            "unexplained_retired_energy_fraction": 0.0001,
        })
        for layer in range(1, 17):
            slot_progression.append({
                "experiment_phase": "13A",
                "row_id": spec.row_id,
                "pair_id": spec.pair_id,
                "residual_layer": layer,
                "active_atom_slot": 7,
                "eligible_residual_count": 100,
                "training_median_rmse": p95 / (layer + 1),
                "training_p95_rmse": p95 / (1.0 + layer * 0.2),
                "training_max_abs_error_p95": p95 * 2.0,
            })
            atom_construction.append({
                "experiment_phase": "13A",
                "row_id": spec.row_id,
                "pair_id": spec.pair_id,
                "residual_layer": layer,
                "slot_index": 1,
                "layer_role": "Repair" if layer % 2 == 0 else "Broad",
                "atom_source_kind": "observed_residual" if layer % 2 == 0 else "synthesized_prototype",
                "training_p95_rmse_before": p95 + 0.002,
                "training_p95_rmse_after": p95,
                "exact_duplicate_alignment_reused": False,
                "prototype_converged": layer % 2 == 1,
                "prototype_iterations_executed": 1,
            })
            candidate_search.append({
                "experiment_phase": "13A",
                "row_id": spec.row_id,
                "pair_id": spec.pair_id,
                "residual_layer": layer,
                "slot_index": 1,
                "candidate_count": 48 if spec.utility_candidate_budget == "CandidateBudget48" else 24 if spec.utility_candidate_budget == "CandidateBudget24" else 0,
            })
    runtime.write_csv(source / "summary.csv", summaries)
    runtime.write_csv(source / "partial_codebook_validation.csv", partial)
    runtime.write_csv(source / "layer_epsilon_quantiles.csv", layer_quantiles)
    runtime.write_csv(source / "slot_epsilon_quantiles.csv", slot_quantiles)
    runtime.write_csv(source / "epsilon_coverage.csv", coverage)
    runtime.write_csv(source / "retired_error_mass.csv", retired)
    runtime.write_csv(source / "slot_progression.csv", slot_progression)
    runtime.write_csv(source / "atom_construction.csv", atom_construction)
    runtime.write_csv(source / "candidate_search_diagnostics.csv", candidate_search)

    matched = summaries[:4]
    for sampled_summary in matched:
        row_id = str(sampled_summary["row_id"])
        runtime.write_csv(source / "rows" / row_id / "summary.csv", [sampled_summary])
        baseline_summary = dict(sampled_summary)
        baseline_summary["validation_median_rmse"] = float(sampled_summary["validation_median_rmse"]) - 0.001
        baseline_summary["validation_p95_rmse"] = float(sampled_summary["validation_p95_rmse"]) - 0.001
        runtime.write_csv(baseline / "rows" / row_id / "summary.csv", [baseline_summary])
    assignments = [
        {"dataset_split": "training", "dataset_index": 1},
        {"dataset_split": "validation", "dataset_index": 10},
        {"dataset_split": "validation", "dataset_index": 11},
    ]
    for sampled_summary in matched:
        row_id = str(sampled_summary["row_id"])
        runtime.write_csv(source / "rows" / row_id / "atom_assignments.csv", assignments)
        runtime.write_csv(baseline / "rows" / row_id / "atom_assignments.csv", assignments)

    selection = {
        "selection_passed": False,
        "selected_epsilon": None,
        "selection_notes": "no candidate epsilon satisfied all automatic selection conditions; restricted pilot required",
        "training_statistics_used": {
            "candidate_statistics": {
                "0.001": {
                    "max_early_middle_median_retired_lfo_fraction": 0.02,
                    "median_unexplained_retired_energy_fraction": 0.0001,
                    "p95_unexplained_retired_energy_fraction": 0.001,
                }
            }
        },
    }
    return [dict(spec.manifest_dict("fixture", "fixture")) for spec in specs], selection


def _write_partial_fixture(source: Path) -> None:
    specs = [
        spec
        for spec in grid.experiment13a_specs()
        if spec.construction_family == "BroadMeanGlobalRepair"
    ]
    self_contained = [
        spec
        for spec in specs
        if spec.layer_schedule in {"Interleaved", "TwoPhase"}
        and spec.utility_candidate_budget in {"CandidateBudget24", "CandidateBudget48"}
        and spec.layer_normalization_policy in {"FinalClipOnly", "LayerClip0To1"}
    ]
    if len(self_contained) != 8:
        raise AssertionError("fixture requires the complete eight-row BroadMeanGlobalRepair cell")
    for index, spec in enumerate(self_contained):
        row_dir = source / "rows" / spec.row_id
        schedule = 0.001 if spec.layer_schedule == "TwoPhase" else 0.0
        budget = -0.002 if spec.utility_candidate_budget == "CandidateBudget48" else 0.0
        clipping = -0.01 if spec.layer_normalization_policy == "LayerClip0To1" else 0.0
        p95 = 0.1 + schedule + budget + clipping
        summary = {
            **spec.manifest_dict("fixture", "fixture"),
            "validation_median_rmse": p95 / 2.0,
            "validation_strict_perfect_lfo_rate": 0.01 + index * 0.0001,
            "validation_p95_rmse": p95,
            "validation_node_max_error_p95": p95 * 2.0,
            "validation_p99_rmse": p95 * 1.1,
            "validation_max_rmse": p95 * 1.2,
            "validation_max_abs_error_p95": p95 * 2.0,
            "oracle_construction_time": 10.0 + index,
            "train_encoding_time": 2.0,
            "validation_encoding_time": 1.0,
        }
        runtime.write_csv(row_dir / "summary.csv", [summary])
        runtime.write_csv(
            row_dir / "partial_codebook_validation.csv",
            [
                {
                    "experiment_phase": "13A",
                    "row_id": spec.row_id,
                    "pair_id": spec.pair_id,
                    "active_atom_count": active,
                    "validation_median_rmse": p95 / 2.0 + (7 - active) * 0.001,
                    "validation_strict_perfect_lfo_rate": 0.01,
                    "validation_p95_rmse": p95 + (7 - active) * 0.002,
                    "validation_node_max_error_p95": p95 * 2.0,
                }
                for active in range(1, 8)
            ],
        )
        runtime.write_csv(
            row_dir / "layer_epsilon_quantiles.csv",
            [{"experiment_phase": "13A", "row_id": spec.row_id, "dataset_split": "training", "residual_layer": 1, "percentile": 0.5, "epsilon_value": p95}],
        )
        runtime.write_csv(
            row_dir / "slot_epsilon_quantiles.csv",
            [{"experiment_phase": "13A", "row_id": spec.row_id, "dataset_split": "training", "active_atom_slot": 0, "percentile": 0.5, "epsilon_value": p95}],
        )
        runtime.write_csv(
            row_dir / "epsilon_coverage.csv",
            [
                {"experiment_phase": "13A", "row_id": spec.row_id, "dataset_split": "training", "residual_layer": 1, "active_atom_slot": "", "epsilon": 0.001, "resolved_fraction": 0.1},
                {"experiment_phase": "13A", "row_id": spec.row_id, "dataset_split": "training", "residual_layer": 1, "active_atom_slot": 0, "epsilon": 0.001, "resolved_fraction": 0.05},
            ],
        )
        runtime.write_csv(
            row_dir / "retired_error_mass.csv",
            [{"experiment_phase": "13A", "row_id": spec.row_id, "epsilon": 0.001, "retired_lfo_fraction": 0.1, "incoming_retired_energy_fraction": 0.02, "unexplained_retired_energy_fraction": 0.01}],
        )


def _write_threshold_fixture(path: Path, expected: list[dict[str, object]]) -> None:
    rows = []
    for index, spec in enumerate(expected):
        for tolerance, base in ((1e-2, 0.80), (1e-3, 0.45), (1e-4, 0.12), (1e-5, 0.02 if spec["construction_policy"] == "CommonCaseRepair" else 0.01)):
            rate = base + (index % 5) * (0.001 if tolerance != 1e-5 else 0.0)
            rows.append({
                "schema_version": "experiment13_strict_threshold_sweep_v1",
                "row_id": spec["row_id"],
                "dataset_split": "validation",
                "max_abs_tolerance": tolerance,
                "rmse_tolerance": tolerance / 10,
                "strict_perfect_lfo_count": round(rate * 1605),
                "row_count": 1605,
                "strict_perfect_lfo_rate": rate,
            })
    runtime.write_csv(path, rows)


def _fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.as_posix()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


if __name__ == "__main__":
    unittest.main()

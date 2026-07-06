from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from lfo_era2.component_ladder import (  # noqa: E402
    D,
    W,
    ComponentEncoding,
    CONSTRUCTION_POLICY_VALUES,
    LAYER_NORMALIZATION_POLICY_VALUES,
    PATH_SEARCH_POLICY_VALUES,
    SCALAR_SCHEMA_VALUES,
    _usage_summary,
    analyze_component_ladder,
    default_component_specs,
    budget_for_spec,
    run_component_ladder,
    validate_residual_gain_contract,
)
from lfo_era2.dataset import make_tiny_curve_dataset  # noqa: E402


class ComponentLadderTests(unittest.TestCase):
    def test_default_specs_are_fixed_w8d16_ladder(self) -> None:
        specs = default_component_specs()
        self.assertEqual(len(specs), 74)
        self.assertTrue(all(W == 8 and D == 16 for _ in specs))
        self.assertTrue(all(spec.scalar_schema in SCALAR_SCHEMA_VALUES for spec in specs))
        self.assertTrue(all(spec.screening_value[:1].isupper() for spec in specs))
        self.assertEqual([spec.path_search_policy for spec in specs if spec.screening_variable == "path_search_policy"][::2], list(PATH_SEARCH_POLICY_VALUES))
        self.assertEqual([spec.construction_policy for spec in specs if spec.screening_variable == "construction_policy"][::2], list(CONSTRUCTION_POLICY_VALUES))
        self.assertEqual([spec.layer_normalization_policy for spec in specs if spec.screening_variable == "layer_normalization_policy"][::2], list(LAYER_NORMALIZATION_POLICY_VALUES))
        self.assertIn("GreedyPath", {spec.path_search_policy for spec in specs if spec.screening_variable == "path_search_policy"})
        self.assertNotIn("OutlierChaser", {spec.construction_policy for spec in specs})

    def test_budget_accounting_matches_component_scalars(self) -> None:
        indices = _spec("construction_policy", "BestOverallRepair", "IndicesOnly")
        phase_gain = _spec("construction_policy", "BestOverallRepair", "PhaseAndResidualGain")
        greedy = _spec("path_search_policy", "GreedyPath", "IndicesOnly")
        beam8 = _spec("path_search_policy", "Beam8Path", "IndicesOnly")
        self.assertEqual(budget_for_spec(indices)["head_outputs_actual"], 160)
        self.assertEqual(budget_for_spec(phase_gain)["head_outputs_actual"], 193)
        self.assertEqual(greedy.beam_width, 1)
        self.assertEqual(budget_for_spec(beam8)["head_outputs_actual"], 160)

    def test_optimized_residual_gain_must_be_model_facing(self) -> None:
        with self.assertRaises(ValueError):
            validate_residual_gain_contract("optimized", model_facing=False)
        validate_residual_gain_contract("optimized", model_facing=True)
        validate_residual_gain_contract("fixed", model_facing=False)

    def test_phase_disabled_schema_has_no_phase_or_gain_targets(self) -> None:
        spec = _spec("construction_policy", "BestOverallRepair", "IndicesOnly")
        encoding = _empty_test_encoding(4)
        schema = encoding.target_schema(spec)
        names = [field["name"] for field in schema["fields"]]
        self.assertIn("base_index", names)
        self.assertIn("residual_layer_16_index", names)
        self.assertNotIn("base_phase", names)
        self.assertNotIn("residual_layer_1_phase", names)
        self.assertNotIn("residual_layer_1_gain", names)

    def test_phase_gain_schema_includes_only_model_facing_scalars(self) -> None:
        spec = _spec("construction_policy", "BestOverallRepair", "PhaseAndResidualGain")
        schema = _empty_test_encoding(4).target_schema(spec)
        names = [field["name"] for field in schema["fields"]]
        self.assertIn("base_phase", names)
        self.assertIn("residual_layer_1_phase", names)
        self.assertIn("residual_layer_1_gain", names)

    def test_topology_balanced_row_is_offline_only(self) -> None:
        spec = _spec("construction_policy", "FamilyBalancedRepair", "IndicesOnly")
        self.assertTrue(spec.topology_used_in_construction)
        self.assertEqual(spec.construction_policy, "FamilyBalancedRepair")
        self.assertEqual(budget_for_spec(spec)["head_outputs_actual"], 160)

    def test_smoke_run_writes_report_artifacts_and_schema(self) -> None:
        dataset = make_tiny_curve_dataset(resolution=17, row_count=30)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = run_component_ladder(
                output_dir=root / "artifacts",
                dataset=dataset,
                backend="numpy",
                smoke=True,
                row_ids={
                    "x12_screen_construction_policy_BestOverallRepair_IndicesOnly",
                    "x12_screen_construction_policy_BestOverallRepair_PhaseAndResidualGain",
                },
                chunk_size=8,
                report_path=root / "reports" / "EXPERIMENT_12.md",
                report_image_dir=root / "reports" / "images",
                progress=None,
            )
            self.assertTrue(Path(result["summary"]).exists())
            self.assertTrue(Path(result["component_deltas"]).exists())
            self.assertTrue(Path(result["report"]).exists())
            schema = (root / "artifacts" / "rows" / "x12_screen_construction_policy_BestOverallRepair_IndicesOnly" / "targets_schema.json").read_text(encoding="utf-8")
            self.assertNotIn("phase", schema)
            text = Path(result["report"]).read_text(encoding="utf-8")
            self.assertIn("## Main Findings", text)
            self.assertIn("## Why This Happens", text)
            self.assertIn("## Independent Variable Chapters", text)
            self.assertIn("## Grouped Evidence Tables", text)
            phase_text = Path(result["phase_gain_report"]).read_text(encoding="utf-8")
            self.assertIn("PhaseAndResidualGain Screening Read", phase_text)
            self.assertNotIn("ScalarSchema", phase_text)
            manifest = (root / "artifacts" / "rows" / "x12_screen_construction_policy_BestOverallRepair_IndicesOnly" / "manifest.json").read_text(encoding="utf-8")
            self.assertIn("NoOpAtom", manifest)
            self.assertIn("BestOverallRepair", manifest)

    def test_non_default_screening_row_runs_with_pascal_values(self) -> None:
        dataset = make_tiny_curve_dataset(resolution=17, row_count=30)
        row_id = "x12_screen_layer_normalization_policy_LayerClipNeg0p1To1p1_PhaseAndResidualGain"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = run_component_ladder(
                output_dir=root / "artifacts",
                dataset=dataset,
                backend="numpy",
                smoke=True,
                row_ids={row_id},
                chunk_size=8,
                report_path=root / "reports" / "EXPERIMENT_12.md",
                report_image_dir=root / "reports" / "images",
                progress=None,
            )
            self.assertTrue(Path(result["screening_results"]).exists())
            summary = (root / "artifacts" / "rows" / row_id / "summary.csv").read_text(encoding="utf-8")
            self.assertIn("LayerClipNeg0p1To1p1", summary)
            self.assertIn("PhaseAndResidualGain", summary)
            self.assertIn("validation_overshoot_rate_before_final_clip", summary)
            self.assertIn("residual_layer_effective_no_op_usage_rate_median", summary)

    def test_analyze_filters_legacy_rows(self) -> None:
        current = _spec("construction_policy", "BestOverallRepair", "IndicesOnly")
        rows = [
            {
                "row_id": current.row_id,
                "row_number": 1,
                "screening_variable": current.screening_variable,
                "screening_value": current.screening_value,
                "scalar_schema": current.scalar_schema,
            },
            {"row_id": "x12_add_phase", "row_number": 999, "screening_variable": "legacy"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = analyze_component_ladder(output_dir=root / "artifacts", rows=rows, write_report=False)
            summary = (Path(result["summary"]).read_text(encoding="utf-8"))
            self.assertIn(current.row_id, summary)
            self.assertNotIn("x12_add_phase", summary)

    def test_effective_no_op_counts_near_zero_gain_active_atoms(self) -> None:
        encoding = ComponentEncoding(
            base_index=np.zeros(4, dtype=np.int32),
            base_phase=np.zeros(4, dtype=np.float32),
            base_gain=np.ones(4, dtype=np.float32),
            residual_layer_indices=[np.asarray([0, 1, 2, 3], dtype=np.int32) for _ in range(D)],
            residual_layer_phases=[np.zeros(4, dtype=np.float32) for _ in range(D)],
            residual_layer_gains=[np.asarray([0.0, 0.0, 1e-5, 1e-3], dtype=np.float32) for _ in range(D)],
        )
        usage = _usage_summary(encoding, widths=[W for _ in range(D)])
        self.assertEqual(usage["residual_layer_no_op_usage_rate_median"], 0.25)
        self.assertEqual(usage["residual_layer_effective_no_op_usage_rate_median"], 0.75)


def _spec(variable: str, value: str, scalar_schema: str):
    return next(
        spec
        for spec in default_component_specs()
        if spec.screening_variable == variable and spec.screening_value == value and spec.scalar_schema == scalar_schema
    )


def _empty_test_encoding(row_count: int) -> ComponentEncoding:
    return ComponentEncoding(
        base_index=np.zeros(row_count, dtype=np.int32),
        base_phase=np.zeros(row_count, dtype=np.float32),
        base_gain=np.ones(row_count, dtype=np.float32),
        residual_layer_indices=[np.zeros(row_count, dtype=np.int32) for _ in range(D)],
        residual_layer_phases=[np.zeros(row_count, dtype=np.float32) for _ in range(D)],
        residual_layer_gains=[np.ones(row_count, dtype=np.float32) for _ in range(D)],
    )


if __name__ == "__main__":
    unittest.main()

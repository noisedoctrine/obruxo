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
    default_component_specs,
    budget_for_spec,
    run_component_ladder,
    validate_residual_gain_contract,
)
from lfo_era2.dataset import make_tiny_curve_dataset  # noqa: E402


class ComponentLadderTests(unittest.TestCase):
    def test_default_specs_are_fixed_w8d16_ladder(self) -> None:
        specs = default_component_specs()
        self.assertEqual(
            [spec.row_id for spec in specs],
            [
                "x12_c0_indices_only",
                "x12_add_phase",
                "x12_add_residual_gain",
                "x12_add_beam4",
                "x12_add_utility_construction",
                "x12_add_topology_balanced_utility_construction",
                "x12_phase_gain",
                "x12_phase_beam4",
                "x12_gain_beam4",
                "x12_phase_gain_beam4",
                "x12_phase_gain_beam4_utility",
                "x12_phase_gain_beam4_topology_balanced_utility",
            ],
        )
        self.assertTrue(all(W == 8 and D == 16 for _ in specs))

    def test_budget_accounting_matches_component_scalars(self) -> None:
        specs = {spec.row_id: spec for spec in default_component_specs()}
        self.assertEqual(budget_for_spec(specs["x12_c0_indices_only"])["head_outputs_actual"], 160)
        self.assertEqual(budget_for_spec(specs["x12_add_phase"])["head_outputs_actual"], 177)
        self.assertEqual(budget_for_spec(specs["x12_add_residual_gain"])["head_outputs_actual"], 176)
        self.assertEqual(budget_for_spec(specs["x12_phase_gain"])["head_outputs_actual"], 193)
        self.assertEqual(budget_for_spec(specs["x12_add_beam4"])["head_outputs_actual"], 160)
        self.assertEqual(budget_for_spec(specs["x12_add_utility_construction"])["head_outputs_actual"], 160)

    def test_optimized_residual_gain_must_be_model_facing(self) -> None:
        with self.assertRaises(ValueError):
            validate_residual_gain_contract("optimized", model_facing=False)
        validate_residual_gain_contract("optimized", model_facing=True)
        validate_residual_gain_contract("fixed", model_facing=False)

    def test_phase_disabled_schema_has_no_phase_or_gain_targets(self) -> None:
        spec = default_component_specs()[0]
        encoding = _empty_test_encoding(4)
        schema = encoding.target_schema(spec)
        names = [field["name"] for field in schema["fields"]]
        self.assertIn("base_index", names)
        self.assertIn("residual_layer_16_index", names)
        self.assertNotIn("base_phase", names)
        self.assertNotIn("residual_layer_1_phase", names)
        self.assertNotIn("residual_layer_1_gain", names)

    def test_phase_gain_schema_includes_only_model_facing_scalars(self) -> None:
        spec = next(item for item in default_component_specs() if item.row_id == "x12_phase_gain")
        schema = _empty_test_encoding(4).target_schema(spec)
        names = [field["name"] for field in schema["fields"]]
        self.assertIn("base_phase", names)
        self.assertIn("residual_layer_1_phase", names)
        self.assertIn("residual_layer_1_gain", names)

    def test_topology_balanced_row_is_offline_only(self) -> None:
        spec = next(item for item in default_component_specs() if item.row_id == "x12_add_topology_balanced_utility_construction")
        self.assertTrue(spec.topology_used_in_construction)
        self.assertEqual(spec.construction_policy, "topology_balanced_utility")
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
                row_ids={"x12_c0_indices_only"},
                chunk_size=8,
                report_path=root / "reports" / "EXPERIMENT_12.md",
                report_image_dir=root / "reports" / "images",
                progress=None,
            )
            self.assertTrue(Path(result["summary"]).exists())
            self.assertTrue(Path(result["component_deltas"]).exists())
            self.assertTrue(Path(result["report"]).exists())
            self.assertTrue((root / "reports" / "images" / "experiment12_validation_p95_by_row.png").exists())
            schema = (root / "artifacts" / "rows" / "x12_c0_indices_only" / "targets_schema.json").read_text(encoding="utf-8")
            self.assertNotIn("phase", schema)
            text = Path(result["report"]).read_text(encoding="utf-8")
            self.assertIn("## Main Findings", text)
            self.assertNotIn("| row", text)


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

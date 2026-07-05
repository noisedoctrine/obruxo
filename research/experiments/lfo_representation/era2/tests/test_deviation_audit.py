from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from lfo_era2.alignment import alignment_matrix  # noqa: E402
from lfo_era2.curve import circular_shift  # noqa: E402
from lfo_era2.dataset import make_tiny_curve_dataset  # noqa: E402
from lfo_era2.deviation_audit import (  # noqa: E402
    CONTROL_POINT_COUNT,
    D,
    W,
    DiagnosticRowSpec,
    _run_topology_free_row,
    _summary_base,
    _write_report_section,
)


class DeviationAuditTests(unittest.TestCase):
    def test_exact_phase_beats_lattice_for_subsample_shift(self) -> None:
        x = np.linspace(0.0, 1.0, 17, dtype=np.float32)
        code = (0.5 + 0.4 * np.sin(2.0 * np.pi * x)).astype(np.float32)
        target = circular_shift(code, 0.21)
        lattice = alignment_matrix(target, code[None, :], phase_policy="fft_lattice", gain_policy="fixed")
        exact = alignment_matrix(target, code[None, :], phase_policy="exact", gain_policy="fixed")
        self.assertLess(float(exact.losses[0, 0]), float(lattice.losses[0, 0]))

    def test_topology_balanced_construction_keeps_runtime_contract_clean(self) -> None:
        dataset = make_tiny_curve_dataset(resolution=17, row_count=30).subset(train_count=12, validation_count=6)
        spec = DiagnosticRowSpec(
            row_id="topology_balanced_test",
            description="test",
            construction_policy="topology_balanced_farthest",
            topology_used_in_construction=True,
        )
        summary, _, _ = _run_topology_free_row(
            spec,
            dataset,
            backend="numpy",
            chunk_size=8,
            progress=None,
        )
        self.assertTrue(summary["topology_contract_pass"])
        self.assertTrue(summary["runtime_contract_valid"])
        self.assertTrue(summary["topology_used_in_construction"])
        self.assertFalse(summary["topology_used_at_runtime"])
        self.assertFalse(summary["topology_used_in_decoder_lookup"])

    def test_quarantined_topology_runtime_is_marked_invalid(self) -> None:
        dataset = make_tiny_curve_dataset(resolution=17, row_count=12)
        spec = DiagnosticRowSpec(
            row_id="quarantined",
            description="invalid",
            topology_runtime=True,
            topology_used_in_construction=True,
        )
        summary = _summary_base(spec, dataset, contract_pass=False)
        self.assertFalse(summary["topology_contract_pass"])
        self.assertFalse(summary["runtime_contract_valid"])
        self.assertTrue(summary["topology_used_at_runtime"])
        self.assertTrue(summary["topology_used_in_decoder_lookup"])

    def test_w8d16_phase_and_residual_gain_budget_rows_are_explicit(self) -> None:
        dataset = make_tiny_curve_dataset(resolution=17, row_count=12)
        phase_only = _summary_base(DiagnosticRowSpec(row_id="phase", description=""), dataset, contract_pass=True)
        gain = _summary_base(
            DiagnosticRowSpec(
                row_id="gain",
                description="",
                residual_gain_policy="optimized",
                residual_gain_model_facing=True,
            ),
            dataset,
            contract_pass=True,
        )
        self.assertEqual(phase_only["W"], W)
        self.assertEqual(phase_only["D"], D)
        self.assertEqual(phase_only["lfo_control_point_count"], CONTROL_POINT_COUNT)
        self.assertEqual(phase_only["head_outputs_actual"], 177)
        self.assertEqual(gain["head_outputs_actual"], 193)
        self.assertIn("residual_gain_scalars", gain["head_outputs_formula"])

    def test_report_section_can_be_inserted_without_tables(self) -> None:
        rows = [
            {
                "row_id": "current_endpoint_excluded_lattice_greedy_farthest",
                "description": "baseline",
                "validation_p95_rmse": 0.2,
                "validation_median_rmse": 0.1,
                "head_outputs_actual": 177,
                "topology_used_at_runtime": False,
            },
            {
                "row_id": "exact_phase_only",
                "description": "exact phase",
                "validation_p95_rmse": 0.15,
                "validation_median_rmse": 0.08,
                "head_outputs_actual": 177,
                "topology_used_at_runtime": False,
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.md"
            path.write_text("# Experiment 11 Flat-Categorical Report\n\n## Main Findings\nold\n", encoding="utf-8")
            _write_report_section(path, rows, [])
            text = path.read_text(encoding="utf-8")
            self.assertIn("## W8D16 Deviation Audit", text)
            self.assertIn("metric_delta", text)
            self.assertNotIn("| row", text)


if __name__ == "__main__":
    unittest.main()


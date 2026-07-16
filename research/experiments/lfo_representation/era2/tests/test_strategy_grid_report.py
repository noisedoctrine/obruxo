from __future__ import annotations

import hashlib
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
            images = root / "reports" / "images" / "experiment_13" / "provisional"

            result = grid.analyze_partial_strategy_grid(
                run_dir=source,
                analysis_output_dir=analysis,
                report_path=report,
                image_dir=images,
            )

            self.assertEqual(_fingerprint(source), before)
            self.assertEqual(Path(result["report"]), report.resolve())
            self.assertFalse((source / "summary.csv").exists())
            text = report.read_text(encoding="utf-8")
            self.assertIn("Provisional evidence only", text)
            self.assertIn("`8/90`", text)
            self.assertIn("no eligibility epsilon has been selected", text)
            self.assertIn("Zero completed rows are available", text)
            self.assertNotIn("The frozen eligibility epsilon is", text)

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
            first_deltas = (analysis / "matched_factor_deltas.csv").read_bytes()
            grid.analyze_partial_strategy_grid(
                run_dir=source,
                analysis_output_dir=analysis,
                report_path=report,
                image_dir=images,
            )
            self.assertEqual(report.read_text(encoding="utf-8"), first_text)
            self.assertEqual((analysis / "matched_factor_deltas.csv").read_bytes(), first_deltas)
            self.assertEqual(_fingerprint(source), before)

    def test_partial_report_rejects_source_run_destinations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "legacy_fragment"
            _write_partial_fixture(source)
            with self.assertRaisesRegex(ValueError, "outside the immutable source run"):
                grid.analyze_partial_strategy_grid(
                    run_dir=source,
                    analysis_output_dir=source / "analysis",
                    report_path=Path(tmp) / "report.md",
                    image_dir=Path(tmp) / "images",
                )


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


def _fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.as_posix()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


if __name__ == "__main__":
    unittest.main()

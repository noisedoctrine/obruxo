from __future__ import annotations

from pathlib import Path
import sys
import unittest


ERA2_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ERA2_ROOT / "code"
sys.path.insert(0, str(CODE_DIR))

from lfo_era2 import strategy_grid as grid  # noqa: E402


class StrategyGridEnumerationTests(unittest.TestCase):
    def test_exact_row_and_pair_counts(self) -> None:
        rows_a = grid.experiment13a_specs()
        rows_b = grid.experiment13b_specs(0.001)
        self.assertEqual(len(rows_a), 90)
        self.assertEqual(len(rows_b), 90)
        self.assertEqual(len(rows_a + rows_b), 180)
        self.assertEqual(len({row.row_id for row in rows_a + rows_b}), 180)
        self.assertEqual(len({row.pair_id for row in rows_a + rows_b}), 90)
        grid.validate_pairing(rows_a, rows_b)

    def test_row_and_pair_ids_are_stable(self) -> None:
        first = grid.experiment13a_specs()
        second = grid.experiment13a_specs()
        self.assertEqual([row.row_id for row in first], [row.row_id for row in second])
        self.assertEqual([row.pair_id for row in first], [row.pair_id for row in second])
        self.assertEqual(first[0].row_id, "x13a_common_case_repair_candidate_budget24_final_clip_only")
        self.assertEqual(first[-1].pair_id, "x13_pair_all_dominant_directions_null_layer_clip0_to1")

    def test_each_pair_has_one_row_per_phase_and_only_allowed_differences(self) -> None:
        rows = grid.all_strategy_specs(0.0025)
        by_pair: dict[str, list[grid.StrategyRowSpec]] = {}
        for row in rows:
            by_pair.setdefault(row.pair_id, []).append(row)
        self.assertEqual(len(by_pair), 90)
        for pair in by_pair.values():
            self.assertEqual({row.experiment_phase for row in pair}, {"13A", "13B"})
            self.assertEqual(len(pair), 2)
            a = next(row for row in pair if row.experiment_phase == "13A")
            b = next(row for row in pair if row.experiment_phase == "13B")
            self.assertEqual(a.paired_settings, b.paired_settings)
            self.assertEqual(a.residual_population_policy, "AllResiduals")
            self.assertEqual(b.residual_population_policy, "UnresolvedOnly")
            self.assertIsNone(a.eligibility_epsilon)
            self.assertEqual(b.eligibility_epsilon, 0.0025)

    def test_fixed_runtime_contract_and_threshold_separation(self) -> None:
        rows_a = grid.experiment13a_specs()
        rows_b = grid.experiment13b_specs(0.02)
        self.assertTrue(all(row.finish_threshold == 1e-5 for row in rows_a + rows_b))
        self.assertTrue(all(row.head_outputs_actual == 193 for row in rows_a + rows_b))
        self.assertTrue(all(row.residual_width == 8 and row.residual_depth == 16 for row in rows_a + rows_b))
        self.assertTrue(all(row.reserved_atom == "NoOpAtom" and row.active_atoms_per_layer == 7 for row in rows_a + rows_b))
        self.assertTrue(all(row.scalar_schema == "PhaseAndResidualGain" for row in rows_a + rows_b))
        self.assertTrue(all(row.path_search_policy == "Beam4Path" for row in rows_a + rows_b))
        self.assertTrue(all(row.runtime_topology is None for row in rows_a + rows_b))
        self.assertTrue(all(row.eligibility_epsilon is None for row in rows_a))
        self.assertEqual({row.eligibility_epsilon for row in rows_b}, {0.02})

    def test_layer_schedules_are_layer_level_and_exact(self) -> None:
        interleaved = grid.layer_roles("Interleaved")
        two_phase = grid.layer_roles("TwoPhase")
        self.assertEqual(interleaved, tuple("Broad" if layer % 2 else "Repair" for layer in range(1, 17)))
        self.assertEqual(two_phase, ("Broad",) * 8 + ("Repair",) * 8)
        self.assertEqual(interleaved.count("Broad"), 8)
        self.assertEqual(interleaved.count("Repair"), 8)
        self.assertEqual(two_phase.count("Broad"), 8)
        self.assertEqual(two_phase.count("Repair"), 8)

    def test_effective_budgets_follow_layer_roles(self) -> None:
        interleaved = next(
            row
            for row in grid.experiment13a_specs()
            if row.construction_policy == "BroadMeanGlobalRepairInterleaved"
            and row.utility_candidate_budget == "CandidateBudget48"
            and row.layer_normalization_policy == "FinalClipOnly"
        )
        self.assertEqual(interleaved.effective_candidate_budget_by_layer, (None, 48) * 8)
        self.assertTrue(all(len(layer) == 7 for layer in interleaved.effective_candidate_budget_by_slot))
        for layer_index, layer in enumerate(interleaved.effective_candidate_budget_by_slot):
            expected = None if layer_index % 2 == 0 else 48
            self.assertEqual(layer, (expected,) * 7)

        two_phase = next(
            row
            for row in grid.experiment13a_specs()
            if row.construction_policy == "BroadMeanGlobalRepairTwoPhase"
            and row.utility_candidate_budget == "CandidateBudget24"
            and row.layer_normalization_policy == "FinalClipOnly"
        )
        self.assertEqual(two_phase.effective_candidate_budget_by_layer, (None,) * 8 + (24,) * 8)

    def test_anchors_and_pure_prototypes_use_correct_schedule_and_budget(self) -> None:
        rows = grid.experiment13a_specs()
        anchors = [row for row in rows if row.construction_family == "Experiment12Anchor"]
        pure = [row for row in rows if row.construction_family == "PurePrototype"]
        self.assertEqual(len(anchors), 12)
        self.assertEqual(len(pure), 6)
        self.assertTrue(all(row.layer_schedule == "AnchorNative" for row in anchors))
        self.assertTrue(all(row.layer_schedule == "AllBroad" for row in pure))
        self.assertTrue(all(row.utility_candidate_budget is None for row in pure))
        self.assertTrue(all(value is None for row in pure for value in row.effective_candidate_budget_by_layer))


if __name__ == "__main__":
    unittest.main()

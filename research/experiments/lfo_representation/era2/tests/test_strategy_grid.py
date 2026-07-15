from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ERA2_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ERA2_ROOT / "code"
sys.path.insert(0, str(CODE_DIR))

from lfo_era2 import strategy_grid as grid  # noqa: E402


class StrategyGridEnumerationTests(unittest.TestCase):
    def test_counts_ids_and_pairing(self) -> None:
        rows_a, rows_b = grid.experiment13a_specs(), grid.experiment13b_specs(0.001)
        self.assertEqual((len(rows_a), len(rows_b), len(rows_a + rows_b)), (90, 90, 180))
        self.assertEqual(len({row.row_id for row in rows_a + rows_b}), 180)
        self.assertEqual(len({row.pair_id for row in rows_a + rows_b}), 90)
        grid.validate_pairing(rows_a, rows_b)
        self.assertEqual(rows_a[0].row_id, "x13a_common_case_repair_candidate_budget24_final_clip_only")
        self.assertEqual(rows_a[-1].pair_id, "x13_pair_all_dominant_directions_null_layer_clip0_to1")
        self.assertEqual([row.row_id for row in rows_a], [row.row_id for row in grid.experiment13a_specs()])

    def test_pairs_differ_only_by_phase_population_and_epsilon(self) -> None:
        rows = grid.all_strategy_specs(0.0025)
        for pair_id in {row.pair_id for row in rows}:
            pair = [row for row in rows if row.pair_id == pair_id]
            self.assertEqual(len(pair), 2)
            a = next(row for row in pair if row.experiment_phase == "13A")
            b = next(row for row in pair if row.experiment_phase == "13B")
            self.assertEqual(a.paired_settings, b.paired_settings)
            self.assertEqual((a.residual_population_policy, b.residual_population_policy), ("AllResiduals", "UnresolvedOnly"))
            self.assertIsNone(a.eligibility_epsilon)
            self.assertEqual(b.eligibility_epsilon, 0.0025)

    def test_fixed_contract_and_distinct_thresholds(self) -> None:
        rows_a, rows_b = grid.experiment13a_specs(), grid.experiment13b_specs(0.02)
        rows = rows_a + rows_b
        self.assertTrue(all(row.finish_threshold == 1e-5 for row in rows))
        self.assertEqual({row.eligibility_epsilon for row in rows_b}, {0.02})
        self.assertTrue(all(row.head_outputs_actual == 193 for row in rows))
        self.assertTrue(all((row.residual_width, row.residual_depth) == (8, 16) for row in rows))
        self.assertTrue(all((row.reserved_atom, row.active_atoms_per_layer) == ("NoOpAtom", 7) for row in rows))
        self.assertTrue(all(row.scalar_schema == "PhaseAndResidualGain" for row in rows))
        self.assertTrue(all(row.path_search_policy == "Beam4Path" for row in rows))
        self.assertTrue(all(row.runtime_topology is None for row in rows))

    def test_layer_schedules_and_effective_budgets(self) -> None:
        self.assertEqual(grid.layer_roles("Interleaved"), tuple("Broad" if layer % 2 else "Repair" for layer in range(1, 17)))
        self.assertEqual(grid.layer_roles("TwoPhase"), ("Broad",) * 8 + ("Repair",) * 8)
        interleaved = _row("BroadMeanGlobalRepairInterleaved", "CandidateBudget48", "FinalClipOnly")
        two_phase = _row("BroadMeanGlobalRepairTwoPhase", "CandidateBudget24", "FinalClipOnly")
        self.assertEqual(interleaved.effective_candidate_budget_by_layer, (None, 48) * 8)
        self.assertEqual(two_phase.effective_candidate_budget_by_layer, (None,) * 8 + (24,) * 8)
        self.assertTrue(all(len(layer) == 7 for layer in interleaved.effective_candidate_budget_by_slot))

    def test_anchor_and_pure_prototype_contracts(self) -> None:
        rows = grid.experiment13a_specs()
        anchors = [row for row in rows if row.construction_family == "Experiment12Anchor"]
        pure = [row for row in rows if row.construction_family == "PurePrototype"]
        self.assertEqual((len(anchors), len(pure)), (12, 6))
        self.assertTrue(all(row.layer_schedule == "AnchorNative" for row in anchors))
        self.assertTrue(all(row.layer_schedule == "AllBroad" for row in pure))
        self.assertTrue(all(row.utility_candidate_budget is None for row in pure))
        self.assertTrue(all(value is None for row in pure for value in row.effective_candidate_budget_by_layer))
        for row in anchors:
            self.assertEqual(row.native_slot_roles, grid.ANCHOR_SLOT_ROLES[row.construction_policy])
            self.assertEqual(row.topology_used_in_construction, row.construction_policy == "FamilyBalancedRepair")
        self.assertEqual(
            grid.ANCHOR_SLOT_ROLES["FinishRepairRescue"],
            ("finish", "finish", "common", "common", "common", "hard", "hard"),
        )
        self.assertTrue(all(row.native_slot_roles is None for row in pure))
        self.assertTrue(all(not row.topology_used_in_construction for row in pure))


class StrategyGridGateTests(unittest.TestCase):
    def test_smoke_and_filtered_runs_are_partial_without_fake_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            row_id = grid.experiment13a_specs()[0].row_id
            with self.assertRaises(grid.ConstructionNotImplementedError):
                grid.run_13a(output_dir=run_dir, backend="numpy", smoke=True, row_ids={row_id})
            status = grid.read_phase_status(run_dir, "13A")
            manifest = json.loads((run_dir / "manifest.json").read_text())
            self.assertEqual(status["state"], "partial")
            self.assertTrue(status["smoke"] and status["filtered"])
            self.assertEqual(manifest["phases"]["13A"]["row_count"], 1)
            self.assertFalse(manifest["phases"]["13A"]["complete_design"])
            self.assertTrue((run_dir / "rows" / row_id / "manifest.json").exists())
            self.assertFalse(any((run_dir / name).exists() for name in ("summary.csv", "strategy_results.csv", "epsilon_selection.json")))
            with self.assertRaises(grid.PhaseGateError):
                grid.validate_completed_13a(run_dir)

    def test_full_preflight_is_failed_not_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            with self.assertRaises(grid.ConstructionNotImplementedError):
                grid.run_13a(output_dir=run_dir, backend="numpy")
            status = grid.read_phase_status(run_dir, "13A")
            self.assertEqual((status["state"], status["completed_rows"], status["expected_row_count"]), ("failed", 0, 90))
            self.assertIn("failed", grid.status_text(run_dir))

    def test_select_epsilon_requires_complete_13a_and_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            with self.assertRaises(grid.PhaseGateError):
                grid.select_epsilon(run_dir=run_dir)
            identity, fingerprint = _write_completed_13a(run_dir)
            with self.assertRaisesRegex(grid.PhaseGateError, "calibration"):
                grid.select_epsilon(run_dir=run_dir)
            self.assertTrue(identity and fingerprint)
            self.assertFalse((run_dir / "epsilon_selection.json").exists())

    def test_completed_13a_status_must_match_manifest_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            _write_completed_13a(run_dir)
            status_path = run_dir / grid.PHASE_STATUS_FILES["13A"]
            status = json.loads(status_path.read_text())
            status["experiment13a_run_identity"] = "x13a_stale_status"
            status_path.write_text(json.dumps(status))
            with self.assertRaisesRegex(grid.PhaseGateError, "run identity"):
                grid.validate_completed_13a(run_dir)

    def test_completed_13a_rejects_inconsistent_status_and_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            _write_completed_13a(run_dir)
            status_path = run_dir / grid.PHASE_STATUS_FILES["13A"]
            status = json.loads(status_path.read_text())
            status["failed_rows"] = 1
            status_path.write_text(json.dumps(status))
            with self.assertRaisesRegex(grid.PhaseGateError, "all 90 rows"):
                grid.validate_completed_13a(run_dir)

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            _write_completed_13a(run_dir)
            manifest_path = run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["phases"]["13A"]["rows"][0]["finish_threshold"] = 0.001
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(grid.PhaseGateError, "row manifest is incompatible"):
                grid.validate_completed_13a(run_dir)

    def test_missing_partial_stale_and_incompatible_selection_block_13b(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            identity, fingerprint = _write_completed_13a(run_dir)
            path = run_dir / "epsilon_selection.json"
            for payload, message in (
                ({"selection_passed": True}, "missing fields"),
                (_selection_payload("stale", fingerprint), "stale"),
                (_selection_payload(identity, "wrong"), "configuration"),
            ):
                path.write_text(json.dumps(payload))
                with self.assertRaisesRegex(grid.SelectionArtifactError, message):
                    grid.load_epsilon_selection(path, expected_run_identity=identity, expected_configuration_fingerprint=fingerprint, require_passed=True)
            with self.assertRaises(grid.SelectionArtifactError):
                grid.load_epsilon_selection(run_dir / "absent.json", expected_run_identity=identity, expected_configuration_fingerprint=fingerprint, require_passed=True)

    def test_no_default_point_zero_two_and_invalid_selected_epsilon(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            identity, fingerprint = _write_completed_13a(run_dir)
            path = run_dir / "epsilon_selection.json"
            path.write_text(json.dumps(_selection_payload(identity, fingerprint, passed=False, selected=None)))
            with self.assertRaisesRegex(grid.SelectionArtifactError, "has not passed"):
                grid.load_epsilon_selection(path, expected_run_identity=identity, expected_configuration_fingerprint=fingerprint, require_passed=True)
            path.write_text(json.dumps(_selection_payload(identity, fingerprint, passed=True, selected=None)))
            with self.assertRaises(grid.SelectionArtifactError):
                grid.load_epsilon_selection(path, expected_run_identity=identity, expected_configuration_fingerprint=fingerprint, require_passed=True)

    def test_semantically_malformed_selection_artifacts_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            identity, fingerprint = _write_completed_13a(run_dir)
            path = run_dir / "epsilon_selection.json"
            cases = []

            numeric_strings = _selection_payload(identity, fingerprint)
            numeric_strings["candidate_epsilons"] = [str(value) for value in grid.CANDIDATE_EPSILONS]
            cases.append((numeric_strings, "finite numbers"))

            wrong_checkpoints = _selection_payload(identity, fingerprint)
            wrong_checkpoints["selection_checkpoint_definition"]["decision_slots"].append(7)
            cases.append((wrong_checkpoints, "checkpoint definition"))

            missing_timestamp = _selection_payload(identity, fingerprint)
            missing_timestamp["selection_timestamp"] = None
            cases.append((missing_timestamp, "selection_timestamp"))

            invalid_override = _selection_payload(identity, fingerprint, passed=False, selected=None)
            invalid_override["selection_override"] = True
            cases.append((invalid_override, "requires selection_passed=true"))

            for payload, message in cases:
                with self.subTest(message=message):
                    path.write_text(json.dumps(payload))
                    with self.assertRaisesRegex(grid.SelectionArtifactError, message):
                        grid.load_epsilon_selection(
                            path,
                            expected_run_identity=identity,
                            expected_configuration_fingerprint=fingerprint,
                            require_passed=False,
                        )

    def test_valid_selection_freezes_one_epsilon_and_dataset_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            identity, fingerprint = _write_completed_13a(run_dir)
            path = run_dir / "epsilon_selection.json"
            path.write_text(json.dumps(_selection_payload(identity, fingerprint, selected=0.005)))
            selection = grid.load_epsilon_selection(path, expected_run_identity=identity, expected_configuration_fingerprint=fingerprint, require_passed=True)
            self.assertEqual({row.eligibility_epsilon for row in grid.experiment13b_specs(selection.selected_epsilon or -1)}, {0.005})
            with self.assertRaisesRegex(grid.PhaseGateError, "metadata path"):
                grid.run_13b(output_dir=run_dir, epsilon_selection_path=path, metadata_path=run_dir / "other.csv", backend="numpy")
            self.assertEqual(grid.read_phase_status(run_dir, "13B")["state"], "blocked")
            with self.assertRaises(grid.ConstructionNotImplementedError):
                grid.run_13b(output_dir=run_dir, epsilon_selection_path=path, backend="numpy")
            self.assertEqual(grid.read_phase_status(run_dir, "13B")["state"], "failed")
            self.assertFalse((run_dir / "summary.csv").exists())

    def test_pilot_restrictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            identity, fingerprint = _write_completed_13a(run_dir)
            path = run_dir / "epsilon_selection.json"
            path.write_text(json.dumps(_selection_payload(identity, fingerprint, passed=False, selected=None)))
            with self.assertRaisesRegex(grid.PhaseGateError, "restricted"):
                grid.run_13b_pilot(output_dir=run_dir, epsilon_selection_path=path, candidate_epsilons=[0.005])
            non_pilot = next(row.row_id for row in grid.experiment13b_specs(0.001) if row.construction_policy not in grid.PILOT_POLICIES)
            with self.assertRaisesRegex(grid.PhaseGateError, "non-pilot"):
                grid.run_13b_pilot(output_dir=run_dir, epsilon_selection_path=path, candidate_epsilons=[0.001], row_ids={non_pilot})
            with self.assertRaisesRegex(grid.PhaseGateError, "at least one"):
                grid.run_13b_pilot(output_dir=run_dir, epsilon_selection_path=path, candidate_epsilons=[0.001], row_ids=set())
            with self.assertRaisesRegex(grid.PhaseGateError, "duplicates"):
                grid.run_13b_pilot(output_dir=run_dir, epsilon_selection_path=path, candidate_epsilons=[0.001, 0.001])
            with self.assertRaises(grid.ConstructionNotImplementedError):
                grid.run_13b_pilot(output_dir=run_dir, epsilon_selection_path=path, candidate_epsilons=[0.001, 0.0025])
            pilot = json.loads((run_dir / "experiment13b_pilot_manifest.json").read_text())
            self.assertEqual(set(pilot["allowed_construction_policies"]), set(grid.PILOT_POLICIES))
            self.assertFalse(pilot["complete"])

    def test_analyze_refuses_incomplete_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(grid.AnalysisNotReadyError):
                grid.analyze_strategy_grid(run_dir=Path(tmp))
        self.assertTrue(set(grid.REQUIRED_CALIBRATION_FILES).issubset(grid.REQUIRED_ANALYSIS_FILES))
        self.assertIn("epsilon_selection.json", grid.REQUIRED_ANALYSIS_FILES)
        self.assertIn("atom_assignments.csv", grid.REQUIRED_ANALYSIS_FILES)
        self.assertIn("candidate_search_diagnostics.csv", grid.REQUIRED_ANALYSIS_FILES)


class StrategyGridCliTests(unittest.TestCase):
    def test_cli_commands_status_and_preimport_thread_flags(self) -> None:
        script = CODE_DIR / "experiment13_strategy_grid.py"
        help_result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True, check=False)
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        for command in ("run-13a", "select-epsilon", "run-13b-pilot", "run-13b", "analyze", "status"):
            self.assertIn(command, help_result.stdout)
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [sys.executable, str(script), "--mkl-threading-layer", "SEQUENTIAL", "--native-threads", "1", "status", "--run-dir", tmp],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("13A: state=not_started", result.stdout)
            self.assertIn("13B_gate=blocked", result.stdout)


def _row(policy: str, budget: str | None, normalization: str) -> grid.StrategyRowSpec:
    return next(row for row in grid.experiment13a_specs() if row.construction_policy == policy and row.utility_candidate_budget == budget and row.layer_normalization_policy == normalization)


def _write_completed_13a(run_dir: Path) -> tuple[str, str]:
    identity, fingerprint = "x13a_test_complete", grid.configuration_fingerprint()
    grid._write_phase_manifest(
        output_dir=run_dir,
        phase="13A",
        specs=grid.experiment13a_specs(),
        run_identity=identity,
        fingerprint=fingerprint,
        metadata_path=grid.DEFAULT_METADATA,
        backend="numpy",
        smoke=False,
        corpus_sample_fraction=1.0,
        complete_design=True,
    )
    status = grid._phase_status("13A", "complete", identity, 90, 90, False, False, "complete")
    status.update(completed_rows=90, completed_at_utc="2026-07-15T00:00:00+00:00")
    grid._write_phase_status(run_dir, status)
    return identity, fingerprint


def _selection_payload(identity: str, fingerprint: str, passed: bool = True, selected: float | None = 0.001) -> dict[str, object]:
    return {
        "candidate_epsilons": list(grid.CANDIDATE_EPSILONS),
        "selection_rule_version": grid.SELECTION_RULE_VERSION,
        "selection_checkpoint_definition": json.loads(json.dumps(grid.SELECTION_CHECKPOINT_DEFINITION)),
        "selected_epsilon": selected,
        "training_statistics_used": {"dataset_split": "training", "row_count": 90},
        "median_unexplained_retired_energy_fraction": 0.001,
        "p95_unexplained_retired_energy_fraction": 0.01,
        "retired_lfo_fraction_summary": {"max": 0.1},
        "selection_timestamp": "2026-07-15T00:00:00+00:00" if passed else None,
        "experiment13a_run_identity": identity,
        "configuration_fingerprint": fingerprint,
        "selection_passed": passed,
        "selection_override": False,
        "selection_override_rationale": None,
        "selection_override_timestamp": None,
        "pilot_evidence": None,
        "selection_notes": "test fixture",
    }


if __name__ == "__main__":
    unittest.main()

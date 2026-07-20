from __future__ import annotations

import json
import importlib.util
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ERA2_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ERA2_ROOT / "code"
sys.path.insert(0, str(CODE_DIR))

from lfo_era2 import strategy_grid as grid  # noqa: E402
from lfo_era2 import strategy_grid_runtime as runtime  # noqa: E402
from lfo_era2.dataset import make_tiny_curve_dataset  # noqa: E402
from lfo_era2 import strategy_grid_execution as execution  # noqa: E402


class StrategyGridEnumerationTests(unittest.TestCase):
    def test_counts_ids_and_pairing(self) -> None:
        rows_a, rows_b = grid.experiment13a_specs(), grid.experiment13b_specs()
        self.assertEqual((len(rows_a), len(rows_b), len(rows_a + rows_b)), (90, 135, 225))
        self.assertEqual(len({row.row_id for row in rows_a + rows_b}), 225)
        self.assertEqual(len({row.pair_id for row in rows_a + rows_b}), 90)
        self.assertEqual({row.layer_normalization_policy for row in rows_b}, {"LayerClip0To1"})
        self.assertEqual(
            {row.eligibility_epsilon for row in rows_b},
            set(grid.EXPERIMENT13B_ELIGIBILITY_EPSILONS),
        )
        self.assertEqual(len({row.pair_id for row in rows_b[:3]}), 1)
        self.assertEqual(
            tuple(row.eligibility_epsilon for row in rows_b[:3]),
            grid.EXPERIMENT13B_ELIGIBILITY_EPSILONS,
        )
        grid.validate_pairing(rows_a, rows_b)
        with self.assertRaisesRegex(ValueError, "13B must use LayerClip0To1"):
            grid.validate_row_spec(replace(rows_b[0], layer_normalization_policy="FinalClipOnly"))
        self.assertEqual(rows_a[0].row_id, "x13a_common_case_repair_candidate_budget24_final_clip_only")
        self.assertEqual(rows_a[-1].pair_id, "x13_pair_all_dominant_directions_null_layer_clip0_to1")
        self.assertEqual([row.row_id for row in rows_a], [row.row_id for row in grid.experiment13a_specs()])

    def test_pairs_differ_only_by_phase_population_and_epsilon(self) -> None:
        rows = grid.all_strategy_specs()
        paired_ids = {row.pair_id for row in rows if row.experiment_phase == "13B"}
        self.assertEqual(len(paired_ids), 45)
        for pair_id in paired_ids:
            pair = [row for row in rows if row.pair_id == pair_id]
            self.assertEqual(len(pair), 4)
            a = next(row for row in pair if row.experiment_phase == "13A")
            b_rows = [row for row in pair if row.experiment_phase == "13B"]
            self.assertTrue(all(a.paired_settings == b.paired_settings for b in b_rows))
            self.assertTrue(all(b.residual_population_policy == "UnresolvedOnly" for b in b_rows))
            self.assertEqual(a.residual_population_policy, "AllResiduals")
            self.assertIsNone(a.eligibility_epsilon)
            self.assertEqual(
                {b.eligibility_epsilon for b in b_rows},
                set(grid.EXPERIMENT13B_ELIGIBILITY_EPSILONS),
            )
            self.assertTrue(all(b.layer_normalization_policy == "LayerClip0To1" for b in b_rows))
        unpaired_a = [
            row for row in rows
            if row.experiment_phase == "13A" and row.pair_id not in paired_ids
        ]
        self.assertEqual(len(unpaired_a), 45)
        self.assertEqual({row.layer_normalization_policy for row in unpaired_a}, {"FinalClipOnly"})

    def test_fixed_contract_and_distinct_thresholds(self) -> None:
        rows_a, rows_b = grid.experiment13a_specs(), grid.experiment13b_specs()
        rows = rows_a + rows_b
        self.assertTrue(all(row.finish_threshold == 1e-5 for row in rows))
        self.assertEqual(
            {row.eligibility_epsilon for row in rows_b},
            set(grid.EXPERIMENT13B_ELIGIBILITY_EPSILONS),
        )
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
            result = grid.run_13a(
                output_dir=run_dir,
                backend="numpy",
                smoke=True,
                row_ids={row_id},
                dataset=make_tiny_curve_dataset(resolution=16, row_count=8),
                chunk_size=8,
            )
            status = grid.read_phase_status(run_dir, "13A")
            manifest = json.loads((run_dir / "manifest.json").read_text())
            self.assertEqual(status["state"], "partial")
            self.assertTrue(status["smoke"] and status["filtered"])
            self.assertEqual(manifest["phases"]["13A"]["row_count"], 1)
            self.assertFalse(manifest["phases"]["13A"]["complete_design"])
            self.assertTrue((run_dir / "rows" / row_id / "manifest.json").exists())
            self.assertTrue(Path(result["summary"]).exists())
            self.assertTrue((run_dir / "strategy_results.csv").exists())
            self.assertTrue((run_dir / "rows" / row_id / "codebooks.npz").exists())
            timing = runtime.read_csv(run_dir / "rows" / row_id / "execution_timing.csv")
            self.assertEqual(sum(row["stage"] == "residual_layer" for row in timing), 16)
            self.assertEqual(sum(row["stage"] == "active_atom_slot" for row in timing), 16 * 7)
            self.assertTrue(all(float(row["wall_elapsed_seconds"]) >= 0.0 for row in timing))
            for name in (
                "slot_progression.csv", "partial_codebook_validation.csv", "atom_construction.csv",
                "atom_assignments.csv", "candidate_search_diagnostics.csv", "layer_epsilon_quantiles.csv",
                "slot_epsilon_quantiles.csv", "epsilon_coverage.csv", "retired_error_mass.csv", "budget_accounting.csv",
            ):
                self.assertTrue((run_dir / name).exists(), name)
            diagnostics = runtime.read_csv(run_dir / "atom_construction.csv")
            self.assertEqual(len(diagnostics), 16 * 7)
            required_fields = {
                "experiment_phase", "row_id", "pair_id", "residual_layer", "slot_index", "layer_role", "slot_role",
                "eligible_residual_count_before", "eligible_residual_count_after", "newly_finish_threshold_lfo_count",
                "counterfactual_resolved_fraction_by_candidate_epsilon",
                "counterfactual_incoming_retired_energy_fraction_by_candidate_epsilon",
                "counterfactual_unexplained_retired_energy_fraction_by_candidate_epsilon",
            }
            self.assertTrue(required_fields.issubset(diagnostics[0]))
            self.assertTrue(all(row["eligible_residual_count_before"] == "6" for row in diagnostics))
            self.assertFalse((run_dir / "epsilon_selection.json").exists())
            with self.assertRaises(grid.PhaseGateError):
                grid.validate_completed_13a(run_dir)

    def test_numerical_builders_are_deterministic_and_finite(self) -> None:
        import numpy as np

        values = np.random.default_rng(13).normal(size=(12, 16)).astype(np.float32)
        eligible = np.ones(12, dtype=bool)
        loss = np.mean(values * values, axis=1)
        for builder in ("BroadMean", "TrimmedMean", "AlignedMedian", "ClusterMean", "DominantDirection", "DiverseCoverage"):
            with self.subTest(builder=builder):
                left, detail = runtime._build_broad_atom(builder, values, eligible, loss, [], backend="numpy", chunk_size=8)
                right, _ = runtime._build_broad_atom(builder, values, eligible, loss, [], backend="numpy", chunk_size=8)
                self.assertEqual(left.shape, (16,))
                self.assertTrue(np.all(np.isfinite(left)))
                np.testing.assert_allclose(left, right)
                self.assertGreater(detail["prototype_population_size"], 0)

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

    def test_historical_configuration_is_report_only_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            _write_completed_13a(run_dir)
            manifest_path = run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["configuration_fingerprint"] = "historical-fingerprint"
            for row in manifest["phases"]["13A"]["rows"]:
                row["configuration_fingerprint"] = "historical-fingerprint"
            manifest_path.write_text(json.dumps(manifest))

            with self.assertRaisesRegex(grid.PhaseGateError, "configuration fingerprint"):
                grid.validate_completed_13a(run_dir)
            validated, _ = grid.validate_completed_13a(
                run_dir,
                allow_historical_configuration=True,
            )
            self.assertEqual(validated["configuration_fingerprint"], "historical-fingerprint")

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

    def test_completed_13b_requires_all_three_epsilon_sweeps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            identity, fingerprint = _write_completed_13a(run_dir)
            manifest_path = run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["phases"]["13B"] = {
                "experiment_phase": "13B",
                "row_count": 135,
                "expected_row_count": 135,
                "complete_design": True,
                "smoke": False,
                "filtered": False,
                "eligibility_epsilon_sweep": list(grid.EXPERIMENT13B_ELIGIBILITY_EPSILONS),
                "eligibility_epsilon_sweep_version": grid.EXPERIMENT13B_SWEEP_VERSION,
                "rows": [row.manifest_dict(identity, fingerprint) for row in grid.experiment13b_specs()],
            }
            manifest_path.write_text(json.dumps(manifest))
            status = {
                "schema_version": grid.SCHEMA_VERSION,
                "experiment_id": grid.EXPERIMENT_ID,
                "experiment_phase": "13B",
                "state": "complete",
                "experiment13a_run_identity": identity,
                "row_count": 135,
                "expected_row_count": 135,
                "completed_rows": 135,
                "failed_rows": 0,
                "smoke": False,
                "filtered": False,
                "completed_at_utc": "2026-07-17T00:00:00+00:00",
            }
            status_path = run_dir / grid.PHASE_STATUS_FILES["13B"]
            status_path.write_text(json.dumps(status))
            grid.validate_completed_13b(run_dir)
            status["completed_rows"] = 134
            status_path.write_text(json.dumps(status))
            with self.assertRaisesRegex(grid.PhaseGateError, "all 135"):
                grid.validate_completed_13b(run_dir)

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

    def test_valid_failed_selection_allows_fixed_sweep_and_preserves_dataset_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            identity, fingerprint = _write_completed_13a(run_dir)
            path = run_dir / "epsilon_selection.json"
            path.write_text(json.dumps(_selection_payload(identity, fingerprint, passed=False, selected=None)))
            selection = grid.load_epsilon_selection(
                path,
                expected_run_identity=identity,
                expected_configuration_fingerprint=fingerprint,
                require_passed=False,
            )
            self.assertFalse(selection.selection_passed)
            self.assertEqual(
                {row.eligibility_epsilon for row in grid.experiment13b_specs()},
                set(grid.EXPERIMENT13B_ELIGIBILITY_EPSILONS),
            )
            with self.assertRaisesRegex(grid.PhaseGateError, "metadata path"):
                grid.run_13b(output_dir=run_dir, epsilon_selection_path=path, metadata_path=run_dir / "other.csv", backend="numpy")
            self.assertEqual(grid.read_phase_status(run_dir, "13B")["state"], "blocked")
            row = grid.experiment13b_specs()[0]
            row_id = row.row_id
            grid.run_13b(
                output_dir=run_dir,
                epsilon_selection_path=path,
                backend="numpy",
                smoke=True,
                row_ids={row_id},
                dataset=make_tiny_curve_dataset(resolution=16, row_count=8),
                chunk_size=8,
            )
            self.assertEqual(grid.read_phase_status(run_dir, "13B")["state"], "partial")
            self.assertEqual(grid.read_phase_status(run_dir, "13B")["expected_row_count"], 135)
            manifest = json.loads((run_dir / "manifest.json").read_text())
            self.assertEqual(manifest["phases"]["13B"]["expected_row_count"], 135)
            self.assertEqual(
                manifest["phases"]["13B"]["eligibility_epsilon_sweep"],
                list(grid.EXPERIMENT13B_ELIGIBILITY_EPSILONS),
            )
            self.assertEqual(
                {row["layer_normalization_policy"] for row in manifest["phases"]["13B"]["rows"]},
                {"LayerClip0To1"},
            )
            self.assertTrue((run_dir / "summary.csv").exists())
            diagnostics = runtime.read_csv(run_dir / "rows" / row_id / "atom_construction.csv")
            self.assertTrue(diagnostics)
            self.assertTrue(
                all(float(item["selected_eligibility_epsilon"]) == row.eligibility_epsilon for item in diagnostics)
            )

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
            pilot_row = next(row.row_id for row in grid.experiment13b_specs(0.001) if row.construction_policy == "FinishRepairRescue")
            result = grid.run_13b_pilot(
                output_dir=run_dir,
                epsilon_selection_path=path,
                candidate_epsilons=[0.001],
                row_ids={pilot_row},
                backend="numpy",
                dataset=make_tiny_curve_dataset(resolution=16, row_count=8),
                chunk_size=8,
            )
            pilot = json.loads((run_dir / "experiment13b_pilot_manifest.json").read_text())
            self.assertEqual(set(pilot["allowed_construction_policies"]), set(grid.PILOT_POLICIES))
            self.assertTrue(pilot["complete"])
            self.assertTrue(Path(result["pilot_results"]).exists())
            override = grid.override_epsilon(run_dir=run_dir, selected_epsilon=0.001, rationale="pilot tail metrics are acceptable")
            self.assertTrue(override.selection_passed and override.selection_override)
            self.assertEqual(override.selected_epsilon, 0.001)
            loaded = grid.load_epsilon_selection(
                path,
                expected_run_identity=identity,
                expected_configuration_fingerprint=fingerprint,
                require_passed=True,
            )
            self.assertEqual(loaded.pilot_evidence["selected_epsilon_result_row_count"], 1)

    def test_epsilon_selection_uses_training_checkpoint_rule(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            _write_completed_13a(run_dir)
            _write_selection_calibration(run_dir)
            selection = grid.select_epsilon(run_dir=run_dir)
            self.assertTrue(selection.selection_passed)
            self.assertEqual(selection.selected_epsilon, 0.005)
            payload = json.loads((run_dir / "epsilon_selection.json").read_text())
            self.assertEqual(payload["training_statistics_used"]["dataset_split"], "training")
            self.assertEqual(payload["selection_checkpoint_definition"]["excluded_decision_slots"], [7])

    def test_resume_skips_complete_row_and_preserves_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            row_id = grid.experiment13a_specs()[0].row_id
            dataset = make_tiny_curve_dataset(resolution=16, row_count=8)
            grid.run_13a(output_dir=run_dir, backend="numpy", smoke=True, row_ids={row_id}, dataset=dataset, chunk_size=8)
            manifest_path = run_dir / "rows" / row_id / "manifest.json"
            before = manifest_path.read_bytes()
            with mock.patch("lfo_era2.strategy_grid_runtime.run_strategy_row", side_effect=AssertionError("row reran")):
                grid.run_13a(output_dir=run_dir, backend="numpy", smoke=True, row_ids={row_id}, dataset=dataset, chunk_size=8, resume=True)
            self.assertEqual(grid.read_phase_status(run_dir, "13A")["completed_rows"], 1)
            self.assertTrue((run_dir / "rows" / row_id / "summary.csv").exists())
            self.assertEqual(manifest_path.read_bytes(), before)

    def test_sampling_is_deterministic_stratified_and_preserves_full_validation(self) -> None:
        dataset = make_tiny_curve_dataset(resolution=16, row_count=40)
        left, left_provenance = execution.deterministic_sample(
            dataset, train_fraction=0.5, validation_fraction=1.0, seed=13,
        )
        right, right_provenance = execution.deterministic_sample(
            dataset, train_fraction=0.5, validation_fraction=1.0, seed=13,
        )
        self.assertEqual(len(left.train_indices), int(len(dataset.train_indices) * 0.5))
        np.testing.assert_array_equal(left.validation_indices, dataset.validation_indices)
        np.testing.assert_array_equal(left.train_indices, right.train_indices)
        self.assertEqual(left_provenance, right_provenance)
        self.assertTrue(np.all(left.train_indices[:-1] < left.train_indices[1:]))

    def test_finish_repair_vectorization_is_exact(self) -> None:
        rng = np.random.default_rng(13)
        target = rng.normal(size=(33, 16)).astype(np.float32)
        candidates = rng.normal(size=(13, 16)).astype(np.float32)
        phases = rng.random((33, 13), dtype=np.float32)
        gains = rng.normal(size=(33, 13)).astype(np.float32)
        eligible = rng.random(33) > 0.2
        current = np.max(np.abs(target), axis=1)
        legacy = runtime._finish_counts_legacy(target, candidates, phases, gains, eligible, current, 0.25)
        optimized = runtime._finish_counts_vectorized(
            target, candidates, phases, gains, eligible, current, 0.25, candidate_chunk=4,
        )
        np.testing.assert_array_equal(optimized, legacy)

    def test_dataset_and_base_stage_cache_round_trip_exactly(self) -> None:
        dataset = make_tiny_curve_dataset(resolution=16, row_count=20)
        sampled, provenance = execution.deterministic_sample(
            dataset, train_fraction=0.5, validation_fraction=1.0, seed=13,
        )
        with tempfile.TemporaryDirectory() as tmp:
            first = execution.load_or_build_base_stage(
                sampled, provenance, backend="numpy", chunk_size=8, cache_dir=Path(tmp),
            )
            second = execution.load_or_build_base_stage(
                sampled, provenance, backend="numpy", chunk_size=8, cache_dir=Path(tmp),
            )
            self.assertFalse(first.cache_hit)
            self.assertTrue(second.cache_hit)
            np.testing.assert_array_equal(first.base_dictionary, second.base_dictionary)
            np.testing.assert_array_equal(first.train_alignment.values, second.train_alignment.values)
            np.testing.assert_array_equal(first.validation_alignment.values, second.validation_alignment.values)

    def test_cancel_request_is_archived_on_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            grid.request_cancel(run_dir, reason="test stop")
            self.assertTrue((run_dir / "cancel_request.json").exists())
            grid._archive_cancel_request(run_dir)
            self.assertFalse((run_dir / "cancel_request.json").exists())
            self.assertEqual(len(list((run_dir / "cancel_requests").glob("*.json"))), 1)

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
        for command in (
            "run-13a", "select-epsilon", "run-13b-pilot", "override-epsilon", "run-13b",
            "analyze", "analyze-13a", "analyze-partial", "analyze-scaling", "verify-equivalence", "cancel", "status", "monitor",
        ):
            self.assertIn(command, help_result.stdout)
        self.assertIn("135-row LayerClip0To1 three-epsilon", help_result.stdout)
        partial_help = subprocess.run(
            [sys.executable, str(script), "analyze-partial", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(partial_help.returncode, 0, partial_help.stderr)
        self.assertIn("--html-report-path", partial_help.stdout)
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [sys.executable, str(script), "--mkl-threading-layer", "SEQUENTIAL", "--native-threads", "1", "status", "--run-dir", tmp],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("13A: state=not_started", result.stdout)
            self.assertIn("13B: state=not_started completed_rows=0/135", result.stdout)
            self.assertIn("13B_gate=blocked", result.stdout)

    def test_async_command_does_not_recurse_and_preserves_runtime_options(self) -> None:
        module_spec = importlib.util.spec_from_file_location("experiment13_strategy_grid_test", CODE_DIR / "experiment13_strategy_grid.py")
        assert module_spec is not None and module_spec.loader is not None
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        args = module._parser().parse_args([
            "run-13a", "--async", "--backend", "numpy", "--chunk-size", "32", "--resume",
            "--train-sample-fraction", "0.5", "--validation-sample-fraction", "1.0",
            "--sample-seed", "13", "--cache-dir", "cache", "--verify-optimized-kernels", "first-use",
        ])
        command = module._async_command(args)
        self.assertNotIn("--async", command)
        self.assertIn("--resume", command)
        self.assertEqual(command[command.index("--chunk-size") + 1], "32")
        self.assertEqual(command[command.index("--train-sample-fraction") + 1], "0.5")
        self.assertEqual(command[command.index("--cache-dir") + 1], "cache")


def _row(policy: str, budget: str | None, normalization: str) -> grid.StrategyRowSpec:
    return next(row for row in grid.experiment13a_specs() if row.construction_policy == policy and row.utility_candidate_budget == budget and row.layer_normalization_policy == normalization)


def _write_completed_13a(run_dir: Path) -> tuple[str, str]:
    identity, fingerprint = "x13a_test_complete", grid.configuration_fingerprint()
    dataset, sample = execution.deterministic_sample(
        make_tiny_curve_dataset(resolution=16, row_count=8),
        train_fraction=1.0,
        validation_fraction=1.0,
        seed=13,
    )
    execution.write_sample_artifacts(run_dir, dataset, sample)
    grid._write_phase_manifest(
        output_dir=run_dir,
        phase="13A",
        specs=grid.experiment13a_specs(),
        run_identity=identity,
        fingerprint=fingerprint,
        metadata_path=grid.DEFAULT_METADATA,
        backend="numpy",
        smoke=False,
        train_sample_fraction=1.0,
        validation_sample_fraction=1.0,
        sample_seed=13,
        sample=sample,
        dataset_cache={"cache_key": "test", "cache_hit": False, "cache_path": None},
        base_stage_cache_key="test",
        complete_design=True,
        resume=False,
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


def _write_selection_calibration(run_dir: Path) -> None:
    runtime.write_csv(
        run_dir / "layer_epsilon_quantiles.csv",
        [{"experiment_phase": "13A", "row_id": "row", "pair_id": "pair", "dataset_split": "training", "residual_layer": 0, "percentile": 0.5, "epsilon_value": 0.001, "sample_count": 10}],
    )
    runtime.write_csv(
        run_dir / "slot_epsilon_quantiles.csv",
        [{"experiment_phase": "13A", "row_id": "row", "pair_id": "pair", "residual_layer": 1, "active_atom_slot": 0, "percentile": 0.5, "epsilon_value": 0.001, "sample_count": 10}],
    )
    retired = []
    coverage = []
    for epsilon in grid.CANDIDATE_EPSILONS:
        passing = epsilon <= 0.005
        retired.append(
            {
                "experiment_phase": "13A", "row_id": "row", "pair_id": "pair",
                "residual_layer": 1, "active_atom_slot": 0, "epsilon": epsilon,
                "retired_lfo_count": 1, "retired_lfo_fraction": 0.1 if passing else 0.2,
                "incoming_retired_energy": 0.1, "incoming_retired_energy_fraction": 0.01,
                "unexplained_retired_energy": 0.001 if passing else 0.2,
                "unexplained_retired_energy_fraction": 0.005 if passing else 0.1,
                "retained_unexplained_energy_fraction": 0.995 if passing else 0.9,
                "zero_total_energy": False,
            }
        )
        coverage.append(
            {
                "experiment_phase": "13A", "row_id": "row", "pair_id": "pair",
                "dataset_split": "training", "residual_layer": 1, "active_atom_slot": 0,
                "epsilon": epsilon, "resolved_count": 1, "resolved_fraction": 0.1,
                "counterfactual_eligible_count": 9, "counterfactual_eligible_fraction": 0.9,
            }
        )
    runtime.write_csv(run_dir / "retired_error_mass.csv", retired)
    runtime.write_csv(run_dir / "epsilon_coverage.csv", coverage)


if __name__ == "__main__":
    unittest.main()

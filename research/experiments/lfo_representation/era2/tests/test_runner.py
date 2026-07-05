from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
import io


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from lfo_era2.analytics import analyze_run  # noqa: E402
from lfo_era2.cli import _async_runner_command, _validate_run_screen_args, parser  # noqa: E402
from lfo_era2.dataset import make_tiny_curve_dataset  # noqa: E402
from lfo_era2.runner import ExperimentRowSpec, _row_progress_from_message, experiment11_row_specs, run_experiment11_screen, status_text  # noqa: E402


class RunnerTests(unittest.TestCase):
    def test_builtin_experiment11_specs_use_97_control_points(self) -> None:
        specs = experiment11_row_specs(backend="numpy")
        self.assertEqual([spec.row_id for spec in specs], ["w4_d48", "w6_d32", "w8_d28", "w4_d120", "w6_d80", "w8_d72"])
        self.assertTrue(all(spec.resolution == 97 for spec in specs))
        self.assertTrue(all(spec.train_count is None and spec.validation_count is None for spec in specs))
        self.assertTrue(all(spec.oracle_phase_search_policy == "fft_lattice" for spec in specs))
        self.assertTrue(all(spec.oracle_phase_candidate_count is None for spec in specs))

    def test_run_screen_async_defaults_are_simple_launcher_defaults(self) -> None:
        args = parser().parse_args(
            [
                "run-screen",
                "--async",
                "--screen",
                "experiment11",
                "--backend",
                "xpu",
            ]
        )
        self.assertTrue(args.async_run)
        self.assertFalse(args.smoke)
        self.assertEqual(args.corpus_sample_fraction, 1.0)
        self.assertEqual(args.monitor_refresh_seconds, 30)
        command = _async_runner_command(args, Path("run_dir"))
        self.assertIn("run-screen", command)
        self.assertIn("--run-dir", command)
        self.assertIn("--no-monitor-window", command)
        self.assertNotIn("--async", command)
        self.assertNotIn("--profile", command)
        self.assertIn("--corpus-sample-fraction", command)
        self.assertIn("--oracle-phase-search-policy", command)
        self.assertNotIn("--oracle-phase-candidate-count", command)

    def test_run_screen_continuous_phase_policy_maps_to_fft_lattice(self) -> None:
        args = parser().parse_args(["run-screen", "--oracle-phase-search-policy", "continuous"])
        self.assertEqual(args.oracle_phase_search_policy, "continuous")

    def test_run_screen_profile_arg_is_removed(self) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser().parse_args(["run-screen", "--profile", "screen"])

    def test_run_screen_rejects_invalid_fraction_combinations(self) -> None:
        root = parser()
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            _validate_run_screen_args(root.parse_args(["run-screen", "--corpus-sample-fraction", "0"]), root)
        root = parser()
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            _validate_run_screen_args(root.parse_args(["run-screen", "--corpus-sample-fraction", "1.2"]), root)
        root = parser()
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            _validate_run_screen_args(root.parse_args(["run-screen", "--smoke", "--corpus-sample-fraction", "0.5"]), root)

    def test_experiment11_tiny_run_writes_status_rows_and_analytics(self) -> None:
        dataset = make_tiny_curve_dataset(resolution=24, row_count=24)
        specs = [
            ExperimentRowSpec(row_id="tiny_w2_d2", D=2, W=2, budget_band="tiny", resolution=24, train_count=8, validation_count=4, backend="numpy"),
            ExperimentRowSpec(row_id="tiny_w3_d2", D=2, W=3, budget_band="tiny", resolution=24, train_count=8, validation_count=4, backend="numpy"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_tiny"
            result = run_experiment11_screen(
                backend="numpy",
                run_dir=run_dir,
                dataset=dataset,
                row_specs=specs,
            )
            self.assertEqual(result["run_dir"], str(run_dir))
            status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["current_phase"], "complete")
            self.assertEqual(status["rows"]["tiny_w2_d2"]["status"], "completed")
            self.assertTrue((run_dir / "rows" / "tiny_w2_d2" / "manifest.json").exists())
            row_manifest = json.loads((run_dir / "rows" / "tiny_w2_d2" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(row_manifest["lfo_control_point_count"], 24)
            self.assertEqual(row_manifest["resolution"], 24)
            self.assertEqual(row_manifest["oracle_phase_search_policy"], "fft_lattice")
            self.assertEqual(row_manifest["oracle_phase_candidate_count"], 24)
            self.assertEqual(row_manifest["phase_target_kind"], "continuous_scalar")
            self.assertNotIn("phase_bins", row_manifest)
            self.assertIn("zero model prediction head outputs", row_manifest["fixed_x_grid_note"])
            self.assertFalse(row_manifest["smoke"])
            self.assertEqual(row_manifest["corpus_sample_fraction_requested"], 1.0)
            self.assertTrue((run_dir / "analytics" / "summary.csv").exists())
            self.assertTrue((run_dir / "analytics" / "budget_projections.csv").exists())
            with (run_dir / "analytics" / "budget_projections.csv").open("r", encoding="utf-8", newline="") as handle:
                projections = list(csv.DictReader(handle))
            self.assertTrue(any(row["is_actual_runtime_interface"] == "True" for row in projections))
            self.assertTrue(any(row["is_actual_runtime_interface"] == "False" for row in projections))
            report_path = Path(result["analytics"]["report"])
            self.assertTrue(report_path.exists())
            self.assertTrue((Path(result["analytics"]["report_image_dir"]) / "validation_p95_by_row.png").exists())
            report_text = report_path.read_text(encoding="utf-8")
            self.assertIn("## Main Findings", report_text)
            self.assertIn("## Budget Band Read", report_text)
            self.assertIn("## Budget Projection Notes", report_text)
            self.assertIn("decoder-owned", report_text)
            self.assertIn("Oracle phase-search resolution", report_text)
            self.assertIn("Corpus mode", report_text)
            self.assertIn("Overall: 2/2 rows complete (100.0%)", status_text(run_dir))
            self.assertIn("Current: complete", status_text(run_dir))
            self.assertIn("elapsed=", status_text(run_dir))

    def test_analyze_run_is_idempotent(self) -> None:
        dataset = make_tiny_curve_dataset(resolution=16, row_count=18)
        specs = [
            ExperimentRowSpec(row_id="tiny", D=1, W=2, budget_band="tiny", resolution=16, train_count=6, validation_count=3, backend="numpy"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_tiny"
            run_experiment11_screen(
                backend="numpy",
                run_dir=run_dir,
                dataset=dataset,
                row_specs=specs,
                analyze=False,
            )
            first = analyze_run(run_dir)
            second = analyze_run(run_dir)
            self.assertEqual(first["summary"], second["summary"])
            self.assertEqual(first["report"], second["report"])
            self.assertEqual(first["budget_projections"], second["budget_projections"])
            self.assertTrue(Path(first["report"]).exists())
            self.assertTrue((Path(first["report_image_dir"]) / "validation_p95_vs_head_outputs.png").exists())
            self.assertTrue((Path(first["report_image_dir"]) / "validation_p95_by_row.png").exists())

    def test_run_can_emit_monitor_updates(self) -> None:
        dataset = make_tiny_curve_dataset(resolution=16, row_count=18)
        specs = [
            ExperimentRowSpec(row_id="tiny", D=1, W=2, budget_band="tiny", resolution=16, train_count=6, validation_count=3, backend="numpy"),
        ]
        seen = []
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_tiny"

            def monitor(path: Path) -> None:
                seen.append(path)

            run_experiment11_screen(
                backend="numpy",
                run_dir=run_dir,
                dataset=dataset,
                row_specs=specs,
                analyze=False,
                monitor=monitor,
            )
            self.assertGreaterEqual(len(seen), 4)
            self.assertTrue(all(path == run_dir for path in seen))

    def test_run_can_emit_readable_progress_events(self) -> None:
        dataset = make_tiny_curve_dataset(resolution=16, row_count=18)
        specs = [
            ExperimentRowSpec(row_id="tiny", D=1, W=2, budget_band="tiny", resolution=16, train_count=6, validation_count=3, backend="numpy"),
        ]
        messages = []
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_tiny"
            run_experiment11_screen(
                backend="numpy",
                run_dir=run_dir,
                dataset=dataset,
                row_specs=specs,
                analyze=False,
                progress=messages.append,
            )
            self.assertTrue(any(message.startswith("run_prepare") for message in messages))
            self.assertTrue(any("row_start row=tiny" in message for message in messages))
            self.assertTrue(any("construction_start" in message for message in messages))
            self.assertTrue(any("validation_encoding_complete" in message for message in messages))
            self.assertTrue(any("row_complete row=tiny" in message for message in messages))
            self.assertTrue((run_dir / "events.jsonl").exists())

    def test_fractional_sampling_reduces_dataset_split_counts(self) -> None:
        dataset = make_tiny_curve_dataset(resolution=16, row_count=40)
        specs = [
            ExperimentRowSpec(row_id="tiny", D=1, W=2, budget_band="tiny", resolution=16, backend="numpy"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_tiny"
            run_experiment11_screen(
                backend="numpy",
                corpus_sample_fraction=0.5,
                run_dir=run_dir,
                dataset=dataset,
                row_specs=specs,
                analyze=False,
            )
            manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            row_manifest = json.loads((run_dir / "rows" / "tiny" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["corpus_sample_fraction_requested"], 0.5)
            self.assertEqual(manifest["dataset"]["train_count"], 16)
            self.assertEqual(manifest["dataset"]["validation_count"], 4)
            self.assertEqual(row_manifest["train_count"], 16)
            self.assertEqual(row_manifest["validation_count"], 4)

    def test_oracle_phase_candidate_count_does_not_change_head_outputs(self) -> None:
        dataset = make_tiny_curve_dataset(resolution=16, row_count=24)
        specs = [
            ExperimentRowSpec(row_id="phase_grid_8", D=1, W=2, budget_band="tiny", resolution=16, train_count=6, validation_count=3, backend="numpy", oracle_phase_search_policy="grid", oracle_phase_candidate_count=8),
            ExperimentRowSpec(row_id="phase_grid_16", D=1, W=2, budget_band="tiny", resolution=16, train_count=6, validation_count=3, backend="numpy", oracle_phase_search_policy="grid", oracle_phase_candidate_count=16),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_phase_budget"
            run_experiment11_screen(
                backend="numpy",
                run_dir=run_dir,
                dataset=dataset,
                row_specs=specs,
                analyze=False,
            )
            first = json.loads((run_dir / "rows" / "phase_grid_8" / "manifest.json").read_text(encoding="utf-8"))
            second = json.loads((run_dir / "rows" / "phase_grid_16" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(first["oracle_phase_candidate_count"], 8)
            self.assertEqual(second["oracle_phase_candidate_count"], 16)
            self.assertEqual(first["head_outputs_actual"], second["head_outputs_actual"])

    def test_canonical_experiment11_rejects_inactive_phase_search(self) -> None:
        dataset = make_tiny_curve_dataset(resolution=16, row_count=18)
        specs = [
            ExperimentRowSpec(row_id="inactive_phase", D=1, W=2, budget_band="tiny", resolution=16, train_count=6, validation_count=3, backend="numpy", oracle_phase_search_policy="disabled"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_inactive_phase"
            with self.assertRaisesRegex(ValueError, "oracle phase search must have more than one candidate"):
                run_experiment11_screen(
                    backend="numpy",
                    run_dir=run_dir,
                    dataset=dataset,
                    row_specs=specs,
                    analyze=False,
                )

    def test_smoke_uses_fixed_tiny_cap(self) -> None:
        dataset = make_tiny_curve_dataset(resolution=16, row_count=120)
        specs = [
            ExperimentRowSpec(row_id="tiny", D=1, W=2, budget_band="tiny", resolution=16, backend="numpy"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_tiny"
            run_experiment11_screen(
                backend="numpy",
                smoke=True,
                run_dir=run_dir,
                dataset=dataset,
                row_specs=specs,
                analyze=False,
            )
            manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            row_manifest = json.loads((run_dir / "rows" / "tiny" / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["smoke"])
            self.assertEqual(manifest["dataset"]["train_count"], 48)
            self.assertEqual(manifest["dataset"]["validation_count"], 24)
            self.assertTrue(row_manifest["smoke"])

    def test_status_text_shows_running_nested_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_progress"
            run_dir.mkdir()
            status = {
                "run_id": "run_progress",
                "smoke": False,
                "corpus_sample_fraction_requested": 1.0,
                "row_count": 6,
                "started_at_utc": "2026-07-04T20:37:51+00:00",
                "completed_at_utc": "",
                "current_phase": "validation encoding: residual layer 60/72",
                "current_row_id": "w8_d72",
                "current_row_number": 6,
                "current_task_id": "w8_d72",
                "current_task_number": 6,
                "current_task_percent": 91.4,
                "current_task_phase": "validation encoding residual layer 60/72",
                "row_order": ["w4_d48", "w6_d32", "w8_d28", "w4_d120", "w6_d80", "w8_d72"],
                "rows": {
                    "w4_d48": {"status": "completed"},
                    "w6_d32": {"status": "completed"},
                    "w8_d28": {"status": "completed"},
                    "w4_d120": {"status": "completed"},
                    "w6_d80": {"status": "completed"},
                    "w8_d72": {"status": "running"},
                },
            }
            (run_dir / "run_status.json").write_text(json.dumps(status), encoding="utf-8")
            text = status_text(run_dir)
            self.assertIn("Overall: 5/6 rows complete (83.3%)", text)
            self.assertIn("Current: w8_d72 row 6/6 91.4% - validation encoding residual layer 60/72", text)

    def test_status_text_keeps_failed_row_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_failed"
            run_dir.mkdir()
            status = {
                "run_id": "run_failed",
                "smoke": False,
                "corpus_sample_fraction_requested": 1.0,
                "row_count": 6,
                "started_at_utc": "2026-07-04T20:37:51+00:00",
                "completed_at_utc": "",
                "current_phase": "construction: residual layer 35/80",
                "current_row_id": "w6_d80",
                "current_row_number": 5,
                "current_task_id": "w6_d80",
                "current_task_number": 5,
                "current_task_percent": 43.2,
                "current_task_phase": "failed: construction residual layer 35/80",
                "row_order": ["w4_d48", "w6_d32", "w8_d28", "w4_d120", "w6_d80", "w8_d72"],
                "rows": {
                    "w4_d48": {"status": "completed"},
                    "w6_d32": {"status": "completed"},
                    "w8_d28": {"status": "completed"},
                    "w4_d120": {"status": "completed"},
                    "w6_d80": {"status": "failed", "error": "boom"},
                },
            }
            (run_dir / "run_status.json").write_text(json.dumps(status), encoding="utf-8")
            text = status_text(run_dir)
            self.assertIn("Overall: 4/6 rows complete (66.7%) failed=1", text)
            self.assertIn("Current: w6_d80 row 5/6 failed at 43.2% - construction residual layer 35/80", text)

    def test_unknown_progress_message_keeps_previous_percent(self) -> None:
        result = _row_progress_from_message(
            "unexpected internal note",
            residual_layer_count=10,
            previous_percent=12.5,
            previous_phase="construction residual layer 1/10",
        )
        self.assertEqual(result["percent"], 12.5)
        self.assertEqual(result["phase"], "unexpected internal note")


if __name__ == "__main__":
    unittest.main()

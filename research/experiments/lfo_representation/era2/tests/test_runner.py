from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from lfo_era2.analytics import analyze_run  # noqa: E402
from lfo_era2.cli import _async_runner_command, parser  # noqa: E402
from lfo_era2.dataset import make_tiny_curve_dataset  # noqa: E402
from lfo_era2.runner import ExperimentRowSpec, experiment11_row_specs, run_experiment11_screen, status_text  # noqa: E402


class RunnerTests(unittest.TestCase):
    def test_builtin_experiment11_specs_use_97_control_points(self) -> None:
        for profile in ("quick", "screen", "extended"):
            with self.subTest(profile=profile):
                self.assertTrue(all(spec.resolution == 97 for spec in experiment11_row_specs(profile, backend="numpy")))

    def test_run_screen_async_defaults_are_simple_launcher_defaults(self) -> None:
        args = parser().parse_args(
            [
                "run-screen",
                "--async",
                "--screen",
                "experiment11",
                "--profile",
                "screen",
                "--backend",
                "xpu",
            ]
        )
        self.assertTrue(args.async_run)
        self.assertEqual(args.monitor_refresh_seconds, 30)
        command = _async_runner_command(args, Path("run_dir"))
        self.assertIn("run-screen", command)
        self.assertIn("--run-dir", command)
        self.assertIn("--no-monitor-window", command)
        self.assertNotIn("--async", command)

    def test_experiment11_tiny_run_writes_status_rows_and_analytics(self) -> None:
        dataset = make_tiny_curve_dataset(resolution=24, row_count=24)
        specs = [
            ExperimentRowSpec(row_id="tiny_w2_d2", D=2, W=2, budget_band="tiny", resolution=24, train_count=8, validation_count=4, backend="numpy"),
            ExperimentRowSpec(row_id="tiny_w3_d2", D=2, W=3, budget_band="tiny", resolution=24, train_count=8, validation_count=4, backend="numpy"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_tiny"
            result = run_experiment11_screen(
                profile="quick",
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
            self.assertIn("zero model prediction head outputs", row_manifest["fixed_x_grid_note"])
            self.assertTrue((run_dir / "analytics" / "summary.csv").exists())
            self.assertTrue((run_dir / "analytics" / "images" / "validation_p95_by_row.png").exists())
            report_text = (run_dir / "analytics" / "run_report.md").read_text(encoding="utf-8")
            self.assertIn("## Main Findings", report_text)
            self.assertIn("## Budget Band Read", report_text)
            self.assertIn("decoder-owned", report_text)
            self.assertIn("completed=2/2", status_text(run_dir))
            self.assertIn("elapsed=", status_text(run_dir))

    def test_analyze_run_is_idempotent(self) -> None:
        dataset = make_tiny_curve_dataset(resolution=16, row_count=18)
        specs = [
            ExperimentRowSpec(row_id="tiny", D=1, W=2, budget_band="tiny", resolution=16, train_count=6, validation_count=3, backend="numpy"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_tiny"
            run_experiment11_screen(
                profile="quick",
                backend="numpy",
                run_dir=run_dir,
                dataset=dataset,
                row_specs=specs,
                analyze=False,
            )
            first = analyze_run(run_dir)
            second = analyze_run(run_dir)
            self.assertEqual(first["summary"], second["summary"])
            self.assertTrue(Path(first["run_report"]).exists())
            self.assertTrue((run_dir / "analytics" / "images" / "validation_p95_vs_head_outputs.png").exists())
            self.assertTrue((run_dir / "analytics" / "images" / "validation_p95_by_row.png").exists())

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
                profile="quick",
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
                profile="quick",
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


if __name__ == "__main__":
    unittest.main()

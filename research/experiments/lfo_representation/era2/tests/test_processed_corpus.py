from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from experiment10_grid_audit import run_experiment10_grid_audit  # noqa: E402
from lfo_era2.processed_corpus import build_lfo_corpus, load_processed_lfo_corpus, load_processed_shape_corpus  # noqa: E402


class ProcessedCorpusTests(unittest.TestCase):
    def test_build_deduplicates_shapes_and_preserves_occurrences(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata = _write_fixture_corpus(root)
            corpus_dir = root / "processed"
            build_lfo_corpus(metadata_path=metadata, output_dir=corpus_dir, dense_resolution=64)

            shapes = [json.loads(line) for line in (corpus_dir / "shapes.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(shapes), 2)
            self.assertEqual(sorted(shape["active_occurrence_count"] for shape in shapes), [1, 2])
            self.assertTrue((corpus_dir / "curves_r64_f32.npy").exists())

            instance_dataset = load_processed_lfo_corpus(corpus_dir, resolution=64, active_only=True, mmap=False)
            self.assertEqual(instance_dataset.curves.shape, (3, 64))
            shape_corpus = load_processed_shape_corpus(corpus_dir, resolution=64, active_only=True, mmap=False)
            self.assertEqual(shape_corpus.curves.shape, (2, 64))
            np.testing.assert_array_equal(np.sort(shape_corpus.active_occurrence_count), np.asarray([1, 2]))

    def test_experiment10_grid_audit_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata = _write_fixture_corpus(root)
            corpus_dir = root / "processed"
            output_dir = root / "experiment10"

            result = run_experiment10_grid_audit(
                metadata_path=metadata,
                corpus_dir=corpus_dir,
                output_dir=output_dir,
                report_path=output_dir / "EXPERIMENT_10_CONTROL_POINT_X_GRID_REPORT.md",
                grid_point_counts=(4, 5),
            )

            self.assertTrue(Path(result["summary"]).exists())
            self.assertTrue(Path(result["report"]).exists())
            self.assertTrue(Path(result["point_count_frequency"]).exists())
            self.assertTrue(Path(result["control_point_x_lattice_frequency"]).exists())
            self.assertTrue(Path(result["control_point_x_summary"]).exists())
            self.assertTrue(Path(result["factor3_grid_point_comparisons"]).exists())
            self.assertTrue(Path(result["global_nonuniform_grids"]).exists())
            for plot_path in result["plots"].values():
                self.assertTrue(Path(plot_path).exists())
            self.assertIn("subdivision_count + 1", result["manifest"]["grid_count_contract"])
            self.assertIn("W is reserved", result["manifest"]["grid_count_contract"])
            self.assertEqual(result["manifest"]["subdivision_counts"], [3, 4])
            self.assertIn("x only", result["manifest"]["control_point_x_contract"])
            self.assertIn("0.001", result["manifest"]["pass_rate_contract"])
            report_text = Path(result["report"]).read_text(encoding="utf-8")
            self.assertIn("control_point_count = subdivision_count + 1", report_text)
            self.assertIn("images/experiment_10/experiment10_point_count_frequency.png", report_text)
            self.assertIn("images/experiment_10/experiment10_control_point_x_lattice_frequency.png", report_text)
            self.assertIn("images/experiment_10/experiment10_control_point_x_median.png", report_text)
            self.assertIn("images/experiment_10/experiment10_lfo_pass_rate_0p001.png", report_text)
            self.assertIn("images/experiment_10/experiment10_factor3_checks.png", report_text)
            with Path(result["point_count_frequency"]).open("r", encoding="utf-8", newline="") as handle:
                point_rows = list(csv.DictReader(handle))
            self.assertEqual([row["source_point_count"] for row in point_rows], ["3", "4"])
            self.assertEqual([row["deduplicated_lfo_count"] for row in point_rows], ["1", "1"])
            self.assertEqual([row["lfo_corpus_occurrence_count"] for row in point_rows], ["2.0", "1.0"])
            with Path(result["control_point_x_lattice_frequency"]).open("r", encoding="utf-8", newline="") as handle:
                lattice_rows = list(csv.DictReader(handle))
            self.assertEqual(lattice_rows[0]["fraction"], "1/2")
            self.assertIn("occurrence_point_fraction", lattice_rows[0])
            with Path(result["summary"]).open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 6)
            self.assertEqual([row["grid_point_count"] for row in rows[:3]], ["4", "4", "4"])
            self.assertEqual([row["grid_kind"] for row in rows[:3]], ["uniform", "global_quantile", "global_quantile"])
            self.assertEqual(rows[0]["subdivision_count"], "3")
            self.assertEqual(rows[1]["subdivision_count"], "3")
            with Path(result["control_point_x_summary"]).open("r", encoding="utf-8", newline="") as handle:
                control_rows = list(csv.DictReader(handle))
            self.assertEqual(len(control_rows), 6)
            self.assertIn("control_point_x_p95_abs_error_interior_occurrence_weighted", control_rows[0])
            self.assertIn("lfo_all_points_within_0p001_deduplicated_fraction", control_rows[0])
            learned_grids = json.loads(Path(result["global_nonuniform_grids"]).read_text(encoding="utf-8"))
            self.assertEqual(len(learned_grids), 4)
            self.assertEqual(learned_grids[0]["grid_kind"], "global_quantile")


def _write_fixture_corpus(root: Path) -> Path:
    data_dir = root / "presetshare_files" / "data"
    data_dir.mkdir(parents=True)
    shape_a = {
        "name": "triangle",
        "num_points": 3,
        "points": [0.0, 0.0, 0.5, 1.0, 1.0, 0.0],
        "powers": [0.0, 0.0, 0.0],
        "smooth": False,
    }
    shape_b = {
        "name": "four",
        "num_points": 4,
        "points": [0.0, 0.0, 0.25, 1.0, 0.75, 0.25, 1.0, 0.0],
        "powers": [0.0, 0.0, 0.0, 0.0],
        "smooth": False,
    }
    _write_vital(data_dir / "preset_a.vital", [shape_a, shape_b])
    _write_vital(data_dir / "preset_b.vital", [shape_a])
    metadata = root / "presetshare_vital_metadata.csv"
    fieldnames = ["preset_id", "author_id", "author", "title", "genre", "type", "preset_file"]
    with metadata.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "preset_id": "a",
                "author_id": "author_a",
                "author": "Author A",
                "title": "Preset A",
                "genre": "",
                "type": "",
                "preset_file": "presetshare_files/data/preset_a.vital",
            }
        )
        writer.writerow(
            {
                "preset_id": "b",
                "author_id": "author_b",
                "author": "Author B",
                "title": "Preset B",
                "genre": "",
                "type": "",
                "preset_file": "presetshare_files/data/preset_b.vital",
            }
        )
    return metadata


def _write_vital(path: Path, lfos: list[dict[str, object]]) -> None:
    modulations = [{"source": f"lfo_{index}", "destination": "filter_cutoff"} for index in range(1, len(lfos) + 1)]
    settings = {"lfos": lfos, "modulations": modulations}
    for index in range(1, len(lfos) + 1):
        settings[f"modulation_{index}_amount"] = 1.0
        settings[f"modulation_{index}_bypass"] = 0.0
    path.write_text(json.dumps({"settings": settings}), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()

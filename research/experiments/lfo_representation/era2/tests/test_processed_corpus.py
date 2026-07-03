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
                point_budgets=(3, 4),
                subdivisions=(3, 4),
                dense_resolution=64,
            )

            self.assertTrue(Path(result["summary"]).exists())
            self.assertTrue(Path(result["report"]).exists())
            self.assertTrue(Path(result["point_budget_summary"]).exists())
            self.assertTrue(Path(result["subdivision_summary"]).exists())
            self.assertTrue(Path(result["direct_grid_summary"]).exists())
            with Path(result["point_budget_summary"]).open("r", encoding="utf-8", newline="") as handle:
                point_rows = list(csv.DictReader(handle))
            self.assertEqual([row["point_budget"] for row in point_rows], ["3", "4"])
            self.assertLess(float(point_rows[0]["point_budget_coverage_weighted"]), 1.0)
            self.assertEqual(float(point_rows[1]["point_budget_coverage_weighted"]), 1.0)
            with Path(result["summary"]).open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["subdivisions"] for row in rows], ["3", "4"])


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

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from lfo_era2.dataset import LfoShape, make_tiny_curve_dataset, sample_shape  # noqa: E402


class DatasetTests(unittest.TestCase):
    def test_sample_shape_returns_fixed_resolution_curve(self) -> None:
        shape = LfoShape.from_json(
            {
                "name": "line",
                "num_points": 2,
                "points": [0.0, 0.0, 1.0, 1.0],
                "powers": [0.0, 0.0],
                "smooth": False,
            }
        )
        curve = sample_shape(shape, resolution=8)
        self.assertEqual(curve.shape, (8,))
        self.assertTrue(np.all(curve >= 0.0))
        self.assertTrue(np.all(curve <= 1.0))

    def test_tiny_dataset_has_train_validation_split(self) -> None:
        dataset = make_tiny_curve_dataset(resolution=16, row_count=20)
        self.assertEqual(dataset.curves.shape, (20, 16))
        self.assertGreater(len(dataset.train_indices), 0)
        self.assertGreater(len(dataset.validation_indices), 0)
        self.assertEqual(dataset.subset(train_count=3, validation_count=2).train_curves.shape, (3, 16))


if __name__ == "__main__":
    unittest.main()

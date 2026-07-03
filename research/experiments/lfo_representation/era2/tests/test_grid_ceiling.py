from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from lfo_era2.dataset import LfoShape, sample_shape  # noqa: E402
from lfo_era2.grid_ceiling import (  # noqa: E402
    best_fixed_grid_reconstruction,
    fixed_grid_basis,
    parse_grid_points,
)


class GridCeilingTests(unittest.TestCase):
    def test_basis_rows_sum_to_one(self) -> None:
        basis = fixed_grid_basis(atom_grid_points=6, dense_points=24)
        np.testing.assert_allclose(np.sum(basis, axis=1), np.ones(24), atol=1e-8)

    def test_reconstructs_linear_ramp_exactly_on_matching_grid(self) -> None:
        shape = LfoShape.from_json(
            {
                "name": "ramp",
                "num_points": 2,
                "points": [0.0, 0.0, 1.0, 1.0],
                "powers": [0.0, 0.0],
                "smooth": False,
            }
        )
        reference = sample_shape(shape, resolution=24)[None, :]
        reconstructed = best_fixed_grid_reconstruction(
            (shape,),
            atom_grid_points=6,
            dense_points=24,
            reference_dense=reference,
        )
        self.assertLess(float(np.max(np.abs(reference - reconstructed))), 1e-6)

    def test_parse_grid_points(self) -> None:
        self.assertEqual(parse_grid_points("24,36, 48"), (24, 36, 48))

    def test_reconstruction_stays_in_vital_point_range(self) -> None:
        shape = LfoShape.from_json(
            {
                "name": "step",
                "num_points": 4,
                "points": [0.0, 0.0, 0.5, 0.0, 0.5, 1.0, 1.0, 1.0],
                "powers": [0.0, 0.0, 0.0, 0.0],
                "smooth": False,
            }
        )
        reference = sample_shape(shape, resolution=64)[None, :]
        reconstructed = best_fixed_grid_reconstruction(
            (shape,),
            atom_grid_points=8,
            dense_points=64,
            reference_dense=reference,
        )
        self.assertGreaterEqual(float(np.min(reconstructed)), -1e-6)
        self.assertLessEqual(float(np.max(reconstructed)), 1.0 + 1e-6)


if __name__ == "__main__":
    unittest.main()

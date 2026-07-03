from __future__ import annotations

import unittest

import numpy as np

from lfo_experiment.codecs import DirectGridCodec, StockResidualCodec
from lfo_experiment.curve import power_scale, sample_shape
from lfo_experiment.model import LfoShape


class CurveTests(unittest.TestCase):
    def test_triangle_samples(self) -> None:
        shape = LfoShape.from_json(
            {
                "name": "Triangle",
                "num_points": 3,
                "points": [0.0, 1.0, 0.5, 0.0, 1.0, 1.0],
                "powers": [0.0, 0.0, 0.0],
                "smooth": False,
            }
        )
        np.testing.assert_allclose(sample_shape(shape, 4), [1.0, 0.5, 0.0, 0.5])

    def test_zero_power_is_linear(self) -> None:
        values = np.linspace(0.0, 1.0, 11)
        np.testing.assert_allclose(power_scale(values, np.zeros_like(values)), values)

    def test_direct_grid_preserves_matching_grid(self) -> None:
        values = np.asarray([0.0, 0.25, 1.0, 0.5])
        reconstructed = DirectGridCodec(4).reconstruct(values).values
        np.testing.assert_allclose(reconstructed, values)

    def test_stock_residual_reconstructs_zero_residual(self) -> None:
        codebook = np.asarray([[0.0, 0.5, 1.0, 0.5]])
        reconstructed = StockResidualCodec(codebook, 4).reconstruct(codebook[0]).values
        np.testing.assert_allclose(reconstructed, codebook[0])


if __name__ == "__main__":
    unittest.main()


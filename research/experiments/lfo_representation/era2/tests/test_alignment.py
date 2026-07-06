from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from lfo_era2.accelerator import xpu_available  # noqa: E402
from lfo_era2.alignment import alignment_matrix, best_alignment  # noqa: E402


class AlignmentBackendTests(unittest.TestCase):
    def test_lattice_alignment_numpy_backend_reports_numpy(self) -> None:
        targets, codes = _alignment_fixture()
        matrix = alignment_matrix(
            targets,
            codes,
            phase_policy="fft_lattice",
            gain_policy="optimized",
            backend="numpy",
            chunk_size=4,
            phase_candidate_count=17,
        )
        self.assertEqual(matrix.backend_used, "numpy")
        self.assertEqual(matrix.losses.shape, (len(targets), len(codes)))
        self.assertTrue(np.all(np.isfinite(matrix.losses)))

    @unittest.skipUnless(xpu_available(), "torch.xpu is not available")
    def test_lattice_alignment_xpu_matches_numpy_for_phase_and_gain(self) -> None:
        targets, codes = _alignment_fixture()
        numpy_matrix = alignment_matrix(
            targets,
            codes,
            phase_policy="fft_lattice",
            gain_policy="optimized",
            backend="numpy",
            chunk_size=4,
            phase_candidate_count=17,
        )
        xpu_matrix = alignment_matrix(
            targets,
            codes,
            phase_policy="fft_lattice",
            gain_policy="optimized",
            backend="xpu",
            chunk_size=4,
            phase_candidate_count=17,
        )
        self.assertEqual(xpu_matrix.backend_used, "xpu")
        np.testing.assert_allclose(xpu_matrix.losses, numpy_matrix.losses, rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(xpu_matrix.gains, numpy_matrix.gains, rtol=2e-5, atol=2e-6)
        nonzero_gain = np.abs(numpy_matrix.gains) > 1e-5
        np.testing.assert_allclose(xpu_matrix.phases[nonzero_gain], numpy_matrix.phases[nonzero_gain], rtol=0.0, atol=0.0)

    @unittest.skipUnless(xpu_available(), "torch.xpu is not available")
    def test_best_alignment_optimized_reports_xpu_backend(self) -> None:
        targets, codes = _alignment_fixture()
        choice = best_alignment(
            targets,
            codes,
            phase_policy="fft_lattice",
            gain_policy="optimized",
            backend="xpu",
            chunk_size=4,
            phase_candidate_count=17,
        )
        self.assertEqual(choice.backend_used, "xpu")
        self.assertEqual(choice.indices.shape, (len(targets),))


def _alignment_fixture() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1234)
    x = np.arange(17, dtype=np.float32) / 17.0
    codes = np.stack(
        [
            np.zeros_like(x),
            0.35 * np.sin(2.0 * np.pi * x),
            0.25 * np.cos(4.0 * np.pi * x),
            0.2 * (2.0 * x - 1.0),
            0.15 * rng.normal(size=len(x)).astype(np.float32),
        ]
    ).astype(np.float32)
    targets = np.stack(
        [
            0.2 * np.sin(2.0 * np.pi * ((x + 0.18) % 1.0)),
            -0.35 * np.cos(4.0 * np.pi * ((x + 0.31) % 1.0)),
            0.1 * rng.normal(size=len(x)).astype(np.float32),
            np.zeros_like(x),
            0.18 * (2.0 * x - 1.0),
            0.12 * np.sin(6.0 * np.pi * x),
        ]
    ).astype(np.float32)
    return targets, codes


if __name__ == "__main__":
    unittest.main()

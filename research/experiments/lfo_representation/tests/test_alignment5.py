from __future__ import annotations

import unittest
import subprocess
import sys

import numpy as np

from lfo_experiment.alignment5 import (
    dense_align_cpu,
    exact_align_cpu,
    exact_align_xpu,
    clipped_grid_reference,
    phase_distance,
    select_best_code,
)
from lfo_experiment.phase4 import circular_shift


class AlignmentOracleTests(unittest.TestCase):
    def setUp(self) -> None:
        phase = np.linspace(0, 1, 128, endpoint=False)
        self.codes = np.stack([
            np.sin(2 * np.pi * phase),
            .7 * np.sin(4 * np.pi * phase + .2) + .2 * np.cos(6 * np.pi * phase),
            np.where(phase < .35, 1.0, -0.4),
        ]).astype(np.float32)

    def test_known_continuous_shift_and_gain(self) -> None:
        target = 1.37 * circular_shift(self.codes[1], .17321)
        result = exact_align_cpu(target[None], self.codes)
        selected, error, phase, gain = select_best_code(result)
        self.assertEqual(int(selected[0]), 1)
        self.assertLess(float(error[0]), 1e-10)
        self.assertLess(float(phase_distance(phase, np.asarray([.17321]))[0]), 1e-5)
        self.assertAlmostEqual(float(gain[0]), 1.37, places=5)

    def test_exact_never_worse_than_dense_reference(self) -> None:
        targets = np.stack([
            .8 * circular_shift(self.codes[0], .234567),
            -1.2 * circular_shift(self.codes[2], .912345),
        ])
        exact = exact_align_cpu(targets, self.codes)
        dense = dense_align_cpu(targets, self.codes, 65536)
        self.assertTrue(np.all(exact.error <= dense.error + 1e-8))

    def test_gain_clipping_and_noop(self) -> None:
        codes = np.concatenate([np.zeros((1, 128), np.float32), self.codes[:1]])
        target = 3.0 * circular_shift(self.codes[0], .1)
        result = exact_align_cpu(target[None], codes)
        self.assertEqual(result.phase[0, 0], 0.0)
        self.assertEqual(result.gain[0, 0], 0.0)
        self.assertAlmostEqual(abs(result.gain[0, 1]), 2.0, places=6)

    def test_fixed_base_gain(self) -> None:
        target = circular_shift(self.codes[2], .321)
        result = exact_align_cpu(target[None], self.codes, fixed_gain=1.0)
        selected, error, phase, gain = select_best_code(result)
        self.assertEqual(int(selected[0]), 2)
        self.assertLess(float(error[0]), 1e-9)
        self.assertEqual(float(gain[0]), 1.0)

    def test_cpu_xpu_agreement(self) -> None:
        code = """
import numpy as np, torch
from lfo_experiment.alignment5 import exact_align_cpu, exact_align_xpu
from lfo_experiment.phase4 import circular_shift
if not torch.xpu.is_available(): raise SystemExit(0)
p=np.linspace(0,1,128,endpoint=False)
codes=np.stack([np.sin(2*np.pi*p), .7*np.sin(4*np.pi*p+.2)+.2*np.cos(6*np.pi*p)]).astype(np.float32)
targets=np.stack([1.2*circular_shift(codes[0],.1234),-.7*circular_shift(codes[1],.8765)])
cpu=exact_align_cpu(targets,codes); xpu=exact_align_xpu(targets,codes)
np.testing.assert_allclose(xpu.error,cpu.error,atol=1e-6)
"""
        completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_translated_code_beats_zero_phase_alternative(self) -> None:
        target = circular_shift(self.codes[0], .2)
        distractor = target + .08 * self.codes[1]
        codes = np.stack([self.codes[0], distractor])
        result = exact_align_cpu(target[None], codes, fixed_gain=1.0)
        selected, *_ = select_best_code(result)
        self.assertEqual(int(selected[0]), 0)

    def test_clipped_refinement_never_worsens_grid_initializer(self) -> None:
        prefix = np.full((1, 128), .8, dtype=np.float32)
        target = np.clip(prefix[0] + .7 * circular_shift(self.codes[1], .137), 0, 1)[None]
        initial = clipped_grid_reference(
            target, prefix, self.codes[1:2], positions=256, top_peaks=4, refine_rounds=0
        )
        refined = clipped_grid_reference(
            target, prefix, self.codes[1:2], positions=256, top_peaks=4, refine_rounds=3
        )
        self.assertLessEqual(float(refined.error[0, 0]), float(initial.error[0, 0]) + 1e-12)


if __name__ == "__main__":
    unittest.main()

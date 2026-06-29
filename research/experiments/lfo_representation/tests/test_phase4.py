from __future__ import annotations

import unittest

import numpy as np

from lfo_experiment.phase4 import (
    PhaseChain,
    PhaseEncoding,
    canonical_orientation,
    circular_shift,
    decode_phase_chain,
    phase_distances,
    quantize_phases,
)


class PhaseFactorizationTests(unittest.TestCase):
    def test_zero_phase_is_identity_and_fractional_shift_is_periodic(self) -> None:
        curve = np.linspace(0, 1, 128, endpoint=False, dtype=np.float32)
        np.testing.assert_allclose(circular_shift(curve, 0.0), curve)
        np.testing.assert_allclose(circular_shift(curve, 1.0), curve, atol=1e-6)

    def test_shifted_target_keeps_code_and_recovers_phase(self) -> None:
        phase = np.linspace(0, 1, 128, endpoint=False)
        codes = np.stack([np.sin(2 * np.pi * phase), np.sin(4 * np.pi * phase)]).astype(np.float32)
        target = circular_shift(codes[0], .25)[None]
        distance, offsets, _ = phase_distances(target, codes, gains=False)
        self.assertEqual(int(np.argmin(distance[0])), 0)
        self.assertAlmostEqual(float(offsets[0, 0]), .25, places=5)

    def test_noop_ignores_phase_and_gain(self) -> None:
        base = np.linspace(0, 1, 128, endpoint=False, dtype=np.float32)
        stages = (np.zeros((1, 2, 128), dtype=np.float32),)
        stages[0][0, 1] = .1
        chain = PhaseChain(
            "test", base[None], stages, np.asarray([-1]),
            (np.asarray([[-1, 0]]),), ("layer_1",), False, (1,), ("shared",),
            (np.zeros((1, 2), dtype=np.float32),),
        )
        encoding = PhaseEncoding(
            np.asarray([0]), np.asarray([0.0]), [np.asarray([0])],
            [np.asarray([.37])], [np.asarray([1.8])],
        )
        decoded, _, _ = decode_phase_chain(chain, encoding, np.asarray([0]))
        np.testing.assert_allclose(decoded[0], base)

    def test_phase_quantization_preserves_noop(self) -> None:
        encoding = PhaseEncoding(
            np.asarray([1]), np.asarray([.13]), [np.asarray([0])],
            [np.asarray([0.0])], [np.asarray([0.0])],
        )
        quantized = quantize_phases(encoding, 8)
        self.assertEqual(float(quantized.stage_phases[0][0]), 0.0)

    def test_canonical_orientation_maximizes_zero_offset_utility(self) -> None:
        curve = np.zeros(128, dtype=np.float32)
        curve[24:40] = 1.0
        targets = np.stack([curve, curve, circular_shift(curve, .25)])
        current = np.mean(targets * targets, axis=1)
        oriented, _, utility = canonical_orientation(curve, targets, current, gains=False)
        candidate_utility = []
        for shift in range(128):
            value = circular_shift(oriented, shift / 128)
            mse = np.mean((targets - value) ** 2, axis=1)
            candidate_utility.append(np.sum(np.maximum(current - mse, 0)))
        self.assertAlmostEqual(utility, max(candidate_utility), places=5)


if __name__ == "__main__":
    unittest.main()

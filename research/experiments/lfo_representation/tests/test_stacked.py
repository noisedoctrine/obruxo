from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from lfo_experiment.stacked import (
    CurveDataset,
    StackedChain,
    beam_encode,
    decode_encoding,
    fit_observed_codewords,
    metric_arrays,
    train_conditional_chain,
    train_shared_chain,
    validation_conditions,
)
from lfo_experiment.residual3 import (
    envelope_basis,
    fit_frequency_first_codewords,
    flexible_beam_encode,
    flexible_decode,
)


def synthetic_dataset() -> CurveDataset:
    phase = np.linspace(0.0, 1.0, 128, endpoint=False)
    curves = []
    for index in range(16):
        frequency = 1 + index % 4
        offset = (index % 3) * 0.04
        curve = 0.5 + (0.42 - offset) * np.sin(2 * np.pi * frequency * phase + index * 0.11)
        curves.append(np.clip(curve, 0.0, 1.0))
    values = np.asarray(curves, dtype=np.float32)
    frame = pd.DataFrame(
        {
            "shape_signature": [f"shape_{index}" for index in range(len(values))],
            "author_id": [f"author_{index // 2}" for index in range(len(values))],
        }
    )
    return CurveDataset(
        frame=frame,
        curves=values,
        features=values.copy(),
        topology=np.ones(len(values), dtype=np.int8),
        train_indices=np.arange(12, dtype=np.int32),
        validation_indices=np.arange(12, 16, dtype=np.int32),
    )


class StackedCodebookTests(unittest.TestCase):
    def test_codewords_are_observed_and_zero_is_reserved(self) -> None:
        rng = np.random.default_rng(7)
        targets = synthetic_dataset().train_curves
        codes, sources = fit_observed_codewords(
            targets, 4, rng=rng, include_zero=True
        )
        np.testing.assert_allclose(codes[0], 0.0)
        self.assertEqual(sources[0], -1)
        for code, source in zip(codes[1:], sources[1:]):
            np.testing.assert_allclose(code, targets[source])

    def test_training_is_deterministic_and_sources_are_observed(self) -> None:
        dataset = synthetic_dataset()
        stock = dataset.train_curves[:2]
        first = train_shared_chain(
            dataset, stock, base_width=3, residual_width=2, max_depth=2
        )
        second = train_shared_chain(
            dataset, stock, base_width=3, residual_width=2, max_depth=2
        )
        np.testing.assert_allclose(first.bases, second.bases)
        np.testing.assert_allclose(first.residuals, second.residuals)
        learned_base_source = first.base_source_indices[2]
        np.testing.assert_allclose(first.bases[2], dataset.curves[learned_base_source])
        self.assertTrue(np.all(first.residuals[:, 0, 0] == 0.0))

    def test_more_layers_do_not_worsen_and_beam_beats_greedy(self) -> None:
        dataset = synthetic_dataset()
        chain = train_shared_chain(
            dataset,
            dataset.train_curves[:2],
            base_width=3,
            residual_width=3,
            max_depth=2,
        )
        conditions = validation_conditions(dataset, chain)
        one = beam_encode(
            dataset.validation_curves, chain, conditions, depth=1, beam_width=8
        )
        two = beam_encode(
            dataset.validation_curves, chain, conditions, depth=2, beam_width=8
        )
        greedy = beam_encode(
            dataset.validation_curves, chain, conditions, depth=2, beam_width=1
        )
        one_error = metric_arrays(
            dataset.validation_curves, decode_encoding(chain, one, conditions)
        )["rmse"]
        two_error = metric_arrays(
            dataset.validation_curves, decode_encoding(chain, two, conditions)
        )["rmse"]
        greedy_error = metric_arrays(
            dataset.validation_curves, decode_encoding(chain, greedy, conditions)
        )["rmse"]
        self.assertTrue(np.all(two_error <= one_error + 1e-7))
        self.assertTrue(np.all(two_error <= greedy_error + 1e-7))

    def test_conditional_layer_falls_back_when_residual_diversity_is_low(self) -> None:
        dataset = synthetic_dataset()
        shared = train_shared_chain(
            dataset,
            dataset.train_curves[:2],
            base_width=3,
            residual_width=4,
            max_depth=2,
        )
        conditioned = train_conditional_chain(
            dataset, shared, kind="base", min_support=1
        )
        self.assertEqual(conditioned.residuals.shape, (2, 3, 4, 128))
        self.assertTrue(np.all(conditioned.residuals[:, :, 0] == 0.0))

    def test_frequency_first_selection_counts_repeated_observations(self) -> None:
        rng = np.random.default_rng(2)
        common = np.zeros((9, 128), dtype=np.float32)
        common[:, :64] = 0.4
        rare = np.zeros((1, 128), dtype=np.float32)
        rare[:, 64:] = 1.0
        targets = np.concatenate([common, rare])
        codes, sources, utility = fit_frequency_first_codewords(
            targets, 2, rng=rng, include_zero=True
        )
        np.testing.assert_allclose(codes[0], 0.0)
        self.assertLess(sources[1], 9)
        self.assertGreater(utility[1], 0.0)

    def test_compact_envelopes_have_expected_parameter_counts(self) -> None:
        self.assertEqual(envelope_basis("none", 128).shape[0], 0)
        self.assertEqual(envelope_basis("scalar", 128).shape[0], 1)
        self.assertEqual(envelope_basis("linear", 128).shape[0], 2)
        self.assertEqual(envelope_basis("step2", 128).shape[0], 2)

    def test_flexible_noop_is_exact(self) -> None:
        dataset = synthetic_dataset()
        residuals = np.zeros((1, 1, 2, 128), dtype=np.float32)
        residuals[0, 0, 1] = dataset.train_curves[2] - dataset.train_curves[0]
        chain = StackedChain(
            base_width=2, residual_width=2, max_depth=1, strategy="test",
            bases=dataset.train_curves[:2], residuals=residuals,
            base_source_indices=np.asarray([0, 1]),
            residual_source_indices=np.asarray([[[-1, 2]]]),
        )
        conditions = validation_conditions(dataset, chain)
        encoding = flexible_beam_encode(
            dataset.validation_curves, chain, conditions,
            depth=1, mode="scalar", beam_width=4,
        )
        encoding.residual_indices[0][:] = 0
        decoded = flexible_decode(chain, encoding, conditions, mode="scalar")
        np.testing.assert_allclose(decoded, chain.bases[encoding.base_indices])


if __name__ == "__main__":
    unittest.main()

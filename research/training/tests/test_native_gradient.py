from dataclasses import replace
import unittest

import torch
from torch.nn import functional as F

from obruxo_training.codec import PilotCodec, PilotTarget
from obruxo_training.native_gradient import guided_candidates, measure_native_gradient, measured_loss
from obruxo_training.spectral_controls import BasicControlAdapter


class NativeGradientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.codec = PilotCodec()

    def test_quadratic_gradient_and_cost(self):
        anchor = PilotTarget((.4,) * 9, (2, 1))
        calls = []
        def evaluate(target):
            calls.append(target)
            return sum(x * x for x in target.continuous) + target.categorical[0] + target.categorical[1]
        measured = measure_native_gradient(self.codec, anchor, evaluate)
        self.assertEqual(measured.queries, 22)
        self.assertEqual(len(set(calls)), 22)
        for value in measured.continuous:
            self.assertAlmostEqual(value, .8)
        candidates, gradients = guided_candidates(self.codec, measured)
        self.assertLess(evaluate(candidates["continuous"]), measured.value)
        self.assertLess(evaluate(candidates["joint"]), measured.value)
        self.assertTrue(all(value > 0 for value in gradients["category_gradient_l1"]))

    def test_bounds_inactive_defaults_and_latent_category_values(self):
        raw = PilotTarget((0., 1., .8, .9, .8, .5, .5, .5, .5), (0, 0))
        calls = []
        def evaluate(target):
            calls.append(target)
            self.assertEqual(target, self.codec.canonical_target(target))
            return sum(target.continuous)
        measured = measure_native_gradient(self.codec, raw, evaluate)
        self.assertEqual(measured.queries, 14)  # Two bound probes reuse the base.
        for index in (2, 3, 4):
            self.assertEqual(measured.continuous[index], 0.)
        for index in (0, 1, 5, 6, 7, 8):
            self.assertAlmostEqual(measured.continuous[index], 1.)
        activated = next(row for row in calls if row.categorical == (0, 1))
        self.assertEqual(activated.continuous[3:5], raw.continuous[3:5])

    def test_gumbel_gradient_matches_loss_table_without_double_canonicalization(self):
        torch.manual_seed(42)
        raw = torch.full((1, 9), .4, requires_grad=True)
        logits = [torch.zeros(1, size, requires_grad=True) for size in (3, 2)]
        encoded = BasicControlAdapter(self.codec)(raw, logits)
        choices = (encoded[:, 9:12], encoded[:, 12:])
        anchor = PilotTarget(tuple(raw.detach()[0].tolist()), tuple(int(choice.argmax(-1).item()) for choice in choices))
        measured = measure_native_gradient(self.codec, anchor, lambda target: sum(target.continuous) + sum(target.categorical))
        loss = measured_loss(raw, choices, measured)
        self.assertEqual(float(loss.detach()), float(torch.tensor(measured.value)))
        expected = sum((choice * choice.new_tensor([values])).sum()
                       for choice, values in zip(choices, measured.categorical, strict=True))
        expected_grads = torch.autograd.grad(expected, logits, retain_graph=True)
        loss.backward()
        torch.testing.assert_close(raw.grad, torch.tensor([measured.continuous]))
        for scores, gradient in zip(logits, expected_grads, strict=True):
            torch.testing.assert_close(scores.grad, gradient)
            self.assertGreater(float(scores.grad.abs().sum()), 0)

    def test_reject_stale_controls_wrong_choices_and_invalid_loss(self):
        raw = torch.full((1, 9), .4)
        anchor = PilotTarget(tuple(raw[0].tolist()), (0, 0))
        measured = measure_native_gradient(self.codec, anchor, lambda target: 1.)
        choices = [F.one_hot(torch.tensor([0]), size).float() for size in (3, 2)]
        with self.assertRaisesRegex(ValueError, "stale"):
            measured_loss(raw + .01, choices, measured)
        with self.assertRaisesRegex(ValueError, "hard patch"):
            measured_loss(raw, [choices[0].roll(1, -1), choices[1]], measured)
        with self.assertRaisesRegex(ValueError, "coefficients"):
            measured_loss(raw, choices, replace(measured, continuous=(float("nan"),) * 9))
        for value in (float("nan"), float("inf"), -1.):
            with self.assertRaisesRegex(ValueError, "nonnegative"):
                measure_native_gradient(self.codec, anchor, lambda target: value)


if __name__ == "__main__":
    unittest.main()

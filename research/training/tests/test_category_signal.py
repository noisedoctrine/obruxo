import math

import pytest
import torch

from obruxo_training.category_signal import conditional_score_gradient, fit_scales, scaled_logits


def test_training_scale_ignores_common_offset_and_preserves_hard_choices():
    scores = [torch.tensor([[-50., 0., 50.], [50., 0., -50.]]), torch.tensor([[-50., 50.], [50., -50.]])]
    scales = fit_scales(scores)
    assert scales == pytest.approx((math.sqrt(5000 / 3), 50))
    assert fit_scales([row + 1000 for row in scores]) == pytest.approx(scales)
    changed = scaled_logits(scores, scales)
    assert all(torch.equal(before.argmax(-1), after.argmax(-1)) for before, after in zip(scores, changed, strict=True))
    assert fit_scales([torch.zeros(2, 3), torch.zeros(2, 2)]) == (1., 1.)


def test_fixed_binary_scale_restores_a_real_gradient_without_per_row_normalization():
    scores, noise, costs = torch.tensor([[-80., 80.]]), torch.zeros(1, 2), [.1, 1.]
    original = conditional_score_gradient(scores, noise, costs, scale=1., temperature=1.)
    changed = conditional_score_gradient(scores, noise, costs, scale=80., temperature=1.)
    assert original.abs().sum() < 1e-20
    assert torch.isfinite(changed).all() and changed.abs().sum() > 1e-6
    assert changed[0, 0] < 0 < changed[0, 1]
    raw = scores.clone().requires_grad_()
    scaled_logits([torch.zeros(1, 3), raw], (1., 80.))[1].sum().backward()
    torch.testing.assert_close(raw.grad, torch.full_like(raw, 1 / 80.))


def test_invalid_or_mismatched_calibration_contract_is_rejected():
    for scores in ([torch.zeros(1, 3)], [torch.zeros(1, 3), torch.zeros(2, 2)],
                   [torch.full((1, 3), float("nan")), torch.zeros(1, 2)]):
        with pytest.raises(ValueError):
            fit_scales(scores)
    for scales in ((0., 1.), (float("nan"), 1.), (1.,)):
        with pytest.raises(ValueError):
            scaled_logits([torch.zeros(1, 3), torch.zeros(1, 2)], scales)

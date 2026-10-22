import pytest

torch = pytest.importorskip("torch")

from obruxo_training.category_jacobian import check_jacobians  # noqa: E402
from obruxo_training.category_signal import fit_scales  # noqa: E402


def test_fixed_scale_recovers_basis_cost_gradients_without_mutating_scores_or_rng():
    logits = [torch.tensor([[35., -35., -35.], [-35., 35., -35.]]), torch.tensor([[35., -35.], [-35., 35.]])]
    original = [value.clone() for value in logits]
    rng = torch.get_rng_state().clone()
    result = check_jacobians(torch.full((2, 9), .5), logits, fit_scales(logits), draws=2)
    assert result["passed"] and result["all_derivatives_finite"]
    assert result["native_requests"] == result["optimizer_updates"] == 0
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(before, after) for before, after in zip(original, logits, strict=True))
    for point in result["points"]:
        for head in point["heads"]:
            jacobian = torch.tensor(head["jacobian"])
            assert jacobian.norm() > 0
            # A constant native loss table cannot teach a category preference.
            assert torch.allclose(jacobian.T @ torch.ones(len(jacobian)), torch.zeros(len(jacobian)), atol=1e-7)
            assert torch.all(jacobian.diag() > 0)


def test_saturated_unscaled_sampler_fails_while_healthy_sampler_needs_no_gain():
    continuous = torch.full((1, 9), .5)
    saturated = [torch.tensor([[100., -100., -100.]]), torch.tensor([[100., -100.]])]
    assert not check_jacobians(continuous, saturated, (1., 1.), draws=1)["passed"]
    healthy = [torch.zeros(1, 3), torch.zeros(1, 2)]
    result = check_jacobians(continuous, healthy, (1., 1.), draws=1)
    assert result["passed"]
    assert all(row["median_gain"] == pytest.approx(1, abs=1e-6) for head in result["heads"] for row in head["temperatures"])


def test_derivative_check_refuses_nonfrozen_or_invalid_configuration():
    logits = [torch.zeros(1, 3), torch.zeros(1, 2)]
    with pytest.raises(ValueError, match="one through sixteen"):
        check_jacobians(torch.full((1, 9), .5), logits, (1., 1.), draws=True)
    logits[0].requires_grad_(True)
    with pytest.raises(ValueError, match="detached frozen"):
        check_jacobians(torch.full((1, 9), .5), logits, (1., 1.))

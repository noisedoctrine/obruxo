import pytest
import torch
from obruxo_training.fitted_note_head import FittedNoteHead
from obruxo_training.representation_probes import ridge_predict


@pytest.mark.parametrize("count,width", [(20, 6), (6, 20)])
def test_fitted_head_matches_independent_ridge_in_both_solve_branches(count, width):
    generator = torch.Generator().manual_seed(14)
    x, query = torch.randn(count, width, generator=generator), torch.randn(7, width, generator=generator)
    x[:, 0] = 3.
    target = torch.rand(count, 2, generator=generator)
    head = FittedNoteHead(width)
    before = x.clone()
    head.fit(x, target)
    assert torch.allclose(head(query, coefficient=0.), ridge_predict(x, target, query), atol=1e-10, rtol=1e-10)
    assert torch.equal(x, before) and not list(head.parameters())


def test_feature_affine_transform_preserves_raw_mean_and_scale_information_and_gradient_sign():
    torch.manual_seed(9)
    x = torch.randn(30, 6)
    target = torch.stack((x.mean(1), x[:, 0]), 1)
    head = FittedNoteHead(6)
    head.fit(x, target)
    assert torch.allclose(head.normalized(x) * head.scale + head.center, x.double())
    assert not torch.equal(head.normalized(x), head.normalized(x + 1))
    assert float((head(x + 1, coefficient=0.) - head(x, coefficient=0.))[:, 0].mean()) > .5
    a, b = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
    head(a, coefficient=1.).square().mean().backward()
    ((head.normalized(b) @ head.weights + head.bias).square().mean()).backward()
    assert torch.equal(a.grad, -b.grad)


def test_unfitted_or_nonfinite_data_cannot_silently_disable_adversary():
    head = FittedNoteHead(6)
    with pytest.raises(ValueError, match="fitted"):
        head(torch.ones(3, 6), coefficient=1.)
    with pytest.raises(ValueError, match="finite"):
        head.fit(torch.full((3, 6), float("nan")), torch.ones(3, 2))

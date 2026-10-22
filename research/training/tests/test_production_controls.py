import pytest

torch = pytest.importorskip("torch")

from obruxo_training.codec import PilotCodec, PilotTarget  # noqa: E402
from obruxo_training.native_inverse import native_reconstruction_loss  # noqa: E402
from obruxo_training.production_controls import ProductionControlAdapter  # noqa: E402
from obruxo_training.spectral_controls import BasicControlAdapter  # noqa: E402


@pytest.mark.parametrize("temperature", (1., .5))
def test_training_forward_matches_export_for_confident_tied_and_weak_scores(temperature):
    adapter = ProductionControlAdapter(scales=(32., 29.))
    raw = torch.tensor([[.1, .8, .9, .7, .8, .4, .3, .2, .5]]).expand(3, -1).clone().requires_grad_(True)
    logits = [torch.tensor([[100., -100., 0.], [0., 0., 0.], [-.01, .01, 0.]], requires_grad=True),
              torch.tensor([[100., -100.], [0., 0.], [-.01, .01]], requires_grad=True)]
    rng = torch.get_rng_state().clone()
    actual = adapter(raw, logits, temperature=temperature)
    assert torch.equal(actual.detach(), BasicControlAdapter()(raw, logits, sample=False))
    assert adapter.export(raw, logits) == BasicControlAdapter().export(raw, logits)
    assert torch.equal(rng, torch.get_rng_state())
    for head, offset in enumerate((9, 12)):
        scores, scale = logits[head], adapter.scales[head]
        costs = torch.arange(scores.shape[1], dtype=raw.dtype)
        gradient = torch.autograd.grad((actual[:, offset:offset + len(costs)] * costs).sum(), scores, retain_graph=True)[0]
        probability = (scores.detach() / scale / temperature).softmax(-1)
        expected = probability * (costs - (probability * costs).sum(-1, keepdim=True)) / scale / temperature
        torch.testing.assert_close(gradient, expected, rtol=1e-5, atol=1e-8)
        assert torch.all(gradient.abs().sum(-1) > 0)


def test_native_loss_uses_production_anchor_and_counts_inactive_category_jumps_once():
    codec, adapter = PilotCodec(), ProductionControlAdapter(scales=(32., 29.))
    raw = torch.tensor([[.4, .4, .9, .9, .8, .4, .4, .4, .4]], requires_grad=True)
    logits = [torch.tensor([[80., -30., -40.]], requires_grad=True), torch.tensor([[60., -50.]], requires_grad=True)]
    calls = []

    def evaluate(index, candidate):
        assert index == 0
        calls.append(candidate)
        return sum(candidate.continuous) + sum(candidate.categorical)

    loss, estimates = native_reconstruction_loss({"continuous": raw, "categorical": logits}, adapter, evaluate)
    assert len(estimates) == 1 and len(calls) == estimates[0].queries
    assert estimates[0].anchor == PilotTarget(tuple(raw.detach()[0].tolist()), (0, 0))
    assert calls[0] == codec.canonical_target(estimates[0].anchor)
    assert float(loss.detach()) == float(torch.tensor(estimates[0].value))
    loss.backward()
    torch.testing.assert_close(raw.grad, torch.tensor([estimates[0].continuous]))
    for scores, scale, costs in zip(logits, adapter.scales, estimates[0].categorical, strict=True):
        probability = (scores.detach() / scale).softmax(-1)
        values = scores.new_tensor([costs])
        expected = probability * (values - (probability * values).sum(-1, keepdim=True)) / scale
        torch.testing.assert_close(scores.grad, expected, rtol=1e-5, atol=1e-8)
    assert raw.grad[0, 2:5].tolist() == [0., 0., 0.]


@pytest.mark.parametrize("scales", ((0., 1.), (float("nan"), 1.), (1., float("inf")), (True, 1.), (1.,)))
def test_reject_invalid_scales(scales):
    with pytest.raises(ValueError, match="fixed finite"):
        ProductionControlAdapter(scales=scales)


def test_export_has_no_category_gradient_and_scope_checks_remain_strict():
    adapter = ProductionControlAdapter()
    raw = torch.full((1, 9), .4, requires_grad=True)
    logits = [torch.zeros((1, size), requires_grad=True) for size in (3, 2)]
    exported = adapter(raw, logits, sample=False)
    assert torch.autograd.grad(exported[:, 9:].sum(), logits, allow_unused=True) == (None, None)
    with pytest.raises(ValueError, match="selector"):
        adapter(raw, logits, sample=1)
    with pytest.raises(ValueError, match="normalized basic"):
        adapter(raw.double(), logits)

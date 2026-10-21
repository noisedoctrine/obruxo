import copy

import pytest
import torch

from obruxo_training.features import PITCH_SIZE, TIMBRE_SHAPE, tensor_digest
from obruxo_training.model import PilotInverse
from obruxo_training.native_inverse import native_reconstruction_loss
from obruxo_training.spectral_controls import BasicControlAdapter


def test_native_loss_updates_real_inverse_and_restores_gumbel_optimizer_exactly():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    adapter, model = BasicControlAdapter(), PilotInverse()
    features = {"timbre": torch.rand(2, *TIMBRE_SHAPE), "pitch": torch.rand(2, PITCH_SIZE)}
    optimizer = torch.optim.Adam(model.parameters(), lr=.0001)
    calls = []

    def evaluate(index, patch):
        calls.append((index, patch))
        assert patch == adapter.codec.canonical_target(patch)
        # Deterministic stand-in for the costly callback; not native-quality evidence.
        return sum((value - .1 * index) ** 2 for value in patch.continuous) + .4 * patch.categorical[0] + .7 * patch.categorical[1]

    def update(network, opt):
        calls.clear()
        opt.zero_grad(set_to_none=True)
        loss, estimates = native_reconstruction_loss(network(features), adapter, evaluate)
        assert len(calls) == sum(row.queries for row in estimates) <= 44
        assert loss.item() == pytest.approx(sum(row.value for row in estimates) / 2)
        loss.backward()
        for head in (network.continuous, *network.categorical):
            assert torch.isfinite(head.weight.grad).all() and head.weight.grad.abs().sum() > 0
        opt.step()
        return loss.item(), [row.to_dict() for row in estimates]

    before = tensor_digest(model.state_dict())
    update(model, optimizer)
    assert tensor_digest(model.state_dict()) != before
    saved = {"model": copy.deepcopy(model.state_dict()), "optimizer": copy.deepcopy(optimizer.state_dict()),
             "rng": torch.get_rng_state()}
    expected = update(model, optimizer)
    recovered = PilotInverse()
    recovered.load_state_dict(saved["model"])
    opt = torch.optim.Adam(recovered.parameters(), lr=.0001)
    opt.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["rng"])
    assert update(recovered, opt) == expected
    assert tensor_digest(recovered.state_dict()) == tensor_digest(model.state_dict())
    assert adapter.export(recovered(features)["continuous"], recovered(features)["categorical"]) == adapter.export(
        model(features)["continuous"], model(features)["categorical"])


def test_empty_batch_and_callback_failure_do_not_return_partial_objective():
    adapter = BasicControlAdapter()
    with pytest.raises(ValueError, match="nonempty"):
        native_reconstruction_loss({"continuous": torch.empty(0, 9), "categorical": []}, adapter, lambda *_: 0.)
    with pytest.raises(RuntimeError, match="native failure"):
        def rejected(*_):
            raise RuntimeError("native failure")
        native_reconstruction_loss({"continuous": torch.full((1, 9), .5), "categorical": [torch.zeros(1, 3), torch.zeros(1, 2)]},
                                   adapter, rejected)

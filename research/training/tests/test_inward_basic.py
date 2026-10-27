import copy

import pytest
import torch

from obruxo_training.basic_direct import restore_direct
from obruxo_training.inward_basic import InwardClamp, POLICY, export_inward, training_model
from obruxo_training.model import batch_features, predicted_targets
from obruxo_training.trainable_basic import COUNTS
from test_prediction import artifacts  # noqa: F401


@pytest.mark.parametrize("incoming,expected", [([-2., 2., -2., 2., -2., 2.], [-2., 2., -2., 2., -2., 2.]),
                                              ([2., -2., 2., -2., 2., -2.], [0., -2., 2., -2., 2., 0.])])
def test_only_inward_gradients_pass_outside_the_legal_interval(incoming, expected):
    proposal = torch.tensor([-.2, 0., .3, .7, 1., 1.2], requires_grad=True)
    value = InwardClamp.apply(proposal)
    assert torch.equal(value.detach(), proposal.detach().clamp(0., 1.))
    value.backward(torch.tensor(incoming))
    assert torch.equal(proposal.grad, torch.tensor(expected))


def test_saturated_proposal_can_reenter_the_interval_without_pushing_further_out():
    value = torch.tensor([-.05, 1.05], requires_grad=True)
    optimizer = torch.optim.SGD([value], lr=.1)
    loss = ((InwardClamp.apply(value) - .5) ** 2).sum()
    loss.backward()
    optimizer.step()
    assert torch.all((value > 0.) & (value < 1.))


def test_inference_export_keeps_weights_predictions_interface_and_release_status(artifacts):  # noqa: F811
    root, _base, _frontend, _codec, audio, _saved = artifacts
    initial, frontend, codec = restore_direct(torch.load(root / "inputs.pt", map_location="cpu", weights_only=True))
    wrapped = training_model(initial)
    assert wrapped.require_frozen() == COUNTS
    features = batch_features([{"features": frontend(audio)}])
    with torch.no_grad():
        before, after = initial(features), wrapped(features)
    assert torch.equal(before["continuous"], after["continuous"])
    assert all(torch.equal(a, b) for a, b in zip(before["categorical"], after["categorical"], strict=True))
    saved = export_inward(wrapped, frontend, codec, "synthetic-test-pool")
    restored, _front, _codec = restore_direct(copy.deepcopy(saved))
    with torch.no_grad():
        cold = restored(features)
    assert predicted_targets(after, codec) == predicted_targets(cold, codec)
    assert saved["training_gradient_policy"] == POLICY and saved["qualified_for_release"] is False

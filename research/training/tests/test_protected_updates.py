import pytest
import torch

from obruxo_training.protected_update_check import inspect_proposals
from obruxo_training.codec import PilotCodec
from obruxo_training.protected_updates import project_direction, select_proposal
from test_basic_direct_fit import completed  # noqa: F401


@pytest.mark.parametrize("proposal,gradients,expected", [
    ([-1., 2.], [[1., 0.], [0., 1.]], [-1., 0.]),
    ([2., 3.], [[1., 0.], [-1., 0.]], [0., 3.]),
    ([2., 3.], [[1., 0.], [1., 0.], [0., 0.]], [0., 3.]),
    ([-2., -3.], [[1., 0.], [0., 1.]], [-2., -3.]),
])
def test_projection_handles_conflicts_rank_deficiency_and_already_safe_steps(proposal, gradients, expected):
    result, _ = project_direction(torch.tensor(proposal), list(torch.tensor(gradients)))
    assert result.tolist() == pytest.approx(expected, abs=1e-6)
    assert all(float(torch.dot(row, result)) <= 1e-6 for row in torch.tensor(gradients))


def test_nonfinite_proposals_refused():
    with pytest.raises(ValueError, match="finite"):
        project_direction(torch.tensor([float("nan")]), [torch.tensor([1.])])


def test_output_guard_backtracks_for_unsampled_inputs_and_restores_on_failure():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.correction = torch.nn.Linear(1, 9, bias=False)

    class Inputs:
        codec = PilotCodec()
        plan = {"fit_audit_cells": [{"key": i, "x": 1. if i < 4 else 10.} for i in range(5)]}

        def key(self, cell):
            return cell["key"]

        def predict(self, model, cell):
            return {"continuous": .4 + model.correction(torch.tensor([[cell["x"]]])),
                    "categorical": [torch.tensor([[0., 1., 0.]]), torch.tensor([[0., 1.]])]}

    model, inputs = Model(), Inputs()
    with torch.no_grad():
        model.correction.weight.fill_(.3)
    incoming = model.correction.weight.detach().clone()
    before, proposal = torch.zeros(9), torch.tensor([.1] + [0.] * 8)
    estimate = {"anchor": {"continuous": torch.full((9,), .4).tolist(), "categorical": [1, 1]},
                "continuous": [-1.] + [0.] * 8, "categorical": [[0.] * 3, [0.] * 2]}
    estimates, observations = [estimate] * 4, [[i] for i in range(4)]
    _, result = select_proposal(model, before, proposal, [-proposal] * 4, inputs, observations, estimates)
    assert result["accepted"] and result["scale"] == 1 / 64
    assert result["maximum_bank_control_change"] == pytest.approx(.015625)
    assert torch.equal(model.correction.weight, incoming)
    estimate["anchor"]["continuous"][0] = .41
    with pytest.raises(ValueError, match="stale"):
        select_proposal(model, before, proposal, [-proposal] * 4, inputs, observations, estimates)
    assert torch.equal(model.correction.weight, incoming)


def test_temporary_proposals_leave_full_historical_fit_exact(completed):  # noqa: F811
    inputs, source = completed
    result = inspect_proposals(inputs, source)
    assert result["model_adam_rng_replay_exact"] and result["replayed_updates_per_arm"] == 4
    assert result["temporary_protected_proposals"] == 4 and result["scalar_probes_reused"] > 0
    for row in result["rows"]:
        assert row["maximum_bank_control_change"] <= .025 + 1e-8
        assert all(change["estimated_sound_change"] <= 1e-8 for change in row["changes"])
        assert row["accepted"] == (row["scale"] > 0)
        if row["accepted"]:
            assert any(change["estimated_sound_change"] < -1e-8 for change in row["changes"])

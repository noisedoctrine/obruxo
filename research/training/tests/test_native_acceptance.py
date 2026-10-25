import copy
from types import SimpleNamespace

import pytest
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from obruxo_training.codec import PilotCodec
from obruxo_training.native_acceptance import accept_update, capture, objective
from obruxo_training.paired_native_audit import equal_tree
from obruxo_training.production_controls import ProductionControlAdapter


def setup_proposal(amount=.1):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.correction = torch.nn.Linear(1, 9, bias=False)
            torch.nn.init.zeros_(self.correction.weight)

        def require_frozen(self):
            return None

    class Inputs:
        codec = PilotCodec()
        adapter = ProductionControlAdapter(codec)
        plan = {"fit_audit_cells": [{"key": i, "x": 1. if i < 4 else 10.} for i in range(5)]}

        def key(self, cell):
            return cell["key"]

        def predict(self, model, cell):
            return {"continuous": (.4 + model.correction(torch.tensor([[cell["x"]]]))).clamp(0., 1.),
                    "categorical": [torch.tensor([[0., 1., 0.]]), torch.tensor([[0., 1.]])]}

    inputs, model = Inputs(), Model()
    before = capture(inputs, model)
    proposal = before["weights"].clone()
    proposal[0] = amount
    with torch.no_grad():
        vector_to_parameters(proposal, tuple(model.correction.parameters()))
    observations = [{"baseline": .5}] * 4
    estimates = [{"value": 1., "anchor": {"continuous": torch.full((9,), .4).tolist(), "categorical": [1, 1]}} for _ in range(4)]
    result = {"estimates": estimates, "losses": {"spectral": objective([1.] * 4, [.5] * 4)}}
    return SimpleNamespace(inputs=inputs, model=model, before=before, cells=[[i] for i in range(4)], observations=observations, result=result)


def run(ctx, evaluate):
    return accept_update(ctx.inputs, ctx.model, ctx.before, ctx.cells, ctx.observations, ctx.result, evaluate)


def test_real_cost_refuses_first_trial_and_accepts_smaller_trial_with_unsampled_input_cap():
    ctx, measured = setup_proposal(), []

    def evaluate(index, patch):
        measured.append((index, patch))
        return 1.2 if patch.continuous[0] > .401 else .8

    result = run(ctx, evaluate)
    assert result["accepted"] and result["scale"] == 1 / 128
    assert [row["status"] for row in result["trials"]] == ["sound_refused", "accepted"]
    assert result["before_objective"] > result["committed_objective"]
    assert len(measured) == 8 and result["bank_inputs"] == 5
    assert result["cap_checks"][-1]["scale"] == 1 / 64
    assert all(row["maximum_control_change"] <= .025 + 1e-8 for row in result["trials"])
    assert float(parameters_to_vector(ctx.model.correction.parameters()).detach()[0]) == pytest.approx(.1 / 128)


def test_refused_trials_restore_weights_exactly_without_changing_adam_moments():
    ctx = setup_proposal()
    optimizer = torch.optim.Adam(ctx.model.correction.parameters())
    for parameter in ctx.model.correction.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    moments = copy.deepcopy(optimizer.state_dict())
    result = run(ctx, lambda *_args: 1.2)
    assert not result["accepted"] and len(result["trials"]) == 3
    assert result["committed_objective"] == result["before_objective"]
    assert torch.equal(parameters_to_vector(ctx.model.correction.parameters()), ctx.before["weights"])
    assert equal_tree(moments, optimizer.state_dict())


def test_render_failure_restores_pre_update_weights():
    ctx = setup_proposal()

    def fail(*_args):
        raise RuntimeError("native render failed")

    with pytest.raises(RuntimeError, match="render failed"):
        run(ctx, fail)
    assert torch.equal(parameters_to_vector(ctx.model.correction.parameters()), ctx.before["weights"])


def test_stale_anchor_is_refused_before_any_render_and_rolls_back():
    ctx = setup_proposal()
    ctx.result["estimates"][0]["anchor"]["continuous"][0] = .5
    with pytest.raises(ValueError, match="stale"):
        run(ctx, lambda *_args: pytest.fail("a stale anchor must not render"))
    assert torch.equal(parameters_to_vector(ctx.model.correction.parameters()), ctx.before["weights"])


def test_unsafe_bank_refuses_without_spending_native_requests():
    ctx = setup_proposal(amount=1000.)
    result = run(ctx, lambda *_args: pytest.fail("unsafe controls must not render"))
    assert not result["accepted"] and len(result["cap_checks"]) == 9 and not result["trials"]
    assert result["logical_trial_evaluations"] == 0
    assert torch.equal(parameters_to_vector(ctx.model.correction.parameters()), ctx.before["weights"])


def test_objective_uses_original_reference_hinges_and_rejects_invalid_measurements():
    assert objective([.2, .8, .7, .1], [.5] * 4) == pytest.approx(.575)
    with pytest.raises(ValueError, match="finite nonnegative"):
        objective([.2, .8, float("nan"), .1], [.5] * 4)


def test_original_weighted_objective_can_accept_a_tradeoff_between_sampled_sounds():
    ctx = setup_proposal()
    costs = [.7, .8, 1.1, .9]
    result = run(ctx, lambda index, _patch: costs[index])
    assert result["accepted"] and result["trials"][0]["measured_sound_costs"][2] > 1.
    assert result["committed_objective"] < result["before_objective"]

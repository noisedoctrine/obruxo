import copy

import pytest
import torch

from obruxo_training.codec import PilotCodec, PilotTarget
from obruxo_training.direct_basic_control import DirectPatch, SPEC, category_round, continuous_step, quality, step
from obruxo_training.paired_native_audit import equal_tree


def patch(values=None, categories=(2, 1)):
    return DirectPatch(PilotCodec(), PilotTarget(tuple(values or [.5] * 9), categories))


def bowl(candidate, _role):
    return 1. + sum((value - .2) ** 2 for value in candidate.continuous)


def test_boundary_probe_uses_actual_span_and_real_cost_caps_every_move():
    model = patch([0., 1., .5, .5, .5, .5, .5, .5, .5])
    result = continuous_step(model, bowl)
    assert result["gradient"][0] == pytest.approx(-.375, abs=1e-6)
    assert result["gradient"][1] == pytest.approx(1.575, abs=1e-6)
    assert result["accepted"] and result["after_loss"] < result["before_loss"]
    assert 0 < result["maximum_move"] <= .025
    assert all(0 <= value <= 1 for value in model.controls)


def test_all_refused_trials_restore_controls_but_advance_adam_including_invalid_audio():
    model = patch()
    before = model.controls.detach().clone()

    def evaluate(candidate, role):
        return None if role == "trial" else bowl(candidate, role)

    result = continuous_step(model, evaluate)
    assert not result["accepted"] and [row["status"] for row in result["trials"]] == ["invalid_audio"] * 3
    assert torch.equal(model.controls, before) and model.refusals == 1
    state = model.optimizer.state[model.controls]
    assert state["step"] == 1 and bool(state["exp_avg"].abs().sum() > 0)


def test_inactive_stored_values_stay_exact_despite_existing_adam_momentum():
    model = patch()
    continuous_step(model, bowl)
    before = model.controls.detach().clone()
    model.categories = (0, 0)
    continuous_step(model, bowl)
    assert torch.equal(model.controls[2:5], before[2:5])
    assert not torch.equal(model.controls[:2], before[:2])


def test_six_category_choices_prefer_current_on_ties_and_reset_refusals_on_improvement():
    model = patch(categories=(2, 1))
    tied = category_round(model, lambda *_: 1.)
    assert len(tied["rows"]) == 6 and not tied["accepted"] and model.categories == (2, 1)
    before = model.controls.detach().clone()
    model.refusals = 7
    improved = category_round(model, lambda candidate, _role: 1. if candidate.categorical == (0, 0) else 2.)
    assert improved["accepted"] and model.categories == (0, 0) and model.refusals == 0
    assert torch.equal(model.controls, before)


@pytest.mark.parametrize("bad", [None, float("nan"), -1.])
def test_invalid_gradient_cost_fails_before_adam_advance(bad):
    model = patch()
    with pytest.raises(ValueError, match="costs must be valid"):
        continuous_step(model, lambda candidate, role: 1. if role == "anchor" else bad)
    assert not model.optimizer.state


def test_cold_checkpoint_and_partial_measurement_interruption_replay_exactly():
    uninterrupted, resumed = patch(), patch()
    for i in range(9):
        assert step(uninterrupted, i, bowl) == step(resumed, i, bowl)
    saved = resumed.snapshot()
    partial = copy.deepcopy(saved)
    calls = 0

    def interrupt(candidate, role):
        nonlocal calls
        calls += 1
        if calls == 5:
            raise RuntimeError("simulated render timeout")
        return bowl(candidate, role)

    with pytest.raises(RuntimeError, match="timeout"):
        step(resumed, 9, interrupt)
    cold = patch()
    cold.restore(partial)
    for i in range(9, 16):
        assert step(uninterrupted, i, bowl) == step(cold, i, bowl)
    assert equal_tree(uninterrupted.snapshot(), cold.snapshot())


def test_eight_refusals_stop_without_extra_updates_and_category_round_can_reset():
    model = patch()
    for i in range(8):
        result = step(model, i, lambda *_: 1.)
        assert not result["continuous"]["accepted"]
    assert model.refusals == SPEC["refusal_limit"]
    with pytest.raises(ValueError, match="allocation"):
        step(model, 8, lambda *_: pytest.fail("stopped case must not measure"))


def test_quality_requires_every_performance_mrstft_and_the_matched_control():
    initial = [{"spectral": 1., "mrstft": 1.}] * 2
    good = [{"spectral": .7, "mrstft": .9}] * 2
    assert quality(initial, good, good)["pass"]
    bad = [good[0], {"spectral": .7, "mrstft": 1.01}]
    assert not quality(initial, bad, good)["pass"]
    assert not quality(initial, good, [{"spectral": .6}] * 2)["pass"]

import copy

import pytest
import torch

from obruxo_training import basic_direct_fit as fit
from obruxo_training.audit import write_json
from obruxo_training.basic_direct import BasicDirect, export_direct, restore_direct
from obruxo_training.basic_direct_bridge import gradient_access, simulate
from obruxo_training.basic_direct_fit_audit import verify
from obruxo_training.features import tensor_digest
from obruxo_training.model import batch_features
from obruxo_training.native_acceptance import SPEC as ACCEPTANCE
from obruxo_training.paired_native_audit import equal_tree
from obruxo_training.trainable_basic import COUNTS, TrainableBasic, input_gradients
from test_basic_direct import data  # noqa: F401
from test_basic_direct_fit import FakeSession, completed  # noqa: F401
from test_conditioned_basic import measured_direction
from test_native_acceptance_fit import test_timeout_during_actual_trial_recovers_both_models_adam_and_rng_exactly as check_recovery


def test_trainable_features_keep_initial_outputs_rng_and_cold_export_exact(data):  # noqa: F811
    base, codec, features, frontend = data
    before = torch.get_rng_state().clone()
    model, old = TrainableBasic(base), BasicDirect(base)
    assert model.require_frozen() == COUNTS and torch.equal(before, torch.get_rng_state())
    cold, _frontend, _codec = restore_direct(export_direct(model, frontend, codec, "simulation"))
    batch = batch_features([{"features": features}])
    first, previous, restored = [value(batch) for value in (model, old, cold)]
    assert torch.equal(first["continuous"], previous["continuous"]) and torch.equal(first["continuous"], restored["continuous"])
    assert all(torch.equal(a, b) and torch.equal(a, c) for a, b, c in zip(first["categorical"], previous["categorical"], restored["categorical"], strict=True))
    assert first["continuous"].shape == (1, 9) and [value.shape[1] for value in first["categorical"]] == [3, 2]
    assert equal_tree(model.correction[0].timbre.state_dict(), base.timbre.state_dict())
    assert equal_tree(model.correction[0].pitch.state_dict(), base.pitch.state_dict())


def test_sound_gradient_reaches_both_input_branches_after_first_update(data, tmp_path):  # noqa: F811
    base, codec, features, frontend = data
    model = TrainableBasic(base)
    _adapter, direction = measured_direction(base, codec, features)
    gradient_access(model, features, direction)
    assert input_gradients(model) == {"timbre": 0., "pitch": 0., "hidden": 0.}
    initial = tensor_digest(model.state_dict())
    trained, cold, result = simulate(model, frontend, codec, "simulation", [features] * 4, tmp_path)
    assert all(value > 0 for value in input_gradients(trained).values())
    assert result["model_adam_rng_recovery_exact"] and result["simulated_optimizer_step_applications"] == 3
    assert result["native_requests"] == result["experimental_training_updates"] == 0
    assert tensor_digest(model.state_dict()) == initial
    assert equal_tree(trained.base.state_dict(), model.base.state_dict())
    assert not equal_tree(trained.correction[0].timbre.state_dict(), model.correction[0].timbre.state_dict())
    assert not equal_tree(trained.correction[0].pitch.state_dict(), model.correction[0].pitch.state_dict())
    assert tensor_digest(trained.state_dict()) == tensor_digest(cold.state_dict())


@pytest.mark.parametrize("change", ["encoder", "budget", "phase", "mask"])
def test_changed_input_model_or_frozen_contract_is_refused(data, change):  # noqa: F811
    base, codec, _features, frontend = data
    model = TrainableBasic(base)
    if change == "mask":
        model.correction[0].pitch.requires_grad_(False)
        with pytest.raises(ValueError, match="mask"):
            model.require_frozen()
        return
    saved = copy.deepcopy(export_direct(model, frontend, codec, "simulation"))
    if change == "encoder":
        saved["correction"]["0.timbre.0.weight"][0, 0, 0, 0] += .1
    elif change == "budget":
        saved["spec"]["additional_output_heads"] = 1
    else:
        saved["render_phase_controls"]["sampler_random_phase"] = .5
    with pytest.raises(ValueError, match="changed|phase|profile"):
        restore_direct(saved)


@pytest.fixture(scope="module")
def trained(completed, tmp_path_factory):  # noqa: F811
    original, source = completed
    root = tmp_path_factory.mktemp("trainable-input-fit")
    bridge, output = root / "bridge", root / "fit"
    bridge.mkdir()
    output.mkdir()
    ctx = copy.copy(original)
    ctx.model = TrainableBasic(original.model.base)
    ctx.initial_path = bridge / "model.pt"
    torch.save(export_direct(ctx.model, ctx.frontend, ctx.codec, "simulation"), ctx.initial_path)
    ctx.plan = {**copy.deepcopy(original.plan), "native_acceptance": copy.deepcopy(ACCEPTANCE),
                "gates": {**original.plan["gates"], "minimum_native_accepted_updates": 3}}
    write_json(output / "plan.json", ctx.plan)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(fit, "PairedSession", FakeSession)
        result = fit.execute(ctx, output)
    assert result["complete"]
    return ctx, output, source


def test_existing_sound_fit_and_audit_include_encoder_and_adam_state(trained):
    ctx, output, _source = trained
    proof = verify(ctx, output)
    assert proof["offline_training_replay_exact"] and proof["comparison_waveforms_rescored"] == 24
    saved = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
    for arm in ("parameter", "spectral"):
        assert equal_tree(saved["models"][arm]["base.timbre.0.weight"], ctx.model.base.timbre[0].weight)
        assert not torch.equal(saved["models"][arm]["correction.0.timbre.0.weight"], ctx.model.correction[0].timbre[0].weight)
        assert len(saved["optimizers"][arm]["state"]) == len(tuple(ctx.model.correction.parameters()))


def test_encoder_recovers_exactly_after_interrupted_native_trial(trained, tmp_path, monkeypatch):
    check_recovery(trained, tmp_path, monkeypatch)

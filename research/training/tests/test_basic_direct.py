import copy

import numpy as np
import pytest
import torch

from obruxo_data.render.qa import audio_float32_sha256
from obruxo_training.basic_direct import COUNTS, SPEC, BasicDirect, export_direct, reference_loss, rehearsal_indices, restore_direct
from obruxo_training.basic_direct_bridge import gradient_access, preserve, reference_manifest, simulate, validate_references
from obruxo_training.codec import PilotCodec
from obruxo_training.features import AudioFeatures, BasicPitchICASSP2022, PITCH_SIZE, TIMBRE_SHAPE, tensor_digest
from obruxo_training.inference_profile import bind_rendering
from obruxo_training.model import PilotInverse, batch_features
from obruxo_training.production_controls import ProductionControlAdapter
from obruxo_training.profiled_sampler import profile_metadata
from test_conditioned_basic import measured_direction


@pytest.fixture
def data():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    base, codec = PilotInverse().eval(), PilotCodec()
    bind_rendering(base, profile_metadata(32768))
    features = {"timbre": torch.rand(*TIMBRE_SHAPE), "pitch": torch.rand(PITCH_SIZE)}
    frontend = AudioFeatures(BasicPitchICASSP2022().state_dict())
    return base, codec, features, frontend


def test_direct_initialization_preserves_all_outputs_and_fixed_phase_presets(data):
    base, codec, features, frontend = data
    rng = torch.get_rng_state().clone()
    model = BasicDirect(base)
    assert model.require_frozen() == COUNTS and torch.equal(rng, torch.get_rng_state())
    cold, _front, _codec = restore_direct(export_direct(model, frontend, codec, "fixture"))
    rows = preserve(base, model, cold, codec, {"recipe": {"features": features, "state_id": "train", "audio_sha256": "audio"}}, initial=True)
    assert len(rows) == 1
    output = model(batch_features([{"features": features}]))
    assert output["continuous"].shape == (1, 9) and [value.shape[1] for value in output["categorical"]] == [3, 2]


def test_native_coefficients_reach_direct_controls_and_both_category_heads(data):
    base, codec, features, _front = data
    model = BasicDirect(base)
    _adapter, direction = measured_direction(base, codec, features)
    result = gradient_access(model, features, direction)
    assert result["frozen_gradients_absent"] and result["weights_and_rng_unchanged"]
    assert all(value > 0 for value in result["continuous_bias_gradient_l1"] if value is not None)
    assert all(value > 0 for value in result["category_bias_gradient_l1"] if value is not None)
    assert result["gradient_groups"]["hidden"] == 0
    with torch.no_grad():
        model.correction[2].bias[0] = .01
    with pytest.raises(ValueError, match="stale"):
        gradient_access(model, features, direction)


def test_clipping_keeps_controls_legal_and_uses_actual_clamp_derivative(data):
    base, codec, features, _front = data
    model = BasicDirect(base)
    with torch.no_grad():
        model.correction[2].bias[:9].copy_(torch.tensor([20., -20.] * 4 + [20.]))
    output = model(batch_features([{"features": features}]))
    assert output["continuous"][0].tolist() == [1., 0.] * 4 + [1.]
    output["continuous"].sum().backward()
    assert not model.correction[2].bias.grad[:9].any()
    target = ProductionControlAdapter(codec, scales=SPEC["adapter_scales"]).export(output["continuous"], output["categorical"])[0]
    codec.validate_target(target)
    model.require_frozen()


@pytest.mark.parametrize("value,baseline,expected,derivative", [(.2, .3, .2, 1.), (.4, .3, .5, 2.), (.3, .3, .3, 1.)])
def test_reference_sound_regression_receives_extra_gradient(value, baseline, expected, derivative):
    current = torch.tensor(value, requires_grad=True)
    result = reference_loss(current, baseline)
    result.backward()
    assert float(result.detach()) == pytest.approx(expected) and float(current.grad) == derivative


def test_reference_schedule_covers_all_original_recordings_three_times():
    selected = [index for step in range(48) for index in rehearsal_indices(step)]
    assert all(selected.count(index) == 3 for index in range(32))
    assert all(a != b for a, b in (rehearsal_indices(step) for step in range(48)))
    with pytest.raises(ValueError):
        rehearsal_indices(-1)


def test_software_updates_recompute_costs_and_recover_exactly(tmp_path, data):
    base, codec, features, frontend = data
    model = BasicDirect(base)
    rng = torch.get_rng_state().clone()
    trial, cold, result = simulate(model, frontend, codec, "fixture", [features] * 4, tmp_path)
    assert result["model_adam_rng_recovery_exact"] and result["simulated_optimizer_step_applications"] == 3
    assert 0 < result["software_cost_evaluations"] <= 266
    assert result["experimental_training_updates"] == result["native_requests"] == 0
    assert all(value > 0 for value in result["second"]["gradient_groups"].values())
    assert torch.equal(rng, torch.get_rng_state())
    assert tensor_digest(trial.state_dict()) != tensor_digest(model.state_dict())
    assert len(preserve(base, trial, cold, codec, {"recipe": {"features": features, "state_id": "train", "audio_sha256": "audio"}}, initial=False)) == 1


@pytest.mark.parametrize("change", ["base", "correction", "scale", "profile"])
def test_direct_cold_export_rejects_changed_contract_or_weights(data, change):
    base, codec, _features, frontend = data
    saved = copy.deepcopy(export_direct(BasicDirect(base), frontend, codec, "fixture"))
    if change == "base":
        saved["base"]["model"]["continuous.bias"][0] += .1
    elif change == "correction":
        saved["correction"]["2.bias"][0] += .1
    elif change == "scale":
        saved["spec"]["adapter_scales"][0] = 2.
    else:
        saved["render_phase_controls"]["osc_1_random_phase"] = .5
    with pytest.raises(ValueError, match="changed|phase|profile"):
        restore_direct(saved)


def test_unverified_reference_parent_is_refused_before_loading_data(tmp_path):
    from obruxo_training.audit import write_json
    write_json(tmp_path / "verification.json", {"verified": False})
    write_json(tmp_path / "plan.json", {})
    with pytest.raises(ValueError, match="audited unison"):
        reference_manifest(tmp_path, tmp_path / "missing")


def test_reference_baseline_must_be_the_accepted_initial_prediction(tmp_path, data):
    base, codec, features, _front = data
    model = BasicDirect(base)
    adapter = ProductionControlAdapter(codec, scales=SPEC["adapter_scales"])
    output = model(batch_features([{"features": features}]))
    target = adapter.export(output["continuous"], output["categorical"])[0].to_dict()
    target["continuous"][0] += .01
    audio = np.zeros((100, 2), dtype=np.float32)
    sha = audio_float32_sha256(audio)
    torch.save({"audio": torch.from_numpy(audio)}, tmp_path / "source.pt")
    row = {"state_id": "train", "recipe_id": "recipe", "features": features, "audio_sha256": sha}
    reference = {"state_id": "train", "recipe_id": "recipe", "source_audio_sha256": sha, "features_sha256": tensor_digest(features),
                 "source_file": "source.pt", "initial_patch": target}
    with pytest.raises(ValueError, match="accepted initializer"):
        validate_references({"reference_manifest": [reference]}, tmp_path, model, codec, {"recipe": row})

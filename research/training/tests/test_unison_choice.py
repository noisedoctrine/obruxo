import copy

import pytest
import torch

from obruxo_training.features import AudioFeatures, BasicPitchICASSP2022, PITCH_SIZE, TIMBRE_SHAPE, tensor_digest
from obruxo_training.inference_profile import bind_rendering, prediction_preset
from obruxo_training.model import PilotInverse, batch_features
from obruxo_training.codec import PilotCodec
from obruxo_training.production_controls import ProductionControlAdapter
from obruxo_training.profiled_sampler import profile_metadata
from obruxo_training.unison_choice import COUNTS, SCALES, SPEC, UnisonChoice, export_unison, measured_choice_loss, restore_unison
from obruxo_training.unison_choice_bridge import gradient_access, preserve_predictions, simulate_recovery
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


def test_initialization_preserves_production_choices_probabilities_and_rng(data):
    base, codec, features, _front = data
    rng, original = torch.get_rng_state().clone(), tensor_digest(base.state_dict())
    model = UnisonChoice(base)
    assert model.require_frozen() == COUNTS
    assert torch.equal(rng, torch.get_rng_state()) and tensor_digest(base.state_dict()) == original
    inputs = batch_features([{"features": features}])
    first, second = base(inputs), model(inputs)
    assert torch.equal(first["continuous"], second["continuous"])
    assert torch.equal(first["categorical"][1], second["categorical"][1])
    for temperature in (1., .5):
        assert torch.equal((first["categorical"][0] / SCALES[0] / temperature).softmax(-1),
                           (second["categorical"][0] / temperature).softmax(-1))
    old, new = ProductionControlAdapter(codec, scales=SCALES), ProductionControlAdapter(codec, scales=SPEC["adapter_scales"])
    targets = [adapter.export(output["continuous"], output["categorical"])[0] for adapter, output in ((old, first), (new, second))]
    assert targets[0] == targets[1]
    assert prediction_preset(base, codec, targets[0]) == prediction_preset(model, codec, targets[1])


def test_cached_costs_match_analytic_derivative_and_frozen_paths(data):
    base, codec, features, _front = data
    model = UnisonChoice(base)
    _old, direction = measured_direction(base, codec, features)
    result = gradient_access(model, features, direction)
    assert result["responding"] and result["weights_and_rng_unchanged"] and result["frozen_gradients_absent"]
    assert result["gradient_groups"]["output"] > 0 and result["gradient_groups"]["hidden"] == 0
    assert all(row["maximum_derivative_error"] < 1e-7 for row in result["derivative_checks"])
    direction["production_patch"]["categorical"][0] = (direction["production_patch"]["categorical"][0] + 1) % 3
    with pytest.raises(ValueError, match="hard anchor"):
        gradient_access(model, features, direction)


def test_simulated_recovery_preserves_other_controls_after_learning(tmp_path, data):
    base, codec, features, frontend = data
    model = UnisonChoice(base)
    rng = torch.get_rng_state().clone()
    trial, cold, result = simulate_recovery(model, frontend, codec, "fixture", features, [.5, .2, .8], tmp_path)
    assert result["model_adam_rng_recovery_exact"] and result["simulated_optimizer_step_applications"] == 3
    assert result["experimental_training_updates"] == 0 and torch.equal(rng, torch.get_rng_state())
    assert result["second"]["gradient_groups"]["hidden"] > 0
    bank = {"recipe": {"features": features, "state_id": "train", "audio_sha256": "audio"}}
    assert len(preserve_predictions(base, trial, cold, codec, bank, calibrated_initial=False)) == 1
    assert tensor_digest(model.state_dict()) != tensor_digest(trial.state_dict())
    assert tensor_digest(base.state_dict()) == tensor_digest(trial.base.state_dict())


def test_frozen_mask_and_stale_continuous_anchor_are_refused(data):
    base, codec, features, _front = data
    model = UnisonChoice(base)
    _adapter, direction = measured_direction(base, codec, features)
    direction["raw_prediction"]["continuous"][0] += .01
    with pytest.raises(ValueError, match="stale"):
        gradient_access(model, features, direction)
    model.base.continuous.bias.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen predictor"):
        model.require_frozen()


@pytest.mark.parametrize("change", ["scale", "profile", "base", "correction"])
def test_cold_export_rejects_changed_calibration_profile_or_weights(data, change):
    base, codec, _features, frontend = data
    saved = copy.deepcopy(export_unison(UnisonChoice(base), frontend, codec, "fixture"))
    if change == "scale":
        saved["spec"]["base_unison_scale"] = 1.
    elif change == "profile":
        saved["render_phase_controls"]["osc_1_random_phase"] = .5
    elif change == "base":
        saved["base"]["model"]["continuous.bias"][0] += .1
    else:
        saved["correction"]["2.bias"][0] += .1
    with pytest.raises(ValueError, match="changed|profile|phase"):
        restore_unison(saved)


@pytest.mark.parametrize("costs,temperature", [([[-1., 1., 2.]], 1.), ([[0., float("nan"), 1.]], 1.), ([[1., 2.]], 1.), ([[1., 2., 3.]], 0.)])
def test_expected_measured_cost_rejects_invalid_data(costs, temperature):
    with pytest.raises(ValueError, match="measured three-choice"):
        measured_choice_loss({"categorical": [torch.zeros(1, 3)]}, costs, temperature=temperature)

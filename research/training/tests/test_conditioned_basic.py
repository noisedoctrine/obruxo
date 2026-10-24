import copy
from types import SimpleNamespace

import pytest
import torch

from obruxo_training.conditioned_basic import COUNTS, apply_policy, bridge_state, gradient_access, make_conditioned, restore_conditioned
from obruxo_training.conditioned_bridge import training_bank
from obruxo_training.codec import PilotCodec
from obruxo_training.features import AudioFeatures, BasicPitchICASSP2022, PITCH_SIZE, TIMBRE_SHAPE, tensor_digest
from obruxo_training.inference_profile import bind_rendering, prediction_preset
from obruxo_training.model import PilotInverse, batch_features
from obruxo_training.native_inverse import native_reconstruction_loss
from obruxo_training.production_controls import ProductionControlAdapter
from obruxo_training.production_direction import raw_prediction
from obruxo_training.profiled_sampler import profile_metadata


@pytest.fixture
def data():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    base, codec = PilotInverse().eval(), PilotCodec()
    bind_rendering(base, profile_metadata(32768))
    features = {"timbre": torch.rand(*TIMBRE_SHAPE), "pitch": torch.rand(PITCH_SIZE)}
    return base, codec, features


def test_bridge_preserves_predictions_presets_rng_and_named_parameter_budget(data):
    base, codec, features = data
    before, rng = tensor_digest(base.state_dict()), torch.get_rng_state().clone()
    model = make_conditioned(base, codec)
    assert torch.equal(rng, torch.get_rng_state()) and tensor_digest(base.state_dict()) == before
    assert apply_policy(model, model.conditioning_training_policy) == COUNTS
    assert float(model.conditioning_gain.detach()) == 1 and not torch.count_nonzero(model.conditioning_projection.weight)
    with torch.no_grad():
        first, second = [network(batch_features([{"features": features}])) for network in (base, model)]
    assert torch.equal(first["continuous"], second["continuous"])
    assert all(torch.equal(a, b) for a, b in zip(first["categorical"], second["categorical"], strict=True))
    adapter = ProductionControlAdapter(codec)
    target = adapter.export(first["continuous"], first["categorical"])[0]
    assert prediction_preset(base, codec, target) == prediction_preset(model, codec, target)


def measured_direction(base, codec, features):
    adapter = ProductionControlAdapter(codec, scales=(32., 29.))
    prediction = base(batch_features([{"features": features}]))
    def evaluate(_index, candidate):
        return sum((value - .2)**2 for value in candidate.continuous) + .1 * sum(candidate.categorical)
    _loss, estimates = native_reconstruction_loss(prediction, adapter, evaluate)
    return adapter, {"raw_prediction": raw_prediction(prediction), "production_patch": adapter.export(prediction["continuous"], prediction["categorical"])[0].to_dict(),
                     "estimate": estimates[0].to_dict()}


def test_measured_loss_reaches_correction_and_one_step_keeps_shared_weights_fixed(data):
    base, codec, features = data
    model = make_conditioned(base, codec)
    adapter, direction = measured_direction(base, codec, features)
    report = gradient_access(model, features, adapter, direction)
    assert report["frozen_gradients_absent"] and report["weights_and_rng_unchanged"]
    assert report["correction_group_gradient_l1"]["conditioning_projection"] > 0
    original = model.conditioning_training_policy["frozen_weights_sha256"]
    torch.optim.Adam(model.parameters(), lr=.001).step()
    apply_policy(model, model.conditioning_training_policy)
    assert tensor_digest({name: value for name, value in model.state_dict().items() if name in model.conditioning_training_policy["frozen_names"]}) == original
    assert torch.count_nonzero(model.conditioning_projection.weight)


def test_stale_gradient_and_altered_freeze_policy_are_refused(data):
    base, codec, features = data
    model = make_conditioned(base, codec)
    adapter, direction = measured_direction(base, codec, features)
    direction["raw_prediction"]["continuous"][0] += .1
    with pytest.raises(ValueError, match="stale"):
        gradient_access(model, features, adapter, direction)
    policy = copy.deepcopy(model.conditioning_training_policy)
    policy["trainable_names"].append(policy["frozen_names"].pop())
    with pytest.raises(ValueError, match="policy"):
        apply_policy(model, policy)
    with torch.no_grad():
        model.continuous.bias[0] += .1
    with pytest.raises(ValueError, match="frozen"):
        apply_policy(model, model.conditioning_training_policy)


def test_cold_export_reapplies_freeze_policy_and_preserves_fixed_phases(tmp_path, data):
    base, codec, features = data
    model = make_conditioned(base, codec)
    frontend = AudioFeatures(BasicPitchICASSP2022().state_dict())
    artifact = bridge_state(model, frontend, codec, "training-fixture")
    path = tmp_path / "model.pt"
    torch.save(artifact, path)
    restored, restored_frontend, restored_codec = restore_conditioned(torch.load(path, map_location="cpu", weights_only=True))
    assert apply_policy(restored, artifact["conditioning_training_policy"]) == COUNTS
    assert tensor_digest(restored.state_dict()) == tensor_digest(model.state_dict())
    assert restored_frontend.describe() == frontend.describe() and restored_codec.to_dict() == codec.to_dict()
    adapter = ProductionControlAdapter(codec)
    prediction = base(batch_features([{"features": features}]))
    target = adapter.export(prediction["continuous"], prediction["categorical"])[0]
    assert prediction_preset(restored, restored_codec, target) == prediction_preset(base, codec, target)


def test_held_out_inputs_and_inconsistent_duplicate_recipes_are_refused(data):
    _base, _codec, features = data
    pool = SimpleNamespace(manifest=SimpleNamespace(by_state={"train": SimpleNamespace(split="train")}))
    row = {"state_id": "held", "recipe_id": "same", "features": features, "audio_sha256": "same"}
    with pytest.raises(ValueError, match="held-out"):
        training_bank({"replay": [row]}, pool)
    row["state_id"] = "train"
    changed = {**row, "audio_sha256": "changed"}
    with pytest.raises(ValueError, match="duplicate"):
        training_bank({"replay": [row, changed]}, pool)
    with pytest.raises(ValueError, match="404-state"):
        training_bank({"replay": [row]}, pool)

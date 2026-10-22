from collections import Counter
import copy

import pytest

torch = pytest.importorskip("torch")

from obruxo_training.codec import PilotCodec  # noqa: E402
from obruxo_training.features import PITCH_SIZE, TIMBRE_SHAPE, tensor_digest  # noqa: E402
from obruxo_training.model import PilotInverse, batch_features, loss_terms  # noqa: E402
from obruxo_training.paired_native_step import matched_pair_step, reciprocal_updates  # noqa: E402
from obruxo_training.production_controls import ProductionControlAdapter  # noqa: E402


def setup_pair():
    torch.set_num_threads(2)
    torch.manual_seed(12)
    codec, initial = PilotCodec(), PilotInverse()
    with torch.no_grad():
        initial.categorical[0].weight.zero_()
        initial.categorical[0].bias.copy_(torch.tensor([-40., 80., -20.]))
        initial.categorical[1].weight.zero_()
        initial.categorical[1].bias.copy_(torch.tensor([-50., 60.]))
    models = {name: copy.deepcopy(initial) for name in ("parameter", "spectral")}
    optimizers = {name: torch.optim.Adam(model.parameters(), lr=.0001) for name, model in models.items()}
    features = [{"timbre": torch.rand(*TIMBRE_SHAPE), "pitch": torch.rand(PITCH_SIZE)} for _ in range(2)]
    return codec, models, optimizers, features, [codec.default_target()] * 2


def test_schedule_keeps_reciprocal_views_in_every_update_with_original_weight_unchanged():
    rng = torch.get_rng_state().clone()
    updates = reciprocal_updates(64)
    assert updates == reciprocal_updates(64) and torch.equal(rng, torch.get_rng_state())
    assert len(updates) == 128
    examples = [row for update in updates for row in update["examples"]]
    assert len(examples) == 256
    assert Counter(row["input_view"] for row in examples) == {0: 128, 1: 64, 2: 64}
    for update in updates:
        original, wider = update["examples"]
        assert original["state"] == wider["state"]
        assert original["input_view"] == wider["view"] == 0
        assert original["view"] == wider["input_view"] in (1, 2)
    assert Counter((row["state"], row["input_view"], row["view"]) for row in examples) == Counter(
        {(index, source, target): 1 for index in range(64) for source, target in ((0, 1), (1, 0), (0, 2), (2, 0))})


def test_both_observations_are_independently_measured_and_control_cross_entropy_is_unscaled():
    codec, models, optimizers, features, targets = setup_pair()
    before = {name: tensor_digest(model.state_dict()) for name, model in models.items()}
    expected_control = float(loss_terms(models["parameter"](batch_features([{"features": row} for row in features])), targets, codec)["loss"].detach())
    adapter = ProductionControlAdapter(codec, scales=(32., 29.))
    production = []
    with torch.no_grad():
        for row in features:
            prediction = models["spectral"](batch_features([{"features": row}]))
            production.append(adapter.export(prediction["continuous"], prediction["categorical"])[0].to_dict())
    calls = []

    def evaluate(index, patch):
        calls.append((index, patch))
        return sum((value - .2 * index) ** 2 for value in patch.continuous) + .4 * patch.categorical[0] + .7 * patch.categorical[1]

    result = matched_pair_step(models, optimizers, features, targets, codec, evaluate, 0, 128, scales=(32., 29.))
    assert result["losses"]["parameter"] == expected_control and expected_control > 100
    assert result["production_patches"] == production
    assert len(calls) == result["gradient_queries"] <= 44
    assert Counter(index for index, _patch in calls) == {index: row["queries"] for index, row in enumerate(result["estimates"])}
    assert result["losses"]["spectral"] == pytest.approx(sum(row["value"] for row in result["estimates"]) / 2)
    assert len(result["production_patches"]) == 2 and all(value > 0 for value in result["gradients"]["continuous_l1"])
    assert all(tensor_digest(models[name].state_dict()) != digest for name, digest in before.items())


def test_native_failure_does_not_partially_advance_the_parameter_control():
    codec, models, optimizers, features, targets = setup_pair()
    before = {name: tensor_digest(model.state_dict()) for name, model in models.items()}

    def fail_second(index, patch):
        if index == 1:
            raise RuntimeError("native second-example failure")
        return sum(patch.continuous) + sum(patch.categorical)

    with pytest.raises(RuntimeError, match="second-example"):
        matched_pair_step(models, optimizers, features, targets, codec, fail_second, 0, 128, scales=(32., 29.))
    assert before == {name: tensor_digest(model.state_dict()) for name, model in models.items()}
    assert all(not optimizer.state for optimizer in optimizers.values())


def test_restored_models_and_adam_reproduce_the_next_pair_exactly():
    codec, models, optimizers, features, targets = setup_pair()
    def evaluate(index, patch):
        return sum((value - .2 * index) ** 2 for value in patch.continuous) + sum(patch.categorical)
    def update(networks, opts, step):
        return matched_pair_step(networks, opts, features, targets, codec, evaluate, step, 128, scales=(32., 29.))
    update(models, optimizers, 0)
    states = copy.deepcopy({name: model.state_dict() for name, model in models.items()})
    adam = copy.deepcopy({name: optimizer.state_dict() for name, optimizer in optimizers.items()})
    rng = torch.get_rng_state().clone()
    expected = update(models, optimizers, 1)
    _codec, recovered, opts, _features, _targets = setup_pair()
    for name in recovered:
        recovered[name].load_state_dict(states[name])
        opts[name].load_state_dict(adam[name])
    torch.set_rng_state(rng)
    assert update(recovered, opts, 1) == expected
    assert all(tensor_digest(recovered[name].state_dict()) == tensor_digest(models[name].state_dict()) for name in models)


@pytest.mark.parametrize("count", (0, -1, True))
def test_invalid_schedule_rejected(count):
    with pytest.raises(ValueError, match="positive state"):
        reciprocal_updates(count)

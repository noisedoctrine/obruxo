import copy

import pytest

torch = pytest.importorskip("torch")

from obruxo_training.features import PITCH_SIZE, TIMBRE_SHAPE, tensor_digest  # noqa: E402
from obruxo_training.model import PilotInverse  # noqa: E402
from obruxo_training.production_controls import ProductionControlAdapter  # noqa: E402
from obruxo_training.production_direction import GATES, direction_summary, guided_prediction  # noqa: E402


def test_sound_loss_reaches_real_network_and_replays_without_updating_weights_or_rng():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    model, adapter = PilotInverse().eval(), ProductionControlAdapter(scales=(32., 29.))
    features = {"timbre": torch.rand(*TIMBRE_SHAPE), "pitch": torch.rand(PITCH_SIZE)}
    before, rng = tensor_digest(model.state_dict()), torch.get_rng_state().clone()
    calls = []

    def evaluate(index, patch):
        assert index == 0 and patch == adapter.codec.canonical_target(patch)
        calls.append(patch)
        return sum(value * value for value in patch.continuous) + .4 * patch.categorical[0] + .7 * patch.categorical[1]

    result = guided_prediction(model, features, adapter, evaluate)
    assert len(calls) == result["estimate"]["queries"] <= 22
    assert result["gradients"]["model_parameter_gradient_l1"] > 0
    assert all(value > 0 for value in result["gradients"]["category_parameter_gradient_l1"])
    assert before == tensor_digest(model.state_dict()) and torch.equal(rng, torch.get_rng_state())
    assert result["weights_and_rng_unchanged"]
    measured = {patch: sum(value * value for value in patch.continuous) + .4 * patch.categorical[0] + .7 * patch.categorical[1] for patch in calls}
    recovered = copy.deepcopy(model)
    assert guided_prediction(recovered, features, adapter, lambda index, patch: measured[patch]) == result


def direction_cases():
    return [{"index": index, "role": role, "repeat_exact": True,
             "scores": {"start": {"mrstft": 1., "spectral": 1.}, "continuous": {"mrstft": .8, "spectral": .8},
                        "joint": {"mrstft": .7, "spectral": .7}},
             "direction": {"weights_and_rng_unchanged": True, "estimate": {"categorical": [[1., .8, .6], [1., .8]]},
                           "gradients": {"model_parameter_gradient_l1": 1., "categorical": [[.1, -.1, 0.], [.1, -.1]],
                                         "category_parameter_gradient_l1": [1., 1.]}}}
            for index in range(12) for role in GATES["roles"]]


@pytest.mark.parametrize("failure", ("original_regression", "zero_category", "zero_model", "repeat", "missing_case"))
def test_direction_gates_cannot_hide_original_or_gradient_failure(failure):
    cases = direction_cases()
    plan = {"cases": [{"index": row["index"], "role": row["role"]} for row in cases], "gates": GATES}
    assert direction_summary(cases, plan)["direction_gates_passed"]
    if failure == "original_regression":
        for row in cases:
            if row["role"] == "original_input":
                row["scores"]["joint"]["mrstft"] = 1.1
    elif failure == "zero_category":
        cases[0]["direction"]["gradients"]["categorical"][0] = [0., 0., 0.]
    elif failure == "zero_model":
        cases[0]["direction"]["gradients"]["model_parameter_gradient_l1"] = 0.
    elif failure == "repeat":
        cases[0]["repeat_exact"] = False
    else:
        cases.pop()
    assert not direction_summary(cases, plan)["direction_gates_passed"]

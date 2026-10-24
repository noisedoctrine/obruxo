"""A zero-output note-conditioned correction with the accepted predictor frozen."""

import copy

import torch

from .codec import PilotCodec, PilotTarget
from .features import tensor_digest
from .inference_profile import bind_rendering, model_rendering
from .model import PilotInverse, batch_features
from .native_gradient import NativeGradient, measured_loss
from .production_direction import raw_prediction
from .trainer import inference_state, restore_model

SHARED_GROUPS = {"timbre", "pitch", "fusion", "continuous", "categorical"}
CORRECTION_GROUPS = {"performance_projection", "timbre_projection", "conditioning_attention", "conditioning_projection",
                     "performance_position", "timbre_position", "conditioning_gain"}
COUNTS = {"total_parameters": 426271, "frozen_parameters": 355806, "trainable_parameters": 70465, "dense_outputs": 14}
POLICY = "frozen_basic_conditioning_correction_v1"


def apply_policy(model, policy):
    parameters = dict(model.named_parameters())
    shared = sorted(name for name in parameters if name.split(".")[0] in SHARED_GROUPS)
    correction = sorted(name for name in parameters if name.split(".")[0] in CORRECTION_GROUPS)
    if (model.conditioning != "cross_attention" or policy["format"] != POLICY or policy["initialization"] != "zero_output_projection_gain_one_v1"
            or set(shared + correction) != set(parameters)
            or policy["frozen_names"] != shared or policy["trainable_names"] != correction
            or tensor_digest({name: parameters[name] for name in shared}) != policy["frozen_weights_sha256"]):
        raise ValueError("conditioning policy or frozen predictor weights changed")
    actual = {"total_parameters": sum(value.numel() for value in parameters.values()),
              "frozen_parameters": sum(parameters[name].numel() for name in shared),
              "trainable_parameters": sum(parameters[name].numel() for name in correction),
              "dense_outputs": model.continuous.out_features + sum(head.out_features for head in model.categorical)}
    if actual != COUNTS:
        raise ValueError("conditioning bridge changed its parameter or output budget")
    for name, parameter in parameters.items():
        parameter.requires_grad_(name in correction)
    model.conditioning_training_policy = copy.deepcopy(policy)
    return actual


def make_conditioned(base, codec, *, seed=0):
    if type(codec) is not PilotCodec or base.conditioning != "concat" or base.cyclic_head != "sigmoid" or type(seed) is not int:
        raise ValueError("conditioning bridge requires the original basic predictor")
    shared = sorted(dict(base.named_parameters()))
    if any(name.split(".")[0] not in SHARED_GROUPS for name in shared):
        raise ValueError("unexpected shared prediction parameters")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = PilotInverse(codec, conditioning="cross_attention").eval()
    loaded = model.load_state_dict(base.state_dict(), strict=False)
    correction = sorted(set(dict(model.named_parameters())) - set(shared))
    if loaded.unexpected_keys or sorted(loaded.missing_keys) != correction:
        raise ValueError("accepted predictor does not fill the complete shared model")
    with torch.no_grad():
        # A zero final projection preserves predictions and exposes 12,480 weights at the first backward pass.
        model.conditioning_gain.fill_(1.)
        model.conditioning_projection.weight.zero_()
        model.conditioning_projection.bias.zero_()
    policy = {"format": POLICY, "frozen_names": shared, "trainable_names": correction,
              "frozen_weights_sha256": tensor_digest(base.state_dict()), "correction_seed": seed,
              "initialization": "zero_output_projection_gain_one_v1"}
    apply_policy(model, policy)
    bind_rendering(model, model_rendering(base))
    return model


def bridge_state(model, frontend, codec, pool_id):
    policy = model.conditioning_training_policy
    apply_policy(model, policy)
    return {**inference_state(model, frontend, codec, pool_id), "conditioning_training_policy": copy.deepcopy(policy),
            "qualified_for_release": False}


def restore_conditioned(saved):
    model, frontend, codec = restore_model(saved)
    if type(codec) is not PilotCodec:
        raise ValueError("conditioning export requires the basic codec")
    apply_policy(model, saved["conditioning_training_policy"])
    return model, frontend, codec


def gradient_access(model, features, adapter, direction):
    before, rng = tensor_digest(model.state_dict()), torch.get_rng_state().clone()
    model.zero_grad(set_to_none=True)
    prediction = model(batch_features([{"features": features}]))
    if raw_prediction(prediction) != direction["raw_prediction"] or adapter.export(prediction["continuous"], prediction["categorical"])[0].to_dict() != direction["production_patch"]:
        raise ValueError("cached native gradient is stale for the conditioned prediction")
    saved = direction["estimate"]
    estimate = NativeGradient(PilotTarget.from_dict(saved["anchor"]), saved["value"], tuple(saved["continuous"]),
                              tuple(tuple(row) for row in saved["categorical"]), saved["queries"])
    raw = prediction["continuous"]
    encoded = adapter(raw, prediction["categorical"], temperature=1., sample=True)
    choices = (encoded[:, 9:12], encoded[:, 12:])
    loss = measured_loss(raw, choices, estimate)
    projection = tuple(model.conditioning_projection.parameters())
    def norm(component):
        values = torch.autograd.grad(component, projection, retain_graph=True)
        if any(not torch.isfinite(value).all() for value in values):
            raise ValueError("nonfinite correction gradient")
        return sum(float(value.double().abs().sum()) for value in values)
    continuous = [norm(raw[0, index] * coefficient) if abs(coefficient) > 1e-7 else None for index, coefficient in enumerate(estimate.continuous)]
    categorical = [norm((choice * choice.new_tensor([costs])).sum()) if max(costs) - min(costs) > 1e-7 else None
                   for choice, costs in zip(choices, estimate.categorical, strict=True)]
    if any(value is not None and value <= 0 for value in (*continuous, *categorical)):
        raise ValueError("a responding native control cannot reach the correction")
    loss.backward()
    groups = {}
    for name, parameter in model.named_parameters():
        if name in model.conditioning_training_policy["frozen_names"]:
            if parameter.grad is not None:
                raise ValueError("sound loss reached a frozen predictor parameter")
            continue
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise ValueError("missing or nonfinite correction gradient")
        key = name.split(".")[0]
        groups[key] = groups.get(key, 0.) + float(parameter.grad.double().abs().sum())
    if groups["conditioning_projection"] <= 0 or any(value for key, value in groups.items() if key != "conditioning_projection"):
        raise ValueError("zero-projection initialization does not have the declared first-step gradient path")
    if tensor_digest(model.state_dict()) != before or not torch.equal(rng, torch.get_rng_state()):
        raise ValueError("gradient qualification changed weights or RNG")
    return {"continuous_projection_gradient_l1": continuous, "category_projection_gradient_l1": categorical,
            "correction_group_gradient_l1": groups, "weights_and_rng_unchanged": True, "frozen_gradients_absent": True}

"""One fixed output-space sound-loss step through the actual inverse network."""

import torch

from .features import tensor_digest
from .model import batch_features
from .native_gradient_screen import qualification_summary
from .native_inverse import native_reconstruction_loss

UPDATE = {"continuous_radius": .01, "continuous_scaling": "divide_by_gradient_max_absolute",
          "category_step_in_fixed_scale_units": 1., "category_scaling": "each_head_divide_by_gradient_max_absolute",
          "temperature": 1., "steps": 1, "backtracking": False, "optimizer_updates": 0,
          "claim": "local_output_direction_and_model_gradient_connection_not_inverse_training"}
GATES = {"roles": ["original_input", "wider_input"], "cases_per_role": 12, "methods": ["continuous", "joint"],
         "minimum_improved_fraction": .6, "mean_mrstft_delta_below": 0., "mean_spectral_delta_below": 0.,
         "repeat_exact": True, "all_native_accepted": True, "nonzero_model_gradient": True,
         "nonzero_category_gradient_when_native_loss_span_exceeds": 1e-7}


def raw_prediction(prediction):
    return {"continuous": prediction["continuous"].detach()[0].tolist(),
            "categorical": [scores.detach()[0].tolist() for scores in prediction["categorical"]]}


def guided_prediction(model, features, adapter, evaluate):
    if model.training:
        raise ValueError("production-direction check requires evaluation mode")
    before, rng = tensor_digest(model.state_dict()), torch.get_rng_state().clone()
    model.zero_grad(set_to_none=True)
    prediction = model(batch_features([{"features": features}]))
    continuous, logits = prediction["continuous"], prediction["categorical"]
    for value in (continuous, *logits):
        value.retain_grad()
    initial = raw_prediction(prediction)
    production = adapter.export(continuous, logits)[0]
    loss, estimates = native_reconstruction_loss(prediction, adapter, evaluate, temperature=UPDATE["temperature"])
    if len(estimates) != 1 or adapter.codec.canonical_target(estimates[0].anchor) != production or not torch.isfinite(loss):
        raise ValueError("native forward differs from production or has an invalid loss")
    loss.backward()
    parameters = {name: value.grad for name, value in model.named_parameters() if value.grad is not None}
    if not parameters or any(not torch.isfinite(value).all() for value in parameters.values()):
        raise ValueError("invalid model gradients from native sound loss")
    gradients = {"continuous": continuous.grad[0].tolist(), "categorical": [scores.grad[0].tolist() for scores in logits],
                 "model_gradient_sha256": tensor_digest(parameters),
                 "model_parameter_gradient_l1": sum(float(value.double().abs().sum()) for value in parameters.values()),
                 "category_parameter_gradient_l1": [sum(float(value.grad.double().abs().sum()) for value in head.parameters() if value.grad is not None)
                                                     for head in model.categorical]}
    with torch.no_grad():
        changed = (continuous - UPDATE["continuous_radius"] * continuous.grad / continuous.grad.abs().max().clamp_min(1e-12)).clamp(0, 1)
        updated = [scores - scale * UPDATE["category_step_in_fixed_scale_units"] * scores.grad / scores.grad.abs().max().clamp_min(1e-12)
                   for scores, scale in zip(logits, adapter.scales, strict=True)]
        candidates = {"continuous": adapter.export(changed, logits)[0], "joint": adapter.export(changed, updated)[0]}
    if before != tensor_digest(model.state_dict()) or not torch.equal(rng, torch.get_rng_state()):
        raise ValueError("direction check changed weights or RNG")
    return {"raw_prediction": initial, "production_patch": production.to_dict(), "loss": float(loss.detach()),
            "estimate": estimates[0].to_dict(), "gradients": gradients,
            "candidates": {name: candidate.to_dict() for name, candidate in candidates.items()}, "weights_and_rng_unchanged": True}


def direction_summary(cases, plan):
    result = qualification_summary(cases, plan)
    connected = bool(cases) and all(row["direction"]["gradients"]["model_parameter_gradient_l1"] > 0 for row in cases)
    category_checks = []
    for row in cases:
        estimate, gradients = row["direction"]["estimate"], row["direction"]["gradients"]
        for head, costs in enumerate(estimate["categorical"]):
            if max(costs) - min(costs) <= GATES["nonzero_category_gradient_when_native_loss_span_exceeds"]:
                continue
            output_norm = sum(abs(value) for value in gradients["categorical"][head])
            category_checks.append(output_norm > 0 and gradients["category_parameter_gradient_l1"][head] > 0)
    unchanged = bool(cases) and all(row["direction"]["weights_and_rng_unchanged"] for row in cases)
    passed = result["direction_gates_passed"] and connected and bool(category_checks) and all(category_checks) and unchanged
    return {**result, "native_loss_reaches_model_weights": connected, "responding_category_checks": len(category_checks),
            "responding_category_gradients_connected": bool(category_checks) and all(category_checks),
            "weights_and_rng_unchanged": unchanged, "direction_gates_passed": bool(passed)}

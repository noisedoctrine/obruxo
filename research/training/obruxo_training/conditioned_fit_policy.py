"""Enforce the qualified correction mask during optimization and replay."""

import torch

from .conditioned_basic import CORRECTION_GROUPS, apply_policy


def restore_policy(model, saved):
    if "conditioning_training_policy" in saved:
        apply_policy(model, saved["conditioning_training_policy"])
    return model


def require_frozen(model):
    policy = getattr(model, "conditioning_training_policy", None)
    if policy is None:
        return
    parameters = dict(model.named_parameters())
    if any(parameters[name].requires_grad or parameters[name].grad is not None for name in policy["frozen_names"]):
        raise ValueError("frozen predictor acquired a gradient or trainable flag")
    if any(not parameters[name].requires_grad for name in policy["trainable_names"]):
        raise ValueError("correction parameter was unexpectedly frozen")
    apply_policy(model, policy)  # Checks the complete frozen tensor digest, including after every step.


def gradient_groups(model):
    require_frozen(model)
    groups = dict.fromkeys(sorted(CORRECTION_GROUPS), 0.)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise ValueError("missing or nonfinite correction gradient")
        groups[name.split(".")[0]] += float(parameter.grad.double().abs().sum())
    return groups


def learning_access(trace):
    groups = {arm: {group: any(row["correction_gradients"][arm][group] > 0 for row in trace[1:])
                    for group in sorted(CORRECTION_GROUPS)} for arm in ("parameter", "spectral")}
    return {"after_first_update": groups, "pass": len(trace) > 1 and all(all(row.values()) for row in groups.values())}

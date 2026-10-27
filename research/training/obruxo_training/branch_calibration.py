"""Measure encoder-gradient pressure without updating weights or optimizer state."""

import math
from statistics import median

import torch

from .paired import identity_terms

SPEC = {"format": "training_only_equal_branch_gradient_calibration_v1", "paired_batches": 8, "seed": 73,
        "target_note_to_identity_norm_ratio": 1., "fit": "target_divided_by_median_note_to_identity_norm_ratio",
        "coefficient_limits": [.001, 1000.], "on_out_of_range": "fail_without_clamping", "after_detached_warmup": 100,
        "during_training": "fixed_coefficient_linear_ramp_first100_updates", "validation_used": False}


def measure(arm, mel, targets):
    parameters = tuple(arm["encoder"].parameters())
    representation = arm["encoder"](mel)
    primary = identity_terms(representation)["identity_loss"]
    note = (arm["head"](representation, coefficient=1.) - targets).square().mean()
    a = torch.autograd.grad(primary, parameters, retain_graph=True)
    b = torch.autograd.grad(note, parameters)
    a, b = (torch.cat([value.flatten() for value in gradients]).double() for gradients in (a, b))
    primary_norm, note_norm = float(a.norm()), float(b.norm())
    if not all(math.isfinite(value) and value > 0 for value in (primary_norm, note_norm)):
        raise ValueError("calibration requires finite nonzero encoder gradients for both objectives")
    return {"identity_norm": primary_norm, "unscaled_reversed_note_norm": note_norm,
            "unscaled_note_to_identity_norm_ratio": note_norm / primary_norm,
            "gradient_cosine": float(a.dot(b) / (a.norm() * b.norm()))}


def coefficient(measurements):
    if len(measurements) != SPEC["paired_batches"]:
        raise ValueError("calibration needs exactly eight predeclared training batches")
    ratios = []
    for row in measurements:
        primary, note = row["identity_norm"], row["unscaled_reversed_note_norm"]
        if not all(math.isfinite(value) and value > 0 for value in (primary, note)):
            raise ValueError("cannot calibrate from nonfinite or zero gradients")
        ratios.append(note / primary)
    value = SPEC["target_note_to_identity_norm_ratio"] / median(ratios)
    lower, upper = SPEC["coefficient_limits"]
    if not lower <= value <= upper:
        raise ValueError("measured coefficient exceeds its frozen limits; preserve the failed calibration")
    return value

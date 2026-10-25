"""A small source-active continuous label signal alongside the native sound objective."""

import torch

from .basic_direct import REHEARSAL
from .codec import PilotCodec

SPEC = {"format": "paired_basic_continuous_auxiliary_v1", "weight": 1., "paired_weights": [.25, .5],
        "normalization": "mean_squared_error_over_source_active_continuous_controls_per_recording",
        "references": "unchanged_sound_and_reference_excess_objective", "categories": "native_sound_only",
        "label_inputs": False, "dense_outputs": 14}


def masked_continuous_loss(continuous, targets, codec):
    if (type(codec) is not PilotCodec or not targets or continuous.shape != (len(targets), 9)
            or continuous.dtype != torch.float32 or not torch.isfinite(continuous).all()
            or (continuous < 0).any() or (continuous > 1).any()):
        raise ValueError("continuous auxiliary requires bounded finite float32 basic predictions and source targets")
    for target in targets:
        codec.validate_target(target)
    expected = continuous.new_tensor([target.continuous for target in targets])
    mask = continuous.new_tensor([codec.continuous_mask(target) for target in targets])
    return (((continuous - expected).square() * mask).sum(-1) / mask.sum(-1).clamp_min(1)).mean()


def paired_continuous_loss(predictions, targets, codec):
    if len(predictions) != 2 or len(targets) != 2 or targets[0] != targets[1] or SPEC["paired_weights"] != REHEARSAL["paired_weights"]:
        raise ValueError("continuous auxiliary needs the reciprocal original/wider pair of one basic patch")
    return sum(weight * masked_continuous_loss(prediction["continuous"], [target], codec)
               for weight, prediction, target in zip(SPEC["paired_weights"], predictions, targets, strict=True))

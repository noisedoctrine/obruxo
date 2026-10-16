"""Differentiable reconstruction of the saved Era2 curves; no oracle inputs."""
import numpy as np
import torch


def _shift(values, phases):
    width = values.shape[-1]
    displacement = phases * width
    whole = displacement.floor()
    fraction = (displacement - whole)[..., None]
    right = (torch.arange(width, device=values.device) - whole.long()[..., None]).remainder(width)
    left = (right - 1).remainder(width)
    return values.gather(-1, right) * (1 - fraction) + values.gather(-1, left) * fraction


def decoded_curves(prediction, codec):
    device = prediction["continuous"].device
    if not hasattr(codec, "_tensor_dictionaries") or codec._tensor_dictionaries[0].device != device:
        assets = codec.lfo_bank.assets
        codec._tensor_dictionaries = (torch.tensor(assets.base_dictionary, dtype=torch.float32, device=device),
                                     torch.tensor(np.stack(assets.residual_layer_dictionaries), dtype=torch.float32, device=device))
    base, residuals = codec._tensor_dictionaries
    size = prediction["continuous"].shape[0]
    phases = prediction["continuous"][:, codec._ncont:codec._ncont + 136].reshape(size, 8, 17)
    gains = prediction["continuous"][:, codec._ncont + 136:codec._ncont + 264].reshape(size, 8, 16) * 4 - 2
    logits = prediction["categorical"][codec._ncat:]

    def mixed_atoms(layer, atoms):
        probabilities = torch.stack([logits[slot * 17 + layer].softmax(-1) for slot in range(8)], dim=1)
        return probabilities @ atoms

    values = _shift(mixed_atoms(0, base), phases[:, :, 0])
    for layer in range(16):
        values = values + _shift(mixed_atoms(layer + 1, residuals[layer]), phases[:, :, layer + 1]) * gains[:, :, layer, None]
    return values.clamp(0, 1)


def reconstruction_loss(prediction, targets, codec, curve_targets):
    curves = decoded_curves(prediction, codec)
    if (not isinstance(curve_targets, torch.Tensor) or curve_targets.shape != curves.shape or curve_targets.dtype != torch.float32
            or curve_targets.requires_grad or not torch.isfinite(curve_targets).all() or torch.any((curve_targets < 0) | (curve_targets > 1))):
        raise ValueError("original LFO curve targets must be frozen finite float32 [batch,8,97] values within [0,1]")
    expected = curve_targets.to(curves.device)
    mask = torch.tensor([codec.active_lfos(target) for target in targets], dtype=torch.float32, device=curves.device)
    error = (curves - expected).square().mean(-1)
    return (error * mask).sum() / mask.sum().clamp_min(1)

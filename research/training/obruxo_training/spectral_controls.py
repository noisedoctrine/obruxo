"""Basic patch tensors preserve category gradients until the explicit export boundary."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .codec import CATEGORICAL, CONTINUOUS, PilotCodec, PilotTarget
from .reproducibility import FULL_PHASE_CONTROL
from .spectral import spectral_spec


def temperature_at(step, steps):
    if type(step) is not int or type(steps) is not int or steps < 1 or not 0 <= step < steps:
        raise ValueError("invalid Gumbel schedule position")
    return 1. - .5 * step / max(1, steps - 1)


class BasicControlAdapter(nn.Module):
    def __init__(self, codec=None):
        super().__init__()
        self.codec = codec or PilotCodec()
        if type(self.codec) is not PilotCodec:
            raise ValueError("spectral basic scope requires the exact PilotCodec")
        defaults = [(default - lo) / (hi - lo) for _, lo, hi, default in CONTINUOUS]
        self.register_buffer("defaults", torch.tensor(defaults, dtype=torch.float32))

    def contract(self):
        return {"format": "basic_spectral_controls_v1", "codec": self.codec.to_dict(), "features": spectral_spec(),
                "input_size": 14, "canonicalization": "continuous_default_blend_with_hard_one_hot_activity",
                "training_choice": "straight_through_gumbel_softmax", "export_choice": "unperturbed_argmax",
                "temperature": {"start": 1., "end": .5, "schedule": "linear_by_committed_update"},
                "phase_policy": FULL_PHASE_CONTROL.copy(), "phase_application": "separate_hashed_render_copy"}

    def forward(self, continuous, logits, *, temperature=1., sample=True):
        if (continuous.ndim != 2 or continuous.shape[1] != 9 or continuous.dtype != torch.float32
                or not torch.isfinite(continuous).all() or (continuous < 0).any() or (continuous > 1).any()):
            raise ValueError("expected finite normalized basic controls [batch, 9]")
        if type(sample) is not bool or len(logits) != 2 or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("expected two category heads and positive finite temperature")
        choices = []
        for scores, (_, options) in zip(logits, CATEGORICAL, strict=True):
            if (scores.shape != (len(continuous), len(options)) or scores.device != continuous.device
                    or scores.dtype != continuous.dtype or not torch.isfinite(scores).all()):
                raise ValueError("invalid basic categorical scores")
            choice = (F.gumbel_softmax(scores, tau=temperature, hard=True, dim=-1) if sample else
                      F.one_hot(scores.argmax(-1), len(options)).to(continuous.dtype))
            choices.append(choice)
        return self._encode_choices(continuous, choices)

    def _encode_choices(self, continuous, choices):
        """No index conversion in the differentiable canonicalization path."""
        ones = torch.ones_like(continuous[:, 0])
        active = torch.stack((ones, ones, 1 - choices[0][:, 0], choices[1][:, 1], choices[1][:, 1],
                              ones, ones, ones, ones), dim=1)
        values = continuous * active + self.defaults * (1 - active)
        return torch.cat((values, *choices), dim=1)

    @torch.no_grad()
    def export(self, continuous, logits):
        encoded = self(continuous, logits, sample=False)
        categories = [scores.argmax(-1).cpu().tolist() for scores in logits]
        return [self.codec.canonical_target(PilotTarget(tuple(row[:9]), (categories[0][i], categories[1][i])))
                for i, row in enumerate(encoded.cpu().tolist())]

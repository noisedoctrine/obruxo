"""Render production categories and smooth only their backward derivative."""

import math

from .spectral_controls import BasicControlAdapter


class ProductionControlAdapter(BasicControlAdapter):
    def __init__(self, codec=None, *, scales=(1., 1.)):
        super().__init__(codec)
        if (not isinstance(scales, (tuple, list)) or len(scales) != 2
                or any(type(scale) not in (int, float) or not math.isfinite(scale) or scale < 1 for scale in scales)):
            raise ValueError("expected two fixed finite category scales at least one")
        self.scales = tuple(float(scale) for scale in scales)

    def contract(self):
        return {**super().contract(), "format": "basic_production_controls_v1",
                "training_choice": "unperturbed_argmax_with_scaled_softmax_backward",
                "backward_probability": "softmax(raw_logits / fixed_scale / temperature)",
                "fixed_scales": list(self.scales), "random_draws": 0,
                "parameter_control_logits": "raw_unscaled"}

    def forward(self, continuous, logits, *, temperature=1., sample=True):
        # Reuse the basic shape/domain checks and exact production tie handling.
        encoded = super().forward(continuous, logits, temperature=temperature, sample=False)
        if type(sample) is not bool:
            raise ValueError("expected a boolean backward-path selector")
        if not sample:
            return encoded
        choices = []
        for scores, scale, offset in zip(logits, self.scales, (9, 12), strict=True):
            hard = encoded[:, offset:offset + scores.shape[1]]
            soft = (scores / scale / temperature).softmax(-1)
            # Subtract before adding so the float32 forward value is exactly hard.
            choices.append(hard + (soft - soft.detach()))
        return self._encode_choices(continuous, choices)

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .codec import CATEGORICAL, CONTINUOUS, PilotCodec, PilotTarget
from .features import PITCH_SIZE


MODEL_SPEC = {"model": "pilot_stereo_cnn_pooled_pitch_v1", "timbre_channels": [16, 32],
              "timbre_pool": [6, 4], "pitch_width": 64, "hidden_width": 128,
              "continuous_heads": len(CONTINUOUS), "categorical_sizes": [len(options) for _, options in CATEGORICAL]}


def model_spec(codec=None, conditioning="concat"):
    if conditioning not in {"concat", "cross_attention"}:
        raise ValueError("unsupported performance conditioning")
    spec = dict(MODEL_SPEC)
    if codec is not None and codec.generator != "independent_authored_pilot_v1":
        spec.update(model="alpha_stereo_cnn_pooled_pitch_v1", continuous_heads=len(codec.continuous),
                    categorical_sizes=[len(options) for _, options in codec.categorical])
        if getattr(codec, "balanced_loss", False):
            spec.update(model=f"{getattr(codec, 'model_family', 'full_scalar')}_stereo_cnn_pooled_pitch_v1", initialization="template_head_bias_zero_output_weights_v1",
                        objective=codec.to_dict()["loss"])
    if conditioning == "cross_attention":
        spec.update(model=spec["model"].replace("pooled_pitch", "pitch_cross_attention"), conditioning="cross_attention_v1",
                    conditioning_heads=4, conditioning_width=64, conditioning_performance_bins=8, conditioning_initial_gain=0.)
    return spec


def conditioning_from_spec(spec):
    if "conditioning" not in spec:
        return "concat"
    if spec["conditioning"] != "cross_attention_v1":
        raise ValueError("unsupported saved performance conditioning")
    return "cross_attention"


class PilotInverse(nn.Module):
    def __init__(self, codec=None, conditioning="concat"):
        super().__init__()
        spec = model_spec(codec, conditioning)
        self.conditioning = conditioning
        self.timbre = nn.Sequential(nn.Conv2d(2, 16, 5, stride=2, padding=2), nn.GELU(),
                                    nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GELU(),
                                    nn.AdaptiveAvgPool2d((6, 4)), nn.Flatten())
        self.pitch = nn.Sequential(nn.Linear(PITCH_SIZE, 64), nn.GELU())
        self.fusion = nn.Sequential(nn.Linear(32 * 6 * 4 + 64, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU())
        self.continuous = nn.Linear(128, spec["continuous_heads"])
        self.categorical = nn.ModuleList(nn.Linear(128, size) for size in spec["categorical_sizes"])
        if getattr(codec, "balanced_loss", False):
            defaults = codec.default_target()
            with torch.no_grad():
                self.continuous.weight.zero_()
                self.continuous.bias.copy_(torch.tensor(defaults.continuous).clamp(1e-4, 1 - 1e-4).logit())
                for head, index in zip(self.categorical, defaults.categorical, strict=True):
                    head.weight.zero_()
                    head.bias.zero_()
                    head.bias[index] = 5.
        if conditioning == "cross_attention":
            # Construct extras last so shared weights have exactly the baseline's seeded initialization.
            self.performance_projection = nn.Sequential(nn.Linear(440, 64), nn.GELU())
            self.timbre_projection = nn.Linear(32 * 6, 64)
            self.conditioning_attention = nn.MultiheadAttention(64, 4, dropout=0., batch_first=True)
            self.conditioning_projection = nn.Linear(64, 32 * 6)
            self.performance_position = nn.Parameter(torch.randn(8, 64) * .02)
            self.timbre_position = nn.Parameter(torch.randn(4, 64) * .02)
            self.conditioning_gain = nn.Parameter(torch.zeros(()))

    def forward(self, features: dict) -> dict:
        timbre = self.timbre(features["timbre"])
        if self.conditioning == "cross_attention":
            parts = features["pitch"][:, :-1].split((88 * 8, 88 * 8, 264 * 8), dim=1)
            performance = torch.cat([part.reshape(len(timbre), -1, 8).transpose(1, 2) for part in parts], dim=2)
            keys = self.performance_projection(performance) + self.performance_position
            queries = self.timbre_projection(timbre.reshape(len(timbre), 32 * 6, 4).transpose(1, 2)) + self.timbre_position
            conditioned, _ = self.conditioning_attention(queries, keys, keys, need_weights=False)
            residual = self.conditioning_projection(conditioned).transpose(1, 2).flatten(1)
            timbre = timbre + self.conditioning_gain * residual
        hidden = self.fusion(torch.cat((timbre, self.pitch(features["pitch"])), dim=1))
        return {"continuous": self.continuous(hidden).sigmoid(), "categorical": [head(hidden) for head in self.categorical],
                "representation": hidden}


def batch_features(items: list[dict], device: str = "cpu") -> dict:
    return {name: torch.stack([item["features"][name] for item in items]).to(device) for name in ("timbre", "pitch")}


def loss_terms(prediction: dict, targets: list[PilotTarget], codec: PilotCodec) -> dict:
    if getattr(codec, "balanced_loss", False):
        return _balanced_loss(prediction, targets, codec)
    device = prediction["continuous"].device
    values = torch.tensor([target.continuous for target in targets], dtype=torch.float32, device=device)
    mask = torch.tensor([codec.continuous_mask(target) for target in targets], dtype=torch.float32, device=device)
    categories = torch.tensor([target.categorical for target in targets], dtype=torch.int64, device=device)
    error = prediction["continuous"] - values
    continuous = (error.square() * mask).sum() / mask.sum()
    if hasattr(codec, "categorical_mask"):
        cat_mask = torch.tensor([codec.categorical_mask(target) for target in targets], dtype=torch.float32, device=device)
        categorical = sum((F.cross_entropy(logits, categories[:, index], reduction="none") * cat_mask[:, index]).sum()
                          / cat_mask[:, index].sum().clamp_min(1) for index, logits in enumerate(prediction["categorical"]))
        correct = torch.stack([logits.argmax(1) == categories[:, index] for index, logits in enumerate(prediction["categorical"])], dim=1)
        accuracy = (correct * cat_mask).sum() / cat_mask.sum().clamp_min(1)
    else:
        categorical = sum(F.cross_entropy(logits, categories[:, index]) for index, logits in enumerate(prediction["categorical"]))
        accuracy = torch.stack([logits.argmax(1) == categories[:, index]
                                for index, logits in enumerate(prediction["categorical"])]).float().mean()
    return {"loss": continuous + categorical, "continuous_mse": continuous, "categorical_ce": categorical,
            "continuous_mae": (error.abs() * mask).sum() / mask.sum(), "categorical_accuracy": accuracy}


def _family_mean(errors, mask, groups):
    totals, active = [], []
    for group in sorted(set(groups)):
        indices = [index for index, value in enumerate(groups) if value == group]
        count = mask[:, indices].sum()
        totals.append((errors[:, indices] * mask[:, indices]).sum() / count.clamp_min(1))
        active.append((count > 0).to(errors.dtype))
    return torch.stack(totals).sum() / torch.stack(active).sum().clamp_min(1)


def _balanced_loss(prediction, targets, codec):
    targets = [codec.supervision_target(target) for target in targets]
    device = prediction["continuous"].device
    values = torch.tensor([target.continuous for target in targets], dtype=torch.float32, device=device)
    masks = torch.tensor([codec.continuous_mask(target) for target in targets], dtype=torch.float32, device=device)
    categories = torch.tensor([target.categorical for target in targets], dtype=torch.int64, device=device)
    cat_masks = torch.tensor([codec.categorical_mask(target) for target in targets], dtype=torch.float32, device=device)
    error = prediction["continuous"] - values
    if codec.cyclic_indices:
        error = error.clone()
        error[:, codec.cyclic_indices] = (error[:, codec.cyclic_indices] + .5).remainder(1.) - .5
    continuous = _family_mean(error.square(), masks, codec.continuous_groups)
    cat_error = torch.stack([F.cross_entropy(logits, categories[:, index], reduction="none")
                             for index, logits in enumerate(prediction["categorical"])], dim=1)
    categorical = _family_mean(cat_error, cat_masks, codec.categorical_groups)
    correct = torch.stack([logits.argmax(1) == categories[:, index] for index, logits in enumerate(prediction["categorical"])], dim=1)
    return {"loss": continuous + categorical, "continuous_mse": continuous, "categorical_ce": categorical,
            "continuous_mae": _family_mean(error.abs(), masks, codec.continuous_groups),
            "categorical_accuracy": (correct * cat_masks).sum() / cat_masks.sum().clamp_min(1)}


def predicted_targets(prediction: dict, codec: PilotCodec) -> list[PilotTarget]:
    values = prediction["continuous"].detach().cpu().tolist()
    categories = torch.stack([logits.argmax(1) for logits in prediction["categorical"]], dim=1).cpu().tolist()
    return [codec.canonical_target(PilotTarget(tuple(row), tuple(indices))) for row, indices in zip(values, categories, strict=True)]

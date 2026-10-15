from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .codec import CATEGORICAL, CONTINUOUS, PilotCodec, PilotTarget
from .features import PITCH_SIZE


MODEL_SPEC = {"model": "pilot_stereo_cnn_pooled_pitch_v1", "timbre_channels": [16, 32],
              "timbre_pool": [6, 4], "pitch_width": 64, "hidden_width": 128,
              "continuous_heads": len(CONTINUOUS), "categorical_sizes": [len(options) for _, options in CATEGORICAL]}


def model_spec(codec=None):
    if codec is None or codec.generator == "independent_authored_pilot_v1":
        return dict(MODEL_SPEC)
    return {**MODEL_SPEC, "model": "alpha_stereo_cnn_pooled_pitch_v1", "continuous_heads": len(codec.continuous),
            "categorical_sizes": [len(options) for _, options in codec.categorical]}


class PilotInverse(nn.Module):
    def __init__(self, codec=None):
        super().__init__()
        spec = model_spec(codec)
        self.timbre = nn.Sequential(nn.Conv2d(2, 16, 5, stride=2, padding=2), nn.GELU(),
                                    nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GELU(),
                                    nn.AdaptiveAvgPool2d((6, 4)), nn.Flatten())
        self.pitch = nn.Sequential(nn.Linear(PITCH_SIZE, 64), nn.GELU())
        self.fusion = nn.Sequential(nn.Linear(32 * 6 * 4 + 64, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU())
        self.continuous = nn.Linear(128, spec["continuous_heads"])
        self.categorical = nn.ModuleList(nn.Linear(128, size) for size in spec["categorical_sizes"])

    def forward(self, features: dict) -> dict:
        hidden = self.fusion(torch.cat((self.timbre(features["timbre"]), self.pitch(features["pitch"])), dim=1))
        return {"continuous": self.continuous(hidden).sigmoid(), "categorical": [head(hidden) for head in self.categorical]}


def batch_features(items: list[dict], device: str = "cpu") -> dict:
    return {name: torch.stack([item["features"][name] for item in items]).to(device) for name in ("timbre", "pitch")}


def loss_terms(prediction: dict, targets: list[PilotTarget], codec: PilotCodec) -> dict:
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


def predicted_targets(prediction: dict, codec: PilotCodec) -> list[PilotTarget]:
    values = prediction["continuous"].detach().cpu().tolist()
    categories = torch.stack([logits.argmax(1) for logits in prediction["categorical"]], dim=1).cpu().tolist()
    return [codec.canonical_target(PilotTarget(tuple(row), tuple(indices))) for row, indices in zip(values, categories, strict=True)]

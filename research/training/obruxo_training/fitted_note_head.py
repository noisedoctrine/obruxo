"""Training-only ridge readout retaining every timbre coordinate, mean and scale."""

import torch
from torch import nn

from .branch_adversary import ReverseGradient


class FittedNoteHead(nn.Module):
    def __init__(self, width=768):
        super().__init__()
        self.register_buffer("center", torch.zeros(width, dtype=torch.float64))
        self.register_buffer("scale", torch.ones(width, dtype=torch.float64))
        self.register_buffer("weights", torch.zeros(width, 2, dtype=torch.float64))
        self.register_buffer("bias", torch.zeros(2, dtype=torch.float64))
        self.register_buffer("fitted", torch.tensor(False))

    @torch.no_grad()
    def fit(self, features, targets):
        if features.ndim != 2 or features.shape[1] != len(self.center) or targets.shape != (len(features), 2) or not len(features):
            raise ValueError("fitted note head needs aligned training features and two targets")
        features, targets = features.detach().double(), targets.detach().double()
        if not torch.isfinite(features).all() or not torch.isfinite(targets).all():
            raise ValueError("fitted note head requires finite training data")
        center, scale = features.mean(0), features.std(0, unbiased=False)
        scale = torch.where(scale > 1e-8, scale, 1.)
        bias, width = targets.mean(0), features.shape[1]
        x, y = (features - center) / scale, targets - bias
        if width <= len(x):
            weights = torch.linalg.solve(x.T @ x / width + .1 * torch.eye(width, dtype=x.dtype), x.T @ y / width)
        else:
            weights = x.T @ torch.linalg.solve(x @ x.T / width + .1 * torch.eye(len(x), dtype=x.dtype), y) / width
        for name, value in (("center", center), ("scale", scale), ("weights", weights), ("bias", bias)):
            getattr(self, name).copy_(value)
        self.fitted.fill_(True)

    def normalized(self, features):
        if not self.fitted or features.ndim != 2 or features.shape[1] != len(self.center):
            raise ValueError("note head must be fitted before use on aligned features")
        return (features.double() - self.center) / self.scale

    def forward(self, features, *, coefficient):
        values = ReverseGradient.apply(self.normalized(features), coefficient)
        return values @ self.weights + self.bias

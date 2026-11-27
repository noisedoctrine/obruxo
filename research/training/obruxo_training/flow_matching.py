"""Direct and conditional-flow 2048-point Vital spectrum predictors."""

import torch
from torch import nn
from torch.nn import functional as F

from .surrogate_backbones import GRID_HOP, WIDTH, ResidualMLPBlock, VitalSurrogate
from .spectral import SAMPLE_RATE

FFT_SIZE = 2048
FREQUENCY_BINS = FFT_SIZE // 2 + 1
SPECTRUM_WIDTH = 2 * FREQUENCY_BINS
LOG_FLOOR = 1e-4
def _frame_context(backbone, controls, performance, times, lengths):
    encoded = backbone.context(controls, performance, lengths)
    location = (times * (SAMPLE_RATE / GRID_HOP)).clamp(0., encoded.shape[1] - 1)
    left = location.long()
    right = (left + 1).clamp_max(encoded.shape[1] - 1)
    before = encoded.gather(1, left[..., None].expand(-1, -1, WIDTH))
    after = encoded.gather(1, right[..., None].expand(-1, -1, WIDTH))
    return torch.lerp(before, after, (location - left)[..., None])


def log_spectrum(magnitude):
    return torch.log1p(magnitude / LOG_FLOOR)


def physical_spectrum(log_value):
    return LOG_FLOOR * torch.expm1(log_value.clamp(0., 16.))


def as_spectrogram(values):
    return values.reshape(len(values), values.shape[1], 2, FREQUENCY_BINS).permute(0, 2, 3, 1)


class _ConditionedBackbone(nn.Module):
    def __init__(self, backbone, control_width):
        super().__init__()
        self.backbone = VitalSurrogate(backbone, control_width)
        del self.backbone.heads

    def context(self, controls, performance, times, lengths):
        return _frame_context(self.backbone, controls, performance, times, lengths)


class DirectSpectrumModel(_ConditionedBackbone):
    """One-pass prediction of positive log1p magnitude values."""

    def __init__(self, backbone, control_width):
        super().__init__(backbone, control_width)
        self.output = nn.Linear(WIDTH, SPECTRUM_WIDTH)
        nn.init.normal_(self.output.weight, std=.001)
        nn.init.constant_(self.output.bias, 2.)

    def forward(self, controls, performance, times, lengths):
        hidden = self.context(controls, performance, times, lengths)
        return physical_spectrum(F.softplus(self.output(hidden)))


class FlowSpectrumModel(_ConditionedBackbone):
    """Conditional vector field over log1p stereo magnitude spectra."""

    def __init__(self, backbone, control_width):
        super().__init__(backbone, control_width)
        self.noisy_input = nn.Linear(SPECTRUM_WIDTH, WIDTH)
        self.time_input = nn.Sequential(nn.Linear(4, WIDTH), nn.SiLU(), nn.Linear(WIDTH, WIDTH))
        # Event-frame training samples irregular times. Keep denoising framewise so
        # inference on a dense time grid has the same temporal semantics.
        self.denoise_blocks = nn.ModuleList(ResidualMLPBlock() for _ in range(2))
        self.denoise_norm = nn.LayerNorm(WIDTH)
        self.clean_endpoint = nn.Linear(WIDTH, SPECTRUM_WIDTH)
        nn.init.normal_(self.clean_endpoint.weight, std=.001)
        nn.init.constant_(self.clean_endpoint.bias, 2.)

    def velocity_from_context(self, context, noisy, time, frame_mask, initial_noise):
        features = torch.stack((time, torch.sin(torch.pi * time), torch.sin(2 * torch.pi * time),
                                torch.cos(torch.pi * time)), dim=-1)
        hidden = context + self.noisy_input(noisy) + self.time_input(features)[:, None]
        padding = ~frame_mask
        for block in self.denoise_blocks:
            hidden = block(hidden, padding)
        # Predict the clean endpoint, then subtract the full-dimensional base noise.
        # This analytically derived velocity retains sample-dependent noise removal
        # in every spectral dimension despite the compact endpoint projection.
        return self.clean_endpoint(self.denoise_norm(hidden)) - initial_noise

    @torch.no_grad()
    def sample(self, controls, performance, times, lengths, frame_mask, noise, steps):
        context = self.context(controls, performance, times, lengths)
        return self.sample_from_context(context, frame_mask, noise, steps)

    @torch.no_grad()
    def sample_from_context(self, context, frame_mask, noise, steps):
        if steps not in {4, 8, 16}:
            raise ValueError("the frozen Euler sampler uses 4, 8 or 16 steps")
        value, initial_noise = noise, noise
        for index in range(steps):
            time = torch.full((len(value),), index / steps, dtype=value.dtype, device=value.device)
            value = value + self.velocity_from_context(context, value, time, frame_mask, initial_noise) / steps
        return physical_spectrum(value)


def one_resolution_spectral_loss(predicted, expected, frame_mask):
    """The existing spectral-convergence plus log-magnitude MAE objective at FFT=2048."""
    valid = frame_mask[:, None, None].expand_as(expected)
    count = valid.sum((1, 2, 3)).clamp_min(1)
    difference = ((predicted - expected) * valid).flatten(1)
    target = (expected * valid).flatten(1)
    denominator = torch.linalg.vector_norm(target, dim=1).clamp_min(.001 * count.sqrt())
    convergence = torch.linalg.vector_norm(difference, dim=1) / denominator
    log_error = ((torch.log1p(predicted / LOG_FLOOR) - torch.log1p(expected / LOG_FLOOR)).abs()
                 * valid).sum((1, 2, 3)) / count
    return {"loss": .5 * (convergence + log_error), "spectral_convergence": convergence,
            "log_magnitude_mae": log_error}

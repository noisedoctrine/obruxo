"""Versioned, unpooled stereo targets and a sound-only reconstruction objective."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from obruxo_data.hashing import canonical_sha256


SAMPLE_RATE = 44100
FFT_SIZES = (512, 2048, 8192)
MAX_SAMPLES = 16 * SAMPLE_RATE
ENVELOPE_WINDOW, ENVELOPE_HOP = 1024, 256


def spectral_spec():
    return {"format": "basic_temporal_spectra_v1", "sample_rate": SAMPLE_RATE, "maximum_samples": MAX_SAMPLES,
            "fft_sizes": list(FFT_SIZES), "hop": "fft/4", "window": "periodic_hann", "center": True,
            "padding": "constant_zero", "magnitude": "abs_stft/sqrt_fft", "channels": ["left", "right"],
            "valid_frame": "center_sample_less_than_length", "pooling": None, "dtype": "float32",
            "envelope": {"window": ENVELOPE_WINDOW, "hop": ENVELOPE_HOP, "center": True,
                         "channels": ["left", "right", "(left-right)/2"], "rms_floor": 1e-7},
            "objective": {"spectral": "mean_resolution(0.5*(convergence+mean_log1p_l1))",
                          "convergence_rms_floor": .001, "log_floor": .0001,
                          "envelope_weight": .1, "side_weight": .1, "gain_normalization": None}}


class SpectralTargets(nn.Module):
    """Measure real audio; gradients for inverse training enter predicted measurements instead."""

    def __init__(self):
        super().__init__()
        for size in FFT_SIZES:
            self.register_buffer(f"window_{size}", torch.hann_window(size))

    @torch.no_grad()
    def forward(self, audio, lengths=None):
        if (audio.dtype != torch.float32 or audio.ndim != 3 or audio.shape[1] != 2
                or not 1 <= audio.shape[-1] <= MAX_SAMPLES or not torch.isfinite(audio).all()):
            raise ValueError("spectral targets require finite float32 [batch, 2, samples] up to 16 seconds")
        batch, _, samples = audio.shape
        if batch < 1:
            raise ValueError("empty spectral batch")
        if lengths is None:
            lengths = torch.full((batch,), samples, dtype=torch.int64, device=audio.device)
        if (lengths.shape != (batch,) or lengths.dtype != torch.int64 or lengths.device != audio.device
                or (lengths < 1).any() or (lengths > samples).any()):
            raise ValueError("invalid spectral target lengths")
        audio = audio * (torch.arange(samples, device=audio.device)[None] < lengths[:, None])[:, None]
        magnitudes, masks = [], []
        for size in FFT_SIZES:
            window = getattr(self, f"window_{size}")
            spectrum = torch.stft(audio.reshape(batch * 2, samples), size, hop_length=size // 4, window=window,
                                  center=True, pad_mode="constant", return_complex=True).abs() / math.sqrt(size)
            magnitudes.append(spectrum.reshape(batch, 2, size // 2 + 1, -1))
            masks.append(torch.arange(spectrum.shape[-1], device=audio.device)[None] * (size // 4) < lengths[:, None])
        channels = torch.cat((audio, (audio[:, :1] - audio[:, 1:2]) / 2), dim=1)
        padded = F.pad(channels.square(), (ENVELOPE_WINDOW // 2, ENVELOPE_WINDOW // 2))
        energy = F.avg_pool1d(padded, ENVELOPE_WINDOW, stride=ENVELOPE_HOP)
        envelope = (energy + 1e-14).sqrt() - 1e-7
        envelope_mask = torch.arange(envelope.shape[-1], device=audio.device)[None] * ENVELOPE_HOP < lengths[:, None]
        return {"magnitudes": tuple(magnitudes), "masks": tuple(masks), "envelope": envelope,
                "envelope_mask": envelope_mask, "lengths": lengths.clone()}


def _masked_log_error(predicted, expected, mask):
    error = (torch.log1p(predicted / .0001) - torch.log1p(expected / .0001)).abs()
    weights = mask.expand_as(error)
    dims = tuple(range(1, error.ndim))
    return (error * weights).sum(dims) / weights.sum(dims).clamp_min(1)


def spectral_loss(prediction, target):
    """Return one value per example; predictions use the target's frame layout and masks."""
    if len(prediction["magnitudes"]) != len(FFT_SIZES) or len(target["magnitudes"]) != len(FFT_SIZES):
        raise ValueError("spectral resolution count changed")
    losses = []
    for predicted, expected, mask in zip(prediction["magnitudes"], target["magnitudes"], target["masks"], strict=True):
        _validate_prediction(predicted, expected)
        weight = mask[:, None, None].expand_as(expected)
        difference = ((predicted - expected) * weight).flatten(1)
        norm = torch.linalg.vector_norm((expected * weight).flatten(1), dim=1)
        floor = .001 * weight.sum((1, 2, 3)).sqrt()
        convergence = torch.linalg.vector_norm(difference, dim=1) / norm.clamp_min(floor)
        losses.append(.5 * (convergence + _masked_log_error(predicted, expected, weight)))
    predicted, expected = prediction["envelope"], target["envelope"]
    _validate_prediction(predicted, expected)
    mask = target["envelope_mask"][:, None]
    envelope = _masked_log_error(predicted[:, :2], expected[:, :2], mask)
    side = _masked_log_error(predicted[:, 2:], expected[:, 2:], mask)
    spectral = torch.stack(losses).mean(0)
    return {"loss": spectral + .1 * envelope + .1 * side, "spectral": spectral, "envelope": envelope, "side": side}


def _validate_prediction(predicted, expected):
    if (predicted.shape != expected.shape or predicted.dtype != expected.dtype or predicted.device != expected.device
            or not torch.isfinite(predicted).all() or (predicted < 0).any()):
        raise ValueError("predicted measurements must match target shape/device/dtype and be finite nonnegative")


def target_bytes(target):
    tensors = [*target["magnitudes"], *target["masks"], target["envelope"], target["envelope_mask"], target["lengths"]]
    return sum(value.numel() * value.element_size() for value in tensors)


def require_spectral_contract(saved, current):
    if canonical_sha256(saved) != canonical_sha256(current):
        raise ValueError("spectral training contract changed; start a new run")

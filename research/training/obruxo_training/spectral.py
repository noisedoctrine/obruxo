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


def _log_reference(expected, mask):
    dims = tuple(range(1, expected.ndim))
    return {"log": torch.log1p(expected / .0001), "count": mask.expand_as(expected).sum(dims).clamp_min(1)}


@torch.no_grad()
def prepare_spectral_target(target):
    """Cache reference-only arithmetic for repeated losses; returned tensors must remain unchanged."""
    tensors = [*target["magnitudes"], target["envelope"]]
    if any(value.requires_grad for value in tensors):
        raise ValueError("prepared spectral references must be fixed measurements")
    resolutions = []
    for expected, mask in zip(target["magnitudes"], target["masks"], strict=True):
        weight = mask[:, None, None].expand_as(expected)
        norm = torch.linalg.vector_norm((expected * weight).flatten(1), dim=1)
        floor = .001 * weight.sum((1, 2, 3)).sqrt()
        resolutions.append({**_log_reference(expected, weight), "norm": norm.clamp_min(floor)})
    mask = target["envelope_mask"][:, None]
    reference = {"magnitudes": tuple(resolutions), "envelope": _log_reference(target["envelope"][:, :2], mask),
                 "side": _log_reference(target["envelope"][:, 2:], mask)}
    return {**target, "_loss_reference": reference}


def _masked_log_error(predicted, expected, mask, reference=None):
    expected_log = torch.log1p(expected / .0001) if reference is None else reference["log"]
    error = (torch.log1p(predicted / .0001) - expected_log).abs()
    weights = mask.expand_as(error)
    dims = tuple(range(1, error.ndim))
    count = weights.sum(dims).clamp_min(1) if reference is None else reference["count"]
    return (error * weights).sum(dims) / count


def spectral_loss(prediction, target):
    """Return one value per example; predictions use the target's frame layout and masks."""
    if len(prediction["magnitudes"]) != len(FFT_SIZES) or len(target["magnitudes"]) != len(FFT_SIZES):
        raise ValueError("spectral resolution count changed")
    losses = []
    reference = target.get("_loss_reference")
    for index, (predicted, expected, mask) in enumerate(zip(prediction["magnitudes"], target["magnitudes"], target["masks"], strict=True)):
        _validate_prediction(predicted, expected)
        weight = mask[:, None, None].expand_as(expected)
        difference = ((predicted - expected) * weight).flatten(1)
        cached = None if reference is None else reference["magnitudes"][index]
        if cached is None:
            norm = torch.linalg.vector_norm((expected * weight).flatten(1), dim=1)
            floor = .001 * weight.sum((1, 2, 3)).sqrt()
            denominator = norm.clamp_min(floor)
        else:
            denominator = cached["norm"]
        convergence = torch.linalg.vector_norm(difference, dim=1) / denominator
        losses.append(.5 * (convergence + _masked_log_error(predicted, expected, weight, cached)))
    predicted, expected = prediction["envelope"], target["envelope"]
    _validate_prediction(predicted, expected)
    mask = target["envelope_mask"][:, None]
    envelope = _masked_log_error(predicted[:, :2], expected[:, :2], mask, None if reference is None else reference["envelope"])
    side = _masked_log_error(predicted[:, 2:], expected[:, 2:], mask, None if reference is None else reference["side"])
    spectral = torch.stack(losses).mean(0)
    return {"loss": spectral + .1 * envelope + .1 * side, "spectral": spectral, "envelope": envelope, "side": side}


def _validate_prediction(predicted, expected):
    if (predicted.shape != expected.shape or predicted.dtype != expected.dtype or predicted.device != expected.device
            or not torch.isfinite(predicted).all() or (predicted < 0).any()):
        raise ValueError("predicted measurements must match target shape/device/dtype and be finite nonnegative")


def target_bytes(target):
    tensors = [*target["magnitudes"], *target["masks"], target["envelope"], target["envelope_mask"], target["lengths"]]
    if "_loss_reference" in target:
        reference = target["_loss_reference"]
        tensors.extend(value for row in (*reference["magnitudes"], reference["envelope"], reference["side"]) for value in row.values())
    return sum(value.numel() * value.element_size() for value in tensors)


def concatenate_targets(targets):
    """Combine equal-frame references, including prepared reference-only arithmetic."""
    def combine(values):
        first = values[0]
        if isinstance(first, torch.Tensor):
            return torch.cat(values)
        if isinstance(first, tuple):
            return tuple(combine([value[index] for value in values]) for index in range(len(first)))
        return {key: combine([value[key] for value in values]) for key in first}
    return combine(targets)


def require_spectral_contract(saved, current):
    if canonical_sha256(saved) != canonical_sha256(current):
        raise ValueError("spectral training contract changed; start a new run")

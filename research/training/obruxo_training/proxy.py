"""A small performance-conditioned forward model; Vital remains the final judge."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from obruxo_data.midi import TempoMap

from .codec import PilotTarget
from .performances import CAPABILITIES


SAMPLE_RATE = 44100
HORIZON = 16
RESOLUTIONS = (512, 2048, 8192)
BANDS, SPECTRAL_BINS, ENVELOPE_BINS = 24, 8, 32
SPECTRAL_SIZE = 2 * BANDS * SPECTRAL_BINS
GROUP_SIZES = (SPECTRAL_SIZE,) * 3 + (2 * ENVELOPE_BINS, ENVELOPE_BINS)
FEATURE_SIZE = sum(GROUP_SIZES)
PERFORMANCE_SIZE = 16 * 5 + 3


def feature_spec():
    return {"features": "fixed_multiscale_stereo_envelope_v1", "sample_rate": SAMPLE_RATE, "maximum_seconds": HORIZON,
            "fft": list(RESOLUTIONS), "hop": "fft/4", "bands": BANDS, "spectral_time_bins": SPECTRAL_BINS,
            "envelope_time_bins": ENVELOPE_BINS, "alignment": "absolute_time_zero_pad_to_16_seconds",
            "spectral_scale": "log1p(100*unit_area_mel_magnitude/sqrt(fft))",
            "envelope_scale": "log1p(100*rms)", "stereo": "side=(left-right)/2_rms",
            "group_sizes": list(GROUP_SIZES), "size": FEATURE_SIZE, "fitted_statistics": False}


class SoundFeatures(nn.Module):
    """Fixed measurements, without a pretrained model or fitted normalization."""

    def __init__(self):
        super().__init__()
        edges = 700 * (10 ** (torch.linspace(0, 2595 * math.log10(1 + 22050 / 700), BANDS + 2) / 2595) - 1)
        for size in RESOLUTIONS:
            frequencies = torch.linspace(0, SAMPLE_RATE / 2, size // 2 + 1)[None]
            rising = (frequencies - edges[:-2, None]) / (edges[1:-1, None] - edges[:-2, None])
            falling = (edges[2:, None] - frequencies) / (edges[2:, None] - edges[1:-1, None])
            bands = torch.minimum(rising, falling).clamp_min(0)
            # Small FFTs have no sample in some low-frequency bands. Keep those bands explicitly zero.
            self.register_buffer(f"bands_{size}", bands / bands.sum(1, keepdim=True).clamp_min(1e-12))
            self.register_buffer(f"window_{size}", torch.hann_window(size))

    @torch.no_grad()
    def forward(self, audio: np.ndarray, sample_rate=SAMPLE_RATE):
        if (sample_rate != SAMPLE_RATE or audio.dtype != np.float32 or audio.ndim != 2 or audio.shape[1] != 2
                or not 8192 <= len(audio) <= HORIZON * SAMPLE_RATE or not np.isfinite(audio).all()):
            raise ValueError("sound features require finite 44.1 kHz stereo float32 audio, 8192 samples to 16 seconds")
        stereo = F.pad(torch.from_numpy(audio.copy()).T, (0, HORIZON * SAMPLE_RATE - len(audio)))
        features = []
        for size in RESOLUTIONS:
            spectrum = torch.stft(stereo, size, hop_length=size // 4, window=getattr(self, f"window_{size}"),
                                  return_complex=True).abs() / math.sqrt(size)
            mel = torch.log1p(100 * (getattr(self, f"bands_{size}") @ spectrum))
            features.append(F.adaptive_avg_pool1d(mel, SPECTRAL_BINS).flatten())
        envelope = F.adaptive_avg_pool1d(stereo.square(), ENVELOPE_BINS).sqrt()
        side = F.adaptive_avg_pool1d(((stereo[0] - stereo[1]) / 2).square()[None], ENVELOPE_BINS).sqrt()
        result = torch.cat([*features, torch.log1p(100 * envelope).flatten(), torch.log1p(100 * side).flatten()])
        validate_sound_features(result)
        return result


def validate_sound_features(value):
    if value.shape != (FEATURE_SIZE,) or value.dtype != torch.float32 or not torch.isfinite(value).all() or (value < 0).any():
        raise ValueError("invalid fixed sound features")


def performance_inputs(performance, tail_seconds=2.):
    """Actual notes and timing, never a performance ID or learned bank index."""
    performance.validate(CAPABILITIES).require_valid()
    timing = TempoMap.from_performance(performance)
    notes = sorted(performance.note_spans(), key=lambda note: (note.start_tick, note.pitch, note.end_tick, note.velocity))
    duration = float(timing.tick_to_seconds(performance.end_tick))
    if not 1 <= len(notes) <= 16 or not math.isfinite(tail_seconds) or tail_seconds < 0 or duration + tail_seconds > HORIZON:
        raise ValueError("proxy performance needs 1..16 notes and a total duration within 16 seconds")
    values = []
    for note in notes:
        start, end = [float(timing.tick_to_seconds(tick)) for tick in (note.start_tick, note.end_tick)]
        values.extend((note.pitch / 127, note.velocity / 127, start / HORIZON, (end - start) / HORIZON, 1.))
    values.extend([0.] * (5 * (16 - len(notes))))
    tempo = 60_000_000 / timing.segments[0].microseconds_per_beat
    return torch.tensor([*values, tempo / 240, duration / HORIZON, tail_seconds / HORIZON], dtype=torch.float32)


def patch_input_size(codec):
    return 2 * len(codec.continuous) + sum(len(options) for _, options in codec.categorical) + len(codec.categorical)


def patch_inputs(continuous, categorical, codec):
    """Hard categories and canonical inactive fields; gradients reach active continuous values."""
    if (continuous.ndim != 2 or continuous.shape[1] != len(codec.continuous) or continuous.dtype != torch.float32 or categorical.ndim != 2
            or categorical.shape != (len(continuous), len(codec.categorical)) or categorical.dtype != torch.int64
            or not torch.isfinite(continuous).all() or ((continuous < 0) | (continuous > 1)).any()):
        raise ValueError("invalid proxy patch tensors")
    targets = [codec.canonical_target(PilotTarget((.5,) * len(codec.continuous), tuple(row))) for row in categorical.cpu().tolist()]
    canonical = torch.tensor([target.categorical for target in targets], dtype=torch.int64, device=continuous.device)
    mask = torch.tensor([codec.continuous_mask(target) for target in targets], dtype=torch.bool, device=continuous.device)
    cat_mask = torch.tensor([codec.categorical_mask(target) if hasattr(codec, "categorical_mask") else [True] * len(codec.categorical)
                             for target in targets], dtype=torch.float32, device=continuous.device)
    defaults = continuous.new_tensor([(default - lo) / (hi - lo) for _, lo, hi, default in codec.continuous])
    values = torch.where(mask, continuous, defaults)
    one_hot = [F.one_hot(canonical[:, index], len(options)).to(continuous.dtype) for index, (_, options) in enumerate(codec.categorical)]
    return torch.cat([values, mask.to(continuous.dtype), *one_hot, cat_mask], dim=1)


def target_inputs(targets, codec):
    for target in targets:
        codec.validate_target(target)
    return patch_inputs(torch.tensor([target.continuous for target in targets], dtype=torch.float32),
                        torch.tensor([target.categorical for target in targets], dtype=torch.int64), codec)


def feature_distance(predicted, target, scale):
    """Equal group weights; scales come only from accepted training features."""
    errors = ((predicted - target) / scale).square().split(GROUP_SIZES, dim=-1)
    return torch.stack([error.mean(dim=-1) for error in errors], dim=-1).mean(dim=-1)


class SoundProxy(nn.Module):
    def __init__(self, codec, mean=None, scale=None):
        super().__init__()
        self.codec = codec
        self.register_buffer("mean", torch.zeros(FEATURE_SIZE) if mean is None else mean.clone())
        self.register_buffer("scale", torch.ones(FEATURE_SIZE) if scale is None else scale.clone())
        self.network = nn.Sequential(nn.Linear(patch_input_size(codec) + PERFORMANCE_SIZE, 256), nn.SiLU(),
                                     nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, FEATURE_SIZE))
        nn.init.normal_(self.network[-1].weight, std=.001)
        with torch.no_grad():
            self.network[-1].bias.copy_(torch.log(torch.expm1(self.mean.clamp(.001, 20))))

    def forward(self, patch, performance):
        return F.softplus(self.network(torch.cat((patch, performance), dim=-1)))

    def predict_patch(self, continuous, categorical, performance):
        return self(patch_inputs(continuous, categorical, self.codec), performance)

    def freeze(self):
        self.eval().requires_grad_(False)
        self.zero_grad(set_to_none=True)
        return self


def refine_continuous(proxy, target, performance, desired_features, *, steps=16, learning_rate=.03, radius=.15):
    """Bounded qualification probe. No source parameter labels enter the update."""
    if proxy.training or any(parameter.requires_grad for parameter in proxy.parameters()):
        raise ValueError("freeze the proxy before using its input gradients")
    if type(steps) is not int or steps < 1 or not 0 < learning_rate <= .1 or not 0 < radius <= .25:
        raise ValueError("invalid bounded proxy update")
    start = torch.tensor([target.continuous], dtype=torch.float32)
    continuous = start.clone().requires_grad_()
    categorical = torch.tensor([target.categorical], dtype=torch.int64)
    optimizer = torch.optim.Adam([continuous], lr=learning_rate)
    initial, final = None, None
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = feature_distance(proxy.predict_patch(continuous, categorical, performance[None]), desired_features[None], proxy.scale).mean()
        if not torch.isfinite(loss):
            raise ValueError("nonfinite proxy update")
        if initial is None:
            initial = float(loss.detach())
        loss.backward()
        if continuous.grad is None or not torch.isfinite(continuous.grad).all():
            raise ValueError("missing or nonfinite proxy input gradient")
        optimizer.step()
        with torch.no_grad():
            continuous.copy_(torch.maximum(torch.minimum(continuous, start + radius), start - radius).clamp(0, 1))
    with torch.no_grad():
        final = float(feature_distance(proxy.predict_patch(continuous, categorical, performance[None]), desired_features[None], proxy.scale).mean())
    result = proxy.codec.canonical_target(PilotTarget(tuple(continuous.detach()[0].tolist()), target.categorical))
    return result, {"initial_proxy_error": initial, "final_proxy_error": final, "steps": steps,
                    "maximum_continuous_change": float((continuous.detach() - start).abs().max())}


def load_proxy(path, *, require_qualified=True):
    from .codec import load_codec
    from .features import tensor_digest
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if (saved.get("format") != "obruxo_sound_proxy_v1" or saved["features"] != feature_spec()
            or tensor_digest(saved["model"]) != saved["model_sha256"]):
        raise ValueError("incompatible or changed sound proxy artifact")
    if require_qualified and not saved["qualified_for_inverse_training"]:
        raise ValueError("sound proxy has not passed native qualification")
    model = SoundProxy(load_codec(saved["codec"]))
    model.load_state_dict(saved["model"], strict=True)
    return model.freeze()

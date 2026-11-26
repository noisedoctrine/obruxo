"""Three interchangeable backbones with shared note conditioning and stereo heads."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .spectral import FFT_SIZES, SAMPLE_RATE

BACKBONES = ("transformer", "conformer", "residual_mlp")
GRID_HOP = 1024
WIDTH = 64


def feedforward(hidden):
    return nn.Sequential(nn.LayerNorm(WIDTH), nn.Linear(WIDTH, hidden), nn.GELU(), nn.Linear(hidden, WIDTH))


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(WIDTH)
        self.attention = nn.MultiheadAttention(WIDTH, 4, batch_first=True)

    def forward(self, value, padding):
        normalized = self.norm(value)
        return self.attention(normalized, normalized, normalized, key_padding_mask=padding, need_weights=False)[0]


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = Attention()
        self.ff = feedforward(192)

    def forward(self, value, padding):
        value = value + self.attention(value, padding)
        return value + self.ff(value)


class ConformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.before = feedforward(48)
        self.attention = Attention()
        self.conv = nn.Sequential(nn.Conv1d(WIDTH, 2 * WIDTH, 1), nn.GLU(dim=1),
                                  nn.Conv1d(WIDTH, WIDTH, 7, padding=3, groups=WIDTH), nn.SiLU(), nn.Conv1d(WIDTH, WIDTH, 1))
        self.norm = nn.LayerNorm(WIDTH)
        self.after = feedforward(48)

    def forward(self, value, padding):
        value = value + .5 * self.before(value)
        value = value + self.attention(value, padding)
        local = self.norm(value).masked_fill(padding[..., None], 0.)
        value = value + self.conv(local.transpose(1, 2)).transpose(1, 2)
        return value + .5 * self.after(value)


class ResidualMLPBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ff = feedforward(320)

    def forward(self, value, _padding):
        return value + self.ff(value)


class VitalSurrogate(nn.Module):
    def __init__(self, backbone, control_width):
        super().__init__()
        block = dict(zip(BACKBONES, (TransformerBlock, ConformerBlock, ResidualMLPBlock), strict=True))[backbone]
        self.notes = nn.Sequential(nn.Linear(7, 24), nn.SiLU(), nn.Linear(24, 24), nn.SiLU())
        self.input = nn.Linear(control_width + 24 + 3 + 5, WIDTH)
        self.blocks = nn.ModuleList(block() for _ in range(2))
        self.norm = nn.LayerNorm(WIDTH)
        self.heads = nn.ModuleList(nn.Linear(WIDTH, width) for width in (*[2 * (n // 2 + 1) for n in FFT_SIZES], 3))
        for head in self.heads:
            nn.init.normal_(head.weight, std=.001)
            nn.init.constant_(head.bias, -2.)

    def context(self, controls, performance, lengths):
        time = torch.arange(1 + math.ceil(int(lengths.max()) / GRID_HOP), device=controls.device) * (GRID_HOP / SAMPLE_RATE)
        notes = performance[:, :80].reshape(-1, 1, 16, 5)
        pitch, velocity, onset, duration, available = notes.unbind(-1)
        elapsed = time[None, :, None] - onset * 16
        released = elapsed - duration * 16
        visible = (elapsed >= -.1) & available.bool()
        active = ((elapsed >= 0) & (released < 0) & visible).float()
        features = torch.stack((pitch.expand_as(elapsed), velocity.expand_as(elapsed), duration.expand_as(elapsed),
                                elapsed.clamp(-1, 4), released.clamp(-1, 4), active, visible.float()), dim=-1)
        note_context = (self.notes(features) * visible[..., None]).sum(2)
        position = torch.stack((time / 16, time.sin(), time.cos(), (4 * time).sin(), (4 * time).cos()), dim=-1)
        fixed = torch.cat((controls, performance[:, 80:]), dim=1)
        value = self.input(torch.cat((fixed[:, None].expand(-1, time.numel(), -1), note_context,
                                      position[None].expand(len(controls), -1, -1)), dim=-1))
        padding = time[None] * SAMPLE_RATE > lengths[:, None] + GRID_HOP
        for block in self.blocks:
            value = block(value, padding)
        return self.norm(value)

    def forward(self, controls, performance, times, lengths):
        encoded = self.context(controls, performance, lengths)
        outputs = []
        for index, (head, time) in enumerate(zip(self.heads, times, strict=True)):
            location = (time * (SAMPLE_RATE / GRID_HOP)).clamp(0., encoded.shape[1] - 1)
            left = location.long()
            right = (left + 1).clamp_max(encoded.shape[1] - 1)
            before = encoded.gather(1, left[..., None].expand(-1, -1, WIDTH))
            after = encoded.gather(1, right[..., None].expand(-1, -1, WIDTH))
            feature = torch.lerp(before, after, (location - left)[..., None])
            linear = .0001 * torch.expm1(F.softplus(head(feature)).clamp_max(16))
            if index < len(FFT_SIZES):
                outputs.append(linear.reshape(len(linear), linear.shape[1], 2, -1).permute(0, 2, 3, 1))
            else:
                outputs.append(linear.transpose(1, 2))
        return {"magnitudes": tuple(outputs[:3]), "envelope": outputs[3]}

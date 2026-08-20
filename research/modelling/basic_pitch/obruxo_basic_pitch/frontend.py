"""The fixed CQT, normalized-log, and harmonic-stacking frontend."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .constants import (
    AUDIO_N_SAMPLES,
    CQT_INPUT_REFLECTION,
    CQT_KERNEL_LENGTH,
    CQT_N_BINS,
    CQT_N_FILTERS,
    CQT_TOP_HOP,
    DOWNSAMPLE_FILTER_LENGTH,
    DOWNSAMPLE_REFLECTION,
    FRONTEND_BATCHNORM_EPS,
    HARMONIC_OUTPUT_BINS,
    HARMONIC_SHIFTS,
)


def normalized_log(magnitude: Tensor) -> Tensor:
    """Apply the released normalized-log operation, including zero division."""
    power = magnitude * magnitude
    log_magnitude = torch.log(power + 1e-10) * 0.43429446 * 10.0
    minimum = torch.amin(log_magnitude, dim=(1, 2), keepdim=True)
    shifted = log_magnitude - minimum
    maximum = torch.amax(shifted, dim=(1, 2), keepdim=True)
    return torch.where(maximum == 0, torch.zeros_like(shifted), shifted / maximum)


def harmonic_stack(magnitude: Tensor) -> Tensor:
    """Apply the exact zero-padded frequency shifts used by the graph."""
    channels = []
    for shift in HARMONIC_SHIFTS:
        if shift < 0:
            shifted = F.pad(magnitude[:, :, :shift], (-shift, 0))
        elif shift > 0:
            shifted = F.pad(magnitude[:, :, shift:], (0, shift))
        else:
            shifted = magnitude
        channels.append(shifted)
    return torch.stack(channels, dim=1)[:, :, :, :HARMONIC_OUTPUT_BINS]


class BasicPitchFrontend(nn.Module):
    """Reproduce the released ONNX frontend using native torch operations.

    The released graph has already folded the learned convolutional batch
    normalizations. The explicit frontend batch normalization remains live;
    its parameters are imported from the graph's affine scale and offset.
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "cqt_kernels_real", torch.zeros(CQT_N_FILTERS, 1, CQT_KERNEL_LENGTH)
        )
        self.register_buffer(
            "cqt_kernels_imag", torch.zeros(CQT_N_FILTERS, 1, CQT_KERNEL_LENGTH)
        )
        self.register_buffer(
            "lowpass_filter", torch.zeros(1, 1, DOWNSAMPLE_FILTER_LENGTH)
        )
        self.register_buffer("cqt_lengths", torch.zeros(CQT_N_BINS))
        self.normalization = nn.BatchNorm2d(
            1, eps=FRONTEND_BATCHNORM_EPS, momentum=0.01
        )

    def _complex_cqt(self, audio: Tensor, hop: int) -> Tensor:
        padded = F.pad(
            audio, (CQT_INPUT_REFLECTION, CQT_INPUT_REFLECTION), mode="reflect"
        )
        real = F.conv1d(padded, self.cqt_kernels_real, stride=hop)
        imag = -F.conv1d(padded, self.cqt_kernels_imag, stride=hop)
        return torch.stack((real, imag), dim=-1).permute(0, 2, 1, 3)

    def _downsample_by_two(self, audio: Tensor) -> Tensor:
        padded = F.pad(
            audio, (DOWNSAMPLE_REFLECTION, DOWNSAMPLE_REFLECTION), mode="constant"
        )
        return F.conv1d(padded, self.lowpass_filter, stride=2)

    def _cqt_magnitude(self, audio: Tensor) -> Tensor:
        cqt = self._complex_cqt(audio, CQT_TOP_HOP)
        for hop in (128, 64, 32, 16, 8, 4, 2, 1):
            audio = self._downsample_by_two(audio)
            lower = self._complex_cqt(audio, hop)
            cqt = torch.cat((lower, cqt), dim=2)[:, :, -CQT_N_BINS:, :]
        cqt = cqt * self.cqt_lengths.view(1, 1, CQT_N_BINS, 1)
        return torch.sqrt(torch.sum(cqt * cqt, dim=-1))

    def forward(self, audio: Tensor) -> Tensor:
        if audio.ndim != 3 or audio.shape[1:] != (AUDIO_N_SAMPLES, 1):
            raise ValueError(
                f"expected input shape [B,{AUDIO_N_SAMPLES},1], got {tuple(audio.shape)}"
            )
        if audio.dtype is not torch.float32:
            raise TypeError(f"expected torch.float32 input, got {audio.dtype}")
        audio_nchw = audio.transpose(1, 2)
        magnitude = self._cqt_magnitude(audio_nchw)
        normalized = normalized_log(magnitude)
        normalized = self.normalization(normalized.unsqueeze(1)).squeeze(1)
        return harmonic_stack(normalized)

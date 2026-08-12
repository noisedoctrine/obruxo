"""Native PyTorch Basic Pitch ICASSP 2022 network."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .constants import (
    ANNOT_N_FRAMES,
    AUDIO_N_SAMPLES,
    N_FREQ_BINS_CONTOURS,
    N_FREQ_BINS_NOTES,
)
from .frontend import BasicPitchFrontend


def _pad(x: Tensor, top: int, bottom: int, left: int, right: int) -> Tensor:
    return F.pad(x, (left, right, top, bottom))


class BasicPitchICASSP2022(nn.Module):
    """The inference graph with trainable PyTorch modules and no runtime dispatch."""

    def __init__(self) -> None:
        super().__init__()
        self.frontend = BasicPitchFrontend()

        self.contour_conv1 = nn.Conv2d(8, 8, (3, 39))
        self.contour_bn = nn.BatchNorm2d(8, eps=1e-3, momentum=0.01)
        self.contour_conv2 = nn.Conv2d(8, 1, (5, 5))

        self.note_conv1 = nn.Conv2d(1, 32, (7, 7), stride=(1, 3))
        self.note_conv2 = nn.Conv2d(32, 1, (7, 3))

        self.onset_conv1 = nn.Conv2d(8, 32, (5, 5), stride=(1, 3))
        self.onset_bn = nn.BatchNorm2d(32, eps=1e-3, momentum=0.01)
        self.onset_conv2 = nn.Conv2d(33, 1, (3, 3))

    def forward(self, audio: Tensor) -> dict[str, Tensor]:
        if audio.ndim != 3 or audio.shape[1:] != (AUDIO_N_SAMPLES, 1):
            raise ValueError(
                f"expected input shape [B,{AUDIO_N_SAMPLES},1], got {tuple(audio.shape)}"
            )
        if audio.dtype is not torch.float32:
            raise TypeError(f"expected torch.float32 input, got {audio.dtype}")

        features = self.frontend(audio)

        contour = _pad(features, 1, 1, 19, 19)
        contour = F.relu(self.contour_bn(self.contour_conv1(contour)))
        contour = torch.sigmoid(self.contour_conv2(_pad(contour, 2, 2, 2, 2)))

        note = contour
        note = F.relu(self.note_conv1(_pad(note, 3, 3, 2, 2)))
        note = torch.sigmoid(self.note_conv2(_pad(note, 3, 3, 1, 1)))

        onset = _pad(features, 2, 2, 1, 1)
        onset = F.relu(self.onset_bn(self.onset_conv1(onset)))
        onset = torch.sigmoid(
            self.onset_conv2(_pad(torch.cat((note, onset), dim=1), 1, 1, 1, 1))
        )

        return {
            "note": note.squeeze(1),
            "onset": onset.squeeze(1),
            "contour": contour.squeeze(1),
        }


def output_shapes(batch_size: int) -> dict[str, tuple[int, int, int]]:
    """Return the fixed public output shapes for a given batch size."""
    return {
        "note": (batch_size, ANNOT_N_FRAMES, N_FREQ_BINS_NOTES),
        "onset": (batch_size, ANNOT_N_FRAMES, N_FREQ_BINS_NOTES),
        "contour": (batch_size, ANNOT_N_FRAMES, N_FREQ_BINS_CONTOURS),
    }

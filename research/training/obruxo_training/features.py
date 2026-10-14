"""Audio-only, float32 pilot frontend; pretrained weights and statistics stay frozen."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import numpy as np
from scipy import signal
import torch
from torch import nn
from torch.nn import functional as F

from obruxo_data.hashing import canonical_sha256


BASIC_PITCH_ROOT = Path(__file__).resolve().parents[2] / "modelling" / "basic_pitch"
sys.path.insert(0, str(BASIC_PITCH_ROOT))
from obruxo_basic_pitch import BasicPitchICASSP2022  # noqa: E402
from obruxo_basic_pitch.inference import _window_audio, unwrap_window_outputs  # noqa: E402

TIMBRE_SHAPE = (2, 96, 32)
PITCH_SIZE = 8 * (88 + 88 + 264) + 1


def tensor_digest(state: dict) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(f"{name}:{array.dtype}:{array.shape}:".encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def validate_features(features: dict) -> None:
    if set(features) != {"timbre", "pitch"}:
        raise ValueError("expected audio-derived timbre and pitch features only")
    for name, shape in (("timbre", TIMBRE_SHAPE), ("pitch", (PITCH_SIZE,))):
        value = features[name]
        if value.shape != shape or value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError(f"invalid {name} feature shape, dtype or values")


class AudioFeatures(nn.Module):
    """Stereo full-band spectrum plus native Basic Pitch dense posteriors, computed on CPU."""

    def __init__(self, basic_pitch_state: dict | None = None):
        super().__init__()
        self.basic_pitch = BasicPitchICASSP2022()
        if basic_pitch_state is None:
            basic_pitch_state = torch.load(BASIC_PITCH_ROOT / "artifacts/basic_pitch_icassp_2022.pt",
                                          map_location="cpu", weights_only=True)
        self.basic_pitch.load_state_dict(basic_pitch_state, strict=True)
        self.requires_grad_(False)
        self.train(False)
        self.frozen_digest = tensor_digest(self.basic_pitch.state_dict())
        self.register_buffer("window", torch.hann_window(2048), persistent=False)
        # Unit-area triangular mel bands retain both channels and extend to Nyquist.
        edges = 700 * (10 ** (torch.linspace(0, 2595 * np.log10(1 + 22050 / 700), 98) / 2595) - 1)
        frequencies = torch.linspace(0, 22050, 1025)[None, :]
        rising = (frequencies - edges[:-2, None]) / (edges[1:-1, None] - edges[:-2, None])
        falling = (edges[2:, None] - frequencies) / (edges[2:, None] - edges[1:-1, None])
        weights = torch.minimum(rising, falling).clamp_min(0)
        self.register_buffer("bands", weights / weights.sum(1, keepdim=True), persistent=False)

    def train(self, mode: bool = True):
        return super().train(False)

    def describe(self) -> dict:
        return {"frontend": "stereo_mel_and_frozen_basic_pitch_v1", "sample_rate": 44100,
                "timbre_shape": list(TIMBRE_SHAPE), "fft": 2048, "hop": 441, "log_scale": 10,
                "pitch_size": PITCH_SIZE, "pitch_time_bins": 8, "maximum_audio_seconds": 16,
                "basic_pitch_state_sha256": self.frozen_digest, "precision": "float32", "device": "cpu",
                "normalization": "fixed_scaling_no_fitted_statistics", "input": "audio_only"}

    @property
    def frontend_id(self) -> str:
        return canonical_sha256(self.describe())

    def require_frozen(self) -> None:
        if self.training or any(module.training for module in self.basic_pitch.modules()):
            raise RuntimeError("pretrained module statistics must stay frozen")
        if any(parameter.requires_grad for parameter in self.parameters()):
            raise RuntimeError("pretrained parameters must stay frozen")
        if tensor_digest(self.basic_pitch.state_dict()) != self.frozen_digest:
            raise RuntimeError("pretrained weights or buffers changed")

    @torch.no_grad()
    def forward(self, audio: np.ndarray, sample_rate: int = 44100) -> dict:
        if sample_rate != 44100 or audio.dtype != np.float32 or audio.ndim != 2 or audio.shape[1] != 2:
            raise ValueError("frontend requires float32 stereo audio at 44100 Hz")
        if not 2048 <= len(audio) <= 16 * sample_rate or not np.isfinite(audio).all():
            raise ValueError("audio must be finite and between 2048 samples and 16 seconds")
        stereo = torch.from_numpy(audio.copy()).T.contiguous()
        spectrum = torch.stft(stereo, n_fft=2048, hop_length=441, window=self.window, return_complex=True)
        mel = torch.log1p(10 * (self.bands @ (spectrum.abs() / np.sqrt(2048))))
        timbre = F.adaptive_avg_pool2d(mel, TIMBRE_SHAPE[1:]).contiguous()
        mono = signal.resample_poly(audio.mean(axis=1), 1, 2).astype(np.float32)
        windows = _window_audio(mono)
        # A bounded one-window batch also works on CPU-only Colab sessions.
        output = [self.basic_pitch(torch.from_numpy(window[None])) for window in windows]
        arrays = {name: torch.cat([item[name] for item in output]).numpy() for name in ("note", "onset", "contour")}
        posterior = unwrap_window_outputs(arrays, original_sample_count=len(mono))
        pooled = [F.adaptive_avg_pool1d(torch.from_numpy(posterior[name]).T[None], 8).flatten()
                  for name in ("note", "onset", "contour")]
        pitch = torch.cat([*pooled, torch.tensor([len(audio) / sample_rate / 16], dtype=torch.float32)])
        features = {"timbre": timbre, "pitch": pitch}
        validate_features(features)
        return features

"""Bounded full-recording features for explicit longer-observation contracts."""
import hashlib
from pathlib import Path

import numpy as np
import torch

from .features import AudioFeatures
from .observation_bank import PROFILE


class ExtendedAudioFeatures(AudioFeatures):
    def describe(self):
        spec = super().describe()
        spec.update(frontend="stereo_mel_and_frozen_basic_pitch_80s_v1", maximum_audio_seconds=80, duration_scale_seconds=16,
                    temporal_policy="entire_recording_same_fixed_output_bins_no_truncation_one_basic_pitch_window_at_a_time",
                    source_sha256_lf={name: hashlib.sha256((Path(__file__).parent / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                                      for name in ("features.py", "extended_features.py")})
        return spec

    @torch.no_grad()
    def analyze(self, audio, sample_rate=44100):
        if sample_rate != 44100 or audio.dtype != np.float32 or audio.ndim != 2 or audio.shape[1] != 2:
            raise ValueError("frontend requires float32 stereo audio at 44100 Hz")
        if not 2048 <= len(audio) <= 80 * sample_rate or not np.isfinite(audio).all():
            raise ValueError("audio must be finite and between 2048 samples and 80 seconds")
        return self._analyze_validated(audio, sample_rate)


def features_for(codec, basic_pitch_state=None):
    kind = ExtendedAudioFeatures if codec.contract.performance_profile == PROFILE else AudioFeatures
    return kind(basic_pitch_state)

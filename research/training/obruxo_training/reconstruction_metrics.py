"""Full-reference waveform metrics shared by the Issue 187 experiment stages."""

from functools import lru_cache
import math

import numpy as np
from scipy.fft import rfft
from scipy.signal import get_window, resample_poly


SAMPLE_RATE = 44_100
STFT_SIZES = (128, 256, 512, 1024, 2048, 4096)
MEL_SIZES = (1024, 2048, 4096)
CCPC_SIZES = (512, 1024, 2048, 4096)
MEL_BANDS = 64


@lru_cache(maxsize=None)
def _window(size):
    return get_window("hann", size).astype(np.float32)


@lru_cache(maxsize=None)
def _mel(size):
    from librosa.filters import mel

    # Unit-height Slaney triangles, scaled by bands / one-sided FFT bins.
    return mel(sr=SAMPLE_RATE, n_fft=size, n_mels=MEL_BANDS, norm=None) * (MEL_BANDS / (size // 2 + 1))


def _stft(audio, size):
    padded = np.pad(audio, ((0, 0), (0, 0), (size // 2, size // 2)))
    frames = np.lib.stride_tricks.sliding_window_view(padded, size, axis=-1)[..., ::size // 4, :]
    return rfft(frames * _window(size), axis=-1)


def _ccpc(spectrum):
    cross = spectrum[:, 0].astype(np.complex128) * spectrum[:, 1].conj()
    magnitude = np.abs(cross)
    # Joint geometric mean of reference/estimate cross-channel energies.
    weight = np.sqrt(magnitude[0]) * np.sqrt(magnitude[1])
    total = weight.sum(dtype=np.float64)
    if total == 0:
        return None
    phasor = np.divide(cross, magnitude, out=np.zeros_like(cross), where=magnitude > 0)
    resultant = np.sum(weight * phasor[1] * phasor[0].conj(), axis=-1, dtype=np.complex128)
    return float(np.clip(np.abs(resultant).sum() / total, 0, 1))


def _si_sdr(audio):
    reference, estimate = audio.astype(np.float64)
    reference -= reference.mean(axis=-1, keepdims=True)
    estimate -= estimate.mean(axis=-1, keepdims=True)
    energy = np.sum(reference**2, axis=-1)
    valid = (energy > 0) & (np.sum(estimate**2, axis=-1) > 0)
    if not valid.any():
        return None
    reference, estimate, energy = reference[valid], estimate[valid], energy[valid]
    target = reference * (np.sum(reference * estimate, axis=-1) / energy)[:, None]
    signal = np.sum(target**2, axis=-1)
    residual = np.sum((estimate - target)**2, axis=-1)
    # Clip numeric singularities at ±100 dB so identity stays JSON serializable.
    floor = (signal + residual) * 1e-10
    return float(np.mean(10 * np.log10(np.maximum(signal, floor) / np.maximum(residual, floor))))


def waveform_scores(reference, estimate):
    """Score aligned native 44.1 kHz float stereo arrays shaped [samples, 2]."""
    if reference.shape != estimate.shape or reference.ndim != 2 or reference.shape[1] != 2 or not len(reference):
        raise ValueError("waveforms must have equal nonempty [samples, 2] shapes")
    audio = np.stack((reference.T, estimate.T)).astype(np.float32, copy=False)
    if not np.isfinite(audio).all():
        raise ValueError("waveforms must contain finite samples")
    stft, mel, coherence = [], [], []
    for size in STFT_SIZES:
        spectrum = _stft(audio, size)
        magnitude = np.abs(spectrum)
        if size in MEL_SIZES:
            projected = np.log1p(magnitude @ _mel(size).T)
            mel.append(np.abs(projected[0] - projected[1]).mean(dtype=np.float64))
        logged = np.log1p(magnitude)
        stft.append(np.abs(logged[0] - logged[1]).mean(dtype=np.float64))
        if size in CCPC_SIZES:
            value = _ccpc(spectrum)
            if value is not None:
                coherence.append(value)
    return {"mel_log1p": float(np.mean(mel)), "stft_log1p": float(np.mean(stft)),
            "ccpc": 100 * float(np.mean(coherence)) if coherence else None, "si_sdr_db": _si_sdr(audio)}


def cosine_similarity(reference, estimate):
    left, right = np.asarray(reference, dtype=np.float64), np.asarray(estimate, dtype=np.float64)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.clip(np.dot(left, right) / denominator, -1, 1)) if denominator > 0 else None


class VisqolAudio:
    """Official ViSQOL Python binding with the audio SVR, if installed."""

    def __init__(self, model_path):
        from visqol import visqol_lib_py
        from visqol.pb2 import visqol_config_pb2

        config = visqol_config_pb2.VisqolConfig()
        config.audio.sample_rate = 48_000
        config.options.use_speech_scoring = False
        config.options.svr_model_path = str(model_path)
        self.api = visqol_lib_py.VisqolApi()
        self.api.Create(config)

    def __call__(self, reference, estimate):
        divisor = math.gcd(SAMPLE_RATE, 48_000)
        audio = [resample_poly(wave.mean(axis=1, dtype=np.float64), 48_000 // divisor, SAMPLE_RATE // divisor)
                 for wave in (reference, estimate)]
        return float(self.api.Measure(*audio).moslqo)

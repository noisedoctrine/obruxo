import numpy as np
import pytest

from obruxo_training.reconstruction_metrics import cosine_similarity, waveform_scores


def test_identity_gain_and_silence():
    wave = np.random.default_rng(187).normal(0, 0.1, (8192, 2)).astype(np.float32)
    identity = waveform_scores(wave, wave)
    assert identity == pytest.approx({"mel_log1p": 0, "stft_log1p": 0, "ccpc": 100, "si_sdr_db": 100})
    quiet = waveform_scores(wave, wave * 0.5)
    assert quiet["mel_log1p"] > 0 and quiet["stft_log1p"] > 0
    assert quiet["ccpc"] == pytest.approx(100)
    assert quiet["si_sdr_db"] == pytest.approx(100)
    quiet_tail = wave * np.float32(1e-21)
    assert waveform_scores(quiet_tail, quiet_tail)["ccpc"] == pytest.approx(100)
    blank = np.zeros_like(wave)
    assert waveform_scores(blank, blank) == {
        "mel_log1p": 0, "stft_log1p": 0, "ccpc": None, "si_sdr_db": None,
    }


def test_stereo_delay_changes_ccpc_but_common_polarity_does_not():
    rng = np.random.default_rng(187)
    left = rng.normal(0, 0.1, 8192).astype(np.float32)
    wave = np.stack((left, left), axis=1)
    delayed = wave.copy()
    delayed[:13, 1] = 0
    delayed[13:, 1] = wave[:-13, 1]
    assert waveform_scores(wave, -wave)["ccpc"] == pytest.approx(100)
    assert waveform_scores(wave, delayed)["ccpc"] < 50


def test_cosine_similarity_reports_missing_zero_vector():
    assert cosine_similarity([1, 2], [3, 6]) == pytest.approx(1)
    assert cosine_similarity([1, 0], [0, 1]) == 0
    assert cosine_similarity([0, 0], [0, 1]) is None

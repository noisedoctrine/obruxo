from __future__ import annotations

import pytest
import torch
from obruxo_basic_pitch.constants import AUDIO_N_SAMPLES, CQT_N_BINS, HARMONIC_SHIFTS
from obruxo_basic_pitch.frontend import (
    BasicPitchFrontend,
    harmonic_stack,
    normalized_log,
)
from obruxo_basic_pitch.model import BasicPitchICASSP2022


def test_fixed_dimensions_and_shifts() -> None:
    assert AUDIO_N_SAMPLES == 43_844
    assert CQT_N_BINS == 309
    assert HARMONIC_SHIFTS == (-36, 0, 36, 57, 72, 84, 93, 101)


def test_model_rejects_shape_and_dtype() -> None:
    model = BasicPitchICASSP2022()
    with pytest.raises(ValueError, match=r"\(1, 10, 1\)"):
        model(torch.zeros(1, 10, 1, dtype=torch.float32))
    with pytest.raises(TypeError, match="torch.float64"):
        model(torch.zeros(1, AUDIO_N_SAMPLES, 1, dtype=torch.float64))


def test_normalized_log_zero_is_finite_and_zero() -> None:
    result = normalized_log(torch.zeros(2, 172, CQT_N_BINS, dtype=torch.float32))
    assert torch.isfinite(result).all()
    assert torch.count_nonzero(result) == 0


def test_cqt_and_downsampling_padding_are_fixed() -> None:
    frontend = BasicPitchFrontend()
    audio = torch.arange(300, dtype=torch.float32).view(1, 1, 300)
    padded = torch.nn.functional.pad(audio, (128, 128), mode="reflect")
    frontend.cqt_kernels_real.zero_()
    frontend.cqt_kernels_real[0, 0, 0] = 1
    result = frontend._complex_cqt(audio, 1)[0, :, 0, 0]
    assert torch.equal(result, padded[0, 0, : result.shape[0]])

    frontend.lowpass_filter.zero_()
    frontend.lowpass_filter[0, 0, 127] = 1
    downsampled = frontend._downsample_by_two(audio)[0, 0]
    assert torch.equal(downsampled, audio[0, 0, ::2])


def test_harmonic_stack_shift_and_crop() -> None:
    x = torch.arange(CQT_N_BINS, dtype=torch.float32).view(1, 1, CQT_N_BINS)
    result = harmonic_stack(x)
    assert result.shape == (1, 8, 1, 264)
    assert torch.equal(result[0, 0, 0, :36], torch.zeros(36))
    assert torch.equal(result[0, 1, 0], x[0, 0, :264])
    assert torch.equal(result[0, 2, 0], x[0, 0, 36:300])


def test_frontend_output_shape_and_finiteness() -> None:
    frontend = BasicPitchFrontend().eval()
    with torch.inference_mode():
        result = frontend(torch.zeros(2, AUDIO_N_SAMPLES, 1, dtype=torch.float32))
    assert result.shape == (2, 8, 172, 264)
    assert torch.isfinite(result).all()

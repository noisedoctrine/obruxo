from __future__ import annotations

import torch
from obruxo_basic_pitch.constants import AUDIO_N_SAMPLES
from obruxo_basic_pitch.model import BasicPitchICASSP2022


def test_live_topology_has_no_dead_contour_layer() -> None:
    model = BasicPitchICASSP2022()
    assert model.contour_conv1.kernel_size == (3, 39)
    assert model.contour_conv2.kernel_size == (5, 5)
    assert model.note_conv1.stride == (1, 3)
    assert model.onset_conv1.stride == (1, 3)
    assert not hasattr(model, "dead_contour_conv")
    assert sum(isinstance(layer, torch.nn.Conv2d) for layer in model.modules()) == 6


def test_output_shapes_and_sigmoid_bounds() -> None:
    model = BasicPitchICASSP2022().eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, AUDIO_N_SAMPLES, 1, dtype=torch.float32))
    assert set(output) == {"note", "onset", "contour"}
    assert output["note"].shape == (1, 172, 88)
    assert output["onset"].shape == (1, 172, 88)
    assert output["contour"].shape == (1, 172, 264)
    assert all(torch.isfinite(value).all() and value.min() >= 0 and value.max() <= 1 for value in output.values())


def test_backward_reaches_live_parameters_not_frontend_buffers() -> None:
    model = BasicPitchICASSP2022()
    output = model(torch.zeros(1, AUDIO_N_SAMPLES, 1, dtype=torch.float32))
    sum(output[name].mean() for name in output).backward()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert trainable
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in trainable)
    assert all(buffer.grad is None for buffer in model.buffers())

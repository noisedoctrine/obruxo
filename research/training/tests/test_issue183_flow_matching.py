"""Regression checks for octave scoring and the learned flow transformation."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "reports/scripts/run_issue183_flow_matching.py"
SPEC = importlib.util.spec_from_file_location("issue183_runner", SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

from obruxo_training.flow_matching import (LOG_FLOOR, FlowSpectrumModel, as_spectrogram,
                                           physical_spectrum)  # noqa: E402


def performance(pitch=69, polyphonic=False):
    values = torch.zeros(83)
    values[:5] = torch.tensor([pitch / 127, 1., 0., 1. / 16, 1.])
    if polyphonic:
        values[5:10] = torch.tensor([72 / 127, 1., 0., 1. / 16, 1.])
    return values


def spectrum(*frequencies):
    result = np.zeros((2, runner.FREQUENCY_BINS), dtype=np.float32)
    for frequency in frequencies:
        index = round(frequency * runner.FFT_SIZE / runner.SAMPLE_RATE)
        result[:, index] = 1.
    return result


def test_octave_band_metric_scores_general_polyphonic_frames_and_absolute_energy():
    edges = (runner.SAMPLE_RATE / runner.FFT_SIZE) * 2. ** np.arange(11)
    target = spectrum(*(np.sqrt(edges[:-1] * edges[1:])))
    assert runner._octave_band_energy_error_db(target, target) == pytest.approx(0.)
    assert runner._octave_band_energy_error_db(target * 2, target) == pytest.approx(20 * np.log10(2), abs=.02)
    assert runner._octave_band_energy_error_db(spectrum(440), spectrum()) is None


def test_monophonic_octave_diagnostic_uses_normalized_midi_pitch_and_absolute_energy():
    target = spectrum(110, 220, 440, 880, 1760)
    assert runner._monophonic_octave_energy_error_db(target, target, performance(), .3) == pytest.approx(0.)
    assert runner._monophonic_octave_energy_error_db(spectrum(220), target, performance(), .3) > 0
    assert runner._monophonic_octave_energy_error_db(target * 2, target, performance(), .3) == pytest.approx(
        20 * np.log10(2), abs=.02)


def test_octave_metrics_handle_polyphony_and_unresolved_low_pitch_bands():
    target = spectrum(440)
    assert runner._monophonic_octave_energy_error_db(target, target, performance(polyphonic=True), .3) is None
    assert runner._monophonic_octave_energy_error_db(
        spectrum(32.7), spectrum(32.7), performance(24), .3) is None
    assert runner._octave_band_energy_error_db(target, target) == pytest.approx(0.)


@pytest.mark.parametrize("steps", [4, 8, 16])
def test_euler_recovers_endpoint_with_exact_constant_velocity(steps):
    noise = torch.tensor([[[-1., 2.], [0., 1.]]])
    clean = torch.tensor([[[3., 4.], [2., 5.]]])
    context = torch.zeros(1, 2, 64)
    mask = torch.ones(1, 2, dtype=torch.bool)
    oracle = SimpleNamespace(velocity_from_context=lambda *_: clean - noise)
    output = FlowSpectrumModel.sample_from_context(oracle, context, mask, noise, steps)
    torch.testing.assert_close(output, physical_spectrum(clean))


def test_real_flow_model_learns_and_samples_a_fixed_conditioned_target():
    torch.manual_seed(183)
    model = FlowSpectrumModel("residual_mlp", control_width=1)
    with torch.no_grad():
        model.clean_endpoint.bias.zero_()
    controls = torch.zeros(1, 1)
    midi = performance()
    times = torch.full((1, 4), .10)
    lengths = torch.tensor([22_050])
    mask = torch.ones(1, 4, dtype=torch.bool)
    clean = torch.full((1, 4, runner.FREQUENCY_BINS * 2), .5)
    expected = as_spectrogram(physical_spectrum(clean))
    batch = {"controls": controls, "performance": midi[None], "times": times, "lengths": lengths,
             "mask": mask, "spectra": expected, "log_values": clean}
    optimizer = torch.optim.AdamW(model.parameters(), lr=.02, weight_decay=0.)
    noise_rng = runner.cpu_generator(183001)

    def fixed_velocity_mse(seed):
        noise = runner._deterministic_noise(clean.shape, seed, torch.device("cpu"))
        time = torch.full((1,), .5)
        noisy = .5 * noise + .5 * clean
        context = model.context(controls, midi[None], times, lengths)
        predicted = model.velocity_from_context(context, noisy, time, mask, noise)
        return (predicted - (clean - noise)).square().mean()

    initial_mse = fixed_velocity_mse(183002)
    for _ in range(250):
        optimizer.zero_grad(set_to_none=True)
        loss = runner._training_objective(model, "flow", batch, noise_rng)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
    final_mse = fixed_velocity_mse(183002)
    assert final_mse < initial_mse * .5

    context = model.context(controls, midi[None], times, lengths)
    noise_a = runner._deterministic_noise(clean.shape, 183003, torch.device("cpu"))
    noise_b = runner._deterministic_noise(clean.shape, 183004, torch.device("cpu"))
    output_a = model.sample_from_context(context, mask, noise_a, 16)
    output_b = model.sample_from_context(context, mask, noise_b, 16)
    log_a = torch.log1p(output_a / LOG_FLOOR)
    log_b = torch.log1p(output_b / LOG_FLOOR)
    assert (log_a - clean).abs().mean() < .12
    assert (log_a - log_b).abs().mean() < .12

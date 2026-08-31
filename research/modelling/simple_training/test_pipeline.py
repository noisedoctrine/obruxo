from argparse import Namespace
import json
from pathlib import Path
import sys

import pytest
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline  # noqa: E402


def test_decoder_round_trip_and_isolation(tmp_path):
    codec = pipeline.PresetCodec()
    baseline = codec.template.to_dict()
    for category in range(3):
        dense = torch.tensor([-100., -100., -100., 0.])
        dense[category] = 100
        preset = codec.decode(dense)
        assert codec.decode(dense).to_json() == preset.to_json()
        assert preset.get_raw("osc_1_transpose") == pipeline.OCTAVES[category]
        assert preset.get_raw("osc_1_level") == pytest.approx(0.5)
        path = tmp_path / f"{category}.vital"
        preset.save(path)
        assert pipeline.VitalPreset.load(path).to_dict() == preset.to_dict()
        actual = preset.to_dict()
        for name in ("osc_1_transpose", "osc_1_level"):
            actual["settings"][name] = baseline["settings"][name]
        assert actual == baseline  # Includes wavetable assets, routing, disabled effects, etc.
    assert codec.template.to_dict() == baseline
    assert codec.decode(torch.zeros(4)).get_raw("osc_1_transpose") == -12
    assert codec.decode(torch.tensor([0., 1., 0., 1000.])).get_raw("osc_1_level") == pytest.approx(0.8)
    assert codec.decode(torch.tensor([0., 1., 0., -1000.])).get_raw("osc_1_level") == pytest.approx(0.2)
    for invalid in (torch.zeros(3), torch.tensor([0., 1., 0., float("nan")])):
        with pytest.raises(ValueError):
            codec.decode(invalid)


def test_perceptual_spectral_loss_has_audio_gradients():
    torch.set_num_threads(2)
    spectrum = pipeline.PerceptualSpectrum()
    time = torch.arange(pipeline.SAMPLES) / pipeline.SAMPLE_RATE
    target = (0.3 * torch.sin(2 * torch.pi * 440 * time))[None]
    assert spectrum.loss(target, target).item() == 0
    predicted = (target * 0.5).requires_grad_()
    loss = spectrum.loss(predicted, target)
    loss.backward()
    assert loss > 0
    assert torch.isfinite(predicted.grad).all() and predicted.grad.abs().sum() > 0
    assert torch.isfinite(spectrum(torch.zeros_like(target))).all()


def test_spectral_gradient_reaches_both_dense_fields_with_frozen_proxy():
    torch.manual_seed(5)
    proxy = pipeline.SpectralSurrogate().requires_grad_(False)
    dense = torch.zeros(2, 4, requires_grad=True)
    loss = pipeline.surrogate_loss(dense, torch.rand(2, pipeline.PerceptualSpectrum.size), proxy)
    loss.backward()
    assert torch.isfinite(dense.grad).all()
    assert dense.grad[:, :3].abs().sum() > 0
    assert dense.grad[:, 3].abs().sum() > 0
    assert all(parameter.grad is None for parameter in proxy.parameters())


def test_expected_loss_enumerates_only_legal_categories():
    class ExactProxy(nn.Module):
        def forward(self, controls):
            assert torch.equal(controls[:, :3], F.one_hot(controls[:, :3].argmax(1), 3).float())
            return controls[:, :3] + controls[:, 3:]

    dense = nn.Parameter(torch.zeros(1, 4))
    target = torch.tensor([[0.8, 1.8, 0.8]])
    optimizer = torch.optim.Adam([dense], lr=0.2)
    first = pipeline.surrogate_loss(dense, target, ExactProxy()).item()
    for _ in range(100):
        optimizer.zero_grad()
        loss = pipeline.surrogate_loss(dense, target, ExactProxy())
        loss.backward()
        optimizer.step()
    assert loss.item() < first / 10
    assert dense[0, :3].argmax().item() == 1
    assert dense[0, 3].sigmoid().item() == pytest.approx(0.8, abs=0.03)


def test_transformer_summary_attends_to_both_sequences_and_time_order():
    torch.manual_seed(7)
    fusion = pipeline.TinyFusion().eval()
    timbre = torch.randn(2, pipeline.VALID_FRAMES, 440, requires_grad=True)
    performance = torch.randn_like(timbre, requires_grad=True)
    level = torch.zeros(2, 1)
    output = fusion(timbre, performance, level)
    assert output.shape == (2, pipeline.VALID_FRAMES + 1, 32)
    assert not torch.allclose(output[:, 0], fusion(timbre.flip(1), performance.flip(1), level)[:, 0], atol=1e-6)
    output[:, 0, 0].sum().backward()
    assert torch.isfinite(timbre.grad).all() and timbre.grad.abs().sum() > 0
    assert torch.isfinite(performance.grad).all() and performance.grad.abs().sum() > 0
    attention = fusion.transformer.layers[0].self_attn
    assert attention.in_proj_weight.grad.abs().sum() > 0


def test_audio_conditioning_is_independent_of_prediction_head():
    model = pipeline.AudioToDense().eval()
    time = torch.arange(pipeline.SAMPLES) / pipeline.SAMPLE_RATE
    audio = (0.3 * torch.sin(2 * torch.pi * 261.6256 * time))[None]
    with torch.no_grad():
        tokens = model.encode_audio(audio)
        before = model(audio)
        model.head.bias.add_(1)
        assert torch.equal(tokens, model.encode_audio(audio))
        assert torch.allclose(model(audio), before + 1, atol=1e-6)
    assert tokens.shape == (1, 53, 32)
    assert not model.performance.training


def test_train_export_reload_predict_with_test_renderer(tmp_path, monkeypatch):
    # Wiring test only. A sine stand-in is not evidence of native Vital compatibility.
    def render(self, preset):
        self.render_count += 1
        frequency = 261.6256 * 2 ** (preset.get_raw("osc_1_transpose") / 12)
        time = torch.arange(pipeline.SAMPLES) / pipeline.SAMPLE_RATE
        return preset.get_raw("osc_1_level") ** 2 * torch.sin(2 * torch.pi * frequency * time)

    class Renderer:
        renderer_id = "test-only-sine-renderer"

        def __init__(self, *args):
            pass

    monkeypatch.setattr(pipeline, "VitalRenderer", Renderer)
    monkeypatch.setattr(pipeline.DynamicAudio, "render", render)
    output = tmp_path / "run"
    pipeline.train(Namespace(output=str(output), seed=7, proxy_steps=2, steps=2, batch_size=2, plugin_path=None, device="cpu"))
    report = json.loads((output / "report.json").read_text())
    assert report["render_count"] == 31  # 5 control checks + 6 targets + 12 evaluations + 8 training renders.
    assert report["frozen_branch_unchanged"] and report["trainable_branch_changed"]
    assert report["transformer_fusion_changed"]
    assert all(row["fusion_gradient_norm"] > 0 for row in report["history"] if row["stage"] == "audio_to_dense")
    assert all(row["gradient_norm"] > 0 for row in report["history"])
    prediction = tmp_path / "reloaded.vital"
    args = Namespace(checkpoint=str(output / "checkpoint.pt"), audio=str(output / "target.wav"), output=str(prediction), device="cpu")
    pipeline.predict(args)
    # Batch vs single-example BLAS may differ by float32 roundoff before decoding.
    reloaded = pipeline.VitalPreset.load(prediction)
    original = pipeline.VitalPreset.load(output / "predicted.vital")
    assert reloaded.get_raw("osc_1_transpose") == original.get_raw("osc_1_transpose")
    assert reloaded.get_raw("osc_1_level") == pytest.approx(original.get_raw("osc_1_level"), abs=1e-6)
    with pytest.raises(FileExistsError):
        pipeline.predict(args)

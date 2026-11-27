import torch

from obruxo_data.midi import Performance
from obruxo_training.codec import PilotCodec
from obruxo_training.issue187_stage2 import (
    EventSetEncoder, Issue187Adapter, MidiEmbeddingCache, align_midi_tokens, encode_midi_spans,
    midi_cache_key, performance_to_spans, trainable_parameter_count,
)
from obruxo_training.surrogate_codec import SurrogateCodec


def score_notes():
    return [
        {"onset": 960, "offset": 10_080, "pitch": 60, "velocity": 100, "program": 80, "physical_id": 0},
        {"onset": 2_880, "offset": 8_640, "pitch": 64, "velocity": 55, "program": 80, "physical_id": 1},
    ]


def test_span_rows_retain_polyphony_timing_velocity_and_release():
    rows = encode_midi_spans(score_notes(), duration_seconds=.5)
    assert rows["kind_id"].shape == (1, 512, 128)
    assert rows["frame_valid"].sum().item() == 13
    assert rows["event_valid"][0, 0, :].sum().item() == 1
    assert rows["kind_id"][0, 1, :2].tolist() == [2, 1]
    assert rows["pitch_id"][0, 1, :2].tolist() == [61, 65]
    assert rows["numeric"][0, 1, 0, 4].item() != rows["numeric"][0, 1, 1, 4].item()
    assert (rows["kind_id"] == 3).any()
    assert not rows["event_valid"][:, 13:].any()


def test_token_alignment_samples_in_seconds_and_is_exact_at_native_rate():
    tokens = torch.arange(6, dtype=torch.float32).view(1, 6, 1)
    assert torch.equal(align_midi_tokens(tokens, 4, 25), tokens[:, :4])
    assert torch.allclose(align_midi_tokens(tokens, 5, 50).flatten(), torch.tensor([0., .5, 1., 1.5, 2.]))


def test_performance_ticks_and_tempo_changes_map_to_absolute_samples():
    score = Performance(bpm=120)
    score.add_tempo(60, tick=480)
    score.add_note(pitch=48, velocity=70, start_tick=480, duration_ticks=480)
    score.add_note(pitch=55, velocity=90, start_tick=720, duration_ticks=240, channel=9)
    assert performance_to_spans(score, default_program=80) == [
        {"onset": 24_000, "offset": 72_000, "pitch": 48, "velocity": 70, "program": 80, "physical_id": 0},
        {"onset": 48_000, "offset": 72_000, "pitch": 55, "velocity": 90, "program": 128, "physical_id": 1},
    ]


def test_cache_is_keyed_by_content_revision_and_reuses_frozen_embedding(tmp_path):
    notes = score_notes()
    encoder = EventSetEncoder().eval().requires_grad_(False)
    cache = MidiEmbeddingCache(tmp_path)
    first = cache.get_or_encode(notes, .5, encoder, device="cpu")
    second = cache.get_or_encode(notes, .5, encoder, device="cpu")
    assert first.shape == (13, 768)
    assert torch.equal(first, second)
    assert not first.requires_grad
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in encoder.parameters())
    assert midi_cache_key(notes, .5) != midi_cache_key(notes, .5, "another-revision")
    assert len(list(tmp_path.glob("*.pt"))) == 1


def test_adapter_variants_are_parameter_matched_and_backpropagate_for_both_arms():
    counts = {}
    control_width = SurrogateCodec(PilotCodec(), expanded=True).width
    assert control_width == 33
    for arm, output_width in (("direct", 256), ("generator", 768)):
        for backbone in ("residual_mlp", "transformer"):
            model = Issue187Adapter(backbone, control_width, output_width, generator=arm == "generator")
            counts[arm, backbone] = trainable_parameter_count(model)
            controls = torch.randn(2, control_width, requires_grad=True)
            midi = torch.randn(2, 16, 768)
            mask = torch.ones(2, 16, dtype=torch.bool)
            result = model(controls, midi, mask, target_frames=8, target_hz=20)
            values = (result[0] if arm == "generator" else result).float()
            loss = values.square().mean()
            loss.backward()
            assert controls.grad is not None and torch.isfinite(controls.grad).all() and controls.grad.norm() > 0
            assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0
                       for p in model.parameters())
            assert values.shape == ((2, 8, output_width) if arm == "generator" else (2, output_width, 8))
    for arm in ("direct", "generator"):
        assert sum(p.numel() for p in Issue187Adapter("residual_mlp", control_width, 256).blocks.parameters()
                   if p.ndim == 2) == 81_920
        assert sum(p.numel() for p in Issue187Adapter("transformer", control_width, 256).blocks.parameters()
                   if p.ndim == 2) == 81_920
        mlp, transformer = counts[arm, "residual_mlp"], counts[arm, "transformer"]
        assert abs(mlp - transformer) / max(mlp, transformer) < .01
    assert counts == {("direct", "residual_mlp"): 151_488, ("direct", "transformer"): 151_936,
                      ("generator", "residual_mlp"): 184_768, ("generator", "transformer"): 185_216}

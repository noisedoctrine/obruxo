from dataclasses import replace
import inspect
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from obruxo_training.audio_candidates import (RANKING_SPEC, aligned_distance, candidate_bank, export_candidate_model,  # noqa: E402
                                             load_candidate_model, performance_from_notes, rank_audio)
from obruxo_training.cli import main  # noqa: E402
from obruxo_training.evaluate import load_evaluation_checkpoint  # noqa: E402
from obruxo_training.features import AudioFeatures, tensor_digest  # noqa: E402
from obruxo_training.pilot import PilotPool  # noqa: E402
from obruxo_training.ranking_check import check_audio_ranking  # noqa: E402
from obruxo_training.stream import request_frames  # noqa: E402
from obruxo_training.trainer import TrainConfig, train_pilot  # noqa: E402
from test_stream import FakeRendererFactory  # noqa: E402
from obruxo_basic_pitch.postprocess import NoteEvent  # noqa: E402


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    pool = PilotPool.create(32, seed=0)
    root = tmp_path_factory.mktemp("audio-candidate-source")
    train_pilot(pool, root / "source", config=TrainConfig(steps=3, replay_capacity=4, batch_size=4, render_budget=4, cpu_threads=2),
                renderer_factory=FakeRendererFactory())
    saved, model, frontend, _ = load_evaluation_checkpoint(root / "source/checkpoint.pt", pool)
    pool.save(root / "pool.json")
    return pool, root, saved, model, frontend


def posterior(with_notes=True):
    arrays = {"note": np.zeros((100, 88), dtype=np.float32), "onset": np.zeros((100, 88), dtype=np.float32),
              "contour": np.zeros((100, 264), dtype=np.float32)}
    if with_notes:
        arrays["note"][10:75, 51] = .9  # MIDI 72, supplied only through the audio posterior seam.
        arrays["onset"][10, 51] = .9
        arrays["contour"][10:75, 153] = .9
    return arrays


def test_dense_frontend_reuses_frozen_pass_without_changing_existing_features():
    torch.set_num_threads(2)
    frontend = AudioFeatures()
    audio = np.stack([.1 * np.sin(np.arange(44100) * .031)] * 2, 1).astype(np.float32)
    old = frontend(audio)
    features, dense = frontend.analyze(audio)
    assert all(torch.equal(old[name], features[name]) for name in old)
    assert set(dense) == {"note", "onset", "contour"} and dense["note"].shape[1] == 88
    frontend.require_frozen()


def test_estimated_performance_is_bounded_and_has_no_supplied_midi():
    note = NoteEvent(.1, .8, 72, .9)
    played = performance_from_notes([note], 1.)
    spans = played.note_spans()
    assert spans[0].pitch == 72 and spans[0].velocity == 114
    assert spans[0].start_tick == 96 and spans[0].end_tick == 768
    for notes in ([], [note] * 33, [replace(note, pitch_midi=72.5)], [replace(note, end_time_s=2.)]):
        with pytest.raises(ValueError):
            performance_from_notes(notes, 1.)
    assert "performance" not in inspect.signature(rank_audio).parameters
    assert "source_target" not in inspect.signature(rank_audio).parameters
    assert RANKING_SPEC["input"] == "audio_only"


def test_candidate_bundle_is_portable_train_only_and_rejects_tampering(source, tmp_path):
    pool, root, saved, _, _ = source
    identity = export_candidate_model(pool, root / "source/checkpoint.pt", tmp_path / "model.pt")
    with pytest.raises(ValueError, match="native qualification"):
        load_candidate_model(tmp_path / "model.pt")
    model, frontend, codec, bank, renderer = load_candidate_model(tmp_path / "model.pt", allow_unqualified=True)
    assert all(pool.manifest.by_state[item["state_id"]].split == "train" for item in bank)
    assert len(bank) == len({item["state_id"] for item in bank}) <= 32
    assert tensor_digest(model.state_dict()) == identity["model_sha256"] and renderer == saved["renderer_identity"]
    assert codec.to_dict() == pool.codec.to_dict()
    frontend.require_frozen()
    altered = torch.load(tmp_path / "model.pt", weights_only=True)
    altered["bank"][0]["features"]["pitch"][0] += .1
    torch.save(altered, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="changed"):
        load_candidate_model(tmp_path / "bad.pt", allow_unqualified=True)
    with pytest.raises(ValueError, match="match"):
        export_candidate_model(pool, root / "source/checkpoint.pt", tmp_path / "bad-qualification.pt", qualification={"qualified": True})


def test_audio_only_ranking_uses_estimated_notes_and_keeps_tie_order(source, monkeypatch):
    pool, _, saved, model, frontend = source
    features = saved["replay"][0]["features"]
    calls = []
    monkeypatch.setattr(frontend, "analyze", lambda *args: (features, posterior()))
    def render(request, **kwargs):
        calls.append(request)
        audio = np.zeros((request_frames(request), 2), dtype=np.float32)
        return audio, {"status": "accepted", "provenance": {**saved["renderer_identity"], "request_id": request.request_id}}
    monkeypatch.setattr("obruxo_training.audio_candidates.render_isolated", render)
    audio = np.zeros((44100, 2), dtype=np.float32)
    before = tensor_digest(frontend.basic_pitch.state_dict())
    target, result = rank_audio(model, frontend, pool.codec, candidate_bank(saved), audio, expected_renderer=saved["renderer_identity"])
    assert result["ranked"] and result["selected"] == "model" and not result["source_midi_supplied"]
    assert 1 <= len(calls) <= 4 and all(request.performance.note_spans()[0].pitch == 72 for request in calls)
    assert result["candidates"]["model"] == target.to_dict()
    assert tensor_digest(frontend.basic_pitch.state_dict()) == before


def test_no_notes_and_failed_renders_are_explicit_fallbacks(source, monkeypatch):
    pool, _, saved, model, frontend = source
    audio = np.zeros((44100, 2), dtype=np.float32)
    monkeypatch.setattr(frontend, "analyze", lambda *args: (saved["replay"][0]["features"], posterior(False)))
    calls = []
    monkeypatch.setattr("obruxo_training.audio_candidates.render_isolated", lambda *args, **kwargs: calls.append(1))
    _, result = rank_audio(model, frontend, pool.codec, candidate_bank(saved), audio)
    assert not result["ranked"] and "no notes" in result["failure"] and not calls
    monkeypatch.setattr(frontend, "analyze", lambda *args: (saved["replay"][0]["features"], posterior()))
    monkeypatch.setattr("obruxo_training.audio_candidates.render_isolated", lambda *args, **kwargs: (None, {"status": "error"}))
    _, result = rank_audio(model, frontend, pool.codec, candidate_bank(saved), audio)
    assert not result["ranked"] and result["selected"] == "model" and result["failure"] == "all_candidate_renders_rejected"


def test_alignment_keeps_gain_and_tail_mismatch_visible():
    audio = np.stack([.1 * np.sin(np.arange(44100) * .03)] * 2, 1).astype(np.float32)
    assert aligned_distance(audio, audio)["mrstft"] == 0
    assert aligned_distance(audio, audio * .5)["mrstft"] > 0
    assert aligned_distance(audio, np.concatenate((audio, audio)))["mrstft"] > 0


def test_cli_exports_without_native_rendering_and_refuses_existing_output(source, tmp_path):
    _, root, _, _, _ = source
    output = tmp_path / "cli-model.pt"
    arguments = ["candidate-export", "--pool", str(root / "pool.json"), "--checkpoint", str(root / "source/checkpoint.pt"), "--output", str(output)]
    assert main(arguments) == 0
    with pytest.raises(SystemExit):
        main(arguments)


def test_ordinary_inference_accepts_mono_equivalent_to_duplicated_stereo(source, tmp_path):
    from scipy.io import wavfile
    _, root, _, _, _ = source
    pcm = (1000 * np.sin(np.arange(44100) * .03)).astype(np.int16)
    wavfile.write(tmp_path / "mono.wav", 44100, pcm)
    wavfile.write(tmp_path / "stereo.wav", 44100, np.stack((pcm, pcm), 1))
    for name in ("mono", "stereo"):
        assert main(["infer", "--model", str(root / "source/model.pt"), "--audio", str(tmp_path / f"{name}.wav"),
                     "--output", str(tmp_path / f"{name}.vital")]) == 0
    assert (tmp_path / "mono.vital").read_bytes() == (tmp_path / "stereo.vital").read_bytes()


def test_ranking_benchmark_separates_deployed_and_oracle_and_recovers(source, tmp_path):
    pool, root, _, _, _ = source
    def interrupt(event):
        raise InterruptedError
    with pytest.raises(InterruptedError):
        check_audio_ranking(pool, root / "source/checkpoint.pt", tmp_path / "check", count=1,
                            renderer_factory=FakeRendererFactory(), progress=interrupt)
    result = check_audio_ranking(pool, root / "source/checkpoint.pt", tmp_path / "check", count=1,
                                 renderer_factory=FakeRendererFactory(), resume=True)
    case = json.loads((tmp_path / "check/case-000.json").read_text())
    assert set(case["methods"]) >= {"model", "deployed", "oracle_best_four"}
    assert not result["final_test_used"] and result["renders_attempted"] <= 32
    assert not result["qualification"]["qualified"]

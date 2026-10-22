import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from obruxo_training.cli import main
from obruxo_training.codec import PilotCodec
from obruxo_training.features import AudioFeatures, tensor_digest
from obruxo_training.inference_profile import artifact_rendering, bind_rendering, model_rendering, prediction_preset
from obruxo_training.model import PilotInverse
from obruxo_training.profiled_sampler import profile_metadata
from obruxo_training.render_profile import basic_render_profile
from obruxo_training.reproducibility import FULL_PHASE_CONTROL, fixed_generator_phases
from obruxo_training.trainer import export_model, inference_state, load_model, restore_model


@pytest.fixture(scope="module")
def bundle():
    torch.set_num_threads(2)
    codec, frontend = PilotCodec(), AudioFeatures()
    model = PilotInverse(codec).eval()
    return model, frontend, codec


def test_profiled_preset_matches_evaluated_phase_controls_without_changing_predicted_controls(bundle):
    model, frontend, codec = bundle
    target = codec.default_target()
    raw, before = codec.decode(target), tensor_digest(model.state_dict())
    bind_rendering(model, profile_metadata(32768))
    predicted = prediction_preset(model, codec, target)
    assert predicted == fixed_generator_phases(raw)
    assert all(json.loads(predicted)["settings"][key] == value for key, value in FULL_PHASE_CONTROL.items())
    assert codec.encode(raw) == target and tensor_digest(model.state_dict()) == before
    expected = json.loads(raw)
    expected["settings"].update(FULL_PHASE_CONTROL)
    assert json.loads(predicted) == expected
    exported = inference_state(model, frontend, codec, "fixture")
    assert all(exported[key] == value for key, value in profile_metadata(32768).items())
    bind_rendering(model, {})
    assert prediction_preset(model, codec, target) == raw and model_rendering(model) == {}


def test_incomplete_or_changed_artifact_profile_is_rejected_before_model_loading(bundle):
    model, frontend, codec = bundle
    valid = inference_state(model, frontend, codec, "fixture", preroll_frames=32768)
    for change in ("missing", "random_phase", "lead", "bool_phase"):
        altered = copy.deepcopy(valid)
        if change == "missing":
            altered.pop("render_phase_controls")
        elif change == "random_phase":
            altered["render_phase_controls"]["osc_1_random_phase"] = 1.
        elif change == "lead":
            altered["render_profile"]["preroll_frames"] = 8192
        else:
            altered["render_phase_controls"]["sample_random_phase"] = False
        with pytest.raises(ValueError, match="inference render profile"):
            restore_model(altered)
    cold = {"render_profile": basic_render_profile(0), "render_phase_controls": dict(FULL_PHASE_CONTROL)}
    assert artifact_rendering(cold) == cold


def test_actual_cli_exports_profiled_phases_and_keeps_legacy_output_unchanged(bundle, tmp_path):
    from scipy.io import wavfile
    model, frontend, codec = bundle
    audio = (.05 * np.sin(np.arange(22050) * .04)).astype(np.float32)
    wavfile.write(tmp_path / "input.wav", 44100, np.stack((audio, audio), 1))
    outputs = {}
    for name, lead in (("legacy", 0), ("profiled", 32768)):
        path = tmp_path / f"{name}.pt"
        export_model(path, model, frontend, codec, "fixture", preroll_frames=lead)
        assert main(["infer", "--model", str(path), "--audio", str(tmp_path / "input.wav"), "--output", str(tmp_path / f"{name}.vital")]) == 0
        outputs[name] = (tmp_path / f"{name}.vital").read_bytes()
    assert outputs["profiled"] == fixed_generator_phases(outputs["legacy"])
    restored, _, _ = load_model(tmp_path / "profiled.pt")
    export_model(tmp_path / "again.pt", restored, frontend, codec, "fixture")
    again = torch.load(tmp_path / "again.pt", weights_only=True)
    assert all(again[key] == value for key, value in profile_metadata(32768).items())


def test_candidate_bundle_preserves_profile_and_rejects_changed_identity(bundle, tmp_path, monkeypatch):
    from obruxo_training import audio_candidates as candidates
    model, frontend, codec = bundle
    bind_rendering(model, profile_metadata(32768))
    features = {"timbre": torch.zeros(2, 96, 32), "pitch": torch.zeros(3521)}
    source = {"replay": [{"state_id": "fixture", "target": codec.default_target().to_dict(), "features": features}],
              "renderer_identity": {"fixture": True}}
    monkeypatch.setattr(candidates, "load_evaluation_checkpoint", lambda *_args: (source, model, frontend, None))
    path = tmp_path / "candidate.pt"
    candidates.export_candidate_model(SimpleNamespace(codec=codec, pool_id="fixture"), None, path)
    loaded, _, _, _, _ = candidates.load_candidate_model(path, allow_unqualified=True)
    assert model_rendering(loaded) == profile_metadata(32768)
    changed = torch.load(path, weights_only=True)
    changed["identity"].pop("render_phase_controls")
    torch.save(changed, tmp_path / "changed.pt")
    with pytest.raises(ValueError, match="rendering metadata"):
        candidates.load_candidate_model(tmp_path / "changed.pt", allow_unqualified=True)
    bind_rendering(model, {})


def test_candidate_renders_use_declared_phases_and_lead(bundle, monkeypatch):
    from obruxo_basic_pitch.postprocess import NoteEvent
    from obruxo_training import audio_candidates as candidates
    model, frontend, codec = bundle
    model.eval()
    bind_rendering(model, profile_metadata(32768))
    features = {"timbre": torch.zeros(2, 96, 32), "pitch": torch.zeros(3521)}
    audio, requests = np.zeros((44100, 2), np.float32), []
    monkeypatch.setattr(frontend, "analyze", lambda *_args: (features, {}))
    monkeypatch.setattr("obruxo_basic_pitch.postprocess.posteriorgrams_to_note_events", lambda *_args: [NoteEvent(.1, .7, 72, .8)])

    def render(request, **_kwargs):
        requests.append(request)
        return audio, {"status": "accepted", "provenance": {}}

    monkeypatch.setattr(candidates, "render_isolated", render)
    candidates.rank_audio(model, frontend, codec, [{"state_id": "fixture", "target": codec.default_target().to_dict(), "features": features}], audio)
    assert len(requests) == 4
    assert all(request.preroll_frames == 32768 for request in requests)
    assert all(json.loads(request.preset_json)["settings"][key] == value for request in requests for key, value in FULL_PHASE_CONTROL.items())
    bind_rendering(model, {})

import copy
import json

import numpy as np
import pytest
from scipy.io import wavfile
import torch

from obruxo_training.basic_direct import BasicDirect, export_direct
from obruxo_training.cli import main
from obruxo_training.codec import PilotCodec
from obruxo_training.features import AudioFeatures, BasicPitchICASSP2022
from obruxo_training.inference_profile import bind_rendering, prediction_preset
from obruxo_training.model import PilotInverse
from obruxo_training.prediction import load_predictor
from obruxo_training.profiled_sampler import profile_metadata
from obruxo_training.reproducibility import FULL_PHASE_CONTROL
from obruxo_training.trainable_basic import TrainableBasic
from obruxo_training.trainer import inference_state, predict_audio


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    torch.set_num_threads(2)
    torch.manual_seed(1)
    root = tmp_path_factory.mktemp("audio-preset-inference")
    base, codec = PilotInverse().eval(), PilotCodec()
    bind_rendering(base, profile_metadata(32768))
    frontend = AudioFeatures(BasicPitchICASSP2022().state_dict())
    signal = (.03 * np.sin(np.arange(22050) * .04)).astype(np.float32)
    audio = np.stack((signal, .6 * signal), axis=1)
    wavfile.write(root / "input.wav", 44100, audio)
    saved = {}
    for name, kind in (("direct", BasicDirect), ("inputs", TrainableBasic)):
        model = kind(base).eval()
        with torch.no_grad():
            model.correction[2].weight.fill_(.0003)
            model.correction[2].bias.copy_(torch.tensor([.02, -.03, .01, -.01, .02, .01, -.01, .03, -.02, -.2, .1, .3, -.1, .2]))
            if kind is TrainableBasic:
                model.correction[0].timbre[0].weight.add_(.001)
                model.correction[0].pitch[0].bias.add_(.002)
        saved[name] = export_direct(model, frontend, codec, "software-test")
        torch.save(saved[name], root / f"{name}.pt")
    saved["legacy"] = inference_state(base, frontend, codec, "software-test")
    torch.save(saved["legacy"], root / "legacy.pt")
    return root, base, frontend, codec, audio, saved


@pytest.mark.parametrize("name", ["direct", "inputs"])
def test_actual_audio_cli_keeps_the_full_correction_and_evaluated_phase_policy(artifacts, tmp_path, capsys, name):
    root, base, frontend, codec, audio, _saved = artifacts
    output = tmp_path / f"{name}.vital"
    assert main(["infer", "--model", str(root / f"{name}.pt"), "--audio", str(root / "input.wav"), "--output", str(output)]) == 0
    assert torch.get_num_threads() == 2
    model, restored_frontend, restored_codec, settings = load_predictor(root / f"{name}.pt")
    target = predict_audio(model, restored_frontend, restored_codec, audio)
    assert output.read_bytes() == prediction_preset(model, restored_codec, target)
    assert target != predict_audio(base, frontend, codec, audio)
    assert len(target.continuous) == 9 and len(target.categorical) == 2
    preset = json.loads(output.read_bytes())
    assert all(preset["settings"][key] == value for key, value in FULL_PHASE_CONTROL.items())
    report = json.loads(capsys.readouterr().out)
    assert report["target"] == target.to_dict() and report["qualified_for_release"] is False
    assert settings["qualified_for_release"] is False


def test_legacy_cli_thread_setting_is_preserved(artifacts, tmp_path):
    root, _base, _frontend, _codec, _audio, _saved = artifacts
    assert main(["infer", "--model", str(root / "legacy.pt"), "--audio", str(root / "input.wav"), "--output", str(tmp_path / "legacy.vital")]) == 0
    assert torch.get_num_threads() == 4


@pytest.mark.parametrize("change", ["weights", "phase", "released", "unknown", "checkpoint"])
def test_changed_or_unsupported_artifacts_are_rejected_before_prediction(artifacts, tmp_path, change):
    _root, _base, _front, _codec, _audio, saved = artifacts
    altered = copy.deepcopy(saved["inputs"])
    if change == "weights":
        altered["correction"]["0.timbre.0.weight"][0, 0, 0, 0] += .1
    elif change == "phase":
        altered["render_phase_controls"]["sample_random_phase"] = .5
    elif change == "released":
        altered["qualified_for_release"] = True
    else:
        altered = {"format": "unknown" if change == "unknown" else "obruxo_pilot_training_v1"}
    path = tmp_path / "changed.pt"
    torch.save(altered, path)
    with pytest.raises(ValueError, match="changed|phase|profile|artifact"):
        load_predictor(path)


def test_inference_does_not_overwrite_an_existing_preset(artifacts, tmp_path):
    root, _base, _front, _codec, _audio, _saved = artifacts
    output = tmp_path / "existing.vital"
    output.write_bytes(b"keep this preset")
    with pytest.raises(SystemExit) as error:
        main(["infer", "--model", str(root / "direct.pt"), "--audio", str(root / "input.wav"), "--output", str(output)])
    assert error.value.code == 1
    assert output.read_bytes() == b"keep this preset"

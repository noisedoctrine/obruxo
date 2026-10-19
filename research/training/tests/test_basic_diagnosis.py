import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from obruxo_training.codec import PilotCodec, PilotTarget  # noqa: E402

spec = importlib.util.spec_from_file_location("basic_diagnosis", Path(__file__).resolve().parents[1] / "reports/scripts/diagnose_basic_sound.py")
diagnosis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnosis)


@pytest.mark.parametrize("source_choices,predicted_choices", [((0, 0), (2, 1)), ((2, 1), (0, 0))])
def test_interventions_change_only_their_group_and_keep_inactive_values_legal(source_choices, predicted_choices):
    codec = PilotCodec()
    source = codec.decode(codec.canonical_target(PilotTarget((.2,) * 9, source_choices)))
    predicted = codec.decode(codec.canonical_target(PilotTarget((.8,) * 9, predicted_choices)))
    variants = diagnosis.corrections(codec, source, predicted)
    assert variants["source"] == variants["source_repeat"] == source
    assert variants["prediction"] == predicted
    original = json.loads(predicted)
    expected_settings = json.loads(source)["settings"]
    for group, names in diagnosis.GROUPS.items():
        expected = json.loads(predicted)
        expected["settings"].update({name: expected_settings[name] for name in names})
        assert json.loads(variants[group]) == expected
        codec.encode(variants[group])
        assert all(expected["settings"][name] == value for name, value in original["settings"].items() if name not in names)


def test_sound_diagnostics_keep_level_and_stereo_errors_visible():
    mono = np.sin(np.arange(44100) * .03).astype(np.float32) * .1
    source = np.stack((mono, mono * .3), axis=1)
    identical = diagnosis.sound_metrics(source, source.copy())
    assert all(identical[key] == 0 for key in ("mrstft", "envelope_relative_rmse", "stereo_energy_error"))
    quiet = diagnosis.sound_metrics(source, source * .5)
    assert quiet["envelope_relative_rmse"] == pytest.approx(.5)
    assert quiet["mrstft"] > 0
    swapped = diagnosis.sound_metrics(source, source[:, ::-1])
    assert swapped["stereo_energy_error"] > 0
    assert diagnosis.sound_metrics(None, source) == {"failure": "source_or_candidate_not_accepted"}

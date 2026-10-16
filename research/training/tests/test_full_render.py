from copy import deepcopy
from dataclasses import replace
import base64
import json

import numpy as np
import pytest

from obruxo_data.errors import ValidationError
from obruxo_data.render.base import FULL_PRESET_RENDERER_ID, LEGACY_TRAINING_RENDERER_ID, RenderRequest
from obruxo_data.render.full_state import verify_full_nested_state
from obruxo_data.render.vita import VitalRenderer
from obruxo_data.vital.full_contract import FullPresetContract
from obruxo_training.performances import initial_bank
from obruxo_training.stream import NativeRendererFactory


@pytest.fixture(scope="module")
def contract():
    return FullPresetContract()


def loaded_state(contract, document):
    loaded = deepcopy(document)
    settings = loaded["settings"]
    settings.update(contract.spec["fixed_native_ramps"])
    for wave in settings["wavetables"]:
        wave["version"] = "1.6.4"
    for key in {"samples", "samples_stereo"} & settings["sample"].keys():
        before = np.frombuffer(base64.b64decode(settings["sample"][key]), dtype="<i2")
        after = np.zeros_like(before)
        after[4:] = before[:-4]
        settings["sample"][key] = base64.b64encode(after.tobytes()).decode()
    return loaded


def test_full_requests_reject_wrong_versions_unknown_fields_and_duplicate_keys(contract):
    def request(raw):
        return RenderRequest(preset_json=raw, performance=initial_bank()[0].performance(), renderer_id=FULL_PRESET_RENDERER_ID)

    document = contract.template()
    accepted = request(json.dumps(document))
    assert RenderRequest.from_dict(accepted.to_dict()).request_id == accepted.request_id
    assert accepted.renderer_id != LEGACY_TRAINING_RENDERER_ID
    for key, value in (("synth_version", "1.6.4"), ("surprise", 1)):
        with pytest.raises(ValidationError):
            request(json.dumps({**document, key: value}))
    with pytest.raises(ValidationError, match="duplicate"):
        request(json.dumps(document)[:-1] + ',"synth_version":"1.5.5"}')
    with pytest.raises(ValidationError):
        request(json.dumps(document).replace('"osc_1_level":', '"unknown_control":', 1))


def test_full_profile_validates_without_vita_or_plugin_loading(contract, monkeypatch):
    renderer = object.__new__(VitalRenderer)
    renderer.renderer_id, renderer.full_contract = FULL_PRESET_RENDERER_ID, contract
    from obruxo_data.render.capabilities import RendererCapabilities
    renderer._capabilities = RendererCapabilities(max_channels=1)
    monkeypatch.setattr("obruxo_data.vital.VitalPreset.validate", lambda *_a, **_k: pytest.fail("full presets must not use the legacy Vita validator"))
    request = RenderRequest(preset_json=json.dumps(contract.template()), performance=initial_bank()[0].performance(),
                            renderer_id=FULL_PRESET_RENDERER_ID)
    assert not renderer._validate_request(request)
    changed = json.loads(request.preset_json)
    changed["settings"].pop("osc_3_spectral_morph_phase")
    with pytest.raises(ValidationError):
        renderer._validate_request(replace(request, preset_json=json.dumps(changed)))


def test_full_readback_accepts_only_documented_asset_migrations(contract):
    document = contract.template()
    document["settings"]["sample"]["samples_stereo"] = document["settings"]["sample"]["samples"]
    loaded = loaded_state(contract, document)
    original = deepcopy(document)
    result = verify_full_nested_state(document, loaded, contract.spec["fixed_native_ramps"])
    assert len(result["nested_families_verified"]) == 6 and result["native_ramp_defaults_verified"] == 128
    assert result["sample_aligned_max_pcm16_delta"] == {"samples": 0, "samples_stereo": 0}
    assert not result["sample_readback_byte_exact"] and document == original


@pytest.mark.parametrize("family", ["lfos", "custom_warps", "random_values", "modulations", "wavetables", "sample", "ramp"])
def test_nested_readback_catches_ignored_or_corrupted_state(contract, family):
    document = contract.template()
    document["settings"]["modulations"][63] = {"source": "lfo_2", "destination": "modulation_1_amount"}
    loaded = loaded_state(contract, document)
    settings = loaded["settings"]
    if family in {"lfos", "custom_warps"}:
        settings[family][0]["powers"][0] += .25
    elif family == "random_values":
        settings[family][0]["seed"] += 1
    elif family == "modulations":
        settings[family][0], settings[family][63] = settings[family][63], settings[family][0]
    elif family == "wavetables":
        settings[family][0]["groups"][0]["components"][0]["keyframes"][0]["position"] += 1
    elif family == "sample":
        settings[family]["sample_rate"] += 1
    else:
        settings["modulation_64_ramp_down"] = 0
    with pytest.raises(RuntimeError):
        verify_full_nested_state(document, loaded, contract.spec["fixed_native_ramps"])


@pytest.mark.parametrize("damage", ["rounding", "shift", "length", "prefix", "invalid_base64"])
def test_sample_alignment_does_not_hide_corruption(contract, damage):
    document = contract.template()
    loaded = loaded_state(contract, document)
    sample = loaded["settings"]["sample"]
    values = np.frombuffer(base64.b64decode(sample["samples"]), dtype="<i2").copy()
    if damage == "rounding":
        values[20] = int(values[20]) + 2
    elif damage == "shift":
        values = np.roll(values, 1)
    elif damage == "length":
        values = values[:-1]
    elif damage == "prefix":
        values[0] = 1
    sample["samples"] = "!" if damage == "invalid_base64" else base64.b64encode(values.tobytes()).decode()
    with pytest.raises(RuntimeError):
        verify_full_nested_state(document, loaded, contract.spec["fixed_native_ramps"])


def test_factory_keeps_legacy_default_and_requires_explicit_full_identity(monkeypatch):
    monkeypatch.setattr("obruxo_data.render.VitalRenderer", lambda **settings: settings)
    assert NativeRendererFactory()()["renderer_id"] == LEGACY_TRAINING_RENDERER_ID
    assert NativeRendererFactory(renderer_id=FULL_PRESET_RENDERER_ID)()["renderer_id"] == FULL_PRESET_RENDERER_ID


def test_broad_correctness_cases_cover_declared_families_with_fixed_phase(contract):
    from obruxo_training.full_check import full_check_cases
    cases = dict(full_check_cases())
    assert len(cases) == 12
    for document in cases.values():
        assert contract.validate(document).valid
        settings = document["settings"]
        assert len(contract.scalar_names & settings.keys()) == 775
        assert all(settings[f"osc_{index}_random_phase"] == 0 for index in range(1, 4))
        assert settings["sample_random_phase"] == 0
    settings = cases["indexed_routes_and_remap"]["settings"]
    assert settings["modulations"][63]["destination"] == "modulation_1_amount"
    assert "line_mapping" in settings["modulations"][0]
    assert cases["sampler_stereo"]["settings"]["sample"]["samples_stereo"] != cases["sampler_stereo"]["settings"]["sample"]["samples"]

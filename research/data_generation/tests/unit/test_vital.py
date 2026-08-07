from __future__ import annotations

from copy import deepcopy
import json

import pytest

from obruxo_data.errors import ValidationError
from obruxo_data.vital import ComponentKind, ComponentProfile, ComponentRef, VitalPreset, VitalSchema
from obruxo_data.vital.profiles import SlotPolicy


def test_init_fixture_is_complete_and_valid() -> None:
    preset = VitalPreset.init()
    assert preset.validate().valid
    assert len(preset.schema.parameters) == 772
    assert len(preset.to_dict()["settings"]["wavetables"]) == 3
    assert len(preset.to_dict()["settings"]["lfos"]) == 8
    assert len(preset.to_dict()["settings"]["modulations"]) == 64


def test_raw_lookup_rejects_unknown_bounds_ordinals_and_non_finite_values() -> None:
    preset = VitalPreset.init()
    with pytest.raises(KeyError):
        preset.set_raw("not_a_control", 0.0)
    with pytest.raises(ValueError, match="between"):
        preset.set_raw("osc_1_level", 1.1)
    with pytest.raises(ValueError, match="integral"):
        preset.set_raw("osc_1_on", 0.5)
    with pytest.raises(ValueError, match="finite"):
        preset.set_raw("osc_1_level", float("nan"))


def test_normalized_conversion_uses_parameter_scale() -> None:
    preset = VitalPreset.init()
    preset.set_normalized("osc_1_level", 0.25)
    assert preset.get_raw("osc_1_level") == pytest.approx(0.5)
    assert preset.get_normalized("osc_1_level") == pytest.approx(0.25)
    preset.set_normalized("osc_1_on", 0.6)
    assert preset.get_raw("osc_1_on") == 1.0


def test_validation_reports_shape_parameter_lfo_and_route_errors() -> None:
    schema = VitalSchema.load()
    document = schema.load_init_document()
    document["settings"].pop("volume")
    document["settings"]["osc_1_destination"] = 14.0
    document["settings"]["lfos"][0]["num_points"] = 101
    document["settings"]["modulations"][0] = {"source": "not_a_source", "destination": "not_a_destination"}
    report = VitalPreset(document, schema).validate()
    assert not report.valid
    assert {item.code for item in report.diagnostics} >= {
        "vital.parameter.missing",
        "vital.parameter.invalid",
        "vital.line.point_count",
        "vital.modulation.source",
        "vital.modulation.destination",
    }


def test_load_does_not_mutate_source_and_save_is_valid(tmp_path) -> None:
    source = tmp_path / "source.vital"
    output = tmp_path / "output.vital"
    original = VitalPreset.init().to_json()
    source.write_text(original, encoding="utf-8")
    preset = VitalPreset.load(source)
    preset.set_raw("osc_1_level", 0.25)
    preset.save(output)
    assert source.read_text(encoding="utf-8") == original
    assert VitalPreset.load(output).get_raw("osc_1_level") == pytest.approx(0.25)
    assert not list(tmp_path.glob("*.tmp"))


def test_component_reset_is_idempotent_and_preserves_unrelated_state() -> None:
    preset = VitalPreset.init()
    preset.set_raw("lfo_1_frequency", 2.0)
    preset.set_raw("osc_2_level", 0.25)
    preset.connect_modulation(1, "lfo_1", "osc_1_level", amount=0.5)
    preset.set_raw("modulation_1_power", 3.0)
    preset.reset_component(ComponentRef(ComponentKind.LFO, 1))
    first = preset.to_json(canonical=True)
    preset.reset_component(ComponentRef(ComponentKind.LFO, 1))
    assert preset.to_json(canonical=True) == first
    assert preset.get_raw("osc_2_level") == pytest.approx(0.25)
    assert preset.to_dict()["settings"]["modulations"][0] == VitalPreset.init().to_dict()["settings"]["modulations"][0]
    for name in preset.schema.parameters:
        if name.startswith("modulation_1_"):
            assert preset.get_raw(name) == VitalPreset.init().get_raw(name)


def test_reset_clears_routes_targeting_component() -> None:
    preset = VitalPreset.init()
    preset.connect_modulation(1, "lfo_1", "osc_1_level")
    preset.reset_component(ComponentRef(ComponentKind.OSCILLATOR, 1))
    assert preset.to_dict()["settings"]["modulations"][0].get("source", "") == ""


def test_profile_allow_does_not_enable_component() -> None:
    preset = VitalPreset.init()
    preset.set_raw("osc_1_on", 0.0)
    preset.apply_profile(ComponentProfile.only(oscillators=[1], lfos=[1]))
    assert preset.get_raw("osc_1_on") == 0.0


def test_profile_resets_disallowed_components_and_bounds_routes() -> None:
    preset = VitalPreset.init()
    preset.set_raw("osc_2_level", 0.25)
    preset.connect_modulation(1, "lfo_1", "osc_1_level")
    preset.connect_modulation(2, "lfo_1", "osc_1_pan")
    preset.apply_profile(ComponentProfile.only(oscillators=[1], lfos=[1], max_active_routes=1))
    init = VitalPreset.init()
    assert preset.get_raw("osc_2_level") == init.get_raw("osc_2_level")
    live = [item for item in preset.to_dict()["settings"]["modulations"] if item.get("source")]
    assert len(live) == 1


def test_profile_failure_rolls_back_atomically() -> None:
    preset = VitalPreset.init()
    preset.set_raw("osc_2_level", 0.25)
    before = preset.to_json(canonical=True)
    profile = ComponentProfile(oscillators=SlotPolicy((1,), reset_disallowed=False, max_active=1))
    with pytest.raises(ValidationError, match="disallowed"):
        preset.apply_profile(profile)
    assert preset.to_json(canonical=True) == before


def test_public_document_is_a_deep_copy() -> None:
    preset = VitalPreset.init()
    external = preset.to_dict()
    external["settings"]["osc_1_level"] = 0.1
    assert preset.get_raw("osc_1_level") != 0.1


def test_schema_hash_detects_fixture_tampering(tmp_path) -> None:
    source = VitalSchema.load()
    manifest = json.loads((source.init_preset_path.parent / "manifest.json").read_text(encoding="utf-8"))
    for name in ("manifest.json", "parameter_inventory.json", "modulation_vocab.json", "reconciliation.json", "init.vital"):
        (tmp_path / name).write_bytes((source.init_preset_path.parent / name).read_bytes())
    document = json.loads((tmp_path / "init.vital").read_text(encoding="utf-8"))
    document["preset_name"] = "tampered"
    (tmp_path / "init.vital").write_text(json.dumps(document), encoding="utf-8")
    assert manifest["init_preset_sha256"]
    with pytest.raises(ValueError, match="hash mismatch"):
        VitalSchema.load(tmp_path).load_init_document()

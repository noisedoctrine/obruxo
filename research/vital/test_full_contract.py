from copy import deepcopy
from pathlib import Path
import random
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from full_contract import FullPresetContract  # noqa: E402
from obruxo_data.errors import ValidationError  # noqa: E402


@pytest.fixture(scope="module")
def contract():
    return FullPresetContract()


def put(document, path, value):
    parent = document
    for part in path[:-1]:
        parent = parent[part]
    parent[path[-1]] = value


def test_full_inventory_and_factory_template(contract):
    document = contract.template()
    assert len(contract.scalar_names) == 775
    assert len(document["settings"]) == 781
    assert contract.validate(document).valid
    assert set(contract.spec["fixed_native_ramps"].values()) == {-10}
    assert not contract.scalar_names & contract.spec["fixed_native_ramps"].keys()
    assert set(contract.parameters) == contract.scalar_names
    assert contract.parameters["osc_1_spectral_morph_type"]["max"] == 16
    assert contract.parameters["osc_1_stack_style"]["max"] == 12


@pytest.mark.parametrize("path,value", [
    (("synth_version",), "1.6.4"),
    (("settings", "osc_1_level"), True),
    (("settings", "osc_1_level"), "0.5"),
    (("settings", "osc_1_level"), float("nan")),
    (("settings", "osc_1_level"), float("inf")),
    (("settings", "osc_1_level"), 10 ** 400),
    (("settings", "osc_1_on"), 0.5),
    (("settings", "osc_1_on"), 2),
    (("settings", "osc_1_spectral_morph_phase"), 1.1),
    (("settings", "modulation_1_ramp_up"), -10),
    (("settings", "modulation_64_ramp_down"), 0),
    (("settings", "unknown_control"), 0),
    (("settings", "random_values", 0, "seed"), -1),
    (("settings", "random_values", 0, "seed"), 2 ** 32),
    (("settings", "random_values", 0, "seed"), 0.5),
    (("settings", "sample", "samples"), "!bad!"),
    (("settings", "sample", "samples"), "AAAA"),
    (("settings", "sample", "sample_rate"), 0),
    (("settings", "sample", "length"), -1),
    (("settings", "sample", "surprise"), 0),
    (("settings", "lfos", 0, "num_points"), 200),
    (("settings", "lfos", 0, "num_points"), 2),
    (("settings", "lfos", 0, "points"), [1, 1, 0, 0, 1, 1]),
    (("settings", "custom_warps", 0, "points"), [-1, 1, 0.5, 0, 1, 1]),
    (("settings", "custom_warps", 0, "powers"), [0, 0]),
    (("settings", "modulations", 0), {"source": "lfo_1", "destination": ""}),
    (("settings", "modulations", 0), {"source": "bad", "destination": "osc_1_level"}),
    (("settings", "modulations", 0), {"source": "lfo_1", "destination": "modulation_1_ramp_up"}),
    (("settings", "wavetables", 0, "version"), "99999.9.9"),
    (("settings", "wavetables", 0, "groups"), []),
    (("settings", "wavetables", 0, "groups", 0, "components", 0, "type"), "Unknown"),
    (("settings", "wavetables", 0, "groups", 0, "components", 0, "interpolation_style"), 3),
    (("settings", "wavetables", 0, "groups", 0, "components", 0, "interpolation"), 0.5),
    (("settings", "wavetables", 0, "groups", 0, "components", 0, "keyframes"), []),
    (("settings", "wavetables", 0, "groups", 0, "components", 0, "keyframes", 0, "wave_data"), "AAAA"),
    (("settings", "wavetables", 0, "groups", 0, "components", 0, "keyframes", 0, "position"), -1),
])
def test_malformed_data_is_rejected_without_publishing(contract, tmp_path, path, value):
    document = contract.template()
    put(document, path, value)
    assert not contract.validate(document).valid
    with pytest.raises(ValidationError):
        contract.save(document, tmp_path / "invalid.vital")
    assert not (tmp_path / "invalid.vital").exists()


@pytest.mark.parametrize("key", ["lfos", "wavetables", "custom_warps", "random_values", "modulations"])
def test_slot_counts_and_missing_nested_fields(contract, key):
    document = contract.template()
    document["settings"][key].pop()
    assert not contract.validate(document).valid
    del document["settings"][key]
    assert not contract.validate(document).valid


@pytest.mark.parametrize("raw", [b'{"settings":{},"settings":{}}', b'{"x":NaN}', b'{"x":Infinity}',
                                  b'{"settings":{"x":1e999}}', b'{"settings":', b'\xff', b'[]'])
def test_strict_file_parsing(contract, tmp_path, raw):
    path = tmp_path / "invalid.vital"
    path.write_bytes(raw)
    assert not contract.read(path)[1].valid


def test_cycles_and_non_json_objects_fail_closed(contract):
    document = contract.template()
    document["loop"] = document
    assert not contract.validate(document).valid
    assert not contract.validate({1: object(), "settings": {}}).valid


def test_decoder_determinism_and_restricted_output(contract, tmp_path):
    rng = random.Random(155)
    for index in range(24):
        controls = {"osc_1_transpose": rng.choice([-12, 0, 12]), "osc_1_level": rng.uniform(0.2, 0.8),
                    "osc_1_spectral_morph_phase": rng.random()}
        document = contract.decode_scalars(controls)
        assert document == contract.decode_scalars(controls)
        destination = tmp_path / f"{index}.vital"
        contract.save(document, destination)
        loaded, report = contract.read(destination)
        assert report.valid and loaded == document
        with pytest.raises(FileExistsError):
            contract.save(document, destination)
    for blocked in ("modulation_1_ramp_up", "modulation_64_ramp_down", "wavetables", "unknown"):
        with pytest.raises(ValueError):
            contract.decode_scalars({blocked: -10})


def test_range_rejection_is_not_structural_rejection(contract):
    document = contract.template()
    document["settings"]["osc_1_level"] = 10
    assert contract.validate(document, authoring=False).valid
    assert not contract.validate(document).valid


def test_new_phase_destination_and_remap(contract):
    document = contract.template()
    document["settings"]["modulations"][0] = {"source": "lfo_1", "destination": "osc_1_spectral_morph_phase",
                                                "line_mapping": deepcopy(document["settings"]["lfos"][0])}
    assert contract.validate(document).valid


def test_source_defined_component_absent_from_corpus(contract):
    document = contract.template()
    document["settings"]["wavetables"][0]["groups"][0]["components"][0]["type"] = "Shepard Tone Source"
    assert contract.validate(document).valid


def test_newer_enum_values_use_verified_renderer_bounds(contract):
    document = contract.decode_scalars({"osc_1_spectral_morph_type": 16, "osc_1_stack_style": 12, "lfo_1_sync_type": 6})
    assert contract.validate(document).valid
    with pytest.raises(ValidationError):
        contract.decode_scalars({"osc_1_spectral_morph_type": 17})


def test_all_required_scalar_keys_are_enforced(contract):
    # Removing a family, not just one example control, must never slip through.
    document = contract.template()
    for key in list(document["settings"]):
        if key.startswith("env_"):
            del document["settings"][key]
    report = contract.validate(document)
    assert not report.valid
    assert sum(item.code == "contract.scalar" for item in report.diagnostics) == 54

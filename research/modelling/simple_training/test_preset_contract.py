from argparse import Namespace
import json
import os
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import PresetCodec, validate_preset  # noqa: E402
from preset_contract import PresetContract, strict_json  # noqa: E402
from obruxo_data.errors import ValidationError  # noqa: E402


@pytest.fixture(scope="module")
def codec():
    return PresetCodec()


@pytest.mark.parametrize("path,value", [
    (("synth_version",), "99999.9.9"),
    (("author",), 123),
    (("unexpected",), "extra"),
    (("settings", "osc_1_level"), 0.81),
    (("settings", "osc_1_level"), 0.19),
    (("settings", "osc_1_level"), float("nan")),
    (("settings", "osc_1_level"), float("inf")),
    (("settings", "osc_1_level"), 10 ** 400),
    (("settings", "osc_1_level"), "0.5"),
    (("settings", "osc_1_transpose"), 6),
    (("settings", "osc_1_transpose"), 0.5),
    (("settings", "osc_1_transpose"), False),
    (("settings", "osc_1_on"), True),
    (("settings", "reverb_on"), 1),
    (("settings", "osc_2_on"), 1),
    (("settings", "unknown_parameter"), 0),
    (("settings", "sample", "samples"), "broken base64"),
    (("settings", "wavetables", 0), {}),
    (("settings", "wavetables", 0, "version"), "1.7.0"),
    (("settings", "lfos"), []),
    (("settings", "lfos", 0, "num_points"), 999),
    (("settings", "modulations"), []),
    (("settings", "modulations", 0), {"source": "lfo_1", "destination": "osc_1_level"}),
])
def test_contract_rejects_malformed_and_out_of_scope_state(codec, path, value):
    document = codec.preset(1, 0.5).to_dict()
    parent = document
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    report = codec.output_contract.validate(document)
    assert not report.valid
    assert report.diagnostics[0].pointer is not None


@pytest.mark.parametrize("path", [("author",), ("settings", "osc_1_level"), ("settings", "wavetables")])
def test_contract_rejects_missing_fields(codec, path):
    document = codec.template.to_dict()
    parent = document
    for key in path[:-1]:
        parent = parent[key]
    del parent[path[-1]]
    assert not codec.output_contract.validate(document).valid


@pytest.mark.parametrize("payload", [
    b'{"settings": {}, "settings": {}}', b'{"x": NaN}', b'{"x": Infinity}', b'{"x": -Infinity}',
    b'{"x":', b'[]', b'null', b'123', b'false', b'\xff', b'[' * 2000 + b']' * 2000,
])
def test_strict_file_reader_rejects_bad_json(codec, tmp_path, payload):
    path = tmp_path / "invalid.vital"
    path.write_bytes(payload)
    _, report = codec.output_contract.read(path)
    assert not report.valid


def test_file_size_limit_and_numeric_overflow(codec, tmp_path):
    path = tmp_path / "invalid.vital"
    path.write_bytes(b" " * (codec.output_contract.spec["max_file_bytes"] + 1))
    assert not codec.output_contract.read(path)[1].valid
    text = codec.preset(1, 0.5).to_json().replace('"osc_1_level": 0.5', '"osc_1_level": 1e999')
    path.write_text(text, encoding="utf-8")
    assert not codec.output_contract.read(path)[1].valid


def test_random_and_extreme_predictions_remain_in_contract(codec):
    generator = torch.Generator().manual_seed(42)
    for scale in (1e-30, 1, 100, 1e6, 1e30, 1e300):
        predictions = torch.randn(64, 4, generator=generator, dtype=torch.float64) * scale
        for dense in predictions:
            preset = codec.decode(dense)
            document = strict_json(preset.to_json())
            assert codec.output_contract.validate(document).valid
            assert document["settings"]["osc_1_transpose"] in (-12, 0, 12)
            assert 0.2 <= document["settings"]["osc_1_level"] <= 0.8


@pytest.mark.parametrize("dense", [torch.zeros(3), torch.zeros(1, 4), torch.ones(4, dtype=torch.bool),
                                   torch.ones(4, dtype=torch.complex64),
                                   torch.tensor([0., 0., 0., float("nan")]), torch.tensor([float("inf"), 0., 0., 0.])])
def test_invalid_predictions_never_publish(codec, tmp_path, dense):
    destination = tmp_path / "bad.vital"
    with pytest.raises(ValueError):
        codec.save(codec.decode(dense), destination)
    assert not destination.exists()


def test_revalidates_serialized_bytes_before_publish(codec, tmp_path, monkeypatch):
    preset = codec.preset(1, 0.5)
    corrupt = preset.to_dict()
    corrupt["settings"]["sample"]["samples"] = "corrupt"
    monkeypatch.setattr(preset, "to_json", lambda: json.dumps(corrupt))
    with pytest.raises(ValidationError):
        codec.save(preset, tmp_path / "bad.vital")
    assert list(tmp_path.iterdir()) == []


def test_atomic_export_no_overwrite_and_cli_validation(codec, tmp_path, capsys):
    path = tmp_path / "valid.vital"
    codec.save(codec.preset(0, 0.5), path)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        codec.save(codec.preset(2, 1), path)
    assert path.read_bytes() == original
    validate_preset(Namespace(preset=str(path), runtime=False, render=False, plugin_path=None))
    result = json.loads(capsys.readouterr().out)
    assert result["valid"] and result["static_valid"] and not result["runtime_checked"]
    path.write_text('{"settings":{},"settings":{}}', encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        validate_preset(Namespace(preset=str(path), runtime=False, render=False, plugin_path=None))
    assert error.value.code == 1
    assert not json.loads(capsys.readouterr().out)["valid"]


def test_tampered_codec_template_cannot_escape_contract(tmp_path):
    codec = PresetCodec()
    codec.template.set_raw("reverb_on", 1)
    with pytest.raises(ValidationError):
        codec.save(codec.decode(torch.zeros(4)), tmp_path / "bad.vital")
    assert not list(tmp_path.iterdir())


def test_contract_is_independent_and_accepts_equivalent_json_numbers(codec):
    document = codec.preset(1, 0.5).to_dict()
    document["settings"]["osc_1_on"] = 1  # JSON 1 and 1.0 are both numeric, but true is not.
    assert PresetContract().validate(document).valid


@pytest.mark.skipif(os.environ.get("OBRUXO_VALIDATE_NATIVE") != "1", reason="opt-in official Vital boundary sweep")
@pytest.mark.parametrize("category", range(3))
@pytest.mark.parametrize("level_logit", [-1000., 0., 1000.])
def test_native_decoder_boundaries(codec, tmp_path, capsys, category, level_logit):
    dense = torch.full((4,), -1000.)
    dense[category], dense[3] = 1000., level_logit
    path = tmp_path / "boundary.vital"
    codec.save(codec.decode(dense), path)
    validate_preset(Namespace(preset=str(path), runtime=True, render=True, plugin_path=None))
    result = json.loads(capsys.readouterr().out)
    assert result["valid"] and result["runtime_checked"] and result["render_checked"]

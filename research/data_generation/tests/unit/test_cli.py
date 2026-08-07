from __future__ import annotations

import json

from obruxo_data.cli import main
from obruxo_data.midi import Performance
from obruxo_data.vital import VitalPreset
from obruxo_data.vital.probe import probe_schema


def test_vital_cli_init_set_validate_and_no_overwrite(tmp_path, capsys) -> None:
    init = tmp_path / "init.vital"
    edited = tmp_path / "edited.vital"
    assert main(["vital", "init", "--output", str(init)]) == 0
    assert main(["vital", "init", "--output", str(init)]) == 2
    assert "refusing to overwrite" in capsys.readouterr().err
    assert main([
        "vital", "set", str(init), "--raw", "osc_1_level=0.25", "--output", str(edited),
    ]) == 0
    assert VitalPreset.load(edited).get_raw("osc_1_level") == 0.25
    assert main(["vital", "validate", str(edited), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["valid"]


def test_midi_cli_create_validate_and_explicit_end(tmp_path, capsys) -> None:
    output = tmp_path / "note.mid"
    assert main([
        "midi", "create-note", "--pitch", "60", "--velocity", "100", "--beats", "2",
        "--end-beats", "4", "--output", str(output),
    ]) == 0
    performance = Performance.from_midi(output)
    assert performance.end_tick == 1920
    assert performance.note_spans()[0].duration_ticks == 960
    assert main(["midi", "validate", str(output), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["valid"]


def test_schema_probe_refuses_overwrite_before_importing_runtime(tmp_path) -> None:
    output = tmp_path / "schema"
    output.mkdir()
    (output / "reviewed.txt").write_text("keep", encoding="utf-8")
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    try:
        probe_schema(output, source_atlas_path=source)
    except FileExistsError as error:
        assert "refusing to overwrite" in str(error)
    else:
        raise AssertionError("schema probe overwrote a reviewed bundle")

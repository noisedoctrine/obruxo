import json
import os
from threading import Thread
from time import sleep

import pytest

from obruxo_training import atomic_io
from obruxo_training.atomic_io import read_json_snapshot, replace_file
from obruxo_training.audit import write_json


def test_snapshot_reader_closes_before_parsing_and_keeps_its_old_snapshot(tmp_path, monkeypatch):
    path = tmp_path / "result.json"
    write_json(path, {"step": 1})
    original_loads = json.loads
    def parse_after_replace(text):
        write_json(path, {"step": 2})
        return original_loads(text)
    monkeypatch.setattr(atomic_io.json, "loads", parse_after_replace)
    assert read_json_snapshot(path) == {"step": 1}
    assert original_loads(path.read_text()) == {"step": 2}


@pytest.mark.skipif(os.name != "nt", reason="Windows delete-sharing behavior")
def test_atomic_json_publication_recovers_when_an_ordinary_reader_closes(tmp_path):
    path = tmp_path / "result.json"
    write_json(path, {"step": 1})
    reader = path.open()
    partial = path.with_suffix(".json.partial")
    partial.write_text("{}")
    try:
        with pytest.raises(PermissionError):
            os.replace(partial, path)
        thread = Thread(target=lambda: (sleep(.08), reader.close()))
        thread.start()
        try:
            write_json(path, {"step": 2})
        finally:
            thread.join()
    finally:
        reader.close()
    assert read_json_snapshot(path) == {"step": 2}
    assert not partial.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows delete-sharing behavior")
def test_permanent_lock_is_bounded_and_preserves_both_versions(tmp_path):
    path, partial = tmp_path / "result.json", tmp_path / "result.json.partial"
    path.write_text("old")
    partial.write_text("new")
    with path.open():
        with pytest.raises(PermissionError):
            replace_file(partial, path, timeout=.05)
    assert path.read_text() == "old"
    assert partial.read_text() == "new"

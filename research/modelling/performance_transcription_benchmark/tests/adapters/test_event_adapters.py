from __future__ import annotations

from pathlib import Path

import pytest
from obruxo_performance_benchmark.adapters.muscriptor import (
    batch_status,
    normalize_timing_corrected_events,
    stock_decoding_config,
)
from obruxo_performance_benchmark.adapters.yourmt3 import (
    normalize_note_events,
    stock_inference_config,
    write_temporary_midi,
)


def test_yourmt3_event_normalization_ignores_progress() -> None:
    events = [
        {"type": "ProgressEvent", "completed": 0, "total": 1},
        {"type": "NoteStartEvent", "pitch": 60, "start_time": 0.25, "index": 4, "instrument": "strings"},
        {"type": "NoteEndEvent", "end_time": 0.75, "start_event_index": 4},
    ]
    notes = normalize_note_events(events)
    assert len(notes) == 1
    assert notes[0].onset_seconds == 0.25
    assert notes[0].offset_seconds == 0.75
    assert notes[0].instrument_or_program == "strings"


def test_stock_settings_cannot_be_overridden(tmp_path: Path) -> None:
    assert stock_inference_config("ymt3_plus")["deterministic"]
    with pytest.raises(ValueError):
        stock_inference_config("ymt3_plus", {"precision": "float16"})
    payload = write_temporary_midi(b"MThd", tmp_path)
    assert payload.parent == tmp_path.resolve()
    assert payload.read_bytes() == b"MThd"


def test_muscriptor_timing_path_drops_serialization_velocity() -> None:
    notes = normalize_timing_corrected_events([
        {"onset_seconds": 0.0, "offset_seconds": 0.5, "midi_pitch": 60, "velocity": 127, "instrument": 1}
    ])
    assert stock_decoding_config()["prelude_forcing"] is True
    assert notes[0].velocity_midi is None
    assert batch_status(4)["status"] == "not_applicable"

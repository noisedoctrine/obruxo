from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from obruxo_performance_benchmark.adapters.muscriptor import (
    MuScriptorAdapter,
    batch_status,
    normalize_timing_corrected_events,
    stock_decoding_config,
)
from obruxo_performance_benchmark.adapters.timbre_trap import (
    TimbreTrapAdapter,
    normalize_frame_output,
)
from obruxo_performance_benchmark.adapters.yourmt3 import (
    YourMT3Adapter,
    normalize_note_events,
    stock_inference_config,
    write_temporary_midi,
)
from obruxo_performance_benchmark.types import (
    NormalizedNote,
    common_frame_count,
    common_frame_times,
    rasterize_notes,
)
from scipy.io import wavfile


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


def test_all_adapters_use_the_canonical_basic_pitch_frame_grid(tmp_path: Path) -> None:
    sample_count = 22_050 * 10
    audio = tmp_path / "ten-seconds.wav"
    wavfile.write(audio, 22_050, np.zeros(sample_count, dtype=np.float32))

    frame_count = common_frame_count(sample_count)
    assert frame_count == 860
    assert TimbreTrapAdapter._frame_count(audio) == frame_count
    assert YourMT3Adapter._frame_count(audio) == frame_count
    assert MuScriptorAdapter._frame_count(audio) == frame_count

    times = common_frame_times(frame_count)
    timbre = normalize_frame_output(
        np.asarray([0.0]),
        np.asarray([440.0]),
        np.asarray([[0.9]]),
        times,
    )
    event = rasterize_notes((NormalizedNote(0.0, 1.0, 69),), frame_count)
    assert timbre.frame_pitch is not None
    assert event.times_seconds.shape == (frame_count,)
    assert timbre.frame_pitch.active_midi.shape == event.active_midi.shape == (
        frame_count,
        88,
    )
    np.testing.assert_array_equal(timbre.frame_pitch.times_seconds, times)
    np.testing.assert_array_equal(event.times_seconds, times)

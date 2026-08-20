from __future__ import annotations

import numpy as np
from obruxo_basic_pitch.constants import (
    FRAME_THRESHOLD,
    MIN_NOTE_LENGTH_FRAMES,
    ONSET_THRESHOLD,
)
from obruxo_basic_pitch.postprocess import (
    model_frames_to_time,
    posteriorgrams_to_note_events,
)


def test_stock_defaults_and_known_note_event() -> None:
    assert ONSET_THRESHOLD == 0.5
    assert FRAME_THRESHOLD == 0.3
    assert MIN_NOTE_LENGTH_FRAMES == 11
    note = np.zeros((172, 88), dtype=np.float32)
    onset = np.zeros((172, 88), dtype=np.float32)
    contour = np.zeros((172, 264), dtype=np.float32)
    onset[10, 20] = 1.0
    note[10:80, 20] = 0.8
    events = posteriorgrams_to_note_events(
        {"note": note, "onset": onset, "contour": contour}
    )
    assert len(events) == 1
    assert events[0].pitch_midi == 41
    assert events[0].start_time_s == model_frames_to_time(172)[10]
    assert events[0].end_time_s == model_frames_to_time(172)[80]
    assert len(events[0].pitch_bend or ()) == 70

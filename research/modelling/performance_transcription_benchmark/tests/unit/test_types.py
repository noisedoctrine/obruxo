from __future__ import annotations

import numpy as np
import pytest
from obruxo_basic_pitch.postprocess import model_frames_to_time
from obruxo_performance_benchmark.types import (
    NormalizedNote,
    common_frame_times,
    rasterize_notes,
)


def test_common_frame_rasterization_is_half_open() -> None:
    times = model_frames_to_time(3)
    note = NormalizedNote(float(times[0]), float(times[1]), 60)
    prediction = rasterize_notes([note], 3)
    assert np.array_equal(prediction.times_seconds, common_frame_times(3))
    assert prediction.active_midi[0, 60 - 21]
    assert not prediction.active_midi[1, 60 - 21]


def test_normalized_note_rejects_invalid_interval() -> None:
    with pytest.raises(ValueError):
        NormalizedNote(1.0, 1.0, 60)
    with pytest.raises(ValueError):
        NormalizedNote(0.0, 1.0, 128)

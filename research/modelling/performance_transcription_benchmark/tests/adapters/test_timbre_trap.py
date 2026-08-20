from __future__ import annotations

import numpy as np
from obruxo_performance_benchmark.adapters.timbre_trap import (
    normalize_frame_output,
    peak_pick_then_threshold,
)


def test_peak_picking_precedes_fixed_threshold() -> None:
    activations = np.asarray([[0.7, 0.6, 0.1], [0.4, 0.6, 0.4]], dtype=float)
    active = peak_pick_then_threshold(activations)
    assert active.tolist() == [[True, False, False], [False, True, False]]


def test_common_mapping_uses_earlier_frame_on_tie_and_no_notes() -> None:
    centers = np.asarray([440.0, 466.1637615])
    output = normalize_frame_output(
        np.asarray([0.0, 1.0]),
        centers,
        np.asarray([[0.9, 0.1], [0.1, 0.9]]),
        np.asarray([0.5]),
    )
    assert output.notes is None
    assert output.frame_pitch is not None
    assert output.frame_pitch.active_midi[0, 69 - 21]
    assert not output.frame_pitch.active_midi[0, 70 - 21]

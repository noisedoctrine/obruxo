from __future__ import annotations

import numpy as np
from obruxo_basic_pitch.evaluation.aggregate import aggregate_results
from obruxo_basic_pitch.evaluation.labels import ReferenceNote
from obruxo_basic_pitch.evaluation.metrics import (
    frame_metrics,
    frame_target,
    match_note_indices,
    note_metrics,
)
from obruxo_basic_pitch.postprocess import NoteEvent, model_frames_to_time


def _reference(
    *, pitch: int = 60, onset: float = 0.1, offset: float = 0.5, velocity: int = 80
) -> list[ReferenceNote]:
    return [ReferenceNote(onset, offset, pitch, velocity)]


def _prediction(
    *,
    pitch: int = 60,
    onset: float = 0.1,
    offset: float = 0.5,
    amplitude: float = 80 / 127,
) -> list[NoteEvent]:
    return [NoteEvent(onset, offset, pitch, amplitude)]


def test_exact_event_and_frame_metrics_use_fixed_stock_contract() -> None:
    reference = _reference()
    predicted = _prediction()
    notes = note_metrics(reference, predicted)
    assert notes["onset_pitch"]["f1"] == 1.0
    assert notes["onset_pitch_offset"]["f1"] == 1.0
    posterior = frame_target([ReferenceNote(0.0, 1.0, 60, 80)], 172).astype(np.float32)
    frames = frame_metrics([ReferenceNote(0.0, 1.0, 60, 80)], posterior)
    assert frames["f1"] == 1.0
    assert np.array_equal(frame_target(reference, 172), frame_target(reference, 172))


def test_onset_and_offset_tolerances_are_not_loosened() -> None:
    reference = _reference(onset=0.1, offset=0.2)
    inside = _prediction(onset=0.149, offset=0.249)
    outside = _prediction(onset=0.151, offset=0.251)
    assert match_note_indices(reference, inside, include_offsets=False) == [(0, 0)]
    assert match_note_indices(reference, outside, include_offsets=False) == []
    assert match_note_indices(reference, inside, include_offsets=True) == [(0, 0)]
    beyond_short_offset = _prediction(onset=0.1, offset=0.251)
    assert (
        match_note_indices(reference, beyond_short_offset, include_offsets=True) == []
    )
    wrong_offset = note_metrics(reference, _prediction(offset=0.28))
    assert wrong_offset["onset_pitch"]["f1"] == 1.0
    assert wrong_offset["onset_pitch_offset"]["f1"] == 0.0
    assert wrong_offset["timing_diagnostics"]["offset"]["support"] == 1


def test_count_bias_out_of_range_velocity_and_pitch_confusion() -> None:
    reference = _reference(pitch=10, velocity=50) + [ReferenceNote(0.6, 1.0, 60, 100)]
    predicted = [NoteEvent(0.6, 1.0, 72, 100 / 127)]
    result = note_metrics(reference, predicted)
    assert result["onset_pitch"]["count_bias"] == -1
    assert result["out_of_range_reference_notes"] == 1
    assert result["velocity"]["pearson"] is None
    assert result["pitch_confusion"]["octave_error_count"] == 1

    constant_velocity = note_metrics(
        [ReferenceNote(0.1, 0.5, 60, 80), ReferenceNote(0.6, 1.0, 62, 80)],
        [NoteEvent(0.1, 0.5, 60, 80 / 127), NoteEvent(0.6, 1.0, 62, 80 / 127)],
    )
    assert constant_velocity["velocity"]["pearson"] is None
    assert constant_velocity["velocity"]["pearson_status"] == "constant_series"

    ambiguous = note_metrics(
        [ReferenceNote(0.1, 0.5, 60, 80), ReferenceNote(0.1, 0.5, 64, 80)],
        [NoteEvent(0.1, 0.5, 72, 80 / 127), NoteEvent(0.1, 0.5, 76, 80 / 127)],
    )
    assert ambiguous["pitch_confusion"]["assigned_count"] == 0
    assert ambiguous["pitch_confusion"]["ambiguous_reference_count"] == 2
    assert ambiguous["pitch_confusion"]["unassigned_reference_count"] == 2
    assert ambiguous["pitch_confusion"]["unassigned_prediction_count"] == 2
    assert ambiguous["pitch_confusion"]["unassigned_error_count"] == 4

    missing_prediction = note_metrics(_reference(), [])
    assert missing_prediction["pitch_confusion"]["unassigned_error_count"] == 1
    extra_prediction = note_metrics([], _prediction())
    assert extra_prediction["pitch_confusion"]["unassigned_error_count"] == 1


def test_micro_aggregation_is_count_based_and_bootstrap_is_deterministic() -> None:
    rows = [
        {
            "pair_id": "pair-a",
            "preset_id": "preset-a",
            "labels": {"polyphony_class": "monophonic"},
            "status": "ok",
            "metrics": {
                "onset_pitch": {
                    "reference_count": 1,
                    "prediction_count": 1,
                    "tp": 1,
                    "f1": 1.0,
                },
                "onset_pitch_offset": {
                    "reference_count": 1,
                    "prediction_count": 1,
                    "tp": 1,
                    "f1": 1.0,
                },
                "frames": {
                    "reference_count": 1,
                    "prediction_count": 1,
                    "tp": 1,
                    "f1": 1.0,
                },
            },
        },
        {
            "pair_id": "pair-b",
            "preset_id": None,
            "labels": {"polyphony_class": "polyphonic"},
            "status": "ok",
            "metrics": {
                "onset_pitch": {
                    "reference_count": 9,
                    "prediction_count": 0,
                    "tp": 0,
                    "f1": None,
                },
                "onset_pitch_offset": {
                    "reference_count": 9,
                    "prediction_count": 0,
                    "tp": 0,
                    "f1": None,
                },
                "frames": {
                    "reference_count": 9,
                    "prediction_count": 0,
                    "tp": 0,
                    "f1": None,
                },
            },
        },
    ]
    first = aggregate_results(rows, bootstrap_replicates=100, seed=0)
    second = aggregate_results(rows, bootstrap_replicates=100, seed=0)
    assert first == second
    assert first["micro"]["onset_pitch"]["f1"] == 2 / 11
    assert first["pair_macro"]["onset_pitch"]["f1"] == 1.0
    assert first["bootstrap"]["replicates"] == 100


def test_frame_target_uses_model_frame_times() -> None:
    times = model_frames_to_time(20)
    reference = [ReferenceNote(float(times[3]), float(times[7]), 60, 80)]
    target = frame_target(reference, 20)
    assert target[3, 39]
    assert not target[7, 39]

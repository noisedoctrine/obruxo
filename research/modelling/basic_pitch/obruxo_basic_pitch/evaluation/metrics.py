"""Pure transcription, frame, timing, and velocity metrics for Basic Pitch."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import mir_eval
import numpy as np

from ..constants import FRAME_THRESHOLD
from ..postprocess import NoteEvent, model_frames_to_time
from .labels import ReferenceNote

MIDI_LOW = 21
MIDI_HIGH = 108
ONSET_TOLERANCE_SECONDS = 0.05
PITCH_TOLERANCE_CENTS = 50.0
OFFSET_RATIO = 0.2
OFFSET_MIN_TOLERANCE_SECONDS = 0.05


def _midi_to_hz(pitch: int) -> float:
    return 440.0 * 2.0 ** ((pitch - 69.0) / 12.0)


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _counts(reference_count: int, prediction_count: int, true_positive: int) -> dict[str, Any]:
    false_positive = prediction_count - true_positive
    false_negative = reference_count - true_positive
    precision = _safe_ratio(true_positive, prediction_count)
    recall = _safe_ratio(true_positive, reference_count)
    if reference_count == 0 and prediction_count == 0:
        f1 = None
    elif precision is None or recall is None:
        f1 = 0.0
    else:
        f1 = 0.0 if precision + recall == 0 else float(2 * precision * recall / (precision + recall))
    return {
        "reference_count": reference_count,
        "prediction_count": prediction_count,
        "count_bias": prediction_count - reference_count,
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_negative_rate": _safe_ratio(false_negative, reference_count),
        "false_positive_fraction": _safe_ratio(false_positive, prediction_count) if prediction_count else None,
    }


def _note_arrays(notes: Sequence[ReferenceNote | NoteEvent]) -> tuple[np.ndarray, np.ndarray]:
    intervals = np.asarray(
        [
            (note.start_time_s, note.end_time_s) if isinstance(note, NoteEvent) else (note.onset_s, note.offset_s)
            for note in notes
        ],
        dtype=np.float64,
    ).reshape(-1, 2)
    pitches = np.asarray([_midi_to_hz(note.pitch_midi) for note in notes], dtype=np.float64)
    return intervals, pitches


def match_note_indices(
    reference: Sequence[ReferenceNote],
    predicted: Sequence[NoteEvent],
    *,
    include_offsets: bool,
) -> list[tuple[int, int]]:
    """Return `mir_eval` reference/prediction matches under the fixed contract."""
    if not reference or not predicted:
        return []
    ref_intervals, ref_pitches = _note_arrays(reference)
    est_intervals, est_pitches = _note_arrays(predicted)
    return list(
        mir_eval.transcription.match_notes(
            ref_intervals,
            ref_pitches,
            est_intervals,
            est_pitches,
            onset_tolerance=ONSET_TOLERANCE_SECONDS,
            pitch_tolerance=PITCH_TOLERANCE_CENTS,
            offset_ratio=OFFSET_RATIO if include_offsets else None,
            offset_min_tolerance=OFFSET_MIN_TOLERANCE_SECONDS,
        )
    )


def _error_summary(errors: Sequence[float], *, suffix: str = "") -> dict[str, Any]:
    values = np.asarray(errors, dtype=np.float64)
    if values.size == 0:
        return {
            "support": 0,
            "signed_mean": None,
            "mae": None,
            "median_abs": None,
            "p90_abs": None,
            "p95_abs": None,
        }
    absolute = np.abs(values)
    return {
        "support": int(values.size),
        "signed_mean": float(np.mean(values)),
        "mae": float(np.mean(absolute)),
        "median_abs": float(np.median(absolute)),
        "p90_abs": float(np.percentile(absolute, 90)),
        "p95_abs": float(np.percentile(absolute, 95)),
    }


def _velocity_summary(reference: Sequence[ReferenceNote], predicted: Sequence[NoteEvent], matches: Sequence[tuple[int, int]]) -> dict[str, Any]:
    errors = [round(127 * predicted[est].amplitude) - reference[ref].velocity for ref, est in matches]
    summary = _error_summary(errors)
    summary["pearson"] = None
    summary["pearson_status"] = "insufficient_support"
    if len(errors) >= 2:
        reference_values = np.asarray([reference[ref].velocity for ref, _ in matches], dtype=np.float64)
        predicted_values = np.asarray([round(127 * predicted[est].amplitude) for _, est in matches], dtype=np.float64)
        if np.ptp(reference_values) != 0 and np.ptp(predicted_values) != 0:
            summary["pearson"] = float(np.corrcoef(reference_values, predicted_values)[0, 1])
            summary["pearson_status"] = "ok"
        else:
            summary["pearson_status"] = "constant_series"
    return summary


def _pitch_confusion(
    reference: Sequence[ReferenceNote],
    predicted: Sequence[NoteEvent],
    matches: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    matched_reference = {ref for ref, _ in matches}
    matched_prediction = {est for _, est in matches}
    candidate_by_reference: dict[int, list[int]] = {}
    candidate_by_prediction: dict[int, list[int]] = {}
    for ref_index, ref in enumerate(reference):
        if ref_index in matched_reference:
            continue
        candidates = [
            est_index
            for est_index, est in enumerate(predicted)
            if est_index not in matched_prediction and abs(est.start_time_s - ref.onset_s) <= ONSET_TOLERANCE_SECONDS
        ]
        if candidates:
            candidate_by_reference[ref_index] = candidates
            for est_index in candidates:
                candidate_by_prediction.setdefault(est_index, []).append(ref_index)
    assigned: list[tuple[int, int]] = []
    for ref_index, candidates in candidate_by_reference.items():
        if len(candidates) == 1 and len(candidate_by_prediction[candidates[0]]) == 1:
            assigned.append((ref_index, candidates[0]))
    deltas = [predicted[est].pitch_midi - reference[ref].pitch_midi for ref, est in assigned]
    octave_errors = sum(delta != 0 and delta % 12 == 0 for delta in deltas)
    ambiguous_reference_count = sum(
        1 for ref_index, candidates in candidate_by_reference.items() if len(candidates) != 1 or len(candidate_by_prediction[candidates[0]]) != 1
    )
    assigned_reference = {ref for ref, _ in assigned}
    assigned_prediction = {est for _, est in assigned}
    unassigned_reference_count = len(reference) - len(matched_reference) - len(assigned_reference)
    unassigned_prediction_count = len(predicted) - len(matched_prediction) - len(assigned_prediction)
    return {
        "assigned_count": len(assigned),
        "signed_semitone_deltas": deltas,
        "octave_error_count": octave_errors,
        "ambiguous_reference_count": ambiguous_reference_count,
        "unassigned_reference_count": unassigned_reference_count,
        "unassigned_prediction_count": unassigned_prediction_count,
        "unassigned_error_count": unassigned_reference_count + unassigned_prediction_count,
        "candidate_reference_count": len(candidate_by_reference),
    }


def note_metrics(reference: Sequence[ReferenceNote], predicted: Sequence[NoteEvent]) -> dict[str, Any]:
    """Compute the fixed event-level metrics without loading models or writing files."""
    onset_pitch_matches = match_note_indices(reference, predicted, include_offsets=False)
    offset_matches = match_note_indices(reference, predicted, include_offsets=True)
    timing = {
        "onset": _error_summary([predicted[est].start_time_s - reference[ref].onset_s for ref, est in onset_pitch_matches]),
        "offset": _error_summary([predicted[est].end_time_s - reference[ref].offset_s for ref, est in onset_pitch_matches]),
        "duration": _error_summary(
            [
                (predicted[est].end_time_s - predicted[est].start_time_s)
                - (reference[ref].offset_s - reference[ref].onset_s)
                for ref, est in onset_pitch_matches
            ]
        ),
    }
    return {
        "onset_pitch": _counts(len(reference), len(predicted), len(onset_pitch_matches)),
        "onset_pitch_offset": _counts(len(reference), len(predicted), len(offset_matches)),
        "out_of_range_reference_notes": sum(not MIDI_LOW <= note.pitch_midi <= MIDI_HIGH for note in reference),
        "timing_diagnostics": timing,
        "velocity": _velocity_summary(reference, predicted, onset_pitch_matches),
        "pitch_confusion": _pitch_confusion(reference, predicted, onset_pitch_matches),
    }


def _distribution(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {"support": 0, "min": None, "median": None, "p90": None, "p95": None, "max": None}
    return {
        "support": int(values.size),
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def frame_target(reference: Sequence[ReferenceNote], n_frames: int) -> np.ndarray:
    """Build the 88-bin half-open target at #23's exact model frame times."""
    if n_frames < 0:
        raise ValueError("frame count must be non-negative")
    times = model_frames_to_time(n_frames)
    target = np.zeros((n_frames, MIDI_HIGH - MIDI_LOW + 1), dtype=bool)
    for note in reference:
        if not MIDI_LOW <= note.pitch_midi <= MIDI_HIGH:
            continue
        target[:, note.pitch_midi - MIDI_LOW] |= (times >= note.onset_s) & (times < note.offset_s)
    return target


def frame_metrics(reference: Sequence[ReferenceNote], note_posterior: np.ndarray) -> dict[str, Any]:
    """Score the stock note posterior at its native model-frame timestamps."""
    posterior = np.asarray(note_posterior, dtype=np.float64)
    if posterior.ndim != 2 or posterior.shape[1] != MIDI_HIGH - MIDI_LOW + 1:
        raise ValueError("note posterior must have shape [frames, 88]")
    target = frame_target(reference, posterior.shape[0])
    prediction = posterior >= FRAME_THRESHOLD
    true_positive = int(np.count_nonzero(target & prediction))
    result = _counts(int(np.count_nonzero(target)), int(np.count_nonzero(prediction)), true_positive)
    positive_values = posterior[target]
    negative_values = posterior[~target]
    times = model_frames_to_time(posterior.shape[0])
    onset_values = []
    for note in reference:
        if MIDI_LOW <= note.pitch_midi <= MIDI_HIGH and len(times):
            index = int(np.argmin(np.abs(times - note.onset_s)))
            onset_values.append(posterior[index, note.pitch_midi - MIDI_LOW])
    result.update(
        {
            "positive_note_posterior": _distribution(positive_values),
            "negative_note_posterior": _distribution(negative_values),
            "reference_onset_posterior": _distribution(np.asarray(onset_values, dtype=np.float64)),
        }
    )
    return result


def evaluate_notes_and_frames(
    reference: Sequence[ReferenceNote],
    predicted: Sequence[NoteEvent],
    note_posterior: np.ndarray,
) -> dict[str, Any]:
    """Combine event and frame metrics for one successful pair."""
    return {"notes": note_metrics(reference, predicted), "frames": frame_metrics(reference, note_posterior)}

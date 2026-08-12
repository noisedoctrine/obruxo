"""Small, validated result types shared by the explicit model adapters."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

MIDI_LOW = 21
MIDI_HIGH = 108
PITCH_COUNT = MIDI_HIGH - MIDI_LOW + 1

PAIR_STATUSES = ("ok", "runtime_failed", "out_of_memory", "invalid_native_output")
FAILURE_CODES = (
    "source_revision_mismatch",
    "checkpoint_missing",
    "checkpoint_hash_mismatch",
    "dependency_unavailable",
    "model_load_failed",
    "cpu_unsupported",
    "xpu_unsupported",
    "transcription_runtime_error",
    "out_of_memory",
    "invalid_native_output",
)
QUANTIZATION_STATUSES = (
    "ok",
    "not_applicable_no_linear",
    "quantization_unsupported",
    "quantized_runtime_failed",
)


@dataclass(frozen=True)
class NormalizedNote:
    """A lossless note-event projection for models with native note output."""

    onset_seconds: float
    offset_seconds: float
    midi_pitch: int
    velocity_midi: int | None = None
    confidence: float | None = None
    instrument_or_program: str | int | None = None

    def __post_init__(self) -> None:
        onset = float(self.onset_seconds)
        offset = float(self.offset_seconds)
        pitch = int(self.midi_pitch)
        if not np.isfinite(onset) or not np.isfinite(offset) or onset < 0 or offset <= onset:
            raise ValueError("note interval must be finite, non-negative, and non-empty")
        if pitch != self.midi_pitch or not 0 <= pitch <= 127:
            raise ValueError("MIDI pitch must be an integer in [0, 127]")
        if self.velocity_midi is not None and not 0 <= int(self.velocity_midi) <= 127:
            raise ValueError("MIDI velocity must be in [0, 127]")
        if self.confidence is not None and not 0 <= float(self.confidence) <= 1:
            raise ValueError("confidence must be in [0, 1]")

    def as_dict(self) -> dict[str, Any]:
        return {
            "onset_seconds": float(self.onset_seconds),
            "offset_seconds": float(self.offset_seconds),
            "midi_pitch": int(self.midi_pitch),
            "velocity_midi": None if self.velocity_midi is None else int(self.velocity_midi),
            "confidence": None if self.confidence is None else float(self.confidence),
            "instrument_or_program": self.instrument_or_program,
        }


@dataclass(frozen=True)
class FramePitchPrediction:
    """Boolean common-grid frame activity with columns MIDI 21 through 108."""

    times_seconds: np.ndarray
    active_midi: np.ndarray

    def __post_init__(self) -> None:
        times = np.asarray(self.times_seconds, dtype=np.float64)
        active = np.asarray(self.active_midi)
        if times.ndim != 1 or active.ndim != 2 or active.shape != (times.shape[0], PITCH_COUNT):
            raise ValueError("frame prediction must have times [T] and activity [T, 88]")
        if active.dtype != np.bool_:
            raise ValueError("common-grid activity must be boolean")
        if not np.isfinite(times).all() or (times.size and np.any(np.diff(times) < 0)):
            raise ValueError("frame times must be finite and non-decreasing")
        object.__setattr__(self, "times_seconds", np.array(times, copy=True))
        object.__setattr__(self, "active_midi", np.array(active, dtype=bool, copy=True))


@dataclass(frozen=True)
class TranscriptionOutput:
    """The only cross-family prediction shape used by the comparison."""

    notes: tuple[NormalizedNote, ...] | None
    frame_pitch: FramePitchPrediction | None

    def __post_init__(self) -> None:
        if self.notes is None and self.frame_pitch is None:
            raise ValueError("a transcription output must expose notes or frame activity")
        if self.notes is not None:
            object.__setattr__(self, "notes", tuple(self.notes))
            if not all(isinstance(note, NormalizedNote) for note in self.notes):
                raise ValueError("notes must contain NormalizedNote values")

    @property
    def has_native_note_events(self) -> bool:
        return self.notes is not None


def common_frame_times(n_frames: int) -> np.ndarray:
    """Return #25's exact model-frame timestamps, without a second grid."""
    if n_frames < 0:
        raise ValueError("frame count must be non-negative")
    _basic_pitch_root()
    from obruxo_basic_pitch.postprocess import model_frames_to_time

    return np.asarray(model_frames_to_time(n_frames), dtype=np.float64)


def common_frame_count(original_sample_count: int) -> int:
    """Return #25's canonical frame population for resampled audio."""
    if original_sample_count < 0:
        raise ValueError("sample count must be non-negative")
    _basic_pitch_root()
    from obruxo_basic_pitch.constants import ANNOTATIONS_FPS, AUDIO_SAMPLE_RATE

    return int(original_sample_count * ANNOTATIONS_FPS // AUDIO_SAMPLE_RATE)


def _basic_pitch_root() -> Path:
    root = Path(__file__).resolve().parents[2] / "basic_pitch"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def rasterize_notes(notes: tuple[NormalizedNote, ...] | list[NormalizedNote], n_frames: int) -> FramePitchPrediction:
    """Rasterize native notes through #25's half-open frame-target helper."""
    import sys
    from pathlib import Path

    basic_pitch_root = Path(__file__).resolve().parents[2] / "basic_pitch"
    if str(basic_pitch_root) not in sys.path:
        sys.path.insert(0, str(basic_pitch_root))
    from obruxo_basic_pitch.evaluation.labels import ReferenceNote
    from obruxo_basic_pitch.evaluation.metrics import frame_target

    reference_like = [
        ReferenceNote(note.onset_seconds, note.offset_seconds, note.midi_pitch, note.velocity_midi or 0)
        for note in notes
    ]
    return FramePitchPrediction(common_frame_times(n_frames), frame_target(reference_like, n_frames))


def empty_frame_prediction(n_frames: int) -> FramePitchPrediction:
    return FramePitchPrediction(common_frame_times(n_frames), np.zeros((n_frames, PITCH_COUNT), dtype=bool))


def validate_output(value: Any) -> TranscriptionOutput:
    if not isinstance(value, TranscriptionOutput):
        raise TypeError("adapter did not return TranscriptionOutput")
    return value

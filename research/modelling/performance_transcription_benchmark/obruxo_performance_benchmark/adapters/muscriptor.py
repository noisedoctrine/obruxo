"""MuScriptor's deterministic timing-corrected MIDI normalization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..artifacts import (
    ArtifactUnavailable,
    ModelSpec,
    verify_checkout,
    verify_checkpoint,
)
from ..types import NormalizedNote
from .yourmt3 import normalize_note_events

MUSCRIPTOR_IDS = ("muscriptor_small", "muscriptor_medium", "muscriptor_large")


def stock_decoding_config() -> dict[str, Any]:
    return {
        "use_sampling": False,
        "temperature": 1.0,
        "cfg_coef": 1.0,
        "beam_size": 1,
        "prelude_forcing": True,
        "batch_size": 1,
        "detect_tempo": "best_effort",
        "quality_path": "transcribe_to_midi",
    }


def batch_status(batch_size: int) -> dict[str, str]:
    if batch_size == 1:
        return {"status": "ok", "reason": "stock_prelude_forcing"}
    return {"status": "not_applicable", "reason": "prelude_forcing_requires_batch_size_1"}


def normalize_timing_corrected_events(events: Any) -> tuple[NormalizedNote, ...]:
    normalized = normalize_note_events(events)
    return tuple(
        NormalizedNote(
            note.onset_seconds,
            note.offset_seconds,
            note.midi_pitch,
            None,
            note.confidence,
            note.instrument_or_program,
        )
        for note in normalized
    )


class MuScriptorAdapter:
    def __init__(self, spec: ModelSpec, source_root: Path | None, checkpoint: Path | None) -> None:
        self.spec = spec
        self.source_root = None if source_root is None else Path(source_root)
        self.checkpoint = None if checkpoint is None else Path(checkpoint)
        self.inference = stock_decoding_config()

    def preflight(self) -> None:
        if self.source_root is None or self.checkpoint is None:
            raise ArtifactUnavailable("dependency_unavailable")
        verify_checkout(self.spec, self.source_root)
        verify_checkpoint(self.spec, self.checkpoint)

    def load(self) -> None:
        self.preflight()
        try:
            __import__("torch")
            __import__("muscriptor")
        except (ImportError, OSError) as exc:
            raise ArtifactUnavailable("dependency_unavailable") from exc

    def transcribe(self, _audio: Path) -> Any:
        raise ArtifactUnavailable("official MuScriptor runtime is not present")

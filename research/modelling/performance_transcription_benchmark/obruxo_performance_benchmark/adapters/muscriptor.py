"""MuScriptor's deterministic timing-corrected MIDI normalization."""

from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path
from typing import Any

from ..artifacts import (
    ArtifactUnavailable,
    ModelSpec,
    verify_checkout,
    verify_checkpoint,
)
from ..types import NormalizedNote, TranscriptionOutput, rasterize_notes
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
        self.model: Any | None = None
        self.bound_model: Any | None = None

    def preflight(self) -> None:
        if self.source_root is None or self.checkpoint is None:
            raise ArtifactUnavailable("dependency_unavailable")
        verify_checkout(self.spec, self.source_root)
        verify_checkpoint(self.spec, self.checkpoint)

    @property
    def active_model(self) -> Any | None:
        return self.bound_model if self.bound_model is not None else self.model

    def load(self, device: str = "cpu") -> None:
        self.preflight()
        try:
            import torch

            root = self.source_root.resolve(strict=True)
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            from muscriptor import TranscriptionModel
        except (ImportError, OSError) as exc:
            raise ArtifactUnavailable("dependency_unavailable") from exc
        if device not in {"cpu", "xpu"}:
            raise ValueError("MuScriptor supports only the requested CPU/XPU routes")
        self.model = TranscriptionModel.load_model(weights_path=str(self.checkpoint), device=torch.device(device), dtype="float32")
        self.bound_model = self.model

    def bind_model(self, model: Any) -> None:
        self.bound_model = model
        if hasattr(model, "eval"):
            model.eval()

    @staticmethod
    def _frame_count(audio: Path) -> int:
        import sys

        root = Path(__file__).resolve().parents[2] / "basic_pitch"
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from obruxo_basic_pitch.inference import prepare_wav

        return int(prepare_wav(Path(audio)).original_sample_count * 100 // 22050)

    @staticmethod
    def _decode_midi(data: bytes) -> tuple[NormalizedNote, ...]:
        import mido

        midi = mido.MidiFile(file=BytesIO(data))
        merged = mido.merge_tracks(midi.tracks)
        tempo = 500000
        ticks = 0
        seconds = 0.0
        active: dict[tuple[int, int], list[tuple[float, int | None]]] = {}
        notes: list[NormalizedNote] = []
        for message in merged:
            seconds += mido.tick2second(message.time, midi.ticks_per_beat, tempo)
            ticks += message.time
            if message.type == "set_tempo":
                tempo = message.tempo
                continue
            if message.type == "program_change":
                continue
            if message.type == "note_on" and message.velocity > 0:
                active.setdefault((message.channel, message.note), []).append((seconds, message.velocity))
                continue
            if message.type not in {"note_off", "note_on"}:
                continue
            key = (message.channel, message.note)
            pending = active.get(key)
            if not pending:
                continue
            onset, _velocity = pending.pop(0)
            if not pending:
                active.pop(key, None)
            if seconds > onset:
                notes.append(NormalizedNote(onset, seconds, message.note, None, None, message.channel))
        return tuple(notes)

    def transcribe(self, audio: Path) -> TranscriptionOutput:
        if self.active_model is None:
            self.load()
        data = self.active_model.transcribe_to_midi(
            str(Path(audio).resolve(strict=True)),
            use_sampling=False,
            temperature=1.0,
            cfg_coef=1.0,
            beam_size=1,
            prelude_forcing=True,
            batch_size=1,
            detect_tempo="best-effort",
        )
        notes = normalize_timing_corrected_events(self._decode_midi(data))
        return TranscriptionOutput(notes, rasterize_notes(notes, self._frame_count(Path(audio))))

"""Lossless YourMT3+ event normalization and fixed stock inference settings."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..artifacts import (
    ArtifactUnavailable,
    ModelSpec,
    verify_checkout,
    verify_checkpoint,
)
from ..types import NormalizedNote

YOURMT3_IDS = ("ymt3_plus", "yptf_multi", "yptf_moe_multi")


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def stock_inference_config(model_id: str, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if model_id not in YOURMT3_IDS:
        raise ValueError(f"not a YourMT3+ model: {model_id}")
    config = {
        "project": "2024",
        "precision": "float32",
        "deterministic": True,
        "decode": "stock_official_release",
        "checkpoint_selection": "locked_before_corpus_results",
    }
    if overrides:
        changed = {key: value for key, value in overrides.items() if config.get(key) != value}
        if changed:
            raise ValueError("YourMT3+ stock inference settings cannot be overridden")
    return config


def normalize_note_events(events: Any) -> tuple[NormalizedNote, ...]:
    """Normalize upstream event objects/dicts without adding an alternate decoder."""
    if events is None:
        return ()
    rows = events if isinstance(events, (list, tuple)) else list(events)
    starts: dict[int, dict[str, Any]] = {}
    result: list[NormalizedNote] = []
    for event in rows:
        kind = str(_field(event, "event_type", _field(event, "type", event.__class__.__name__))).casefold()
        if "progress" in kind or "tempo" in kind:
            continue
        if _field(event, "onset_seconds") is not None or _field(event, "onset") is not None:
            onset = _field(event, "onset_seconds", _field(event, "onset"))
            offset = _field(event, "offset_seconds", _field(event, "offset"))
            pitch = _field(event, "midi_pitch", _field(event, "pitch"))
            if offset is None:
                offset = _field(event, "end_time", _field(event, "end_seconds"))
            result.append(
                NormalizedNote(
                    float(onset),
                    float(offset),
                    int(pitch),
                    None if _field(event, "velocity") is None else int(_field(event, "velocity")),
                    None if _field(event, "confidence") is None else float(_field(event, "confidence")),
                    _field(event, "instrument_or_program", _field(event, "instrument")),
                )
            )
            continue
        if "start" in kind or _field(event, "start_time") is not None:
            index = _field(event, "index", len(starts))
            starts[int(index)] = {
                "onset": _field(event, "start_time", _field(event, "onset")),
                "pitch": _field(event, "pitch", _field(event, "midi_pitch")),
                "instrument": _field(event, "instrument", _field(event, "program")),
                "velocity": _field(event, "velocity"),
            }
            continue
        if "end" in kind or _field(event, "end_time") is not None:
            start_event = _field(event, "start_event")
            index = _field(event, "start_event_index", _field(start_event, "index") if start_event is not None else None)
            if index is None or int(index) not in starts:
                raise ValueError("YourMT3+ end event does not reference a start event")
            start = starts[int(index)]
            result.append(
                NormalizedNote(
                    float(start["onset"]),
                    float(_field(event, "end_time")),
                    int(start["pitch"]),
                    None if start["velocity"] is None else int(start["velocity"]),
                    None,
                    start["instrument"],
                )
            )
    return tuple(result)


def write_temporary_midi(data: bytes, output_dir: Path, name: str = "upstream.mid") -> Path:
    root = Path(output_dir).resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    destination = (root / name).resolve(strict=False)
    if destination.parent != root or destination.suffix.casefold() != ".mid":
        raise ValueError("temporary MIDI must remain directly under the provided output directory")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.stem}.", suffix=".tmp", dir=root)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        return destination
    finally:
        temporary.unlink(missing_ok=True)


class YourMT3Adapter:
    def __init__(self, spec: ModelSpec, source_root: Path | None, checkpoint: Path | None) -> None:
        self.spec = spec
        self.source_root = None if source_root is None else Path(source_root)
        self.checkpoint = None if checkpoint is None else Path(checkpoint)
        self.inference = stock_inference_config(spec.model_id)

    def preflight(self) -> None:
        if self.source_root is None or self.checkpoint is None:
            raise ArtifactUnavailable("dependency_unavailable")
        verify_checkout(self.spec, self.source_root)
        verify_checkpoint(self.spec, self.checkpoint)

    def load(self) -> None:
        self.preflight()
        try:
            __import__("torch")
            __import__("torchaudio")
        except (ImportError, OSError) as exc:
            raise ArtifactUnavailable("dependency_unavailable") from exc

    def transcribe(self, _audio: Path) -> Any:
        raise ArtifactUnavailable("official YourMT3+ runtime is not present")

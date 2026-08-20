"""Ground-truth MIDI descriptors for the observed PresetShare corpus."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any


@dataclass(frozen=True)
class ReferenceNote:
    onset_s: float
    offset_s: float
    pitch_midi: int
    velocity: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "onset_s": self.onset_s,
            "offset_s": self.offset_s,
            "pitch_midi": self.pitch_midi,
            "velocity": self.velocity,
        }


def _ensure_data_generation_importable() -> None:
    root = str(Path(__file__).resolve().parents[5] / "research" / "data_generation")
    if root not in sys.path:
        sys.path.insert(0, root)


def _max_polyphony(spans: list[Any]) -> int:
    boundaries = [(span.start_tick, 1) for span in spans] + [
        (span.end_tick, -1) for span in spans
    ]
    active = 0
    maximum = 0
    for _, delta in sorted(boundaries, key=lambda item: (item[0], item[1])):
        active += delta
        maximum = max(maximum, active)
    return maximum


def _fixed_class(
    value: float | None, boundaries: tuple[float, float], names: tuple[str, str, str]
) -> str:
    if value is None:
        return "unknown"
    if value < boundaries[0]:
        return names[0]
    if value < boundaries[1]:
        return names[1]
    return names[2]


def performance_labels(path: Path | str) -> tuple[list[ReferenceNote], dict[str, Any]]:
    """Parse one MIDI through the repository's canonical Performance/TempoMap APIs."""
    _ensure_data_generation_importable()
    from obruxo_data.errors import ValidationError
    from obruxo_data.midi import Performance, TempoMap

    try:
        performance = Performance.from_midi(path)
        performance.validate().require_valid()
    except (OSError, ValueError, ValidationError) as exc:
        raise ValueError("reference MIDI is invalid") from exc
    spans = performance.note_spans()
    if not spans:
        raise ValueError("reference MIDI contains no note spans")
    tempo_map = TempoMap.from_performance(performance)
    notes = [
        ReferenceNote(
            onset_s=float(tempo_map.tick_to_seconds(span.start_tick)),
            offset_s=float(tempo_map.tick_to_seconds(span.end_tick)),
            pitch_midi=span.pitch,
            velocity=span.velocity,
        )
        for span in spans
    ]
    duration_s = float(tempo_map.tick_to_seconds(performance.end_tick))
    durations = [note.offset_s - note.onset_s for note in notes]
    pitches = [note.pitch_midi for note in notes]
    onset_ticks = [span.start_tick for span in spans]
    maximum_polyphony = _max_polyphony(spans)
    pitch_mean = float(sum(pitches) / len(pitches))
    labels: dict[str, Any] = {
        "note_count": len(notes),
        "duration_s": duration_s,
        "minimum_pitch_midi": min(pitches),
        "maximum_pitch_midi": max(pitches),
        "pitch_span_semitones": max(pitches) - min(pitches),
        "mean_pitch_midi": pitch_mean,
        "maximum_polyphony": maximum_polyphony,
        "polyphony_class": "polyphonic" if maximum_polyphony > 1 else "monophonic",
        "notes_per_second": float(len(notes) / duration_s) if duration_s > 0 else None,
        "median_note_duration_s": float(median(durations)),
        "simultaneous_note_onsets": len(onset_ticks) != len(set(onset_ticks)),
        "out_of_range_reference_notes": sum(
            not 21 <= pitch <= 108 for pitch in pitches
        ),
        "duration_class": _fixed_class(
            duration_s, (2.0, 8.0), ("short", "medium", "long")
        ),
        "note_density_class": _fixed_class(
            float(len(notes) / duration_s) if duration_s > 0 else None,
            (2.0, 8.0),
            ("low", "medium", "high"),
        ),
        "pitch_register_class": _fixed_class(
            pitch_mean, (48.0, 72.0), ("low", "mid", "high")
        ),
        "label_sources": {
            "note_count": "derived_midi",
            "duration_s": "derived_midi",
            "minimum_pitch_midi": "derived_midi",
            "maximum_pitch_midi": "derived_midi",
            "pitch_span_semitones": "derived_midi",
            "mean_pitch_midi": "derived_midi",
            "maximum_polyphony": "derived_midi",
            "polyphony_class": "derived_midi",
            "notes_per_second": "derived_midi",
            "median_note_duration_s": "derived_midi",
            "simultaneous_note_onsets": "derived_midi",
            "out_of_range_reference_notes": "derived_midi",
            "duration_class": "derived_midi_fixed_bins",
            "note_density_class": "derived_midi_fixed_bins",
            "pitch_register_class": "derived_midi_fixed_bins",
        },
    }
    return notes, labels


def add_source_metadata(
    labels: Mapping[str, Any], metadata: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Add only explicit PresetShare fields; absent metadata remains unknown."""
    result = dict(labels)
    sources = dict(result.get("label_sources", {}))
    metadata = metadata or {}
    for name in ("instrument", "genre", "type", "vital_style"):
        value = str(metadata.get(name, "") or "").strip() or "unknown"
        result[name] = value
        sources[name] = "source_metadata" if value != "unknown" else "unknown"
    result["label_sources"] = sources
    return result

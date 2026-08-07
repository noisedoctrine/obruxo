from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from obruxo_data.errors import Diagnostic, Severity, ValidationError, ValidationReport

from .events import EventKind, MidiEvent, Performance


def _commit(performance: Performance, events: list[MidiEvent], end_tick: int | None = None) -> None:
    candidate = Performance(performance.ticks_per_beat, bpm=None, events=events, end_tick=performance.end_tick if end_tick is None else end_tick)
    candidate.validate().require_valid()
    performance.events = candidate.events
    performance.end_tick = candidate.end_tick


def transpose(performance: Performance, semitones: int, *, out_of_range: str = "error") -> None:
    if out_of_range not in ("error", "drop"):
        raise ValueError("out_of_range must be 'error' or 'drop'")
    invalid = {span.on_event for span in performance.note_spans() if not 0 <= span.pitch + semitones <= 127}
    invalid |= {span.off_event for span in performance.note_spans() if not 0 <= span.pitch + semitones <= 127}
    if invalid and out_of_range == "error":
        raise ValueError("transposition would move a note outside MIDI range")
    events = []
    for event in performance.events:
        if event in invalid:
            continue
        if event.kind in (EventKind.NOTE_ON, EventKind.NOTE_OFF):
            event = replace(event, data=(event.data[0] + semitones, event.data[1]))
        events.append(event)
    _commit(performance, events)


def scale_velocity(performance: Performance, factor: float, *, clipping: str = "error") -> None:
    if factor < 0:
        raise ValueError("velocity factor must be non-negative")
    if clipping not in ("error", "clip"):
        raise ValueError("clipping must be 'error' or 'clip'")
    events = []
    for event in performance.events:
        if event.kind == EventKind.NOTE_ON:
            value = round(event.data[1] * factor)
            if not 1 <= value <= 127 and clipping == "error":
                raise ValueError("scaled velocity is outside 1..127")
            event = replace(event, data=(event.data[0], min(127, max(1, value))))
        events.append(event)
    _commit(performance, events)


def quantize(performance: Performance, grid_ticks: int, *, preserve_duration: bool = True) -> None:
    if grid_ticks <= 0:
        raise ValueError("grid_ticks must be positive")
    replacements: dict[MidiEvent, int] = {}
    if preserve_duration:
        for span in performance.note_spans():
            start = round(span.start_tick / grid_ticks) * grid_ticks
            replacements[span.on_event] = start
            replacements[span.off_event] = start + span.duration_ticks
    events = []
    for event in performance.events:
        tick = replacements.get(event, round(event.tick / grid_ticks) * grid_ticks)
        events.append(replace(event, tick=tick))
    end_tick = max(performance.end_tick, max((event.tick for event in events), default=0))
    _commit(performance, events, end_tick)


@dataclass(frozen=True)
class PerformanceProfile:
    allowed_event_types: frozenset[str]
    channels: frozenset[int]
    max_polyphony: int
    pitch_range: tuple[int, int]
    velocity_range: tuple[int, int]
    fixed_tempo_bpm: float | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PerformanceProfile":
        return cls(
            allowed_event_types=frozenset(str(item) for item in value["allowed_event_types"]),
            channels=frozenset(int(item) for item in value["channels"]),
            max_polyphony=int(value["max_polyphony"]),
            pitch_range=tuple(int(item) for item in value["pitch_range"]),
            velocity_range=tuple(int(item) for item in value["velocity_range"]),
            fixed_tempo_bpm=None if value.get("fixed_tempo_bpm") is None else float(value["fixed_tempo_bpm"]),
        )

    @classmethod
    def load(cls, path: Path | str, name: str) -> "PerformanceProfile":
        document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if document.get("version") != 1:
            raise ValueError("unsupported performance profile version")
        return cls.from_dict(document["performance_profiles"][name])

    def validate(self, performance: Performance) -> ValidationReport:
        diagnostics = []
        kind_names = {
            EventKind.NOTE_ON: "note", EventKind.NOTE_OFF: "note", EventKind.TEMPO: "tempo",
            EventKind.PITCH_BEND: "pitch_bend", EventKind.CHANNEL_PRESSURE: "channel_pressure",
            EventKind.CONTROL_CHANGE: "control_change", EventKind.TIME_SIGNATURE: "time_signature", EventKind.OPAQUE: "opaque",
        }
        for event in performance.events:
            if kind_names[event.kind] not in self.allowed_event_types:
                diagnostics.append(Diagnostic("midi.profile.event_type", Severity.ERROR, "event type is not allowed", context={"event": event.to_dict()}))
            if event.channel is not None and event.channel not in self.channels:
                diagnostics.append(Diagnostic("midi.profile.channel", Severity.ERROR, "MIDI channel is not allowed", context={"event": event.to_dict()}))
            if event.kind == EventKind.NOTE_ON:
                if not self.pitch_range[0] <= event.data[0] <= self.pitch_range[1]:
                    diagnostics.append(Diagnostic("midi.profile.pitch", Severity.ERROR, "note pitch is outside the profile range", context={"event": event.to_dict()}))
                if not self.velocity_range[0] <= event.data[1] <= self.velocity_range[1]:
                    diagnostics.append(Diagnostic("midi.profile.velocity", Severity.ERROR, "note velocity is outside the profile range", context={"event": event.to_dict()}))
        active = 0
        maximum = 0
        for event in performance.canonical_events():
            if event.kind == EventKind.NOTE_OFF:
                active = max(0, active - 1)
            elif event.kind == EventKind.NOTE_ON:
                active += 1
                maximum = max(maximum, active)
        if maximum > self.max_polyphony:
            diagnostics.append(Diagnostic("midi.profile.polyphony", Severity.ERROR, "maximum polyphony exceeds the profile", context={"maximum": maximum}))
        if self.fixed_tempo_bpm is not None:
            expected = round(60_000_000 / self.fixed_tempo_bpm)
            tempos = [event for event in performance.events if event.kind == EventKind.TEMPO]
            if len(tempos) != 1 or tempos[0].tick != 0 or tempos[0].data[0] != expected:
                diagnostics.append(Diagnostic("midi.profile.tempo", Severity.ERROR, "performance does not use the profile's fixed tempo"))
        return ValidationReport(tuple(diagnostics))

    def apply(self, performance: Performance, *, violations: str = "error") -> None:
        report = self.validate(performance)
        if report.valid:
            return
        if violations != "remove":
            if violations != "error":
                raise ValueError("violations must be 'error' or 'remove'")
            raise ValidationError(report)
        allowed_note_events = set()
        active_ends: list[int] = []
        spans = sorted(performance.note_spans(), key=lambda item: (item.start_tick, item.on_event.order))
        for span in spans:
            active_ends = [end for end in active_ends if end > span.start_tick]
            allowed = (
                span.channel in self.channels
                and self.pitch_range[0] <= span.pitch <= self.pitch_range[1]
                and self.velocity_range[0] <= span.velocity <= self.velocity_range[1]
                and len(active_ends) < self.max_polyphony
            )
            if allowed:
                allowed_note_events.update((span.on_event, span.off_event))
                active_ends.append(span.end_tick)
        events = []
        for event in performance.events:
            if event.kind in (EventKind.NOTE_ON, EventKind.NOTE_OFF):
                if "note" in self.allowed_event_types and event in allowed_note_events:
                    events.append(event)
            elif event.kind == EventKind.TEMPO and "tempo" in self.allowed_event_types:
                events.append(event)
        candidate = Performance(performance.ticks_per_beat, bpm=None, events=events, end_tick=performance.end_tick)
        if self.fixed_tempo_bpm is not None:
            candidate.events = [event for event in candidate.events if event.kind != EventKind.TEMPO]
            candidate.add_tempo(self.fixed_tempo_bpm, tick=0)
        profile_report = self.validate(candidate)
        profile_report.require_valid()
        candidate.validate().require_valid()
        performance.events = candidate.events

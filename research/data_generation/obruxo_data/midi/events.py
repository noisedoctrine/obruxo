from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable

from obruxo_data.hashing import canonical_json


class EventKind(StrEnum):
    TEMPO = "tempo"
    NOTE_OFF = "note_off"
    NOTE_ON = "note_on"
    PITCH_BEND = "pitch_bend"
    CHANNEL_PRESSURE = "channel_pressure"
    CONTROL_CHANGE = "control_change"
    TIME_SIGNATURE = "time_signature"
    OPAQUE = "opaque"


EVENT_PRIORITY = {
    EventKind.TEMPO: 0,
    EventKind.NOTE_OFF: 1,
    EventKind.NOTE_ON: 2,
    EventKind.PITCH_BEND: 3,
    EventKind.CHANNEL_PRESSURE: 4,
    EventKind.CONTROL_CHANGE: 5,
    EventKind.TIME_SIGNATURE: 6,
    EventKind.OPAQUE: 7,
}


@dataclass(frozen=True, order=True)
class MidiEvent:
    tick: int
    order: int
    kind: EventKind
    channel: int | None
    data: tuple[int, ...]
    opaque: tuple[tuple[str, str], ...] = ()

    def sort_key(self) -> tuple[int, int, int, int, int]:
        channel = -1 if self.channel is None else self.channel
        pitch = self.data[0] if self.kind in (EventKind.NOTE_ON, EventKind.NOTE_OFF) and self.data else -1
        return self.tick, EVENT_PRIORITY[self.kind], channel, pitch, self.order

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "tick": self.tick,
            "order": self.order,
            "kind": self.kind.value,
            "channel": self.channel,
            "data": list(self.data),
        }
        if self.opaque:
            value["opaque"] = dict(self.opaque)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MidiEvent":
        return cls(
            tick=int(value["tick"]),
            order=int(value["order"]),
            kind=EventKind(value["kind"]),
            channel=None if value.get("channel") is None else int(value["channel"]),
            data=tuple(int(item) for item in value.get("data", [])),
            opaque=tuple(sorted((str(key), str(item)) for key, item in value.get("opaque", {}).items())),
        )


@dataclass(frozen=True)
class NoteSpan:
    channel: int
    pitch: int
    velocity: int
    start_tick: int
    end_tick: int
    on_event: MidiEvent
    off_event: MidiEvent

    @property
    def duration_ticks(self) -> int:
        return self.end_tick - self.start_tick


@dataclass
class Performance:
    ticks_per_beat: int = 480
    events: list[MidiEvent] = field(default_factory=list)
    end_tick: int = 0

    def __init__(self, ticks_per_beat: int = 480, bpm: float | None = 120, events: Iterable[MidiEvent] = (), end_tick: int = 0):
        self.ticks_per_beat = ticks_per_beat
        self.events = list(events)
        self.end_tick = end_tick
        if bpm is not None and not any(event.kind == EventKind.TEMPO for event in self.events):
            self.add_tempo(bpm, tick=0)

    def _next_order(self) -> int:
        return max((event.order for event in self.events), default=-1) + 1

    def add_tempo(self, bpm: float, *, tick: int = 0) -> None:
        if bpm <= 0:
            raise ValueError("bpm must be positive")
        microseconds = round(60_000_000 / bpm)
        self.events.append(MidiEvent(tick, self._next_order(), EventKind.TEMPO, None, (microseconds,)))
        self.end_tick = max(self.end_tick, tick)

    def add_note(self, *, pitch: int, velocity: int, start_tick: int, duration_ticks: int, channel: int = 0) -> None:
        if duration_ticks <= 0:
            raise ValueError("duration_ticks must be positive")
        order = self._next_order()
        end_tick = start_tick + duration_ticks
        self.events.extend([
            MidiEvent(start_tick, order, EventKind.NOTE_ON, channel, (pitch, velocity)),
            MidiEvent(end_tick, order + 1, EventKind.NOTE_OFF, channel, (pitch, 0)),
        ])
        self.end_tick = max(self.end_tick, end_tick)

    def canonical_events(self) -> list[MidiEvent]:
        return sorted(self.events, key=MidiEvent.sort_key)

    def note_spans(self) -> list[NoteSpan]:
        pending: dict[tuple[int, int], list[MidiEvent]] = {}
        spans = []
        for event in self.canonical_events():
            if event.kind == EventKind.NOTE_ON:
                pending.setdefault((event.channel or 0, event.data[0]), []).append(event)
            elif event.kind == EventKind.NOTE_OFF:
                key = (event.channel or 0, event.data[0])
                if pending.get(key):
                    on = pending[key].pop(0)
                    spans.append(NoteSpan(key[0], key[1], on.data[1], on.tick, event.tick, on, event))
        return spans

    def validate(self, renderer_capabilities: Any | None = None):
        from .validation import validate_performance

        return validate_performance(self, renderer_capabilities)

    def transpose(self, semitones: int, *, out_of_range: str = "error") -> None:
        from .transforms import transpose

        transpose(self, semitones, out_of_range=out_of_range)

    def scale_velocity(self, factor: float, *, clipping: str = "error") -> None:
        from .transforms import scale_velocity

        scale_velocity(self, factor, clipping=clipping)

    def quantize(self, grid_ticks: int, *, preserve_duration: bool = True) -> None:
        from .transforms import quantize

        quantize(self, grid_ticks, preserve_duration=preserve_duration)

    def apply_profile(self, profile: Any, *, violations: str = "error") -> None:
        profile.apply(self, violations=violations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticks_per_beat": self.ticks_per_beat,
            "end_tick": self.end_tick,
            "events": [event.to_dict() for event in self.canonical_events()],
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Performance":
        return cls(
            ticks_per_beat=int(value["ticks_per_beat"]),
            bpm=None,
            events=(MidiEvent.from_dict(item) for item in value.get("events", [])),
            end_tick=int(value["end_tick"]),
        )

    @classmethod
    def from_midi(cls, path: Path | str) -> "Performance":
        from .io import load_midi

        return load_midi(Path(path))

    def save_midi(self, path: Path | str) -> None:
        from .io import save_midi

        self.validate().require_valid()
        save_midi(self, Path(path))

    def replace_events(self, events: Iterable[MidiEvent]) -> None:
        self.events = list(events)

    def clone(self) -> "Performance":
        return Performance(self.ticks_per_beat, bpm=None, events=self.events, end_tick=self.end_tick)

    def replace_event_ticks(self, replacements: dict[MidiEvent, int]) -> None:
        self.events = [replace(event, tick=replacements.get(event, event.tick)) for event in self.events]

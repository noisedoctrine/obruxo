from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

import mido

from .events import EventKind, MidiEvent, Performance


def _opaque(message: mido.messages.BaseMessage) -> tuple[tuple[str, str], ...]:
    value = message.dict()
    value.pop("time", None)
    value["is_meta"] = message.is_meta
    return (("message", json.dumps(value, sort_keys=True, separators=(",", ":"))),)


def _event_from_message(message: mido.messages.BaseMessage, tick: int, order: int) -> MidiEvent | None:
    channel = getattr(message, "channel", None)
    if message.type == "note_on":
        kind = EventKind.NOTE_OFF if message.velocity == 0 else EventKind.NOTE_ON
        velocity = 0 if kind == EventKind.NOTE_OFF else message.velocity
        return MidiEvent(tick, order, kind, channel, (message.note, velocity))
    if message.type == "note_off":
        return MidiEvent(tick, order, EventKind.NOTE_OFF, channel, (message.note, message.velocity))
    if message.type == "set_tempo":
        return MidiEvent(tick, order, EventKind.TEMPO, None, (message.tempo,))
    if message.type == "pitchwheel":
        return MidiEvent(tick, order, EventKind.PITCH_BEND, channel, (message.pitch,))
    if message.type == "aftertouch":
        return MidiEvent(tick, order, EventKind.CHANNEL_PRESSURE, channel, (message.value,))
    if message.type == "control_change":
        return MidiEvent(tick, order, EventKind.CONTROL_CHANGE, channel, (message.control, message.value))
    if message.type == "time_signature":
        return MidiEvent(tick, order, EventKind.TIME_SIGNATURE, None, (
            message.numerator, message.denominator, message.clocks_per_click, message.notated_32nd_notes_per_beat,
        ))
    if message.type == "end_of_track":
        return None
    return MidiEvent(tick, order, EventKind.OPAQUE, channel, tuple(message.bytes()) if not message.is_meta else (), _opaque(message))


def load_midi(path: Path) -> Performance:
    midi = mido.MidiFile(path)
    if midi.type == 2:
        raise ValueError("Type 2 MIDI files are not supported")
    pending: list[tuple[int, int, int, mido.messages.BaseMessage]] = []
    end_ticks = []
    for track_index, track in enumerate(midi.tracks):
        tick = 0
        for message_index, message in enumerate(track):
            tick += message.time
            if message.type == "end_of_track":
                end_ticks.append(tick)
            else:
                pending.append((tick, track_index, message_index, message))
        end_ticks.append(tick)
    pending.sort(key=lambda item: (item[0], item[1], item[2]))
    events = []
    for order, (tick, _, _, message) in enumerate(pending):
        event = _event_from_message(message, tick, order)
        if event is not None:
            events.append(event)
    end_tick = max(end_ticks + [event.tick for event in events] + [0])
    return Performance(midi.ticks_per_beat, bpm=None, events=events, end_tick=end_tick)


def _message_from_event(event: MidiEvent) -> mido.messages.BaseMessage:
    if event.kind == EventKind.NOTE_ON:
        return mido.Message("note_on", channel=event.channel or 0, note=event.data[0], velocity=event.data[1], time=0)
    if event.kind == EventKind.NOTE_OFF:
        return mido.Message("note_off", channel=event.channel or 0, note=event.data[0], velocity=event.data[1], time=0)
    if event.kind == EventKind.TEMPO:
        return mido.MetaMessage("set_tempo", tempo=event.data[0], time=0)
    if event.kind == EventKind.PITCH_BEND:
        return mido.Message("pitchwheel", channel=event.channel or 0, pitch=event.data[0], time=0)
    if event.kind == EventKind.CHANNEL_PRESSURE:
        return mido.Message("aftertouch", channel=event.channel or 0, value=event.data[0], time=0)
    if event.kind == EventKind.CONTROL_CHANGE:
        return mido.Message("control_change", channel=event.channel or 0, control=event.data[0], value=event.data[1], time=0)
    if event.kind == EventKind.TIME_SIGNATURE:
        return mido.MetaMessage(
            "time_signature", numerator=event.data[0], denominator=event.data[1], clocks_per_click=event.data[2],
            notated_32nd_notes_per_beat=event.data[3], time=0,
        )
    if event.kind == EventKind.OPAQUE:
        values = json.loads(dict(event.opaque)["message"])
        message_type = values.pop("type")
        message_class = mido.MetaMessage if values.pop("is_meta", False) else mido.Message
        return message_class.from_dict({"type": message_type, **values, "time": 0})
    raise ValueError(f"cannot export event kind {event.kind}")


def save_midi(performance: Performance, path: Path) -> None:
    midi = mido.MidiFile(type=0, ticks_per_beat=performance.ticks_per_beat)
    track = mido.MidiTrack()
    previous_tick = 0
    for event in performance.canonical_events():
        if event.tick > performance.end_tick:
            raise ValueError("event occurs after explicit end_tick")
        message = _message_from_event(event)
        message.time = event.tick - previous_tick
        track.append(message)
        previous_tick = event.tick
    track.append(mido.MetaMessage("end_of_track", time=performance.end_tick - previous_tick))
    midi.tracks.append(track)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
        midi.save(temporary)
        loaded = load_midi(temporary)
        loaded.validate().require_valid()
        if loaded.end_tick != performance.end_tick:
            raise ValueError("MIDI round trip changed the explicit end boundary")
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

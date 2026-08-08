from __future__ import annotations

from collections import defaultdict
from typing import Any

from obruxo_data.errors import Diagnostic, Severity, ValidationReport

from .events import EventKind, Performance


def _error(code: str, message: str, **context: Any) -> Diagnostic:
    return Diagnostic(code, Severity.ERROR, message, context=context)


def validate_performance(performance: Performance, capabilities: Any | None = None) -> ValidationReport:
    diagnostics: list[Diagnostic] = []
    if not isinstance(performance.ticks_per_beat, int) or isinstance(performance.ticks_per_beat, bool) or performance.ticks_per_beat <= 0:
        diagnostics.append(_error("midi.ticks_per_beat", "ticks_per_beat must be a positive integer"))
    if not isinstance(performance.end_tick, int) or isinstance(performance.end_tick, bool) or performance.end_tick < 0:
        diagnostics.append(_error("midi.end_tick", "end_tick must be a non-negative integer"))

    pending: dict[tuple[int, int], list[int]] = defaultdict(list)
    active = 0
    max_polyphony = 0
    tempo_events = []
    for event in performance.canonical_events():
        if not isinstance(event.tick, int) or isinstance(event.tick, bool) or event.tick < 0:
            diagnostics.append(_error("midi.event.tick", "event tick must be a non-negative integer", event=event.to_dict()))
        if event.tick > performance.end_tick:
            diagnostics.append(_error("midi.event.after_end", "event occurs after end_tick", event=event.to_dict()))
        if event.channel is not None and (not isinstance(event.channel, int) or not 0 <= event.channel <= 15):
            diagnostics.append(_error("midi.event.channel", "MIDI channel must be between 0 and 15", event=event.to_dict()))
        if event.kind in (EventKind.NOTE_ON, EventKind.NOTE_OFF):
            if len(event.data) != 2 or not 0 <= event.data[0] <= 127 or not 0 <= event.data[1] <= 127:
                diagnostics.append(_error("midi.note.range", "note pitch and velocity must be between 0 and 127", event=event.to_dict()))
                continue
            key = (event.channel or 0, event.data[0])
            if event.kind == EventKind.NOTE_ON:
                if event.data[1] == 0:
                    diagnostics.append(_error("midi.note_on.zero", "note-on velocity zero must be normalized to note-off", event=event.to_dict()))
                pending[key].append(event.tick)
                active += 1
                max_polyphony = max(max_polyphony, active)
            elif pending[key]:
                pending[key].pop(0)
                active -= 1
            else:
                diagnostics.append(_error("midi.note.unmatched_off", "note-off has no matching note-on", event=event.to_dict()))
        elif event.kind == EventKind.TEMPO:
            tempo_events.append(event)
            if len(event.data) != 1 or event.data[0] <= 0:
                diagnostics.append(_error("midi.tempo", "tempo must be positive microseconds per beat", event=event.to_dict()))
        elif event.kind == EventKind.PITCH_BEND and (len(event.data) != 1 or not -8192 <= event.data[0] <= 8191):
            diagnostics.append(_error("midi.pitch_bend", "pitch bend must be between -8192 and 8191", event=event.to_dict()))
        elif event.kind == EventKind.CHANNEL_PRESSURE and (len(event.data) != 1 or not 0 <= event.data[0] <= 127):
            diagnostics.append(_error("midi.channel_pressure", "channel pressure must be between 0 and 127", event=event.to_dict()))
        elif event.kind == EventKind.CONTROL_CHANGE and (len(event.data) != 2 or not all(0 <= item <= 127 for item in event.data)):
            diagnostics.append(_error("midi.control_change", "controller and value must be between 0 and 127", event=event.to_dict()))

    for (channel, pitch), starts in pending.items():
        for start in starts:
            diagnostics.append(_error("midi.note.unmatched_on", "note-on has no matching note-off within end_tick", channel=channel, pitch=pitch, start_tick=start))

    if capabilities is not None:
        if not capabilities.notes and any(event.kind in (EventKind.NOTE_ON, EventKind.NOTE_OFF) for event in performance.events):
            diagnostics.append(_error("midi.capability.notes", "renderer does not support notes"))
        if not capabilities.polyphony and max_polyphony > 1:
            diagnostics.append(_error("midi.capability.polyphony", "renderer does not support polyphony", maximum=max_polyphony))
        if not capabilities.tempo_changes and (len(tempo_events) > 1 or any(event.tick != 0 for event in tempo_events)):
            diagnostics.append(_error("midi.capability.tempo_changes", "renderer supports one fixed tempo only"))
        for event in performance.events:
            if event.channel is not None and event.channel >= capabilities.max_channels:
                diagnostics.append(_error("midi.capability.channel", "renderer channel limit exceeded", event=event.to_dict()))
            if event.kind == EventKind.PITCH_BEND and not capabilities.pitch_bend:
                diagnostics.append(_error("midi.capability.pitch_bend", "renderer does not support pitch bend"))
            if event.kind == EventKind.CHANNEL_PRESSURE and not capabilities.channel_pressure:
                diagnostics.append(_error("midi.capability.channel_pressure", "renderer does not support channel pressure"))
            if event.kind == EventKind.CONTROL_CHANGE and event.data[0] not in capabilities.control_changes:
                diagnostics.append(_error("midi.capability.control_change", "renderer does not support this controller", controller=event.data[0]))
            if event.kind == EventKind.OPAQUE:
                diagnostics.append(_error("midi.capability.opaque", "renderer cannot consume an opaque MIDI event", event=event.to_dict()))
    return ValidationReport(tuple(diagnostics))

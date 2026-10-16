"""Additional fixed reference/transfer performances with MIDI expression."""
from obruxo_data.midi import EventKind, MidiEvent, Performance
from obruxo_data.render.capabilities import EXPRESSION_CAPABILITIES

from .performances import PerformanceEntry, initial_bank


def expression_bank():
    entries = list(initial_bank())
    for role, bpm, offset, velocity in (("reference", 120, 0, 88), ("transfer", 90, -12, 104)):
        for style in ("gestures", "poly_expression", "sustain", "release"):
            value = Performance(bpm=bpm)

            def add(tick, kind, channel, *data):
                value.events.append(MidiEvent(tick, value._next_order(), kind, channel, tuple(data)))

            if style in {"gestures", "poly_expression"}:
                notes = ((60 + offset, 1),) if style == "gestures" else ((60 + offset, 1), (67 + offset, 15))
                for pitch, channel in notes:
                    value.add_note(pitch=pitch, velocity=velocity, start_tick=120, duration_ticks=1440, channel=channel, release_velocity=72)
                    sign = 1 if channel == 1 else -1
                    add(480, EventKind.PITCH_BEND, channel, sign * (1024 if role == "reference" else 768))
                    add(960, EventKind.PITCH_BEND, channel, 0)
                    add(240, EventKind.CONTROL_CHANGE, channel, 74, 32)
                    add(720, EventKind.CONTROL_CHANGE, channel, 74, 112)
                    add(1080, EventKind.CHANNEL_PRESSURE, channel, 96 if role == "reference" else 112)
                    add(1440, EventKind.CHANNEL_PRESSURE, channel, 16)
                add(360, EventKind.CONTROL_CHANGE, 0, 1, 32)
                add(840, EventKind.CONTROL_CHANGE, 0, 1, 112)
            elif style == "sustain":
                for index, pitch in enumerate((60, 64, 67)):
                    value.add_note(pitch=pitch + offset, velocity=velocity, start_tick=120 + index * 240, duration_ticks=240,
                                   channel=index + 1, release_velocity=48)
                add(240, EventKind.CONTROL_CHANGE, 0, 64, 127)
                add(1440, EventKind.CONTROL_CHANGE, 0, 64, 0)
            else:
                for index, release in enumerate((16, 64, 112)):
                    value.add_note(pitch=60 + offset + index * 4, velocity=velocity, start_tick=120 + index * 480,
                                   duration_ticks=300, channel=index + 1, release_velocity=release)
            value.end_tick = 1920
            value.validate(EXPRESSION_CAPABILITIES).require_valid()
            entries.append(PerformanceEntry(f"expression_{style}_{role}", role, value.to_json()))
    return tuple(entries)


def expression_bank_document():
    return {"version": 2, "bank": "initial_plus_expression_performance_bank_v1", "capabilities": EXPRESSION_CAPABILITIES.to_dict(),
            "entries": [entry.to_dict() for entry in expression_bank()]}

from __future__ import annotations

import mido
import pytest

from obruxo_data.errors import ValidationError
from obruxo_data.midi import EventKind, MidiEvent, Performance, PerformanceProfile
from obruxo_data.render import RendererCapabilities


def test_construct_and_type_zero_round_trip_preserves_explicit_end(tmp_path) -> None:
    performance = Performance(ticks_per_beat=480, bpm=120)
    performance.add_note(pitch=60, velocity=100, start_tick=0, duration_ticks=480)
    performance.end_tick = 960
    path = tmp_path / "note.mid"
    performance.save_midi(path)

    raw = mido.MidiFile(path)
    assert raw.type == 0
    assert sum(message.time for message in raw.tracks[0]) == 960
    assert raw.tracks[0][-1].type == "end_of_track"
    assert Performance.from_midi(path).to_dict() == performance.to_dict()


def test_type_one_tracks_merge_deterministically(tmp_path) -> None:
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    meta = mido.MidiTrack([
        mido.MetaMessage("set_tempo", tempo=500_000, time=0),
        mido.MetaMessage("end_of_track", time=960),
    ])
    notes = mido.MidiTrack([
        mido.Message("note_on", channel=0, note=60, velocity=100, time=0),
        mido.Message("note_off", channel=0, note=60, velocity=0, time=480),
        mido.MetaMessage("end_of_track", time=480),
    ])
    midi.tracks.extend((meta, notes))
    path = tmp_path / "type1.mid"
    midi.save(path)
    performance = Performance.from_midi(path)
    assert performance.end_tick == 960
    assert [event.kind for event in performance.canonical_events()] == [EventKind.TEMPO, EventKind.NOTE_ON, EventKind.NOTE_OFF]


def test_type_two_is_rejected(tmp_path) -> None:
    midi = mido.MidiFile(type=2, ticks_per_beat=480)
    midi.tracks.append(mido.MidiTrack([mido.MetaMessage("end_of_track", time=0)]))
    path = tmp_path / "type2.mid"
    midi.save(path)
    with pytest.raises(ValueError, match="Type 2"):
        Performance.from_midi(path)


def test_note_on_velocity_zero_normalizes_to_note_off(tmp_path) -> None:
    midi = mido.MidiFile(type=0, ticks_per_beat=480)
    midi.tracks.append(mido.MidiTrack([
        mido.Message("note_on", note=60, velocity=100, time=0),
        mido.Message("note_on", note=60, velocity=0, time=480),
        mido.MetaMessage("end_of_track", time=0),
    ]))
    path = tmp_path / "zero.mid"
    midi.save(path)
    performance = Performance.from_midi(path)
    assert [event.kind for event in performance.events] == [EventKind.NOTE_ON, EventKind.NOTE_OFF]
    assert performance.validate().valid


def test_same_tick_note_off_precedes_note_on_for_same_pitch() -> None:
    performance = Performance(bpm=120)
    performance.add_note(pitch=60, velocity=100, start_tick=0, duration_ticks=480)
    performance.add_note(pitch=60, velocity=90, start_tick=480, duration_ticks=480)
    at_boundary = [event.kind for event in performance.canonical_events() if event.tick == 480]
    assert at_boundary == [EventKind.NOTE_OFF, EventKind.NOTE_ON]
    assert performance.validate().valid


def test_unmatched_notes_are_reported() -> None:
    event = MidiEvent(0, 0, EventKind.NOTE_ON, 0, (60, 100))
    report = Performance(bpm=None, events=[event], end_tick=480).validate()
    assert not report.valid
    assert "midi.note.unmatched_on" in {item.code for item in report.diagnostics}


def test_unsupported_event_is_preserved_but_renderer_rejects_it(tmp_path) -> None:
    midi = mido.MidiFile(type=0, ticks_per_beat=480)
    midi.tracks.append(mido.MidiTrack([
        mido.Message("program_change", channel=0, program=7, time=0),
        mido.MetaMessage("end_of_track", time=480),
    ]))
    source = tmp_path / "opaque.mid"
    output = tmp_path / "opaque-roundtrip.mid"
    midi.save(source)
    performance = Performance.from_midi(source)
    assert performance.events[0].kind == EventKind.OPAQUE
    performance.save_midi(output)
    assert Performance.from_midi(output).events[0].kind == EventKind.OPAQUE
    report = performance.validate(RendererCapabilities())
    assert "midi.capability.opaque" in {item.code for item in report.diagnostics}


def test_transforms_are_atomic_and_validate_ranges() -> None:
    performance = Performance(bpm=120)
    performance.add_note(pitch=120, velocity=100, start_tick=5, duration_ticks=100)
    before = performance.to_json()
    with pytest.raises(ValueError, match="outside MIDI range"):
        performance.transpose(12)
    assert performance.to_json() == before
    performance.transpose(-12)
    performance.scale_velocity(1.2)
    performance.quantize(10, preserve_duration=True)
    span = performance.note_spans()[0]
    assert (span.pitch, span.velocity, span.start_tick, span.duration_ticks) == (108, 120, 0, 100)


def test_performance_profile_rejects_or_explicitly_simplifies() -> None:
    performance = Performance(bpm=120)
    performance.add_note(pitch=60, velocity=100, start_tick=0, duration_ticks=480)
    performance.add_note(pitch=64, velocity=100, start_tick=0, duration_ticks=480)
    profile = PerformanceProfile(
        allowed_event_types=frozenset({"note", "tempo"}), channels=frozenset({0}), max_polyphony=1,
        pitch_range=(48, 72), velocity_range=(1, 127), fixed_tempo_bpm=120,
    )
    before = performance.to_json()
    with pytest.raises(ValidationError):
        performance.apply_profile(profile)
    assert performance.to_json() == before
    performance.apply_profile(profile, violations="remove")
    assert len(performance.note_spans()) == 1
    assert profile.validate(performance).valid


def test_capabilities_reject_tempo_changes_and_expressive_events() -> None:
    performance = Performance(bpm=120)
    performance.add_tempo(90, tick=480)
    performance.events.extend([
        MidiEvent(0, 20, EventKind.PITCH_BEND, 0, (100,)),
        MidiEvent(0, 21, EventKind.CONTROL_CHANGE, 0, (1, 64)),
    ])
    performance.end_tick = 480
    report = performance.validate(RendererCapabilities())
    assert {item.code for item in report.diagnostics} >= {
        "midi.capability.tempo_changes", "midi.capability.pitch_bend", "midi.capability.control_change",
    }

"""Compile absolute half-even sample offsets into an all-event SMF clock."""
import hashlib
from math import gcd
from pathlib import Path

import mido

from obruxo_data.midi import EventKind, TempoMap
from obruxo_data.midi.io import _message_from_event

POLICY = "half_even_sample_offsets_quarter_sample_smf_clock_v1"


def write_sample_midi(performance, path, sample_rate, *, preroll_frames=0):
    """The quarter-sample inset survives either floor or nearest host quantization."""
    performance.validate().require_valid()
    if type(sample_rate) is not int or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    if type(preroll_frames) is not int or preroll_frames < 0:
        raise ValueError("preroll_frames must be a nonnegative integer")
    divisor = gcd(sample_rate * 4, 1_000_000)
    ppq, tempo = sample_rate * 4 // divisor, 1_000_000 // divisor
    if ppq > 32767:
        raise ValueError("sample rate cannot be represented by this exact SMF clock")
    clock = TempoMap.from_performance(performance)
    if any(clock.tick_to_sample(note.end_tick, sample_rate) <= clock.tick_to_sample(note.start_tick, sample_rate) for note in performance.note_spans()):
        raise ValueError("note duration rounds to zero samples")
    midi = mido.MidiFile(type=0, ticks_per_beat=ppq)
    track = mido.MidiTrack([mido.MetaMessage("set_tempo", tempo=tempo, time=0)])
    previous, records = 0, []
    for event in performance.canonical_events():
        if event.kind == EventKind.TEMPO:
            continue  # Source tempo has already been applied to the absolute sample offsets.
        if event.kind == EventKind.OPAQUE:
            raise ValueError("opaque MIDI is outside the reviewed transport")
        sample = clock.tick_to_sample(event.tick, sample_rate) + preroll_frames
        tick = sample * 4 + 1
        message = _message_from_event(event)
        message.time = tick - previous
        track.append(message)
        previous = tick
        records.append({"sample": sample, "kind": event.kind.value, "channel": event.channel, "data": list(event.data)})
    end_tick = (clock.tick_to_sample(performance.end_tick, sample_rate) + preroll_frames) * 4 + 1
    track.append(mido.MetaMessage("end_of_track", time=end_tick - previous))
    midi.tracks.append(track)
    path = Path(path)
    midi.save(path)
    return {"transport": POLICY, "sample_rate": sample_rate, "smf_ticks_per_beat": ppq, "smf_microseconds_per_beat": tempo,
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "events": records,
            "note_off_velocity": "preserved", "channels": "preserved", "all_events": True,
            **({"preroll_frames": preroll_frames} if preroll_frames else {})}

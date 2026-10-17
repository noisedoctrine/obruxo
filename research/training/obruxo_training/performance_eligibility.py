"""Select observations long enough to expose the requested amplitude attack."""
import hashlib
from pathlib import Path

from obruxo_data.midi import EventKind, TempoMap

from .codec import PilotTarget

RULE = "complete_amplitude_attack_under_note_or_sustain_v1"


def rule_document():
    return {"rule": RULE, "source_sha256_lf": hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
            "scope": "temporal_observability_not_guaranteed_audibility", "preset_changes": False}


def attack_seconds(codec, target):
    scalar = getattr(getattr(codec, "base", None), "scalar", None)
    if scalar is None:
        raise ValueError("attack-compatible sampling requires a complete nested-state codec")
    settings = scalar._values(PilotTarget(target.continuous[:len(scalar.continuous)], target.categorical[:len(scalar.categorical)]))
    return settings["env_1_delay"] ** 4 + settings["env_1_attack"] ** 4


def longest_held_seconds(performance):
    timing = TempoMap.from_performance(performance)
    pedals = {}
    for event in performance.canonical_events():
        if event.kind == EventKind.CONTROL_CHANGE and event.data[0] == 64:
            pedals.setdefault(event.channel, []).append((event.sort_key(), event.data[1] >= 64))
    longest = 0.
    for note in performance.note_spans():
        end = note.end_tick
        for channel in {0, note.channel}:
            events = pedals.get(channel, [])
            preceding = [down for key, down in events if key < note.off_event.sort_key()]
            if preceding and preceding[-1]:
                release = next((key[0] for key, down in events if key > note.off_event.sort_key() and not down), performance.end_tick)
                end = max(end, release)
        duration = (timing.tick_to_sample(end, 44100) - timing.tick_to_sample(note.start_tick, 44100)) / 44100
        longest = max(longest, duration)
    return longest


def eligible(codec, target, performance):
    return longest_held_seconds(performance) >= attack_seconds(codec, target)

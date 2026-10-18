"""Explicit longer reference/transfer notes without changing older performance banks."""
import hashlib
from pathlib import Path

from obruxo_data.midi import Performance
from obruxo_data.render.base import FULL_CORPUS_RENDERER_ID
from obruxo_data.render.capabilities import EXPRESSION_CAPABILITIES

from .expression_bank import expression_bank
from .performances import PerformanceEntry

PROFILE = "bounded_long_observations_v1"


def require_profile(renderer_id, profile):
    if renderer_id != FULL_CORPUS_RENDERER_ID or profile != PROFILE:
        raise ValueError("unsupported performance profile for this renderer")


def observation_bank():
    entries = list(expression_bank())
    for seconds in (4, 12, 40):
        for role, pitch, velocity in (("reference", 60, 32), ("reference", 60, 64), ("transfer", 72, 80)):
            value = Performance(bpm=120)
            value.add_note(pitch=pitch, velocity=velocity, start_tick=240, duration_ticks=seconds * 960)
            value.end_tick += 480
            value.validate(EXPRESSION_CAPABILITIES).require_valid()
            entries.append(PerformanceEntry(f"held_{seconds}s_{role}_v{velocity}", role, value.to_json()))
    return tuple(entries)


def observation_bank_document():
    return {"version": 3, "bank": PROFILE, "source_sha256_lf": hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
            "capabilities": EXPRESSION_CAPABILITIES.to_dict(), "entries": [entry.to_dict() for entry in observation_bank()]}

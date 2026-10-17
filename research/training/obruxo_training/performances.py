from __future__ import annotations

from dataclasses import dataclass
import json

from obruxo_data.hashing import canonical_sha256
from obruxo_data.midi import Performance
from obruxo_data.render import RendererCapabilities


CAPABILITIES = RendererCapabilities(max_channels=1)


@dataclass(frozen=True)
class PerformanceEntry:
    name: str
    role: str
    performance_json: str

    @property
    def performance_id(self) -> str:
        return canonical_sha256(json.loads(self.performance_json))

    def performance(self) -> Performance:
        return Performance.from_dict(json.loads(self.performance_json))

    def to_dict(self) -> dict:
        return {"name": self.name, "role": self.role, "performance_id": self.performance_id,
                "performance": json.loads(self.performance_json)}


def initial_bank() -> tuple[PerformanceEntry, ...]:
    """A small exact bank for contract tests/pilots; not the eventual online distribution."""
    patterns = (
        ("held_mid", "reference", 120, [(60, 96, 0, 1440)]),
        ("staccato_mid", "reference", 120, [(60, 80, tick, 120) for tick in range(0, 1920, 480)]),
        ("melody_mid", "reference", 120, [(pitch, 96, index * 480, 360) for index, pitch in enumerate((60, 64, 67, 62))]),
        ("chord_mid", "reference", 120, [(pitch, 80, 0, 1440) for pitch in (60, 64, 67)]),
        ("held_low", "transfer", 90, [(36, 64, 0, 1440)]),
        ("staccato_high", "transfer", 150, [(84, 112, tick, 180) for tick in (0, 360, 960, 1560)]),
        ("melody_wide", "transfer", 90, [(pitch, 72, index * 480, 600) for index, pitch in enumerate((48, 67, 55, 72))]),
        ("chord_wide", "transfer", 150, [(pitch, 112, 0, 1680) for pitch in (36, 55, 76)]),
    )
    entries = []
    for name, role, bpm, notes in patterns:
        performance = Performance(bpm=bpm)
        for pitch, velocity, start, duration in notes:
            performance.add_note(pitch=pitch, velocity=velocity, start_tick=start, duration_ticks=duration)
        performance.end_tick = max(performance.end_tick, 1920)
        performance.validate(CAPABILITIES).require_valid()
        entries.append(PerformanceEntry(name, role, performance.to_json()))
    return tuple(entries)


def bank_document() -> dict:
    return {"version": 1, "bank": "initial_performance_bank_v1", "entries": [entry.to_dict() for entry in initial_bank()]}


def bank_for(renderer_id):
    from obruxo_data.render.base import FULL_CORPUS_RENDERER_ID, FULL_CURVE_RENDERER_ID, FULL_EXPRESSION_RENDERER_ID
    if renderer_id in {FULL_EXPRESSION_RENDERER_ID, FULL_CURVE_RENDERER_ID, FULL_CORPUS_RENDERER_ID}:
        from .expression_bank import expression_bank
        return expression_bank()
    return initial_bank()


def bank_document_for(renderer_id):
    from obruxo_data.render.base import FULL_CORPUS_RENDERER_ID, FULL_CURVE_RENDERER_ID, FULL_EXPRESSION_RENDERER_ID
    if renderer_id in {FULL_EXPRESSION_RENDERER_ID, FULL_CURVE_RENDERER_ID, FULL_CORPUS_RENDERER_ID}:
        from .expression_bank import expression_bank_document
        return expression_bank_document()
    return bank_document()


def capabilities_for(renderer_id):
    from obruxo_data.render.base import FULL_CORPUS_RENDERER_ID, FULL_CURVE_RENDERER_ID, FULL_EXPRESSION_RENDERER_ID
    from obruxo_data.render.capabilities import EXPRESSION_CAPABILITIES
    return EXPRESSION_CAPABILITIES if renderer_id in {FULL_EXPRESSION_RENDERER_ID, FULL_CURVE_RENDERER_ID, FULL_CORPUS_RENDERER_ID} else CAPABILITIES

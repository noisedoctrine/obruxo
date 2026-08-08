from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache

from .atlas import VitalSchema


class ComponentKind(StrEnum):
    OSCILLATOR = "oscillator"
    SAMPLER = "sampler"
    FILTER = "filter"
    ENVELOPE = "envelope"
    LFO = "lfo"
    RANDOM = "random"
    EFFECT = "effect"
    MODULATION_SLOT = "modulation_slot"
    GLOBAL = "global"


@dataclass(frozen=True, order=True)
class ComponentRef:
    kind: ComponentKind
    slot: int | str | None = None

    @classmethod
    def parse(cls, value: str) -> "ComponentRef":
        if "[" not in value:
            return cls(ComponentKind(value))
        kind, slot_text = value.rstrip("]").split("[", 1)
        slot: int | str = int(slot_text) if slot_text.isdigit() else slot_text
        return cls(ComponentKind(kind), slot)

    def __str__(self) -> str:
        return self.kind.value if self.slot is None else f"{self.kind.value}[{self.slot}]"


@dataclass(frozen=True)
class ComponentDefinition:
    kind: ComponentKind
    slot: int | str | None
    enable_parameter: str | None
    owned_parameters: tuple[str, ...]
    owned_json_pointers: tuple[str, ...]
    modulation_source_names: tuple[str, ...]
    modulation_destination_prefixes: tuple[str, ...]
    dependencies: tuple[ComponentRef, ...] = ()

    @property
    def ref(self) -> ComponentRef:
        return ComponentRef(self.kind, self.slot)


def _matching(names: set[str], prefix: str) -> tuple[str, ...]:
    return tuple(sorted(name for name in names if name.startswith(prefix)))


@lru_cache(maxsize=4)
def component_registry(schema_id: str | None = None) -> dict[ComponentRef, ComponentDefinition]:
    schema = VitalSchema.load()
    if schema_id is not None and schema.schema_id != schema_id:
        raise ValueError(f"unsupported schema id {schema_id}")
    names = set(schema.parameters)
    definitions: list[ComponentDefinition] = []

    def add(kind: ComponentKind, slot: int | str | None, prefix: str, enable: str | None = None,
            pointers: tuple[str, ...] = (), sources: tuple[str, ...] = ()) -> None:
        definitions.append(ComponentDefinition(
            kind=kind,
            slot=slot,
            enable_parameter=enable,
            owned_parameters=_matching(names, prefix),
            owned_json_pointers=pointers,
            modulation_source_names=sources,
            modulation_destination_prefixes=(prefix,),
        ))

    for slot in range(1, 4):
        add(ComponentKind.OSCILLATOR, slot, f"osc_{slot}_", f"osc_{slot}_on")
    add(ComponentKind.SAMPLER, None, "sample_", "sample_on", ("/settings/sample",))
    for slot in range(1, 3):
        add(ComponentKind.FILTER, slot, f"filter_{slot}_", f"filter_{slot}_on")
    for slot in range(1, 7):
        add(ComponentKind.ENVELOPE, slot, f"env_{slot}_", sources=(f"env_{slot}",))
    for slot in range(1, 9):
        add(ComponentKind.LFO, slot, f"lfo_{slot}_", pointers=(f"/settings/lfos/{slot - 1}",), sources=(f"lfo_{slot}",))
    for slot in range(1, 5):
        add(ComponentKind.RANDOM, slot, f"random_{slot}_", sources=(f"random_{slot}",))
    for effect in ("chorus", "compressor", "delay", "distortion", "eq", "filter_fx", "flanger", "phaser", "reverb"):
        add(ComponentKind.EFFECT, effect, f"{effect}_", f"{effect}_on")
    for slot in range(1, 65):
        add(ComponentKind.MODULATION_SLOT, slot, f"modulation_{slot}_", pointers=(f"/settings/modulations/{slot - 1}",))

    owned = {name for definition in definitions for name in definition.owned_parameters}
    global_names = tuple(sorted(names - owned))
    definitions.append(ComponentDefinition(
        kind=ComponentKind.GLOBAL,
        slot=None,
        enable_parameter=None,
        owned_parameters=global_names,
        owned_json_pointers=(),
        modulation_source_names=("aftertouch", "lift", "macro_control_1", "macro_control_2", "macro_control_3", "macro_control_4", "mod_wheel", "note", "note_in_octave", "pitch_wheel", "random", "slide", "stereo", "velocity"),
        modulation_destination_prefixes=(),
    ))
    return {definition.ref: definition for definition in definitions}

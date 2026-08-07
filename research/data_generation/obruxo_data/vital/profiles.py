from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

import yaml

from .components import ComponentKind, ComponentRef, component_registry

if TYPE_CHECKING:
    from .document import VitalPreset


@dataclass(frozen=True)
class SlotPolicy:
    allow: tuple[int | str, ...]
    reset_disallowed: bool = True
    max_active: int | None = None
    max_routed: int | None = None


@dataclass(frozen=True)
class ComponentProfile:
    oscillators: SlotPolicy = SlotPolicy((1, 2, 3))
    sampler_allowed: bool = True
    sampler_reset_disallowed: bool = True
    filters: SlotPolicy = SlotPolicy((1, 2))
    lfos: SlotPolicy = SlotPolicy(tuple(range(1, 9)))
    effects: SlotPolicy = SlotPolicy(("chorus", "compressor", "delay", "distortion", "eq", "filter_fx", "flanger", "phaser", "reverb"))
    max_active_routes: int = 64

    @classmethod
    def only(cls, *, oscillators: list[int] | None = None, lfos: list[int] | None = None,
             filters: list[int] | None = None, effects: list[str] | None = None, sampler: bool = False,
             max_active_routes: int | None = None) -> "ComponentProfile":
        osc = tuple(oscillators or ())
        lfo = tuple(lfos or ())
        filt = tuple(filters or ())
        fx = tuple(effects or ())
        return cls(
            oscillators=SlotPolicy(osc, max_active=len(osc)),
            sampler_allowed=sampler,
            filters=SlotPolicy(filt, max_active=len(filt)),
            lfos=SlotPolicy(lfo, max_routed=len(lfo)),
            effects=SlotPolicy(fx, max_active=len(fx)),
            max_active_routes=max_active_routes if max_active_routes is not None else len(lfo),
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ComponentProfile":
        def slots(name: str, defaults: tuple[int | str, ...]) -> SlotPolicy:
            raw = value.get(name, {})
            allow = raw.get("allow", defaults)
            return SlotPolicy(tuple(allow), bool(raw.get("reset_disallowed", True)), raw.get("max_active"), raw.get("max_routed"))

        sampler = value.get("sampler", {})
        modulation = value.get("modulation", {})
        return cls(
            oscillators=slots("oscillators", (1, 2, 3)),
            sampler_allowed=bool(sampler.get("allow", True)),
            sampler_reset_disallowed=bool(sampler.get("reset_disallowed", True)),
            filters=slots("filters", (1, 2)),
            lfos=slots("lfos", tuple(range(1, 9))),
            effects=slots("effects", ("chorus", "compressor", "delay", "distortion", "eq", "filter_fx", "flanger", "phaser", "reverb")),
            max_active_routes=int(modulation.get("max_active_routes", 64)),
        )

    @classmethod
    def load(cls, path: Path | str, name: str) -> "ComponentProfile":
        document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if document.get("version") != 1:
            raise ValueError("unsupported component profile version")
        return cls.from_dict(document["profiles"][name])

    def apply(self, preset: "VitalPreset") -> None:
        candidate = preset.to_dict()
        working = type(preset)(candidate, preset.schema)
        registry = component_registry(preset.schema.schema_id)
        policies = {
            ComponentKind.OSCILLATOR: self.oscillators,
            ComponentKind.FILTER: self.filters,
            ComponentKind.LFO: self.lfos,
            ComponentKind.EFFECT: self.effects,
        }
        for kind, policy in policies.items():
            for ref in sorted((item for item in registry if item.kind == kind), key=lambda item: str(item.slot)):
                if ref.slot not in policy.allow and policy.reset_disallowed:
                    working._reset_document_component(candidate, ref)
        sampler = ComponentRef(ComponentKind.SAMPLER)
        if not self.sampler_allowed and self.sampler_reset_disallowed:
            working._reset_document_component(candidate, sampler)

        self._enforce_active_limit(working, candidate, ComponentKind.OSCILLATOR, self.oscillators)
        self._enforce_active_limit(working, candidate, ComponentKind.FILTER, self.filters)
        self._enforce_active_limit(working, candidate, ComponentKind.EFFECT, self.effects)
        self._enforce_lfo_limit(working, candidate)
        self._enforce_route_limit(working, candidate)
        working._replace_document(candidate)
        self._assert_constraints(working)
        preset._replace_document(working.to_dict())

    @staticmethod
    def _enforce_active_limit(working: "VitalPreset", candidate: dict[str, Any], kind: ComponentKind, policy: SlotPolicy) -> None:
        if policy.max_active is None:
            return
        registry = component_registry(working.schema.schema_id)
        active = []
        for ref, definition in registry.items():
            if ref.kind == kind and ref.slot in policy.allow and definition.enable_parameter:
                if candidate["settings"][definition.enable_parameter] != 0:
                    active.append(ref)
        for ref in active[policy.max_active:]:
            working._reset_document_component(candidate, ref)

    def _enforce_lfo_limit(self, working: "VitalPreset", candidate: dict[str, Any]) -> None:
        if self.lfos.max_routed is None:
            return
        allowed_sources = [f"lfo_{slot}" for slot in self.lfos.allow]
        routed: list[str] = []
        for index, connection in enumerate(candidate["settings"]["modulations"]):
            source = connection.get("source", "")
            if source not in allowed_sources:
                continue
            if source not in routed:
                routed.append(source)
            if len(routed) > self.lfos.max_routed:
                working._clear_route(candidate, index)

    def _enforce_route_limit(self, working: "VitalPreset", candidate: dict[str, Any]) -> None:
        live = 0
        for index, connection in enumerate(candidate["settings"]["modulations"]):
            bypassed = candidate["settings"][f"modulation_{index + 1}_bypass"] != 0
            if connection.get("source") and connection.get("destination") and not bypassed:
                live += 1
                if live > self.max_active_routes:
                    working._clear_route(candidate, index)

    def _assert_constraints(self, preset: "VitalPreset") -> None:
        document = preset.to_dict()
        init = preset.schema.load_init_document()
        registry = component_registry(preset.schema.schema_id)
        policies = {
            ComponentKind.OSCILLATOR: self.oscillators,
            ComponentKind.FILTER: self.filters,
            ComponentKind.LFO: self.lfos,
            ComponentKind.EFFECT: self.effects,
        }
        for kind, policy in policies.items():
            active = 0
            for ref, definition in registry.items():
                if ref.kind != kind:
                    continue
                if ref.slot not in policy.allow and not policy.reset_disallowed:
                    changed = any(document["settings"][name] != init["settings"][name] for name in definition.owned_parameters)
                    if changed:
                        raise preset._profile_failure("vital.profile.disallowed", "disallowed component retains non-stock state", component=str(ref))
                if ref.slot in policy.allow and definition.enable_parameter and document["settings"][definition.enable_parameter] != 0:
                    active += 1
            if policy.max_active is not None and active > policy.max_active:
                raise preset._profile_failure("vital.profile.max_active", "profile active-component limit exceeded", kind=kind.value)

        sampler = registry[ComponentRef(ComponentKind.SAMPLER)]
        if not self.sampler_allowed and not self.sampler_reset_disallowed:
            changed = any(document["settings"][name] != init["settings"][name] for name in sampler.owned_parameters)
            if changed:
                raise preset._profile_failure("vital.profile.sampler", "disallowed sampler retains non-stock state")

        live_routes = []
        for index, connection in enumerate(document["settings"]["modulations"]):
            bypassed = document["settings"][f"modulation_{index + 1}_bypass"] != 0
            if connection.get("source") and connection.get("destination") and not bypassed:
                live_routes.append(connection)
        if len(live_routes) > self.max_active_routes:
            raise preset._profile_failure("vital.profile.max_routes", "profile active-route limit exceeded")
        if self.lfos.max_routed is not None:
            routed = {item["source"] for item in live_routes if item["source"].startswith("lfo_")}
            if len(routed) > self.lfos.max_routed:
                raise preset._profile_failure("vital.profile.max_routed", "profile routed-LFO limit exceeded")

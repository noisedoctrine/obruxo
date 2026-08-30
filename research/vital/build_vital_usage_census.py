#!/usr/bin/env python3
"""Build a sanitized, semantic usage census for a Vital preset corpus.

The corpus itself is an external, read-only input.  This module keeps only
aggregate counters and semantic descriptors; it never writes a preset back to
disk and never publishes preset paths, names, authors, or embedded payloads.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import platform
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

repository_root = Path(__file__).resolve().parents[2]
if str(repository_root) not in sys.path:
    sys.path.insert(0, str(repository_root))
data_generation_root = repository_root / "research" / "data_generation"
if str(data_generation_root) not in sys.path:
    sys.path.insert(0, str(data_generation_root))

from research.data_generation.obruxo_data.vital.atlas import VitalSchema
from research.data_generation.obruxo_data.vital.components import (
    ComponentDefinition,
    ComponentKind,
    ComponentRef,
    component_registry,
)
from research.vital.build_vital_corpus_audit import iter_vital_files, load_json_bytes


SCRIPT_VERSION = "1.1.0"
DEFAULT_ROOT = Path("datasets") / "presetshare" / "raw" / "presetshare_files" / "data"
DEFAULT_OUTPUT = Path("research") / "vital" / "vital_usage_census.json.gz"
DEFAULT_REPORT = Path("research") / "vital" / "VITAL_USAGE_REPORT.md"
DEFAULT_FIGURES = Path("research") / "vital" / "vital_usage_figures"
FLOAT_TOLERANCE = 1e-7
DISTRIBUTION_BIN_COUNT = 64
FEATURE_DEFAULT_UNAVAILABLE = object()
PAYLOAD_KEYS = frozenset({"audio_file", "samples", "samples_stereo", "wave_data", "line"})
STOCK_SAMPLE_NAMES = frozenset({
    "BART",
    "Box Fan",
    "Brown Noise",
    "Grinder",
    "HVAC Unit",
    "Jack Hammer",
    "Pink Noise",
    "River",
    "Waves",
    "White Noise",
})
STOCK_WAVETABLE_NAMES = frozenset({
    "Acid Rock ft Maynix",
    "Additive Squish 2",
    "Alternating Harmonics",
    "Basic Shapes",
    "Brown Noise",
    "Classic Blend",
    "Classic Fade",
    "Clicky Robot",
    "Corpusbode Phaser",
    "Crappy Toilet",
    "Creepy Solar",
    "Didg",
    "Drink the Juice",
    "Flange Sqrowl",
    "Granular Upgrade",
    "Harmonic Series",
    "Hollow Distorted FM",
    "Init",
    "Jaw Harp",
    "Low High Fold",
    "Pink Noise",
    "Post Modern Pulse",
    "Pulse Width",
    "Quad Saw",
    "Saws for Days",
    "Solar Powered",
    "Squish Flange",
    "Stabbed",
    "Thank u False Noise",
    "Three Graces",
    "Vital Sine",
    "Vital Sine 2",
    "Water Razor",
    "White Noise",
})
EFFECTS = (
    "chorus",
    "compressor",
    "delay",
    "distortion",
    "eq",
    "filter_fx",
    "flanger",
    "phaser",
    "reverb",
)
MAJOR_FAMILIES = ("oscillator", "sampler", "filter", "effect", "envelope", "lfo", "random", "modulation")
AUDIO_DESTINATIONS = {
    0: "filter_1",
    1: "filter_2",
    2: "filter_1_and_2",
    3: "effects",
    4: "direct_out",
    5: "chorus",
    6: "compressor",
    7: "delay",
    8: "distortion",
    9: "eq",
    10: "filter_fx",
    11: "flanger",
    12: "phaser",
    13: "reverb",
}
SPECTRAL_PHASE_RE = re.compile(r"^osc_[123]_spectral_morph_phase$")
MODULATION_RAMP_RE = re.compile(r"^modulation_(?:[1-9]|[1-5][0-9]|6[0-4])_ramp_(?:up|down)$")


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def as_float(value: Any, fallback: float = 0.0) -> float:
    return float(value) if is_number(value) else fallback


def nonzero(value: Any) -> bool:
    return is_number(value) and abs(float(value)) > FLOAT_TOLERANCE


def version_tuple(version: str) -> tuple[int, ...]:
    parts = []
    for part in version.split("."):
        match = re.match(r"\d+", part)
        if not match:
            break
        parts.append(int(match.group()))
    return tuple(parts) if parts else (0,)


def version_family(version: str) -> str:
    parsed = version_tuple(version)
    if parsed[:1] == (1,) and len(parsed) > 1:
        if parsed[1] == 0:
            return "1.0.x"
        if parsed[1] == 5:
            return "1.5.x"
        if parsed[1] == 6:
            return "1.6.x"
    return "unknown"


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _payload_summary(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        serialized = value
        length = len(value)
    else:
        serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        length = len(value) if isinstance(value, (dict, list)) else len(serialized)
    return {"length": length, "sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest()}


def semantic_nested_state(value: Any, *, key: str | None = None) -> Any:
    """Return a comparison descriptor without retaining embedded payloads."""
    if key in PAYLOAD_KEYS:
        return _payload_summary(value)
    if is_number(value):
        return float(value)
    if isinstance(value, dict):
        result = {}
        for child_key, child_value in sorted(value.items()):
            if child_key in {"author", "name", "version"}:
                continue
            result[child_key] = semantic_nested_state(child_value, key=child_key)
        return result
    if isinstance(value, list):
        return [semantic_nested_state(item, key=key) for item in value]
    return value


def nested_signature(value: Any) -> str:
    return stable_json_hash(semantic_nested_state(value))


def lfo_shape_signature(value: Any) -> str:
    if not isinstance(value, dict):
        return stable_json_hash(None)
    return stable_json_hash(semantic_nested_state({key: child for key, child in value.items() if key != "name"}))


def wavetable_component_types(value: Any) -> tuple[str, ...]:
    types: set[str] = set()
    if not isinstance(value, dict):
        return ()
    for group in value.get("groups", []):
        if not isinstance(group, dict):
            continue
        for component in group.get("components", []):
            if isinstance(component, dict) and component.get("type"):
                types.add(str(component["type"]))
    return tuple(sorted(types))


def sample_status(sample: Any, enabled: bool) -> str:
    if not enabled:
        return "disabled"
    if not isinstance(sample, dict):
        return "malformed"
    name = str(sample.get("name", "")).strip()
    if name in STOCK_SAMPLE_NAMES:
        return "named_stock_or_unresolved_content"
    if name:
        return "named_nonstock_or_unresolved_content"
    return "unnamed_or_unresolved_content"


def wavetable_status(wavetable: Any) -> str:
    if not isinstance(wavetable, dict):
        return "malformed_or_unresolved_content"
    name = str(wavetable.get("name", "")).strip()
    if name in STOCK_WAVETABLE_NAMES:
        return "named_stock_or_unresolved_content"
    if name:
        return "named_nonstock_or_unresolved_content"
    return "unnamed_or_unresolved_content"


def line_mapping_is_linear(mapping: Any) -> bool:
    if not isinstance(mapping, dict):
        return False
    points = mapping.get("points")
    powers = mapping.get("powers")
    count = mapping.get("num_points")
    if count != 2 or not isinstance(points, list) or len(points) != 4 or not isinstance(powers, list) or len(powers) != 2:
        return False
    return (
        all(math.isclose(float(left), float(right), abs_tol=FLOAT_TOLERANCE) for left, right in zip(points, (0.0, 0.0, 1.0, 1.0)))
        and all(math.isclose(float(power), 1.0, abs_tol=FLOAT_TOLERANCE) for power in powers if is_number(power))
        and all(is_number(power) for power in powers)
    )


def component_key(ref: ComponentRef) -> str:
    return str(ref)


def fallback_component_kind(name: str) -> ComponentKind:
    if name.startswith("osc_"):
        return ComponentKind.OSCILLATOR
    if name.startswith("sample_"):
        return ComponentKind.SAMPLER
    if name.startswith("filter_"):
        return ComponentKind.FILTER
    if name.startswith("env_"):
        return ComponentKind.ENVELOPE
    if name.startswith("lfo_"):
        return ComponentKind.LFO
    if name.startswith("random_"):
        return ComponentKind.RANDOM
    if name.startswith("modulation_"):
        return ComponentKind.MODULATION_SLOT
    if name.split("_", 1)[0] in EFFECTS:
        return ComponentKind.EFFECT
    return ComponentKind.GLOBAL


def owner_for_parameter(name: str, registry: dict[ComponentRef, ComponentDefinition]) -> ComponentDefinition | None:
    for definition in registry.values():
        if name in definition.owned_parameters:
            return definition
    return None


def owner_is_active(name: str, observation: PresetObservation, registry: dict[ComponentRef, ComponentDefinition]) -> bool:
    owner = owner_for_parameter(name, registry)
    if owner is not None:
        if owner.enable_parameter:
            if owner.enable_parameter == name:
                return True
            return observation.enabled_components.get(component_key(owner.ref), False)
        if owner.kind in {ComponentKind.LFO, ComponentKind.ENVELOPE, ComponentKind.RANDOM}:
            return str(owner.slot) in observation.active_slots[owner.kind.value]
        if owner.kind == ComponentKind.MODULATION_SLOT:
            return str(owner.slot) in observation.connected_modulation_slots
        return True
    kind = fallback_component_kind(name)
    if kind in {ComponentKind.OSCILLATOR, ComponentKind.FILTER, ComponentKind.EFFECT}:
        match = re.match(r"^[a-z_]+_(\d+)_", name)
        if match:
            return match.group(1) in observation.active_slots[kind.value]
        effect = name.split("_", 1)[0]
        return effect in observation.active_slots["effect"]
    if kind in {ComponentKind.LFO, ComponentKind.ENVELOPE, ComponentKind.RANDOM}:
        match = re.match(r"^[a-z]+_(\d+)_", name)
        return bool(match and match.group(1) in observation.active_slots[kind.value])
    if kind == ComponentKind.MODULATION_SLOT:
        match = re.match(r"^modulation_(\d+)_", name)
        return bool(match and match.group(1) in observation.connected_modulation_slots)
    return True


def route_source_family(source: str, registry: dict[ComponentRef, ComponentDefinition]) -> str:
    for definition in registry.values():
        if source in definition.modulation_source_names:
            return definition.kind.value
    return fallback_component_kind(source).value if source else "unknown"


def route_destination_family(destination: str, registry: dict[ComponentRef, ComponentDefinition]) -> str:
    definition = owner_for_parameter(destination, registry)
    if definition is not None:
        return definition.kind.value
    return fallback_component_kind(destination).value if destination else "unknown"


def introduced_feature(name: str) -> str | None:
    if SPECTRAL_PHASE_RE.match(name):
        return "vital_1.5_spectral_morph_phase"
    if MODULATION_RAMP_RE.match(name):
        return "vital_1.6_modulation_ramps"
    if name == "flanger_depth":
        return "vital_1.0.7_flanger_depth_extension"
    return None


def feature_support(name: str, version: str, settings: dict[str, Any]) -> tuple[bool, str | None]:
    feature = introduced_feature(name)
    if feature == "vital_1.5_spectral_morph_phase":
        return version_tuple(version) >= (1, 5), feature
    if feature == "vital_1.6_modulation_ramps":
        return version_tuple(version) >= (1, 6), feature
    if feature == "vital_1.0.7_flanger_depth_extension":
        return name in settings, feature
    return True, None


def default_for_parameter(name: str, schema: VitalSchema, parameter_specs: dict[str, Any] | None = None) -> tuple[float | object, str | None]:
    parameter = (parameter_specs or schema.parameters).get(name)
    if parameter is not None:
        return parameter.default, "pinned_parameter_atlas"
    return FEATURE_DEFAULT_UNAVAILABLE, introduced_feature(name)


def normalized_parameter_value(parameter: Any, value: float) -> float:
    """Convert a raw value to atlas position, including the Quartic scale."""
    if parameter.scale == "Quartic":
        span = parameter.maximum - parameter.minimum
        if span == 0:
            return 0.0
        position = (value - parameter.minimum) / span
        return position ** 4
    return parameter.normalized_from_raw(value)


@dataclass
class RouteObservation:
    slot: int
    source: str
    destination: str
    source_family: str
    destination_family: str
    connected: bool
    bypassed: bool
    amount_nonzero: bool
    live: bool
    bipolar: bool
    stereo: bool
    has_mapping: bool
    custom_mapping: bool
    explicit_linear_mapping: bool


@dataclass
class PresetObservation:
    version: str
    scalar_values: dict[str, float]
    missing_scalar_names: set[str]
    enabled_components: dict[str, bool]
    changed_components: dict[str, bool]
    routes: list[RouteObservation]
    active_slots: dict[str, tuple[str, ...]]
    connected_modulation_slots: tuple[str, ...]
    lfo_shape_custom_slots: tuple[str, ...]
    active_custom_lfo_slots: tuple[str, ...]
    active_wavetable_custom_slots: tuple[str, ...]
    active_wavetable_states: Counter[str]
    active_wavetable_types: Counter[str]
    sample_state: str
    audio_destinations: dict[str, str]
    major_families: dict[str, bool]


def analyze_document(
    document: dict[str, Any],
    schema: VitalSchema | None = None,
    registry: dict[ComponentRef, ComponentDefinition] | None = None,
    init_document: dict[str, Any] | None = None,
    parameter_specs: dict[str, Any] | None = None,
) -> PresetObservation:
    schema = schema or VitalSchema.load()
    registry = registry or component_registry(schema.schema_id)
    init_document = init_document or schema.load_init_document()
    parameter_specs = parameter_specs or schema.parameters
    settings = document.get("settings")
    if not isinstance(settings, dict):
        raise ValueError("settings must be an object")

    version = str(document.get("synth_version", ""))
    init_settings = init_document["settings"]
    scalar_values = {name: float(value) for name, value in settings.items() if is_number(value)}
    missing_scalar_names = set(parameter_specs) - set(scalar_values)
    enabled_components: dict[str, bool] = {}
    changed_components: dict[str, bool] = {}
    for ref, definition in registry.items():
        key = component_key(ref)
        if definition.enable_parameter:
            enabled_components[key] = nonzero(settings.get(definition.enable_parameter))
        changed_components[key] = any(
            not math.isclose(as_float(settings.get(name, init_settings.get(name, 0.0))), as_float(init_settings.get(name, 0.0)), abs_tol=FLOAT_TOLERANCE)
            for name in definition.owned_parameters
            if name in settings or name in init_settings
        )

    routes: list[RouteObservation] = []
    connected_modulation_slots: list[str] = []
    for index, connection in enumerate(settings.get("modulations", []), start=1):
        if not isinstance(connection, dict):
            continue
        source = str(connection.get("source", ""))
        destination = str(connection.get("destination", ""))
        connected = bool(source and destination)
        bypassed = nonzero(settings.get(f"modulation_{index}_bypass"))
        amount_nonzero = nonzero(settings.get(f"modulation_{index}_amount"))
        live = connected and not bypassed and amount_nonzero
        mapping = connection.get("line_mapping")
        route = RouteObservation(
            slot=index,
            source=source,
            destination=destination,
            source_family=route_source_family(source, registry),
            destination_family=route_destination_family(destination, registry),
            connected=connected,
            bypassed=bypassed,
            amount_nonzero=amount_nonzero,
            live=live,
            bipolar=nonzero(settings.get(f"modulation_{index}_bipolar")),
            stereo=nonzero(settings.get(f"modulation_{index}_stereo")),
            has_mapping=isinstance(mapping, dict),
            custom_mapping=isinstance(mapping, dict) and not line_mapping_is_linear(mapping),
            explicit_linear_mapping=isinstance(mapping, dict) and line_mapping_is_linear(mapping),
        )
        routes.append(route)
        if connected:
            connected_modulation_slots.append(str(index))

    live_routes = [route for route in routes if route.live]
    active_slots: dict[str, tuple[str, ...]] = {}
    for kind in (ComponentKind.OSCILLATOR, ComponentKind.FILTER, ComponentKind.EFFECT):
        active_slots[kind.value] = tuple(
            str(ref.slot)
            for ref, definition in registry.items()
            if ref.kind == kind and definition.enable_parameter and enabled_components.get(component_key(ref), False)
        )
    for kind in (ComponentKind.ENVELOPE, ComponentKind.LFO, ComponentKind.RANDOM):
        source_names = {
            source
            for route in live_routes
            for source in (route.source,)
            if route.source_family == kind.value
        }
        active_slots[kind.value] = tuple(sorted(
            (re.search(r"_(\d+)$", source).group(1) for source in source_names if re.search(r"_(\d+)$", source)),
            key=lambda value: int(value),
        ))
    active_slots["modulation"] = tuple(str(route.slot) for route in live_routes)
    active_slots["sampler"] = ("sampler",) if enabled_components.get("sampler", False) else ()

    lfos = settings.get("lfos", [])
    init_lfos = init_settings.get("lfos", [])
    lfo_shape_custom_slots = tuple(
        str(index)
        for index, lfo in enumerate(lfos[:8], start=1)
        if index > len(init_lfos) or lfo_shape_signature(lfo) != lfo_shape_signature(init_lfos[index - 1])
    )
    active_custom_lfo_slots = tuple(sorted(set(active_slots["lfo"]) & set(lfo_shape_custom_slots), key=int))

    wavetables = settings.get("wavetables", [])
    init_wavetables = init_settings.get("wavetables", [])
    active_custom_wavetable_slots: list[str] = []
    active_wavetable_states: Counter[str] = Counter()
    active_wavetable_types: Counter[str] = Counter()
    for index in range(1, 4):
        wavetable = wavetables[index - 1] if index <= len(wavetables) else None
        if enabled_components.get(f"oscillator[{index}]", False):
            active_wavetable_states[wavetable_status(wavetable)] += 1
            active_wavetable_types.update(wavetable_component_types(wavetable))
            init_wavetable = init_wavetables[index - 1] if index <= len(init_wavetables) else None
            if nested_signature(wavetable) != nested_signature(init_wavetable):
                active_custom_wavetable_slots.append(str(index))

    audio_destinations: dict[str, str] = {}
    for name in ("osc_1", "osc_2", "osc_3", "sample"):
        if name == "sample" and not enabled_components.get("sampler", False):
            continue
        slot = settings.get(f"{name}_destination")
        audio_destinations[name] = AUDIO_DESTINATIONS.get(int(slot), "unknown") if is_number(slot) else "unknown"

    major_families = {
        "oscillator": bool(active_slots["oscillator"]),
        "sampler": bool(active_slots["sampler"]),
        "filter": bool(active_slots["filter"]),
        "effect": bool(active_slots["effect"]),
        "envelope": bool(active_slots["envelope"]),
        "lfo": bool(active_slots["lfo"]),
        "random": bool(active_slots["random"]),
        "modulation": bool(live_routes),
    }
    return PresetObservation(
        version=version,
        scalar_values=scalar_values,
        missing_scalar_names=missing_scalar_names,
        enabled_components=enabled_components,
        changed_components=changed_components,
        routes=routes,
        active_slots=active_slots,
        connected_modulation_slots=tuple(connected_modulation_slots),
        lfo_shape_custom_slots=lfo_shape_custom_slots,
        active_custom_lfo_slots=active_custom_lfo_slots,
        active_wavetable_custom_slots=tuple(active_custom_wavetable_slots),
        active_wavetable_states=active_wavetable_states,
        active_wavetable_types=active_wavetable_types,
        sample_state=sample_status(settings.get("sample"), enabled_components.get("sampler", False)),
        audio_destinations=audio_destinations,
        major_families=major_families,
    )


@dataclass
class DistributionHistogram:
    """A bounded histogram for normalized or raw scalar observations."""

    fixed_minimum: float | None = None
    fixed_maximum: float | None = None
    bin_count: int = DISTRIBUTION_BIN_COUNT
    counts: list[int] = field(default_factory=lambda: [0] * DISTRIBUTION_BIN_COUNT)
    minimum: float | None = None
    maximum: float | None = None
    total: int = 0

    def __post_init__(self) -> None:
        if len(self.counts) != self.bin_count:
            self.counts = [0] * self.bin_count
        if self.fixed_minimum is not None:
            self.minimum = self.fixed_minimum
            self.maximum = self.fixed_maximum

    @property
    def fixed(self) -> bool:
        return self.fixed_minimum is not None and self.fixed_maximum is not None

    @property
    def count(self) -> int:
        return self.total

    def _index(self, value: float, minimum: float, maximum: float) -> int:
        if maximum <= minimum:
            return 0
        position = (value - minimum) / (maximum - minimum)
        return max(0, min(self.bin_count - 1, int(position * self.bin_count)))

    def _add_to_domain(self, value: float, minimum: float, maximum: float) -> None:
        self.counts[self._index(value, minimum, maximum)] += 1

    def add(self, value: float) -> None:
        if not math.isfinite(value):
            return
        if self.fixed:
            self._add_to_domain(value, float(self.fixed_minimum), float(self.fixed_maximum))
            self.total += 1
            return
        if self.count == 0:
            self.minimum = value
            self.maximum = value
            self.counts[0] = 1
            self.total = 1
            return
        assert self.minimum is not None and self.maximum is not None
        if value < self.minimum or value > self.maximum:
            old_minimum, old_maximum = self.minimum, self.maximum
            old_counts = self.counts
            self.minimum = min(self.minimum, value)
            self.maximum = max(self.maximum, value)
            self.counts = [0] * self.bin_count
            old_total = sum(old_counts)
            if old_maximum == old_minimum:
                self._add_to_domain(old_minimum, self.minimum, self.maximum)
                self.counts[self._index(old_minimum, self.minimum, self.maximum)] += old_total - 1
            else:
                for index, count in enumerate(old_counts):
                    if not count:
                        continue
                    center = old_minimum + (index + 0.5) * (old_maximum - old_minimum) / self.bin_count
                    self.counts[self._index(center, self.minimum, self.maximum)] += count
        self._add_to_domain(value, self.minimum, self.maximum)
        self.total += 1

    def _bounds(self, index: int) -> tuple[float, float] | None:
        if self.minimum is None or self.maximum is None:
            return None
        if self.maximum <= self.minimum:
            return self.minimum, self.maximum
        width = (self.maximum - self.minimum) / self.bin_count
        return self.minimum + index * width, self.minimum + (index + 1) * width

    def quantile(self, quantile: float) -> float | None:
        if not self.count:
            return None
        target = max(0.0, min(1.0, quantile)) * (self.count - 1)
        cumulative = 0
        for index, count in enumerate(self.counts):
            if not count:
                continue
            if cumulative + count > target:
                bounds = self._bounds(index)
                if bounds is None:
                    return None
                fraction = (target - cumulative) / count
                return bounds[0] + (bounds[1] - bounds[0]) * fraction
            cumulative += count
        return self.maximum

    def dominant_bins(self, limit: int = 5) -> list[dict[str, float | int]]:
        if not self.count:
            return []
        total = self.count
        rows = []
        for index, count in sorted(enumerate(self.counts), key=lambda item: (-item[1], item[0]))[:limit]:
            if not count:
                continue
            bounds = self._bounds(index)
            if bounds is None:
                continue
            rows.append({
                "bin": index,
                "lower": bounds[0],
                "upper": bounds[1],
                "count": count,
                "frequency": count / total,
            })
        return rows

    def as_dict(self) -> dict[str, Any]:
        quantiles = {
            f"p{int(level * 100):02d}": self.quantile(level)
            for level in (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
        }
        return {
            "domain": "normalized" if self.fixed else "raw",
            "bin_count": self.bin_count,
            "count": self.count,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "counts": self.counts,
            "quantiles": quantiles,
            "dominant_bins": self.dominant_bins(),
        }


@dataclass
class NumericSummary:
    count: int = 0
    minimum: float | None = None
    maximum: float | None = None
    total: float = 0.0
    total_squares: float = 0.0

    def add(self, value: float) -> None:
        self.count += 1
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)
        self.total += value
        self.total_squares += value * value

    def as_dict(self) -> dict[str, Any]:
        if not self.count:
            return {"count": 0, "min": None, "max": None, "mean": None, "stddev": None}
        mean = self.total / self.count
        variance = max(0.0, self.total_squares / self.count - mean * mean)
        return {
            "count": self.count,
            "min": self.minimum,
            "max": self.maximum,
            "mean": mean,
            "stddev": math.sqrt(variance),
        }


@dataclass
class ParameterAggregate:
    default: float | None
    default_source: str | None
    feature: str | None
    track_value_counts: bool
    parameter_type: str = "continuous"
    display_name: str = ""
    display_units: str = ""
    scale: str | None = None
    minimum: float | None = None
    maximum: float | None = None
    options: tuple[str, ...] = ()
    eligible: int = 0
    observed: int = 0
    missing: int = 0
    non_default: int = 0
    nonzero: int = 0
    active_eligible: int = 0
    active_non_default: int = 0
    modulated: int = 0
    default_count: int = 0
    value_counts: Counter[str] = field(default_factory=Counter)
    observed_values: NumericSummary = field(default_factory=NumericSummary)
    non_default_values: NumericSummary = field(default_factory=NumericSummary)
    distribution: DistributionHistogram | None = None
    raw_distribution: DistributionHistogram | None = None

    def __post_init__(self) -> None:
        if self.parameter_type == "continuous" and self.distribution is None:
            if self.minimum is not None and self.maximum is not None:
                self.distribution = DistributionHistogram(fixed_minimum=0.0, fixed_maximum=1.0)
                self.raw_distribution = DistributionHistogram()
            else:
                self.distribution = DistributionHistogram()

    @property
    def default_known(self) -> bool:
        return self.default is not None

    def add(
        self,
        value: float | None,
        *,
        owner_active: bool,
        modulated: bool,
        missing: bool = False,
        use_default: float | object = FEATURE_DEFAULT_UNAVAILABLE,
        distribution_value: float | None = None,
    ) -> None:
        self.eligible += 1
        if owner_active:
            self.active_eligible += 1
        self.missing += int(missing)
        if modulated:
            self.modulated += 1
        if value is None:
            return
        self.observed += 1
        self.observed_values.add(value)
        if abs(value) > FLOAT_TOLERANCE:
            self.nonzero += 1
        if self.track_value_counts:
            self.value_counts[json.dumps(value, separators=(",", ":"))] += 1
        elif self.distribution is not None:
            if self.raw_distribution is not None:
                self.raw_distribution.add(value)
            if distribution_value is not None:
                self.distribution.add(distribution_value)
        if use_default is not FEATURE_DEFAULT_UNAVAILABLE:
            changed = not math.isclose(value, float(use_default), abs_tol=FLOAT_TOLERANCE)
            if changed:
                self.non_default += 1
                self.non_default_values.add(value)
            if owner_active and changed:
                self.active_non_default += 1
            if not changed:
                self.default_count += 1

    def _value_label(self, value: float) -> str | None:
        if not self.options or self.minimum is None:
            return None
        index = int(round(value - self.minimum))
        if abs(value - (self.minimum + index)) > FLOAT_TOLERANCE or not 0 <= index < len(self.options):
            return None
        return self.options[index]

    def value_frequency_rows(self) -> list[dict[str, Any]]:
        total = sum(self.value_counts.values())
        rows = []
        for raw, count in sorted(self.value_counts.items(), key=lambda item: (-item[1], item[0])):
            value = json.loads(raw)
            row = {"value": value, "count": count, "frequency": count / total if total else 0.0}
            label = self._value_label(float(value)) if is_number(value) else None
            if label is not None:
                row["label"] = label
            rows.append(row)
        return rows

    def as_dict(self) -> dict[str, Any]:
        values = {
            "default": self.default,
            "default_known": self.default_known,
            "default_source": self.default_source,
            "introduced_feature": self.feature,
            "parameter_type": self.parameter_type,
            "display_name": self.display_name,
            "display_units": self.display_units,
            "scale": self.scale,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "options": list(self.options),
            "value_counts_supported": self.track_value_counts,
            "eligible": self.eligible,
            "observed": self.observed,
            "missing": self.missing,
            "non_default": self.non_default if self.default_known else None,
            "nonzero": self.nonzero,
            "active_eligible": self.active_eligible,
            "active_non_default": self.active_non_default if self.default_known else None,
            "modulated": self.modulated,
            "default_count": self.default_count if self.default_known else None,
            "default_frequency": self.default_count / self.observed if self.default_known and self.observed else None,
            "value_counts": dict(sorted(self.value_counts.items())),
            "value_frequencies": self.value_frequency_rows(),
            "observed_value_summary": self.observed_values.as_dict(),
            "non_default_value_summary": self.non_default_values.as_dict() if self.default_known else None,
        }
        if self.distribution is not None:
            normalized = self.distribution.as_dict()
            if self.raw_distribution is not None:
                raw = self.raw_distribution.as_dict()
                if normalized["count"] != self.observed:
                    values["distribution"] = raw
                    values["normalized_distribution"] = normalized
                    values["distribution_note"] = "raw fallback because observations exceeded pinned atlas bounds"
                else:
                    values["distribution"] = normalized
                    values["raw_distribution"] = raw
            else:
                values["distribution"] = normalized
        return values


@dataclass
class RouteAggregate:
    connected: int = 0
    bypassed: int = 0
    amount_nonzero: int = 0
    amount_zero: int = 0
    live: int = 0
    bipolar: int = 0
    stereo: int = 0
    custom_mapping: int = 0
    explicit_linear_mapping: int = 0
    connected_count_distribution: Counter[int] = field(default_factory=Counter)
    live_count_distribution: Counter[int] = field(default_factory=Counter)
    connected_source: Counter[str] = field(default_factory=Counter)
    connected_destination: Counter[str] = field(default_factory=Counter)
    connected_source_family: Counter[str] = field(default_factory=Counter)
    connected_destination_family: Counter[str] = field(default_factory=Counter)
    connected_family_pairs: Counter[tuple[str, str]] = field(default_factory=Counter)
    source: Counter[str] = field(default_factory=Counter)
    destination: Counter[str] = field(default_factory=Counter)
    source_family: Counter[str] = field(default_factory=Counter)
    destination_family: Counter[str] = field(default_factory=Counter)
    family_pairs: Counter[tuple[str, str]] = field(default_factory=Counter)

    def add(self, routes: list[RouteObservation]) -> None:
        connected = [route for route in routes if route.connected]
        live = [route for route in routes if route.live]
        self.connected_count_distribution[len(connected)] += 1
        self.live_count_distribution[len(live)] += 1
        for route in connected:
            self.connected += 1
            self.connected_source[route.source] += 1
            self.connected_destination[route.destination] += 1
            self.connected_source_family[route.source_family] += 1
            self.connected_destination_family[route.destination_family] += 1
            self.connected_family_pairs[(route.source_family, route.destination_family)] += 1
            self.bypassed += int(route.bypassed)
            self.amount_nonzero += int(route.amount_nonzero)
            self.amount_zero += int(not route.amount_nonzero)
            self.bipolar += int(route.bipolar)
            self.stereo += int(route.stereo)
            self.custom_mapping += int(route.custom_mapping)
            self.explicit_linear_mapping += int(route.explicit_linear_mapping)
        for route in live:
            self.live += 1
            self.source[route.source] += 1
            self.destination[route.destination] += 1
            self.source_family[route.source_family] += 1
            self.destination_family[route.destination_family] += 1
            self.family_pairs[(route.source_family, route.destination_family)] += 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "connected": self.connected,
            "bypassed": self.bypassed,
            "amount_nonzero": self.amount_nonzero,
            "amount_zero": self.amount_zero,
            "live": self.live,
            "bipolar": self.bipolar,
            "stereo": self.stereo,
            "custom_mapping": self.custom_mapping,
            "explicit_linear_mapping": self.explicit_linear_mapping,
            "connected_count_distribution": dict(sorted(self.connected_count_distribution.items())),
            "live_count_distribution": dict(sorted(self.live_count_distribution.items())),
            "connected_source": dict(self.connected_source.most_common()),
            "connected_destination": dict(self.connected_destination.most_common()),
            "connected_source_family": dict(self.connected_source_family.most_common()),
            "connected_destination_family": dict(self.connected_destination_family.most_common()),
            "connected_family_pairs": {f"{source} -> {destination}": count for (source, destination), count in self.connected_family_pairs.most_common()},
            "source": dict(self.source.most_common()),
            "destination": dict(self.destination.most_common()),
            "source_family": dict(self.source_family.most_common()),
            "destination_family": dict(self.destination_family.most_common()),
            "family_pairs": {f"{source} -> {destination}": count for (source, destination), count in self.family_pairs.most_common()},
        }


@dataclass
class Aggregate:
    parsed_files: int = 0
    failed_files: int = 0
    failure_reasons: Counter[str] = field(default_factory=Counter)
    versions: Counter[str] = field(default_factory=Counter)
    scalar_shapes: Counter[int] = field(default_factory=Counter)
    components: dict[str, dict[str, Any]] = field(default_factory=dict)
    family_count_distributions: dict[str, Counter[int]] = field(default_factory=lambda: defaultdict(Counter))
    family_patterns: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    slot_prevalence: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    parameters: dict[str, ParameterAggregate] = field(default_factory=dict)
    routes: RouteAggregate = field(default_factory=RouteAggregate)
    nested: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    audio_destinations: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    cooccurrence: Counter[tuple[str, str]] = field(default_factory=Counter)

    def _component(self, key: str, family: str) -> dict[str, Any]:
        if key not in self.components:
            self.components[key] = {"family": family, "eligible": 0, "enabled": 0, "changed": 0, "routed": 0}
        return self.components[key]

    def add_failure(self, reason: str) -> None:
        self.failed_files += 1
        self.failure_reasons[reason] += 1

    def add_observation(
        self,
        observation: PresetObservation,
        schema: VitalSchema,
        registry: dict[ComponentRef, ComponentDefinition],
        parameter_specs: dict[str, Any] | None = None,
    ) -> None:
        parameter_specs = parameter_specs or schema.parameters
        self.parsed_files += 1
        self.versions[observation.version or "<missing>"] += 1
        self.scalar_shapes[len(observation.scalar_values)] += 1
        self.routes.add(observation.routes)

        for key, enabled in observation.enabled_components.items():
            family = key.split("[", 1)[0]
            component = self._component(key, family)
            component["eligible"] += 1
            component["enabled"] += int(enabled)
            component["changed"] += int(observation.changed_components.get(key, False))
        for ref in registry:
            key = component_key(ref)
            component = self._component(key, ref.kind.value)
            component["eligible"] = max(component["eligible"], self.parsed_files)
            if ref.kind in {ComponentKind.LFO, ComponentKind.ENVELOPE, ComponentKind.RANDOM} and ref.slot is not None:
                component["routed"] += int(str(ref.slot) in observation.active_slots[ref.kind.value])
            elif ref.kind == ComponentKind.MODULATION_SLOT and ref.slot is not None:
                component["routed"] += int(str(ref.slot) in observation.connected_modulation_slots)

        for family, slots in observation.active_slots.items():
            pattern = "+".join(slots) if slots else "none"
            family_name = "modulation_slot" if family == "modulation" else family
            self.family_count_distributions[family_name][len(slots)] += 1
            self.family_patterns[family_name][pattern] += 1
            self.slot_prevalence[family_name].update(slots)

        self.nested["lfo_shape"]["custom"] += int(bool(observation.lfo_shape_custom_slots))
        self.nested["lfo_shape"]["stock_or_default"] += int(not observation.lfo_shape_custom_slots)
        self.nested["lfo_shape"]["custom_slot"] += len(observation.lfo_shape_custom_slots)
        self.nested["lfo_shape"]["operational_custom_slot"] += len(observation.active_custom_lfo_slots)
        self.nested["wavetable"]["active_non_init"] += int(bool(observation.active_wavetable_custom_slots))
        self.nested["wavetable"]["active_non_init_slot"] += len(observation.active_wavetable_custom_slots)
        self.nested["wavetable"]["active_stock_or_init"] += int(not observation.active_wavetable_custom_slots)
        for status, count in observation.active_wavetable_states.items():
            self.nested["wavetable"][f"{status}_slot"] += count
            self.nested["wavetable"][f"{status}_preset"] += int(count > 0)
        self.nested["sampler"][observation.sample_state] += 1
        self.nested["wavetable_component_type"].update(observation.active_wavetable_types)
        for name, destination in observation.audio_destinations.items():
            self.audio_destinations[name][destination] += 1

        active_major = [family for family in MAJOR_FAMILIES if observation.major_families.get(family, False)]
        for index, left in enumerate(active_major):
            for right in active_major[index + 1 :]:
                self.cooccurrence[(left, right)] += 1

        modulated_destinations = {route.destination for route in observation.routes if route.live}
        for name in set(parameter_specs) | set(observation.scalar_values):
            supported, feature = feature_support(name, observation.version, observation.scalar_values)
            if not supported:
                continue
            default, default_source = default_for_parameter(name, schema, parameter_specs)
            if name not in self.parameters:
                spec = parameter_specs.get(name)
                parameter_type = "categorical" if spec is not None and (spec.is_discrete or spec.scale == "Indexed") else "continuous"
                self.parameters[name] = ParameterAggregate(
                    default=float(default) if default is not FEATURE_DEFAULT_UNAVAILABLE else None,
                    default_source=default_source if default is not FEATURE_DEFAULT_UNAVAILABLE else None,
                    feature=feature,
                    track_value_counts=parameter_type == "categorical",
                    parameter_type=parameter_type,
                    display_name=spec.display_name if spec is not None else name,
                    display_units=spec.display_units if spec is not None else "",
                    scale=spec.scale if spec is not None else None,
                    minimum=spec.minimum if spec is not None else None,
                    maximum=spec.maximum if spec is not None else None,
                    options=spec.options if spec is not None else (),
                )
            parameter = self.parameters[name]
            value = observation.scalar_values.get(name)
            if value is None and default is not FEATURE_DEFAULT_UNAVAILABLE:
                value = float(default)
            distribution_value = None
            if value is not None and parameter.parameter_type == "continuous":
                spec = parameter_specs.get(name)
                if spec is None:
                    distribution_value = value
                else:
                    try:
                        distribution_value = normalized_parameter_value(spec, value)
                    except ValueError:
                        distribution_value = None
            parameter.add(
                value,
                owner_active=owner_is_active(name, observation, registry),
                modulated=name in modulated_destinations,
                missing=name not in observation.scalar_values,
                use_default=default,
                distribution_value=distribution_value,
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "parsed_files": self.parsed_files,
            "failed_files": self.failed_files,
            "failure_reasons": dict(self.failure_reasons.most_common()),
            "versions": dict(self.versions.most_common()),
            "scalar_shapes": {str(key): value for key, value in sorted(self.scalar_shapes.items())},
            "components": dict(sorted(self.components.items())),
            "family_count_distributions": {
                family: {str(key): value for key, value in sorted(counter.items())}
                for family, counter in sorted(self.family_count_distributions.items())
            },
            "family_patterns": {family: dict(counter.most_common()) for family, counter in sorted(self.family_patterns.items())},
            "slot_prevalence": {family: dict(counter.most_common()) for family, counter in sorted(self.slot_prevalence.items())},
            "parameters": {name: value.as_dict() for name, value in sorted(self.parameters.items())},
            "routes": self.routes.as_dict(),
            "nested": {name: dict(counter.most_common()) for name, counter in sorted(self.nested.items())},
            "audio_destinations": {name: dict(counter.most_common()) for name, counter in sorted(self.audio_destinations.items())},
            "cooccurrence": {f"{left} + {right}": count for (left, right), count in sorted(self.cooccurrence.items())},
        }


def build_census(root: Path, schema: VitalSchema | None = None, progress_every: int = 250) -> dict[str, Any]:
    schema = schema or VitalSchema.load()
    parameter_specs = schema.parameters
    registry = component_registry(schema.schema_id)
    init_document = schema.load_init_document()
    files = sorted(iter_vital_files(root), key=lambda path: path.as_posix().lower())
    weighted = Aggregate()
    deduplicated = Aggregate()
    seen_hashes: set[str] = set()
    duplicate_files = 0
    duplicate_groups = 0
    started = time.monotonic()
    for file_id, path in enumerate(files, start=1):
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        duplicate = digest in seen_hashes
        if duplicate:
            duplicate_files += 1
        else:
            seen_hashes.add(digest)
            duplicate_groups += 1
        try:
            _, document = load_json_bytes(path)
            observation = analyze_document(document, schema, registry, init_document, parameter_specs)
        except Exception as exc:
            weighted.add_failure(type(exc).__name__)
            if not duplicate:
                deduplicated.add_failure(type(exc).__name__)
        else:
            weighted.add_observation(observation, schema, registry, parameter_specs)
            if not duplicate:
                deduplicated.add_observation(observation, schema, registry, parameter_specs)
        if progress_every and (file_id % progress_every == 0 or file_id == len(files)):
            elapsed = time.monotonic() - started
            rate = file_id / elapsed if elapsed else 0.0
            print(f"[{file_id:,}/{len(files):,}] parsed={weighted.parsed_files:,} failed={weighted.failed_files:,} rate={rate:.1f} files/s", flush=True)

    return {
        "artifact_schema": "obruxo_vital_usage_census_v2",
        "script_version": SCRIPT_VERSION,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "corpus_root_policy": "local external input; paths and raw payloads are not retained",
        "platform": platform.platform(),
        "python": sys.version,
        "schema_id": schema.schema_id,
        "discovered_files": len(files),
        "exact_duplicate_files": duplicate_files,
        "exact_unique_content_groups": duplicate_groups,
        "exact_unique_parsed_content_groups": deduplicated.parsed_files,
        "definitions": {
            "enabled": "direct *_on control is non-zero",
            "connected_route": "modulation source and destination are both populated",
            "live_route": "connected, not bypassed, and modulation amount is non-zero",
            "zero_amount_route": "connected route whose modulation amount is zero or unavailable",
            "operational_envelope_lfo_random": "the source appears in at least one live route; always-wired synth behavior is not inferred",
            "non_default": "scalar differs from the pinned atlas default; missing common scalar keys are treated as defaults",
            "categorical_value_frequency": "exact raw ordinal frequency among observed values; atlas options are included when available",
            "continuous_distribution": "64-bin distribution in normalized atlas control position when bounds are known, otherwise adaptive raw-value bins; quantiles are histogram estimates",
            "custom_lfo_shape": "shape descriptor differs from the canonical init shape, ignoring display name",
            "non_init_wavetable": "semantic descriptor differs from the canonical init wavetable, with payloads represented only by in-process hashes; this is diagnostic and is not itself a custom-content claim",
            "wavetable_content": "active wavetable slots are classified by known stock names or left unresolved; names do not prove payload provenance",
            "sampler_content": "named stock/unresolved classification; sample payloads are never compared or published",
            "introduced_parameter_defaults": "not established by the pinned atlas; report nonzero observations and leave non-default unsupported",
            "duplicate_sensitivity": "file_weighted counts every parsed file; exact_deduplicated counts one representative per raw-byte SHA-256",
        },
        "file_weighted": weighted.as_dict(),
        "exact_deduplicated": deduplicated.as_dict(),
    }


def read_json(path: Path) -> Any:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    payload = json.dumps(value, separators=(",", ":"), sort_keys=True, ensure_ascii=False).encode("utf-8")
    # A fixed timestamp keeps unchanged snapshots byte-identical across runs.
    temporary.write_bytes(gzip.compress(payload, compresslevel=9, mtime=0) if path.suffix == ".gz" else payload)
    temporary.replace(path)


def write_parameter_csv(path: Path, census: dict[str, Any]) -> None:
    rows = []
    for name, values in census["file_weighted"]["parameters"].items():
        unique = census["exact_deduplicated"]["parameters"].get(name, {})
        distribution = values.get("distribution", {})
        summary = values.get("observed_value_summary", {})
        unique_distribution = unique.get("distribution", {})
        unique_summary = unique.get("observed_value_summary", {})
        rows.append({
            "parameter": name,
            "introduced_feature": values["introduced_feature"] or "shared",
            "parameter_type": values.get("parameter_type", "continuous"),
            "display_name": values.get("display_name", name),
            "display_units": values.get("display_units", ""),
            "scale": values.get("scale", ""),
            "minimum": values.get("minimum"),
            "maximum": values.get("maximum"),
            "default": values.get("default"),
            "options": json.dumps(values.get("options", []), ensure_ascii=False),
            "default_known": values["default_known"],
            "file_weighted_eligible": values["eligible"],
            "file_weighted_observed": values["observed"],
            "file_weighted_missing": values["missing"],
            "file_weighted_default_count": values.get("default_count"),
            "file_weighted_default_frequency": values.get("default_frequency"),
            "file_weighted_non_default": values["non_default"],
            "file_weighted_active_eligible": values["active_eligible"],
            "file_weighted_active_non_default": values["active_non_default"],
            "file_weighted_modulated": values["modulated"],
            "file_weighted_nonzero": values["nonzero"],
            "file_weighted_observed_min": summary.get("min"),
            "file_weighted_observed_max": summary.get("max"),
            "file_weighted_observed_mean": summary.get("mean"),
            "file_weighted_observed_stddev": summary.get("stddev"),
            "file_weighted_distribution_domain": distribution.get("domain"),
            "file_weighted_p01": distribution.get("quantiles", {}).get("p01"),
            "file_weighted_p05": distribution.get("quantiles", {}).get("p05"),
            "file_weighted_p25": distribution.get("quantiles", {}).get("p25"),
            "file_weighted_p50": distribution.get("quantiles", {}).get("p50"),
            "file_weighted_p75": distribution.get("quantiles", {}).get("p75"),
            "file_weighted_p95": distribution.get("quantiles", {}).get("p95"),
            "file_weighted_p99": distribution.get("quantiles", {}).get("p99"),
            "file_weighted_dominant_bins": json.dumps(distribution.get("dominant_bins", []), separators=(",", ":")),
            "deduplicated_eligible": unique.get("eligible", 0),
            "deduplicated_observed": unique.get("observed", 0),
            "deduplicated_missing": unique.get("missing", 0),
            "deduplicated_default_count": unique.get("default_count"),
            "deduplicated_default_frequency": unique.get("default_frequency"),
            "deduplicated_non_default": unique.get("non_default"),
            "deduplicated_active_non_default": unique.get("active_non_default"),
            "deduplicated_modulated": unique.get("modulated", 0),
            "deduplicated_nonzero": unique.get("nonzero", 0),
            "deduplicated_observed_min": unique_summary.get("min"),
            "deduplicated_observed_max": unique_summary.get("max"),
            "deduplicated_observed_mean": unique_summary.get("mean"),
            "deduplicated_observed_stddev": unique_summary.get("stddev"),
            "deduplicated_distribution_domain": unique_distribution.get("domain"),
            "deduplicated_p01": unique_distribution.get("quantiles", {}).get("p01"),
            "deduplicated_p05": unique_distribution.get("quantiles", {}).get("p05"),
            "deduplicated_p25": unique_distribution.get("quantiles", {}).get("p25"),
            "deduplicated_p50": unique_distribution.get("quantiles", {}).get("p50"),
            "deduplicated_p75": unique_distribution.get("quantiles", {}).get("p75"),
            "deduplicated_p95": unique_distribution.get("quantiles", {}).get("p95"),
            "deduplicated_p99": unique_distribution.get("quantiles", {}).get("p99"),
            "deduplicated_dominant_bins": json.dumps(unique_distribution.get("dominant_bins", []), separators=(",", ":")),
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["parameter"])
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_categorical_values_csv(path: Path, census: dict[str, Any]) -> None:
    rows = []
    weighted_parameters = census["file_weighted"]["parameters"]
    unique_parameters = census["exact_deduplicated"]["parameters"]
    for name, values in weighted_parameters.items():
        if values.get("parameter_type") != "categorical":
            continue
        weighted_rows = {json.dumps(row["value"], separators=(",", ":")): row for row in values.get("value_frequencies", [])}
        unique_rows = {
            json.dumps(row["value"], separators=(",", ":")): row
            for row in unique_parameters.get(name, {}).get("value_frequencies", [])
        }
        for raw in sorted(set(weighted_rows) | set(unique_rows)):
            weighted = weighted_rows.get(raw, {})
            unique = unique_rows.get(raw, {})
            rows.append({
                "parameter": name,
                "display_name": values.get("display_name", name),
                "scale": values.get("scale", ""),
                "minimum": values.get("minimum"),
                "maximum": values.get("maximum"),
                "value": json.loads(raw),
                "label": weighted.get("label", unique.get("label", "")),
                "file_weighted_count": weighted.get("count", 0),
                "file_weighted_frequency": weighted.get("frequency", 0.0),
                "deduplicated_count": unique.get("count", 0),
                "deduplicated_frequency": unique.get("frequency", 0.0),
            })
    fieldnames = [
        "parameter", "display_name", "scale", "minimum", "maximum", "value", "label",
        "file_weighted_count", "file_weighted_frequency", "deduplicated_count", "deduplicated_frequency",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def histogram_bin_rows(distribution: dict[str, Any]) -> list[dict[str, Any]]:
    counts = distribution.get("counts", [])
    minimum = distribution.get("minimum")
    maximum = distribution.get("maximum")
    if not counts or minimum is None or maximum is None:
        return []
    bin_count = len(counts)
    width = (maximum - minimum) / bin_count if maximum > minimum else 0.0
    return [
        {
            "bin": index,
            "lower": minimum + index * width,
            "upper": minimum + (index + 1) * width,
            "count": count,
        }
        for index, count in enumerate(counts)
    ]


def write_continuous_bins_csv(path: Path, census: dict[str, Any]) -> None:
    rows = []
    weighted_parameters = census["file_weighted"]["parameters"]
    unique_parameters = census["exact_deduplicated"]["parameters"]
    for name, values in weighted_parameters.items():
        if values.get("parameter_type") != "continuous":
            continue
        weighted_distribution = values.get("distribution", {})
        unique_distribution = unique_parameters.get(name, {}).get("distribution", {})
        weighted_bins = {row["bin"]: row for row in histogram_bin_rows(weighted_distribution)}
        unique_bins = {row["bin"]: row for row in histogram_bin_rows(unique_distribution)}
        for index in range(max(len(weighted_distribution.get("counts", [])), len(unique_distribution.get("counts", [])))):
            weighted = weighted_bins.get(index, {})
            unique = unique_bins.get(index, {})
            rows.append({
                "parameter": name,
                "display_name": values.get("display_name", name),
                "scale": values.get("scale", ""),
                "minimum": values.get("minimum"),
                "maximum": values.get("maximum"),
                "domain": weighted_distribution.get("domain", unique_distribution.get("domain", "")),
                "bin": index,
                "file_weighted_lower": weighted.get("lower"),
                "file_weighted_upper": weighted.get("upper"),
                "file_weighted_count": weighted.get("count", 0),
                "file_weighted_frequency": weighted.get("count", 0) / values.get("observed", 1) if values.get("observed") else 0.0,
                "deduplicated_lower": unique.get("lower"),
                "deduplicated_upper": unique.get("upper"),
                "deduplicated_count": unique.get("count", 0),
                "deduplicated_frequency": unique.get("count", 0) / unique_parameters.get(name, {}).get("observed", 1) if unique_parameters.get(name, {}).get("observed") else 0.0,
            })
    fieldnames = [
        "parameter", "display_name", "scale", "minimum", "maximum", "domain", "bin",
        "file_weighted_lower", "file_weighted_upper", "file_weighted_count", "file_weighted_frequency",
        "deduplicated_lower", "deduplicated_upper", "deduplicated_count", "deduplicated_frequency",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a sanitized semantic Vital preset usage census.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("root", nargs="?", type=Path, help=f"Read-only corpus root (default: {DEFAULT_ROOT})")
    source.add_argument("--from-census", type=Path, help="Read an existing .json or .json.gz aggregate without scanning the corpus")
    parser.add_argument("--output", type=Path, help=f"Write aggregate as .json.gz or .json (default when scanning: {DEFAULT_OUTPUT})")
    parser.add_argument("--parameter-csv", type=Path, help="Optionally export the per-parameter summary CSV")
    parser.add_argument("--categorical-csv", type=Path, help="Optionally export categorical value frequencies as CSV")
    parser.add_argument("--continuous-csv", type=Path, help="Optionally export continuous histogram bins as CSV")
    parser.add_argument("--progress-every", type=int, default=250)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.progress_every < 0:
        print("error: --progress-every cannot be negative", file=sys.stderr)
        return 2
    if args.from_census:
        census = read_json(args.from_census.expanduser().resolve())
    else:
        root = (args.root or DEFAULT_ROOT).expanduser().resolve()
        if not root.is_dir():
            print(f"error: corpus root does not exist or is not a directory: {root}", file=sys.stderr)
            return 2
        census = build_census(root, progress_every=args.progress_every)
    output = args.output or (DEFAULT_OUTPUT if not args.from_census else None)
    if output:
        write_json(output.expanduser().resolve(), census)
        print(f"Wrote {output.resolve()}")
    for path, writer in ((args.parameter_csv, write_parameter_csv), (args.categorical_csv, write_categorical_values_csv), (args.continuous_csv, write_continuous_bins_csv)):
        if path:
            writer(path.expanduser().resolve(), census)
            print(f"Wrote {path.resolve()}")
    print(f"Parsed files: {census['file_weighted']['parsed_files']:,}; exact unique parsed content: {census['exact_unique_parsed_content_groups']:,}")
    return 0 if census["file_weighted"]["parsed_files"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

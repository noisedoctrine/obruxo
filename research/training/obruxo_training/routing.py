"""All 64 stored connections, with explicit presence and fixed-grid line maps."""
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random
import re

import numpy as np

from obruxo_data.hashing import canonical_json, canonical_sha256

from .codec import PilotCodec, PilotTarget
from .full_scalar import EFFECTS
from .lfo_bank import curve_to_lfo, sample_lfo
from .lfo_codec import LfoCurveCodec

FAMILIES = ("direct", "parallel", "indexed_amount", "modulator_feedback", "dense")
LINEAR = tuple(float(value) for value in np.linspace(1, 0, 97, dtype=np.float32))


def authored_routes(index, ordinal, primary, sampler):
    """Independent authored targets; positional references are never relabeled."""
    routes = [{"source": "", "destination": ""} for _ in range(64)]
    controls = {f"modulation_{slot}_{name}": 0. for slot in range(1, 65) for name in ("amount", "power", "bipolar", "stereo", "bypass")}
    family = ordinal % len(FAMILIES)
    slots = [(ordinal + 13 * offset) % 64 for offset in range(16)]
    source, other = f"lfo_{primary + 1}", f"lfo_{(primary + 1) % 8 + 1}"
    destination = "sample_pan" if sampler else "osc_1_pan"

    def add(position, name, target, amount=.03, mapping=False):
        slot = slots[position]
        routes[slot] = {"source": name, "destination": target}
        controls[f"modulation_{slot + 1}_amount"] = amount
        controls[f"modulation_{slot + 1}_bipolar"] = float((index + position) % 2)
        controls[f"modulation_{slot + 1}_power"] = float(np.float32((position % 3 - 1) * .2))
        if mapping:
            routes[slot]["line_mapping"] = {"name": f"Authored route {slot + 1}", "num_points": 3,
                "points": [0., 1., .5, float(np.float32(.2 + .1 * (index % 5))), 1., 0.],
                "powers": [float(np.float32(.25)), 0., 0.], "smooth": bool(index % 2)}

    if family == 0:
        add(0, source, destination, mapping=True)
    elif family == 1:
        for offset, name in enumerate((source, other, "env_1", "velocity")):
            add(offset, name, destination, .008, mapping=offset == 2)
    elif family == 2:
        add(0, source, destination, .02, mapping=True)
        add(1, other, f"modulation_{slots[0] + 1}_amount", .01)
    elif family == 3:
        add(0, source, f"{other}_frequency", .01)
        add(1, other, f"{source}_frequency", .01)
        add(2, source, destination, .02, mapping=True)
    else:
        for offset in range(16):
            add(offset, f"lfo_{offset % 8 + 1}", destination, .002, mapping=offset % 4 == 0)
    return routes, controls, FAMILIES[family]


class OrderedRoutingCodec:
    generator = LfoCurveCodec.generator
    model_family = "ordered_routing"
    balanced_loss = True
    lossy_curves = True
    lossy_route_maps = True

    def __init__(self, asset_bank, lfo_bank):
        self.curve = LfoCurveCodec(asset_bank, lfo_bank)
        self.base, self.lfo_bank = self.curve.base, self.curve.lfo_bank
        self.schema, self.catalog, self.contract = self.curve.schema, self.curve.catalog, self.curve.contract
        self.roots = self.curve.roots
        self._ncont, self._ncat = self.curve._ncont, self.curve._ncat
        self._curve_cont, self._curve_cat = len(self.curve.continuous), len(self.curve.categorical)
        self._map_start = self._curve_cont
        self.sources, self.destinations = tuple(sorted(self.schema.sources)), tuple(sorted(self.schema.destinations))
        fields = (("present", (0, 1)), ("source", self.sources), ("destination", self.destinations), ("map_present", (0, 1)))
        self.continuous = self.curve.continuous + tuple((f"route_{slot}_map_{point}", 0., 1., LINEAR[point])
                                                       for slot in range(1, 65) for point in range(97))
        self.categorical = self.curve.categorical + tuple((f"route_{slot}_{name}", values) for slot in range(1, 65) for name, values in fields)
        self.continuous_groups = self.curve.continuous_groups + ("route_maps",) * (64 * 97)
        self.categorical_groups = self.curve.categorical_groups + tuple(name for _ in range(64) for name in
                                              ("route_presence", "route_structure", "route_structure", "route_map_presence"))
        self.cyclic_indices = self.curve.cyclic_indices
        self._ordinal = {}
        for split in ("train", "validation", "test"):
            for index, entry in enumerate(entry for entry in self.roots.assignments if entry.split == split):
                self._ordinal[entry.state_id] = index
        for name in ("active_routes", "active_lfos", "continuous_mask", "categorical_mask", "categorical_weights", "active_route_maps"):
            setattr(self, name, lru_cache(maxsize=512)(getattr(self, name)))

    @property
    def codec_id(self):
        return canonical_sha256(self.to_dict())

    def to_dict(self):
        spec = self.curve.to_dict()
        spec.update(codec="authored_ordered_routing_v1", continuous=spec["continuous"] + [
                    {"name": name, "minimum": lo, "maximum": hi, "inactive_default": default}
                    for name, lo, hi, default in self.continuous[self._curve_cont:]],
                    categorical=[{"name": name, "values": list(values)} for name, values in self.categorical],
                    loss="family_balanced_routing_presence_and_original_curve_mse_v1", route_map_loss_weight=1.,
                    fixed_routes=None, families=list(FAMILIES), route_slots=64,
                    routing_policy="preserve_slot_order_all_five_controls_and_literal_destination_references",
                    presence_loss="equal_positive_negative_means_per_example", remap_policy="connected_slots_only_97_float32_samples_canonical_name_lossy",
                    source_policy="original_authored_lfos_and_route_maps_audio_with_lossy_curve_targets_v1",
                    routing_curriculum={"policy": "original_ordered_routes_v1", "source_sha256": hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
                                        "master_volume_stored": 3844., "active_routes": [1, 4, 2, 3, 16]},
                    nested_policy="learned_64_ordered_connections_and_maps_fixed_custom_warps_and_seeds",
                    scope="all_775_scalars_registered_assets_eight_lfos_64_ordered_routes_and_maps")
        return spec

    def validate_target(self, target):
        PilotCodec.validate_target(self, target)

    def _curve_target(self, target):
        return PilotTarget(target.continuous[:self._curve_cont], target.categorical[:self._curve_cat])

    def canonical_target(self, target):
        self.validate_target(target)
        curve = self.curve.canonical_target(self._curve_target(target))
        maps = tuple(float(np.float32(value)) for value in target.continuous[self._map_start:])
        return PilotTarget(curve.continuous + maps, curve.categorical + target.categorical[self._curve_cat:])

    def _route_data(self, routes):
        flags, values = [], []
        for route in routes:
            present = bool(route["source"])
            if not present and "line_mapping" in route:
                raise ValueError("Vital does not preserve a line mapping on an empty connection")
            flags.extend((int(present), self.sources.index(route["source"]) if present else 0,
                          self.destinations.index(route["destination"]) if present else 0, int("line_mapping" in route)))
            values.extend(sample_lfo(route["line_mapping"]).tolist() if "line_mapping" in route else LINEAR)
        return tuple(flags), tuple(values)

    def routes(self, target):
        self.validate_target(target)
        flags = target.categorical[self._curve_cat:]
        maps = target.continuous[self._map_start:]
        result = []
        for slot in range(64):
            present, source, destination, mapped = flags[slot * 4:slot * 4 + 4]
            route = {"source": self.sources[source] if present else "", "destination": self.destinations[destination] if present else ""}
            if present and mapped:
                route["line_mapping"] = curve_to_lfo(maps[slot * 97:(slot + 1) * 97], name=f"OBRUXO route {slot + 1}")
            result.append(route)
        return result

    def _join(self, curve, routes):
        flags, maps = self._route_data(routes)
        return self.canonical_target(PilotTarget(curve.continuous + maps, curve.categorical + flags))

    def settings(self, target):
        settings = self.curve.settings(self._curve_target(target))
        settings["modulations"] = self.routes(target)
        return settings

    def decode(self, target):
        target = self.canonical_target(target)
        document = json.loads(self.curve.decode(self._curve_target(target)))
        document["settings"]["modulations"] = self.routes(target)
        self.schema.validate(document).require_valid()
        return canonical_json(document).encode()

    def encode(self, raw):
        document, report = self.schema.loads(raw)
        report.require_valid()
        routes = deepcopy(document["settings"]["modulations"])
        document["settings"]["modulations"] = deepcopy(self.curve.routes)
        return self._join(self.curve.encode(canonical_json(document).encode()), routes)

    def default_target(self):
        curve = self.curve.default_target()
        settings = self.base.settings(self.curve._base_target(curve))
        settings["volume"] = 3844.
        scalar = self.base.scalar._target(settings)
        curve = PilotTarget(scalar.continuous + curve.continuous[self._ncont:], scalar.categorical + curve.categorical[len(scalar.categorical):])
        return self._join(curve, [{"source": "", "destination": ""} for _ in range(64)])

    def supervision_target(self, target):
        curve = self.curve.supervision_target(self._curve_target(target))
        return PilotTarget(curve.continuous + target.continuous[self._map_start:], curve.categorical + target.categorical[self._curve_cat:])

    def curves(self, target):
        return self.curve.curves(self._curve_target(target))

    def source_curves(self, raw):
        return self.curve.source_curves(raw)

    def source_route_maps(self, raw):
        routes = json.loads(raw)["settings"]["modulations"]
        return np.asarray(self._route_data(routes)[1], dtype=np.float32).reshape(64, 97)

    def lineage_root(self, rng, index):
        return self.curve.lineage_root(rng, index)

    def validate_manifest(self, manifest):
        self.curve.validate_manifest(manifest)

    def _authored(self, index, root, sampler):
        return authored_routes(index, self._ordinal[root], self.curve._primary[root], sampler)

    def sample_target(self, rng, index):
        probe = random.Random()
        probe.setstate(rng.getstate())
        root = self.lineage_root(probe, index)
        curve = self.curve.sample_target(rng, index)
        settings = self.base.settings(self.curve._base_target(curve))
        routes, controls, _ = self._authored(index, root, bool(settings["sample_on"]))
        settings.update(controls)
        settings["volume"] = 3844.
        for slot in range(1, 9):
            settings[f"lfo_{slot}_frequency"], settings[f"lfo_{slot}_sync"] = float(np.float32((slot % 3 - 1) * .25)), 0.
        scalar = self.base.scalar._target(settings)
        curve = PilotTarget(scalar.continuous + curve.continuous[self._ncont:], scalar.categorical + curve.categorical[len(scalar.categorical):])
        return self._join(curve, routes)

    def source_preset(self, target, index):
        from .pilot import seeded_random
        root = self.lineage_root(seeded_random(self.generator, self.roots.seed, index), index)
        curve = self._curve_target(target)
        document = json.loads(self.curve.source_preset(curve, index))
        routes, _, _ = self._authored(index, root, bool(document["settings"]["sample_on"]))
        if self._join(curve, routes) != self.canonical_target(target):
            raise ValueError("saved routing target does not match its original ordered source")
        document["settings"]["modulations"] = routes
        self.schema.validate(document).require_valid()
        return canonical_json(document).encode()

    def active_routes(self, target):
        routes = self.routes(target)
        settings = self.base._values(self.curve._base_target(self._curve_target(target)))
        enabled = [bool(route["source"]) and not settings[f"modulation_{slot + 1}_bypass"] for slot, route in enumerate(routes)]
        potential = {slot for slot, value in enumerate(enabled) if value and settings[f"modulation_{slot + 1}_amount"] != 0}
        for _ in range(64):
            driven = {int(match[1]) - 1 for slot in potential if (match := re.fullmatch(r"modulation_(\d+)_amount", routes[slot]["destination"]))}
            found = {slot for slot in driven if enabled[slot]}
            if found <= potential:
                break
            potential |= found
        enabled = [value and slot in potential for slot, value in enumerate(enabled)]
        active, sources = set(), set()

        def audible(destination):
            match = re.match(r"modulation_(\d+)_", destination)
            if match:
                return int(match[1]) - 1 in active
            match = re.match(r"(lfo_\d|random_\d|env_\d)_", destination)
            if match:
                return match[1] in sources or match[1] == "env_1"
            match = re.match(r"(osc_\d|filter_[12]|sample)_", destination)
            if match:
                return bool(settings.get(match[1] + "_on", 1))
            for effect in EFFECTS:
                if destination.startswith(effect + "_"):
                    return bool(settings[effect + "_on"])
            return bool(settings["osc_1_on"] or settings["osc_2_on"] or settings["osc_3_on"] or settings["sample_on"])

        for _ in range(65):
            found = {slot for slot, route in enumerate(routes) if enabled[slot] and audible(route["destination"])}
            if found <= active:
                break
            active |= found
            sources.update(routes[slot]["source"] for slot in active)
        return tuple(slot in active for slot in range(64))

    def active_lfos(self, target):
        routes, active = self.routes(target), self.active_routes(target)
        sources = {route["source"] for route, enabled in zip(routes, active) if enabled}
        return tuple(f"lfo_{slot}" in sources for slot in range(1, 9))

    def _scalar_mask(self, name, present, sources, fallback):
        match = re.match(r"modulation_(\d+)_", name)
        if match:
            return present[int(match[1]) - 1]
        match = re.match(r"(lfo_\d|random_\d|env_\d)_", name)
        if match and match[1] != "env_1":
            return match[1] in sources
        return fallback

    def _masks(self, target):
        routes, active = self.routes(target), self.active_routes(target)
        present = tuple(bool(route["source"]) for route in routes)
        sources = {route["source"] for route, enabled in zip(routes, active) if enabled}
        return routes, active, present, sources

    def continuous_mask(self, target):
        curve = self._curve_target(target)
        base = self.curve._base_target(curve)
        routes, active, present, sources = self._masks(target)
        scalar = tuple(self._scalar_mask(name, present, sources, value) for (name, *_), value in zip(self.base.continuous, self.base.continuous_mask(base)))
        curves = self.curve.curve_parameter_masks(curve, tuple(f"lfo_{slot}" in sources for slot in range(1, 9)))
        maps = tuple(bool(enabled and "line_mapping" in route) for route, enabled in zip(routes, active) for _ in range(97))
        return scalar + curves + maps

    def categorical_mask(self, target):
        curve = self._curve_target(target)
        base = self.curve._base_target(curve)
        _, active, present, sources = self._masks(target)
        mask = self.base.categorical_mask(base)
        names = self.base.scalar._categorical_controls
        scalar = tuple(self._scalar_mask(name, present, sources, value) for name, value in zip(names, mask[:len(names)]))
        curves = tuple(f"lfo_{slot}" in sources for slot in range(1, 9) for _ in range(17))
        routes = tuple(value for slot in range(64) for value in (not present[slot] or active[slot], active[slot], active[slot], active[slot]))
        return scalar + mask[len(names):] + curves + routes

    def categorical_weights(self, target):
        flags = target.categorical[self._curve_cat:]
        mask = self.categorical_mask(target)
        positive = sum(bool(flags[slot * 4]) and mask[self._curve_cat + slot * 4] for slot in range(64))
        negative = sum(not flags[slot * 4] for slot in range(64))
        groups = int(positive > 0) + int(negative > 0)
        result = [1.] * len(self.categorical)
        for slot in range(64):
            count = positive if flags[slot * 4] else negative
            result[self._curve_cat + slot * 4] = 1. / max(1, groups * count)
        return tuple(result)

    def active_route_maps(self, target):
        return tuple(enabled and "line_mapping" in route for route, enabled in zip(self.routes(target), self.active_routes(target)))

    def family(self, target):
        routes = [route for route in self.routes(target) if route["source"]]
        if any(route["destination"].startswith("lfo_") for route in routes):
            return "modulator_feedback"
        if any(route["destination"].startswith("modulation_") for route in routes):
            return "indexed_amount"
        return "dense" if len(routes) > 4 else "parallel" if len(routes) > 1 else "direct"

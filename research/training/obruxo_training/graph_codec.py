"""Broad/coverage variants under frozen roots, with every destination scheduled."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

from obruxo_data.hashing import canonical_json, canonical_sha256

from .curve_codec import CurveInputCodec
from .full_scalar import EFFECTS
from .pilot import seeded_random


def destination_schedule(roots, destinations, width):
    """Cover the inventory in training and spread each held-out split across it."""
    schedule = {}
    for split in ("train", "validation", "test"):
        order = list(destinations)
        seeded_random("graph_destination_order_v1", roots.seed, split).shuffle(order)
        entries = (entry for entry in roots.assignments if entry.split == split)
        for index, entry in enumerate(entries):
            schedule[entry.state_id] = tuple(order[(index * width + offset) % len(order)] for offset in range(width))
    return schedule


def coverage_routes(settings, sources, destinations, offset, *, width=1):
    """Protect referenced slots so control-to-control probes always have an audible exit."""
    selected = tuple(destinations[(offset + index) % len(destinations)] for index in range(width))
    protected = {int(match[1]) - 1 for name in selected if (match := re.fullmatch(r"modulation_(\d+)_(?:amount|power)", name))}
    protected |= {slot + 32 for slot in protected if slot < 32}
    original = settings["modulations"]
    owners = [slot for slot in range(63, 31, -1) if slot not in protected and "line_mapping" not in original[slot]][:width]
    if len(owners) != width:
        raise ValueError("insufficient unreferenced probe slots")
    routes = [{"source": sources[slot % len(sources)], "destination": ("osc_1_pan", "osc_2_pan", "osc_3_pan", "sample_pan")[slot % 4],
               **({"line_mapping": original[slot]["line_mapping"]} if "line_mapping" in original[slot] else {})} for slot in range(64)]
    for slot in range(64):
        settings.update({f"modulation_{slot + 1}_amount": .003, f"modulation_{slot + 1}_bypass": 0.})
    for slot, destination in zip(owners, selected, strict=True):
        routes[slot] = {**routes[slot], "source": "env_1", "destination": destination}
        settings[f"modulation_{slot + 1}_amount"] = .03
        match = re.fullmatch(r"modulation_(\d+)_(?:amount|power)", destination)
        if match:
            child = routes[int(match[1]) - 1]
            child["source"] = "lfo_2"
            child.pop("line_mapping", None)
    return routes, selected, owners


class GraphCoverageCodec:
    generator = "frozen_root_graph_coverage_variants_v1"
    multiple_variants_per_root = True
    variants_per_root = 4

    def __init__(self, asset_bank, lfo_bank):
        self.parent = CurveInputCodec(asset_bank, lfo_bank)
        self.sources = tuple(sorted(set(self.routing.sources) - {""}))
        self.destinations = tuple(sorted(set(self.routing.destinations) - {""}))
        self._destination_schedule = destination_schedule(self.roots, self.destinations, self.variants_per_root - 1)
        self._authored = lru_cache(maxsize=16)(self._authored)

    def __getattr__(self, name):
        return getattr(self.parent, name)

    @property
    def required_count(self):
        return self.variants_per_root * len(self.roots.assignments)

    @property
    def codec_id(self):
        return canonical_sha256(self.to_dict())

    def to_dict(self):
        spec = self.parent.to_dict()
        spec.update(codec="authored_complete_graph_v1", generator=self.generator,
                    source_policy="original_curve_sources_and_complete_connection_inventory_variants_v1",
                    graph_curriculum={"policy": "three_single_destination_variants_per_split_root_preserve_referenced_exits_v1",
                        "destination_schedule": "graph_destination_order_v1_independent_seeded_permutation_per_split",
                        "source_sha256_lf": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                                             for name in ("graph_codec.py", "graph_context.py")},
                        "variants_per_root": self.variants_per_root, "sources": list(self.sources), "destinations": list(self.destinations),
                        "coverage_context": "all_generators_filters_effects_and_named_sources_enabled",
                        "coverage_claim": "declared_supervision_not_native_audibility_or_learning"},
                    scope=spec["scope"] + "_with_all_destination_inventory_variants")
        return spec

    def _root_index(self, rng, index):
        if type(index) is not int or not 0 <= index < self.required_count:
            raise ValueError("graph variants must remain within their frozen roots")
        if rng.getstate() != seeded_random(self.generator, self.roots.seed, index).getstate():
            raise ValueError("graph curriculum seed differs from the frozen roots")
        return index // self.variants_per_root

    def lineage_root(self, rng, index):
        root_index = self._root_index(rng, index)
        return self.parent.lineage_root(seeded_random(self.parent.generator, self.roots.seed, root_index), root_index)

    def _authored(self, index):
        root_index = index // self.variants_per_root
        target = self.parent.sample_target(seeded_random(self.parent.generator, self.roots.seed, root_index), root_index)
        raw = self.parent.source_preset(target, root_index)
        variant = index % self.variants_per_root
        if variant == 0:
            return target, raw
        document = json.loads(raw)
        settings = document["settings"]
        settings.update({f"osc_{slot}_on": 1. for slot in range(1, 4)})
        settings.update(sample_on=1., filter_1_on=1., filter_2_on=1., **{effect + "_on": 1. for effect in EFFECTS})
        root = self.lineage_root(seeded_random(self.generator, self.roots.seed, index), index)
        destination = self._destination_schedule[root][variant - 1]
        from .graph_context import prepare_destination_context
        prepare_destination_context(settings, destination, self.base._waves, self.base._samples)
        settings["modulations"], selected, owners = coverage_routes(settings, self.sources, (destination,), 0)
        settings.update(self.base.scalar._values(self.base.scalar._target(settings)))
        raw = canonical_json(document).encode()
        target = self.parent.encode(raw)
        active = self.routing.active_routes(self._routing_target(target))
        if not all(active[slot] for slot in owners):
            raise ValueError(f"coverage destinations are not supervised: {selected}")
        return target, raw

    def sample_target(self, rng, index):
        self._root_index(rng, index)
        return self._authored(index)[0]

    def source_preset(self, target, index):
        self._root_index(seeded_random(self.generator, self.roots.seed, index), index)
        expected, raw = self._authored(index)
        if self.canonical_target(target) != expected:
            raise ValueError("saved graph target differs from its original source")
        return raw

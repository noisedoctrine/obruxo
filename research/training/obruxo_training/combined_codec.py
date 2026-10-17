"""Combined control curriculum with original source curves and frozen ancestry."""
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random

from obruxo_data.hashing import canonical_json

from .codec import PilotTarget
from .combined_policy import POLICY, control_policy
from .contracts import TrainingContract
from .expression_codec import ExpressionCodec
from .full_scalar import EFFECTS, effect_rank
from .pilot import seeded_random


class CombinedCodec(ExpressionCodec):
    model_family = "combined_nested_state"

    def __init__(self, asset_bank, lfo_bank):
        super().__init__(asset_bank, lfo_bank)
        self.contract = TrainingContract.create(self.schema, self.catalog, tail_seconds=4.)
        self.policy = control_policy(self.base.scalar)
        self.categorical = tuple((name, tuple(range(5)) if name in self.policy and name.endswith("_destination") else values)
                                 for name, values in self.categorical)
        self._authored = lru_cache(maxsize=16)(self._authored)

    def to_dict(self):
        spec = super().to_dict()
        sources = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                   for name in ("combined_codec.py", "combined_policy.py")}
        spec.update(codec="authored_combined_nested_state_v1", contract_id=self.contract.contract_id, source_policy=POLICY,
                    inherited_curricula="parent_source_construction_before_combined_control_overrides",
                    combined_curriculum={"policy": POLICY, "source_sha256_lf": sources, "controls": self.policy,
                        "raw_source_cache_entries": 16, "tail_seconds": 4., "source_bindings": "original_LFO_map_warp_and_uint32_seed_labels",
                        "scenes": ["oscillator", "sampler", "hybrid", "dense_all_modules"],
                        "audio_destination_values": list(range(5)), "inactive_native_destination_values": list(range(5, 15)),
                        "master_gain_db": [-29., -24.], "sparse_additional_sources": 2,
                        "dense_destination_stride": 24, "full_range_coverage_claim": False},
                    families=["oscillators", "sampler", "hybrid"], scope=spec["scope"] + "_combined_control_and_factory_asset_curriculum")
        return spec

    def _root(self, index):
        root = self.lineage_root(seeded_random(self.generator, self.roots.seed, index), index)
        if root not in self.routing._ordinal:
            raise ValueError("combined curriculum requires its frozen authored roots")
        return root

    def _controls(self, rng, ordinal):
        settings = {}
        for name, rule in self.policy.items():
            kind = rule["kind"]
            if kind == "fixed":
                settings[name] = rule["value"]
            elif kind == "choice":
                settings[name] = float(rng.choice(rule["values"]))
            elif kind == "pitch_mask":
                settings[name] = float(rng.choice((0, 0b101010110101, 0b100100010001)))
            elif kind == "permutation_rank":
                order = list(range(9))
                rng.shuffle(order)
                settings[name] = float(effect_rank(order))
            else:
                settings[name] = rng.uniform(rule["minimum"], rule["maximum"])
        scene = ordinal % 4
        enabled = ((1, 0, 0), (0, 0, 0), (rng.randrange(2), 1, rng.randrange(2)), (1, 1, 1))[scene]
        settings.update(sample_on=float(scene != 0), filter_1_on=float(scene == 3 or rng.random() < .75),
                        filter_2_on=float(scene == 3 or rng.random() < .5))
        settings.update({f"osc_{slot}_on": float(value) for slot, value in enumerate(enabled, 1)})
        primary = ordinal % len(EFFECTS)
        selected = set(range(9)) if scene == 3 else set() if ordinal % 10 == 0 else {primary, (primary + 4) % 9}
        settings.update({effect + "_on": float(index in selected) for index, effect in enumerate(EFFECTS)})
        return settings

    def _connections(self, settings, original, rng, ordinal):
        routes = deepcopy(original)
        free = [slot for slot, route in enumerate(routes) if not route["source"]]
        rng.shuffle(free)
        present = {route["source"] for route in routes}
        audible = next((f"osc_{slot}_pan" for slot in range(1, 4) if settings[f"osc_{slot}_on"]), "sample_pan")
        missing = sorted(set(self.routing.sources) - {""} - present)
        if ordinal % 4 != 3:
            missing = [missing[((ordinal // 4) * 2 + offset) % len(missing)] for offset in range(min(2, len(missing)))]
        for source in missing:
            slot = free.pop()
            routes[slot] = {"source": source, "destination": audible}
        if ordinal % 4 == 3:
            destinations = tuple(name for name in self.routing.destinations if name)
            if len(free) < 24:
                raise ValueError("dense coverage needs at least 24 remaining connection slots")
            for offset, slot in enumerate(free):
                destination = destinations[((ordinal // 4) * 24 + offset) % len(destinations)]
                routes[slot] = {"source": rng.choice(tuple(sorted(present - {""}))), "destination": destination}
        for slot, route in enumerate(routes, 1):
            if not route["source"]:
                continue
            amount = settings[f"modulation_{slot}_amount"]
            if abs(amount) < .005:
                settings[f"modulation_{slot}_amount"] = .005 if amount >= 0 else -.005
        return routes

    def _authored(self, index):
        root = self._root(index)
        ordinal = self.routing._ordinal[root]
        parent = ExpressionCodec.sample_target(self, seeded_random(self.generator, self.roots.seed, index), index)
        document = json.loads(ExpressionCodec.source_preset(self, parent, index))
        settings = document["settings"]
        rng = seeded_random(POLICY, self.roots.seed, index)
        settings.update(self._controls(rng, ordinal))
        routes = self._connections(settings, settings["modulations"], rng, ordinal)
        scalar = self.base.scalar._target(settings)
        settings.update(self.base.scalar._values(scalar))
        choices = tuple((ordinal // 4 + slot * 7) % len(self.base._waves) for slot in range(3)) + ((ordinal // 4) % len(self.base._samples),)
        settings["wavetables"] = [deepcopy(self.base._waves[choice]) for choice in choices[:3]]
        settings["sample"] = deepcopy(self.base._samples[choices[3]])
        settings["modulations"] = routes
        prefix = len(scalar.categorical) + 4
        target = PilotTarget(scalar.continuous + parent.continuous[len(scalar.continuous):],
                             scalar.categorical + choices + parent.categorical[prefix:])
        target = self.canonical_target(self._route_target(target, routes))
        self.schema.validate(document).require_valid()
        return target, canonical_json(document).encode()

    def sample_target(self, rng, index):
        probe = random.Random()
        probe.setstate(rng.getstate())
        if self.lineage_root(probe, index) != self._root(index):
            raise ValueError("combined curriculum seed differs from the frozen roots")
        return self._authored(index)[0]

    def source_preset(self, target, index):
        expected, raw = self._authored(index)
        if self.canonical_target(target) != expected:
            raise ValueError("saved combined target does not match its original source policy")
        return raw

    def family(self, target):
        scalar = self.base.scalar
        settings = scalar._values(PilotTarget(target.continuous[:len(scalar.continuous)], target.categorical[:len(scalar.categorical)]))
        if not settings["sample_on"]:
            return "oscillators"
        return "hybrid" if any(settings[f"osc_{slot}_on"] for slot in range(1, 4)) else "sampler"

"""Three learned custom warps and exact uint32 seed outputs over ordered routing."""
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random

import numpy as np

from obruxo_data.hashing import canonical_json, canonical_sha256
from obruxo_data.render.base import FULL_SEED_RENDERER_ID
from obruxo_data.vital.full_contract import FullPresetContract

from .codec import PilotCodec, PilotTarget
from .contracts import TrainingContract
from .lfo_bank import curve_to_lfo, sample_lfo
from .routing import OrderedRoutingCodec

WARP_MODES = (13, 14, 15)
SEED_MODE = 6
FAMILIES = tuple(f"warp_{slot}_mode_{mode}" for slot in range(1, 4) for mode in WARP_MODES) + tuple(f"seed_{slot}" for slot in range(1, 4)) + ("mixed",)


def seed_bits(seeds):
    if len(seeds) != 3 or any(type(seed) not in {int, float} or not float(seed).is_integer() or not 0 <= seed <= 0xFFFFFFFF for seed in seeds):
        raise ValueError("three unsigned 32-bit seeds are required")
    return tuple((int(seed) >> bit) & 1 for seed in seeds for bit in range(32))


def authored_nested(index):
    rng = np.random.default_rng(0x4F425255584F + index)
    warps = []
    for slot in range(3):
        count = 3 + (index + slot) % 4
        points = np.column_stack((np.linspace(0, 1, count, dtype=np.float32), rng.uniform(.05, .95, count).astype(np.float32)))
        warps.append({"name": f"Authored warp {slot + 1}", "num_points": count, "points": points.reshape(-1).tolist(),
                      "powers": rng.uniform(-1.5, 1.5, count).astype(np.float32).tolist(), "smooth": bool((index + slot) % 2)})
    return warps, [int(value) for value in rng.integers(0, 2**32, 3, dtype=np.uint64)]


def authored_modes(ordinal):
    family = ordinal % len(FAMILIES)
    modes = [0, 0, 0]
    if family < 9:
        modes[family // 3] = WARP_MODES[family % 3]
    elif family < 12:
        modes[family - 9] = SEED_MODE
    else:
        modes = [13, 6, 15]
    return tuple(modes)


class NestedStateCodec:
    generator = OrderedRoutingCodec.generator
    model_family = "nested_state"
    balanced_loss = True
    lossy_curves = True
    lossy_route_maps = True
    lossy_warps = True

    def __init__(self, asset_bank, lfo_bank):
        self.routing = OrderedRoutingCodec(asset_bank, lfo_bank)
        self.base, self.lfo_bank = self.routing.base, self.routing.lfo_bank
        self.schema, self.catalog, self.contract, self.roots = self.routing.schema, self.routing.catalog, self.routing.contract, self.routing.roots
        self.schema = FullPresetContract(renderer_id=FULL_SEED_RENDERER_ID)
        self.contract = TrainingContract.create(self.schema, self.catalog)
        self._ncont, self._ncat, self._map_start = self.routing._ncont, self.routing._ncat, self.routing._map_start
        self._routing_cont, self._routing_cat = len(self.routing.continuous), len(self.routing.categorical)
        self._warp_start = self._routing_cont
        defaults = np.stack([sample_lfo(shape) for shape in self.schema.template()["settings"]["custom_warps"]])
        self.continuous = self.routing.continuous + tuple((f"custom_warp_{slot + 1}_{point}", 0., 1., float(defaults[slot, point])) for slot in range(3) for point in range(97))
        self.categorical = self.routing.categorical + tuple((f"osc_{slot}_seed_bit_{bit}", (0, 1)) for slot in range(1, 4) for bit in range(32))
        self.continuous_groups = self.routing.continuous_groups + ("custom_warps",) * 291
        self.categorical_groups = self.routing.categorical_groups + ("oscillator_seeds",) * 96
        self.cyclic_indices = self.routing.cyclic_indices
        self._wave_choice = next((index for index, value in enumerate(self.base._waves) if value.get("name") == "Basic Shapes"), 1)
        for name in ("active_warps", "active_seeds", "continuous_mask", "categorical_mask", "categorical_weights"):
            setattr(self, name, lru_cache(maxsize=512)(getattr(self, name)))

    @property
    def codec_id(self):
        return canonical_sha256(self.to_dict())

    def to_dict(self):
        spec = self.routing.to_dict()
        spec.update(codec="authored_nested_state_v1", contract_id=self.contract.contract_id,
                    native_seed_readback="separate_v2_renderer_exact_signed_int32_to_uint32_bits",
                    continuous=spec["continuous"] + [{"name": name, "minimum": lo, "maximum": hi, "inactive_default": default}
                                                                                     for name, lo, hi, default in self.continuous[self._warp_start:]],
                    categorical=[{"name": name, "values": list(values)} for name, values in self.categorical],
                    loss="family_balanced_ordered_routing_original_lfo_map_warp_mse_and_seed_bits_v1", families=list(FAMILIES),
                    seed_policy="three_exact_uint32_little_endian_32_binary_heads_no_float_regression",
                    warp_policy="three_97_point_float32_linear_curves_canonical_names_lossy", warp_loss_weight=1.,
                    source_policy="original_authored_lfos_route_maps_custom_warps_and_uint32_seeds_v1",
                    nested_curriculum={"policy": "authored_nested_state_v1", "source_sha256": hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
                                       "wave_asset_id": self.base._wave_ids[self._wave_choice], "oscillator_level_stored": .06,
                                       "warp_modes": list(WARP_MODES), "seed_mode": SEED_MODE, "root_policy": "preserve_all_frozen_root_assignments"},
                    nested_policy="all_six_nested_families_registered_assets_learned_lfos_warps_seeds_and_ordered_connections",
                    scope="all_775_scalars_registered_assets_eight_lfos_64_ordered_routes_three_custom_warps_and_three_uint32_seeds")
        return spec

    def validate_target(self, target):
        PilotCodec.validate_target(self, target)

    def _routing_target(self, target):
        return PilotTarget(target.continuous[:self._routing_cont], target.categorical[:self._routing_cat])

    def canonical_target(self, target):
        self.validate_target(target)
        routing = self.routing.canonical_target(self._routing_target(target))
        return PilotTarget(routing.continuous + tuple(float(np.float32(value)) for value in target.continuous[self._warp_start:]),
                           routing.categorical + target.categorical[self._routing_cat:])

    def _join(self, routing, warps, seeds):
        values = tuple(float(value) for shape in warps for value in sample_lfo(shape))
        return self.canonical_target(PilotTarget(routing.continuous + values, routing.categorical + seed_bits(seeds)))

    def seeds(self, target):
        self.validate_target(target)
        flags = target.categorical[self._routing_cat:]
        return tuple(sum(flags[slot * 32 + bit] << bit for bit in range(32)) for slot in range(3))

    def warps(self, target):
        self.validate_target(target)
        return np.asarray(target.continuous[self._warp_start:], dtype=np.float32).reshape(3, 97)

    def settings(self, target):
        settings = self.routing.settings(self._routing_target(target))
        settings["custom_warps"] = [curve_to_lfo(values, name=f"OBRUXO custom warp {slot}") for slot, values in enumerate(self.warps(target), 1)]
        settings["random_values"] = [{"seed": value} for value in self.seeds(target)]
        return settings

    def decode(self, target):
        target = self.canonical_target(target)
        document = json.loads(self.routing.decode(self._routing_target(target)))
        settings = self.settings(target)
        document["settings"].update(custom_warps=settings["custom_warps"], random_values=settings["random_values"])
        self.schema.validate(document).require_valid()
        return canonical_json(document).encode()

    def encode(self, raw):
        document, report = self.schema.loads(raw)
        report.require_valid()
        settings = document["settings"]
        warps, seeds = deepcopy(settings["custom_warps"]), [value["seed"] for value in settings["random_values"]]
        template = self.schema.template()["settings"]
        settings.update(custom_warps=template["custom_warps"], random_values=template["random_values"])
        return self._join(self.routing.encode(canonical_json(document).encode()), warps, seeds)

    def default_target(self):
        settings = self.schema.template()["settings"]
        return self._join(self.routing.default_target(), settings["custom_warps"], [value["seed"] for value in settings["random_values"]])

    def supervision_target(self, target):
        routing = self.routing.supervision_target(self._routing_target(target))
        return PilotTarget(routing.continuous + target.continuous[self._warp_start:], routing.categorical + target.categorical[self._routing_cat:])

    def source_curves(self, raw):
        return self.routing.source_curves(raw)

    def source_route_maps(self, raw):
        return self.routing.source_route_maps(raw)

    def source_warp_curves(self, raw):
        return np.stack([sample_lfo(shape) for shape in json.loads(raw)["settings"]["custom_warps"]])

    def curves(self, target):
        return self.routing.curves(self._routing_target(target))

    def active_lfos(self, target):
        return self.routing.active_lfos(self._routing_target(target))

    def active_route_maps(self, target):
        return self.routing.active_route_maps(self._routing_target(target))

    def lineage_root(self, rng, index):
        return self.routing.lineage_root(rng, index)

    def validate_manifest(self, manifest):
        self.routing.validate_manifest(manifest)

    def sample_target(self, rng, index):
        probe = random.Random()
        probe.setstate(rng.getstate())
        root = self.lineage_root(probe, index)
        routing = self.routing.sample_target(rng, index)
        curve = self.routing._curve_target(routing)
        settings = self.routing.settings(routing)
        routes, controls, _ = self.routing._authored(index, root, False)
        settings.update(controls)
        settings["sample_on"] = 0.
        modes = authored_modes(self.routing._ordinal[root])
        for slot, mode in enumerate(modes, 1):
            settings.update({f"osc_{slot}_on": 1., f"osc_{slot}_level": .06, f"osc_{slot}_wave_frame": 128.,
                             f"osc_{slot}_spectral_morph_type": float(mode), f"osc_{slot}_spectral_morph_amount": .6})
        scalar = self.base.scalar._target(settings)
        base_categories = scalar.categorical + (self._wave_choice,) * 3 + (0,)
        curve = PilotTarget(scalar.continuous + curve.continuous[self._ncont:], base_categories + curve.categorical[self._ncat:])
        return self._join(self.routing._join(curve, routes), *authored_nested(index))

    def source_preset(self, target, index):
        routing = self._routing_target(target)
        document = json.loads(self.routing.source_preset(routing, index))
        warps, seeds = authored_nested(index)
        if self._join(routing, warps, seeds) != self.canonical_target(target):
            raise ValueError("saved nested target does not match its original custom curves and seeds")
        document["settings"].update(custom_warps=warps, random_values=[{"seed": value} for value in seeds])
        self.schema.validate(document).require_valid()
        return canonical_json(document).encode()

    def _active_modes(self, target):
        settings = self.base._values(self.routing.curve._base_target(self.routing._curve_target(self._routing_target(target))))
        return tuple(int(settings[f"osc_{slot}_spectral_morph_type"]) if settings[f"osc_{slot}_on"] else -1 for slot in range(1, 4))

    def active_warps(self, target):
        return tuple(mode in WARP_MODES for mode in self._active_modes(target))

    def active_seeds(self, target):
        return tuple(mode == SEED_MODE for mode in self._active_modes(target))

    def continuous_mask(self, target):
        return self.routing.continuous_mask(self._routing_target(target)) + tuple(active for active in self.active_warps(target) for _ in range(97))

    def categorical_mask(self, target):
        return self.routing.categorical_mask(self._routing_target(target)) + tuple(active for active in self.active_seeds(target) for _ in range(32))

    def categorical_weights(self, target):
        return self.routing.categorical_weights(self._routing_target(target)) + (1.,) * 96

    def family(self, target):
        modes = self._active_modes(target)
        active = [(slot, mode) for slot, mode in enumerate(modes, 1) if mode in (*WARP_MODES, SEED_MODE)]
        if len(active) != 1:
            return "mixed" if active else "inactive"
        slot, mode = active[0]
        return f"seed_{slot}" if mode == SEED_MODE else f"warp_{slot}_mode_{mode}"

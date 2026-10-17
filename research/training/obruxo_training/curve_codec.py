"""Combined sources with explicit one-point curve ingestion, without changing older codecs."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

from obruxo_data.hashing import canonical_json, canonical_sha256
from obruxo_data.render.base import FULL_CURVE_RENDERER_ID
from obruxo_data.render.curve_inputs import curve_input_policy, prepare_curve_inputs
from obruxo_data.vital.full_contract import FullPresetContract

from .combined_codec import CombinedCodec
from .contracts import TrainingContract
from .pilot import seeded_random

LAYOUTS = ("lfo", "mapping", "warp", "mixed")


class CurveInputCodec:
    def __init__(self, asset_bank, lfo_bank):
        self.parent = CombinedCodec(asset_bank, lfo_bank)
        self.schema = FullPresetContract(renderer_id=FULL_CURVE_RENDERER_ID)
        self.catalog = self.parent.catalog
        self.contract = TrainingContract.create(self.schema, self.catalog, tail_seconds=4.)
        self._authored = lru_cache(maxsize=16)(self._authored)

    def __getattr__(self, name):
        return getattr(self.parent, name)

    @property
    def codec_id(self):
        return canonical_sha256(self.to_dict())

    def to_dict(self):
        spec = self.parent.to_dict()
        spec.update(codec="authored_curve_inputs_v1", renderer_id=self.contract.renderer_id, contract_id=self.contract.contract_id,
                    source_policy="original_singleton_curve_sources_explicit_native_constant_import_v1",
                    curve_input_policy=curve_input_policy(), curve_layout_curriculum={"families": list(LAYOUTS),
                        "source_sha256_lf": hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
                        "layout_scene_assignment": "orthogonal_four_root_groups_v1", "raw_source_cache_entries": 16,
                        "activation": "selected_lfo_and_map_connected_to_enabled_generator_pan_bypass_zero_amount_0_05",
                        "warp_activation": "selected_oscillator_on_mode_13_14_15_amount_0_5_slot_and_mode_cycle_orthogonally"},
                    scope=spec["scope"] + "_with_original_singleton_curve_inputs")
        return spec

    def _applied(self, raw):
        document, report = self.schema.loads(raw)
        report.require_valid()
        applied, _ = prepare_curve_inputs(document)
        return canonical_json(applied).encode()

    def encode(self, raw):
        return self.parent.encode(self._applied(raw))

    def source_curves(self, raw):
        return self.parent.source_curves(self._applied(raw))

    def source_route_maps(self, raw):
        return self.parent.source_route_maps(self._applied(raw))

    def source_warp_curves(self, raw):
        return self.parent.source_warp_curves(self._applied(raw))

    def _authored(self, index):
        target = self.parent.sample_target(seeded_random(self.generator, self.roots.seed, index), index)
        document = json.loads(self.parent.source_preset(target, index))
        ordinal = self.routing._ordinal[self.parent._root(index)]
        family, turn = (ordinal // 4) % 4, ordinal // 16
        settings = document["settings"]

        def singleton(offset):
            return {"name": f"OBRUXO singleton {offset}", "num_points": 1, "points": [(0., .37, 1.)[turn % 3],
                    (0., .2, .5, .8, 1.)[(ordinal // 4 + offset) % 5]],
                    "powers": [(-2., .37, 2.)[turn % 3]], "smooth": bool(ordinal % 2)}

        if family in {0, 3}:
            settings["lfos"][turn % 8] = singleton(0)
        if family in {2, 3}:
            slot = turn % 3 + 1
            settings["custom_warps"][slot - 1] = singleton(2)
            settings.update({f"osc_{slot}_on": 1., f"osc_{slot}_spectral_morph_type": float((13, 14, 15)[(turn // 3) % 3]),
                             f"osc_{slot}_spectral_morph_amount": .5})
        audible = next((f"osc_{slot}_pan" for slot in range(1, 4) if settings[f"osc_{slot}_on"]), "sample_pan")
        used = set()

        def connect(source, mapping=False):
            routes = settings["modulations"]
            candidates = [slot for slot, route in enumerate(routes) if slot not in used and route["source"] == source]
            candidates += [slot for slot, route in enumerate(routes) if slot not in used and not route["source"]]
            candidates += [slot for slot in range(64) if slot not in used]
            slot = candidates[0]
            used.add(slot)
            routes[slot] = {"source": source, "destination": audible, **({"line_mapping": singleton(1)} if mapping else {})}
            settings.update({f"modulation_{slot + 1}_amount": .05, f"modulation_{slot + 1}_bypass": 0.})

        if family in {0, 3}:
            connect(f"lfo_{turn % 8 + 1}")
        if family in {1, 3}:
            connect("velocity", mapping=True)
        settings.update(self.base.scalar._values(self.base.scalar._target(settings)))
        raw = canonical_json(document).encode()
        target = self.encode(raw)
        flags = {"lfos": self.active_lfos(target), "modulations": self.active_route_maps(target), "custom_warps": self.active_warps(target)}
        for group, shapes in (("lfos", settings["lfos"]), ("custom_warps", settings["custom_warps"]),
                              ("modulations", [route.get("line_mapping") for route in settings["modulations"]])):
            if any(shape and shape["num_points"] == 1 and not flags[group][slot] for slot, shape in enumerate(shapes)):
                raise ValueError("authored singleton curve is not actively supervised")
        return target, raw

    def sample_target(self, rng, index):
        # Preserve the parent's frozen-root seed check before deriving a new source layout.
        self.parent.sample_target(rng, index)
        return self._authored(index)[0]

    def source_preset(self, target, index):
        expected, raw = self._authored(index)
        if self.canonical_target(target) != expected:
            raise ValueError("saved curve-input target differs from its original source")
        return raw

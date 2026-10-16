"""Audio-only outputs for eight Era2 curves alongside scalar and factory assets."""
from copy import deepcopy
from functools import lru_cache
import json
import random
import re

import numpy as np

from obruxo_data.hashing import canonical_json, canonical_sha256

from .asset_codec import FactoryAssetCodec
from .codec import PilotCodec, PilotTarget
from .lfo_bank import INDEX_SIZES, LfoBank, LfoCode, curve_to_lfo, sample_lfo
from .lfo_curriculum import POLICY, authored_lfos, source_sha256
from .splits import SplitManifest


class LfoCurveCodec:
    generator = FactoryAssetCodec.generator
    model_family = "lfo_curve"
    balanced_loss = True
    lossy_curves = True

    def __init__(self, asset_bank, lfo_bank):
        self.base, self.lfo_bank = FactoryAssetCodec(asset_bank), LfoBank(lfo_bank)
        self.schema, self.catalog, self.contract = self.base.schema, self.base.catalog, self.base.contract
        manifest = SplitManifest.from_dict(self.lfo_bank.to_dict()["bank"]["split_manifest"])
        self.roots = getattr(manifest, "parent", manifest)
        self._primary = {}
        for split in ("train", "validation", "test"):
            for index, entry in enumerate(entry for entry in self.roots.assignments if entry.split == split):
                self._primary[entry.state_id] = index % 8
        self._ncont, self._ncat = len(self.base.continuous), len(self.base.categorical)
        controls = tuple((f"lfo_{slot}_code_phase_{layer}", 0., 1., 0.) for slot in range(1, 9) for layer in range(17))
        gains = tuple((f"lfo_{slot}_code_gain_{layer}", -2., 2., 0.) for slot in range(1, 9) for layer in range(1, 17))
        self.continuous = self.base.continuous + controls + gains
        self.categorical = self.base.categorical + tuple((f"lfo_{slot}_code_index_{layer}", tuple(range(size)))
                                                        for slot in range(1, 9) for layer, size in enumerate(INDEX_SIZES))
        self.continuous_groups = self.base.continuous_groups + ("lfo_curves",) * 264
        self.categorical_groups = self.base.categorical_groups + ("lfo_curves",) * 136
        self.cyclic_indices = self.base.cyclic_indices + tuple(range(self._ncont, self._ncont + 136))
        template = self.schema.template()["settings"]["modulations"]
        self.routes = deepcopy(template)
        for slot in range(8):
            self.routes[2 * slot] = {"source": f"lfo_{slot + 1}", "destination": "osc_1_level"}
            self.routes[2 * slot + 1] = {"source": f"lfo_{slot + 1}", "destination": "sample_level"}
        self._authored_codes = {}
        self.curves = lru_cache(maxsize=512)(self._curves)

    @property
    def codec_id(self):
        return canonical_sha256(self.to_dict())

    def to_dict(self):
        spec = self.base.to_dict()
        spec.update(codec="authored_lfo_curves_v1", asset_bank=deepcopy(self.base.bank), lfo_bank=self.lfo_bank.to_dict(),
                    continuous=spec["continuous"] + [{"name": name, "minimum": lo, "maximum": hi, "inactive_default": default}
                                                       for name, lo, hi, default in self.continuous[self._ncont:]],
                    categorical=[{"name": name, "values": list(values)} for name, values in self.categorical],
                    loss="family_balanced_masked_atlas_and_era2_soft_curve_mse_v1", curve_loss_weight=1.,
                    curriculum={"policy": POLICY, "source_sha256": source_sha256(), "headroom_multiplier": .25,
                                "primary_lfo": "round_robin_within_frozen_split_root_order", "active_lfos_per_state": 1},
                    fixed_routes=deepcopy(self.routes), families=[f"lfo_{slot}" for slot in range(1, 9)],
                    source_policy="original_authored_lfo_audio_with_lossy_era2_code_targets_v1",
                    curve_supervision="original_sampled_97_point_targets_never_model_inputs",
                    nested_policy="eight_lossy_era2_curves_fixed_16_routes_fixed_custom_warps_and_seeds",
                    curve_projection="97_point_era2_beam4_lossy_raw_reencoding_not_a_lossless_roundtrip",
                    scope="all_775_scalars_registered_assets_eight_lfo_codes_16_fixed_connections")
        return spec

    def validate_target(self, target):
        PilotCodec.validate_target(self, target)

    def _base_target(self, target):
        return PilotTarget(target.continuous[:self._ncont], target.categorical[:self._ncat])

    def canonical_target(self, target):
        self.validate_target(target)
        base = self.base.canonical_target(self._base_target(target))
        phases = tuple(float(np.float32(value) % np.float32(1.)) for value in target.continuous[self._ncont:self._ncont + 136])
        gains = tuple((float(np.float32(np.float32(4.) * np.float32(value) - np.float32(2.))) + 2.) / 4.
                      for value in target.continuous[self._ncont + 136:])
        return PilotTarget(base.continuous + phases + gains, base.categorical + target.categorical[self._ncat:])

    def _join(self, base, codes):
        phases = tuple(value for code in codes for value in code.phases)
        gains = tuple((value + 2.) / 4. for code in codes for value in code.gains)
        indices = tuple(value for code in codes for value in code.indices)
        return self.canonical_target(PilotTarget(base.continuous + phases + gains, base.categorical + indices))

    def codes(self, target):
        self.validate_target(target)
        phases, gains = target.continuous[self._ncont:self._ncont + 136], target.continuous[self._ncont + 136:]
        indices = target.categorical[self._ncat:]
        return [LfoCode(indices[slot * 17:(slot + 1) * 17], phases[slot * 17:(slot + 1) * 17],
                        tuple(float(np.float32(value * 4. - 2.)) for value in gains[slot * 16:(slot + 1) * 16])) for slot in range(8)]

    def _curves(self, target):
        return self.lfo_bank.decode(self.codes(target))

    def settings(self, target):
        self.validate_target(target)
        settings = self.base.settings(self._base_target(target))
        settings["modulations"] = deepcopy(self.routes)
        settings["lfos"] = [curve_to_lfo(curve) for curve in self.curves(target)]
        return settings

    def decode(self, target):
        document = json.loads(self.base.decode(self._base_target(target)))
        document["settings"] = self.settings(self.canonical_target(target))
        self.schema.validate(document).require_valid()
        return canonical_json(document).encode()

    def _source_codes(self, index):
        if index not in self._authored_codes:
            self._authored_codes[index] = self.lfo_bank.encode(np.stack([sample_lfo(shape) for shape in authored_lfos(index)]))[0]
        return self._authored_codes[index]

    def source_preset(self, target, index):
        """Original curve audio is the input; lossy codes are supervision only."""
        if self._join(self._base_target(target), self._source_codes(index)) != self.canonical_target(target):
            raise ValueError("saved curve target does not match its original authored source")
        document = json.loads(self.base.decode(self._base_target(target)))
        document["settings"].update(lfos=authored_lfos(index), modulations=deepcopy(self.routes))
        self.schema.validate(document).require_valid()
        return canonical_json(document).encode()

    def source_curves(self, raw):
        return np.stack([sample_lfo(shape) for shape in json.loads(raw)["settings"]["lfos"]])

    def encode(self, raw):
        document, report = self.schema.loads(raw)
        report.require_valid()
        if document["settings"]["modulations"] != self.routes:
            raise ValueError("curve codec requires its declared fixed ordered connections")
        codes, _ = self.lfo_bank.encode(np.stack([sample_lfo(shape) for shape in document["settings"]["lfos"]]))
        base = deepcopy(document)
        template = self.schema.template()["settings"]
        base["settings"]["lfos"], base["settings"]["modulations"] = template["lfos"], template["modulations"]
        return self._join(self.base.encode(canonical_json(base).encode()), codes)

    def default_target(self):
        if hasattr(self, "_default_target"):
            return self._default_target
        base = self.base.default_target()
        settings = self.base.settings(base)
        settings.update(osc_1_level=.08, sample_level=.06)
        scalar = self.base.scalar._target(settings)
        base = PilotTarget(scalar.continuous, scalar.categorical + base.categorical[-4:])
        codes, _ = self.lfo_bank.encode(np.stack([sample_lfo(shape) for shape in self.schema.template()["settings"]["lfos"]]))
        self._default_target = self._join(base, codes)
        return self._default_target

    def supervision_target(self, target):
        base = self.base.supervision_target(self._base_target(target))
        return PilotTarget(base.continuous + target.continuous[self._ncont:], base.categorical + target.categorical[self._ncat:])

    def active_lfos(self, target):
        settings = self.base._values(self._base_target(target))
        return tuple(any(settings[f"modulation_{2 * slot + channel + 1}_amount"] != 0
                         and not settings[f"modulation_{2 * slot + channel + 1}_bypass"]
                         and settings["osc_1_on" if channel == 0 else "sample_on"] for channel in range(2)) for slot in range(8))

    def _scalar_mask(self, name, active, fallback):
        match = re.match(r"lfo_(\d)_", name)
        if match:
            return active[int(match[1]) - 1]
        match = re.match(r"modulation_(\d+)_", name)
        return int(match[1]) <= 16 if match else fallback

    def continuous_mask(self, target):
        base = self._base_target(target)
        active = self.active_lfos(target)
        mask = tuple(self._scalar_mask(name, active, value) for (name, *_), value in zip(self.base.continuous, self.base.continuous_mask(base)))
        return mask + self.curve_parameter_masks(target, active)

    def curve_parameter_masks(self, target, active):
        codes = self.codes(target)
        phases, gains = [], []
        for slot, code in enumerate(codes):
            atoms = [self.lfo_bank.assets.base_dictionary[code.indices[0]],
                     *(layer[index] for layer, index in zip(self.lfo_bank.assets.residual_layer_dictionaries, code.indices[1:]))]
            phases.extend(bool(active[slot] and np.ptp(atom) != 0 and (layer == 0 or code.gains[layer - 1] != 0)) for layer, atom in enumerate(atoms))
            gains.extend(bool(active[slot] and np.any(atom != 0)) for atom in atoms[1:])
        return tuple(phases) + tuple(gains)

    def categorical_mask(self, target):
        base = self._base_target(target)
        active = self.active_lfos(target)
        names = self.base.scalar._categorical_controls
        mask = self.base.categorical_mask(base)
        scalar = tuple(self._scalar_mask(name, active, value) for name, value in zip(names, mask[:len(names)]))
        return scalar + mask[len(names):] + tuple(value for value in active for _ in range(17))

    def lineage_root(self, rng, index):
        root = self.base.lineage_root(rng, index)
        if root not in self.roots.by_state:
            raise ValueError("authored root is outside the curve bank's frozen split population; refit under the new population")
        return root

    def validate_manifest(self, manifest):
        parent = getattr(manifest, "parent", manifest)
        if any(self.roots.by_state.get(entry.state_id) != entry for entry in parent.assignments):
            raise ValueError("curve pool moved a root across the dictionary's frozen splits")

    def sample_target(self, rng, index):
        probe = random.Random()
        probe.setstate(rng.getstate())
        root = self.lineage_root(probe, index)
        base = self.base.sample_target(rng, index)
        settings = self.base.settings(base)
        for name in ("osc_1_level", "osc_2_level", "osc_3_level", "sample_level"):
            settings[name] *= .25
        for route in range(1, 17):
            settings[f"modulation_{route}_amount"] = 0.
        active = self._primary[root]
        channel = 1 if settings["sample_on"] else 0
        settings[f"modulation_{2 * active + channel + 1}_amount"] = rng.uniform(.015, .04)
        settings[f"lfo_{active + 1}_frequency"] = rng.uniform(-1, 1)
        settings[f"lfo_{active + 1}_sync"] = 0.
        scalar = self.base.scalar._target(settings)
        base = PilotTarget(scalar.continuous, scalar.categorical + base.categorical[-4:])
        return self._join(base, self._source_codes(index))

    def family(self, target):
        active = self.active_lfos(target)
        return f"lfo_{next((slot for slot, value in enumerate(active, 1) if value), 1)}"

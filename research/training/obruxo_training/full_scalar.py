"""All 775 scalar targets, with legal order/mask outputs and explicit fixed nested state."""
import json
import math
import re

import numpy as np

from obruxo_data.hashing import canonical_json, canonical_sha256
from obruxo_data.vital.atlas import ParameterSpec
from obruxo_data.vital.full_contract import FullPresetContract

from .assets import AssetCatalog, AssetReference, asset_digest
from .codec import PilotCodec, PilotTarget
from .contracts import TrainingContract
from .identity import DISPLAY_FIELDS, PresetIdentity, parse_document


FAMILIES = ("oscillators", "filters", "envelopes", "voice", "effects")
EFFECTS = ("chorus", "compressor", "delay", "distortion", "eq", "filter_fx", "flanger", "phaser", "reverb")
QUANTIZE = tuple(f"osc_{index}_transpose_quantize" for index in range(1, 4)) + ("sample_transpose_quantize",)
CYCLIC = {f"osc_{index}_phase" for index in range(1, 4)} | {f"lfo_{index}_phase" for index in range(1, 9)}


def effect_order(rank):
    """Decode Vital's inversion rank: choose each last item from the remaining sorted IDs."""
    available, order = list(range(9)), [0] * 9
    for size in range(9, 0, -1):
        rank, inversions = divmod(rank, size)
        order[size - 1] = available.pop(size - 1 - inversions)
    return tuple(order)


def effect_rank(order):
    if sorted(order) != list(range(9)):
        raise ValueError("effect order must be a permutation of nine IDs")
    rank = 0
    for index, value in enumerate(order[1:], 1):
        rank = rank * (index + 1) + sum(previous > value for previous in order[:index])
    return rank


def scalar_family(name):
    if name.startswith("modulation_"):
        return "modulation"
    if name.startswith(("lfo_", "random_")):
        return "modulators"
    if name.startswith("env_"):
        return "envelopes"
    if name.startswith("osc_"):
        return "oscillators"
    if name.startswith("sample_"):
        return "sampler"
    if name.startswith(tuple(effect + "_" for effect in EFFECTS)) or name == "effect_chain_order":
        return "effects" if name.startswith("filter_fx_") or not name.startswith("filter_") else "filters"
    if name.startswith("filter_"):
        return "filters"
    return "voice_global"


class FullScalarCodec:
    generator = "independent_authored_full_scalar_v1"
    balanced_loss = True

    @property
    def codec_id(self):
        return canonical_sha256(self.to_dict())

    def validate_target(self, target):
        PilotCodec.validate_target(self, target)

    def __init__(self):
        self.schema = FullPresetContract()
        template = self.schema.template()
        template.update(author="OBRUXO", comments="Authored full scalar v1", preset_name="OBRUXO scalar coverage")
        self._template_json = canonical_json(template)
        self._defaults = {name: template["settings"][name] for name in self.schema.scalar_names}
        self._routes = template["settings"]["modulations"]
        typed = parse_document(self._template_json.encode())["settings"]
        references, seen = [], set()
        for kind, assets in (("wavetable", typed["wavetables"]), ("sample", [typed["sample"]])):
            for index, asset in enumerate(assets):
                digest = asset_digest(kind, asset)
                if (kind, digest) not in seen:
                    references.append(AssetReference(f"full-init-{kind}-{index}", kind, digest, self.schema.spec["template_sha256"], "reviewed_init"))
                    seen.add((kind, digest))
        self.catalog = AssetCatalog(tuple(references))
        self.contract = TrainingContract.create(self.schema, self.catalog)
        self._parameters, continuous, categorical = {}, [], []
        self._categorical_controls = []
        settings = template["settings"]
        for name in sorted(self.schema.scalar_names - {"effect_chain_order", *QUANTIZE}):
            raw = self.schema.parameters[name]
            baseline = self.schema.baseline_parameters.get(name, {})
            discrete = raw["is_discrete"] or baseline.get("is_discrete", False) or baseline.get("scale") == "Indexed"
            metadata = {**baseline, "min": raw["min"], "max": raw["max"], "default": settings[name], "is_discrete": discrete}
            spec = ParameterSpec.from_dict(name, {"scale": "Linear", **metadata})
            self._parameters[name] = spec
            if discrete:
                options = tuple(range(math.ceil(spec.minimum), math.floor(spec.maximum) + 1))
                categorical.append((name, options))
                self._categorical_controls.append(name)
            else:
                continuous.append((name, spec.minimum, spec.maximum, settings[name]))
        self._ordinary_categories = len(categorical)
        for radix in range(2, 10):
            categorical.append((f"effect_order_digit_{radix}", tuple(range(radix))))
            self._categorical_controls.append("effect_chain_order")
        for name in QUANTIZE:
            for bit in range(13):
                categorical.append((f"{name}_bit_{bit}", (0, 1)))
                self._categorical_controls.append(name)
        self.continuous, self.categorical = tuple(continuous), tuple(categorical)
        self.continuous_groups = tuple(scalar_family(name) for name, *_ in self.continuous)
        self.categorical_groups = tuple(scalar_family(name) for name in self._categorical_controls)
        self.cyclic_indices = tuple(index for index, (name, *_) in enumerate(self.continuous) if name in CYCLIC)
        if len(self._parameters) + 1 + len(QUANTIZE) != 775 or max(len(options) for _, options in categorical) > 128:
            raise ValueError("full scalar output inventory changed; review its head budget")

    def to_dict(self):
        return {"version": 1, "codec": "authored_full_scalar_v1", "contract_id": self.contract.contract_id,
                "template_sha256": canonical_sha256(json.loads(self._template_json)), "scalar_count": 775,
                "continuous": [{"name": name, "minimum": lo, "maximum": hi, "inactive_default": default,
                                "mapping": self._parameters[name].scale} for name, lo, hi, default in self.continuous],
                "categorical": [{"name": name, "values": list(options)} for name, options in self.categorical],
                "effect_order": "mixed_radix_digits_2_through_9_bijection_to_9_factorial_ranks",
                "effect_order_supervision": "retain_relative_enabled_order_append_disabled_ids_in_ascending_order",
                "pitch_quantization": "four_13_bit_masks_no_fractional_predictions",
                "loss": "family_balanced_masked_atlas_coordinates_cyclic_phase_v1", "cyclic_controls": sorted(CYCLIC),
                "serialization": "atlas_coordinates_projected_to_native_float32_linear_control_grid_v1",
                "lineage": "unprojected_authored_recipe_roots_frozen_before_native_grid_v1",
                "inactive_policy": "mask_supervision_preserve_stored_values", "families": list(FAMILIES),
                "nested_policy": "all_six_structures_fixed_to_reviewed_full_template_reject_other_state",
                "scope": "all_775_scalar_fields_with_fixed_factory_assets_curves_seeds_and_empty_routes"}

    def canonical_target(self, target):
        self.validate_target(target)
        values = []
        for (name, lo, hi, _), value in zip(self.continuous, target.continuous, strict=True):
            spec = self._parameters[name]
            raw, minimum = np.float32(spec.raw_from_normalized(value)), np.float32(lo)
            span = np.float32(np.float32(hi) - minimum)
            position = np.float32(np.float32(raw - minimum) / span)
            native = float(np.float32(minimum + np.float32(position * span)))
            values.append(spec.normalized_from_raw(min(hi, max(lo, native))))
        return PilotTarget(tuple(values), tuple(target.categorical))

    def _target(self, settings, *, project=True):
        values = tuple(self._parameters[name].normalized_from_raw(settings[name]) for name, *_ in self.continuous)
        categories = [options.index(settings[name]) for name, options in self.categorical[:self._ordinary_categories]]
        order = settings["effect_chain_order"]
        if type(order) not in {int, float} or not float(order).is_integer() or not 0 <= order < math.factorial(9):
            raise ValueError("effect order requires an integer permutation rank")
        order = int(order)
        for radix in range(2, 10):
            order, digit = divmod(order, radix)
            categories.append(digit)
        for name in QUANTIZE:
            value = settings[name]
            if type(value) not in {int, float} or not float(value).is_integer() or not 0 <= value < 8192:
                raise ValueError("pitch quantization requires an integer 13-bit mask")
            categories.extend((int(value) >> bit) & 1 for bit in range(13))
        target = PilotTarget(values, tuple(categories))
        return self.canonical_target(target) if project else target

    def _values(self, target):
        self.validate_target(target)
        settings = self._defaults.copy()
        for (name, *_), value in zip(self.continuous, target.continuous, strict=True):
            settings[name] = self._parameters[name].raw_from_normalized(value)
        for (name, options), index in zip(self.categorical[:self._ordinary_categories], target.categorical[:self._ordinary_categories], strict=True):
            settings[name] = float(options[index])
        cursor, multiplier, order = self._ordinary_categories, 1, 0
        for radix in range(2, 10):
            order += target.categorical[cursor] * multiplier
            multiplier *= radix
            cursor += 1
        settings["effect_chain_order"] = float(order)
        for name in QUANTIZE:
            settings[name] = float(sum(target.categorical[cursor + bit] << bit for bit in range(13)))
            cursor += 13
        return settings

    def settings(self, target):
        settings = json.loads(self._template_json)["settings"]
        settings.update(self._values(target))
        return settings

    def decode(self, target):
        document = json.loads(self._template_json)
        document["settings"] = self.settings(self.canonical_target(target))
        self.schema.validate(document).require_valid()
        return canonical_json(document).encode()

    def encode(self, raw):
        document, report = self.schema.loads(raw)
        report.require_valid()
        try:
            target = self._target(document["settings"])
        except ValueError as error:
            raise ValueError("scalar state is outside the full codec domain") from error
        expected = json.loads(self._template_json)
        expected["settings"].update({name: document["settings"][name] for name in self.schema.scalar_names})
        if {k: v for k, v in document.items() if k not in DISPLAY_FIELDS} != {k: v for k, v in expected.items() if k not in DISPLAY_FIELDS}:
            raise ValueError("nested or metadata state outside the full scalar codec; no silent projection")
        return target

    def default_target(self):
        return self._target(json.loads(self._template_json)["settings"])

    def supervision_target(self, target):
        settings = self._values(target)
        order = effect_order(int(settings["effect_chain_order"]))
        active = [index for index in order if settings[EFFECTS[index] + "_on"]]
        canonical = effect_rank((*active, *(index for index in range(9) if index not in active)))
        categories = list(target.categorical)
        for index, radix in enumerate(range(2, 10), self._ordinary_categories):
            canonical, categories[index] = divmod(canonical, radix)
        return PilotTarget(target.continuous, tuple(categories))

    def _active(self, name, settings):
        match = re.match(r"(osc_\d|filter_[12]|sample|env_\d|lfo_\d|random_\d|modulation_\d+)_", name)
        if match:
            module = match[1]
            if module.startswith("modulation_"):
                return bool(self._routes[int(module.split("_")[1]) - 1]["source"])
            if module.startswith(("lfo_", "random_")) or (module.startswith("env_") and module != "env_1"):
                return any(route["source"] == module for route in self._routes)
            return name == module + "_on" or bool(settings.get(module + "_on", 1))
        if name == "effect_chain_order":
            return sum(settings[effect + "_on"] != 0 for effect in EFFECTS) >= 2
        for effect in EFFECTS:
            if name.startswith(effect + "_"):
                return name == effect + "_on" or bool(settings[effect + "_on"])
        return True

    def continuous_mask(self, target):
        settings = self._values(target)
        return tuple(self._active(name, settings) for name, *_ in self.continuous)

    def categorical_mask(self, target):
        settings = self._values(target)
        return tuple(self._active(name, settings) for name in self._categorical_controls)

    def _sample_settings(self, rng, index):
        settings = json.loads(self._template_json)["settings"]
        settings.update(osc_1_level=rng.uniform(.18, .35), osc_1_pan=rng.uniform(-.6, .6),
                        osc_1_unison_voices=float(rng.choice((1, 2, 4))), osc_1_unison_detune=rng.uniform(.5, 1.5))
        family = index % len(FAMILIES)
        if family == 0:
            settings.update(osc_2_on=1., osc_2_level=rng.uniform(.08, .18), osc_2_transpose=float(rng.choice((-12, 0, 12))),
                            osc_3_on=float(index % 2), osc_3_level=.1)
        elif family == 1:
            settings.update(filter_1_on=1., filter_1_cutoff=rng.uniform(60, 110), filter_1_resonance=rng.uniform(.15, .5))
        elif family == 2:
            settings.update(env_1_attack=rng.uniform(.01, .45), env_1_decay=rng.uniform(.1, .8),
                            env_1_sustain=rng.uniform(.25, .85), env_1_release=rng.uniform(.1, .6))
        elif family == 3:
            settings.update(voice_override=1., voice_priority=float(rng.randrange(5)), legato=float(rng.randrange(2)),
                            portamento_time=rng.uniform(-10, -5))
        else:
            settings.update(chorus_on=1., chorus_dry_wet=rng.uniform(.1, .25), distortion_on=1., distortion_mix=.15,
                            effect_chain_order=float(rng.randrange(math.factorial(9))))
        return settings

    def sample_target(self, rng, index):
        return self._target(self._sample_settings(rng, index))

    def lineage_root(self, rng, index):
        target = self._target(self._sample_settings(rng, index), project=False)
        document = json.loads(self._template_json)
        document["settings"] = self.settings(target)
        return PresetIdentity.from_bytes(canonical_json(document).encode()).state

    def family(self, target):
        settings = self._values(target)
        if settings["chorus_on"] or settings["distortion_on"]:
            return "effects"
        if settings["voice_override"]:
            return "voice"
        if settings["filter_1_on"]:
            return "filters"
        if settings["osc_2_on"]:
            return "oscillators"
        return "envelopes"

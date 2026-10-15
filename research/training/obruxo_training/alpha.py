"""A broader authored vocabulary: factory assets, layering, filter motion and chorus."""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from obruxo_data.hashing import canonical_json, canonical_sha256
from obruxo_data.vital import VitalPreset, VitalSchema

from .assets import AssetCatalog, AssetReference, asset_digest
from .codec import CONTINUOUS, PilotCodec, PilotTarget
from .contracts import TrainingContract
from .identity import DISPLAY_FIELDS, parse_document


ALPHA_CONTINUOUS = (*CONTINUOUS, ("osc_1_wave_frame", 0., 255., 0.), ("osc_2_level", .08, .25, .15),
                    ("osc_2_pan", -.7, .7, 0.), ("osc_2_tune", -.2, .2, 0.), ("osc_2_wave_frame", 0., 255., 0.),
                    ("lfo_1_frequency", -2., 2., 1.), ("modulation_1_amount", 0., .35, 0.),
                    ("chorus_dry_wet", .1, .35, .2), ("chorus_mod_depth", .1, .6, .3), ("chorus_frequency", -3., 0., -2.))
FAMILIES = ("wavetable", "layered", "filtered", "modulated", "chorus")
LFO_SHAPES = (
    {"name": "Triangle", "num_points": 3, "points": [0., 1., .5, 0., 1., 1.], "powers": [0., 0., 0.], "smooth": False},
    {"name": "Ramp up", "num_points": 2, "points": [0., 1., 1., 0.], "powers": [0., 0.], "smooth": False},
    {"name": "Ramp down", "num_points": 2, "points": [0., 0., 1., 1.], "powers": [0., 0.], "smooth": False},
)
FACTORY_IMPORTS = {
    "4bc81073a8b6f12921d3bd3df4ce677ab313eefebc529018b0a2202b0282c834": "header_0_8_1",
    "d3e99ef899b0d45c08e692fdd0cc7094dcaf8da5941257ae6d3f2ab8eeb7a266": "audio_source_0_3_9",
    "617cc3abb43b534532432b5fd435269c0b39bcfb509c085e0723ed671adf232f": "audio_source_0_3_9",
}


def import_wavetable(raw: bytes) -> dict:
    """Apply only the three hash-pinned, native-compared factory import migrations."""
    parse_document(raw)
    wave = json.loads(raw)
    policy = FACTORY_IMPORTS.get(hashlib.sha256(raw).hexdigest())
    if policy is None:
        return wave
    if policy == "audio_source_0_3_9":
        wave["full_normalize"] = wave.pop("post_normalize")
        wave.setdefault("author", "")
        for group in wave["groups"]:
            for component in group["components"]:
                if component["type"] == "Audio File Source":
                    component.update(interpolation_style=1, normalize_mult=True)
    wave["version"] = "1.0.8"
    return wave


def read_wavetable_bank(manifest_path: Path, root: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or type(manifest.get("version")) is not int or manifest["version"] != 1 or not isinstance(manifest.get("references"), list):
        raise ValueError("expected wavetable reference manifest version 1")
    root = root.resolve(strict=True)
    entries = []
    for entry in manifest["references"]:
        path = (root / entry["path"]).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError("reference path escapes the selected asset root")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["source_sha256"]:
            raise ValueError("wavetable reference source hash mismatch")
        entries.append({"asset_id": entry["asset_id"], "provenance": entry["provenance"],
                        "source_sha256": entry["source_sha256"], "source_json": raw.decode("utf-8")})
    return {"version": 1, "references": entries}


class AlphaCodec(PilotCodec):
    continuous = ALPHA_CONTINUOUS
    generator = "independent_authored_alpha_v1"

    def __init__(self, bank: dict, schema=None):
        self.schema = schema or VitalSchema.load()
        if (not isinstance(bank, dict) or type(bank.get("version")) is not int or bank["version"] != 1
                or not isinstance(bank.get("references"), list) or not 1 <= len(bank["references"]) <= 7):
            raise ValueError("alpha needs one through seven pinned wavetable references")
        # Keep payloads in pools/bundles once, never once per state; inference needs no asset installation.
        self._bank_json = canonical_json(bank)
        template = self.schema.load_init_document()
        self._waves = [deepcopy(template["settings"]["wavetables"][0])]
        references = list(AssetCatalog.reviewed_init(self.schema).references)
        self.asset_ids = ["reviewed-init-wavetable"]
        for entry in bank["references"]:
            raw = entry["source_json"].encode("utf-8")
            if hashlib.sha256(raw).hexdigest() != entry["source_sha256"] or entry["provenance"] not in {"factory_export", "authored_reference"}:
                raise ValueError("embedded wavetable source or provenance changed")
            wave = import_wavetable(raw)
            if not isinstance(wave.get("groups"), list) or not wave["groups"] or not isinstance(wave.get("version"), str):
                raise ValueError("alpha references require explicit-version wavetable exports; legacy tables need separate qualification")
            digest = asset_digest("wavetable", parse_document(canonical_json(wave).encode()))
            references.append(AssetReference(entry["asset_id"], "wavetable", digest, entry["source_sha256"], entry["provenance"]))
            self._waves.append(wave)
            self.asset_ids.append(entry["asset_id"])
        self.catalog = AssetCatalog(tuple(references))
        self.contract = TrainingContract.create(self.schema, self.catalog)
        self.categorical = (("osc_1_unison_voices", (1, 2, 4)), ("filter_1_on", (0, 1)), ("osc_2_on", (0, 1)),
                            ("osc_2_transpose", (0, -12, 12)), ("osc_1_asset", tuple(self.asset_ids)),
                            ("osc_2_asset", tuple(self.asset_ids)), ("lfo_1_filter_route", (0, 1)),
                            ("lfo_1_shape", ("triangle", "ramp_up", "ramp_down")), ("chorus_on", (0, 1)))
        template.update(author="OBRUXO", comments="Authored alpha v1", preset_name="OBRUXO alpha", synth_version="1.0.8")
        for name, lo, hi, default in self.continuous:
            if any(self.schema.parameters[name].validate_raw(value) for value in (lo, hi, default)):
                raise ValueError(f"alpha domain exceeds reviewed schema: {name}")
            template["settings"][name] = default
        for name, options in self.categorical:
            if name in self.schema.parameters:
                if any(self.schema.parameters[name].validate_raw(value) for value in options):
                    raise ValueError(f"alpha category exceeds reviewed schema: {name}")
                template["settings"][name] = float(options[0])
        template["settings"].update(osc_2_destination=0., lfo_1_sync=0., lfo_1_sync_type=0.)
        template["settings"]["lfos"][0] = deepcopy(LFO_SHAPES[0])
        VitalPreset(template, self.schema).validate().require_valid()
        self._template_json = canonical_json(template)

    def to_dict(self):
        return {"version": 1, "codec": "authored_alpha_v1", "contract_id": self.contract.contract_id,
                "template_sha256": canonical_sha256(json.loads(self._template_json)), "wavetable_bank": json.loads(self._bank_json),
                "wavetable_import": "hash_pinned_factory_to_1_0_8_v1",
                "imported_wavetable_sha256": [canonical_sha256(wave) for wave in self._waves],
                "continuous": [{"name": name, "minimum": lo, "maximum": hi, "inactive_default": default}
                               for name, lo, hi, default in self.continuous],
                "categorical": [{"name": name, "values": list(values)} for name, values in self.categorical],
                "normalization": "linear_in_stored_units", "inactive_policy": "fixed_defaults_and_masked_supervision",
                "routes": "slot_1_lfo_1_to_filter_1_cutoff_only_when_filter_enabled; all other ordered slots fixed",
                "lfo_shapes": deepcopy(list(LFO_SHAPES)), "effect_order": "reviewed_init_fixed", "families": list(FAMILIES),
                "scope": "two_oscillators_pinned_factory_assets_filter1_env1_lfo1_filter_route_chorus"}

    def categorical_mask(self, target):
        self.validate_target(target)
        c = target.categorical
        return (True, True, True, bool(c[2]), True, bool(c[2]), bool(c[1]), bool(c[1] and c[6]), True)

    def continuous_mask(self, target):
        self.validate_target(target)
        c = target.categorical
        # Reviewed init is a single static frame: scanning it has no audible target.
        return (True, True, c[0] != 0, bool(c[1]), bool(c[1]), True, True, True, True, bool(c[4]),
                *([bool(c[2])] * 3), bool(c[2] and c[5]), bool(c[1] and c[6]), bool(c[1] and c[6]), *([bool(c[8])] * 3))

    def canonical_target(self, target):
        self.validate_target(target)
        categories = tuple(index if active else 0 for index, active in zip(target.categorical, self.categorical_mask(target), strict=True))
        target = PilotTarget(target.continuous, categories)
        values = tuple(float(value) if active else (default-lo)/(hi-lo)
                       for value, active, (_, lo, hi, default) in zip(target.continuous, self.continuous_mask(target), self.continuous, strict=True))
        return PilotTarget(values, categories)

    def decode(self, target):
        target = self.canonical_target(target)
        document = json.loads(self._template_json)
        settings = document["settings"]
        for (name, lo, hi, _), value in zip(self.continuous, target.continuous, strict=True):
            settings[name] = lo + value * (hi-lo)
        for (name, options), index in zip(self.categorical, target.categorical, strict=True):
            if name in self.schema.parameters:
                settings[name] = float(options[index])
        c = target.categorical
        settings["wavetables"][:2] = [deepcopy(self._waves[c[4]]), deepcopy(self._waves[c[5]])]
        settings["lfos"][0] = deepcopy(LFO_SHAPES[c[7]])
        if c[6]:
            settings["modulations"][0] = {"source": "lfo_1", "destination": "filter_1_cutoff"}
        return canonical_json(document).encode()

    def encode(self, raw):
        parse_document(raw)
        document = json.loads(raw.decode("utf-8-sig"))
        VitalPreset(document, self.schema).validate().require_valid()
        settings = document["settings"]
        try:
            virtual = {"osc_1_asset": self._waves.index(settings["wavetables"][0]),
                       "osc_2_asset": self._waves.index(settings["wavetables"][1]),
                       "lfo_1_filter_route": ({"source": "", "destination": ""}, {"source": "lfo_1", "destination": "filter_1_cutoff"}).index(settings["modulations"][0]),
                       "lfo_1_shape": LFO_SHAPES.index(settings["lfos"][0])}
            categories = tuple(virtual[name] if name in virtual else options.index(settings[name]) for name, options in self.categorical)
        except ValueError as error:
            raise ValueError("asset, route, shape or category outside alpha vocabulary") from error
        values = tuple((settings[name]-lo)/(hi-lo) for name, lo, hi, _ in self.continuous)
        target = self.canonical_target(PilotTarget(values, categories))
        expected = json.loads(self.decode(target))
        for name, *_ in self.continuous:
            if not math.isclose(settings[name], expected["settings"][name], rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"inactive alpha control changed: {name}")
            expected["settings"][name] = settings[name]
        if {k: v for k, v in document.items() if k not in DISPLAY_FIELDS} != {k: v for k, v in expected.items() if k not in DISPLAY_FIELDS}:
            raise ValueError("preset has fixed or unsupported state outside alpha codec")
        return target

    def sample_target(self, rng, index):
        stage = index % len(FAMILIES)
        values = [rng.random() for _ in self.continuous]
        categories = [rng.randrange(len(options)) for _, options in self.categorical]
        categories[1], categories[2], categories[6], categories[8] = int(stage >= 2), int(stage >= 1), int(stage >= 3), int(stage >= 4)
        if stage >= 3:
            values[15] = rng.uniform(.25, .75)
        return self.canonical_target(PilotTarget(tuple(values), tuple(categories)))

    def family(self, target):
        c = target.categorical
        return FAMILIES[4 if c[8] else 3 if c[6] else 2 if c[1] else 1 if c[2] else 0]

"""Full scalar learning plus registered factory wavetable and sample choices."""
from copy import deepcopy
import json

from obruxo_data.hashing import canonical_json, canonical_sha256
from obruxo_data.render.base import FULL_PRESET_RENDERER_ID

from .assets import AssetCatalog, AssetReference, asset_digest
from .codec import PilotCodec, PilotTarget
from .contracts import TrainingContract
from .factory_bank import IMPORT_POLICY
from .full_scalar import FullScalarCodec
from .identity import parse_document, require_digest


def _content(kind, value):
    typed = parse_document(canonical_json({"asset": value}).encode())["asset"]
    return asset_digest(kind, typed)


def validated_bank(document):
    if document.get("format") != "obruxo_factory_asset_bank_v1":
        raise ValueError("unsupported factory asset bank")
    bank, qualification = document["bank"], document["qualification"]
    digest = canonical_sha256(bank)
    if document["bank_id"] != digest or qualification["bank_id"] != digest:
        raise ValueError("factory bank content or qualification identity changed")
    if bank.get("version") != 1 or bank.get("policy") != IMPORT_POLICY or not bank.get("references"):
        raise ValueError("unsupported or empty factory import vocabulary")
    count = len(bank["references"])
    if (qualification.get("qualified") is not True or qualification.get("source_unchanged") is not True
            or qualification.get("references_passed") != count or qualification.get("references_requested") != count
            or qualification.get("renderer_id") != FULL_PRESET_RENDERER_ID):
        raise ValueError("factory bank needs complete native qualification for the pinned renderer")
    require_digest(bank["source_manifest_sha256"])
    ids = set()
    for entry in bank["references"]:
        if entry["asset_id"] in ids or entry["kind"] not in {"wavetable", "sample", "lfo"}:
            raise ValueError("duplicate or unsupported factory reference")
        if entry["provenance"] not in {"factory_export", "authored_reference"}:
            raise ValueError("factory vocabulary references need reviewed provenance")
        require_digest(entry["source_sha256"])
        if canonical_sha256(entry["asset"]) != require_digest(entry["asset_sha256"]):
            raise ValueError("canonical factory asset changed")
        ids.add(entry["asset_id"])
    # Keep execution timings outside codec identity; payloads and qualification
    # policy remain pinned and are embedded once in the private inference bundle.
    qualification = {name: qualification[name] for name in ("qualified", "bank_id", "renderer_id", "source_unchanged",
                                                            "references_passed", "references_requested", "plan_sha256")}
    return {"format": document["format"], "bank_id": digest, "bank": deepcopy(bank), "qualification": qualification}


class FactoryAssetCodec:
    # Asset mutations keep the original authored scalar roots and reserved splits.
    generator = FullScalarCodec.generator
    model_family = "factory_asset"
    balanced_loss = True

    def __init__(self, bank):
        self.scalar = FullScalarCodec()
        self.schema = self.scalar.schema
        self.bank = validated_bank(bank)
        template = json.loads(self.scalar._template_json)
        self._waves, self._samples = [template["settings"]["wavetables"][0]], [template["settings"]["sample"]]
        self._wave_ids = [self.scalar.catalog.identify("wavetable", parse_document(self.scalar._template_json.encode())["settings"]["wavetables"][0]).asset_id]
        self._sample_ids = [self.scalar.catalog.identify("sample", parse_document(self.scalar._template_json.encode())["settings"]["sample"]).asset_id]
        references = list(self.scalar.catalog.references)
        seen = {(entry.kind, entry.content_sha256): entry.asset_id for entry in references}
        self.aliases = {}
        for entry in self.bank["bank"]["references"]:
            document = deepcopy(template)
            kind = entry["kind"]
            if kind == "lfo":
                document["settings"]["lfos"][0] = deepcopy(entry["asset"])
                self.schema.validate(document).require_valid()
                continue
            if kind == "wavetable":
                document["settings"]["wavetables"][0] = deepcopy(entry["asset"])
            else:
                document["settings"]["sample"] = deepcopy(entry["asset"])
            self.schema.validate(document).require_valid()
            content = _content(kind, entry["asset"])
            key = kind, content
            if key in seen:
                self.aliases[entry["asset_id"]] = seen[key]
                continue
            references.append(AssetReference(entry["asset_id"], kind, content, entry["source_sha256"], entry["provenance"]))
            seen[key] = entry["asset_id"]
            assets, asset_ids = (self._waves, self._wave_ids) if kind == "wavetable" else (self._samples, self._sample_ids)
            assets.append(deepcopy(entry["asset"]))
            asset_ids.append(entry["asset_id"])
        if len(self._waves) < 2 or len(self._samples) < 2:
            raise ValueError("factory learning needs non-default wavetable and sample choices")
        self.catalog = AssetCatalog(tuple(references))
        self.contract = TrainingContract.create(self.schema, self.catalog)
        self.continuous = self.scalar.continuous
        self._scalar_categories = len(self.scalar.categorical)
        self.categorical = self.scalar.categorical + tuple((f"osc_{index}_asset", tuple(self._wave_ids)) for index in range(1, 4)) + (("sample_asset", tuple(self._sample_ids)),)
        self.continuous_groups = self.scalar.continuous_groups
        self.categorical_groups = self.scalar.categorical_groups + ("oscillators",) * 3 + ("sampler",)
        self.cyclic_indices = self.scalar.cyclic_indices
        self._wave_index = {_content("wavetable", value): index for index, value in enumerate(self._waves)}
        self._sample_index = {_content("sample", value): index for index, value in enumerate(self._samples)}
        self._static_waves = []
        for value in self._waves:
            components = [component for group in value["groups"] for component in group["components"]]
            self._static_waves.append(len(components) == 1 and components[0]["type"] == "Wave Source" and len(components[0]["keyframes"]) == 1)

    @property
    def codec_id(self):
        return canonical_sha256(self.to_dict())

    def to_dict(self):
        spec = self.scalar.to_dict()
        spec.update(codec="authored_factory_assets_v1", contract_id=self.contract.contract_id, asset_bank=deepcopy(self.bank),
                    categorical=[{"name": name, "values": list(options)} for name, options in self.categorical],
                    asset_aliases=dict(self.aliases), families=[*spec["families"], "sampler"],
                    asset_metadata_policy="canonical_registered_names_and_authors",
                    nested_policy="registered_wavetables_and_samples_all_curves_seeds_and_routes_fixed_to_template",
                    scope="all_775_scalars_three_registered_wavetable_slots_and_registered_sampler_fixed_curves_routes_seeds")
        return spec

    def validate_target(self, target):
        PilotCodec.validate_target(self, target)

    def _split(self, target):
        return PilotTarget(target.continuous, target.categorical[:self._scalar_categories])

    def canonical_target(self, target):
        self.validate_target(target)
        scalar = self.scalar.canonical_target(self._split(target))
        return PilotTarget(scalar.continuous, scalar.categorical + target.categorical[self._scalar_categories:])

    def _values(self, target):
        self.validate_target(target)
        return self.scalar._values(self._split(target))

    def settings(self, target):
        self.validate_target(target)
        settings = self.scalar.settings(self._split(target))
        first, second, third, sample = target.categorical[self._scalar_categories:]
        settings["wavetables"] = [deepcopy(self._waves[index]) for index in (first, second, third)]
        settings["sample"] = deepcopy(self._samples[sample])
        return settings

    def decode(self, target):
        document = json.loads(self.scalar._template_json)
        document["settings"] = self.settings(self.canonical_target(target))
        self.schema.validate(document).require_valid()
        return canonical_json(document).encode()

    def encode(self, raw):
        document, report = self.schema.loads(raw)
        report.require_valid()
        settings = document["settings"]
        try:
            assets = tuple(self._wave_index[_content("wavetable", value)] for value in settings["wavetables"])
            sample = self._sample_index[_content("sample", settings["sample"])]
        except KeyError as error:
            raise ValueError("unregistered factory asset; no silent replacement") from error
        base = deepcopy(document)
        template = json.loads(self.scalar._template_json)["settings"]
        base["settings"]["wavetables"], base["settings"]["sample"] = template["wavetables"], template["sample"]
        scalar = self.scalar.encode(canonical_json(base).encode())
        return self.canonical_target(PilotTarget(scalar.continuous, scalar.categorical + assets + (sample,)))

    def default_target(self):
        scalar = self.scalar.default_target()
        return PilotTarget(scalar.continuous, scalar.categorical + (0, 0, 0, 0))

    def supervision_target(self, target):
        scalar = self.scalar.supervision_target(self._split(target))
        return PilotTarget(scalar.continuous, scalar.categorical + target.categorical[self._scalar_categories:])

    def continuous_mask(self, target):
        mask = list(self.scalar.continuous_mask(self._split(target)))
        for slot, asset in enumerate(target.categorical[self._scalar_categories:self._scalar_categories + 3], 1):
            if self._static_waves[asset]:
                index = next(index for index, (name, *_) in enumerate(self.continuous) if name == f"osc_{slot}_wave_frame")
                mask[index] = False
        return tuple(mask)

    def categorical_mask(self, target):
        settings = self._values(target)
        return self.scalar.categorical_mask(self._split(target)) + tuple(bool(settings[f"osc_{index}_on"]) for index in range(1, 4)) + (bool(settings["sample_on"]),)

    def lineage_root(self, rng, index):
        return self.scalar.lineage_root(rng, index)

    def sample_target(self, rng, index):
        scalar = self.scalar.sample_target(rng, index)
        settings = self.scalar.settings(scalar)
        waves, sample = [0, 0, 0], 0
        if index % 4 == 3:
            settings.update(osc_1_on=0., osc_2_on=0., osc_3_on=0., sample_on=1., sample_level=rng.uniform(.1, .25),
                            sample_keytrack=1., sample_loop=float(rng.randrange(2)))
            sample = (index // 4) % len(self._samples)
        else:
            position = index - (index + 1) // 4
            waves = [position % len(self._waves), rng.randrange(len(self._waves)), rng.randrange(len(self._waves))]
            for slot in range(1, 4):
                settings[f"osc_{slot}_wave_frame"] = float(rng.choice((0, 64, 128, 192, 256)))
        scalar = self.scalar._target(settings)
        return self.canonical_target(PilotTarget(scalar.continuous, scalar.categorical + tuple(waves) + (sample,)))

    def family(self, target):
        if self._values(target)["sample_on"]:
            return "sampler"
        return self.scalar.family(self._split(target))

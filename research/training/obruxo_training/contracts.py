from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

from obruxo_data.hashing import canonical_json, canonical_sha256
from obruxo_data.midi import Performance
from obruxo_data.render.base import LEGACY_TRAINING_RENDERER_ID, RenderRequest
from obruxo_data.vital import VitalSchema
from obruxo_data.vital.full_contract import FullPresetContract

from .assets import ASSET_RULE, AssetCatalog
from .eligibility import eligibility_policy, inspect_preset
from .identity import IDENTITY_RULE
from .performances import bank_document, bank_document_for, bank_for, capabilities_for
from .splits import SPLIT_RULE, SplitLedger


FULL_SCHEMA_ID = "vital-1.5.5-fields-native-1.6.4"


@dataclass(frozen=True)
class TrainingContract:
    schema_id: str
    schema_sha256: str
    asset_catalog_id: str
    performance_bank_id: str
    renderer_id: str = LEGACY_TRAINING_RENDERER_ID
    sample_rate: int = 44100
    channels: int = 2
    dtype: str = "float32"
    tail_seconds: float = 2.0
    render_workers: int = 1
    prefetch_clips: int = 32
    prefetch_bytes: int = 256 * 1024 * 1024

    @property
    def contract_id(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict:
        full = self.schema_id == FULL_SCHEMA_ID
        return {"version": 2 if full else 1, "identity_rule": IDENTITY_RULE, "asset_rule": ASSET_RULE, "split_rule": SPLIT_RULE,
                "eligibility": eligibility_policy(full_profile=full), "midi": capabilities_for(self.renderer_id).to_dict(),
                "recipe_rule": "render_recipe_v1", "audio_mode": "on_demand", **asdict(self)}

    @classmethod
    def create(cls, schema: VitalSchema, catalog: AssetCatalog) -> TrainingContract:
        if isinstance(schema, FullPresetContract):
            schema.template()  # Verify the reviewed factory bytes, as for the legacy contract.
            return cls(FULL_SCHEMA_ID, canonical_sha256(schema.spec), catalog.catalog_id,
                       canonical_sha256(bank_document_for(schema.renderer_id)), renderer_id=schema.renderer_id)
        paths = (schema.init_preset_path, schema.parameter_atlas_path, schema.modulation_vocab_path,
                 schema.init_preset_path.parent / "manifest.json")
        schema.load_init_document()
        digest = canonical_sha256({path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
        return cls(schema.schema_id, digest, catalog.catalog_id, canonical_sha256(bank_document()))

    def require_match(self, saved: dict) -> None:
        if saved != self.to_dict():
            raise ValueError("training contract changed; use an explicit new run instead of resuming")


@dataclass(frozen=True)
class RenderRecipe:
    contract_id: str
    split_manifest_id: str
    raw_sha256: str
    canonical_sha256: str
    state_id: str
    lineage_id: str
    split: str
    purpose: str
    performance_id: str
    performance_role: str
    sequence_id: int
    request_json: str

    @property
    def recipe_id(self) -> str:
        return canonical_sha256(asdict(self))

    def render_request(self) -> RenderRequest:
        return RenderRequest.from_dict(json.loads(self.request_json))

    @classmethod
    def create(cls, raw_preset: bytes, performance: Performance, *, schema: VitalSchema, catalog: AssetCatalog,
               contract: TrainingContract, ledger: SplitLedger, purpose: str, sequence_id: int) -> RenderRecipe:
        if purpose not in {"train", "evaluate"} or type(sequence_id) is not int or sequence_id < 0:
            raise ValueError("recipe requires a valid purpose and nonnegative sequence ID")
        TrainingContract.create(schema, catalog).require_match(contract.to_dict())
        inspection = inspect_preset(raw_preset, schema, catalog)
        if not inspection["static_eligible"]:
            raise ValueError("preset is not statically eligible: " + ", ".join(inspection["reasons"]))
        assignment = ledger.lookup(inspection["identity"]["state"])
        performance.validate(capabilities_for(contract.renderer_id)).require_valid()
        performance_id = canonical_sha256(performance.to_dict())
        entry = next((entry for entry in bank_for(contract.renderer_id) if entry.performance_id == performance_id), None)
        if entry is None:
            raise ValueError("performance is outside the frozen initial bank for this renderer")
        if purpose == "train" and (assignment.split != "train" or entry.role != "reference"):
            raise ValueError("training cannot consume a held-out state or transfer performance")
        request = RenderRequest(preset_json=raw_preset.decode("utf-8-sig"), performance=performance,
                                sample_rate=contract.sample_rate, channels=contract.channels, tail_seconds=contract.tail_seconds,
                                renderer_id=contract.renderer_id)
        return cls(contract.contract_id, ledger.manifest.manifest_id, inspection["identity"]["raw"], inspection["identity"]["canonical"],
                   assignment.state_id, assignment.lineage_id,
                   assignment.split, purpose, performance_id, entry.role, sequence_id, canonical_json(request.to_dict()))

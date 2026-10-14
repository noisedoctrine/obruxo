from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from obruxo_data.hashing import canonical_sha256
from obruxo_data.vital import VitalSchema

from .identity import parse_document, require_digest, typed_tree


ASSET_RULE = "asset_content_v1"


def asset_digest(kind: str, asset: dict) -> str:
    if kind not in {"wavetable", "sample"} or not isinstance(asset, dict):
        raise ValueError("asset must be a wavetable or sample object")
    # Deliberately conservative: retain versions, encodings, numeric spelling and all DSP state.
    content = {key: value for key, value in asset.items() if key not in {"name", "author"}}
    payload = json.dumps(typed_tree(content), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(f"{ASSET_RULE}:{kind}:{payload}".encode()).hexdigest()


@dataclass(frozen=True)
class AssetReference:
    asset_id: str
    kind: str
    content_sha256: str
    source_sha256: str
    provenance: str

    def __post_init__(self):
        if not isinstance(self.asset_id, str) or not self.asset_id or self.kind not in {"wavetable", "sample"}:
            raise ValueError("invalid asset reference")
        if self.provenance not in {"reviewed_init", "factory_export", "authored_reference"}:
            raise ValueError("asset reference requires explicit provenance")
        require_digest(self.content_sha256)
        require_digest(self.source_sha256)


@dataclass(frozen=True)
class AssetCatalog:
    references: tuple[AssetReference, ...]

    def __post_init__(self):
        ids, contents = set(), set()
        for item in self.references:
            key = (item.kind, item.content_sha256)
            if item.asset_id in ids or key in contents:
                raise ValueError("duplicate asset ID or ambiguous content reference")
            ids.add(item.asset_id)
            contents.add(key)
        object.__setattr__(self, "references", tuple(sorted(self.references, key=lambda item: item.asset_id)))

    @property
    def catalog_id(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict:
        return {"rule": ASSET_RULE, "references": [asdict(item) for item in sorted(self.references, key=lambda item: item.asset_id)]}

    def identify(self, kind: str, asset: dict) -> AssetReference | None:
        digest = asset_digest(kind, asset)
        return next((item for item in self.references if item.kind == kind and item.content_sha256 == digest), None)

    @classmethod
    def reviewed_init(cls, schema: VitalSchema) -> AssetCatalog:
        schema.load_init_document()  # Verify the reviewed init buffer before trusting its assets.
        raw = schema.init_preset_path.read_bytes()
        settings = parse_document(raw)["settings"]
        source = hashlib.sha256(raw).hexdigest()
        return cls((
            AssetReference("reviewed-init-wavetable", "wavetable", asset_digest("wavetable", settings["wavetables"][0]), source, "reviewed_init"),
            AssetReference("reviewed-init-sample", "sample", asset_digest("sample", settings["sample"]), source, "reviewed_init"),
        ))

    @classmethod
    def from_reference_manifest(cls, path: Path) -> AssetCatalog:
        """Verify each caller-declared reference export; never infer stock status from names."""
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or type(manifest.get("version")) is not int or manifest["version"] != 1 or not isinstance(manifest.get("references"), list):
            raise ValueError("expected asset reference manifest version 1")
        references = []
        for entry in manifest["references"]:
            source_path = path.parent / entry["path"]
            raw = source_path.read_bytes()
            source_hash = hashlib.sha256(raw).hexdigest()
            if source_hash != require_digest(entry["source_sha256"]):
                raise ValueError("asset reference source hash mismatch")
            kind, slot = entry["kind"], entry.get("slot")
            settings = parse_document(raw)["settings"]
            if kind == "wavetable":
                if type(slot) is not int or not 1 <= slot <= 3:
                    raise ValueError("wavetable reference slot must be 1, 2 or 3")
                asset = settings["wavetables"][slot - 1]
            elif kind == "sample" and slot is None:
                asset = settings["sample"]
            else:
                raise ValueError("invalid reference kind or slot")
            if entry["provenance"] not in {"factory_export", "authored_reference"}:
                raise ValueError("external references must declare factory_export or authored_reference provenance")
            references.append(AssetReference(entry["asset_id"], kind, asset_digest(kind, asset), source_hash, entry["provenance"]))
        return cls(tuple(references))

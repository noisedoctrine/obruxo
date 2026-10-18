"""Private frozen training membership with original presets read and checked on demand."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath

from obruxo_data.hashing import canonical_sha256
from obruxo_data.render.base import FULL_CURVE_RENDERER_ID
from obruxo_data.vital.full_contract import FullPresetContract

from .audit import load_snapshot, write_json
from .codec import PilotTarget, load_codec
from .contracts import TrainingContract
from .corpus_codec import CorpusNativeCodec
from .identity import PresetIdentity, require_digest
from .splits import SplitManifest

GENERATOR = "frozen_original_corpus_training_v1"


def source_policy():
    return {"rule": GENERATOR, "source_sha256_lf": hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
            "membership": "complete_static_eligible_training_subset_of_original_full_manifest",
            "audio": "original_source_bytes_phase_controls_unchanged", "checks_per_read": "raw_hash_state_identity_and_reencoded_target",
            "root_mapping": "private_relocatable_outside_pool_identity", "held_out_sources": "never_opened"}


def relative_source(value):
    if not isinstance(value, str) or not value or PureWindowsPath(value).drive or "\\" in value:
        raise ValueError("corpus source requires a portable relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value or not path.parts:
        raise ValueError("corpus source escapes its declared root")
    return path


@dataclass(frozen=True)
class CorpusEntry:
    state_id: str
    target: PilotTarget
    relative_path: str
    raw_sha256: str

    def to_dict(self):
        return {"state_id": self.state_id, "target": self.target.to_dict(), "relative_path": self.relative_path, "raw_sha256": self.raw_sha256}


@dataclass(frozen=True)
class CorpusPool:
    codec: CorpusNativeCodec
    manifest: SplitManifest
    entries: tuple[CorpusEntry, ...]
    corpus_root: Path
    source_contract: dict
    source_snapshot_id: str
    candidates_sha256: str

    def __post_init__(self):
        root = Path(self.corpus_root).resolve(strict=True)
        if not root.is_dir() or not isinstance(self.codec, CorpusNativeCodec) or not self.entries:
            raise ValueError("corpus pool requires original sources and the native corpus codec")
        object.__setattr__(self, "corpus_root", root)
        require_digest(self.source_snapshot_id)
        require_digest(self.candidates_sha256)
        schema = FullPresetContract(renderer_id=FULL_CURVE_RENDERER_ID)
        if schema.spec != {name: value for name, value in self.codec.schema.spec.items() if name != "corpus_native_policy"}:
            raise ValueError("corpus source and training schemas differ beyond the declared native policy")
        TrainingContract.create(schema, self.codec.catalog).require_match(self.source_contract)
        states = set()
        for entry in self.entries:
            require_digest(entry.raw_sha256)
            relative_source(entry.relative_path)
            assignment = self.manifest.by_state.get(entry.state_id)
            if entry.state_id in states or assignment is None or assignment.split != "train":
                raise ValueError("corpus pool contains duplicate, unregistered or held-out sources")
            if self.codec.canonical_target(entry.target) != entry.target:
                raise ValueError("corpus pool target is not canonical")
            states.add(entry.state_id)

    @property
    def seed(self):
        return self.manifest.seed

    @property
    def pool_id(self):
        return canonical_sha256(self.to_dict())

    def to_dict(self):
        return {"version": 2, "generator": GENERATOR, "source_policy": source_policy(), "codec": self.codec.to_dict(),
                "contract": self.codec.contract.to_dict(), "splits": self.manifest.to_dict(), "source_contract": self.source_contract,
                "source_snapshot_id": self.source_snapshot_id, "candidates_sha256": self.candidates_sha256,
                "entries": [entry.to_dict() for entry in self.entries]}

    @classmethod
    def create(cls, snapshot, corpus_root, candidates):
        labels = json.loads(Path(candidates).read_text(encoding="utf-8"))
        if labels["codec"].get("codec") != "canonical_corpus_inputs_v1":
            raise ValueError("corpus pool requires the qualified original corpus label policy")
        parent = load_codec(labels["codec"])
        codec = CorpusNativeCodec(parent.base.bank, parent.lfo_bank.to_dict())
        contract, catalog, manifest = load_snapshot(Path(snapshot), parent.schema)
        if labels["codec"] != codec.parent.to_dict() or catalog.catalog_id != codec.catalog.catalog_id or labels["split_manifest"] != manifest.to_dict():
            raise ValueError("corpus labels disagree with the frozen source snapshot or codec")
        records = json.loads((Path(snapshot) / "members.json").read_text(encoding="utf-8"))["records"]
        eligible = {}
        for row in records:
            if row["static_eligible"] and row["split"] == "train":
                eligible.setdefault(row["identity"]["state"], row)
        entries, states = [], set()
        for row in labels["entries"]:
            state = row["state_id"]
            member = eligible.get(state)
            if member is None or state in states or (row["relative_path"], row["raw_sha256"], row["lineage_id"]) != (
                member["relative_path"], member["identity"]["raw"], member["lineage_id"]):
                raise ValueError("corpus label source is missing, duplicated or outside frozen training membership")
            states.add(state)
            entries.append(CorpusEntry(state, PilotTarget.from_dict(row["target"]), row["relative_path"], row["raw_sha256"]))
        if states != set(eligible):
            raise ValueError("corpus labels must cover every eligible training source")
        status = json.loads((Path(snapshot) / "status.json").read_text(encoding="utf-8"))
        return cls(codec, manifest, tuple(sorted(entries, key=lambda entry: entry.state_id)), Path(corpus_root), contract.to_dict(),
                   canonical_sha256(status["artifact_ids"]), canonical_sha256(labels))

    def source_preset(self, target, index):
        if type(index) is not int or not 0 <= index < len(self.entries):
            raise ValueError("corpus source index is outside its frozen population")
        entry = self.entries[index]
        if target != entry.target:
            raise ValueError("corpus target differs from its frozen source entry")
        source = self.corpus_root
        for part in relative_source(entry.relative_path).parts:
            source = source / part
            if source.is_symlink() or source.is_junction():
                raise ValueError("corpus source cannot traverse links")
        path = source.resolve(strict=True)
        if not path.is_relative_to(self.corpus_root):
            raise ValueError("corpus source escapes its declared root")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry.raw_sha256:
            raise ValueError("original corpus source bytes changed")
        if PresetIdentity.from_bytes(raw).state != entry.state_id or self.codec.encode(raw) != target:
            raise ValueError("corpus target does not match its original source")
        return raw

    def save(self, path):
        path = Path(path)
        if path.resolve().is_relative_to(self.corpus_root):
            raise ValueError("corpus pool output must be outside the source corpus")
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, {"pool_id": self.pool_id, "pool": self.to_dict(), "corpus_root": str(self.corpus_root)})

    @classmethod
    def from_saved(cls, saved, *, corpus_root=None):
        value = saved["pool"]
        if saved["pool_id"] != canonical_sha256(value) or value.get("version") != 2 or value.get("generator") != GENERATOR:
            raise ValueError("corpus pool artifact changed")
        if value["source_policy"] != source_policy():
            raise ValueError("corpus source policy changed; create an explicit new pool")
        codec = load_codec(value["codec"])
        codec.contract.require_match(value["contract"])
        entries = tuple(CorpusEntry(row["state_id"], PilotTarget.from_dict(row["target"]), row["relative_path"], row["raw_sha256"])
                        for row in value["entries"])
        result = cls(codec, SplitManifest.from_dict(value["splits"]), entries, Path(corpus_root or saved["corpus_root"]),
                     value["source_contract"], value["source_snapshot_id"], value["candidates_sha256"])
        if result.to_dict() != value:
            raise ValueError("unsupported corpus pool fields or ordering")
        return result

    def summary(self):
        return {"pool_id": self.pool_id, "codec_id": self.codec.codec_id, "states": len(self.entries),
                "states_by_split": {"train": len(self.entries), "validation": 0, "test": 0},
                "frozen_manifest_states": len(self.manifest.assignments), "split_manifest_id": self.manifest.manifest_id,
                "audio_qa": "on_demand_not_pre_rendered", "source_policy": source_policy()}

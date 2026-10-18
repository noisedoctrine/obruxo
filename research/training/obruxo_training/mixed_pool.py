"""Original corpus and authored sources under one model, with frozen splits and alias checks."""
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
import hashlib
import json
from pathlib import Path

from obruxo_data.hashing import canonical_sha256
from obruxo_data.render.base import FULL_CURVE_RENDERER_ID
from obruxo_data.render.curve_inputs import prepare_curve_inputs
from obruxo_data.vital.full_contract import FullPresetContract

from .asset_codec import _content
from .audit import load_snapshot, write_json
from .codec import PilotTarget
from .corpus_inputs import prepare_corpus_labels
from .corpus_pool import CorpusEntry, CorpusPool, relative_source
from .identity import DISPLAY_FIELDS, PresetIdentity, require_digest
from .mixed_manifest import MixedManifest
from .observation_codec import ObservationCodec
from .pilot import PilotPool

GENERATOR = "frozen_authored_and_original_corpus_v1"


def mixture_policy():
    return {"rule": GENERATOR, "source_sha256_lf": hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
            "splits": "unchanged_parent_ledgers", "final_test": "static_alias_check_only_no_scheduled_entries",
            "aliases": "before_lossy_encoding_reject_cross_split_group_same_split_lineages_for_evaluation",
            "canonicalization": "corpus_label_policy_native_curve_precision_registered_asset_metadata_curve_names_outer_display_fields",
            "audio": "original_documents_and_phases_no_label_decode", "source_locations": "private_relocatable_id_checked_references"}


def read_source(root, relative_path, raw_sha256, state_id):
    path = root
    for part in relative_source(relative_path).parts:
        path = path / part
        if path.is_symlink() or path.is_junction():
            raise ValueError("mixed source cannot traverse links")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError("mixed source escapes its original root")
    raw = resolved.read_bytes()
    if hashlib.sha256(raw).hexdigest() != require_digest(raw_sha256) or PresetIdentity.from_bytes(raw).state != state_id:
        raise ValueError("mixed source identity or original bytes changed")
    return raw


class ProgramIdentity:
    def __init__(self, codec):
        self.codec = codec
        self.known = {kind: {_content(kind, value): value for value in values}
                      for kind, values in (("wavetable", codec.base._waves), ("sample", codec.base._samples))}

    def __call__(self, raw):
        return self.identify(raw)[0]

    def identify(self, raw):
        document, changes = prepare_corpus_labels(json.loads(raw), self.codec.base, self.codec.schema)
        document, _ = prepare_curve_inputs(document)
        for name in DISPLAY_FIELDS:
            document.pop(name, None)
        settings = document["settings"]
        settings["wavetables"] = [self.known["wavetable"][_content("wavetable", value)] for value in settings["wavetables"]]
        settings["sample"] = self.known["sample"][_content("sample", settings["sample"])]
        shapes = settings["lfos"] + settings["custom_warps"] + [route["line_mapping"] for route in settings["modulations"] if "line_mapping" in route]
        # Registered assets are shared immutable references; curve dictionaries belong to this copy.
        for shape in shapes:
            shape.pop("name", None)
        return canonical_sha256(document), changes


def evaluation_groups(programs, manifest):
    representatives = {entry.lineage_id: entry.lineage_id for entry in manifest.assignments}

    def find(value):
        while representatives[value] != value:
            representatives[value] = representatives[representatives[value]]
            value = representatives[value]
        return value

    for program, members in programs:
        require_digest(program)
        entries = [manifest.by_state[state] for _, state in members]
        if len({entry.split for entry in entries}) != 1:
            raise ValueError("canonical program alias crosses frozen splits")
        for entry in entries[1:]:
            left, right = find(entries[0].lineage_id), find(entry.lineage_id)
            representatives[max(left, right)] = min(left, right)
    return {lineage: find(lineage) for lineage in representatives}


@dataclass(frozen=True)
class MixedEntry:
    state_id: str
    target: PilotTarget
    source: str
    source_index: int
    raw_sha256: str
    program_id: str

    def to_dict(self):
        return {"state_id": self.state_id, "source": self.source, "source_index": self.source_index,
                "raw_sha256": self.raw_sha256, "program_id": self.program_id}


@dataclass(frozen=True)
class MixedPool:
    corpus: CorpusPool
    authored: PilotPool
    validation: tuple
    entries: tuple
    programs: tuple
    source_paths: dict

    def __post_init__(self):
        if not isinstance(self.corpus, CorpusPool) or not isinstance(self.authored, PilotPool) or not self.entries:
            raise ValueError("mixed pool requires original corpus and authored sources")
        if self.corpus.codec.base.bank != self.authored.codec.base.bank or self.corpus.codec.lfo_bank.bank_id != self.authored.codec.lfo_bank.bank_id:
            raise ValueError("mixed sources require identical registered asset and LFO banks")
        if len({entry.state_id for entry in self.validation}) != len(self.validation):
            raise ValueError("mixed validation sources are duplicated")
        states = set()
        expected = {entry.state_id for entry in self.corpus.entries} | {entry.state_id for entry in self.validation}
        expected |= {entry.state_id for entry in self.authored.entries if self.authored.manifest.by_state[entry.state_id].split != "test"}
        for entry in self.validation:
            if self.corpus.manifest.by_state[entry.state_id].split != "validation":
                raise ValueError("mixed corpus evaluation entries must be frozen validation states")
            relative_source(entry.relative_path)
            require_digest(entry.raw_sha256)
        for entry in self.entries:
            if entry.state_id in states or self.manifest.by_state[entry.state_id].split == "test":
                raise ValueError("mixed scheduled sources are duplicated or include final-test states")
            source = self._source_entry(entry.source, entry.source_index)
            if source.state_id != entry.state_id or self.codec.canonical_target(entry.target) != entry.target or entry.target != source.target:
                raise ValueError("mixed entry disagrees with its frozen source target")
            if entry.source != "authored" and entry.raw_sha256 != source.raw_sha256:
                raise ValueError("mixed entry changed its original corpus byte hash")
            require_digest(entry.raw_sha256)
            require_digest(entry.program_id)
            if self._program_by_state.get(entry.state_id) != (entry.source, entry.program_id):
                raise ValueError("mixed entry disagrees with its declared program alias census")
            states.add(entry.state_id)
        if states != expected:
            raise ValueError("mixed pool must preserve every scheduled source")
        if {source for source in self.source_paths} != {"corpus", "authored"}:
            raise ValueError("mixed source references are incomplete")
        self._groups

    @cached_property
    def codec(self):
        return ObservationCodec(self.corpus.codec.base.bank, self.corpus.codec.lfo_bank.to_dict())

    @cached_property
    def manifest(self):
        return MixedManifest((("authored", self.authored.manifest), ("corpus", self.corpus.manifest)))

    @cached_property
    def _identity(self):
        return ProgramIdentity(self.codec)

    @cached_property
    def _groups(self):
        return evaluation_groups(self.programs, self.manifest)

    @cached_property
    def _program_by_state(self):
        result = {}
        for program, members in self.programs:
            for source, state in members:
                if source not in {"authored", "corpus", "corpus_validation"} or state in result:
                    raise ValueError("mixed program census contains duplicate or invalid source members")
                parent = self.authored.manifest if source == "authored" else self.corpus.manifest
                if state not in parent.by_state:
                    raise ValueError("mixed program census disagrees with a frozen source ledger")
                if source != "authored" and (parent.by_state[state].split == "validation") != (source == "corpus_validation"):
                    raise ValueError("mixed program census has an invalid validation source")
                result[state] = (source, program)
        if {state for state, (source, _) in result.items() if source == "authored"} != set(self.authored.manifest.by_state):
            raise ValueError("mixed program census omits authored held-out sources")
        return result

    @cached_property
    def pool_id(self):
        return canonical_sha256(self.to_dict())

    def _source_entry(self, source, index):
        if source not in {"authored", "corpus", "corpus_validation"} or type(index) is not int or index < 0:
            raise ValueError("invalid mixed source reference")
        values = self.authored.entries if source == "authored" else self.corpus.entries if source == "corpus" else self.validation
        if index >= len(values):
            raise ValueError("mixed source index is outside its frozen pool")
        return values[index]

    def source_preset(self, target, index):
        if type(index) is not int or not 0 <= index < len(self.entries):
            raise ValueError("mixed source index is outside its frozen population")
        entry = self.entries[index]
        if target != entry.target:
            raise ValueError("mixed target differs from its original source")
        source = self._source_entry(entry.source, entry.source_index)
        raw = (self.authored.source_preset(source.target, entry.source_index) if entry.source == "authored" else
               read_source(self.corpus.corpus_root, source.relative_path, entry.raw_sha256, entry.state_id))
        if PresetIdentity.from_bytes(raw).state != entry.state_id or hashlib.sha256(raw).hexdigest() != entry.raw_sha256:
            raise ValueError("mixed source identity or original bytes changed")
        if self.codec.encode(raw) != target or self._identity(raw) != entry.program_id:
            raise ValueError("mixed target or program identity does not match original source")
        return raw

    def sampling_group(self, entry):
        return "authored" if entry.source == "authored" else "corpus"

    def evaluation_group(self, state):
        return self._groups[self.manifest.by_state[state].lineage_id]

    def to_dict(self):
        return {"version": 3, "generator": GENERATOR, "policy": mixture_policy(), "codec": self.codec.to_dict(),
                "contract": self.codec.contract.to_dict(), "sources": {"corpus": self.corpus.pool_id, "authored": self.authored.pool_id},
                "splits": self.manifest.to_dict(), "validation": [entry.to_dict() for entry in self.validation],
                "entries": [entry.to_dict() for entry in self.entries],
                "programs": [{"program_id": program, "members": [list(member) for member in members]} for program, members in self.programs]}

    @classmethod
    def create(cls, corpus_path, authored_path, snapshot, *, progress=None):
        if progress:
            progress({"stage": "load_source_metadata"})
        corpus, authored = PilotPool.load(Path(corpus_path)), PilotPool.load(Path(authored_path), verify_sources=False)
        if not isinstance(corpus, CorpusPool) or not isinstance(authored, PilotPool):
            raise ValueError("mixture requires original corpus and authored pool artifacts")
        codec = ObservationCodec(corpus.codec.base.bank, corpus.codec.lfo_bank.to_dict())
        from .curve_codec import CurveInputCodec
        from .graph_codec import GraphCoverageCodec
        authored_encoder = authored.codec.parent if isinstance(authored.codec, GraphCoverageCodec) else authored.codec
        if type(authored_encoder) is not CurveInputCodec or type(codec.parent.parent) is not CurveInputCodec or codec.parent.codec_id != corpus.codec.parent.codec_id:
            raise ValueError("mixed label caches require the reviewed common curve encoder")
        source_schema = FullPresetContract(renderer_id=FULL_CURVE_RENDERER_ID)
        contract, catalog, manifest = load_snapshot(Path(snapshot), source_schema)
        status = json.loads((Path(snapshot) / "status.json").read_text())
        if manifest != corpus.manifest or catalog.catalog_id != codec.catalog.catalog_id or contract.to_dict() != corpus.source_contract or (
                canonical_sha256(status["artifact_ids"]) != corpus.source_snapshot_id):
            raise ValueError("mixed snapshot differs from the frozen corpus source pool")
        if codec.continuous != authored.codec.continuous or len(codec.categorical) != len(authored.codec.categorical):
            raise ValueError("authored outputs are incompatible with the corpus model")
        for (name, values), (old_name, old_values) in zip(codec.categorical, authored.codec.categorical, strict=True):
            if name != old_name or values[:len(old_values)] != old_values:
                raise ValueError("authored categorical domain changed its meaning")
        records = json.loads((Path(snapshot) / "members.json").read_text())["records"]
        members = {}
        for row in records:
            if row["static_eligible"]:
                members.setdefault(row["identity"]["state"], row)
        identity, validation, entries, groups = ProgramIdentity(codec), [], [], defaultdict(list)
        corpus_indices = {entry.state_id: index for index, entry in enumerate(corpus.entries)}
        total = len(members) + len(authored.entries)

        def record(source, state, raw, target, index, program=None):
            program = identity(raw) if program is None else program
            groups[program].append((source, state))
            if manifest_union.by_state[state].split != "test":
                entries.append(MixedEntry(state, target, source, index, hashlib.sha256(raw).hexdigest(), program))

        manifest_union = MixedManifest((("authored", authored.manifest), ("corpus", corpus.manifest)))
        checked = 0
        if progress:
            progress({"stage": "check_source_programs_and_aliases", "total": total})
        for state, row in sorted(members.items()):
            raw = read_source(corpus.corpus_root, row["relative_path"], row["identity"]["raw"], state)
            split = manifest.by_state[state].split
            if split == "validation":
                target = codec.encode(raw)
                index = len(validation)
                validation.append(CorpusEntry(state, target, row["relative_path"], row["identity"]["raw"]))
            else:
                index = corpus_indices.get(state, -1)
                if split == "train" and index < 0:
                    raise ValueError("mixed corpus training member is missing from the original pool")
                target = corpus.entries[index].target if split == "train" else None
            record("corpus_validation" if split == "validation" else "corpus", state, raw, target, index)
            checked += 1
            if progress and checked % 250 == 0:
                progress({"checked": checked, "total": total})
        for index, entry in enumerate(authored.entries):
            raw = authored.source_preset(entry.target, index)
            if PresetIdentity.from_bytes(raw).state != entry.state_id:
                raise ValueError("authored source state differs from its frozen entry")
            program, changes = identity.identify(raw)
            if changes and authored.manifest.by_state[entry.state_id].split != "test" and codec.encode(raw) != entry.target:
                raise ValueError("mixed authored target differs under the full corpus codec")
            record("authored", entry.state_id, raw, entry.target, index, program)
            checked += 1
            if progress and (checked % 250 == 0 or checked == total):
                progress({"checked": checked, "total": total})
        programs = tuple((program, tuple(sorted(values))) for program, values in sorted(groups.items()))
        return cls(corpus, authored, tuple(validation), tuple(sorted(entries, key=lambda entry: entry.state_id)), programs,
                   {"corpus": str(Path(corpus_path).resolve()), "authored": str(Path(authored_path).resolve())})

    def save(self, path):
        path = Path(path)
        if path.exists() or path.resolve().is_relative_to(self.corpus.corpus_root):
            raise ValueError("mixed pool requires a new output outside the corpus")
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, {"pool_id": self.pool_id, "pool": self.to_dict(), "source_paths": self.source_paths,
                          "corpus_root": str(self.corpus.corpus_root)})

    @classmethod
    def from_saved(cls, saved, *, corpus_root=None):
        value = saved["pool"]
        if saved["pool_id"] != canonical_sha256(value) or value.get("version") != 3 or value.get("generator") != GENERATOR or value["policy"] != mixture_policy():
            raise ValueError("mixed pool artifact or source policy changed")
        paths = saved["source_paths"]
        corpus = PilotPool.load(Path(paths["corpus"]), corpus_root=corpus_root or saved["corpus_root"])
        authored = PilotPool.load(Path(paths["authored"]), verify_sources=False)
        validation = tuple(CorpusEntry(row["state_id"], PilotTarget.from_dict(row["target"]), row["relative_path"], row["raw_sha256"])
                           for row in value["validation"])
        entries = []
        for row in value["entries"]:
            if row["source"] not in {"authored", "corpus", "corpus_validation"}:
                raise ValueError("mixed source reference changed")
            values = authored.entries if row["source"] == "authored" else corpus.entries if row["source"] == "corpus" else validation
            if type(row["source_index"]) is not int or not 0 <= row["source_index"] < len(values):
                raise ValueError("mixed source reference changed")
            entries.append(MixedEntry(target=values[row["source_index"]].target, **row))
        programs = tuple((row["program_id"], tuple(tuple(member) for member in row["members"])) for row in value["programs"])
        result = cls(corpus, authored, validation, tuple(entries), programs, dict(paths))
        if result.to_dict() != value:
            raise ValueError("mixed sources, codec, membership or frozen split ledgers changed")
        return result

    def summary(self):
        return {"pool_id": self.pool_id, "codec_id": self.codec.codec_id, "states": len(self.entries),
                "states_by_split": {split: sum(self.manifest.by_state[entry.state_id].split == split for entry in self.entries) for split in ("train", "validation", "test")},
                "sources_by_split": {split: {source: sum(self.manifest.by_state[entry.state_id].split == split and self.sampling_group(entry) == source
                    for entry in self.entries) for source in ("authored", "corpus")} for split in ("train", "validation")},
                "frozen_manifest_states": len(self.manifest.assignments), "split_manifest_id": self.manifest.manifest_id,
                "canonical_programs": len(self.programs), "alias_groups": sum(len(members) > 1 for _, members in self.programs),
                "audio_qa": "on_demand_not_pre_rendered", "final_test_entries": 0}

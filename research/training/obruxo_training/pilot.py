from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random

from obruxo_data.hashing import canonical_sha256

from .audit import write_json
from .codec import CATEGORICAL, CONTINUOUS, PilotCodec, PilotTarget
from .contracts import RenderRecipe
from .eligibility import inspect_preset
from .identity import PresetIdentity
from .performances import initial_bank
from .splits import SplitLedger, SplitManifest, build_splits


def seeded_random(*parts) -> random.Random:
    return random.Random(int(hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest(), 16))


@dataclass(frozen=True)
class PilotEntry:
    state_id: str
    target: PilotTarget


@dataclass(frozen=True)
class PilotPool:
    codec: PilotCodec
    seed: int
    manifest: SplitManifest
    entries: tuple[PilotEntry, ...]

    @property
    def pool_id(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict:
        return {"version": 1, "generator": "independent_authored_pilot_v1", "seed": self.seed,
                "codec": self.codec.to_dict(), "contract": self.codec.contract.to_dict(), "splits": self.manifest.to_dict(),
                "entries": [{"state_id": entry.state_id, "target": entry.target.to_dict()} for entry in self.entries]}

    @classmethod
    def create(cls, count: int = 256, seed: int = 0) -> PilotPool:
        if type(count) is not int or not 1 <= count <= 4096 or type(seed) is not int:
            raise ValueError("pilot needs an integer seed and 1..4096 independent states")
        codec, entries = PilotCodec(), []
        for index in range(count):
            rng = seeded_random("independent_authored_pilot_v1", seed, index)
            target = codec.canonical_target(PilotTarget(tuple(rng.random() for _ in CONTINUOUS),
                                                       tuple(rng.randrange(len(options)) for _, options in CATEGORICAL)))
            raw = codec.decode(target)
            inspection = inspect_preset(raw, codec.schema, codec.catalog)
            if not inspection["static_eligible"]:
                raise ValueError(f"generated pilot state is ineligible: {inspection['reasons']}")
            entries.append(PilotEntry(inspection["identity"]["state"], target))
        manifest = build_splits((entry.state_id for entry in entries), seed=seed)
        if len(manifest.assignments) != count:
            raise ValueError("pilot generator produced duplicate states")
        return cls(codec, seed, manifest, tuple(entries))

    def save(self, path: Path) -> None:
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, {"pool_id": self.pool_id, "pool": self.to_dict()})

    def expand(self, count: int) -> PilotPool:
        if type(count) is not int or not len(self.entries) < count <= 4096:
            raise ValueError("expanded pilot count must exceed the parent and remain at most 4096")
        expanded = self.create(count, self.seed)
        expanded.expansion_from(self)
        return expanded

    def expansion_from(self, parent: PilotPool) -> dict:
        """Prove a strict authored prefix extension without moving any existing split."""
        if (self.seed != parent.seed or self.codec.to_dict() != parent.codec.to_dict()
                or self.codec.contract.to_dict() != parent.codec.contract.to_dict()):
            raise ValueError("pool expansion requires the same seed, codec and render contract")
        size = len(parent.entries)
        if len(self.entries) <= size or self.entries[:size] != parent.entries:
            raise ValueError("pool expansion must preserve the complete ordered parent population")
        if any(self.manifest.by_state.get(entry.state_id) != entry for entry in parent.manifest.assignments):
            raise ValueError("pool expansion moved or removed a frozen lineage or split")
        if (len(self.manifest.assignments) != len(self.entries)
                or len({entry.state_id for entry in self.entries}) != len(self.entries)
                or self.manifest.manifest_id != build_splits((entry.state_id for entry in self.entries), seed=self.seed).manifest_id):
            raise ValueError("pool expansion has an invalid independent-root population or split manifest")
        added = self.entries[size:]
        return {"policy": "independent_authored_prefix_expansion_v1", "parent_pool_id": parent.pool_id,
                "expanded_pool_id": self.pool_id, "parent_manifest_id": parent.manifest.manifest_id,
                "expanded_manifest_id": self.manifest.manifest_id, "preserved_states": size,
                "added_states": len(added), "preserved_states_by_split": parent.summary()["states_by_split"],
                "added_states_by_split": {split: sum(self.manifest.by_state[entry.state_id].split == split for entry in added)
                                          for split in ("train", "validation", "test")},
                "evaluation_policy": "fixed_parent_validation_recipes_and_parent_training_retrieval",
                "feature_reuse": "cross_pool_seed_and_resume_forbidden_start_fresh"}

    @classmethod
    def load(cls, path: Path) -> PilotPool:
        saved = json.loads(path.read_text(encoding="utf-8"))
        value = saved["pool"]
        if saved["pool_id"] != canonical_sha256(value):
            raise ValueError("pilot pool artifact changed")
        if value.get("version") != 1 or value.get("generator") != "independent_authored_pilot_v1" or type(value.get("seed")) is not int:
            raise ValueError("unsupported pilot generator")
        codec = PilotCodec()
        codec.contract.require_match(value["contract"])
        if value["codec"] != codec.to_dict():
            raise ValueError("pilot codec changed")
        manifest = SplitManifest.from_dict(value["splits"])
        entries = tuple(PilotEntry(entry["state_id"], PilotTarget.from_dict(entry["target"])) for entry in value["entries"])
        if not entries or len(entries) != len(manifest.assignments) or len({entry.state_id for entry in entries}) != len(entries):
            raise ValueError("pilot population is incomplete or duplicated")
        for entry in entries:
            raw = codec.decode(entry.target)
            if codec.canonical_target(entry.target) != entry.target or PresetIdentity.from_bytes(raw).state != entry.state_id:
                raise ValueError("pilot target does not match its source state")
        expected = build_splits((entry.state_id for entry in entries), seed=value["seed"])
        if expected.manifest_id != manifest.manifest_id:
            raise ValueError("pilot split manifest differs from its independent roots")
        return cls(codec, value["seed"], manifest, entries)

    def summary(self) -> dict:
        return {"pool_id": self.pool_id, "codec_id": self.codec.codec_id, "states": len(self.entries),
                "states_by_split": {split: sum(entry.split == split for entry in self.manifest.assignments)
                                    for split in ("train", "validation", "test")}, "audio_qa": "not_run"}


class PilotSampler:
    """Address recipes by sequence so unacknowledged prefetch can be regenerated after interruption."""

    def __init__(self, pool: PilotPool, *, split: str = "train", role: str = "reference", seed: int = 0, policy: str = "pairs"):
        if split not in {"train", "validation", "test"} or role not in {"reference", "transfer"} or type(seed) is not int:
            raise ValueError("invalid pilot sampling policy")
        if split == "train" and role != "reference":
            raise ValueError("transfer performances are reserved for held-out evaluation")
        if policy not in {"pairs", "state_round_robin"}:
            raise ValueError("unknown pilot sampling policy")
        self.pool, self.split, self.role, self.seed = pool, split, role, seed
        self.policy = policy
        self._entries = tuple(entry for entry in pool.entries if pool.manifest.by_state[entry.state_id].split == split)
        if not self._entries:
            raise ValueError("selected split has no pilot states")
        self._performances = tuple(entry for entry in initial_bank() if entry.role == role)
        self._ledger = SplitLedger(pool.manifest)
        self._epoch, self._order = None, []
        rule = "shuffled_pilot_pairs_v1" if policy == "pairs" else "shuffled_states_rotating_performances_v1"
        self.sampler_id = canonical_sha256({"sampler": rule, "pool_id": pool.pool_id,
                                           "split": split, "role": role, "seed": seed})

    @property
    def epoch_size(self) -> int:
        return len(self._entries) * (len(self._performances) if self.policy == "pairs" else 1)

    def recipe_at(self, sequence_id: int) -> RenderRecipe:
        if type(sequence_id) is not int or sequence_id < 0:
            raise ValueError("sequence ID must be a nonnegative integer")
        epoch, position = divmod(sequence_id, self.epoch_size)
        if self._epoch != epoch:
            if self.policy == "pairs":
                self._order = [(entry, performance) for entry in self._entries for performance in self._performances]
            else:
                self._order = [(entry, self._performances[(index + epoch) % len(self._performances)])
                               for index, entry in enumerate(self._entries)]
            seeded_random(self.sampler_id, epoch).shuffle(self._order)
            self._epoch = epoch
        entry, performance = self._order[position]
        codec = self.pool.codec
        return RenderRecipe.create(codec.decode(entry.target), performance.performance(), schema=codec.schema,
                                   catalog=codec.catalog, contract=codec.contract, ledger=self._ledger,
                                   purpose="train" if self.split == "train" else "evaluate", sequence_id=sequence_id)

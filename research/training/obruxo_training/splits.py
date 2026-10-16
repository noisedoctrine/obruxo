from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass, replace
from functools import cached_property
import hashlib
from types import MappingProxyType

from obruxo_data.hashing import canonical_sha256

from .identity import IDENTITY_RULE, require_digest


SPLIT_RULE = "lineage_hash_80_10_10_v1"


def normalize_parents(parents: dict[str, list[str]] | None) -> dict[str, list[str]]:
    if parents is None:
        return {}
    if not isinstance(parents, dict):
        raise ValueError("parents must be an object mapping child state IDs to parent lists")
    result = {}
    for child, ancestors in sorted(parents.items()):
        require_digest(child)
        if not isinstance(ancestors, list):
            raise ValueError("each lineage entry must contain a parent list")
        result[child] = sorted({require_digest(parent) for parent in ancestors})
    return result


def split_for(group_id: str, seed: int) -> str:
    require_digest(group_id)
    if type(seed) is not int:
        raise ValueError("split seed must be an integer")
    bucket = int(hashlib.sha256(f"{SPLIT_RULE}:{seed}:{group_id}".encode()).hexdigest()[:16], 16) * 10000 // 2**64
    return "train" if bucket < 8000 else "validation" if bucket < 9000 else "test"


@dataclass(frozen=True)
class Assignment:
    state_id: str
    lineage_id: str
    split: str


@dataclass(frozen=True)
class SplitManifest:
    seed: int
    assignments: tuple[Assignment, ...]

    def __post_init__(self):
        groups = defaultdict(list)
        states = set()
        for entry in self.assignments:
            require_digest(entry.state_id)
            require_digest(entry.lineage_id)
            if entry.state_id in states or entry.split != split_for(entry.lineage_id, self.seed):
                raise ValueError("duplicate state or invalid split assignment")
            states.add(entry.state_id)
            groups[entry.lineage_id].append(entry.state_id)
        if type(self.seed) is not int or any(group != min(members) for group, members in groups.items()):
            raise ValueError("invalid seed or lineage representative")
        object.__setattr__(self, "assignments", tuple(sorted(self.assignments, key=lambda entry: entry.state_id)))

    @cached_property
    def by_state(self):
        return MappingProxyType({entry.state_id: entry for entry in self.assignments})

    @property
    def manifest_id(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict:
        return {"version": 1, "identity_rule": IDENTITY_RULE, "split_rule": SPLIT_RULE, "seed": self.seed,
                "assignments": [asdict(entry) for entry in sorted(self.assignments, key=lambda entry: entry.state_id)]}

    @classmethod
    def from_dict(cls, value: dict) -> SplitManifest:
        if value.get("version") == 2:
            return FrozenVariantManifest.from_dict(value)
        if (value.get("version"), value.get("identity_rule"), value.get("split_rule")) != (1, IDENTITY_RULE, SPLIT_RULE):
            raise ValueError("unsupported split contract")
        return cls(value["seed"], tuple(Assignment(**entry) for entry in value["assignments"]))


@dataclass(frozen=True)
class FrozenVariantManifest:
    """A complete variant population inheriting a separately verified frozen root split."""
    parent: SplitManifest
    variants: tuple[tuple[str, str], ...]

    def __post_init__(self):
        if not isinstance(self.parent, SplitManifest):
            raise ValueError("variant population requires an original root manifest")
        parents, children = set(), set()
        for state, parent in self.variants:
            require_digest(state)
            assignment = self.parent.by_state.get(parent)
            if assignment is None or parent in parents or state in children:
                raise ValueError("variant population must map unique children to every frozen parent")
            collision = self.parent.by_state.get(state)
            if collision is not None and collision != replace(assignment, state_id=state):
                raise ValueError("variant collides with another frozen lineage")
            parents.add(parent)
            children.add(state)
        if parents != set(self.parent.by_state):
            raise ValueError("variant population must preserve every parent including the test reserve")
        object.__setattr__(self, "variants", tuple(sorted(self.variants)))

    @property
    def seed(self):
        return self.parent.seed

    @cached_property
    def assignments(self):
        return tuple(replace(self.parent.by_state[parent], state_id=state) for state, parent in self.variants)

    @cached_property
    def by_state(self):
        return MappingProxyType({entry.state_id: entry for entry in self.assignments})

    @property
    def manifest_id(self):
        return canonical_sha256(self.to_dict())

    def to_dict(self):
        return {"version": 2, "identity_rule": IDENTITY_RULE, "split_rule": SPLIT_RULE,
                "policy": "complete_variants_of_frozen_parent_v1", "parent": self.parent.to_dict(),
                "variants": [{"state_id": state, "parent_state_id": parent} for state, parent in self.variants]}

    @classmethod
    def from_dict(cls, value):
        if (value.get("identity_rule"), value.get("split_rule"), value.get("policy")) != (IDENTITY_RULE, SPLIT_RULE, "complete_variants_of_frozen_parent_v1"):
            raise ValueError("unsupported variant split contract")
        parent = SplitManifest.from_dict(value["parent"])
        if not isinstance(parent, SplitManifest):
            raise ValueError("variant population requires an original root manifest")
        return cls(parent, tuple((entry["state_id"], entry["parent_state_id"]) for entry in value["variants"]))


def build_splits(states, *, parents: dict[str, list[str]] | None = None, seed: int = 0) -> SplitManifest:
    states = set(states)
    for state in states:
        require_digest(state)
    parents = normalize_parents(parents)
    if set(parents) - states or any(set(ancestors) - states for ancestors in parents.values()):
        raise ValueError("lineage refers to a state outside the audited population")
    children = defaultdict(list)
    indegree = {state: len(parents.get(state, ())) for state in states}
    representatives = {state: state for state in states}

    def find(state):
        while representatives[state] != state:
            representatives[state] = representatives[representatives[state]]
            state = representatives[state]
        return state

    for child, ancestors in parents.items():
        for parent in ancestors:
            children[parent].append(child)
            left, right = find(child), find(parent)
            representatives[max(left, right)] = min(left, right)
    ready = deque(state for state, degree in indegree.items() if degree == 0)
    visited = 0
    while ready:
        visited += 1
        for child in children[ready.popleft()]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if visited != len(states):
        raise ValueError("lineage contains a cycle")
    entries = tuple(Assignment(state, find(state), split_for(find(state), seed)) for state in sorted(states))
    return SplitManifest(seed, entries)


class SplitLedger:
    """Extend a frozen split with online mutations without rehashing or moving its groups."""

    def __init__(self, manifest: SplitManifest):
        self.manifest = manifest
        self._entries = dict(manifest.by_state)
        self._variants = []

    def lookup(self, state_id: str) -> Assignment:
        try:
            return self._entries[state_id]
        except KeyError as error:
            raise ValueError("state is not registered in this split ledger") from error

    def register_variant(self, state_id: str, parent_state_id: str) -> Assignment:
        require_digest(state_id)
        parent = self.lookup(parent_state_id)
        entry = replace(parent, state_id=state_id)
        existing = self._entries.get(state_id)
        if existing is not None:
            if existing != entry:
                raise ValueError("variant collides with a different frozen lineage or split")
            return existing
        self._entries[state_id] = entry
        self._variants.append({"state_id": state_id, "parent_state_id": parent_state_id})
        return entry

    def to_dict(self) -> dict:
        return {"version": 1, "manifest_id": self.manifest.manifest_id, "variants": [dict(entry) for entry in self._variants]}

    @classmethod
    def from_dict(cls, value: dict, manifest: SplitManifest) -> SplitLedger:
        if value.get("version") != 1 or value.get("manifest_id") != manifest.manifest_id:
            raise ValueError("split ledger belongs to a different frozen manifest")
        result = cls(manifest)
        for entry in value["variants"]:
            result.register_variant(**entry)
        return result

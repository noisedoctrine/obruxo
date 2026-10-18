"""Union of frozen source ledgers without recomputing any split or lineage."""
from dataclasses import dataclass
from functools import cached_property
from types import MappingProxyType

from obruxo_data.hashing import canonical_sha256

from .identity import IDENTITY_RULE
from .splits import SPLIT_RULE, SplitManifest


@dataclass(frozen=True)
class MixedManifest:
    parents: tuple

    def __post_init__(self):
        if len(self.parents) != 2 or {name for name, _ in self.parents} != {"authored", "corpus"}:
            raise ValueError("mixed manifest requires both frozen source ledgers")
        if len({parent.seed for _, parent in self.parents}) != 1:
            raise ValueError("mixed source ledgers must retain the same frozen split seed")
        object.__setattr__(self, "parents", tuple(sorted(self.parents)))
        self.assignments

    @property
    def seed(self):
        return self.parents[0][1].seed

    @cached_property
    def assignments(self):
        entries = {}
        for _, parent in self.parents:
            for entry in parent.assignments:
                if entry.state_id in entries and entries[entry.state_id] != entry:
                    raise ValueError("source ledgers disagree on a frozen state lineage or split")
                entries[entry.state_id] = entry
        return tuple(entries[state] for state in sorted(entries))

    @cached_property
    def by_state(self):
        return MappingProxyType({entry.state_id: entry for entry in self.assignments})

    @cached_property
    def manifest_id(self):
        return canonical_sha256(self.to_dict())

    def to_dict(self):
        return {"version": 4, "identity_rule": IDENTITY_RULE, "split_rule": SPLIT_RULE, "policy": "frozen_source_union_v1",
                "parents": {name: parent.to_dict() for name, parent in self.parents}}

    @classmethod
    def from_dict(cls, value):
        if (value.get("version"), value.get("identity_rule"), value.get("split_rule"), value.get("policy")) != (
                4, IDENTITY_RULE, SPLIT_RULE, "frozen_source_union_v1"):
            raise ValueError("unsupported mixed split contract")
        result = cls(tuple((name, SplitManifest.from_dict(parent)) for name, parent in value["parents"].items()))
        if result.to_dict() != value:
            raise ValueError("mixed split contract has unsupported fields")
        return result

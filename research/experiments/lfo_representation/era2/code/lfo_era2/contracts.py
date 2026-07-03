"""Era 2 contract checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TopologyFlags:
    topology_used_in_construction: bool = False
    topology_used_at_runtime: bool = False
    topology_used_in_targets: bool = False
    topology_used_in_loss: bool = False
    topology_used_in_decoder_lookup: bool = False
    topology_used_in_head_accounting: bool = False

    def as_dict(self) -> dict[str, bool]:
        return {
            "topology_used_in_construction": self.topology_used_in_construction,
            "topology_used_at_runtime": self.topology_used_at_runtime,
            "topology_used_in_targets": self.topology_used_in_targets,
            "topology_used_in_loss": self.topology_used_in_loss,
            "topology_used_in_decoder_lookup": self.topology_used_in_decoder_lookup,
            "topology_used_in_head_accounting": self.topology_used_in_head_accounting,
        }


@dataclass(frozen=True)
class ContractResult:
    passed: bool
    violations: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "violations": list(self.violations)}


def validate_topology_contract(flags: TopologyFlags) -> ContractResult:
    allowed_true = {"topology_used_in_construction"}
    violations = [
        key
        for key, value in flags.as_dict().items()
        if value and key not in allowed_true
    ]
    return ContractResult(passed=not violations, violations=violations)


def find_stage_keys(value: Any, *, prefix: str = "") -> list[str]:
    """Find public mapping keys that still use the old 'stage' vocabulary."""
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            if "stage" in key_text.lower():
                found.append(path)
            found.extend(find_stage_keys(child, prefix=path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(find_stage_keys(child, prefix=f"{prefix}[{index}]"))
    return found


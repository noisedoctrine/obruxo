"""Manifest helpers for Era 2 rows."""

from __future__ import annotations

from dataclasses import dataclass, field
import csv
import json
from pathlib import Path
from typing import Any

from .accounting import BudgetBreakdown
from .contracts import TopologyFlags


REQUIRED_MANIFEST_FIELDS = (
    "experiment_id",
    "oracle_construction_id",
    "runtime_interface_id",
    "decoder_policy_id",
    "base_dictionary_size",
    "D",
    "scalar_families",
    "scalar_outputs",
    "categorical_outputs",
    "continuous_outputs",
    "head_outputs_formula",
    "head_outputs_actual",
    "dictionary_scope",
    "codebook_storage_count",
    "oracle_construction_time",
    "oracle_encoding_time",
    "topology_used_in_construction",
    "topology_used_at_runtime",
    "topology_used_in_targets",
    "topology_used_in_loss",
    "topology_used_in_decoder_lookup",
    "topology_used_in_head_accounting",
)


@dataclass(frozen=True)
class ExperimentRowManifest:
    experiment_id: str
    oracle_construction_id: str
    runtime_interface_id: str
    decoder_policy_id: str
    base_dictionary_size: int
    residual_layer_count: int
    scalar_families: list[str]
    dictionary_scope: str
    codebook_storage_count: int
    budget: BudgetBreakdown
    topology_flags: TopologyFlags
    oracle_construction_time: float = 0.0
    oracle_encoding_time: float = 0.0
    method_parameters: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "oracle_construction_id": self.oracle_construction_id,
            "runtime_interface_id": self.runtime_interface_id,
            "decoder_policy_id": self.decoder_policy_id,
            "base_dictionary_size": int(self.base_dictionary_size),
            "D": int(self.residual_layer_count),
            "scalar_families": list(self.scalar_families),
            "scalar_outputs": int(self.budget.scalar_outputs),
            "categorical_outputs": int(self.budget.categorical_outputs),
            "continuous_outputs": int(self.budget.continuous_outputs),
            "residual_atom_selection_outputs": int(self.budget.residual_atom_selection_outputs),
            "head_outputs_formula": self.budget.head_outputs_formula,
            "head_outputs_actual": int(self.budget.head_outputs_actual),
            "dictionary_scope": self.dictionary_scope,
            "codebook_storage_count": int(self.codebook_storage_count),
            "oracle_construction_time": float(self.oracle_construction_time),
            "oracle_encoding_time": float(self.oracle_encoding_time),
            "addressing_scheme": self.budget.addressing_scheme,
            "notes": self.notes,
            **self.topology_flags.as_dict(),
            **self.method_parameters,
        }
        return payload

    def missing_required_fields(self) -> list[str]:
        payload = self.as_dict()
        return [field for field in REQUIRED_MANIFEST_FIELDS if field not in payload]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def write_summary_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = _jsonable(row)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(serializable.keys()))
        writer.writeheader()
        writer.writerow(serializable)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    if hasattr(value, "item"):
        return value.item()
    return value


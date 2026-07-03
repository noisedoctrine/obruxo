"""Core data structures for Era 2 reconstruction assets and targets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .curve import as_curve_matrix


@dataclass(frozen=True)
class DecoderPolicy:
    decoder_policy_id: str = "final_clip"
    final_clip: bool = True

    @property
    def model_head_outputs(self) -> int:
        return 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "decoder_policy_id": self.decoder_policy_id,
            "final_clip": self.final_clip,
            "decoder_policy_model_head_outputs": self.model_head_outputs,
        }


@dataclass
class ReconstructionAssets:
    base_dictionary: np.ndarray
    residual_layer_dictionaries: list[np.ndarray]
    dictionary_scope: str = "per_residual_layer"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.base_dictionary = as_curve_matrix(self.base_dictionary)
        self.residual_layer_dictionaries = [
            as_curve_matrix(layer) for layer in self.residual_layer_dictionaries
        ]
        resolution = self.base_dictionary.shape[1]
        for layer in self.residual_layer_dictionaries:
            if layer.shape[1] != resolution:
                raise ValueError("all dictionaries must share one resolution")

    @property
    def residual_layer_count(self) -> int:
        return len(self.residual_layer_dictionaries)

    @property
    def base_dictionary_size(self) -> int:
        return int(len(self.base_dictionary))

    @property
    def resolution(self) -> int:
        return int(self.base_dictionary.shape[1])

    @property
    def codebook_storage_count(self) -> int:
        return int(self.base_dictionary_size + sum(len(layer) for layer in self.residual_layer_dictionaries))

    def residual_widths(self) -> list[int]:
        return [int(len(layer)) for layer in self.residual_layer_dictionaries]

    def as_manifest_fields(self) -> dict[str, Any]:
        return {
            "base_dictionary_size": self.base_dictionary_size,
            "D": self.residual_layer_count,
            "dictionary_scope": self.dictionary_scope,
            "codebook_storage_count": self.codebook_storage_count,
            "W_by_residual_layer": self.residual_widths(),
            "resolution": self.resolution,
            **self.metadata,
        }


@dataclass
class OracleEncoding:
    base_index: np.ndarray
    base_phase: np.ndarray
    residual_layer_indices: list[np.ndarray]
    residual_layer_phases: list[np.ndarray]

    def __post_init__(self) -> None:
        self.base_index = np.asarray(self.base_index, dtype=np.int32)
        self.base_phase = np.asarray(self.base_phase, dtype=np.float32)
        self.residual_layer_indices = [
            np.asarray(values, dtype=np.int32) for values in self.residual_layer_indices
        ]
        self.residual_layer_phases = [
            np.asarray(values, dtype=np.float32) for values in self.residual_layer_phases
        ]
        if len(self.residual_layer_indices) != len(self.residual_layer_phases):
            raise ValueError("residual_layer_indices and residual_layer_phases must have equal length")
        row_count = len(self.base_index)
        if self.base_phase.shape != (row_count,):
            raise ValueError("base_phase must have one value per encoded row")
        for values in self.residual_layer_indices + self.residual_layer_phases:
            if values.shape != (row_count,):
                raise ValueError("each residual-layer target must have one value per encoded row")

    @property
    def row_count(self) -> int:
        return int(len(self.base_index))

    @property
    def residual_layer_count(self) -> int:
        return len(self.residual_layer_indices)

    def target_schema(self) -> dict[str, Any]:
        fields = [
            {"name": "base_index", "kind": "categorical"},
            {"name": "base_phase", "kind": "continuous"},
        ]
        for residual_layer in range(self.residual_layer_count):
            number = residual_layer + 1
            fields.append(
                {
                    "name": f"residual_layer_{number}_index",
                    "kind": "categorical",
                }
            )
            fields.append(
                {
                    "name": f"residual_layer_{number}_phase",
                    "kind": "continuous",
                }
            )
        return {
            "runtime_interface_id": "flat_categorical_per_residual_layer",
            "row_count": self.row_count,
            "fields": fields,
        }

    def as_arrays(self) -> dict[str, np.ndarray]:
        payload = {"base_index": self.base_index, "base_phase": self.base_phase}
        for residual_layer, values in enumerate(self.residual_layer_indices, start=1):
            payload[f"residual_layer_{residual_layer}_index"] = values
        for residual_layer, values in enumerate(self.residual_layer_phases, start=1):
            payload[f"residual_layer_{residual_layer}_phase"] = values
        return payload


"""Topology-free flat-categorical residual-layer interface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np

from .accelerator import BackendPreference, NearestResult, nearest_indices
from .accounting import RuntimeInterfaceSpec
from .assets import DecoderPolicy, OracleEncoding, ReconstructionAssets
from .contracts import TopologyFlags, find_stage_keys, validate_topology_contract
from .curve import (
    phase_shift_bank,
    synthetic_base_dictionary,
    synthetic_residual_dictionaries,
)
from .manifest import ExperimentRowManifest, write_json, write_summary_csv
from .metrics import flat_atom_usage, reconstruction_summary


@dataclass(frozen=True)
class FlatEncodingResult:
    encoding: OracleEncoding
    nearest_reports: list[NearestResult]

    @property
    def backend_used(self) -> list[str]:
        return sorted({report.backend_used for report in self.nearest_reports})


def construct_flat_assets_from_curves(
    targets: np.ndarray,
    *,
    base_dictionary: np.ndarray | None = None,
    base_dictionary_size: int = 32,
    residual_layer_count: int,
    width: int,
    backend: BackendPreference = "auto",
    chunk_size: int = 256,
) -> ReconstructionAssets:
    """Build a simple topology-free observed residual dictionary stack.

    This is intentionally basic. It gives Experiment 11 a clean construction
    path without importing Era 1's monolithic training code.
    """
    target_matrix = np.asarray(targets, dtype=np.float32)
    if target_matrix.ndim != 2:
        raise ValueError("targets must have shape [rows, resolution]")
    if residual_layer_count < 1:
        raise ValueError("residual_layer_count must be positive")
    if width < 1:
        raise ValueError("width must be positive")
    if base_dictionary is None:
        base_dictionary = _select_observed_atoms(
            target_matrix,
            width=base_dictionary_size,
            backend=backend,
            chunk_size=chunk_size,
            include_zero=False,
        )

    base_result = nearest_indices(target_matrix, base_dictionary, backend=backend, chunk_size=chunk_size)
    prefix = np.asarray(base_dictionary, dtype=np.float32)[base_result.indices].copy()
    residual_layers: list[np.ndarray] = []
    for _ in range(residual_layer_count):
        residual = target_matrix - prefix
        atoms = _select_observed_atoms(
            residual,
            width=width,
            backend=backend,
            chunk_size=chunk_size,
            include_zero=True,
        )
        residual_layers.append(atoms)
        layer_result = nearest_indices(residual, atoms, backend=backend, chunk_size=chunk_size)
        prefix = prefix + atoms[layer_result.indices]
    return ReconstructionAssets(
        base_dictionary=np.asarray(base_dictionary, dtype=np.float32),
        residual_layer_dictionaries=residual_layers,
        dictionary_scope="per_residual_layer",
        metadata={"construction_policy": "topology_blind_observed_farthest_residual"},
    )


def synthetic_flat_assets(
    *,
    residual_layer_count: int = 3,
    width: int = 4,
    base_dictionary_size: int = 32,
    resolution: int = 64,
) -> ReconstructionAssets:
    return ReconstructionAssets(
        base_dictionary=synthetic_base_dictionary(base_dictionary_size, resolution),
        residual_layer_dictionaries=synthetic_residual_dictionaries(residual_layer_count, width, resolution),
        dictionary_scope="per_residual_layer",
        metadata={"construction_policy": "synthetic_fixed_flat_assets"},
    )


def encode_flat(
    targets: np.ndarray,
    assets: ReconstructionAssets,
    *,
    phase_bins: int = 1,
    backend: BackendPreference = "auto",
    chunk_size: int = 256,
) -> FlatEncodingResult:
    target_matrix = np.asarray(targets, dtype=np.float32)
    if target_matrix.ndim != 2:
        raise ValueError("targets must have shape [rows, resolution]")
    nearest_reports: list[NearestResult] = []

    base_index, base_phase, base_values, report = _nearest_shifted(
        target_matrix,
        assets.base_dictionary,
        phase_bins=phase_bins,
        backend=backend,
        chunk_size=chunk_size,
    )
    nearest_reports.append(report)
    prefix = base_values.copy()

    residual_indices: list[np.ndarray] = []
    residual_phases: list[np.ndarray] = []
    for dictionary in assets.residual_layer_dictionaries:
        residual = target_matrix - prefix
        index, phase, values, report = _nearest_shifted(
            residual,
            dictionary,
            phase_bins=phase_bins,
            backend=backend,
            chunk_size=chunk_size,
        )
        nearest_reports.append(report)
        residual_indices.append(index)
        residual_phases.append(phase)
        prefix = prefix + values

    return FlatEncodingResult(
        encoding=OracleEncoding(
            base_index=base_index,
            base_phase=base_phase,
            residual_layer_indices=residual_indices,
            residual_layer_phases=residual_phases,
        ),
        nearest_reports=nearest_reports,
    )


def decode_flat(
    assets: ReconstructionAssets,
    encoding: OracleEncoding,
    *,
    decoder_policy: DecoderPolicy | None = None,
) -> np.ndarray:
    policy = decoder_policy or DecoderPolicy()
    rows = np.arange(encoding.row_count)
    reconstructed = assets.base_dictionary[encoding.base_index].copy()
    if np.any(encoding.base_phase != 0.0):
        from .curve import circular_shift

        reconstructed = circular_shift(reconstructed, encoding.base_phase)
    for residual_layer, indices in enumerate(encoding.residual_layer_indices):
        additions = assets.residual_layer_dictionaries[residual_layer][indices]
        phases = encoding.residual_layer_phases[residual_layer]
        if np.any(phases != 0.0):
            from .curve import circular_shift

            additions = circular_shift(additions, phases)
        reconstructed = reconstructed + additions
    if policy.final_clip:
        reconstructed = np.clip(reconstructed, 0.0, 1.0)
    return reconstructed.astype(np.float32)


def make_smoke_targets(assets: ReconstructionAssets, row_count: int = 12) -> tuple[np.ndarray, OracleEncoding]:
    base_index = np.zeros(row_count, dtype=np.int32)
    base_phase = np.zeros(row_count, dtype=np.float32)
    residual_indices = []
    residual_phases = []
    for residual_layer, dictionary in enumerate(assets.residual_layer_dictionaries):
        width = len(dictionary)
        residual_indices.append(((np.arange(row_count) + residual_layer) % width).astype(np.int32))
        residual_phases.append(np.zeros(row_count, dtype=np.float32))
    encoding = OracleEncoding(
        base_index=base_index,
        base_phase=base_phase,
        residual_layer_indices=residual_indices,
        residual_layer_phases=residual_phases,
    )
    targets = decode_flat(assets, encoding, decoder_policy=DecoderPolicy())
    return targets, encoding


def run_flat_smoke(
    output_dir: Path,
    *,
    residual_layer_count: int = 3,
    width: int = 4,
    base_dictionary_size: int = 32,
    resolution: int = 64,
    phase_bins: int = 1,
    backend: BackendPreference = "auto",
) -> dict[str, Any]:
    start_construction = time.perf_counter()
    assets = synthetic_flat_assets(
        residual_layer_count=residual_layer_count,
        width=width,
        base_dictionary_size=base_dictionary_size,
        resolution=resolution,
    )
    targets, _ = make_smoke_targets(assets)
    construction_time = time.perf_counter() - start_construction

    start_encoding = time.perf_counter()
    encoded = encode_flat(targets, assets, phase_bins=phase_bins, backend=backend)
    encoding_time = time.perf_counter() - start_encoding
    reconstructed = decode_flat(assets, encoded.encoding, decoder_policy=DecoderPolicy())

    runtime_spec = RuntimeInterfaceSpec(
        addressing_scheme="flat_categorical",
        residual_layer_count=residual_layer_count,
        dictionary_scope="per_residual_layer",
        parameters={"width": width},
    )
    budget = runtime_spec.budget(base_dictionary_size=base_dictionary_size)
    flags = TopologyFlags()
    contract = validate_topology_contract(flags)
    schema = encoded.encoding.target_schema()
    schema_stage_keys = find_stage_keys(schema)
    manifest = ExperimentRowManifest(
        experiment_id="smoke_flat",
        oracle_construction_id="synthetic_fixed_flat_assets",
        runtime_interface_id="flat_categorical_per_residual_layer",
        decoder_policy_id="final_clip",
        base_dictionary_size=base_dictionary_size,
        residual_layer_count=residual_layer_count,
        scalar_families=["phase"],
        dictionary_scope=assets.dictionary_scope,
        codebook_storage_count=assets.codebook_storage_count,
        budget=budget,
        topology_flags=flags,
        oracle_construction_time=construction_time,
        oracle_encoding_time=encoding_time,
        method_parameters={
            "W_by_residual_layer": assets.residual_widths(),
            "phase_bins": phase_bins,
            "backend_preference": backend,
            "backend_used": encoded.backend_used,
            "schema_stage_key_violations": schema_stage_keys,
        },
        notes="Tiny deterministic framework smoke run; not an Experiment 11 result.",
    )

    metrics = reconstruction_summary(targets, reconstructed)
    usage = flat_atom_usage(
        encoded.encoding.as_arrays(),
        residual_layer_count=residual_layer_count,
        widths_by_residual_layer=assets.residual_widths(),
    )
    summary = {
        **manifest.as_dict(),
        **metrics,
        **usage,
        "topology_contract_pass": contract.passed,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "manifest.json", manifest.as_dict())
    write_json(output_dir / "targets_schema.json", schema)
    write_json(output_dir / "topology_contract.json", contract.as_dict())
    write_summary_csv(output_dir / "summary.csv", summary)
    return {
        "output_dir": str(output_dir),
        "manifest": manifest.as_dict(),
        "summary": summary,
        "target_schema": schema,
        "topology_contract": contract.as_dict(),
    }


def _nearest_shifted(
    targets: np.ndarray,
    codes: np.ndarray,
    *,
    phase_bins: int,
    backend: BackendPreference,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, NearestResult]:
    bank, phases = phase_shift_bank(codes, phase_bins)
    flat_bank = bank.reshape(bank.shape[0] * bank.shape[1], bank.shape[2])
    result = nearest_indices(targets, flat_bank, backend=backend, chunk_size=chunk_size)
    code_index = (result.indices // phase_bins).astype(np.int32)
    phase_index = (result.indices % phase_bins).astype(np.int32)
    selected = flat_bank[result.indices]
    return code_index, phases[phase_index].astype(np.float32), selected.astype(np.float32), result


def _select_observed_atoms(
    residuals: np.ndarray,
    *,
    width: int,
    backend: BackendPreference,
    chunk_size: int,
    include_zero: bool = True,
) -> np.ndarray:
    matrix = np.asarray(residuals, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("residuals must have shape [rows, resolution]")
    atoms = []
    selected: set[int] = set()
    if include_zero:
        atoms.append(np.zeros(matrix.shape[1], dtype=np.float32))
    elif len(matrix):
        center = np.mean(matrix, axis=0, dtype=np.float32)
        distances = np.mean((matrix - center[None, :]) ** 2, axis=1)
        first = int(np.argmin(distances))
        atoms.append(matrix[first].astype(np.float32))
        selected.add(first)
    else:
        atoms.append(np.zeros(matrix.shape[1], dtype=np.float32))
    while len(atoms) < width:
        current = np.stack(atoms).astype(np.float32)
        result = nearest_indices(matrix, current, backend=backend, chunk_size=chunk_size)
        order = np.argsort(result.losses)[::-1]
        candidate = next((int(index) for index in order if int(index) not in selected), None)
        if candidate is None:
            atoms.append(np.zeros(matrix.shape[1], dtype=np.float32))
        else:
            selected.add(candidate)
            atoms.append(matrix[candidate].astype(np.float32))
    return np.stack(atoms).astype(np.float32)

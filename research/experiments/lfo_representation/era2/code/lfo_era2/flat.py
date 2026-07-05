"""Topology-free flat-categorical residual-layer interface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np

from .accelerator import BackendPreference, NearestResult, nearest_indices
from .accounting import RuntimeInterfaceSpec
from .assets import DecoderPolicy, OracleEncoding, ReconstructionAssets
from .contracts import TopologyFlags, find_stage_keys, validate_topology_contract
from .curve import (
    as_curve_matrix,
    circular_shift,
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


@dataclass(frozen=True)
class PhaseSearchSpec:
    """Oracle-only phase search policy for continuous phase targets."""

    policy: str = "fft_lattice"
    candidate_count: int | None = None

    def resolved_candidate_count(self, resolution: int) -> int:
        if self.policy == "disabled":
            return 1
        if self.policy == "fft_lattice":
            return int(resolution)
        if self.policy == "grid":
            return int(self.candidate_count or resolution)
        raise ValueError(f"unsupported oracle phase search policy: {self.policy}")

    def as_manifest_fields(self, resolution: int) -> dict[str, Any]:
        return {
            "oracle_phase_search_policy": self.policy,
            "oracle_phase_candidate_count": self.resolved_candidate_count(resolution),
            "phase_target_kind": "continuous_scalar",
            "phase_search_cost_note": "Oracle phase search resolution is not part of model prediction head budget.",
        }


def construct_flat_assets_from_curves(
    targets: np.ndarray,
    *,
    base_dictionary: np.ndarray | None = None,
    base_dictionary_size: int = 32,
    residual_layer_count: int,
    width: int,
    backend: BackendPreference = "auto",
    chunk_size: int = 256,
    phase_search: PhaseSearchSpec | None = None,
    progress: Callable[[str], None] | None = None,
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
    phase_spec = phase_search or PhaseSearchSpec()
    if base_dictionary is None:
        _progress(progress, "construction: selecting base dictionary")
        base_dictionary = _select_observed_atoms(
            target_matrix,
            width=base_dictionary_size,
            backend=backend,
            chunk_size=chunk_size,
            include_zero=False,
        )

    _progress(progress, "construction: assigning base dictionary")
    _, _, base_values, _ = _nearest_shifted(
        target_matrix,
        base_dictionary,
        phase_search=phase_spec,
        backend=backend,
        chunk_size=chunk_size,
    )
    prefix = base_values.copy()
    residual_layers: list[np.ndarray] = []
    for residual_layer_index in range(residual_layer_count):
        layer_number = residual_layer_index + 1
        if _should_report_layer(layer_number, residual_layer_count):
            _progress(progress, f"construction: residual layer {layer_number}/{residual_layer_count} selecting atoms")
        residual = target_matrix - prefix
        atoms = _select_observed_atoms(
            residual,
            width=width,
            backend=backend,
            chunk_size=chunk_size,
            include_zero=True,
        )
        residual_layers.append(atoms)
        if _should_report_layer(layer_number, residual_layer_count):
            _progress(progress, f"construction: residual layer {layer_number}/{residual_layer_count} updating prefix")
        _, _, layer_values, _ = _nearest_shifted(
            residual,
            atoms,
            phase_search=phase_spec,
            backend=backend,
            chunk_size=chunk_size,
        )
        prefix = prefix + layer_values
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
    phase_search: PhaseSearchSpec | None = None,
    backend: BackendPreference = "auto",
    chunk_size: int = 256,
    progress: Callable[[str], None] | None = None,
    progress_label: str = "encoding",
) -> FlatEncodingResult:
    target_matrix = np.asarray(targets, dtype=np.float32)
    if target_matrix.ndim != 2:
        raise ValueError("targets must have shape [rows, resolution]")
    nearest_reports: list[NearestResult] = []
    phase_spec = phase_search or PhaseSearchSpec()

    _progress(progress, f"{progress_label}: base choice")
    base_index, base_phase, base_values, report = _nearest_shifted(
        target_matrix,
        assets.base_dictionary,
        phase_search=phase_spec,
        backend=backend,
        chunk_size=chunk_size,
    )
    nearest_reports.append(report)
    prefix = base_values.copy()

    residual_indices: list[np.ndarray] = []
    residual_phases: list[np.ndarray] = []
    residual_layer_count = len(assets.residual_layer_dictionaries)
    for residual_layer_index, dictionary in enumerate(assets.residual_layer_dictionaries):
        layer_number = residual_layer_index + 1
        if _should_report_layer(layer_number, residual_layer_count):
            _progress(progress, f"{progress_label}: residual layer {layer_number}/{residual_layer_count}")
        residual = target_matrix - prefix
        index, phase, values, report = _nearest_shifted(
            residual,
            dictionary,
            phase_search=phase_spec,
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
    phase_search = PhaseSearchSpec(
        policy="disabled" if phase_bins <= 1 else "grid",
        candidate_count=max(1, int(phase_bins)),
    )
    encoded = encode_flat(targets, assets, phase_search=phase_search, backend=backend)
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
        lfo_control_point_count=resolution,
        oracle_construction_time=construction_time,
        oracle_encoding_time=encoding_time,
        method_parameters={
            "W_by_residual_layer": assets.residual_widths(),
            **phase_search.as_manifest_fields(resolution),
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
    phase_search: PhaseSearchSpec,
    backend: BackendPreference,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, NearestResult]:
    target_matrix = as_curve_matrix(targets)
    code_matrix = as_curve_matrix(codes)
    if target_matrix.shape[1] != code_matrix.shape[1]:
        raise ValueError("targets and codes must share the same resolution")
    candidate_count = phase_search.resolved_candidate_count(target_matrix.shape[1])
    if phase_search.policy == "fft_lattice":
        return _nearest_fft_lattice(target_matrix, code_matrix, chunk_size=chunk_size)
    bank, phases = phase_shift_bank(code_matrix, candidate_count)
    flat_bank = bank.reshape(bank.shape[0] * bank.shape[1], bank.shape[2])
    result = nearest_indices(targets, flat_bank, backend=backend, chunk_size=chunk_size)
    code_index = (result.indices // candidate_count).astype(np.int32)
    phase_index = (result.indices % candidate_count).astype(np.int32)
    selected = flat_bank[result.indices]
    return code_index, phases[phase_index].astype(np.float32), selected.astype(np.float32), result


def _nearest_fft_lattice(
    targets: np.ndarray,
    codes: np.ndarray,
    *,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, NearestResult]:
    """Find best code and integer-sample phase by circular correlation."""
    target_matrix = as_curve_matrix(targets)
    code_matrix = as_curve_matrix(codes)
    resolution = target_matrix.shape[1]
    if code_matrix.shape[1] != resolution:
        raise ValueError("targets and codes must share the same resolution")
    if len(code_matrix) == 0:
        raise ValueError("codes cannot be empty")

    chunk_size = max(1, int(chunk_size))
    code_fft = np.fft.rfft(code_matrix, axis=-1)
    code_energy = np.sum(code_matrix * code_matrix, axis=1, dtype=np.float64)[None, :, None]
    indices = np.empty(len(target_matrix), dtype=np.int32)
    phases = np.empty(len(target_matrix), dtype=np.float32)
    selected = np.empty_like(target_matrix, dtype=np.float32)
    losses = np.empty(len(target_matrix), dtype=np.float32)

    for start in range(0, len(target_matrix), chunk_size):
        stop = min(start + chunk_size, len(target_matrix))
        batch = target_matrix[start:stop]
        target_fft = np.fft.rfft(batch, axis=-1)
        correlations = np.fft.irfft(
            target_fft[:, None, :] * np.conj(code_fft[None, :, :]),
            n=resolution,
            axis=-1,
        ).real
        target_energy = np.sum(batch * batch, axis=1, dtype=np.float64)[:, None, None]
        mse = (target_energy + code_energy - 2.0 * correlations) / float(resolution)
        flat_choice = np.argmin(mse.reshape(len(batch), -1), axis=1)
        code_index = (flat_choice // resolution).astype(np.int32)
        phase_index = (flat_choice % resolution).astype(np.int32)
        row = np.arange(len(batch))
        indices[start:stop] = code_index
        phases[start:stop] = phase_index.astype(np.float32) / float(resolution)
        losses[start:stop] = np.maximum(mse[row, code_index, phase_index], 0.0).astype(np.float32)
        selected[start:stop] = circular_shift(code_matrix[code_index], phases[start:stop])

    result = NearestResult(
        indices=indices,
        losses=losses,
        backend_used="fft_lattice",
        chunk_size=chunk_size,
    )
    return indices, phases, selected.astype(np.float32), result


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


def _progress(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _should_report_layer(layer_number: int, total_layers: int) -> bool:
    return layer_number == 1 or layer_number == total_layers or layer_number % 10 == 0

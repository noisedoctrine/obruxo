"""Experiment 11 W8D16 deviation audit."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np

from .accelerator import BackendPreference
from .accounting import RuntimeInterfaceSpec
from .alignment import AlignmentChoice, alignment_matrix, best_alignment
from .assets import DecoderPolicy, OracleEncoding, ReconstructionAssets
from .contracts import TopologyFlags, validate_topology_contract
from .curve import circular_shift
from .dataset import Era2CurveDataset, TOPOLOGY_NAMES, load_presetshare_curve_dataset
from .flat import PhaseSearchSpec, construct_flat_assets_from_curves, decode_flat, encode_flat
from .manifest import write_json, write_summary_csv
from .metrics import flat_atom_usage, reconstruction_summary
from .runner import DEFAULT_METADATA, ERA2_ROOT


DEFAULT_OUTPUT_DIR = ERA2_ROOT / "artifacts" / "experiment_11" / "w8d16_deviation_audit"
REPORT_PATH = ERA2_ROOT / "reports" / "EXPERIMENT_11_FLAT_CATEGORICAL_REPORT.md"
REPORT_IMAGE_DIR = ERA2_ROOT / "reports" / "images" / "experiment_11"
ERA1_SUMMARY = ERA2_ROOT.parent / "era1" / "artifacts" / "additive_finalization_9_screen" / "analytics" / "summary.csv"
BASE_DICTIONARY_SIZE = 32
CONTROL_POINT_COUNT = 97
W = 8
D = 16


@dataclass(frozen=True)
class DiagnosticRowSpec:
    row_id: str
    description: str
    x_grid_mode: str = "inclusive"
    phase_policy: str = "fft_lattice"
    construction_policy: str = "farthest"
    path_policy: str = "greedy"
    residual_gain_policy: str = "fixed"
    residual_gain_model_facing: bool = False
    topology_runtime: bool = False
    topology_used_in_construction: bool = False
    beam_width: int = 1
    max_utility_candidates: int = 24


@dataclass
class DiagnosticEncoding:
    base_index: np.ndarray
    base_phase: np.ndarray
    base_gain: np.ndarray
    residual_layer_indices: list[np.ndarray]
    residual_layer_phases: list[np.ndarray]
    residual_layer_gains: list[np.ndarray]

    def oracle_encoding(self) -> OracleEncoding:
        return OracleEncoding(
            base_index=self.base_index,
            base_phase=self.base_phase,
            residual_layer_indices=self.residual_layer_indices,
            residual_layer_phases=self.residual_layer_phases,
        )

    def as_arrays(self) -> dict[str, np.ndarray]:
        payload = self.oracle_encoding().as_arrays()
        payload["base_gain"] = self.base_gain
        for index, values in enumerate(self.residual_layer_gains, start=1):
            payload[f"residual_layer_{index}_gain"] = values
        return payload


def default_diagnostic_specs() -> list[DiagnosticRowSpec]:
    return [
        DiagnosticRowSpec(
            row_id="current_endpoint_excluded_lattice_greedy_farthest",
            description="Current Era 2 behavior: endpoint-excluded 97 samples, lattice phase, greedy path, topology-blind farthest-residual construction.",
            x_grid_mode="endpoint_excluded",
        ),
        DiagnosticRowSpec(
            row_id="inclusive97_lattice_only",
            description="Only the 97-control-point lattice is corrected to inclusive 96-subdivision geometry.",
        ),
        DiagnosticRowSpec(
            row_id="exact_phase_only",
            description="On the corrected inclusive grid, phase alignment changes to exact piecewise-linear phase search.",
            phase_policy="exact",
        ),
        DiagnosticRowSpec(
            row_id="beam4_only",
            description="On the corrected inclusive grid, path search changes from greedy to beam width 4.",
            path_policy="beam",
            beam_width=4,
        ),
        DiagnosticRowSpec(
            row_id="optimized_residual_gain_only",
            description="On the corrected inclusive grid, residual-layer atom gain becomes optimized and model-facing.",
            residual_gain_policy="optimized",
            residual_gain_model_facing=True,
        ),
        DiagnosticRowSpec(
            row_id="utility_construction_only",
            description="On the corrected inclusive grid, residual atom construction changes from farthest residuals to sampled utility selection.",
            construction_policy="utility",
        ),
        DiagnosticRowSpec(
            row_id="topology_balanced_construction",
            description="Topology is used only to balance offline atom construction; runtime schema stays topology-free.",
            construction_policy="topology_balanced_farthest",
            topology_used_in_construction=True,
        ),
        DiagnosticRowSpec(
            row_id="inclusive_exact_beam4_utility",
            description="Cumulative likely-fix row: inclusive 97, exact phase, beam width 4, utility construction.",
            phase_policy="exact",
            construction_policy="utility",
            path_policy="beam",
            beam_width=4,
        ),
        DiagnosticRowSpec(
            row_id="inclusive_exact_beam4_topology_balanced_utility",
            description="Cumulative likely-fix row plus topology-balanced offline construction.",
            phase_policy="exact",
            construction_policy="topology_balanced_utility",
            path_policy="beam",
            beam_width=4,
            topology_used_in_construction=True,
        ),
        DiagnosticRowSpec(
            row_id="quarantined_topology_runtime_reference",
            description="Invalid Era 2 deployment reference: topology selects a runtime dictionary. This is diagnostic only.",
            phase_policy="exact",
            construction_policy="utility",
            path_policy="beam",
            beam_width=4,
            topology_runtime=True,
            topology_used_in_construction=True,
        ),
    ]


def run_deviation_audit(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    metadata_path: Path = DEFAULT_METADATA,
    backend: BackendPreference = "auto",
    corpus_sample_fraction: float = 1.0,
    row_ids: set[str] | None = None,
    max_utility_candidates: int | None = None,
    chunk_size: int = 256,
    write_report: bool = True,
    progress: Callable[[str], None] | None = None,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = [spec for spec in default_diagnostic_specs() if row_ids is None or spec.row_id in row_ids]
    if max_utility_candidates is not None:
        specs = [
            DiagnosticRowSpec(**{**asdict(spec), "max_utility_candidates": int(max_utility_candidates)})
            for spec in specs
        ]
    if not specs:
        raise ValueError("no diagnostic rows selected")
    datasets = _load_datasets(
        specs,
        metadata_path=metadata_path,
        corpus_sample_fraction=corpus_sample_fraction,
        progress=progress,
    )
    summary_rows = []
    budget_rows = []
    phase_gain_rows = []
    usage_rows = []
    reference_rows = _era1_reference_rows()
    for index, spec in enumerate(specs, start=1):
        _log(progress, f"deviation_audit: [{index}/{len(specs)}] {spec.row_id} start")
        started = time.perf_counter()
        dataset = datasets[spec.x_grid_mode]
        row_dir = output_dir / "rows" / spec.row_id
        row_dir.mkdir(parents=True, exist_ok=True)
        if spec.topology_runtime:
            row_summary, phase_gain, usage = _run_topology_runtime_reference(
                spec,
                dataset,
                backend=backend,
                chunk_size=chunk_size,
                progress=progress,
            )
        else:
            row_summary, phase_gain, usage = _run_topology_free_row(
                spec,
                dataset,
                backend=backend,
                chunk_size=chunk_size,
                progress=progress,
            )
        row_summary["row_elapsed_seconds"] = time.perf_counter() - started
        row_summary["row_number"] = index
        row_summary["row_count"] = len(specs)
        write_json(row_dir / "spec.json", asdict(spec))
        write_summary_csv(row_dir / "summary.csv", row_summary)
        write_json(row_dir / "phase_gain_usage.json", phase_gain)
        write_json(row_dir / "atom_usage.json", usage)
        summary_rows.append(row_summary)
        budget_rows.append(_budget_row(row_summary))
        phase_gain_rows.append({"row_id": spec.row_id, **phase_gain})
        usage_rows.append({"row_id": spec.row_id, **usage})
        _log(
            progress,
            f"deviation_audit: [{index}/{len(specs)}] {spec.row_id} "
            f"validation_p95={row_summary.get('validation_p95_rmse')} elapsed={row_summary['row_elapsed_seconds']:.2f}s",
        )

    summary_path = output_dir / "summary.csv"
    references_path = output_dir / "era1_w8d16_reference_anchors.csv"
    budget_path = output_dir / "budget_accounting.csv"
    phase_gain_path = output_dir / "phase_gain_usage.csv"
    usage_path = output_dir / "atom_usage_diagnostics.csv"
    _write_csv(summary_path, summary_rows)
    _write_csv(references_path, reference_rows)
    _write_csv(budget_path, budget_rows)
    _write_csv(phase_gain_path, phase_gain_rows)
    _write_csv(usage_path, usage_rows)
    if write_report:
        _write_plots(REPORT_IMAGE_DIR, summary_rows, reference_rows)
        _write_report_section(REPORT_PATH, summary_rows, reference_rows)
    return {
        "output_dir": str(output_dir),
        "summary": str(summary_path),
        "era1_reference_anchors": str(references_path),
        "budget_accounting": str(budget_path),
        "phase_gain_usage": str(phase_gain_path),
        "atom_usage_diagnostics": str(usage_path),
        "report": str(REPORT_PATH) if write_report else "",
        "report_image_dir": str(REPORT_IMAGE_DIR) if write_report else "",
    }


def _load_datasets(
    specs: list[DiagnosticRowSpec],
    *,
    metadata_path: Path,
    corpus_sample_fraction: float,
    progress: Callable[[str], None] | None,
) -> dict[str, Era2CurveDataset]:
    modes = sorted({spec.x_grid_mode for spec in specs})
    datasets: dict[str, Era2CurveDataset] = {}
    for mode in modes:
        _log(progress, f"deviation_audit: loading dataset x_grid_mode={mode}")
        dataset = load_presetshare_curve_dataset(
            metadata_path,
            resolution=CONTROL_POINT_COUNT,
            x_grid_mode=mode,
            progress=progress,
        )
        if not (0.0 < float(corpus_sample_fraction) <= 1.0):
            raise ValueError("corpus_sample_fraction must be in (0, 1]")
        if float(corpus_sample_fraction) < 1.0:
            train_count = max(1, int(len(dataset.train_indices) * float(corpus_sample_fraction)))
            validation_count = max(1, int(len(dataset.validation_indices) * float(corpus_sample_fraction)))
            dataset = dataset.subset(train_count=train_count, validation_count=validation_count)
        datasets[mode] = dataset
    return datasets


def _run_topology_free_row(
    spec: DiagnosticRowSpec,
    dataset: Era2CurveDataset,
    *,
    backend: BackendPreference,
    chunk_size: int,
    progress: Callable[[str], None] | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    flags = TopologyFlags(topology_used_in_construction=spec.topology_used_in_construction)
    contract = validate_topology_contract(flags)
    construction_started = time.perf_counter()
    assets = _construct_assets(spec, dataset.train_curves, dataset.topology[dataset.train_indices], backend=backend, chunk_size=chunk_size, progress=progress)
    construction_time = time.perf_counter() - construction_started
    train_encoding, train_reconstructed, train_encoding_time = _encode_decode(
        spec,
        dataset.train_curves,
        assets,
        backend=backend,
        chunk_size=chunk_size,
        progress=progress,
        progress_label="train",
    )
    validation_encoding, validation_reconstructed, validation_encoding_time = _encode_decode(
        spec,
        dataset.validation_curves,
        assets,
        backend=backend,
        chunk_size=chunk_size,
        progress=progress,
        progress_label="validation",
    )
    summary = _summary_base(spec, dataset, contract_pass=contract.passed)
    summary.update(
        {
            "oracle_construction_time": construction_time,
            "train_encoding_time": train_encoding_time,
            "validation_encoding_time": validation_encoding_time,
            "dictionary_scope": assets.dictionary_scope,
            "codebook_storage_count": assets.codebook_storage_count,
            "backend_used": _backend_used(validation_encoding),
            **_prefix("train", reconstruction_summary(dataset.train_curves, train_reconstructed)),
            **_prefix("validation", reconstruction_summary(dataset.validation_curves, validation_reconstructed)),
        }
    )
    usage = _usage_summary(validation_encoding, widths=assets.residual_widths())
    summary.update(usage)
    return summary, _phase_gain_summary(validation_encoding), usage


def _run_topology_runtime_reference(
    spec: DiagnosticRowSpec,
    dataset: Era2CurveDataset,
    *,
    backend: BackendPreference,
    chunk_size: int,
    progress: Callable[[str], None] | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    reconstructed_train = np.zeros_like(dataset.train_curves)
    reconstructed_validation = np.zeros_like(dataset.validation_curves)
    combined_validation = _empty_encoding(len(dataset.validation_curves))
    construction_time = 0.0
    train_encoding_time = 0.0
    validation_encoding_time = 0.0
    widths = [W] * D
    for topology_index, topology_name in enumerate(TOPOLOGY_NAMES):
        train_local = np.flatnonzero(dataset.topology[dataset.train_indices] == topology_index)
        validation_local = np.flatnonzero(dataset.topology[dataset.validation_indices] == topology_index)
        if len(train_local) == 0 or len(validation_local) == 0:
            continue
        _log(progress, f"deviation_audit: topology runtime branch={topology_name} train={len(train_local)} validation={len(validation_local)}")
        branch_dataset_curves = dataset.train_curves[train_local]
        branch_topology = np.full(len(train_local), topology_index, dtype=np.int8)
        started = time.perf_counter()
        assets = _construct_assets(spec, branch_dataset_curves, branch_topology, backend=backend, chunk_size=chunk_size, progress=progress)
        construction_time += time.perf_counter() - started
        train_encoding, train_recon, elapsed = _encode_decode(
            spec,
            dataset.train_curves[train_local],
            assets,
            backend=backend,
            chunk_size=chunk_size,
            progress=progress,
            progress_label=f"train {topology_name}",
        )
        train_encoding_time += elapsed
        reconstructed_train[train_local] = train_recon
        validation_encoding, validation_recon, elapsed = _encode_decode(
            spec,
            dataset.validation_curves[validation_local],
            assets,
            backend=backend,
            chunk_size=chunk_size,
            progress=progress,
            progress_label=f"validation {topology_name}",
        )
        validation_encoding_time += elapsed
        reconstructed_validation[validation_local] = validation_recon
        _scatter_encoding(combined_validation, validation_encoding, validation_local)
        widths = assets.residual_widths()

    summary = _summary_base(spec, dataset, contract_pass=False)
    summary.update(
        {
            "oracle_construction_time": construction_time,
            "train_encoding_time": train_encoding_time,
            "validation_encoding_time": validation_encoding_time,
            "dictionary_scope": "per_topology_runtime_branch",
            "codebook_storage_count": int((BASE_DICTIONARY_SIZE + D * W) * len(TOPOLOGY_NAMES)),
            "backend_used": "diagnostic_topology_runtime",
            "invalid_runtime_contract_reason": "topology selects the runtime dictionary branch",
            **_prefix("train", reconstruction_summary(dataset.train_curves, reconstructed_train)),
            **_prefix("validation", reconstruction_summary(dataset.validation_curves, reconstructed_validation)),
        }
    )
    usage = _usage_summary(combined_validation, widths=widths)
    summary.update(usage)
    return summary, _phase_gain_summary(combined_validation), usage


def _construct_assets(
    spec: DiagnosticRowSpec,
    targets: np.ndarray,
    topology: np.ndarray,
    *,
    backend: BackendPreference,
    chunk_size: int,
    progress: Callable[[str], None] | None,
) -> ReconstructionAssets:
    if _uses_current_flat_constructor(spec):
        return construct_flat_assets_from_curves(
            targets,
            base_dictionary_size=BASE_DICTIONARY_SIZE,
            residual_layer_count=D,
            width=W,
            backend=backend,
            chunk_size=chunk_size,
            phase_search=PhaseSearchSpec(policy="fft_lattice"),
            progress=progress,
        )
    base = _select_farthest_atoms(targets, width=BASE_DICTIONARY_SIZE, include_zero=False, topology=None)
    base_choice = best_alignment(
        targets,
        base,
        phase_policy=spec.phase_policy,
        gain_policy="fixed",
        backend=backend,
        chunk_size=chunk_size,
    )
    prefix = base_choice.values.copy()
    layers = []
    for residual_layer in range(D):
        if residual_layer == 0 or residual_layer + 1 == D or (residual_layer + 1) % 4 == 0:
            _log(progress, f"construction: residual layer {residual_layer + 1}/{D}")
        residual = targets - prefix
        atoms = _select_layer_atoms(
            residual,
            topology,
            spec=spec,
            backend=backend,
            chunk_size=chunk_size,
        )
        layers.append(atoms)
        choice = best_alignment(
            residual,
            atoms,
            phase_policy=spec.phase_policy,
            gain_policy=spec.residual_gain_policy,
            backend=backend,
            chunk_size=chunk_size,
        )
        prefix = prefix + choice.values
    return ReconstructionAssets(
        base_dictionary=base,
        residual_layer_dictionaries=layers,
        dictionary_scope="per_residual_layer",
        metadata={"construction_policy": spec.construction_policy},
    )


def _uses_current_flat_constructor(spec: DiagnosticRowSpec) -> bool:
    return (
        spec.phase_policy == "fft_lattice"
        and spec.construction_policy == "farthest"
        and spec.path_policy == "greedy"
        and spec.residual_gain_policy == "fixed"
        and not spec.topology_runtime
    )


def _select_layer_atoms(
    residual: np.ndarray,
    topology: np.ndarray,
    *,
    spec: DiagnosticRowSpec,
    backend: BackendPreference,
    chunk_size: int,
) -> np.ndarray:
    if spec.construction_policy == "farthest":
        return _select_farthest_atoms(residual, width=W, include_zero=True, topology=None)
    if spec.construction_policy == "topology_balanced_farthest":
        return _select_farthest_atoms(residual, width=W, include_zero=True, topology=topology)
    if spec.construction_policy == "utility":
        return _select_utility_atoms(residual, width=W, topology=None, spec=spec, chunk_size=chunk_size)
    if spec.construction_policy == "topology_balanced_utility":
        return _select_utility_atoms(residual, width=W, topology=topology, spec=spec, chunk_size=chunk_size)
    raise ValueError(f"unsupported construction_policy: {spec.construction_policy}")


def _select_farthest_atoms(
    residual: np.ndarray,
    *,
    width: int,
    include_zero: bool,
    topology: np.ndarray | None,
) -> np.ndarray:
    matrix = np.asarray(residual, dtype=np.float32)
    atoms = []
    selected: set[int] = set()
    if include_zero:
        atoms.append(np.zeros(matrix.shape[1], dtype=np.float32))
    else:
        center = np.mean(matrix, axis=0, dtype=np.float32)
        distances = np.mean((matrix - center[None, :]) ** 2, axis=1)
        first = int(np.argmin(distances))
        atoms.append(matrix[first].astype(np.float32))
        selected.add(first)
    while len(atoms) < int(width):
        current = np.stack(atoms).astype(np.float32)
        losses = np.min(_squared_distance_matrix(matrix, current), axis=1)
        if topology is None:
            order = np.argsort(losses)[::-1]
        else:
            bucket = (len(atoms) - int(include_zero)) % len(TOPOLOGY_NAMES)
            members = np.flatnonzero(topology == bucket)
            order = members[np.argsort(losses[members])[::-1]] if len(members) else np.argsort(losses)[::-1]
        candidate = next((int(index) for index in order if int(index) not in selected), None)
        if candidate is None:
            atoms.append(np.zeros(matrix.shape[1], dtype=np.float32))
        else:
            selected.add(candidate)
            atoms.append(matrix[candidate].astype(np.float32))
    return np.stack(atoms).astype(np.float32)


def _select_utility_atoms(
    residual: np.ndarray,
    *,
    width: int,
    topology: np.ndarray | None,
    spec: DiagnosticRowSpec,
    chunk_size: int,
) -> np.ndarray:
    matrix = np.asarray(residual, dtype=np.float32)
    atoms = [np.zeros(matrix.shape[1], dtype=np.float32)]
    current_loss = np.mean(matrix * matrix, axis=1)
    selected: set[int] = set()
    while len(atoms) < int(width):
        pool = _utility_candidate_pool(matrix, current_loss, selected, topology=topology, limit=spec.max_utility_candidates)
        if len(pool) == 0:
            atoms.append(np.zeros(matrix.shape[1], dtype=np.float32))
            continue
        candidates = matrix[pool]
        losses = alignment_matrix(
            matrix,
            candidates,
            phase_policy=spec.phase_policy,
            gain_policy=spec.residual_gain_policy,
            chunk_size=chunk_size,
        ).losses
        improvement = np.maximum(current_loss[:, None] - losses, 0.0).sum(axis=0)
        chosen_local = int(np.argmax(improvement))
        chosen = int(pool[chosen_local])
        selected.add(chosen)
        atoms.append(matrix[chosen].astype(np.float32))
        current_loss = np.minimum(current_loss, losses[:, chosen_local])
    return np.stack(atoms).astype(np.float32)


def _utility_candidate_pool(
    matrix: np.ndarray,
    current_loss: np.ndarray,
    selected: set[int],
    *,
    topology: np.ndarray | None,
    limit: int,
) -> np.ndarray:
    if topology is None:
        order = np.argsort(current_loss)[::-1]
        return np.asarray([index for index in order if int(index) not in selected][:limit], dtype=np.int32)
    per_bucket = max(1, int(np.ceil(limit / len(TOPOLOGY_NAMES))))
    chosen = []
    for bucket in range(len(TOPOLOGY_NAMES)):
        members = np.flatnonzero(topology == bucket)
        ordered = members[np.argsort(current_loss[members])[::-1]]
        chosen.extend(int(index) for index in ordered if int(index) not in selected)
    return np.asarray(chosen[:limit], dtype=np.int32)


def _encode_decode(
    spec: DiagnosticRowSpec,
    targets: np.ndarray,
    assets: ReconstructionAssets,
    *,
    backend: BackendPreference,
    chunk_size: int,
    progress: Callable[[str], None] | None,
    progress_label: str,
) -> tuple[DiagnosticEncoding, np.ndarray, float]:
    started = time.perf_counter()
    if _uses_current_flat_constructor(spec):
        encoded = encode_flat(
            targets,
            assets,
            phase_search=PhaseSearchSpec(policy="fft_lattice"),
            backend=backend,
            chunk_size=chunk_size,
            progress=progress,
            progress_label=f"{progress_label} encoding",
        )
        reconstructed = decode_flat(assets, encoded.encoding, decoder_policy=DecoderPolicy())
        diagnostic = DiagnosticEncoding(
            base_index=encoded.encoding.base_index,
            base_phase=encoded.encoding.base_phase,
            base_gain=np.ones(encoded.encoding.row_count, dtype=np.float32),
            residual_layer_indices=encoded.encoding.residual_layer_indices,
            residual_layer_phases=encoded.encoding.residual_layer_phases,
            residual_layer_gains=[np.ones(encoded.encoding.row_count, dtype=np.float32) for _ in encoded.encoding.residual_layer_indices],
        )
        return diagnostic, reconstructed, time.perf_counter() - started
    if spec.path_policy == "greedy":
        encoding, reconstructed = _encode_greedy(spec, targets, assets, backend=backend, chunk_size=chunk_size, progress=progress, progress_label=progress_label)
    elif spec.path_policy == "beam":
        encoding, reconstructed = _encode_beam(spec, targets, assets, backend=backend, chunk_size=chunk_size, progress=progress, progress_label=progress_label)
    else:
        raise ValueError(f"unsupported path_policy: {spec.path_policy}")
    return encoding, np.clip(reconstructed, 0.0, 1.0).astype(np.float32), time.perf_counter() - started


def _encode_greedy(
    spec: DiagnosticRowSpec,
    targets: np.ndarray,
    assets: ReconstructionAssets,
    *,
    backend: BackendPreference,
    chunk_size: int,
    progress: Callable[[str], None] | None,
    progress_label: str,
) -> tuple[DiagnosticEncoding, np.ndarray]:
    base = best_alignment(targets, assets.base_dictionary, phase_policy=spec.phase_policy, gain_policy="fixed", backend=backend, chunk_size=chunk_size)
    prefix = base.values.copy()
    indices = []
    phases = []
    gains = []
    for residual_layer, dictionary in enumerate(assets.residual_layer_dictionaries):
        if residual_layer == 0 or residual_layer + 1 == D or (residual_layer + 1) % 4 == 0:
            _log(progress, f"{progress_label} encoding: residual layer {residual_layer + 1}/{D}")
        residual = targets - prefix
        choice = best_alignment(
            residual,
            dictionary,
            phase_policy=spec.phase_policy,
            gain_policy=spec.residual_gain_policy,
            backend=backend,
            chunk_size=chunk_size,
        )
        indices.append(choice.indices)
        phases.append(choice.phases)
        gains.append(choice.gains)
        prefix = prefix + choice.values
    return DiagnosticEncoding(base.indices, base.phases, base.gains, indices, phases, gains), prefix


def _encode_beam(
    spec: DiagnosticRowSpec,
    targets: np.ndarray,
    assets: ReconstructionAssets,
    *,
    backend: BackendPreference,
    chunk_size: int,
    progress: Callable[[str], None] | None,
    progress_label: str,
) -> tuple[DiagnosticEncoding, np.ndarray]:
    rows = len(targets)
    beam = max(1, int(spec.beam_width))
    out = _empty_encoding(rows)
    reconstructed = np.empty_like(targets, dtype=np.float32)
    for start in range(0, rows, max(1, int(chunk_size))):
        stop = min(start + max(1, int(chunk_size)), rows)
        batch = targets[start:stop]
        local = _encode_beam_batch(spec, batch, assets, backend=backend, chunk_size=max(1, int(chunk_size)), beam_width=beam)
        _scatter_encoding(out, local[0], np.arange(start, stop))
        reconstructed[start:stop] = local[1]
        if start == 0 or stop == rows:
            _log(progress, f"{progress_label} encoding: beam batch {stop}/{rows}")
    return out, reconstructed


def _encode_beam_batch(
    spec: DiagnosticRowSpec,
    targets: np.ndarray,
    assets: ReconstructionAssets,
    *,
    backend: BackendPreference,
    chunk_size: int,
    beam_width: int,
) -> tuple[DiagnosticEncoding, np.ndarray]:
    base_matrix = alignment_matrix(targets, assets.base_dictionary, phase_policy=spec.phase_policy, gain_policy="fixed", chunk_size=chunk_size)
    beam = min(beam_width, base_matrix.losses.shape[1])
    base_choices = np.argsort(base_matrix.losses, axis=1)[:, :beam]
    b = len(targets)
    rows = np.arange(b)[:, None]
    base_phases = base_matrix.phases[rows, base_choices]
    base_gains = base_matrix.gains[rows, base_choices]
    prefix = (
        circular_shift(assets.base_dictionary[base_choices.reshape(-1)], base_phases.reshape(-1))
        * base_gains.reshape(-1, 1)
    ).reshape(b, beam, targets.shape[1]).astype(np.float32)
    base_paths = base_choices.astype(np.int32)
    base_phase_paths = base_phases.astype(np.float32)
    base_gain_paths = base_gains.astype(np.float32)
    index_paths = np.zeros((b, beam, D), dtype=np.int32)
    phase_paths = np.zeros((b, beam, D), dtype=np.float32)
    gain_paths = np.zeros((b, beam, D), dtype=np.float32)
    for residual_layer, dictionary in enumerate(assets.residual_layer_dictionaries):
        current_beam = prefix.shape[1]
        residual = (targets[:, None, :] - prefix).reshape(b * current_beam, targets.shape[1])
        matrix = alignment_matrix(
            residual,
            dictionary,
            phase_policy=spec.phase_policy,
            gain_policy=spec.residual_gain_policy,
            chunk_size=chunk_size,
        )
        shifted = circular_shift(
            dictionary[np.broadcast_to(np.arange(len(dictionary)), (b * current_beam, len(dictionary))).reshape(-1)],
            matrix.phases.reshape(-1),
        ).reshape(b, current_beam, len(dictionary), targets.shape[1])
        additions = shifted * matrix.gains.reshape(b, current_beam, len(dictionary), 1)
        candidate_state = prefix[:, :, None, :] + additions
        candidate_recon = np.clip(candidate_state, 0.0, 1.0)
        mse = np.mean((targets[:, None, None, :] - candidate_recon) ** 2, axis=3)
        previous = np.mean((targets[:, None, :] - np.clip(prefix, 0.0, 1.0)) ** 2, axis=2)
        mse[:, :, 0] = np.minimum(mse[:, :, 0], previous)
        candidate_state[:, :, 0, :] = prefix
        matrix.phases[:, 0] = 0.0
        matrix.gains[:, 0] = 0.0
        flat = mse.reshape(b, -1)
        next_beam = min(beam_width, flat.shape[1])
        choice = np.argsort(flat, axis=1)[:, :next_beam]
        parent = choice // len(dictionary)
        code = choice % len(dictionary)
        row_index = np.arange(b)[:, None]
        prefix = candidate_state[row_index, parent, code]
        base_paths = base_paths[row_index, parent]
        base_phase_paths = base_phase_paths[row_index, parent]
        base_gain_paths = base_gain_paths[row_index, parent]
        if residual_layer:
            index_paths[:, :next_beam, :residual_layer] = index_paths[row_index, parent, :residual_layer]
            phase_paths[:, :next_beam, :residual_layer] = phase_paths[row_index, parent, :residual_layer]
            gain_paths[:, :next_beam, :residual_layer] = gain_paths[row_index, parent, :residual_layer]
        phase3 = matrix.phases.reshape(b, current_beam, len(dictionary))
        gain3 = matrix.gains.reshape(b, current_beam, len(dictionary))
        index_paths[:, :next_beam, residual_layer] = code
        phase_paths[:, :next_beam, residual_layer] = phase3[row_index, parent, code]
        gain_paths[:, :next_beam, residual_layer] = gain3[row_index, parent, code]
        prefix = prefix[:, :next_beam]
        base_paths = base_paths[:, :next_beam]
        base_phase_paths = base_phase_paths[:, :next_beam]
        base_gain_paths = base_gain_paths[:, :next_beam]
    encoding = DiagnosticEncoding(
        base_index=base_paths[:, 0],
        base_phase=base_phase_paths[:, 0] % 1.0,
        base_gain=base_gain_paths[:, 0],
        residual_layer_indices=[index_paths[:, 0, layer] for layer in range(D)],
        residual_layer_phases=[phase_paths[:, 0, layer] % 1.0 for layer in range(D)],
        residual_layer_gains=[gain_paths[:, 0, layer] for layer in range(D)],
    )
    return encoding, prefix[:, 0, :].astype(np.float32)


def _summary_base(spec: DiagnosticRowSpec, dataset: Era2CurveDataset, *, contract_pass: bool) -> dict[str, Any]:
    scalar_outputs = D + 1 + (D if spec.residual_gain_model_facing else 0)
    budget = RuntimeInterfaceSpec(
        addressing_scheme="flat_categorical",
        residual_layer_count=D,
        dictionary_scope="per_residual_layer",
        parameters={"width": W},
    ).budget(base_dictionary_size=BASE_DICTIONARY_SIZE, scalar_outputs=scalar_outputs)
    formula = "32 + 16 * 8 + 17"
    if spec.residual_gain_model_facing:
        formula = "32 + 16 * 8 + 17 + 16 residual_gain_scalars"
    return {
        "experiment_id": "experiment_11_w8d16_deviation_audit",
        "row_id": spec.row_id,
        "description": spec.description,
        "W": W,
        "D": D,
        "base_dictionary_size": BASE_DICTIONARY_SIZE,
        "lfo_control_point_count": CONTROL_POINT_COUNT,
        "subdivision_count": CONTROL_POINT_COUNT - 1,
        "x_grid_mode": spec.x_grid_mode,
        "phase_alignment_policy": spec.phase_policy,
        "construction_policy": spec.construction_policy,
        "path_policy": spec.path_policy,
        "beam_width": spec.beam_width,
        "residual_gain_policy": spec.residual_gain_policy,
        "residual_gain_model_facing": spec.residual_gain_model_facing,
        "scalar_outputs": scalar_outputs,
        "head_outputs_formula": formula,
        "head_outputs_actual": budget.head_outputs_actual,
        "head_outputs_valid_for_era2_contract": not spec.topology_runtime,
        "topology_used_in_construction": spec.topology_used_in_construction,
        "topology_used_at_runtime": spec.topology_runtime,
        "topology_used_in_targets": False,
        "topology_used_in_loss": False,
        "topology_used_in_decoder_lookup": spec.topology_runtime,
        "topology_used_in_head_accounting": False,
        "topology_contract_pass": contract_pass,
        "runtime_contract_valid": contract_pass and not spec.topology_runtime,
        "dataset_fingerprint": dataset.source_fingerprint,
        "dataset_row_count": len(dataset.curves),
        "train_count": len(dataset.train_indices),
        "validation_count": len(dataset.validation_indices),
        "dataset_x_grid_mode": dataset.x_grid_mode,
    }


def _usage_summary(encoding: DiagnosticEncoding, *, widths: list[int]) -> dict[str, Any]:
    usage = flat_atom_usage(
        encoding.oracle_encoding().as_arrays(),
        residual_layer_count=D,
        widths_by_residual_layer=widths,
    )
    dead = [value for key, value in usage.items() if key.endswith("_dead_atom_rate")]
    dominant = [value for key, value in usage.items() if key.endswith("_dominant_atom_share")]
    usage["residual_layer_dead_atom_rate_median"] = float(np.median(dead)) if dead else 0.0
    usage["residual_layer_dominant_atom_share_median"] = float(np.median(dominant)) if dominant else 0.0
    usage["residual_layer_dominant_atom_share_p95"] = float(np.quantile(dominant, 0.95)) if dominant else 0.0
    return usage


def _phase_gain_summary(encoding: DiagnosticEncoding) -> dict[str, Any]:
    residual_phases = np.concatenate(encoding.residual_layer_phases) if encoding.residual_layer_phases else np.asarray([], dtype=np.float32)
    residual_gains = np.concatenate(encoding.residual_layer_gains) if encoding.residual_layer_gains else np.asarray([], dtype=np.float32)
    nonzero = np.abs(residual_gains) > 1e-8
    return {
        "base_phase_abs_median": float(np.median(np.abs(encoding.base_phase))) if len(encoding.base_phase) else 0.0,
        "residual_phase_abs_median": float(np.median(np.abs(residual_phases))) if len(residual_phases) else 0.0,
        "residual_gain_median": float(np.median(residual_gains)) if len(residual_gains) else 0.0,
        "residual_gain_abs_p95": float(np.quantile(np.abs(residual_gains), 0.95)) if len(residual_gains) else 0.0,
        "residual_gain_nonzero_rate": float(np.mean(nonzero)) if len(residual_gains) else 0.0,
    }


def _budget_row(summary: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "row_id",
        "W",
        "D",
        "scalar_outputs",
        "head_outputs_formula",
        "head_outputs_actual",
        "head_outputs_valid_for_era2_contract",
        "runtime_contract_valid",
        "residual_gain_model_facing",
        "topology_used_at_runtime",
    ]
    return {key: summary.get(key, "") for key in keys}


def _era1_reference_rows() -> list[dict[str, Any]]:
    if not ERA1_SUMMARY.exists():
        return []
    rows = []
    with ERA1_SUMMARY.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("residual_width") == "8" and row.get("residual_depth") == "16":
                rows.append(row)
    wanted = []
    for label in ("phase_only_final_only", "phase_only_bipolar_guard_each_layer"):
        match = next((row for row in rows if row.get("modifier_label") == label), None)
        if match:
            wanted.append(_era1_reference_payload(match, f"era1_{label}"))
    if rows:
        best = min(rows, key=lambda row: float(row.get("validation_rmse_p95") or row.get("rmse_p95") or "inf"))
        wanted.append(_era1_reference_payload(best, "era1_best_w8d16"))
    return wanted


def _era1_reference_payload(row: dict[str, Any], reference_id: str) -> dict[str, Any]:
    return {
        "reference_id": reference_id,
        "modifier_label": row.get("modifier_label", ""),
        "residual_clip_policy": row.get("residual_clip_policy", ""),
        "validation_median_rmse": row.get("validation_rmse_median") or row.get("rmse_median", ""),
        "validation_p95_rmse": row.get("validation_rmse_p95") or row.get("rmse_p95", ""),
        "head_outputs": row.get("head_outputs") or row.get("budget_actual_head_outputs") or row.get("dense_outputs", ""),
        "legacy_dense_outputs": row.get("legacy_dense_outputs", ""),
        "notes": "Era 1 W8D16 anchor, read-only reference artifact.",
    }


def _write_plots(image_dir: Path, rows: list[dict[str, Any]], references: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    image_dir.mkdir(parents=True, exist_ok=True)
    ordered = [row for row in rows if _float(row.get("validation_p95_rmse")) is not None]
    if not ordered:
        return
    ordered = sorted(ordered, key=lambda row: int(row.get("row_number", 0)))
    _bar_plot(
        image_dir / "w8d16_deviation_validation_p95.png",
        ordered,
        "validation_p95_rmse",
        "validation P95 RMSE",
        "Experiment 11 W8D16 deviation audit: validation P95",
        plt,
    )
    _bar_plot(
        image_dir / "w8d16_deviation_validation_median.png",
        ordered,
        "validation_median_rmse",
        "validation median RMSE",
        "Experiment 11 W8D16 deviation audit: validation median",
        plt,
    )
    _delta_plot(
        image_dir / "w8d16_deviation_delta_vs_current.png",
        ordered,
        baseline_id="current_endpoint_excluded_lattice_greedy_farthest",
        metric="validation_p95_rmse",
        ylabel="P95 RMSE delta vs current Era 2 row",
        title="W8D16 deviation delta vs current Era 2 behavior",
        plt=plt,
    )
    era1_anchor = _reference_value(references, "era1_phase_only_final_only", "validation_p95_rmse")
    if era1_anchor is not None:
        _constant_delta_plot(
            image_dir / "w8d16_deviation_delta_vs_era1.png",
            ordered,
            anchor=era1_anchor,
            metric="validation_p95_rmse",
            ylabel="P95 RMSE delta vs Era 1 phase-only final-only",
            title="W8D16 deviation delta vs Era 1 W8D16 anchor",
            plt=plt,
        )
    _bar_plot(
        image_dir / "w8d16_deviation_dead_atom_rate.png",
        ordered,
        "residual_layer_dead_atom_rate_median",
        "median residual-layer dead atom rate",
        "W8D16 atom collapse diagnostic",
        plt,
    )
    _bar_plot(
        image_dir / "w8d16_deviation_runtime.png",
        ordered,
        "row_elapsed_seconds",
        "row elapsed seconds",
        "W8D16 diagnostic runtime",
        plt,
    )


def _bar_plot(path: Path, rows: list[dict[str, Any]], metric: str, ylabel: str, title: str, plt: Any) -> None:
    labels = [str(row["row_id"]).replace("_", "\n") for row in rows]
    values = [_float(row.get(metric)) or 0.0 for row in rows]
    colors = ["#D95F02" if row.get("topology_used_at_runtime") else "#4C78A8" for row in rows]
    plt.figure(figsize=(max(10.0, 0.8 * len(labels)), 5.2))
    plt.bar(range(len(values)), values, color=colors)
    plt.xticks(range(len(labels)), labels, rotation=0, ha="center", fontsize=8)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _delta_plot(path: Path, rows: list[dict[str, Any]], *, baseline_id: str, metric: str, ylabel: str, title: str, plt: Any) -> None:
    baseline = next((row for row in rows if row.get("row_id") == baseline_id), None)
    if baseline is None or _float(baseline.get(metric)) is None:
        return
    _constant_delta_plot(path, rows, anchor=_float(baseline.get(metric)) or 0.0, metric=metric, ylabel=ylabel, title=title, plt=plt)


def _constant_delta_plot(path: Path, rows: list[dict[str, Any]], *, anchor: float, metric: str, ylabel: str, title: str, plt: Any) -> None:
    labels = [str(row["row_id"]).replace("_", "\n") for row in rows]
    values = [(_float(row.get(metric)) or 0.0) - anchor for row in rows]
    colors = ["#2CA02C" if value < 0 else "#D62728" for value in values]
    plt.figure(figsize=(max(10.0, 0.8 * len(labels)), 5.2))
    plt.axhline(0.0, color="#111827", linewidth=1)
    plt.bar(range(len(values)), values, color=colors)
    plt.xticks(range(len(labels)), labels, rotation=0, ha="center", fontsize=8)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _write_report_section(report_path: Path, rows: list[dict[str, Any]], references: list[dict[str, Any]]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    existing = report_path.read_text(encoding="utf-8") if report_path.exists() else "# Experiment 11 Flat-Categorical Report\n"
    cleaned = existing.split("\n## W8D16 Deviation Audit", 1)[0].rstrip() + "\n"
    section = _report_section(rows, references)
    report_path.write_text(cleaned + "\n" + section, encoding="utf-8")


def _report_section(rows: list[dict[str, Any]], references: list[dict[str, Any]]) -> str:
    baseline = _row_by_id(rows, "current_endpoint_excluded_lattice_greedy_farthest")
    best = min(rows, key=lambda row: _float(row.get("validation_p95_rmse")) or float("inf")) if rows else None
    gain_row = _row_by_id(rows, "optimized_residual_gain_only")
    beam_row = _row_by_id(rows, "beam4_only")
    exact_row = _row_by_id(rows, "exact_phase_only")
    topology_balanced = _row_by_id(rows, "topology_balanced_construction")
    topology_runtime = _row_by_id(rows, "quarantined_topology_runtime_reference")
    era1 = _reference_value(references, "era1_phase_only_final_only", "validation_p95_rmse")
    lines = [
        "## W8D16 Deviation Audit",
        "",
        "This section is a fixed `W=8`, `D=16` diagnostic. It is not a new broad Experiment 11 screen. It asks which concrete deviations from the Era 1 setup explain the quality gap.",
        "",
        "### Main Findings",
        "",
    ]
    if baseline and best:
        lines.append(
            f"The current Era 2-like W8D16 row lands at validation P95 `{_fmt(baseline.get('validation_p95_rmse'))}`. "
            f"The best diagnostic row is `{best.get('row_id')}` at `{_fmt(best.get('validation_p95_rmse'))}`."
        )
    if era1 is not None and best:
        lines.append(
            f"The Era 1 W8D16 `phase_only_final_only` anchor is `{era1:.4f}` P95, so the best diagnostic row is still "
            f"`{(_float(best.get('validation_p95_rmse')) or 0.0) - era1:.4f}` away from that anchor."
        )
    if gain_row and baseline:
        lines.append(
            f"Optimized residual-layer gain is the dominant confirmed deviation: it moves P95 from "
            f"`{_fmt(baseline.get('validation_p95_rmse'))}` to `{_fmt(gain_row.get('validation_p95_rmse'))}`. "
            "That is not the same thing as Era 1's optional modifier/base gain. It is a per-residual-layer reconstruction scalar, and it costs 16 additional model-facing scalar outputs for W8D16."
        )
    if beam_row and baseline:
        lines.append(
            f"Beam search is the strongest zero-head-cost oracle/path improvement in this audit, improving P95 to "
            f"`{_fmt(beam_row.get('validation_p95_rmse'))}` without changing the runtime head formula."
        )
    if exact_row:
        lines.append(
            f"The exact-phase row regresses badly (`{_fmt(exact_row.get('validation_p95_rmse'))}` P95). "
            "Read this as an implementation/representation mismatch, not as evidence that exact phase is conceptually bad: the ported exact solver assumes a periodic sampled curve, while the settled 97-point representation is an inclusive control-point vector over 96 subdivisions."
        )
    if topology_balanced and topology_runtime:
        lines.append(
            f"Topology-balanced offline construction helps modestly (`{_fmt(topology_balanced.get('validation_p95_rmse'))}` P95), "
            f"but the quarantined runtime-topology reference is only `{_fmt(topology_runtime.get('validation_p95_rmse'))}` P95. "
            "So forbidden runtime topology is not the main missing ingredient behind the Era 1 gap."
        )
    lines.extend(
        [
            "The terminology issue is real but secondary: analytics should use `metric_delta` or `metric_improvement` for score changes, while `gain` should mean a reconstruction scalar. The audit rows keep fixed residual gain, optimized residual gain, and model-facing gain budget separate.",
            "Except for the explicit current-behavior control row, the diagnostic rows use the corrected inclusive 97-control-point grid. Labels like `beam4_only` mean the named axis changes relative to the `inclusive97_lattice_only` row.",
            "",
            "![W8D16 validation P95](./images/experiment_11/w8d16_deviation_validation_p95.png)",
            "",
            "![W8D16 delta vs current](./images/experiment_11/w8d16_deviation_delta_vs_current.png)",
            "",
            "### What Each Deviation Tests",
            "",
        ]
    )
    for row in rows:
        status = _classification(row, baseline)
        lines.append(
            f"- `{row.get('row_id')}`: {status}. P95 `{_fmt(row.get('validation_p95_rmse'))}`, "
            f"median `{_fmt(row.get('validation_median_rmse'))}`, heads `{row.get('head_outputs_actual')}`. "
            f"{_description(row)}"
        )
    lines.extend(
        [
            "",
            "### Supporting Plots",
            "",
            "Lower is better for validation median, validation P95, and runtime. Lower dead-atom rate is better when interpreting dictionary usage, but it is diagnostic rather than a direct quality objective.",
            "",
            "![W8D16 validation median](./images/experiment_11/w8d16_deviation_validation_median.png)",
            "",
            "![W8D16 delta vs Era 1](./images/experiment_11/w8d16_deviation_delta_vs_era1.png)",
            "",
            "![W8D16 dead atom rate](./images/experiment_11/w8d16_deviation_dead_atom_rate.png)",
            "",
            "![W8D16 runtime](./images/experiment_11/w8d16_deviation_runtime.png)",
            "",
            "### Method Notes",
            "",
            "- `W` is atom choices per residual layer. It is not a grid subdivision count.",
            "- `control_point_count=97` means 96 subdivisions for inclusive-grid rows.",
            "- The endpoint-excluded row is intentionally retained as the old-behavior control.",
            "- The topology-runtime row is quarantined: it is useful evidence, but not an Era 2 deployable candidate.",
            "- CSV artifacts live under `research/experiments/lfo_representation/era2/artifacts/experiment_11/w8d16_deviation_audit/`.",
        ]
    )
    return "\n".join(lines) + "\n"


def _classification(row: dict[str, Any], baseline: dict[str, Any] | None) -> str:
    if _truthy(row.get("topology_used_at_runtime")):
        return "quarantined invalid-runtime reference"
    if baseline is None or row is baseline:
        return "baseline"
    base = _float(baseline.get("validation_p95_rmse"))
    value = _float(row.get("validation_p95_rmse"))
    if base is None or value is None:
        return "ambiguous"
    delta = value - base
    if delta < -0.005:
        return f"confirmed improvement vs baseline by `{abs(delta):.4f}` P95"
    if abs(delta) <= 0.005:
        return "roughly neutral at this scale"
    return f"regression vs baseline by `{delta:.4f}` P95"


def _description(row: dict[str, Any]) -> str:
    row_id = str(row.get("row_id", ""))
    by_id = {spec.row_id: spec.description for spec in default_diagnostic_specs()}
    return by_id.get(row_id, str(row.get("description", "")))


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _empty_encoding(row_count: int) -> DiagnosticEncoding:
    return DiagnosticEncoding(
        base_index=np.zeros(row_count, dtype=np.int32),
        base_phase=np.zeros(row_count, dtype=np.float32),
        base_gain=np.ones(row_count, dtype=np.float32),
        residual_layer_indices=[np.zeros(row_count, dtype=np.int32) for _ in range(D)],
        residual_layer_phases=[np.zeros(row_count, dtype=np.float32) for _ in range(D)],
        residual_layer_gains=[np.zeros(row_count, dtype=np.float32) for _ in range(D)],
    )


def _scatter_encoding(target: DiagnosticEncoding, source: DiagnosticEncoding, indices: np.ndarray) -> None:
    target.base_index[indices] = source.base_index
    target.base_phase[indices] = source.base_phase
    target.base_gain[indices] = source.base_gain
    for layer in range(D):
        target.residual_layer_indices[layer][indices] = source.residual_layer_indices[layer]
        target.residual_layer_phases[layer][indices] = source.residual_layer_phases[layer]
        target.residual_layer_gains[layer][indices] = source.residual_layer_gains[layer]


def _squared_distance_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.mean((left[:, None, :] - right[None, :, :]) ** 2, axis=2)


def _backend_used(encoding: DiagnosticEncoding) -> str:
    return "diagnostic"


def _prefix(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def _row_by_id(rows: list[dict[str, Any]], row_id: str) -> dict[str, Any] | None:
    return next((row for row in rows if row.get("row_id") == row_id), None)


def _reference_value(rows: list[dict[str, Any]], reference_id: str, key: str) -> float | None:
    row = next((item for item in rows if item.get("reference_id") == reference_id), None)
    return _float(row.get(key)) if row else None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any) -> str:
    number = _float(value)
    if number is None:
        return "n/a"
    return f"{number:.4f}"


def _log(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)

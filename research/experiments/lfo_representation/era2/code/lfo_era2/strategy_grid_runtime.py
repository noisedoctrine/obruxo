"""Numerical execution helpers for the Experiment 13 strategy grid.

The module deliberately reuses Experiment 12's frozen W8D16 encoder and
alignment primitives.  Only offline dictionary construction and Experiment 13
diagnostics live here.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Sequence

import numpy as np

from . import component_ladder as x12
from .accelerator import BackendPreference
from .alignment import alignment_matrix, best_alignment
from .assets import ReconstructionAssets
from .curve import circular_shift
from .dataset import Era2CurveDataset
from .manifest import write_json, write_summary_csv
from .metrics import max_abs_error_per_curve, reconstruction_summary
from .strategy_grid_execution import BaseStage, OPTIMIZATION_VERSION, SampleProvenance


PROTOTYPE_ITERATIONS = 8
PROTOTYPE_TOLERANCE = 1e-6
PROTOTYPE_GAIN_FLOOR = 1e-3
CLUSTER_COUNT = 4
DIVERSE_PROPOSAL_COUNT = 6
MEANINGFUL_IMPROVEMENT = 1e-8
FINISH_VECTOR_CANDIDATE_CHUNK = 8
_FINISH_KERNEL_VERIFIED = False


@dataclass
class RowArtifacts:
    summary: dict[str, Any]
    atom_construction: list[dict[str, Any]]
    atom_assignments: list[dict[str, Any]]
    candidate_search_diagnostics: list[dict[str, Any]]
    slot_progression: list[dict[str, Any]]
    partial_codebook_validation: list[dict[str, Any]]
    layer_epsilon_quantiles: list[dict[str, Any]]
    slot_epsilon_quantiles: list[dict[str, Any]]
    epsilon_coverage: list[dict[str, Any]]
    retired_error_mass: list[dict[str, Any]]


class ExecutionRecorder:
    """Append interruption-safe stage timings and emit a completed CSV."""

    def __init__(self, row_dir: Path, spec: Any) -> None:
        self.path = Path(row_dir) / "execution_timing.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.spec = spec
        self.rows: list[dict[str, Any]] = []

    def start(self, stage: str, *, layer: int | None = None, slot: int | None = None) -> tuple[str, float, float]:
        return _now_utc(), time.perf_counter(), time.process_time()

    def finish(
        self,
        stage: str,
        token: tuple[str, float, float],
        *,
        layer: int | None = None,
        slot: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        started_at, wall_started, cpu_started = token
        row = {
            "experiment_phase": self.spec.experiment_phase,
            "row_id": self.spec.row_id,
            "stage": stage,
            "residual_layer": "" if layer is None else int(layer),
            "active_atom_slot": "" if slot is None else int(slot),
            "started_at_utc": started_at,
            "completed_at_utc": _now_utc(),
            "wall_elapsed_seconds": time.perf_counter() - wall_started,
            "process_cpu_seconds": time.process_time() - cpu_started,
            **(metadata or {}),
        }
        self.rows.append(row)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return row

    def write_csv(self) -> None:
        write_csv(self.path.with_name("execution_timing.csv"), self.rows)


def component_spec(spec: Any) -> x12.ComponentRowSpec:
    budget = int(str(spec.utility_candidate_budget or "CandidateBudget24").replace("CandidateBudget", ""))
    return x12.ComponentRowSpec(
        row_id=spec.row_id,
        components=(spec.construction_policy, spec.residual_population_policy),
        description=f"Experiment 13 {spec.experiment_phase} {spec.construction_policy}",
        scalar_schema="PhaseAndResidualGain",
        screening_variable="construction_policy",
        screening_value=spec.construction_policy,
        phase_enabled=True,
        residual_gain_enabled=True,
        beam_width=4,
        path_search_policy="Beam4Path",
        construction_policy="BestOverallRepair",
        topology_used_in_construction=bool(spec.topology_used_in_construction),
        max_utility_candidates=budget,
        utility_candidate_budget=f"CandidateBudget{budget}",
        layer_normalization_policy=spec.layer_normalization_policy,
        no_damage_policy="NoDamageOff",
        atom_preprocessing_policy="RawAtoms",
        duplicate_suppression_policy="DuplicateSuppressionOff",
    )


def run_strategy_row(
    spec: Any,
    dataset: Era2CurveDataset,
    row_dir: Path,
    *,
    run_identity: str,
    configuration_fingerprint: str,
    backend: BackendPreference,
    chunk_size: int = 256,
    progress: Callable[[str], None] | None = None,
    base_stage: BaseStage | None = None,
    sample_provenance: SampleProvenance | None = None,
    cancel_check: Callable[[], None] | None = None,
    verify_optimized_kernels: bool = False,
) -> RowArtifacts:
    """Construct, encode, score, and persist one Experiment 13 row."""
    started = time.perf_counter()
    row_dir = Path(row_dir)
    row_dir.mkdir(parents=True, exist_ok=True)
    recorder = ExecutionRecorder(row_dir, spec)
    row_timer = recorder.start("row")
    train, validation = dataset.train_curves, dataset.validation_curves
    if not len(train) or not len(validation):
        raise ValueError("Experiment 13 requires non-empty training and validation splits")
    runtime_spec = component_spec(spec)
    phase_count = train.shape[1]
    if base_stage is None:
        base = x12._select_farthest_atoms(train, width=32, include_zero=False, topology=None)
        train_base = best_alignment(train, base, phase_policy="fft_lattice", gain_policy="fixed", backend=backend, chunk_size=chunk_size, phase_candidate_count=phase_count)
        validation_base = best_alignment(validation, base, phase_policy="fft_lattice", gain_policy="fixed", backend=backend, chunk_size=chunk_size, phase_candidate_count=phase_count)
        base_cache_hit = False
        base_cache_key = ""
    else:
        base = np.asarray(base_stage.base_dictionary, dtype=np.float32).copy()
        train_base = base_stage.train_alignment
        validation_base = base_stage.validation_alignment
        base_cache_hit = bool(base_stage.cache_hit)
        base_cache_key = base_stage.cache_key
    train_prefix = train_base.values.copy()
    validation_prefix = validation_base.values.copy()

    construction_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    slot_rows: list[dict[str, Any]] = []
    layer_quantiles: list[dict[str, Any]] = []
    slot_quantiles: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    retired_rows: list[dict[str, Any]] = []
    dictionaries: list[np.ndarray] = []

    _record_layer_calibration(spec, 0, "training", train, train_prefix, layer_quantiles, coverage_rows)
    _record_layer_calibration(spec, 0, "validation", validation, validation_prefix, layer_quantiles, coverage_rows)

    for layer_index in range(16):
        if cancel_check is not None:
            cancel_check()
        layer_timer = recorder.start("residual_layer", layer=layer_index + 1)
        _log(progress, f"construction: residual layer {layer_index + 1}/16")
        incoming = train - train_prefix
        atoms: list[np.ndarray] = [np.zeros(train.shape[1], dtype=np.float32)]
        previous_broad: list[np.ndarray] = []
        for slot_index in range(0, 8):
            if cancel_check is not None:
                cancel_check()
            slot_timer = recorder.start("active_atom_slot", layer=layer_index + 1, slot=slot_index)
            partial = np.stack(atoms).astype(np.float32)
            choice = best_alignment(incoming, partial, phase_policy="fft_lattice", gain_policy="optimized", backend=backend, chunk_size=chunk_size, phase_candidate_count=phase_count)
            unexplained = incoming - choice.values
            max_error = np.max(np.abs(unexplained), axis=1)
            current_loss = np.mean(unexplained * unexplained, axis=1)
            eligible = _eligibility_mask(spec, max_error)
            _record_slot_calibration(spec, layer_index + 1, slot_index, incoming, unexplained, max_error, slot_quantiles, coverage_rows, retired_rows)
            if slot_index == 7:
                recorder.finish("layer_final_evaluation", slot_timer, layer=layer_index + 1)
                break
            before = _slot_snapshot(current_loss, max_error, eligible, spec)
            if not np.any(eligible):
                atom = np.zeros(train.shape[1], dtype=np.float32)
                detail = _empty_builder_detail("early_completion_no_eligible_residuals")
            else:
                layer_role, slot_role = _roles(spec, layer_index, slot_index + 1)
                if layer_role == "Broad":
                    atom, detail = _build_broad_atom(
                        str(spec.broad_atom_builder), incoming, eligible, current_loss, previous_broad,
                        backend=backend, chunk_size=chunk_size,
                    )
                    previous_broad.append(atom)
                else:
                    builder = str(spec.repair_atom_builder or spec.construction_policy)
                    atom, detail = _build_repair_atom(
                        builder, slot_role, incoming, eligible, current_loss, max_error,
                        topology=dataset.topology[dataset.train_indices],
                        budget=spec.effective_candidate_budget_by_slot[layer_index][slot_index],
                        finish_threshold=float(spec.finish_threshold), backend=backend, chunk_size=chunk_size,
                        verify_optimized_kernel=verify_optimized_kernels,
                    )
            atoms.append(atom.astype(np.float32))
            exact_duplicate = any(np.array_equal(atoms[-1], previous) for previous in atoms[:-1])
            next_choice = choice if exact_duplicate else best_alignment(incoming, np.stack(atoms), phase_policy="fft_lattice", gain_policy="optimized", backend=backend, chunk_size=chunk_size, phase_candidate_count=phase_count)
            next_unexplained = incoming - next_choice.values
            next_max = np.max(np.abs(next_unexplained), axis=1)
            next_loss = np.mean(next_unexplained * next_unexplained, axis=1)
            next_eligible = _eligibility_mask(spec, next_max)
            after = _slot_snapshot(next_loss, next_max, next_eligible, spec)
            layer_role, slot_role = _roles(spec, layer_index, slot_index + 1)
            row = {
                "experiment_phase": spec.experiment_phase, "row_id": spec.row_id, "pair_id": spec.pair_id,
                "residual_layer": layer_index + 1, "slot_index": slot_index + 1,
                "layer_role": layer_role, "slot_role": slot_role,
                "atom_source_kind": (
                    "no_op" if detail.get("prototype_seed_rule") == "early_completion_no_eligible_residuals"
                    else "observed_residual" if detail.get("source_index") is not None
                    else "synthesized_prototype"
                ),
                "effective_candidate_budget": spec.effective_candidate_budget_by_slot[layer_index][slot_index],
                "finish_threshold": spec.finish_threshold, "selected_eligibility_epsilon": spec.eligibility_epsilon,
                "eligibility_selection_rule_version": spec.eligibility_selection_rule_version,
                "eligible_residual_count_before": int(np.sum(eligible)), "eligible_residual_count_after": int(np.sum(next_eligible)),
                "resolved_lfo_rate_before": before["resolved_rate"], "resolved_lfo_rate_after": after["resolved_rate"],
                "newly_eligibility_resolved_lfo_count": int(np.sum(eligible & ~next_eligible)) if spec.experiment_phase == "13B" else 0,
                "newly_finish_threshold_lfo_count": int(np.sum((max_error > spec.finish_threshold) & (next_max <= spec.finish_threshold))),
                "training_median_rmse_before": before["median_rmse"], "training_median_rmse_after": after["median_rmse"],
                "training_p95_rmse_before": before["p95_rmse"], "training_p95_rmse_after": after["p95_rmse"],
                "assigned_residual_count": int(np.sum(next_choice.indices == slot_index + 1)),
                "atom_phase_scale_similarity_to_previous_max": _max_similarity(atom, atoms[1:-1]),
                "exact_duplicate_alignment_reused": exact_duplicate,
                **_counterfactual_slot(max_error, incoming, unexplained),
                **detail,
            }
            if detail.get("source_index") is not None:
                dataset_source = int(dataset.train_indices[int(detail["source_index"])])
                row["repair_source_dataset_index"] = dataset_source
                detail["repair_source_dataset_index"] = dataset_source
            row["newly_resolved_lfo_count"] = row["newly_eligibility_resolved_lfo_count"]
            construction_rows.append(row)
            candidate_rows.append({
                "experiment_phase": spec.experiment_phase, "row_id": spec.row_id, "pair_id": spec.pair_id,
                "residual_layer": layer_index + 1, "slot_index": slot_index + 1,
                "shortlist_rule": detail.get("repair_shortlist_rule", ""),
                "candidate_count": detail.get("candidate_count", 0),
                "selected_source_dataset_index": detail.get("repair_source_dataset_index", ""),
                "selected_utility_score": detail.get("repair_utility_score", ""),
                "builder": slot_role, "construction_loss": "mean_squared_error",
            })
            slot_rows.append({
                "experiment_phase": spec.experiment_phase, "row_id": spec.row_id, "pair_id": spec.pair_id,
                "residual_layer": layer_index + 1, "active_atom_slot": slot_index + 1,
                "eligible_residual_count": int(np.sum(next_eligible)),
                "training_median_rmse": after["median_rmse"], "training_p95_rmse": after["p95_rmse"],
                "training_max_abs_error_p95": float(np.quantile(next_max, 0.95)),
            })
            recorder.finish(
                "active_atom_slot", slot_timer, layer=layer_index + 1, slot=slot_index + 1,
                metadata={"exact_duplicate_alignment_reused": exact_duplicate},
            )
        dictionary = np.stack(atoms).astype(np.float32)
        dictionaries.append(dictionary)
        train_choice = best_alignment(incoming, dictionary, phase_policy="fft_lattice", gain_policy="optimized", backend=backend, chunk_size=chunk_size, phase_candidate_count=phase_count)
        train_prefix = x12._apply_layer_state_policy(train_prefix, train_choice.values, runtime_spec)
        validation_incoming = validation - validation_prefix
        validation_choice = best_alignment(validation_incoming, dictionary, phase_policy="fft_lattice", gain_policy="optimized", backend=backend, chunk_size=chunk_size, phase_candidate_count=validation.shape[1])
        validation_prefix = x12._apply_layer_state_policy(validation_prefix, validation_choice.values, runtime_spec)
        _record_layer_calibration(spec, layer_index + 1, "training", train, train_prefix, layer_quantiles, coverage_rows)
        _record_layer_calibration(spec, layer_index + 1, "validation", validation, validation_prefix, layer_quantiles, coverage_rows)
        recorder.finish("residual_layer", layer_timer, layer=layer_index + 1)

    assets = ReconstructionAssets(base, dictionaries, metadata={"experiment_id": "experiment_13", "row_id": spec.row_id})
    train_timer = recorder.start("train_encoding")
    train_encoding, train_reconstructed, train_raw, train_encoding_time = x12._encode_decode(runtime_spec, train, assets, backend=backend, chunk_size=chunk_size, progress=progress, progress_label="train")
    recorder.finish("train_encoding", train_timer)
    validation_timer = recorder.start("validation_encoding")
    validation_encoding, validation_reconstructed, validation_raw, validation_encoding_time = x12._encode_decode(runtime_spec, validation, assets, backend=backend, chunk_size=chunk_size, progress=progress, progress_label="validation")
    recorder.finish("validation_encoding", validation_timer)
    partial_timer = recorder.start("partial_validation")
    partial_rows = _partial_codebook_rows(spec, runtime_spec, dataset, assets, backend=backend, chunk_size=chunk_size)
    recorder.finish("partial_validation", partial_timer)
    manifest = {
        **spec.manifest_dict(run_identity, configuration_fingerprint),
        **dataset.manifest_fields(),
        "codebook_storage_count": assets.codebook_storage_count,
        "oracle_construction_time": time.perf_counter() - started,
        "train_encoding_time": train_encoding_time,
        "validation_encoding_time": validation_encoding_time,
        "backend_preference": backend,
        "optimization_version": OPTIMIZATION_VERSION,
        "base_stage_cache_key": base_cache_key,
        "base_stage_cache_hit": base_cache_hit,
        **(sample_provenance.as_dict() if sample_provenance is not None else {}),
    }
    summary = {
        **manifest,
        **_prefix("train", reconstruction_summary(train, train_reconstructed)),
        **_prefix("validation", reconstruction_summary(validation, validation_reconstructed)),
        "validation_max_abs_error_p95": float(np.quantile(max_abs_error_per_curve(validation, validation_reconstructed), 0.95)),
        **_prefix("train", x12._overshoot_summary(train_raw)),
        **_prefix("validation", x12._overshoot_summary(validation_raw)),
        **x12._usage_summary(validation_encoding, widths=assets.residual_widths()),
        **x12._scalar_summary(validation_encoding, runtime_spec),
        **x12._asset_diagnostics(assets),
    }
    write_json(row_dir / "manifest.json", manifest)
    write_json(row_dir / "targets_schema.json", validation_encoding.target_schema(runtime_spec))
    write_summary_csv(row_dir / "summary.csv", summary)
    np.savez_compressed(row_dir / "codebooks.npz", base_dictionary=base, **{f"residual_layer_{i + 1}": value for i, value in enumerate(dictionaries)})
    tables = {
        "atom_construction.csv": construction_rows,
        "candidate_search_diagnostics.csv": candidate_rows, "slot_progression.csv": slot_rows,
        "partial_codebook_validation.csv": partial_rows, "layer_epsilon_quantiles.csv": layer_quantiles,
        "slot_epsilon_quantiles.csv": slot_quantiles, "epsilon_coverage.csv": coverage_rows,
        "retired_error_mass.csv": retired_rows,
    }
    for name, rows in tables.items():
        write_csv(row_dir / name, rows)
    _write_assignment_csv(row_dir / "atom_assignments.csv", spec, dataset, train_encoding, validation_encoding)
    recorder.finish("row", row_timer, metadata={"base_stage_cache_hit": base_cache_hit})
    recorder.write_csv()
    return RowArtifacts(summary, construction_rows, [], candidate_rows, slot_rows, partial_rows, layer_quantiles, slot_quantiles, coverage_rows, retired_rows)


def load_row_artifacts(row_dir: Path) -> RowArtifacts:
    return RowArtifacts(
        read_one_csv(Path(row_dir) / "summary.csv"),
        read_csv(Path(row_dir) / "atom_construction.csv"),
        read_csv(Path(row_dir) / "atom_assignments.csv"),
        read_csv(Path(row_dir) / "candidate_search_diagnostics.csv"),
        read_csv(Path(row_dir) / "slot_progression.csv"),
        read_csv(Path(row_dir) / "partial_codebook_validation.csv"),
        read_csv(Path(row_dir) / "layer_epsilon_quantiles.csv"),
        read_csv(Path(row_dir) / "slot_epsilon_quantiles.csv"),
        read_csv(Path(row_dir) / "epsilon_coverage.csv"),
        read_csv(Path(row_dir) / "retired_error_mass.csv"),
    )


def _roles(spec: Any, layer_index: int, active_slot: int) -> tuple[str, str]:
    if spec.layer_schedule == "AnchorNative":
        assert spec.native_slot_roles is not None
        return "AnchorNative", str(spec.native_slot_roles[active_slot - 1])
    role = str(spec.layer_schedule)
    if role == "Interleaved":
        layer_role = "Broad" if (layer_index + 1) % 2 else "Repair"
    elif role == "TwoPhase":
        layer_role = "Broad" if layer_index < 8 else "Repair"
    else:
        layer_role = "Broad"
    return layer_role, str(spec.broad_atom_builder if layer_role == "Broad" else spec.repair_atom_builder)


def _eligibility_mask(spec: Any, max_error: np.ndarray) -> np.ndarray:
    if spec.experiment_phase == "13A":
        return np.ones(len(max_error), dtype=bool)
    return np.asarray(max_error > float(spec.eligibility_epsilon), dtype=bool)


def _build_broad_atom(
    builder: str, residual: np.ndarray, eligible: np.ndarray, current_loss: np.ndarray,
    previous: Sequence[np.ndarray], *, backend: BackendPreference, chunk_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    indices = np.flatnonzero(eligible)
    matrix = np.asarray(residual[indices], dtype=np.float32)
    weights = np.maximum(current_loss[indices], 1e-12).astype(np.float64)
    if builder == "ClusterMean":
        proposals, sizes = _cluster_prototypes(matrix, weights, backend=backend, chunk_size=chunk_size)
        atom, score, choice = _best_proposal(matrix, proposals, weights, previous, diverse=False, backend=backend, chunk_size=chunk_size)
        detail = _prototype_detail(len(matrix), 1, True, "deterministic_phase_invariant_farthest_cluster", score)
        detail.update(cluster_count=len(proposals), cluster_size=sizes[choice])
        return atom, detail
    if builder == "DiverseCoverage":
        proposals = _partition_prototypes(matrix, weights, backend=backend, chunk_size=chunk_size)
        atom, score, _ = _best_proposal(matrix, proposals, weights, previous, diverse=True, backend=backend, chunk_size=chunk_size)
        detail = _prototype_detail(len(matrix), 1, True, "deterministic_weight_quantile_partition", score)
        return atom, detail
    atom, iterations, converged, objective_before, objective_after, executed = _alternating_prototype(
        matrix, weights, builder, backend=backend, chunk_size=chunk_size,
    )
    detail = _prototype_detail(len(matrix), iterations, converged, "highest_weight_canonical_residual", objective_after)
    detail["prototype_objective_before"] = objective_before
    detail["explained_weighted_energy"] = max(0.0, 1.0 - objective_after / objective_before) if objective_before > 0.0 else 0.0
    detail["prototype_iterations_executed"] = executed
    return atom, detail


def _alternating_prototype(
    matrix: np.ndarray, weights: np.ndarray, builder: str, *, backend: BackendPreference, chunk_size: int,
) -> tuple[np.ndarray, int, bool, float, float, int]:
    seed = int(np.argmax(weights))
    atom = _canonical_sign(matrix[seed].copy())
    objective_before = float(np.average(np.mean(matrix * matrix, axis=1), weights=weights))
    objective_after = objective_before
    converged = False
    states = [atom.copy()]
    objectives_by_state: list[float] = []
    executed = 0
    for iteration in range(1, PROTOTYPE_ITERATIONS + 1):
        executed += 1
        fit = best_alignment(matrix, np.stack([atom]), phase_policy="fft_lattice", gain_policy="optimized", backend=backend, chunk_size=chunk_size, phase_candidate_count=matrix.shape[1])
        canonical = circular_shift(matrix, -fit.phases)
        use = np.arange(len(matrix))
        if builder == "TrimmedMean" and len(use) > 1:
            keep = max(1, int(np.ceil(0.9 * len(use))))
            use = np.argsort(fit.losses, kind="stable")[:keep]
        if builder == "AlignedMedian":
            gain = np.maximum(np.abs(fit.gains[use]), PROTOTYPE_GAIN_FLOOR)
            sign = np.sign(np.where(np.abs(fit.gains[use]) > 1e-12, fit.gains[use], 1.0))
            normalized = canonical[use] * sign[:, None] / gain[:, None]
            updated = _weighted_median_matrix(normalized, weights[use])
        elif builder == "DominantDirection":
            aligned = canonical * np.sign(np.where(np.abs(fit.gains) > 1e-12, fit.gains, 1.0))[:, None]
            covariance = (aligned * weights[:, None]).T @ aligned
            values, vectors = np.linalg.eigh(covariance)
            updated = vectors[:, int(np.argmax(values))]
            target_rms = float(np.median(np.sqrt(np.mean(matrix * matrix, axis=1))))
            updated = updated / max(float(np.sqrt(np.mean(updated * updated))), 1e-12) * target_rms
        else:
            gains = fit.gains[use].astype(np.float64)
            numerator = np.sum(canonical[use] * (weights[use] * gains)[:, None], axis=0)
            denominator = float(np.sum(weights[use] * gains * gains))
            updated = numerator / denominator if denominator > 1e-15 else atom
        updated = _canonical_sign(np.asarray(updated, dtype=np.float32))
        change = float(np.sqrt(np.mean((updated - atom) ** 2)))
        atom = updated
        objective_after = float(np.average(fit.losses, weights=weights))
        objectives_by_state.append(objective_after)
        if change <= PROTOTYPE_TOLERANCE:
            converged = True
            break
        repeated = next((index for index, state in enumerate(states) if np.array_equal(updated, state)), None)
        states.append(updated.copy())
        if repeated is not None and iteration < PROTOTYPE_ITERATIONS:
            cycle = iteration - repeated

            def mapped_state(target: int) -> int:
                if target < repeated:
                    return target
                return repeated + ((target - repeated) % cycle)

            atom = states[mapped_state(PROTOTYPE_ITERATIONS)].copy()
            objective_after = objectives_by_state[mapped_state(PROTOTYPE_ITERATIONS - 1)]
            iteration = PROTOTYPE_ITERATIONS
            break
    return atom.astype(np.float32), iteration, converged, objective_before, objective_after, executed


def _cluster_prototypes(matrix: np.ndarray, weights: np.ndarray, *, backend: BackendPreference, chunk_size: int) -> tuple[list[np.ndarray], list[int]]:
    count = min(CLUSTER_COUNT, len(matrix))
    seeds = [int(np.argmax(weights))]
    while len(seeds) < count:
        distances = alignment_matrix(matrix, matrix[seeds], phase_policy="fft_lattice", gain_policy="optimized", backend=backend, chunk_size=chunk_size, phase_candidate_count=matrix.shape[1]).losses
        farthest = np.min(distances, axis=1)
        farthest[seeds] = -np.inf
        seeds.append(int(np.argmax(farthest)))
    distances = alignment_matrix(matrix, matrix[seeds], phase_policy="fft_lattice", gain_policy="optimized", backend=backend, chunk_size=chunk_size, phase_candidate_count=matrix.shape[1]).losses
    labels = np.argmin(distances, axis=1)
    proposals, sizes = [], []
    minimum = max(1, int(np.ceil(0.02 * len(matrix))))
    for cluster in range(count):
        members = np.flatnonzero(labels == cluster)
        if len(members) < minimum:
            continue
        atom, *_ = _alternating_prototype(matrix[members], weights[members], "BroadMean", backend=backend, chunk_size=chunk_size)
        proposals.append(atom)
        sizes.append(len(members))
    if not proposals:
        atom, *_ = _alternating_prototype(matrix, weights, "BroadMean", backend=backend, chunk_size=chunk_size)
        return [atom], [len(matrix)]
    return proposals, sizes


def _partition_prototypes(matrix: np.ndarray, weights: np.ndarray, *, backend: BackendPreference, chunk_size: int) -> list[np.ndarray]:
    order = np.argsort(weights, kind="stable")
    proposals = []
    for members in np.array_split(order, min(DIVERSE_PROPOSAL_COUNT, len(order))):
        if len(members):
            atom, *_ = _alternating_prototype(matrix[members], weights[members], "BroadMean", backend=backend, chunk_size=chunk_size)
            proposals.append(atom)
    return proposals


def _best_proposal(
    matrix: np.ndarray, proposals: Sequence[np.ndarray], weights: np.ndarray, previous: Sequence[np.ndarray],
    *, diverse: bool, backend: BackendPreference, chunk_size: int,
) -> tuple[np.ndarray, float, int]:
    codes = np.stack(proposals).astype(np.float32)
    losses = alignment_matrix(matrix, codes, phase_policy="fft_lattice", gain_policy="optimized", backend=backend, chunk_size=chunk_size, phase_candidate_count=matrix.shape[1]).losses
    baseline = np.mean(matrix * matrix, axis=1)
    improvement = np.maximum(baseline[:, None] - losses, 0.0)
    scores = np.sum(improvement * weights[:, None], axis=0)
    if diverse:
        threshold = np.maximum(MEANINGFUL_IMPROVEMENT, baseline[:, None] * 0.01)
        scores += np.sum(improvement >= threshold, axis=0) * max(float(np.max(scores)), 1.0) / max(1, len(matrix))
        if previous:
            scores -= np.asarray([_max_similarity(code, previous) for code in codes]) * max(float(np.max(scores)), 1.0) * 0.25
    choice = int(np.argmax(scores))
    return codes[choice], float(scores[choice]), choice


def _build_repair_atom(
    builder: str, slot_role: str, residual: np.ndarray, eligible: np.ndarray, current_loss: np.ndarray,
    current_max: np.ndarray, *, topology: np.ndarray, budget: int | None, finish_threshold: float,
    backend: BackendPreference, chunk_size: int, verify_optimized_kernel: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    eligible_indices = np.flatnonzero(eligible)
    limit = max(1, int(budget or 24))
    local_order = np.argsort(current_loss[eligible_indices], kind="stable")[::-1]
    if slot_role in {"finish", "FinishRepair"}:
        local_order = np.argsort(current_max[eligible_indices], kind="stable")
    shortlist = eligible_indices[local_order[:limit]]
    candidates = np.asarray(residual[shortlist], dtype=np.float32)
    target = np.asarray(residual, dtype=np.float32)
    fit = alignment_matrix(target, candidates, phase_policy="fft_lattice", gain_policy="optimized", backend=backend, chunk_size=chunk_size, phase_candidate_count=target.shape[1])
    improvement = np.maximum(current_loss[:, None] - fit.losses, 0.0)
    weights = np.maximum(current_loss, 1e-12)
    masked = improvement * eligible[:, None] * weights[:, None]
    role = slot_role.lower()
    if builder == "FinishRepair" or role == "finish":
        finish_counts = _finish_counts_vectorized(
            target, candidates, fit.phases, fit.gains, eligible, current_max, finish_threshold,
            candidate_chunk=FINISH_VECTOR_CANDIDATE_CHUNK,
        )
        global _FINISH_KERNEL_VERIFIED
        if verify_optimized_kernel and not _FINISH_KERNEL_VERIFIED:
            legacy = _finish_counts_legacy(
                target, candidates, fit.phases, fit.gains, eligible, current_max, finish_threshold,
            )
            if not np.array_equal(finish_counts, legacy):
                raise RuntimeError("vectorized FinishRepair failed exact first-use verification")
            _FINISH_KERNEL_VERIFIED = True
        total = np.sum(masked, axis=0)
        choice = int(max(range(len(candidates)), key=lambda index: (finish_counts[index], total[index], -index)))
        score = float(finish_counts[choice] * 1_000_000.0 + total[choice])
    elif builder == "HardRepair" or role in {"hard", "rescue"}:
        threshold = float(np.quantile(current_loss[eligible], 0.90))
        hard = eligible & (current_loss >= threshold)
        hard_score = np.sum(improvement * hard[:, None] * weights[:, None], axis=0)
        total = np.sum(masked, axis=0)
        choice = int(max(range(len(candidates)), key=lambda index: (hard_score[index], total[index], -index)))
        score = float(hard_score[choice])
    elif builder == "CommonCaseRepair" or role == "common":
        eligible_improvement = improvement[eligible]
        score_values = np.median(eligible_improvement, axis=0) * len(eligible_improvement) + np.sum(np.minimum(eligible_improvement, np.median(current_loss[eligible])), axis=0)
        choice = int(np.argmax(score_values)); score = float(score_values[choice])
    elif builder == "FamilyBalancedRepair":
        family_scores = []
        for family in np.unique(topology[eligible]):
            mask = eligible & (topology == family)
            family_scores.append(np.mean(improvement[mask], axis=0) if np.any(mask) else np.zeros(len(candidates)))
        score_values = np.mean(np.stack(family_scores), axis=0)
        choice = int(np.argmax(score_values)); score = float(score_values[choice])
    else:
        score_values = np.sum(masked, axis=0)
        choice = int(np.argmax(score_values)); score = float(score_values[choice])
    detail = _empty_builder_detail("")
    detail.update(
        source_index=int(shortlist[choice]), repair_source_dataset_index=int(shortlist[choice]),
        repair_shortlist_rule="eligible_error_rank_stable", repair_utility_score=score,
        candidate_count=len(candidates), prototype_population_size=0,
    )
    return candidates[choice].astype(np.float32), detail


def _finish_counts_legacy(
    target: np.ndarray,
    candidates: np.ndarray,
    phases: np.ndarray,
    gains: np.ndarray,
    eligible: np.ndarray,
    current_max: np.ndarray,
    finish_threshold: float,
) -> np.ndarray:
    counts = np.zeros(len(candidates), dtype=np.int64)
    finishable = eligible & (current_max > finish_threshold)
    for candidate in range(len(candidates)):
        values = circular_shift(
            np.repeat(candidates[candidate][None, :], len(target), axis=0), phases[:, candidate]
        ) * gains[:, candidate, None]
        candidate_max = np.max(np.abs(target - values), axis=1)
        counts[candidate] = int(np.sum(finishable & (candidate_max <= finish_threshold)))
    return counts


def _finish_counts_vectorized(
    target: np.ndarray,
    candidates: np.ndarray,
    phases: np.ndarray,
    gains: np.ndarray,
    eligible: np.ndarray,
    current_max: np.ndarray,
    finish_threshold: float,
    *,
    candidate_chunk: int = FINISH_VECTOR_CANDIDATE_CHUNK,
) -> np.ndarray:
    counts = np.zeros(len(candidates), dtype=np.int64)
    finishable = eligible & (current_max > finish_threshold)
    for start in range(0, len(candidates), max(1, int(candidate_chunk))):
        stop = min(len(candidates), start + max(1, int(candidate_chunk)))
        code_block = np.broadcast_to(
            candidates[None, start:stop, :], (len(target), stop - start, target.shape[1])
        )
        shifted = circular_shift(code_block, phases[:, start:stop])
        values = shifted * gains[:, start:stop, None]
        candidate_max = np.max(np.abs(target[:, None, :] - values), axis=2)
        counts[start:stop] = np.sum(
            finishable[:, None] & (candidate_max <= finish_threshold), axis=0, dtype=np.int64
        )
    return counts


def _record_layer_calibration(spec: Any, layer: int, split: str, targets: np.ndarray, reconstructed: np.ndarray, quantiles: list[dict[str, Any]], coverage: list[dict[str, Any]]) -> None:
    errors = max_abs_error_per_curve(targets, np.clip(reconstructed, 0.0, 1.0))
    for percentile in _percentiles(len(errors)):
        quantiles.append({"experiment_phase": spec.experiment_phase, "row_id": spec.row_id, "pair_id": spec.pair_id, "dataset_split": split, "residual_layer": layer, "percentile": percentile, "epsilon_value": float(np.quantile(errors, percentile)), "sample_count": len(errors)})
    for epsilon in (0.001, 0.0025, 0.005, 0.01, 0.02):
        resolved = int(np.sum(errors <= epsilon))
        coverage.append(_coverage_row(spec, split, layer, None, epsilon, resolved, len(errors)))


def _record_slot_calibration(spec: Any, layer: int, slot: int, incoming: np.ndarray, unexplained: np.ndarray, errors: np.ndarray, quantiles: list[dict[str, Any]], coverage: list[dict[str, Any]], retired: list[dict[str, Any]]) -> None:
    for percentile in _percentiles(len(errors)):
        quantiles.append({"experiment_phase": spec.experiment_phase, "row_id": spec.row_id, "pair_id": spec.pair_id, "residual_layer": layer, "active_atom_slot": slot, "percentile": percentile, "epsilon_value": float(np.quantile(errors, percentile)), "sample_count": len(errors)})
    incoming_energy = np.sum(incoming * incoming, axis=1)
    unexplained_energy = np.sum(unexplained * unexplained, axis=1)
    incoming_total, unexplained_total = float(np.sum(incoming_energy)), float(np.sum(unexplained_energy))
    for epsilon in (0.001, 0.0025, 0.005, 0.01, 0.02):
        mask = errors <= epsilon
        resolved = int(np.sum(mask))
        coverage.append(_coverage_row(spec, "training", layer, slot, epsilon, resolved, len(errors)))
        incoming_retired = float(np.sum(incoming_energy[mask])); unexplained_retired = float(np.sum(unexplained_energy[mask]))
        zero = unexplained_total <= 0.0
        retired.append({
            "experiment_phase": spec.experiment_phase, "row_id": spec.row_id, "pair_id": spec.pair_id,
            "residual_layer": layer, "active_atom_slot": slot, "epsilon": epsilon,
            "retired_lfo_count": resolved, "retired_lfo_fraction": resolved / len(errors),
            "incoming_retired_energy": incoming_retired,
            "incoming_retired_energy_fraction": incoming_retired / incoming_total if incoming_total > 0.0 else 0.0,
            "unexplained_retired_energy": unexplained_retired,
            "unexplained_retired_energy_fraction": unexplained_retired / unexplained_total if not zero else 0.0,
            "retained_unexplained_energy_fraction": 1.0 - (unexplained_retired / unexplained_total if not zero else 0.0),
            "zero_total_energy": zero,
        })


def _partial_codebook_rows(spec: Any, runtime_spec: x12.ComponentRowSpec, dataset: Era2CurveDataset, assets: ReconstructionAssets, *, backend: BackendPreference, chunk_size: int) -> list[dict[str, Any]]:
    rows = []
    for active_count in range(1, 8):
        partial_assets = ReconstructionAssets(assets.base_dictionary, [layer[: active_count + 1] for layer in assets.residual_layer_dictionaries])
        _, reconstructed, _, _ = x12._encode_decode(runtime_spec, dataset.validation_curves, partial_assets, backend=backend, chunk_size=chunk_size, progress=None, progress_label="partial_validation")
        metrics = reconstruction_summary(dataset.validation_curves, reconstructed)
        rows.append({
            "experiment_phase": spec.experiment_phase, "row_id": spec.row_id, "pair_id": spec.pair_id,
            "finish_threshold": spec.finish_threshold, "selected_eligibility_epsilon": spec.eligibility_epsilon,
            "active_atom_count": active_count,
            "validation_median_rmse": metrics["median_rmse"], "validation_strict_perfect_lfo_rate": metrics["strict_perfect_lfo_rate"],
            "validation_p95_rmse": metrics["p95_rmse"], "validation_node_max_error_p95": metrics["node_max_error_p95"],
        })
    return rows


def _write_assignment_csv(path: Path, spec: Any, dataset: Era2CurveDataset, train: x12.ComponentEncoding, validation: x12.ComponentEncoding) -> None:
    fields = ("experiment_phase", "row_id", "pair_id", "dataset_split", "dataset_index", "residual_layer", "atom_index", "phase", "gain")
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields)); writer.writeheader()
        for split, dataset_indices, encoding in (("training", dataset.train_indices, train), ("validation", dataset.validation_indices, validation)):
            for layer in range(16):
                for local, dataset_index in enumerate(dataset_indices):
                    writer.writerow({
                        "experiment_phase": spec.experiment_phase, "row_id": spec.row_id, "pair_id": spec.pair_id,
                        "dataset_split": split, "dataset_index": int(dataset_index), "residual_layer": layer + 1,
                        "atom_index": int(encoding.residual_layer_indices[layer][local]),
                        "phase": float(encoding.residual_layer_phases[layer][local]), "gain": float(encoding.residual_layer_gains[layer][local]),
                    })
    tmp.replace(path)


def _coverage_row(spec: Any, split: str, layer: int, slot: int | None, epsilon: float, resolved: int, count: int) -> dict[str, Any]:
    return {"experiment_phase": spec.experiment_phase, "row_id": spec.row_id, "pair_id": spec.pair_id, "dataset_split": split, "residual_layer": layer, "active_atom_slot": slot, "epsilon": epsilon, "resolved_count": resolved, "resolved_fraction": resolved / count, "counterfactual_eligible_count": count - resolved, "counterfactual_eligible_fraction": (count - resolved) / count}


def _slot_snapshot(loss: np.ndarray, max_error: np.ndarray, eligible: np.ndarray, spec: Any) -> dict[str, float]:
    rmse = np.sqrt(np.maximum(loss, 0.0))
    resolved = float(np.mean(~eligible)) if spec.experiment_phase == "13B" else 0.0
    return {"median_rmse": float(np.median(rmse)), "p95_rmse": float(np.quantile(rmse, 0.95)), "resolved_rate": resolved, "max_error_p95": float(np.quantile(max_error, 0.95))}


def _prototype_detail(population: int, iterations: int, converged: bool, seed_rule: str, objective: float) -> dict[str, Any]:
    detail = _empty_builder_detail("")
    detail.update(prototype_iteration_count=iterations, prototype_iterations_executed=iterations, prototype_converged=converged, prototype_population_size=population, prototype_objective_after=objective, prototype_seed_rule=seed_rule)
    return detail


def _counterfactual_slot(max_error: np.ndarray, incoming: np.ndarray, unexplained: np.ndarray) -> dict[str, str]:
    incoming_energy = np.sum(incoming * incoming, axis=1)
    unexplained_energy = np.sum(unexplained * unexplained, axis=1)
    incoming_total = float(np.sum(incoming_energy))
    unexplained_total = float(np.sum(unexplained_energy))
    resolved, incoming_fraction, unexplained_fraction = {}, {}, {}
    for epsilon in (0.001, 0.0025, 0.005, 0.01, 0.02):
        mask = max_error <= epsilon
        key = f"{epsilon:g}"
        resolved[key] = float(np.mean(mask))
        incoming_fraction[key] = float(np.sum(incoming_energy[mask]) / incoming_total) if incoming_total > 0.0 else 0.0
        unexplained_fraction[key] = float(np.sum(unexplained_energy[mask]) / unexplained_total) if unexplained_total > 0.0 else 0.0
    return {
        "counterfactual_resolved_fraction_by_candidate_epsilon": json.dumps(resolved, sort_keys=True),
        "counterfactual_incoming_retired_energy_fraction_by_candidate_epsilon": json.dumps(incoming_fraction, sort_keys=True),
        "counterfactual_unexplained_retired_energy_fraction_by_candidate_epsilon": json.dumps(unexplained_fraction, sort_keys=True),
    }


def _empty_builder_detail(reason: str) -> dict[str, Any]:
    return {"prototype_iteration_count": 0, "prototype_iterations_executed": 0, "prototype_converged": False, "prototype_population_size": 0, "prototype_objective_before": "", "prototype_objective_after": "", "prototype_seed_rule": reason, "cluster_count": 0, "cluster_size": 0, "explained_weighted_energy": "", "repair_source_dataset_index": "", "repair_shortlist_rule": "", "repair_utility_score": "", "source_index": None, "candidate_count": 0}


def _weighted_median_matrix(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    order = np.argsort(values, axis=0, kind="stable")
    sorted_values = np.take_along_axis(values, order, axis=0)
    sorted_weights = np.take_along_axis(np.broadcast_to(weights[:, None], values.shape), order, axis=0)
    cumulative = np.cumsum(sorted_weights, axis=0)
    threshold = np.sum(sorted_weights, axis=0) * 0.5
    index = np.argmax(cumulative >= threshold[None, :], axis=0)
    return sorted_values[index, np.arange(values.shape[1])]


def _canonical_sign(atom: np.ndarray) -> np.ndarray:
    value = np.asarray(atom, dtype=np.float32)
    pivot = int(np.argmax(np.abs(value)))
    value = np.roll(value, -pivot)
    return (-value if value[0] < 0.0 else value).astype(np.float32)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    norm = np.sqrt(np.sum(values * values, axis=1, keepdims=True))
    return np.divide(values, norm, out=np.zeros_like(values), where=norm > 1e-12)


def _max_similarity(atom: np.ndarray, previous: Sequence[np.ndarray]) -> float:
    if not previous:
        return 0.0
    left = _normalize_rows(np.asarray(atom)[None, :])[0]
    best = 0.0
    for other in previous:
        right = _normalize_rows(np.asarray(other)[None, :])[0]
        best = max(best, max(abs(float(np.dot(left, np.roll(right, shift)))) for shift in range(len(left))))
    return best


def _percentiles(count: int) -> tuple[float, ...]:
    values = (0.50, 0.25, 0.10, 0.05, 0.02, 0.01)
    return values + ((0.001,) if count >= 1000 else ())


def _prefix(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fieldnames or _fieldnames(rows))
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    tmp.replace(path)


def merge_csv_files(path: Path, sources: Sequence[Path]) -> None:
    """Merge equal-schema row artifacts without materializing them in memory."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    fieldnames: list[str] | None = None
    with tmp.open("w", encoding="utf-8", newline="") as output:
        writer: csv.DictWriter[str] | None = None
        for source in sources:
            if not Path(source).exists():
                continue
            with Path(source).open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                names = list(reader.fieldnames or [])
                if not names:
                    continue
                if fieldnames is None:
                    fieldnames = names
                    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
                    writer.writeheader()
                elif names != fieldnames:
                    raise ValueError(f"incompatible CSV schema while merging {source}")
                assert writer is not None
                writer.writerows(reader)
        if fieldnames is None:
            writer = csv.DictWriter(output, fieldnames=["row_id"])
            writer.writeheader()
    tmp.replace(path)


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_one_csv(path: Path) -> dict[str, Any]:
    rows = read_csv(path)
    if len(rows) != 1:
        raise ValueError(f"expected exactly one row in {path}")
    return rows[0]


def _fieldnames(rows: Sequence[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for row in rows:
        for name in row:
            if name not in names:
                names.append(name)
    return names or ["row_id"]


def _log(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()

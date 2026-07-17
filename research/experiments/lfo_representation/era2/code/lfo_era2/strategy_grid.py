"""Experiment 13 execution for the fixed-W8D16 strategy grid."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any, Callable, Literal, Sequence
from uuid import uuid4

from .strategy_grid_execution import (
    OPTIMIZATION_VERSION,
    SampleProvenance,
    deterministic_sample,
    implementation_fingerprint,
    load_dataset_cached,
    load_or_build_base_stage,
    sample_manifest_matches,
    write_sample_artifacts,
)

ExperimentPhase = Literal["13A", "13B"]
PhaseState = Literal["not_started", "running", "partial", "blocked", "failed", "cancelled", "complete"]

EXPERIMENT_ID = "experiment_13"
SCHEMA_VERSION = "experiment13_strategy_grid_v1"
SELECTION_RULE_VERSION = "experiment13_epsilon_selection_v1"
BASE_DICTIONARY_SIZE = 32
W = 8
D = 16
CONTROL_POINT_COUNT = 97
RESERVED_ATOM = "NoOpAtom"
ACTIVE_ATOMS_PER_LAYER = 7
PHASE_EXPECTED_ROW_COUNTS = {"13A": 90, "13B": 135}
PHASE_NORMALIZATION_POLICIES = {
    "13A": ("FinalClipOnly", "LayerClip0To1"),
    "13B": ("LayerClip0To1",),
}
EXPERIMENT13B_ELIGIBILITY_EPSILONS = (0.01, 0.001, 0.0001)
EXPERIMENT13B_SWEEP_VERSION = "experiment13b_fixed_epsilon_sweep_v1"
SCALAR_SCHEMA = "PhaseAndResidualGain"
PATH_SEARCH_POLICY = "Beam4Path"
NO_DAMAGE_POLICY = "NoDamageOff"
ATOM_PREPROCESSING_POLICY = "RawAtoms"
DUPLICATE_SUPPRESSION_POLICY = "DuplicateSuppressionOff"
FINISH_THRESHOLD = 1e-5
HEAD_OUTPUTS = 193
RUNTIME_TOPOLOGY = None
DEFAULT_SAMPLE_SEED = 13
OPTIMIZED_KERNEL_MODES = ("off", "first-use")
CANDIDATE_EPSILONS = (0.001, 0.0025, 0.005, 0.01, 0.02)
PILOT_EPSILONS = (0.001, 0.0025)
SELECTION_CHECKPOINT_DEFINITION = {
    "experiment_phase": "13A",
    "dataset_split": "training",
    "row_count": 90,
    "residual_layers": tuple(range(1, D + 1)),
    "decision_slots": tuple(range(0, 7)),
    "excluded_decision_slots": (7,),
    "early_middle_residual_layers": tuple(range(1, 13)),
    "early_middle_decision_slots": tuple(range(0, 6)),
}
PILOT_POLICIES = (
    "BroadMeanGlobalRepairInterleaved",
    "BroadMeanGlobalRepairTwoPhase",
    "ClusterMeanHardRepairTwoPhase",
    "FinishRepairRescue",
)
ANCHOR_SLOT_ROLES = {
    "CommonCaseRepair": ("common",) * ACTIVE_ATOMS_PER_LAYER,
    "FinishRepairRescue": ("finish", "finish", "common", "common", "common", "hard", "hard"),
    "FamilyBalancedRepair": ("overall",) * ACTIVE_ATOMS_PER_LAYER,
}

ERA2_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ERA2_ROOT.parents[3]
DEFAULT_METADATA = REPO_ROOT / "datasets" / "presetshare" / "raw" / "presetshare_vital_metadata.csv"
DEFAULT_OUTPUT_DIR = ERA2_ROOT / "artifacts" / "experiment_13" / "strategy_grid"
PHASE_STATUS_FILES = {"13A": "experiment13a_status.json", "13B": "experiment13b_status.json"}
REQUIRED_CALIBRATION_FILES = (
    "layer_epsilon_quantiles.csv",
    "slot_epsilon_quantiles.csv",
    "epsilon_coverage.csv",
    "retired_error_mass.csv",
)
REQUIRED_ANALYSIS_FILES = (
    "summary.csv",
    "strategy_results.csv",
    "slot_progression.csv",
    "partial_codebook_validation.csv",
    "atom_construction.csv",
    "atom_assignments.csv",
    "candidate_search_diagnostics.csv",
    "budget_accounting.csv",
    *REQUIRED_CALIBRATION_FILES,
    "epsilon_selection.json",
)
REQUIRED_ROW_FILES = (
    "manifest.json",
    "targets_schema.json",
    "summary.csv",
    "atom_construction.csv",
    "atom_assignments.csv",
    "candidate_search_diagnostics.csv",
    "slot_progression.csv",
    "partial_codebook_validation.csv",
    "layer_epsilon_quantiles.csv",
    "slot_epsilon_quantiles.csv",
    "epsilon_coverage.csv",
    "retired_error_mass.csv",
    "codebooks.npz",
    "execution_timing.jsonl",
    "execution_timing.csv",
)
REQUIRED_SELECTION_FIELDS = (
    "candidate_epsilons",
    "selection_rule_version",
    "selection_checkpoint_definition",
    "selected_epsilon",
    "training_statistics_used",
    "median_unexplained_retired_energy_fraction",
    "p95_unexplained_retired_energy_fraction",
    "retired_lfo_fraction_summary",
    "selection_timestamp",
    "experiment13a_run_identity",
    "configuration_fingerprint",
    "selection_passed",
    "selection_override",
    "selection_override_rationale",
    "selection_override_timestamp",
    "pilot_evidence",
    "selection_notes",
)


class Experiment13Error(RuntimeError):
    pass


class PhaseGateError(Experiment13Error):
    pass


class SelectionArtifactError(PhaseGateError):
    pass


class AnalysisNotReadyError(PhaseGateError):
    pass


class RunCancelled(Experiment13Error):
    pass


@dataclass(frozen=True)
class ConstructionPolicy:
    name: str
    family: str
    schedule: str
    broad_builder: str | None
    repair_builder: str | None
    observed_residual_value: bool
    repair_budget: bool


@dataclass(frozen=True)
class StrategyRowSpec:
    experiment_phase: ExperimentPhase
    row_id: str
    pair_id: str
    construction_policy: str
    construction_family: str
    layer_schedule: str
    residual_population_policy: str
    utility_candidate_budget: str | None
    layer_normalization_policy: str
    broad_atom_builder: str | None
    repair_atom_builder: str | None
    prototype_uses_observed_residual_value: bool
    native_slot_roles: tuple[str, ...] | None = None
    topology_used_in_construction: bool = False
    eligibility_epsilon: float | None = None
    eligibility_selection_rule_version: str | None = None
    finish_threshold: float = FINISH_THRESHOLD
    scalar_schema: str = SCALAR_SCHEMA
    path_search_policy: str = PATH_SEARCH_POLICY
    no_damage_policy: str = NO_DAMAGE_POLICY
    atom_preprocessing_policy: str = ATOM_PREPROCESSING_POLICY
    duplicate_suppression_policy: str = DUPLICATE_SUPPRESSION_POLICY
    base_dictionary_size: int = BASE_DICTIONARY_SIZE
    residual_width: int = W
    residual_depth: int = D
    control_point_count: int = CONTROL_POINT_COUNT
    reserved_atom: str = RESERVED_ATOM
    active_atoms_per_layer: int = ACTIVE_ATOMS_PER_LAYER
    runtime_topology: None = RUNTIME_TOPOLOGY
    head_outputs_actual: int = HEAD_OUTPUTS

    @property
    def paired_settings(self) -> tuple[Any, ...]:
        ignored = {
            "experiment_phase",
            "row_id",
            "residual_population_policy",
            "eligibility_epsilon",
            "eligibility_selection_rule_version",
        }
        values = asdict(self)
        return tuple(values[key] for key in values if key not in ignored)

    @property
    def effective_candidate_budget_by_layer(self) -> tuple[int | None, ...]:
        budget = _budget_value(self.utility_candidate_budget)
        if self.layer_schedule == "AnchorNative":
            return (budget,) * D
        return tuple(budget if role == "Repair" else None for role in layer_roles(self.layer_schedule))

    @property
    def effective_candidate_budget_by_slot(self) -> tuple[tuple[int | None, ...], ...]:
        return tuple((budget,) * ACTIVE_ATOMS_PER_LAYER for budget in self.effective_candidate_budget_by_layer)

    def manifest_dict(self, run_identity: str, fingerprint: str) -> dict[str, Any]:
        return {
            **asdict(self),
            "schema_version": SCHEMA_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "experiment13a_run_identity": run_identity,
            "configuration_fingerprint": fingerprint,
            "effective_candidate_budget_by_layer": list(self.effective_candidate_budget_by_layer),
            "effective_candidate_budget_by_slot": [list(row) for row in self.effective_candidate_budget_by_slot],
            "runtime_interface_id": "flat_categorical_per_residual_layer",
            "dictionary_scope": "per_residual_layer",
            "topology_used_at_runtime": False,
            "topology_used_in_targets": False,
            "topology_used_in_loss": False,
            "topology_used_in_decoder_lookup": False,
            "topology_used_in_head_accounting": False,
        }


@dataclass(frozen=True)
class EpsilonSelection:
    candidate_epsilons: tuple[float, ...]
    selection_rule_version: str
    selected_epsilon: float | None
    experiment13a_run_identity: str
    configuration_fingerprint: str
    selection_passed: bool
    selection_override: bool
    selection_override_rationale: Any
    selection_override_timestamp: Any
    pilot_evidence: Any
    payload: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EpsilonSelection":
        missing = [key for key in REQUIRED_SELECTION_FIELDS if key not in payload]
        if missing:
            raise SelectionArtifactError("epsilon selection artifact missing fields: " + ", ".join(missing))
        candidate_values = payload["candidate_epsilons"]
        if not isinstance(candidate_values, list) or not candidate_values or any(
            not _is_json_number(value) for value in candidate_values
        ):
            raise SelectionArtifactError("candidate_epsilons must be a non-empty JSON array of finite numbers")
        candidates = tuple(float(value) for value in candidate_values)
        selected_value = payload["selected_epsilon"]
        if selected_value is not None and not _is_json_number(selected_value):
            raise SelectionArtifactError("selected_epsilon must be null or a finite JSON number")
        selected = None if selected_value is None else float(selected_value)
        if type(payload["selection_passed"]) is not bool or type(payload["selection_override"]) is not bool:
            raise SelectionArtifactError("selection_passed and selection_override must be booleans")
        _validate_selection_metadata(payload)
        return cls(
            candidate_epsilons=candidates,
            selection_rule_version=str(payload["selection_rule_version"]),
            selected_epsilon=selected,
            experiment13a_run_identity=str(payload["experiment13a_run_identity"]),
            configuration_fingerprint=str(payload["configuration_fingerprint"]),
            selection_passed=payload["selection_passed"],
            selection_override=payload["selection_override"],
            selection_override_rationale=payload["selection_override_rationale"],
            selection_override_timestamp=payload["selection_override_timestamp"],
            pilot_evidence=payload["pilot_evidence"],
            payload=payload,
        )


def construction_policies() -> tuple[ConstructionPolicy, ...]:
    anchors = tuple(
        ConstructionPolicy(name, "Experiment12Anchor", "AnchorNative", None, name, True, True)
        for name in ("CommonCaseRepair", "FinishRepairRescue", "FamilyBalancedRepair")
    )
    recipes = (
        ("BroadMeanGlobalRepair", "BroadMean", "GlobalRepair"),
        ("BroadMeanFinishRepair", "BroadMean", "FinishRepair"),
        ("BroadMeanHardRepair", "BroadMean", "HardRepair"),
        ("TrimmedMeanGlobalRepair", "TrimmedMean", "GlobalRepair"),
        ("AlignedMedianGlobalRepair", "AlignedMedian", "GlobalRepair"),
        ("ClusterMeanGlobalRepair", "ClusterMean", "GlobalRepair"),
        ("ClusterMeanHardRepair", "ClusterMean", "HardRepair"),
        ("DominantDirectionGlobalRepair", "DominantDirection", "GlobalRepair"),
        ("DiverseCoverageHardRepair", "DiverseCoverage", "HardRepair"),
    )
    mixed = tuple(
        ConstructionPolicy(f"{family}{schedule}", family, schedule, broad, repair, False, True)
        for family, broad, repair in recipes
        for schedule in ("Interleaved", "TwoPhase")
    )
    pure = (
        ConstructionPolicy("AllBroadAlignedMeans", "PurePrototype", "AllBroad", "BroadMean", None, False, False),
        ConstructionPolicy("AllClusterMeans", "PurePrototype", "AllBroad", "ClusterMean", None, False, False),
        ConstructionPolicy(
            "AllDominantDirections", "PurePrototype", "AllBroad", "DominantDirection", None, False, False
        ),
    )
    return anchors + mixed + pure


def layer_roles(schedule: str) -> tuple[str, ...]:
    if schedule == "Interleaved":
        return tuple("Broad" if layer % 2 else "Repair" for layer in range(1, D + 1))
    if schedule == "TwoPhase":
        return ("Broad",) * 8 + ("Repair",) * 8
    if schedule == "AllBroad":
        return ("Broad",) * D
    if schedule == "AnchorNative":
        return ("AnchorNative",) * D
    raise ValueError(f"unsupported layer_schedule: {schedule}")


def experiment13a_specs() -> list[StrategyRowSpec]:
    return _phase_specs("13A", None)


def experiment13b_specs(
    eligibility_epsilons: float | Sequence[float] = EXPERIMENT13B_ELIGIBILITY_EPSILONS,
) -> list[StrategyRowSpec]:
    values = (
        (_validate_epsilon(eligibility_epsilons),)
        if isinstance(eligibility_epsilons, (int, float))
        else tuple(_validate_epsilon(value) for value in eligibility_epsilons)
    )
    if not values:
        raise ValueError("Experiment 13B requires at least one eligibility epsilon")
    if len(set(values)) != len(values):
        raise ValueError("Experiment 13B eligibility epsilons must be unique")
    rows_by_epsilon = [_phase_specs("13B", epsilon) for epsilon in values]
    rows = [
        rows_by_epsilon[epsilon_index][strategy_index]
        for strategy_index in range(45)
        for epsilon_index in range(len(values))
    ]
    expected = 45 * len(values)
    if len(rows) != expected or len({row.row_id for row in rows}) != expected:
        raise AssertionError(f"Experiment 13B sweep must contain {expected} unique rows")
    return rows


def all_strategy_specs(
    eligibility_epsilons: float | Sequence[float] = EXPERIMENT13B_ELIGIBILITY_EPSILONS,
) -> list[StrategyRowSpec]:
    return experiment13a_specs() + experiment13b_specs(eligibility_epsilons)


def _phase_specs(phase: ExperimentPhase, epsilon: float | None) -> list[StrategyRowSpec]:
    rows: list[StrategyRowSpec] = []
    for policy in construction_policies():
        budgets = ("CandidateBudget24", "CandidateBudget48") if policy.repair_budget else (None,)
        for budget in budgets:
            for normalization in PHASE_NORMALIZATION_POLICIES[phase]:
                pair_id = _pair_id(policy.name, budget, normalization)
                row = StrategyRowSpec(
                    experiment_phase=phase,
                    row_id=_row_id(phase, pair_id, epsilon),
                    pair_id=pair_id,
                    construction_policy=policy.name,
                    construction_family=policy.family,
                    layer_schedule=policy.schedule,
                    residual_population_policy="AllResiduals" if phase == "13A" else "UnresolvedOnly",
                    utility_candidate_budget=budget,
                    layer_normalization_policy=normalization,
                    broad_atom_builder=policy.broad_builder,
                    repair_atom_builder=policy.repair_builder,
                    prototype_uses_observed_residual_value=policy.observed_residual_value,
                    native_slot_roles=ANCHOR_SLOT_ROLES.get(policy.name),
                    topology_used_in_construction=policy.name == "FamilyBalancedRepair",
                    eligibility_epsilon=epsilon,
                    eligibility_selection_rule_version=EXPERIMENT13B_SWEEP_VERSION if phase == "13B" else None,
                )
                validate_row_spec(row)
                rows.append(row)
    expected = PHASE_EXPECTED_ROW_COUNTS["13A"] if phase == "13A" else 45
    if len(rows) != expected or len({row.row_id for row in rows}) != expected or len({row.pair_id for row in rows}) != expected:
        raise AssertionError(f"Experiment {phase} grid must contain {expected} unique rows and pairs")
    return rows


def validate_row_spec(row: StrategyRowSpec) -> None:
    if (row.residual_width, row.residual_depth) != (W, D):
        raise ValueError("Experiment 13 rows must remain fixed at W8D16")
    if row.reserved_atom != RESERVED_ATOM or row.active_atoms_per_layer != ACTIVE_ATOMS_PER_LAYER:
        raise ValueError("Experiment 13 requires Atom0=NoOpAtom and seven active atoms per layer")
    if row.scalar_schema != SCALAR_SCHEMA or row.path_search_policy != PATH_SEARCH_POLICY:
        raise ValueError("Experiment 13 runtime scalar and path-search settings are fixed")
    if row.finish_threshold != FINISH_THRESHOLD or row.head_outputs_actual != HEAD_OUTPUTS:
        raise ValueError("Experiment 13 finish threshold and head accounting are fixed")
    if row.runtime_topology is not None:
        raise ValueError("runtime_topology must remain absent")
    if row.layer_schedule == "AnchorNative":
        if row.native_slot_roles != ANCHOR_SLOT_ROLES.get(row.construction_policy):
            raise ValueError("Experiment 12 anchors must preserve their native slot-role schedule")
        if row.topology_used_in_construction != (row.construction_policy == "FamilyBalancedRepair"):
            raise ValueError("only FamilyBalancedRepair may preserve construction-only topology")
    elif row.native_slot_roles is not None or row.topology_used_in_construction:
        raise ValueError("non-anchor rows cannot carry Experiment 12 native construction semantics")
    if row.layer_schedule == "AllBroad" and row.utility_candidate_budget is not None:
        raise ValueError("pure-prototype rows require a Null utility candidate budget")
    if row.layer_schedule != "AllBroad" and row.utility_candidate_budget not in {
        "CandidateBudget24",
        "CandidateBudget48",
    }:
        raise ValueError("repair-containing rows require CandidateBudget24 or CandidateBudget48")
    if row.experiment_phase == "13A":
        if row.residual_population_policy != "AllResiduals" or row.eligibility_epsilon is not None:
            raise ValueError("13A must use AllResiduals without a construction epsilon")
    elif row.experiment_phase == "13B":
        if row.residual_population_policy != "UnresolvedOnly":
            raise ValueError("13B must use UnresolvedOnly")
        if row.layer_normalization_policy != "LayerClip0To1":
            raise ValueError("13B must use LayerClip0To1")
        _validate_epsilon(row.eligibility_epsilon)
        if row.eligibility_selection_rule_version != EXPERIMENT13B_SWEEP_VERSION:
            raise ValueError("13B must carry the fixed epsilon-sweep version")
    else:
        raise ValueError(f"unsupported experiment phase: {row.experiment_phase}")


def validate_pairing(rows_a: Sequence[StrategyRowSpec], rows_b: Sequence[StrategyRowSpec]) -> None:
    pairs_a = {row.pair_id: row for row in rows_a}
    expected_b = {
        pair_id for pair_id, row in pairs_a.items()
        if row.layer_normalization_policy == "LayerClip0To1"
    }
    if len(rows_a) != 90 or len(pairs_a) != 90:
        raise ValueError("Experiment 13A requires exactly 90 unique rows")
    expected_keys = {
        (pair_id, epsilon)
        for pair_id in expected_b
        for epsilon in EXPERIMENT13B_ELIGIBILITY_EPSILONS
    }
    pairs_b = {(row.pair_id, _validate_epsilon(row.eligibility_epsilon)): row for row in rows_b}
    if len(rows_b) != 135 or len(pairs_b) != 135 or set(pairs_b) != expected_keys:
        raise ValueError(
            "Experiment 13B requires all 45 LayerClip0To1 counterparts at each of the three fixed epsilons"
        )
    for (pair_id, _epsilon), right in pairs_b.items():
        left = pairs_a[pair_id]
        if left.paired_settings != right.paired_settings:
            raise ValueError(f"paired settings differ for {pair_id}")
        if left.residual_population_policy != "AllResiduals" or right.residual_population_policy != "UnresolvedOnly":
            raise ValueError(f"invalid population-policy pairing for {pair_id}")


def configuration_fingerprint() -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "optimization_version": OPTIMIZATION_VERSION,
        "implementation_fingerprint": implementation_fingerprint(),
        "selection_rule_version": SELECTION_RULE_VERSION,
        "candidate_epsilons": CANDIDATE_EPSILONS,
        "selection_checkpoint_definition": SELECTION_CHECKPOINT_DEFINITION,
        "pairs": [row.paired_settings for row in experiment13a_specs()],
    }
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


def run_13a(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    metadata_path: Path = DEFAULT_METADATA,
    backend: str = "auto",
    smoke: bool = False,
    corpus_sample_fraction: float | None = None,
    train_sample_fraction: float = 1.0,
    validation_sample_fraction: float = 1.0,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
    cache_dir: Path | None = None,
    rebuild_cache: bool = False,
    verify_optimized_kernels: str = "off",
    resume: bool = False,
    row_ids: set[str] | None = None,
    chunk_size: int = 256,
    dataset: Any | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, str]:
    train_fraction, validation_fraction = _resolve_sample_fractions(
        corpus_sample_fraction, train_sample_fraction, validation_sample_fraction
    )
    return _run_phase(
        phase="13A", specs=experiment13a_specs(), output_dir=output_dir,
        metadata_path=metadata_path, backend=backend, smoke=smoke,
        train_sample_fraction=train_fraction, validation_sample_fraction=validation_fraction,
        sample_seed=sample_seed, cache_dir=cache_dir, rebuild_cache=rebuild_cache,
        verify_optimized_kernels=verify_optimized_kernels, resume=resume,
        row_ids=row_ids, chunk_size=chunk_size, dataset=dataset, progress=progress,
    )


def select_epsilon(*, run_dir: Path = DEFAULT_OUTPUT_DIR) -> EpsilonSelection:
    run_dir = Path(run_dir)
    validate_completed_13a(run_dir)
    missing = [name for name in REQUIRED_CALIBRATION_FILES if not _nonempty(run_dir / name)]
    if missing:
        reason = "missing required 13A calibration artifacts: " + ", ".join(missing)
        _write_json(run_dir / "epsilon_selection_status.json", {"state": "blocked", "reason": reason})
        _failure(run_dir, "select-epsilon", reason)
        raise PhaseGateError(reason)
    payload = _compute_epsilon_selection(run_dir)
    _write_json(run_dir / "epsilon_selection.json", payload)
    _write_json(
        run_dir / "epsilon_selection_status.json",
        {"state": "complete" if payload["selection_passed"] else "not_passed", "reason": payload["selection_notes"]},
    )
    _event(run_dir, "select-epsilon", "selection_complete", payload["selection_notes"])
    return EpsilonSelection.from_dict(payload)


def override_epsilon(
    *,
    run_dir: Path = DEFAULT_OUTPUT_DIR,
    selected_epsilon: float,
    rationale: str,
) -> EpsilonSelection:
    """Record the explicit, pilot-backed threshold decision required by the plan."""
    run_dir = Path(run_dir)
    manifest, _ = validate_completed_13a(run_dir)
    selection_path = run_dir / "epsilon_selection.json"
    selection = load_epsilon_selection(
        selection_path,
        expected_run_identity=manifest["experiment13a_run_identity"],
        expected_configuration_fingerprint=manifest["configuration_fingerprint"],
        require_passed=False,
    )
    if selection.selection_passed:
        raise PhaseGateError("epsilon selection has already passed")
    epsilon = _validate_epsilon(selected_epsilon)
    if epsilon not in PILOT_EPSILONS:
        raise PhaseGateError("pilot override epsilon must be 0.001 or 0.0025")
    if not isinstance(rationale, str) or not rationale.strip():
        raise PhaseGateError("pilot override requires a non-empty rationale")
    pilot_manifest_path = run_dir / "experiment13b_pilot_manifest.json"
    pilot_results_path = run_dir / "experiment13b_pilot_results.csv"
    if not pilot_manifest_path.exists() or not pilot_results_path.exists():
        raise PhaseGateError("pilot override requires completed pilot artifacts")
    pilot_manifest = _read_json(pilot_manifest_path)
    if not pilot_manifest.get("complete") or epsilon not in tuple(float(value) for value in pilot_manifest.get("candidate_epsilons", [])):
        raise PhaseGateError("pilot override epsilon is not covered by the completed pilot")
    with pilot_results_path.open(encoding="utf-8", newline="") as handle:
        pilot_rows = list(csv.DictReader(handle))
    matching = [row for row in pilot_rows if math.isclose(float(row.get("epsilon", "nan")), epsilon)]
    if not matching:
        raise PhaseGateError("pilot override has no matching result rows")
    timestamp = _now()
    payload = dict(selection.payload)
    payload.update(
        selected_epsilon=epsilon,
        selection_passed=True,
        selection_override=True,
        selection_override_rationale=rationale.strip(),
        selection_override_timestamp=timestamp,
        selection_timestamp=timestamp,
        pilot_evidence={
            "pilot_manifest": pilot_manifest_path.name,
            "pilot_results": pilot_results_path.name,
            "selected_epsilon_result_row_count": len(matching),
            "construction_policies": sorted({str(row.get("construction_policy", "")) for row in matching}),
        },
        selection_notes="explicit pilot-backed epsilon override",
    )
    _write_json(selection_path, payload)
    _write_json(run_dir / "epsilon_selection_status.json", {"state": "complete", "reason": payload["selection_notes"]})
    _event(run_dir, "override-epsilon", "selection_override", rationale.strip())
    return EpsilonSelection.from_dict(payload)


def run_13b(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    epsilon_selection_path: Path | None = None,
    metadata_path: Path = DEFAULT_METADATA,
    backend: str = "auto",
    smoke: bool = False,
    corpus_sample_fraction: float | None = None,
    train_sample_fraction: float = 1.0,
    validation_sample_fraction: float = 1.0,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
    cache_dir: Path | None = None,
    rebuild_cache: bool = False,
    verify_optimized_kernels: str = "off",
    resume: bool = False,
    row_ids: set[str] | None = None,
    chunk_size: int = 256,
    dataset: Any | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, str]:
    train_fraction, validation_fraction = _resolve_sample_fractions(
        corpus_sample_fraction, train_sample_fraction, validation_sample_fraction
    )
    _validate_run_options(smoke, train_fraction, validation_fraction, backend, verify_optimized_kernels)
    output_dir = Path(output_dir)
    selection_path = Path(epsilon_selection_path or output_dir / "epsilon_selection.json")
    try:
        manifest, _ = validate_completed_13a(output_dir)
        selection = load_epsilon_selection(
            selection_path,
            expected_run_identity=manifest["experiment13a_run_identity"],
            expected_configuration_fingerprint=manifest["configuration_fingerprint"],
            require_passed=False,
        )
        _validate_13b_invocation(manifest, metadata_path, train_fraction, validation_fraction, sample_seed)
    except PhaseGateError as exc:
        _blocked_13b(output_dir, str(exc))
        raise
    return _run_phase(
        phase="13B", specs=experiment13b_specs(), output_dir=output_dir,
        metadata_path=metadata_path, backend=backend, smoke=smoke,
        train_sample_fraction=train_fraction, validation_sample_fraction=validation_fraction,
        sample_seed=sample_seed, cache_dir=cache_dir, rebuild_cache=rebuild_cache,
        verify_optimized_kernels=verify_optimized_kernels, resume=resume, row_ids=row_ids,
        chunk_size=chunk_size, dataset=dataset, progress=progress,
        run_identity=selection.experiment13a_run_identity,
        fingerprint=selection.configuration_fingerprint,
    )


def run_13b_pilot(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    epsilon_selection_path: Path | None = None,
    candidate_epsilons: Sequence[float] = PILOT_EPSILONS,
    row_ids: set[str] | None = None,
    metadata_path: Path = DEFAULT_METADATA,
    backend: str = "auto",
    chunk_size: int = 256,
    cache_dir: Path | None = None,
    rebuild_cache: bool = False,
    verify_optimized_kernels: str = "off",
    dataset: Any | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, str]:
    _validate_run_options(False, 1.0, 1.0, backend, verify_optimized_kernels)
    output_dir = Path(output_dir)
    manifest, _ = validate_completed_13a(output_dir)
    selection = load_epsilon_selection(
        Path(epsilon_selection_path or output_dir / "epsilon_selection.json"),
        expected_run_identity=manifest["experiment13a_run_identity"],
        expected_configuration_fingerprint=manifest["configuration_fingerprint"],
        require_passed=False,
    )
    phase_a = manifest.get("phases", {}).get("13A", {})
    train_fraction = float(phase_a.get("train_sample_fraction", 1.0))
    validation_fraction = float(phase_a.get("validation_sample_fraction", 1.0))
    sample_seed = int(phase_a.get("sample_seed", DEFAULT_SAMPLE_SEED))
    _validate_13b_invocation(manifest, metadata_path, train_fraction, validation_fraction, sample_seed)
    if selection.selection_passed:
        raise PhaseGateError("run-13b-pilot requires selection_passed=false")
    epsilons = tuple(_validate_epsilon(value) for value in candidate_epsilons)
    if not epsilons or any(value not in PILOT_EPSILONS for value in epsilons):
        raise PhaseGateError("pilot epsilons are restricted to 0.001 and 0.0025")
    if len(set(epsilons)) != len(epsilons):
        raise PhaseGateError("pilot candidate epsilons must not contain duplicates")
    rows = [row for row in experiment13b_specs(PILOT_EPSILONS[0]) if row.construction_policy in PILOT_POLICIES]
    if row_ids is not None:
        invalid = row_ids - {row.row_id for row in rows}
        if invalid:
            raise PhaseGateError("pilot row filter contains non-pilot rows: " + ", ".join(sorted(invalid)))
        rows = [row for row in rows if row.row_id in row_ids]
    if not rows:
        raise PhaseGateError("pilot execution requires at least one prespecified row")
    pilot_manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "experiment_phase": "13B-pilot",
        "experiment13a_run_identity": selection.experiment13a_run_identity,
        "configuration_fingerprint": selection.configuration_fingerprint,
        "candidate_epsilons": list(epsilons),
        "allowed_construction_policies": list(PILOT_POLICIES),
        "row_ids": [row.row_id for row in rows],
        "metadata_path": str(Path(metadata_path)),
        "backend": backend,
        "sample_fingerprint": phase_a.get("sample_fingerprint"),
        "complete": False,
    }
    _write_json(
        output_dir / "experiment13b_pilot_manifest.json",
        pilot_manifest,
    )
    from .strategy_grid_runtime import run_strategy_row, write_csv
    if dataset is None:
        dataset, _ = load_dataset_cached(
            metadata_path, cache_dir=cache_dir, resolution=CONTROL_POINT_COUNT,
            x_grid_mode="inclusive", rebuild=rebuild_cache, progress=progress,
        )
    dataset, sample = deterministic_sample(
        dataset, train_fraction=train_fraction, validation_fraction=validation_fraction, seed=sample_seed,
    )
    if not sample_manifest_matches(output_dir / "sample_manifest.json", sample):
        raise PhaseGateError("pilot sample identity does not match completed 13A")
    base_stage = load_or_build_base_stage(
        dataset, sample, backend=backend, chunk_size=chunk_size, cache_dir=cache_dir,
        rebuild=rebuild_cache, progress=progress,
    )
    evidence: list[dict[str, Any]] = []
    for epsilon in epsilons:
        specs = {row.pair_id: row for row in experiment13b_specs(epsilon)}
        for template in rows:
            spec = specs[template.pair_id]
            pilot_dir = output_dir / "pilot" / f"epsilon_{epsilon:g}" / "rows" / spec.row_id
            result = run_strategy_row(
                spec, dataset, pilot_dir, run_identity=selection.experiment13a_run_identity,
                configuration_fingerprint=selection.configuration_fingerprint, backend=backend,
                chunk_size=chunk_size, progress=progress, base_stage=base_stage,
                sample_provenance=sample, verify_optimized_kernels=verify_optimized_kernels == "first-use",
            )
            evidence.append({"epsilon": epsilon, **result.summary})
    write_csv(output_dir / "experiment13b_pilot_results.csv", evidence)
    pilot_manifest.update(complete=True, completed_at_utc=_now(), result_row_count=len(evidence))
    _write_json(output_dir / "experiment13b_pilot_manifest.json", pilot_manifest)
    _event(output_dir, "13B-pilot", "pilot_complete", f"rows={len(evidence)}")
    return {"pilot_manifest": str(output_dir / "experiment13b_pilot_manifest.json"), "pilot_results": str(output_dir / "experiment13b_pilot_results.csv")}


def analyze_strategy_grid(*, run_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, str]:
    run_dir = Path(run_dir)
    status_a, status_b = read_phase_status(run_dir, "13A"), read_phase_status(run_dir, "13B")
    if status_a["state"] != "complete" or status_b["state"] != "complete":
        raise AnalysisNotReadyError(
            f"analysis requires complete 13A and 13B phases; got 13A={status_a['state']} 13B={status_b['state']}"
        )
    validate_completed_13b(run_dir)
    missing = [name for name in REQUIRED_ANALYSIS_FILES if not _nonempty(run_dir / name)]
    if missing:
        raise AnalysisNotReadyError("analysis inputs are incomplete: " + ", ".join(missing))
    return _write_strategy_analysis(run_dir)


def verify_equivalence(
    *,
    baseline_run: Path,
    output_dir: Path,
    metadata_path: Path = DEFAULT_METADATA,
    cache_dir: Path | None = None,
    backend: str = "auto",
    row_ids: set[str] | None = None,
    chunk_size: int = 256,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Rerun completed legacy rows and require exact scientific artifacts."""
    baseline_run, output_dir = Path(baseline_run), Path(output_dir)
    _require_fresh_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_manifest = _read_json(baseline_run / "manifest.json")
    identity = str(baseline_manifest.get("experiment13a_run_identity") or "equivalence")
    fingerprint = str(baseline_manifest.get("configuration_fingerprint") or configuration_fingerprint())
    available = {
        path.parent.name for path in (baseline_run / "rows").glob("*/summary.csv") if path.is_file()
    }
    requested = row_ids or available
    unknown = requested - available
    if unknown:
        raise Experiment13Error("equivalence baseline lacks completed rows: " + ", ".join(sorted(unknown)))
    specs = [spec for spec in experiment13a_specs() if spec.row_id in requested]
    if not specs:
        raise Experiment13Error("equivalence verification requires at least one completed row")
    dataset, dataset_cache = load_dataset_cached(
        metadata_path, cache_dir=cache_dir, resolution=CONTROL_POINT_COUNT,
        x_grid_mode="inclusive", progress=progress,
    )
    dataset, sample = deterministic_sample(
        dataset, train_fraction=1.0, validation_fraction=1.0, seed=DEFAULT_SAMPLE_SEED,
    )
    base_stage = load_or_build_base_stage(
        dataset, sample, backend=backend, chunk_size=chunk_size, cache_dir=cache_dir, progress=progress,
    )
    from .strategy_grid_runtime import run_strategy_row

    results: list[dict[str, Any]] = []
    for index, spec in enumerate(specs, start=1):
        if progress:
            progress(f"equivalence {index}/{len(specs)}: {spec.row_id}")
        candidate = output_dir / "rows" / spec.row_id
        run_strategy_row(
            spec, dataset, candidate, run_identity=identity, configuration_fingerprint=fingerprint,
            backend=backend, chunk_size=chunk_size, progress=progress,
            base_stage=replace(base_stage, cache_hit=base_stage.cache_hit or index > 1),
            sample_provenance=sample, verify_optimized_kernels=True,
        )
        mismatches = _compare_scientific_row(baseline_run / "rows" / spec.row_id, candidate)
        results.append({"row_id": spec.row_id, "mismatches": mismatches, "passed": not mismatches})
    report = {
        "schema_version": "experiment13_equivalence_v1",
        "baseline_run": str(baseline_run),
        "output_dir": str(output_dir),
        "backend": backend,
        "dataset_cache": dataset_cache,
        "sample_fingerprint": sample.sample_fingerprint,
        "row_count": len(results),
        "passed": all(row["passed"] for row in results),
        "rows": results,
        "completed_at_utc": _now(),
    }
    _write_json(output_dir / "equivalence_report.json", report)
    if not report["passed"]:
        raise Experiment13Error("scientific equivalence failed; see equivalence_report.json")
    return report


def analyze_scaling_ablation(*, full_run: Path, sampled_run: Path, output_dir: Path) -> dict[str, str]:
    """Compare matched 13A methods while explicitly excluding historical runtime."""
    full_run, sampled_run, output_dir = Path(full_run), Path(sampled_run), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    full_rows = _completed_row_summaries(full_run, "13A")
    sampled_rows = _completed_row_summaries(sampled_run, "13A")
    matched = sorted(set(full_rows) & set(sampled_rows))
    if not matched:
        raise AnalysisNotReadyError("scaling analysis has no matched completed 13A rows")
    full_validation = _validation_membership_sha256(full_run / "rows" / matched[0] / "atom_assignments.csv")
    sampled_validation = _validation_membership_sha256(sampled_run / "rows" / matched[0] / "atom_assignments.csv")
    if full_validation != sampled_validation:
        raise AnalysisNotReadyError("scaling analysis requires identical validation membership")
    metrics = (
        "validation_median_rmse",
        "validation_p95_rmse",
        "validation_strict_perfect_lfo_rate",
        "validation_node_max_error_p95",
        "validation_max_abs_error_p95",
    )
    rows: list[dict[str, Any]] = []
    for row_id in matched:
        row: dict[str, Any] = {
            "row_id": row_id,
            "experiment_phase": "13A",
            "full_train_fraction": 1.0,
            "sampled_train_fraction": 0.5,
            "validation_fraction": 1.0,
            "runtime_comparison_allowed": False,
        }
        for metric in metrics:
            left, right = float(full_rows[row_id][metric]), float(sampled_rows[row_id][metric])
            row[f"full_{metric}"] = left
            row[f"sampled_{metric}"] = right
            row[f"delta_{metric}"] = right - left
        rows.append(row)
    from .strategy_grid_runtime import write_csv

    csv_path = output_dir / "training_data_scaling_ablation.csv"
    write_csv(csv_path, rows)
    report = {
        "schema_version": "experiment13_training_scaling_ablation_v1",
        "matched_row_count": len(rows),
        "validation_membership_sha256": full_validation,
        "full_run": str(full_run),
        "sampled_run": str(sampled_run),
        "runtime_comparison_allowed": False,
        "runtime_exclusion_reason": "legacy timings include Modern Standby and use a different execution implementation",
        "completed_at_utc": _now(),
    }
    report_path = output_dir / "training_data_scaling_ablation.json"
    _write_json(report_path, report)
    return {"scaling_csv": str(csv_path), "scaling_report": str(report_path)}


def _run_phase(
    *,
    phase: ExperimentPhase,
    specs: Sequence[StrategyRowSpec],
    output_dir: Path,
    metadata_path: Path,
    backend: str,
    smoke: bool,
    train_sample_fraction: float,
    validation_sample_fraction: float,
    sample_seed: int,
    cache_dir: Path | None,
    rebuild_cache: bool,
    verify_optimized_kernels: str,
    resume: bool,
    row_ids: set[str] | None,
    chunk_size: int,
    dataset: Any | None,
    progress: Callable[[str], None] | None,
    run_identity: str | None = None,
    fingerprint: str | None = None,
) -> dict[str, str]:
    _validate_run_options(
        smoke, train_sample_fraction, validation_sample_fraction, backend, verify_optimized_kernels
    )
    if int(chunk_size) < 1:
        raise ValueError("chunk_size must be positive")
    rows = _filter_rows(specs, row_ids)
    expected_row_count = PHASE_EXPECTED_ROW_COUNTS[phase]
    partial = smoke or row_ids is not None or len(rows) != expected_row_count
    provenance = _git_provenance()
    if not partial and provenance.get("dirty"):
        raise Experiment13Error("production Experiment 13 runs require a clean git worktree")
    output_dir = Path(output_dir)
    if phase == "13A" and not resume:
        _require_fresh_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if resume:
        _archive_cancel_request(output_dir)
    identity = run_identity or _run_identity(output_dir, resume)
    config = fingerprint or configuration_fingerprint()
    if resume:
        existing_manifest = _read_json(output_dir / "manifest.json")
        if existing_manifest.get("experiment13a_run_identity") != identity:
            raise Experiment13Error("resume run identity does not match the existing manifest")
        if existing_manifest.get("configuration_fingerprint") != config:
            raise Experiment13Error("resume implementation/configuration fingerprint does not match the existing run")
    status = _phase_status(
        phase, "running", identity, len(rows), expected_row_count,
        smoke, row_ids is not None, "loading dataset",
    )
    if phase == "13B":
        status.update(
            eligibility_epsilon_sweep=list(EXPERIMENT13B_ELIGIBILITY_EPSILONS),
            eligibility_epsilon_sweep_version=EXPERIMENT13B_SWEEP_VERSION,
        )
    status.update(
        rows={row.row_id: {"status": "pending"} for row in rows},
        current_row_id="", current_row_number=0, current_task_phase="dataset_load",
        current_task_percent=0.0, overall_tasks_completed=0, overall_tasks_total=len(rows),
        overall_tasks_percent=0.0,
    )
    _write_phase_status(output_dir, status)
    _event(output_dir, phase, "run_start", f"rows={len(rows)} backend={backend} smoke={smoke}")
    if not (output_dir / "failures.csv").exists():
        _write_csv(output_dir / "failures.csv", [], ("experiment_phase", "row_id", "error", "timestamp_utc"))

    from .strategy_grid_runtime import read_one_csv, run_strategy_row

    if dataset is None:
        dataset, dataset_cache = load_dataset_cached(
            metadata_path, cache_dir=cache_dir, resolution=CONTROL_POINT_COUNT,
            x_grid_mode="inclusive", rebuild=rebuild_cache, progress=progress,
        )
    else:
        dataset_cache = {"cache_key": "injected", "cache_hit": False, "cache_path": None}
    if smoke:
        from . import component_ladder as x12
        dataset = x12._subset_dataset(dataset, smoke=True, corpus_sample_fraction=1.0)
        dataset, sample = deterministic_sample(
            dataset, train_fraction=1.0, validation_fraction=1.0, seed=sample_seed,
        )
    else:
        dataset, sample = deterministic_sample(
            dataset, train_fraction=train_sample_fraction,
            validation_fraction=validation_sample_fraction, seed=sample_seed,
        )
    sample_path = output_dir / "sample_manifest.json"
    if sample_path.exists() and not sample_manifest_matches(sample_path, sample):
        raise Experiment13Error("run sample identity does not match the existing sample manifest")
    if not sample_path.exists():
        write_sample_artifacts(output_dir, dataset, sample)
    base_stage = load_or_build_base_stage(
        dataset, sample, backend=backend, chunk_size=chunk_size, cache_dir=cache_dir,
        rebuild=rebuild_cache, progress=progress,
    )
    _write_phase_manifest(
        output_dir=output_dir, phase=phase, specs=rows, run_identity=identity,
        fingerprint=config, metadata_path=metadata_path, backend=backend,
        smoke=smoke,
        train_sample_fraction=train_sample_fraction,
        validation_sample_fraction=validation_sample_fraction,
        sample_seed=sample_seed, sample=sample, dataset_cache=dataset_cache,
        base_stage_cache_key=base_stage.cache_key,
        complete_design=not partial, resume=resume,
    )
    status.update(
        train_count=len(dataset.train_indices), validation_count=len(dataset.validation_indices),
        sample_fingerprint=sample.sample_fingerprint,
        train_sample_fraction=train_sample_fraction,
        validation_sample_fraction=validation_sample_fraction,
        sample_seed=sample_seed,
        current_task_phase="base_stage_ready", current_task_percent=0.0,
    )
    _write_phase_status(output_dir, status)
    _event(
        output_dir, phase, "dataset_ready",
        f"train={len(dataset.train_indices)} validation={len(dataset.validation_indices)} sample={sample.sample_fingerprint[:12]}",
    )
    completed = 0
    rows_root = output_dir / "rows"
    staging_root = output_dir / ".in_progress"
    interrupted_root = output_dir / "interrupted_rows"
    rows_root.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir(parents=True, exist_ok=True)
    for index, spec in enumerate(rows, start=1):
        row_dir = rows_root / spec.row_id
        staging_dir = staging_root / spec.row_id
        try:
            if resume and _row_is_complete(row_dir):
                completed_summary = read_one_csv(row_dir / "summary.csv")
                if completed_summary.get("configuration_fingerprint") != config:
                    raise Experiment13Error(f"completed row fingerprint mismatch during resume: {spec.row_id}")
                if completed_summary.get("sample_fingerprint") != sample.sample_fingerprint:
                    raise Experiment13Error(f"completed row sample mismatch during resume: {spec.row_id}")
                event = "row_skipped"
            else:
                if row_dir.exists():
                    _archive_incomplete_row(row_dir, interrupted_root, "incomplete_final")
                if staging_dir.exists():
                    _archive_incomplete_row(staging_dir, interrupted_root, "interrupted_staging")
                staging_dir.mkdir(parents=True, exist_ok=False)
                _check_cancel_requested(output_dir)
                status.update(
                    current_row_id=spec.row_id, current_row_number=index,
                    current_task_phase="starting", current_task_percent=0.0,
                    reason=f"running row {index}/{len(rows)}",
                )
                status["rows"][spec.row_id] = {"status": "running", "started_at_utc": _now()}
                _write_phase_status(output_dir, status)
                _event(output_dir, phase, "row_start", f"{index}/{len(rows)} {spec.row_id}")

                def row_progress(message: str) -> None:
                    status["current_task_phase"] = message
                    status["current_task_percent"] = _strategy_progress_percent(message)
                    _write_phase_status(output_dir, status)
                    if progress is not None:
                        progress(f"experiment13: {spec.row_id}: {message}")

                run_strategy_row(
                    spec, dataset, staging_dir, run_identity=identity,
                    configuration_fingerprint=config, backend=backend,
                    chunk_size=chunk_size, progress=row_progress,
                    base_stage=replace(base_stage, cache_hit=base_stage.cache_hit or index > 1),
                    sample_provenance=sample, cancel_check=lambda: _check_cancel_requested(output_dir),
                    verify_optimized_kernels=verify_optimized_kernels == "first-use",
                )
                if not _row_is_complete(staging_dir):
                    missing = _missing_row_files(staging_dir)
                    raise Experiment13Error(
                        f"row staging did not produce a complete artifact set for {spec.row_id}: {', '.join(missing)}"
                    )
                staging_dir.replace(row_dir)
                event = "row_complete"
            completed += 1
            status["rows"][spec.row_id] = {"status": "completed", "completed_at_utc": _now()}
            status.update(
                completed_rows=completed, overall_tasks_completed=completed,
                overall_tasks_percent=100.0 * completed / len(rows),
                current_task_percent=100.0,
            )
            _write_phase_status(output_dir, status)
            _event(output_dir, phase, event, f"{index}/{len(rows)} {spec.row_id}")
        except RunCancelled as exc:
            if staging_dir.exists():
                _archive_incomplete_row(staging_dir, interrupted_root, "cancelled")
            status["rows"][spec.row_id] = {"status": "cancelled", "cancelled_at_utc": _now()}
            status.update(state="cancelled", reason=str(exc), cancelled_at_utc=_now())
            _write_phase_status(output_dir, status)
            _event(output_dir, phase, "run_cancelled", f"{spec.row_id}: {exc}")
            raise
        except Exception as exc:
            if staging_dir.exists():
                _archive_incomplete_row(staging_dir, interrupted_root, "failed")
            status["rows"][spec.row_id] = {"status": "failed", "failed_at_utc": _now(), "error": str(exc)}
            status.update(state="failed", failed_rows=int(status.get("failed_rows", 0)) + 1, reason=str(exc), failed_at_utc=_now())
            _write_phase_status(output_dir, status)
            _failure(output_dir, phase, str(exc), row_id=spec.row_id)
            _event(output_dir, phase, "row_failed", f"{spec.row_id}: {exc}")
            raise

    status.update(
        state="partial" if partial else "complete",
        completed_rows=completed,
        current_row_id="", current_row_number=0,
        current_task_phase="complete", current_task_percent=100.0,
        completed_at_utc=_now(),
        reason="selected diagnostic rows completed" if partial else f"all {phase} rows completed",
    )
    _write_phase_status(output_dir, status)
    outputs = _aggregate_strategy_artifacts(output_dir)
    _event(output_dir, phase, "run_complete", status["reason"])
    return outputs


def _aggregate_strategy_artifacts(run_dir: Path) -> dict[str, str]:
    from .strategy_grid_runtime import merge_csv_files, read_one_csv, write_csv

    run_dir = Path(run_dir)
    manifest = _read_json(run_dir / "manifest.json")
    allowed_ids = {
        str(row["row_id"])
        for phase_payload in manifest.get("phases", {}).values()
        for row in phase_payload.get("rows", [])
        if isinstance(row, dict) and row.get("row_id")
    }
    row_dirs = [
        row_dir for row_dir in sorted((run_dir / "rows").glob("x13*/"))
        if row_dir.name in allowed_ids and (row_dir / "summary.csv").exists()
    ]
    filenames = (
        "atom_construction.csv", "atom_assignments.csv", "candidate_search_diagnostics.csv",
        "slot_progression.csv", "partial_codebook_validation.csv", "layer_epsilon_quantiles.csv",
        "slot_epsilon_quantiles.csv", "epsilon_coverage.csv", "retired_error_mass.csv",
        "execution_timing.csv",
    )
    summaries = [read_one_csv(row_dir / "summary.csv") for row_dir in row_dirs]
    write_csv(run_dir / "summary.csv", summaries)
    write_csv(run_dir / "strategy_results.csv", summaries)
    for filename in filenames:
        merge_csv_files(run_dir / filename, [row_dir / filename for row_dir in row_dirs])
    budget_rows = [
        {
            "experiment_phase": row.get("experiment_phase", ""), "row_id": row.get("row_id", ""),
            "pair_id": row.get("pair_id", ""), "W": row.get("residual_width", W),
            "D": row.get("residual_depth", D), "reserved_atom": row.get("reserved_atom", RESERVED_ATOM),
            "active_atoms_per_layer": row.get("active_atoms_per_layer", ACTIVE_ATOMS_PER_LAYER),
            "categorical_outputs": BASE_DICTIONARY_SIZE + D * W,
            "continuous_outputs": D + 1 + D,
            "head_outputs_formula": "32 + 16*8 + 17 + 16",
            "head_outputs_actual": row.get("head_outputs_actual", HEAD_OUTPUTS),
        }
        for row in summaries
    ]
    write_csv(run_dir / "budget_accounting.csv", budget_rows)
    return {
        "output_dir": str(run_dir), "summary": str(run_dir / "summary.csv"),
        "strategy_results": str(run_dir / "strategy_results.csv"),
        "atom_construction": str(run_dir / "atom_construction.csv"),
        "epsilon_coverage": str(run_dir / "epsilon_coverage.csv"),
        "retired_error_mass": str(run_dir / "retired_error_mass.csv"),
    }


def _compute_epsilon_selection(run_dir: Path) -> dict[str, Any]:
    from .strategy_grid_runtime import read_csv

    manifest, _ = validate_completed_13a(run_dir)
    retired = read_csv(run_dir / "retired_error_mass.csv")
    coverage = read_csv(run_dir / "epsilon_coverage.csv")
    statistics: dict[float, dict[str, Any]] = {}
    passing: list[float] = []
    for epsilon in CANDIDATE_EPSILONS:
        values = [
            float(row["unexplained_retired_energy_fraction"])
            for row in retired
            if row.get("experiment_phase") == "13A"
            and int(float(row["residual_layer"])) in range(1, 17)
            and int(float(row["active_atom_slot"])) in range(0, 7)
            and math.isclose(float(row["epsilon"]), epsilon)
            and str(row.get("zero_total_energy", "")).lower() not in {"true", "1"}
        ]
        if not values:
            raise PhaseGateError(f"no valid retired-error checkpoints for epsilon {epsilon}")
        median = _quantile(values, 0.50)
        p95 = _quantile(values, 0.95)
        checkpoint_fractions: dict[tuple[int, int], list[float]] = {}
        for row in coverage:
            slot = row.get("active_atom_slot")
            if (
                row.get("experiment_phase") == "13A" and row.get("dataset_split") == "training"
                and slot not in {None, "", "None"}
                and math.isclose(float(row["epsilon"]), epsilon)
            ):
                layer_i, slot_i = int(float(row["residual_layer"])), int(float(slot))
                if layer_i in range(1, 13) and slot_i in range(0, 6):
                    checkpoint_fractions.setdefault((layer_i, slot_i), []).append(float(row["resolved_fraction"]))
        early_middle_max = max((_quantile(values_, 0.50) for values_ in checkpoint_fractions.values()), default=0.0)
        retired_lfo = [
            float(row["retired_lfo_fraction"]) for row in retired
            if row.get("experiment_phase") == "13A" and math.isclose(float(row["epsilon"]), epsilon)
        ]
        result = {
            "median_unexplained_retired_energy_fraction": median,
            "p95_unexplained_retired_energy_fraction": p95,
            "max_early_middle_median_retired_lfo_fraction": early_middle_max,
            "median_retired_lfo_fraction": _quantile(retired_lfo, 0.50),
            "checkpoint_count": len(values),
        }
        statistics[epsilon] = result
        if median <= 0.01 and p95 <= 0.05 and early_middle_max >= 0.05:
            passing.append(epsilon)
    selected = max(passing) if passing else None
    reported = statistics[selected if selected is not None else CANDIDATE_EPSILONS[0]]
    passed = selected is not None
    note = (
        f"selected largest passing candidate epsilon {selected:g}"
        if passed else "no candidate epsilon satisfied all automatic selection conditions; restricted pilot required"
    )
    return {
        "candidate_epsilons": list(CANDIDATE_EPSILONS),
        "selection_rule_version": SELECTION_RULE_VERSION,
        "selection_checkpoint_definition": SELECTION_CHECKPOINT_DEFINITION,
        "selected_epsilon": selected,
        "training_statistics_used": {"dataset_split": "training", "row_count": 90, "candidate_statistics": {str(key): value for key, value in statistics.items()}},
        "median_unexplained_retired_energy_fraction": reported["median_unexplained_retired_energy_fraction"],
        "p95_unexplained_retired_energy_fraction": reported["p95_unexplained_retired_energy_fraction"],
        "retired_lfo_fraction_summary": {str(key): value["median_retired_lfo_fraction"] for key, value in statistics.items()},
        "selection_timestamp": _now() if passed else None,
        "experiment13a_run_identity": manifest["experiment13a_run_identity"],
        "configuration_fingerprint": manifest["configuration_fingerprint"],
        "selection_passed": passed,
        "selection_override": False,
        "selection_override_rationale": None,
        "selection_override_timestamp": None,
        "pilot_evidence": None,
        "selection_notes": note,
    }


def _write_strategy_analysis(run_dir: Path) -> dict[str, str]:
    from .strategy_grid_runtime import read_csv, write_csv

    run_dir = Path(run_dir)
    rows = read_csv(run_dir / "summary.csv")
    phase_a_by_pair = {
        str(row.get("pair_id")): row for row in rows if row.get("experiment_phase") == "13A"
    }
    paired = []
    phase_b_rows = sorted(
        (row for row in rows if row.get("experiment_phase") == "13B"),
        key=lambda row: (str(row.get("pair_id")), float(row.get("eligibility_epsilon") or 0.0)),
    )
    for right in phase_b_rows:
        pair_id = str(right.get("pair_id"))
        left = phase_a_by_pair.get(pair_id)
        if not left:
            continue
        paired.append({
            "pair_id": pair_id,
            "experiment13b_row_id": right.get("row_id", ""),
            "eligibility_epsilon": right.get("eligibility_epsilon", ""),
            "construction_policy": left.get("construction_policy", ""),
            "utility_candidate_budget": left.get("utility_candidate_budget", ""),
            "layer_normalization_policy": left.get("layer_normalization_policy", ""),
            "experiment13a_validation_median_rmse": left.get("validation_median_rmse", ""),
            "experiment13b_validation_median_rmse": right.get("validation_median_rmse", ""),
            "median_rmse_delta_13b_minus_13a": float(right["validation_median_rmse"]) - float(left["validation_median_rmse"]),
            "experiment13a_validation_p95_rmse": left.get("validation_p95_rmse", ""),
            "experiment13b_validation_p95_rmse": right.get("validation_p95_rmse", ""),
            "p95_rmse_delta_13b_minus_13a": float(right["validation_p95_rmse"]) - float(left["validation_p95_rmse"]),
        })
    paired_path = run_dir / "paired_strategy_deltas.csv"
    write_csv(paired_path, paired)
    report_path = ERA2_ROOT / "reports" / "EXPERIMENT_13_W8D16_STRATEGY_GRID_REPORT.md"
    image_dir = ERA2_ROOT / "reports" / "images" / "experiment_13"
    _write_calibration_plots(run_dir, image_dir)
    selection = _read_json(run_dir / "epsilon_selection.json")
    ordered_a = sorted((row for row in rows if row.get("experiment_phase") == "13A"), key=lambda row: float(row["validation_p95_rmse"]))
    ordered_b = sorted((row for row in rows if row.get("experiment_phase") == "13B"), key=lambda row: float(row["validation_p95_rmse"]))
    lines = [
        "# Experiment 13: Fixed-W8D16 Strategy Grid", "", "## Main Findings", "",
        "Experiment 13A completed first using `AllResiduals`. Experiment 13B then evaluated every clipped strategy independently at eligibility epsilons `1e-2`, `1e-3`, and `1e-4`, producing 135 `UnresolvedOnly` + `LayerClip0To1` rows.", "",
        f"The automatic 13A selector recorded `selection_passed={selection.get('selection_passed')}`. It is retained as calibration provenance, not used to label one threshold as selected; the 13B sweep is exploratory and was fixed after 13A completed under `{EXPERIMENT13B_SWEEP_VERSION}`.", "",
        _best_row_sentence("13A", ordered_a), "", _best_row_sentence("13B", ordered_b), "",
        "The tables and plots retain the co-primary metrics separately; no automatic scalar ranking is applied.", "",
        "## Experiment 13A Unfiltered Results", "", *_top_rows_table(ordered_a), "",
        "## Experiment 13A Calibration and Frozen Epsilon", "",
        "![Completed-layer epsilon quantiles](./images/experiment_13/layer_epsilon_quantiles.png)", "",
        "![Slot epsilon quantiles](./images/experiment_13/slot_epsilon_quantiles.png)", "",
        "![Completed-layer coverage](./images/experiment_13/completed_layer_coverage.png)", "",
        "![Slot coverage](./images/experiment_13/slot_coverage.png)", "",
        "![Retired LFO and unexplained energy](./images/experiment_13/retired_fraction_vs_energy.png)", "",
        "![Incoming and unexplained retired energy](./images/experiment_13/incoming_vs_unexplained_energy.png)", "",
        "## Experiment 13B Filtered Results", "", *_top_rows_table(ordered_b), "",
        "## AllResiduals Versus UnresolvedOnly", "",
        f"Paired deltas for all `{len(paired)}` strategy-by-epsilon comparisons are available in `{paired_path.name}`. Negative RMSE deltas favor the filtered Experiment 13B construction.", "",
        _paired_delta_sentence(paired), "",
        "## Prototype Policies Versus Observed-Residual Anchors", "", *_factor_table(rows, "construction_family"), "",
        "## Interleaved Versus TwoPhase", "", *_factor_table(rows, "layer_schedule"), "",
        "## CandidateBudget24 Versus CandidateBudget48", "", *_factor_table(rows, "utility_candidate_budget"), "",
        "## Experiment 13A FinalClipOnly Versus LayerClip0To1", "",
        *_factor_table([row for row in rows if row.get("experiment_phase") == "13A"], "layer_normalization_policy"), "",
        "## Broad-Builder and Repair-Objective Interactions", "", *_factor_table(rows, "broad_atom_builder"), "", *_factor_table(rows, "repair_atom_builder"), "",
        "## Partial-Codebook Progression", "",
        "`partial_codebook_validation.csv` reports each row with one through seven active atoms per layer, retaining median RMSE, strict-perfect rate, P95 RMSE, and node-max P95 separately.", "",
        "## Reproducibility", "",
        f"Configuration fingerprint: `{selection.get('configuration_fingerprint')}`.", "",
        "All rows preserve W8D16, Atom0 as `NoOpAtom`, Beam4 encoding, phase and residual gain, the 193-output head contract, and topology-free runtime behavior.", "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return {"output_dir": str(run_dir), "summary": str(run_dir / "summary.csv"), "paired_strategy_deltas": str(paired_path), "report": str(report_path), "report_image_dir": str(image_dir)}


def _write_calibration_plots(run_dir: Path, image_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    from .strategy_grid_runtime import read_csv
    image_dir.mkdir(parents=True, exist_ok=True)
    layer = read_csv(run_dir / "layer_epsilon_quantiles.csv")
    slot = read_csv(run_dir / "slot_epsilon_quantiles.csv")
    coverage = read_csv(run_dir / "epsilon_coverage.csv")
    retired = read_csv(run_dir / "retired_error_mass.csv")
    _median_line_plot(plt, image_dir / "layer_epsilon_quantiles.png", layer, "residual_layer", "epsilon_value", "Completed-layer epsilon quantiles", "residual layer", "epsilon")
    _median_line_plot(plt, image_dir / "slot_epsilon_quantiles.png", slot, "active_atom_slot", "epsilon_value", "Slot epsilon quantiles", "active atom slot", "epsilon")
    figure, axis = plt.subplots(figsize=(7, 5))
    for epsilon in CANDIDATE_EPSILONS:
        selected = [row for row in retired if math.isclose(float(row["epsilon"]), epsilon)]
        if selected:
            axis.scatter([float(row["retired_lfo_fraction"]) for row in selected], [float(row["unexplained_retired_energy_fraction"]) for row in selected], s=5, alpha=0.2, label=f"{epsilon:g}")
    axis.set(xlabel="retired LFO fraction", ylabel="retired unexplained-error energy fraction", title="Retired population versus unexplained energy")
    axis.legend(); figure.tight_layout(); figure.savefig(image_dir / "retired_fraction_vs_energy.png", dpi=160); plt.close(figure)
    _coverage_plot(plt, image_dir / "completed_layer_coverage.png", coverage, completed=True)
    _coverage_plot(plt, image_dir / "slot_coverage.png", coverage, completed=False)
    figure, axis = plt.subplots(figsize=(7, 5))
    incoming = [float(row["incoming_retired_energy_fraction"]) for row in retired]
    unexplained = [float(row["unexplained_retired_energy_fraction"]) for row in retired]
    axis.scatter(incoming, unexplained, s=5, alpha=0.2)
    axis.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="gray", linewidth=1)
    axis.set(xlabel="incoming retired-energy fraction", ylabel="unexplained retired-energy fraction", title="Incoming versus unexplained retired energy")
    figure.tight_layout(); figure.savefig(image_dir / "incoming_vs_unexplained_energy.png", dpi=160); plt.close(figure)


def _median_line_plot(plt: Any, path: Path, rows: list[dict[str, Any]], x_key: str, y_key: str, title: str, xlabel: str, ylabel: str) -> None:
    grouped: dict[int, list[float]] = {}
    for row in rows:
        if row.get("experiment_phase") == "13A" and row.get("dataset_split", "training") == "training" and math.isclose(float(row.get("percentile", 0.0)), 0.5):
            grouped.setdefault(int(float(row[x_key])), []).append(float(row[y_key]))
    x = sorted(grouped); y = [_quantile(grouped[value], 0.5) for value in x]
    figure, axis = plt.subplots(figsize=(7, 4)); axis.plot(x, y, marker="o"); axis.set(xlabel=xlabel, ylabel=ylabel, title=title); figure.tight_layout(); figure.savefig(path, dpi=160); plt.close(figure)


def _coverage_plot(plt: Any, path: Path, rows: list[dict[str, Any]], *, completed: bool) -> None:
    figure, axis = plt.subplots(figsize=(7, 4))
    for epsilon in CANDIDATE_EPSILONS:
        grouped: dict[int, list[float]] = {}
        for row in rows:
            slot = row.get("active_atom_slot")
            is_completed = slot in {None, "", "None"}
            if row.get("experiment_phase") != "13A" or row.get("dataset_split") != "training" or is_completed != completed or not math.isclose(float(row["epsilon"]), epsilon):
                continue
            key = int(float(row["residual_layer"] if completed else slot))
            grouped.setdefault(key, []).append(float(row["resolved_fraction"]))
        x = sorted(grouped)
        if x:
            axis.plot(x, [_quantile(grouped[value], 0.5) for value in x], marker="o", label=f"{epsilon:g}")
    axis.set(
        xlabel="residual layer" if completed else "active atom slot",
        ylabel="median reconstructed fraction",
        title="Completed-layer reconstructed fraction" if completed else "Slot-level reconstructed fraction",
    )
    axis.legend(); figure.tight_layout(); figure.savefig(path, dpi=160); plt.close(figure)


def _best_row_sentence(phase: str, rows: list[dict[str, Any]]) -> str:
    if not rows:
        return f"No completed {phase} rows were available."
    row = rows[0]
    epsilon = f" at epsilon `{row.get('eligibility_epsilon')}`" if phase == "13B" else ""
    return f"The lowest validation P95 RMSE in {phase} is `{row.get('validation_p95_rmse')}` from `{row.get('construction_policy')}` with `{row.get('utility_candidate_budget')}` and `{row.get('layer_normalization_policy')}`{epsilon}; its validation median is `{row.get('validation_median_rmse')}`."


def _top_rows_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["| Policy | Budget | Normalization | Eligibility epsilon | Median RMSE | Strict perfect | P95 RMSE | Node-max P95 |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in rows[:10]:
        lines.append(
            f"| {row.get('construction_policy', '')} | {row.get('utility_candidate_budget', '')} | {row.get('layer_normalization_policy', '')} | "
            f"{row.get('eligibility_epsilon', '')} | "
            f"{row.get('validation_median_rmse', '')} | {row.get('validation_strict_perfect_lfo_rate', '')} | {row.get('validation_p95_rmse', '')} | {row.get('validation_node_max_error_p95', '')} |"
        )
    return lines


def _factor_table(rows: list[dict[str, Any]], field: str) -> list[str]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        value = str(row.get(field, "Null") or "Null")
        grouped.setdefault((str(row.get("experiment_phase", "")), value), []).append(float(row["validation_p95_rmse"]))
    lines = [f"| Phase | {field} | Median validation P95 RMSE | Rows |", "|---|---|---:|---:|"]
    for (phase, value), metrics in sorted(grouped.items()):
        lines.append(f"| {phase} | {value} | {_quantile(metrics, 0.5):.8g} | {len(metrics)} |")
    return lines


def _paired_delta_sentence(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No complete stable pairs were available."
    median = _quantile([float(row["median_rmse_delta_13b_minus_13a"]) for row in rows], 0.5)
    p95 = _quantile([float(row["p95_rmse_delta_13b_minus_13a"]) for row in rows], 0.5)
    return f"Across stable pairs, the median 13B-minus-13A change is `{median:.8g}` for validation median RMSE and `{p95:.8g}` for validation P95 RMSE."


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * float(q)
    lower, upper = int(math.floor(position)), int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _strategy_progress_percent(message: str) -> float:
    match = re.search(r"residual layer (\d+)/16", message)
    if match:
        return min(70.0, 70.0 * int(match.group(1)) / 16.0)
    if message.startswith("train encoding"):
        return 82.0
    if message.startswith("validation encoding"):
        return 92.0
    if message.startswith("partial_validation"):
        return 96.0
    return 1.0


def _reset_run_outputs(output_dir: Path) -> None:
    names = {
        "manifest.json", "summary.csv", "strategy_results.csv", "slot_progression.csv",
        "partial_codebook_validation.csv", "atom_construction.csv", "atom_assignments.csv",
        "candidate_search_diagnostics.csv", "layer_epsilon_quantiles.csv", "slot_epsilon_quantiles.csv",
        "epsilon_coverage.csv", "retired_error_mass.csv", "epsilon_selection.json",
        "epsilon_selection_status.json", "experiment13a_status.json", "experiment13b_status.json",
        "budget_accounting.csv", "failures.csv", "run_status.json", "run_events.jsonl",
        "paired_strategy_deltas.csv", "experiment13b_pilot_manifest.json", "experiment13b_pilot_results.csv",
    }
    for name in names:
        (Path(output_dir) / name).unlink(missing_ok=True)


def _require_fresh_output_dir(output_dir: Path) -> None:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise Experiment13Error(
            f"fresh 13A output directory must be absent or empty; use --resume or a new path: {output_dir}"
        )


def _missing_row_files(row_dir: Path) -> list[str]:
    row_dir = Path(row_dir)
    return [name for name in REQUIRED_ROW_FILES if not (row_dir / name).is_file() or (row_dir / name).stat().st_size == 0]


def _row_is_complete(row_dir: Path) -> bool:
    return Path(row_dir).is_dir() and not _missing_row_files(row_dir)


def _archive_incomplete_row(row_dir: Path, archive_root: Path, reason: str) -> Path:
    row_dir = Path(row_dir)
    archive_root = Path(archive_root)
    archive_root.mkdir(parents=True, exist_ok=True)
    target = archive_root / f"{row_dir.name}_{reason}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:6]}"
    shutil.move(str(row_dir), str(target))
    return target


def request_cancel(run_dir: Path, *, reason: str) -> dict[str, Any]:
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise Experiment13Error(f"run directory does not exist: {run_dir}")
    payload = {"requested_at_utc": _now(), "reason": str(reason).strip() or "operator requested cancellation"}
    _write_json(run_dir / "cancel_request.json", payload)
    return payload


def _check_cancel_requested(output_dir: Path) -> None:
    path = Path(output_dir) / "cancel_request.json"
    if path.exists():
        payload = _read_json(path)
        raise RunCancelled(str(payload.get("reason") or "operator requested cancellation"))


def _archive_cancel_request(output_dir: Path) -> None:
    path = Path(output_dir) / "cancel_request.json"
    if not path.exists():
        return
    root = Path(output_dir) / "cancel_requests"
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"cancel_request_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:6]}.json"
    path.replace(target)


def _git_provenance() -> dict[str, Any]:
    def command(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL, timeout=10
            ).strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    status = command("status", "--porcelain")
    return {
        "commit": command("rev-parse", "HEAD"),
        "branch": command("branch", "--show-current"),
        "dirty": bool(status),
    }


def load_epsilon_selection(
    path: Path,
    *,
    expected_run_identity: str,
    expected_configuration_fingerprint: str,
    require_passed: bool,
) -> EpsilonSelection:
    path = Path(path)
    if not path.exists():
        raise SelectionArtifactError(f"missing epsilon selection artifact: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SelectionArtifactError(f"invalid epsilon selection artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise SelectionArtifactError("epsilon selection artifact must contain a JSON object")
    selection = EpsilonSelection.from_dict(payload)
    if selection.candidate_epsilons != CANDIDATE_EPSILONS:
        raise SelectionArtifactError("epsilon selection candidate set is incompatible with Experiment 13")
    if selection.selection_rule_version != SELECTION_RULE_VERSION:
        raise SelectionArtifactError("epsilon selection rule version is incompatible with Experiment 13")
    if selection.experiment13a_run_identity != expected_run_identity:
        raise SelectionArtifactError("epsilon selection artifact is stale for the current 13A run")
    if selection.configuration_fingerprint != expected_configuration_fingerprint:
        raise SelectionArtifactError("epsilon selection artifact does not match the current Experiment 13 configuration")
    if require_passed and not selection.selection_passed:
        raise SelectionArtifactError("epsilon selection has not passed; full 13B is blocked")
    if selection.selection_passed:
        try:
            epsilon = _validate_epsilon(selection.selected_epsilon)
        except ValueError as exc:
            raise SelectionArtifactError("selected_epsilon must be a positive finite number") from exc
        if epsilon not in CANDIDATE_EPSILONS:
            raise SelectionArtifactError("selected_epsilon is not in the configured candidate set")
        if selection.selection_override and not (
            _is_nonempty_string(selection.selection_override_rationale)
            and _is_nonempty_string(selection.selection_override_timestamp)
            and isinstance(selection.pilot_evidence, (dict, list))
            and bool(selection.pilot_evidence)
        ):
            raise SelectionArtifactError("selection override requires rationale, timestamp, and pilot evidence")
    elif selection.selected_epsilon is not None:
        raise SelectionArtifactError("selected_epsilon must be null while selection_passed=false")
    return selection


def validate_completed_13a(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = Path(run_dir) / "manifest.json"
    if not manifest_path.exists():
        raise PhaseGateError(f"missing Experiment 13 manifest: {manifest_path}")
    manifest = _read_json(manifest_path)
    phase = manifest.get("phases", {}).get("13A")
    status = read_phase_status(run_dir, "13A")
    if not isinstance(phase, dict) or status["state"] != "complete":
        raise PhaseGateError(f"Experiment 13A is not complete: state={status['state']}")
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("experiment_id") != EXPERIMENT_ID:
        raise PhaseGateError("Experiment 13A manifest schema or experiment identity is incompatible")
    if (
        status.get("schema_version") != SCHEMA_VERSION
        or status.get("experiment_id") != EXPERIMENT_ID
        or status.get("experiment_phase") != "13A"
    ):
        raise PhaseGateError("Experiment 13A status schema or experiment identity is incompatible")
    if status.get("experiment13a_run_identity") != manifest.get("experiment13a_run_identity"):
        raise PhaseGateError("Experiment 13A status does not match the manifest run identity")
    if status.get("smoke") or status.get("filtered"):
        raise PhaseGateError("smoke and filtered 13A runs cannot satisfy the completion gate")
    if (
        status.get("row_count") != 90
        or status.get("completed_rows") != 90
        or status.get("failed_rows") != 0
        or status.get("expected_row_count") != 90
        or not str(status.get("completed_at_utc") or "").strip()
    ):
        raise PhaseGateError("Experiment 13A completion gate requires all 90 rows")
    if (
        not phase.get("complete_design")
        or phase.get("smoke")
        or phase.get("filtered")
        or phase.get("row_count") != 90
        or phase.get("expected_row_count") != 90
    ):
        raise PhaseGateError("Experiment 13A manifest does not represent the complete 90-row design")
    fingerprint = configuration_fingerprint()
    if manifest.get("configuration_fingerprint") != fingerprint:
        raise PhaseGateError("Experiment 13A manifest uses an incompatible configuration fingerprint")
    _validate_completed_13a_rows(
        phase.get("rows"),
        run_identity=str(manifest.get("experiment13a_run_identity") or ""),
        fingerprint=fingerprint,
    )
    return manifest, status


def validate_completed_13b(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest, _ = validate_completed_13a(run_dir)
    status = read_phase_status(run_dir, "13B")
    phase = manifest.get("phases", {}).get("13B")
    if not isinstance(phase, dict) or status["state"] != "complete":
        raise PhaseGateError(f"Experiment 13B is not complete: state={status['state']}")
    if (
        status.get("experiment13a_run_identity") != manifest.get("experiment13a_run_identity")
        or status.get("row_count") != 135
        or status.get("expected_row_count") != 135
        or status.get("completed_rows") != 135
        or status.get("failed_rows") != 0
        or status.get("smoke")
        or status.get("filtered")
        or not str(status.get("completed_at_utc") or "").strip()
    ):
        raise PhaseGateError("Experiment 13B completion gate requires all 135 sweep rows")
    if (
        not phase.get("complete_design")
        or phase.get("row_count") != 135
        or phase.get("expected_row_count") != 135
        or phase.get("smoke")
        or phase.get("filtered")
        or phase.get("eligibility_epsilon_sweep_version") != EXPERIMENT13B_SWEEP_VERSION
        or not _json_equivalent(
            phase.get("eligibility_epsilon_sweep"), list(EXPERIMENT13B_ELIGIBILITY_EPSILONS)
        )
    ):
        raise PhaseGateError("Experiment 13B manifest does not represent the complete fixed epsilon sweep")
    run_identity = str(manifest.get("experiment13a_run_identity") or "")
    fingerprint = str(manifest.get("configuration_fingerprint") or "")
    expected_by_id = {
        row.row_id: row.manifest_dict(run_identity, fingerprint) for row in experiment13b_specs()
    }
    rows = phase.get("rows")
    if not isinstance(rows, list) or len(rows) != 135:
        raise PhaseGateError("Experiment 13B manifest must contain exactly 135 row manifests")
    actual_by_id = {
        str(row.get("row_id")): row for row in rows if isinstance(row, dict) and row.get("row_id")
    }
    if len(actual_by_id) != 135 or set(actual_by_id) != set(expected_by_id):
        raise PhaseGateError("Experiment 13B manifest row set is incomplete or incompatible")
    for row_id, expected in expected_by_id.items():
        actual = actual_by_id[row_id]
        mismatched = [key for key, value in expected.items() if not _json_equivalent(actual.get(key), value)]
        if mismatched:
            raise PhaseGateError(
                f"Experiment 13B row manifest is incompatible for {row_id}: {', '.join(mismatched)}"
            )
    return manifest, status


def read_phase_status(run_dir: Path, phase: ExperimentPhase) -> dict[str, Any]:
    path = Path(run_dir) / PHASE_STATUS_FILES[phase]
    if not path.exists():
        return _phase_status(
            phase, "not_started", "", 0, PHASE_EXPECTED_ROW_COUNTS[phase],
            False, False, "phase has not started",
        )
    payload = _read_json(path)
    if payload.get("state") not in {"not_started", "running", "partial", "blocked", "failed", "cancelled", "complete"}:
        raise Experiment13Error(f"invalid phase state in {path}")
    return payload


def status_text(run_dir: Path = DEFAULT_OUTPUT_DIR) -> str:
    run_dir = Path(run_dir)
    status_a, status_b = read_phase_status(run_dir, "13A"), read_phase_status(run_dir, "13B")
    selection_path = run_dir / "epsilon_selection.json"
    selection_state, selected = "not_started", None
    if selection_path.exists():
        try:
            payload = _read_json(selection_path)
            selection_state = "passed" if payload.get("selection_passed") else "not_passed"
            selected = payload.get("selected_epsilon")
        except Experiment13Error:
            selection_state = "invalid"
    lines = [
        f"run_dir={run_dir}",
        _status_line("13A", status_a),
        f"epsilon_selection={selection_state}" + (f" selected_epsilon={selected}" if selected is not None else ""),
        _status_line("13B", status_b),
    ]
    if status_a["state"] != "complete" and status_b["state"] == "not_started":
        lines.append("13B_gate=blocked until a complete full 13A run and valid epsilon_selection.json exist")
    active = status_a if status_a.get("state") == "running" else status_b if status_b.get("state") == "running" else None
    if active is not None:
        lines.append(
            f"Current: {active.get('current_row_id') or 'starting'} "
            f"{float(active.get('current_task_percent', 0.0)):.1f}% - {active.get('current_task_phase', 'waiting')}"
        )
    return "\n".join(lines)


def _write_phase_manifest(
    *,
    output_dir: Path,
    phase: ExperimentPhase,
    specs: Sequence[StrategyRowSpec],
    run_identity: str,
    fingerprint: str,
    metadata_path: Path,
    backend: str,
    smoke: bool,
    train_sample_fraction: float,
    validation_sample_fraction: float,
    sample_seed: int,
    sample: SampleProvenance,
    dataset_cache: dict[str, Any],
    base_stage_cache_key: str,
    complete_design: bool,
    resume: bool,
) -> None:
    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.exists() else {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "experiment13a_run_identity": run_identity,
        "configuration_fingerprint": fingerprint,
        "phases": {},
    }
    if phase == "13B" and (
        manifest.get("experiment13a_run_identity") != run_identity
        or manifest.get("configuration_fingerprint") != fingerprint
    ):
        raise PhaseGateError("13B manifest update does not match the completed 13A run")
    manifest["experiment13a_run_identity"] = run_identity
    manifest["configuration_fingerprint"] = fingerprint
    rows = [row.manifest_dict(run_identity, fingerprint) for row in specs]
    expected_row_count = PHASE_EXPECTED_ROW_COUNTS[phase]
    phase_manifest = {
        "experiment_phase": phase,
        "row_count": len(rows),
        "expected_row_count": expected_row_count,
        "complete_design": complete_design,
        "smoke": smoke,
        "filtered": len(rows) != expected_row_count,
        "metadata_path": str(Path(metadata_path)),
        "backend": backend,
        "train_sample_fraction": float(train_sample_fraction),
        "validation_sample_fraction": float(validation_sample_fraction),
        "sample_seed": int(sample_seed),
        "sample_fingerprint": sample.sample_fingerprint,
        "sample_policy_version": sample.policy_version,
        "source_fingerprint": sample.source_fingerprint,
        "dataset_cache": dataset_cache,
        "base_stage_cache_key": base_stage_cache_key,
        "optimization_version": OPTIMIZATION_VERSION,
        "implementation_fingerprint": implementation_fingerprint(),
        "keep_awake": {
            "runner_scoped_system_required": os.name == "nt",
            "power_toys_controlled_by_runner": False,
            "power_toys_awake_enabled_externally": True,
        },
        "git_provenance": _git_provenance(),
        "resumed": bool(resume),
        "rows": rows,
    }
    if phase == "13B":
        phase_manifest.update(
            eligibility_epsilon_sweep=list(EXPERIMENT13B_ELIGIBILITY_EPSILONS),
            eligibility_epsilon_sweep_version=EXPERIMENT13B_SWEEP_VERSION,
            epsilon_axis_kind="exploratory_fixed_sweep",
        )
    manifest["phases"][phase] = phase_manifest
    _write_json(manifest_path, manifest)


def _phase_status(
    phase: ExperimentPhase,
    state: PhaseState,
    run_identity: str,
    row_count: int,
    expected_row_count: int,
    smoke: bool,
    filtered: bool,
    reason: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "experiment_phase": phase,
        "state": state,
        "experiment13a_run_identity": run_identity,
        "row_count": row_count,
        "expected_row_count": expected_row_count,
        "completed_rows": 0,
        "failed_rows": 0,
        "smoke": smoke,
        "filtered": filtered,
        "reason": reason,
        "started_at_utc": None if state == "not_started" else _now(),
        "completed_at_utc": None,
    }


def _write_phase_status(output_dir: Path, status: dict[str, Any]) -> None:
    _write_json(Path(output_dir) / PHASE_STATUS_FILES[status["experiment_phase"]], status)
    run_path = Path(output_dir) / "run_status.json"
    run = _read_json(run_path) if run_path.exists() else {"schema_version": SCHEMA_VERSION, "phases": {}}
    run["phases"][status["experiment_phase"]] = status
    _write_json(run_path, run)


def _validate_13b_invocation(
    manifest: dict[str, Any],
    metadata_path: Path,
    train_fraction: float,
    validation_fraction: float,
    sample_seed: int,
) -> None:
    phase = manifest.get("phases", {}).get("13A", {})
    if Path(str(phase.get("metadata_path", ""))).resolve(strict=False) != Path(metadata_path).resolve(strict=False):
        raise PhaseGateError("13B metadata path does not match the completed 13A run")
    expected = (
        float(phase.get("train_sample_fraction", -1.0)),
        float(phase.get("validation_sample_fraction", -1.0)),
        int(phase.get("sample_seed", -1)),
    )
    actual = (float(train_fraction), float(validation_fraction), int(sample_seed))
    if expected != actual:
        raise PhaseGateError("13B sample fractions or seed do not match the completed 13A run")


def _blocked_13b(output_dir: Path, reason: str) -> None:
    status = _phase_status(
        "13B", "blocked", "", 0, PHASE_EXPECTED_ROW_COUNTS["13B"],
        False, False, reason,
    )
    _write_phase_status(output_dir, status)
    _failure(output_dir, "13B", reason)


def _filter_rows(rows: Sequence[StrategyRowSpec], row_ids: set[str] | None) -> list[StrategyRowSpec]:
    if row_ids is None:
        return list(rows)
    known = {row.row_id for row in rows}
    unknown = row_ids - known
    if unknown:
        raise ValueError("unknown Experiment 13 row ids: " + ", ".join(sorted(unknown)))
    selected = [row for row in rows if row.row_id in row_ids]
    if not selected:
        raise ValueError("no Experiment 13 rows selected")
    return selected


def _validate_run_options(
    smoke: bool,
    train_fraction: float,
    validation_fraction: float,
    backend: str,
    verify_optimized_kernels: str = "off",
) -> None:
    if backend not in {"auto", "numpy", "xpu"}:
        raise ValueError(f"unsupported backend: {backend}")
    if smoke and (train_fraction != 1.0 or validation_fraction != 1.0):
        raise ValueError("smoke cannot be combined with sample fractions")
    for name, value in (
        ("train_sample_fraction", train_fraction),
        ("validation_sample_fraction", validation_fraction),
    ):
        if not 0.0 < float(value) <= 1.0:
            raise ValueError(f"{name} must be in (0, 1]")
    if verify_optimized_kernels not in OPTIMIZED_KERNEL_MODES:
        raise ValueError("verify_optimized_kernels must be 'off' or 'first-use'")


def _resolve_sample_fractions(
    corpus_fraction: float | None,
    train_fraction: float,
    validation_fraction: float,
) -> tuple[float, float]:
    if corpus_fraction is None:
        return float(train_fraction), float(validation_fraction)
    if float(train_fraction) != 1.0 or float(validation_fraction) != 1.0:
        raise ValueError("--corpus-sample-fraction cannot be mixed with split-specific sample fractions")
    return float(corpus_fraction), float(corpus_fraction)


def _run_identity(output_dir: Path, resume: bool) -> str:
    path = output_dir / "manifest.json"
    if resume and path.exists():
        return str(_read_json(path).get("experiment13a_run_identity", "")) or f"x13a_{uuid4().hex[:12]}"
    return f"x13a_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"


def _pair_id(policy: str, budget: str | None, normalization: str) -> str:
    return f"x13_pair_{_slug(policy)}_{_slug(budget or 'Null')}_{_slug(normalization)}"


def _row_id(phase: ExperimentPhase, pair_id: str, epsilon: float | None = None) -> str:
    prefix = f"{'x13a' if phase == '13A' else 'x13b'}_{pair_id.removeprefix('x13_pair_')}"
    if phase == "13A":
        return prefix
    value = f"{_validate_epsilon(epsilon):.0e}".replace("-", "_").replace("+", "")
    return f"{prefix}_epsilon_{value}"


def _slug(value: str) -> str:
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def _budget_value(value: str | None) -> int | None:
    if value is None:
        return None
    if not value.startswith("CandidateBudget"):
        raise ValueError(f"unsupported utility candidate budget: {value}")
    return int(value.removeprefix("CandidateBudget"))


def _validate_epsilon(value: float | None) -> float:
    if value is None:
        raise ValueError("eligibility_epsilon is required")
    epsilon = float(value)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("eligibility_epsilon must be finite and positive")
    return epsilon


def _is_json_number(value: Any) -> bool:
    return type(value) in {int, float} and math.isfinite(float(value))


def _validate_selection_metadata(payload: dict[str, Any]) -> None:
    checkpoint = payload["selection_checkpoint_definition"]
    if not isinstance(checkpoint, dict) or not _json_equivalent(checkpoint, SELECTION_CHECKPOINT_DEFINITION):
        raise SelectionArtifactError("selection checkpoint definition is incompatible with Experiment 13")
    statistics = payload["training_statistics_used"]
    if (
        not isinstance(statistics, dict)
        or statistics.get("dataset_split") != "training"
        or type(statistics.get("row_count")) is not int
        or statistics["row_count"] != 90
    ):
        raise SelectionArtifactError("selection statistics must cover all 90 Experiment 13A training rows")
    for field in (
        "median_unexplained_retired_energy_fraction",
        "p95_unexplained_retired_energy_fraction",
    ):
        value = payload[field]
        if not _is_json_number(value) or not 0.0 <= float(value) <= 1.0:
            raise SelectionArtifactError(f"{field} must be a finite fraction in [0, 1]")
    retired_summary = payload["retired_lfo_fraction_summary"]
    if not isinstance(retired_summary, dict) or not retired_summary:
        raise SelectionArtifactError("retired_lfo_fraction_summary must be a non-empty JSON object")
    if (
        not isinstance(payload["experiment13a_run_identity"], str)
        or not payload["experiment13a_run_identity"].strip()
        or not isinstance(payload["configuration_fingerprint"], str)
        or not payload["configuration_fingerprint"].strip()
    ):
        raise SelectionArtifactError("selection artifact run identity and configuration fingerprint are required")
    if not isinstance(payload["selection_notes"], str):
        raise SelectionArtifactError("selection_notes must be a string")
    if payload["selection_passed"] and not _is_nonempty_string(payload["selection_timestamp"]):
        raise SelectionArtifactError("a passed selection requires selection_timestamp")
    if not payload["selection_passed"] and payload["selection_timestamp"] is not None:
        raise SelectionArtifactError("selection_timestamp must be null while selection_passed=false")
    if payload["selection_override"] and not payload["selection_passed"]:
        raise SelectionArtifactError("selection_override=true requires selection_passed=true")
    if payload["selection_passed"] and not payload["selection_override"]:
        if float(payload["median_unexplained_retired_energy_fraction"]) > 0.01 or float(
            payload["p95_unexplained_retired_energy_fraction"]
        ) > 0.05:
            raise SelectionArtifactError("automatic selection statistics do not satisfy the configured thresholds")


def _validate_completed_13a_rows(rows: Any, *, run_identity: str, fingerprint: str) -> None:
    if not isinstance(rows, list) or len(rows) != 90:
        raise PhaseGateError("Experiment 13A manifest must contain exactly 90 row manifests")
    expected_specs = experiment13a_specs()
    expected_by_id = {row.row_id: row.manifest_dict(run_identity, fingerprint) for row in expected_specs}
    actual_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("row_id"), str):
            raise PhaseGateError("Experiment 13A manifest contains a malformed row entry")
        row_id = row["row_id"]
        if row_id in actual_by_id:
            raise PhaseGateError(f"Experiment 13A manifest contains duplicate row_id: {row_id}")
        actual_by_id[row_id] = row
    if set(actual_by_id) != set(expected_by_id):
        raise PhaseGateError("Experiment 13A manifest row set is incomplete or incompatible")
    for row_id, expected in expected_by_id.items():
        actual = actual_by_id[row_id]
        mismatched = [key for key, value in expected.items() if not _json_equivalent(actual.get(key), value)]
        if mismatched:
            raise PhaseGateError(
                f"Experiment 13A row manifest is incompatible for {row_id}: {', '.join(mismatched)}"
            )


def _status_line(label: str, status: dict[str, Any]) -> str:
    return (
        f"{label}: state={status.get('state')} completed_rows={status.get('completed_rows', 0)}/"
        f"{status.get('expected_row_count', 90)} reason={status.get('reason', '')}"
    )


def _failure(output_dir: Path, phase: str, error: str, *, row_id: str = "") -> None:
    path = Path(output_dir) / "failures.csv"
    rows: list[dict[str, str]] = []
    if path.exists():
        with path.open(encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    rows.append({"experiment_phase": phase, "row_id": row_id, "error": error, "timestamp_utc": _now()})
    _write_csv(path, rows, ("experiment_phase", "row_id", "error", "timestamp_utc"))


def _event(output_dir: Path, phase: str, event: str, message: str) -> None:
    path = Path(output_dir) / "run_events.jsonl"
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    line = json.dumps({"timestamp_utc": _now(), "experiment_phase": phase, "event": event, "message": message})
    _write_text(path, existing + line + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)
    _replace(tmp, path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    _replace(tmp, path)


def _replace(tmp: Path, path: Path) -> None:
    for attempt in range(50):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            time.sleep(min(0.05 * (attempt + 1), 0.5))
    tmp.unlink(missing_ok=True)
    raise PermissionError(f"could not replace file: {path}")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Experiment13Error(f"invalid JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise Experiment13Error(f"JSON artifact must contain an object: {path}")
    return payload


def _nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _compare_scientific_row(baseline: Path, candidate: Path) -> list[str]:
    import numpy as np

    mismatches: list[str] = []
    baseline_npz, candidate_npz = baseline / "codebooks.npz", candidate / "codebooks.npz"
    with np.load(baseline_npz, allow_pickle=False) as left, np.load(candidate_npz, allow_pickle=False) as right:
        if left.files != right.files:
            mismatches.append("codebooks.npz: array names differ")
        else:
            for name in left.files:
                if not np.array_equal(left[name], right[name]):
                    mismatches.append(f"codebooks.npz:{name}")
    if _read_json(baseline / "targets_schema.json") != _read_json(candidate / "targets_schema.json"):
        mismatches.append("targets_schema.json")
    assignments = "atom_assignments.csv"
    if _file_sha256(baseline / assignments) != _file_sha256(candidate / assignments):
        mismatches.append(assignments)
    ignored = {"oracle_construction_time", "train_encoding_time", "validation_encoding_time"}
    csv_names = (
        "summary.csv", "atom_construction.csv", "candidate_search_diagnostics.csv",
        "slot_progression.csv", "partial_codebook_validation.csv", "layer_epsilon_quantiles.csv",
        "slot_epsilon_quantiles.csv", "epsilon_coverage.csv", "retired_error_mass.csv",
    )
    for name in csv_names:
        with (baseline / name).open(encoding="utf-8", newline="") as handle:
            left_rows = list(csv.DictReader(handle))
        with (candidate / name).open(encoding="utf-8", newline="") as handle:
            right_rows = list(csv.DictReader(handle))
        if len(left_rows) != len(right_rows):
            mismatches.append(f"{name}: row count")
            continue
        for index, (left, right) in enumerate(zip(left_rows, right_rows, strict=True)):
            differing = [
                key for key, value in left.items()
                if key not in ignored and (key not in right or right[key] != value)
            ]
            if differing:
                mismatches.append(f"{name}: row {index + 1}: {', '.join(differing[:8])}")
                break
    return mismatches


def _completed_row_summaries(run_dir: Path, phase: str) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for path in (Path(run_dir) / "rows").glob("*/summary.csv"):
        try:
            with path.open(encoding="utf-8", newline="") as handle:
                values = list(csv.DictReader(handle))
        except OSError:
            continue
        if len(values) == 1 and values[0].get("experiment_phase") == phase:
            rows[path.parent.name] = values[0]
    return rows


def _validation_membership_sha256(path: Path) -> str:
    indices: list[int] = []
    seen: set[int] = set()
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("dataset_split") != "validation":
                continue
            index = int(row["dataset_index"])
            if index not in seen:
                indices.append(index)
                seen.add(index)
    return hashlib.sha256(_canonical_json(indices).encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _json_equivalent(left: Any, right: Any) -> bool:
    try:
        return _canonical_json(left) == _canonical_json(right)
    except (TypeError, ValueError):
        return False


def _is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

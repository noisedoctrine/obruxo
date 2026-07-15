"""Experiment 13 execution scaffold for the fixed-W8D16 strategy grid."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Literal, Sequence
from uuid import uuid4

ExperimentPhase = Literal["13A", "13B"]
PhaseState = Literal["not_started", "partial", "blocked", "failed", "complete"]

EXPERIMENT_ID = "experiment_13"
SCHEMA_VERSION = "experiment13_strategy_grid_v1"
SELECTION_RULE_VERSION = "experiment13_epsilon_selection_v1"
BASE_DICTIONARY_SIZE = 32
W = 8
D = 16
CONTROL_POINT_COUNT = 97
RESERVED_ATOM = "NoOpAtom"
ACTIVE_ATOMS_PER_LAYER = 7
SCALAR_SCHEMA = "PhaseAndResidualGain"
PATH_SEARCH_POLICY = "Beam4Path"
NO_DAMAGE_POLICY = "NoDamageOff"
ATOM_PREPROCESSING_POLICY = "RawAtoms"
DUPLICATE_SUPPRESSION_POLICY = "DuplicateSuppressionOff"
FINISH_THRESHOLD = 1e-5
HEAD_OUTPUTS = 193
RUNTIME_TOPOLOGY = None
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


class ConstructionNotImplementedError(Experiment13Error):
    pass


class AnalysisNotReadyError(PhaseGateError):
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


def experiment13b_specs(eligibility_epsilon: float) -> list[StrategyRowSpec]:
    return _phase_specs("13B", _validate_epsilon(eligibility_epsilon))


def all_strategy_specs(eligibility_epsilon: float) -> list[StrategyRowSpec]:
    return experiment13a_specs() + experiment13b_specs(eligibility_epsilon)


def _phase_specs(phase: ExperimentPhase, epsilon: float | None) -> list[StrategyRowSpec]:
    rows: list[StrategyRowSpec] = []
    for policy in construction_policies():
        budgets = ("CandidateBudget24", "CandidateBudget48") if policy.repair_budget else (None,)
        for budget in budgets:
            for normalization in ("FinalClipOnly", "LayerClip0To1"):
                pair_id = _pair_id(policy.name, budget, normalization)
                row = StrategyRowSpec(
                    experiment_phase=phase,
                    row_id=_row_id(phase, pair_id),
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
                    eligibility_selection_rule_version=SELECTION_RULE_VERSION if phase == "13B" else None,
                )
                validate_row_spec(row)
                rows.append(row)
    if len(rows) != 90 or len({row.row_id for row in rows}) != 90 or len({row.pair_id for row in rows}) != 90:
        raise AssertionError(f"Experiment {phase} grid must contain 90 unique rows and pairs")
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
        _validate_epsilon(row.eligibility_epsilon)
        if row.eligibility_selection_rule_version != SELECTION_RULE_VERSION:
            raise ValueError("13B must carry the frozen selection-rule version")
    else:
        raise ValueError(f"unsupported experiment phase: {row.experiment_phase}")


def validate_pairing(rows_a: Sequence[StrategyRowSpec], rows_b: Sequence[StrategyRowSpec]) -> None:
    pairs_a = {row.pair_id: row for row in rows_a}
    pairs_b = {row.pair_id: row for row in rows_b}
    if set(pairs_a) != set(pairs_b) or len(pairs_a) != 90:
        raise ValueError("Experiment 13 requires exactly 90 stable pairs")
    for pair_id, left in pairs_a.items():
        right = pairs_b[pair_id]
        if left.paired_settings != right.paired_settings:
            raise ValueError(f"paired settings differ for {pair_id}")
        if left.residual_population_policy != "AllResiduals" or right.residual_population_policy != "UnresolvedOnly":
            raise ValueError(f"invalid population-policy pairing for {pair_id}")


def configuration_fingerprint() -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
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
    corpus_sample_fraction: float = 1.0,
    resume: bool = False,
    row_ids: set[str] | None = None,
) -> dict[str, str]:
    _validate_run_options(smoke, corpus_sample_fraction, backend)
    full_rows = experiment13a_specs()
    rows = _filter_rows(full_rows, row_ids)
    partial = smoke or row_ids is not None or len(rows) != 90
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_identity = _run_identity(output_dir, resume)
    fingerprint = configuration_fingerprint()
    _write_phase_manifest(
        output_dir=output_dir,
        phase="13A",
        specs=rows,
        run_identity=run_identity,
        fingerprint=fingerprint,
        metadata_path=metadata_path,
        backend=backend,
        smoke=smoke,
        corpus_sample_fraction=corpus_sample_fraction,
        complete_design=not partial,
    )
    status = _phase_status(
        phase="13A",
        state="partial",
        run_identity=run_identity,
        row_count=len(rows),
        expected_row_count=90,
        smoke=smoke,
        filtered=row_ids is not None,
        reason="preflight complete; numerical construction has not started",
    )
    _write_phase_status(output_dir, status)
    _event(output_dir, "13A", "preflight_complete", status["reason"])
    error = ConstructionNotImplementedError("Experiment 13A construction strategies are not implemented")
    status.update(state="partial" if partial else "failed", failed_at_utc=_now(), reason=str(error), error=str(error))
    _write_phase_status(output_dir, status)
    _failure(output_dir, "13A", str(error))
    _event(output_dir, "13A", "construction_not_implemented", str(error))
    raise error


def select_epsilon(*, run_dir: Path = DEFAULT_OUTPUT_DIR) -> EpsilonSelection:
    run_dir = Path(run_dir)
    validate_completed_13a(run_dir)
    missing = [name for name in REQUIRED_CALIBRATION_FILES if not _nonempty(run_dir / name)]
    if missing:
        reason = "missing required 13A calibration artifacts: " + ", ".join(missing)
        _write_json(run_dir / "epsilon_selection_status.json", {"state": "blocked", "reason": reason})
        _failure(run_dir, "select-epsilon", reason)
        raise PhaseGateError(reason)
    reason = "epsilon-selection computation is not implemented"
    _write_json(run_dir / "epsilon_selection_status.json", {"state": "failed", "reason": reason})
    _failure(run_dir, "select-epsilon", reason)
    raise ConstructionNotImplementedError(reason)


def run_13b(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    epsilon_selection_path: Path | None = None,
    metadata_path: Path = DEFAULT_METADATA,
    backend: str = "auto",
    smoke: bool = False,
    corpus_sample_fraction: float = 1.0,
    resume: bool = False,
    row_ids: set[str] | None = None,
) -> dict[str, str]:
    del resume
    _validate_run_options(smoke, corpus_sample_fraction, backend)
    output_dir = Path(output_dir)
    selection_path = Path(epsilon_selection_path or output_dir / "epsilon_selection.json")
    try:
        manifest, _ = validate_completed_13a(output_dir)
        selection = load_epsilon_selection(
            selection_path,
            expected_run_identity=manifest["experiment13a_run_identity"],
            expected_configuration_fingerprint=manifest["configuration_fingerprint"],
            require_passed=True,
        )
        _validate_13b_invocation(manifest, metadata_path, corpus_sample_fraction)
    except PhaseGateError as exc:
        _blocked_13b(output_dir, str(exc))
        raise
    assert selection.selected_epsilon is not None
    full_rows = experiment13b_specs(selection.selected_epsilon)
    rows = _filter_rows(full_rows, row_ids)
    partial = smoke or row_ids is not None or len(rows) != 90
    _write_phase_manifest(
        output_dir=output_dir,
        phase="13B",
        specs=rows,
        run_identity=selection.experiment13a_run_identity,
        fingerprint=selection.configuration_fingerprint,
        metadata_path=metadata_path,
        backend=backend,
        smoke=smoke,
        corpus_sample_fraction=corpus_sample_fraction,
        complete_design=not partial,
    )
    status = _phase_status(
        phase="13B",
        state="partial",
        run_identity=selection.experiment13a_run_identity,
        row_count=len(rows),
        expected_row_count=90,
        smoke=smoke,
        filtered=row_ids is not None,
        reason="selection gate passed; numerical construction has not started",
    )
    _write_phase_status(output_dir, status)
    error = ConstructionNotImplementedError("Experiment 13B construction strategies are not implemented")
    status.update(state="partial" if partial else "failed", failed_at_utc=_now(), reason=str(error), error=str(error))
    _write_phase_status(output_dir, status)
    _failure(output_dir, "13B", str(error))
    raise error


def run_13b_pilot(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    epsilon_selection_path: Path | None = None,
    candidate_epsilons: Sequence[float] = PILOT_EPSILONS,
    row_ids: set[str] | None = None,
    metadata_path: Path = DEFAULT_METADATA,
    backend: str = "auto",
) -> dict[str, str]:
    _validate_run_options(False, 1.0, backend)
    output_dir = Path(output_dir)
    manifest, _ = validate_completed_13a(output_dir)
    selection = load_epsilon_selection(
        Path(epsilon_selection_path or output_dir / "epsilon_selection.json"),
        expected_run_identity=manifest["experiment13a_run_identity"],
        expected_configuration_fingerprint=manifest["configuration_fingerprint"],
        require_passed=False,
    )
    _validate_13b_invocation(manifest, metadata_path, 1.0)
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
    _write_json(
        output_dir / "experiment13b_pilot_manifest.json",
        {
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
            "complete": False,
        },
    )
    reason = "Experiment 13B pilot construction is not implemented"
    _failure(output_dir, "13B-pilot", reason)
    raise ConstructionNotImplementedError(reason)


def analyze_strategy_grid(*, run_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, str]:
    run_dir = Path(run_dir)
    status_a, status_b = read_phase_status(run_dir, "13A"), read_phase_status(run_dir, "13B")
    if status_a["state"] != "complete" or status_b["state"] != "complete":
        raise AnalysisNotReadyError(
            f"analysis requires complete 13A and 13B phases; got 13A={status_a['state']} 13B={status_b['state']}"
        )
    missing = [name for name in REQUIRED_ANALYSIS_FILES if not _nonempty(run_dir / name)]
    if missing:
        raise AnalysisNotReadyError("analysis inputs are incomplete: " + ", ".join(missing))
    raise ConstructionNotImplementedError("Experiment 13 report generation is not implemented")


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


def read_phase_status(run_dir: Path, phase: ExperimentPhase) -> dict[str, Any]:
    path = Path(run_dir) / PHASE_STATUS_FILES[phase]
    if not path.exists():
        return _phase_status(phase, "not_started", "", 0, 90, False, False, "phase has not started")
    payload = _read_json(path)
    if payload.get("state") not in {"not_started", "partial", "blocked", "failed", "complete"}:
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
    corpus_sample_fraction: float,
    complete_design: bool,
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
    manifest["phases"][phase] = {
        "experiment_phase": phase,
        "row_count": len(rows),
        "expected_row_count": 90,
        "complete_design": complete_design,
        "smoke": smoke,
        "filtered": len(rows) != 90,
        "metadata_path": str(Path(metadata_path)),
        "backend": backend,
        "corpus_sample_fraction": float(corpus_sample_fraction),
        "rows": rows,
    }
    _write_json(manifest_path, manifest)
    for spec, row in zip(specs, rows, strict=True):
        _write_json(output_dir / "rows" / spec.row_id / "manifest.json", row)


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


def _validate_13b_invocation(manifest: dict[str, Any], metadata_path: Path, fraction: float) -> None:
    phase = manifest.get("phases", {}).get("13A", {})
    if Path(str(phase.get("metadata_path", ""))).resolve(strict=False) != Path(metadata_path).resolve(strict=False):
        raise PhaseGateError("13B metadata path does not match the completed 13A run")
    if float(phase.get("corpus_sample_fraction", -1.0)) != float(fraction):
        raise PhaseGateError("13B corpus sample fraction does not match the completed 13A run")


def _blocked_13b(output_dir: Path, reason: str) -> None:
    status = _phase_status("13B", "blocked", "", 0, 90, False, False, reason)
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


def _validate_run_options(smoke: bool, fraction: float, backend: str) -> None:
    if backend not in {"auto", "numpy", "xpu"}:
        raise ValueError(f"unsupported backend: {backend}")
    if smoke and fraction != 1.0:
        raise ValueError("smoke cannot be combined with a corpus sample fraction")
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("corpus_sample_fraction must be in (0, 1]")


def _run_identity(output_dir: Path, resume: bool) -> str:
    path = output_dir / "manifest.json"
    if resume and path.exists():
        return str(_read_json(path).get("experiment13a_run_identity", "")) or f"x13a_{uuid4().hex[:12]}"
    return f"x13a_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"


def _pair_id(policy: str, budget: str | None, normalization: str) -> str:
    return f"x13_pair_{_slug(policy)}_{_slug(budget or 'Null')}_{_slug(normalization)}"


def _row_id(phase: ExperimentPhase, pair_id: str) -> str:
    return f"{'x13a' if phase == '13A' else 'x13b'}_{pair_id.removeprefix('x13_pair_')}"


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


def _failure(output_dir: Path, phase: str, error: str) -> None:
    path = Path(output_dir) / "failures.csv"
    rows: list[dict[str, str]] = []
    if path.exists():
        with path.open(encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    rows.append({"experiment_phase": phase, "row_id": "", "error": error, "timestamp_utc": _now()})
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

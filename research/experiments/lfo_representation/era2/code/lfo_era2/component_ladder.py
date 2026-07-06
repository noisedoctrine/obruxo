"""Experiment 12 W8D16 first-principles component ladder."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np

from .accelerator import BackendPreference
from .accounting import RuntimeInterfaceSpec
from .alignment import alignment_matrix, best_alignment
from .assets import ReconstructionAssets
from .contracts import TopologyFlags, validate_topology_contract
from .curve import circular_shift
from .dataset import Era2CurveDataset, TOPOLOGY_NAMES, load_presetshare_curve_dataset
from .manifest import write_json, write_summary_csv
from .metrics import flat_atom_usage, reconstruction_summary
from .runner import DEFAULT_METADATA, ERA2_ROOT


BASE_DICTIONARY_SIZE = 32
W = 8
D = 16
CONTROL_POINT_COUNT = 97
DEFAULT_OUTPUT_DIR = ERA2_ROOT / "artifacts" / "experiment_12" / "component_ladder"
REPORT_PATH = ERA2_ROOT / "reports" / "EXPERIMENT_12_W8D16_COMPONENT_LADDER_REPORT.md"
REPORT_IMAGE_DIR = ERA2_ROOT / "reports" / "images" / "experiment_12"
SMOKE_TRAIN_COUNT = 48
SMOKE_VALIDATION_COUNT = 32
NO_OP_ATOM_INDEX = 0
NO_OP_ACTIVE_ATOMS = W - 1
PERFECT_MAX_ABS_EPS = 1e-5
NO_DAMAGE_MIN_IMPROVEMENT = 1e-10

SCALAR_SCHEMA_VALUES = ("IndicesOnly", "PhaseAndResidualGain")
PATH_SEARCH_POLICY_VALUES = ("Beam4Path", "Beam8Path")
CONSTRUCTION_POLICY_VALUES = (
    "BestOverallRepair",
    "FamilyBalancedRepair",
    "FinishMoreLfos",
    "FinishAndRepair",
    "AlternatingFinishRepair",
    "FinishRepairRescue",
    "CommonCaseRepair",
    "HardCaseRepair",
    "MetricBalancedRepair",
    "ShapeClusterRepair",
    "TuneAtomsAfterUse",
    "PathAwareRepair",
)
UTILITY_CANDIDATE_BUDGET_VALUES = ("CandidateBudget8", "CandidateBudget12", "CandidateBudget24", "CandidateBudget48")
LAYER_NORMALIZATION_POLICY_VALUES = (
    "FinalClipOnly",
    "LayerClip0To1",
    "LayerClipNeg0p1To1p1",
    "LayerClipNeg1To1",
    "LayerSoftClip0To1",
    "LayerSoftClipNeg0p1To1p1",
    "LayerCenterPreserveClip",
    "OvershootPenaltyNoClip",
    "BoundedResidualStep",
)
NO_DAMAGE_POLICY_VALUES = (
    "NoDamageOff",
    "LateLayerNoDamage",
    "PerfectLocking",
    "LateLayerNoDamageAndPerfectLocking",
)
ATOM_PREPROCESSING_POLICY_VALUES = ("RawAtoms", "EnergyNormalizedAtoms", "CenteredEnergyNormalizedAtoms")
DUPLICATE_SUPPRESSION_POLICY_VALUES = ("DuplicateSuppressionOff", "PhaseScaleDuplicateSuppression")

DEFAULT_SCREENING_VALUES = {
    "path_search_policy": "Beam4Path",
    "construction_policy": "BestOverallRepair",
    "utility_candidate_budget": "CandidateBudget24",
    "layer_normalization_policy": "FinalClipOnly",
    "no_damage_policy": "NoDamageOff",
    "atom_preprocessing_policy": "RawAtoms",
    "duplicate_suppression_policy": "DuplicateSuppressionOff",
}


@dataclass(frozen=True)
class ComponentRowSpec:
    row_id: str
    components: tuple[str, ...]
    description: str
    scalar_schema: str = "IndicesOnly"
    screening_variable: str = "control"
    screening_value: str = "Control"
    phase_enabled: bool = False
    residual_gain_enabled: bool = False
    beam_width: int = 1
    path_search_policy: str = "Beam4Path"
    construction_policy: str = "BestOverallRepair"
    topology_used_in_construction: bool = False
    max_utility_candidates: int = 24
    utility_candidate_budget: str = "CandidateBudget24"
    layer_normalization_policy: str = "FinalClipOnly"
    no_damage_policy: str = "NoDamageOff"
    atom_preprocessing_policy: str = "RawAtoms"
    duplicate_suppression_policy: str = "DuplicateSuppressionOff"

    @property
    def path_policy(self) -> str:
        return "beam" if self.beam_width > 1 else "greedy"

    @property
    def residual_gain_policy(self) -> str:
        return "optimized" if self.residual_gain_enabled else "fixed"

    @property
    def scalar_families(self) -> list[str]:
        families: list[str] = []
        if self.phase_enabled:
            families.append("phase")
        if self.residual_gain_enabled:
            families.append("residual_gain")
        return families

    @property
    def scalar_outputs(self) -> int:
        return (D + 1 if self.phase_enabled else 0) + (D if self.residual_gain_enabled else 0)


@dataclass
class ComponentEncoding:
    base_index: np.ndarray
    base_phase: np.ndarray
    base_gain: np.ndarray
    residual_layer_indices: list[np.ndarray]
    residual_layer_phases: list[np.ndarray]
    residual_layer_gains: list[np.ndarray]

    @property
    def row_count(self) -> int:
        return int(len(self.base_index))

    def index_arrays(self) -> dict[str, np.ndarray]:
        payload = {"base_index": self.base_index}
        for residual_layer, values in enumerate(self.residual_layer_indices, start=1):
            payload[f"residual_layer_{residual_layer}_index"] = values
        return payload

    def target_schema(self, spec: ComponentRowSpec) -> dict[str, Any]:
        fields = [{"name": "base_index", "kind": "categorical"}]
        if spec.phase_enabled:
            fields.append({"name": "base_phase", "kind": "continuous"})
        for residual_layer in range(D):
            number = residual_layer + 1
            fields.append({"name": f"residual_layer_{number}_index", "kind": "categorical"})
            if spec.phase_enabled:
                fields.append({"name": f"residual_layer_{number}_phase", "kind": "continuous"})
            if spec.residual_gain_enabled:
                fields.append({"name": f"residual_layer_{number}_gain", "kind": "continuous"})
        return {
            "runtime_interface_id": "flat_categorical_per_residual_layer",
            "row_count": self.row_count,
            "fields": fields,
        }


def default_component_specs() -> list[ComponentRowSpec]:
    specs: list[ComponentRowSpec] = []
    screening_grid = [
        ("path_search_policy", PATH_SEARCH_POLICY_VALUES),
        ("construction_policy", CONSTRUCTION_POLICY_VALUES),
        ("utility_candidate_budget", UTILITY_CANDIDATE_BUDGET_VALUES),
        ("layer_normalization_policy", LAYER_NORMALIZATION_POLICY_VALUES),
        ("no_damage_policy", NO_DAMAGE_POLICY_VALUES),
        ("atom_preprocessing_policy", ATOM_PREPROCESSING_POLICY_VALUES),
        ("duplicate_suppression_policy", DUPLICATE_SUPPRESSION_POLICY_VALUES),
    ]
    for variable, values in screening_grid:
        for value in values:
            for scalar_schema in SCALAR_SCHEMA_VALUES:
                specs.append(_screening_spec(variable, value, scalar_schema))
    return specs


def _screening_spec(variable: str, value: str, scalar_schema: str) -> ComponentRowSpec:
    values = {**DEFAULT_SCREENING_VALUES, variable: value}
    phase_enabled = scalar_schema == "PhaseAndResidualGain"
    residual_gain_enabled = scalar_schema == "PhaseAndResidualGain"
    beam_width = _beam_width(values["path_search_policy"])
    max_utility_candidates = _candidate_budget(values["utility_candidate_budget"])
    construction_policy = values["construction_policy"]
    components = tuple(
        item
        for item, enabled in [
            ("phase", phase_enabled),
            ("residual_gain", residual_gain_enabled),
            (values["path_search_policy"], True),
            (construction_policy, True),
            (values["layer_normalization_policy"], True),
            (values["no_damage_policy"], values["no_damage_policy"] != "NoDamageOff"),
            (values["atom_preprocessing_policy"], values["atom_preprocessing_policy"] != "RawAtoms"),
            (values["duplicate_suppression_policy"], values["duplicate_suppression_policy"] != "DuplicateSuppressionOff"),
        ]
        if enabled
    )
    return ComponentRowSpec(
        row_id=f"x12_screen_{variable}_{value}_{scalar_schema}",
        components=components,
        description=f"Screen {variable}={value} with scalar_schema={scalar_schema}.",
        scalar_schema=scalar_schema,
        screening_variable=variable,
        screening_value=value,
        phase_enabled=phase_enabled,
        residual_gain_enabled=residual_gain_enabled,
        beam_width=beam_width,
        path_search_policy=values["path_search_policy"],
        construction_policy=construction_policy,
        topology_used_in_construction=construction_policy == "FamilyBalancedRepair",
        max_utility_candidates=max_utility_candidates,
        utility_candidate_budget=values["utility_candidate_budget"],
        layer_normalization_policy=values["layer_normalization_policy"],
        no_damage_policy=values["no_damage_policy"],
        atom_preprocessing_policy=values["atom_preprocessing_policy"],
        duplicate_suppression_policy=values["duplicate_suppression_policy"],
    )


def validate_component_spec(spec: ComponentRowSpec) -> None:
    validate_residual_gain_contract(spec.residual_gain_policy, model_facing=spec.residual_gain_enabled)
    if spec.scalar_schema not in SCALAR_SCHEMA_VALUES:
        raise ValueError(f"unsupported scalar_schema: {spec.scalar_schema}")
    if spec.path_search_policy not in PATH_SEARCH_POLICY_VALUES:
        raise ValueError(f"unsupported path_search_policy: {spec.path_search_policy}")
    if spec.construction_policy not in CONSTRUCTION_POLICY_VALUES:
        raise ValueError(f"unsupported construction_policy: {spec.construction_policy}")
    if not spec.utility_candidate_budget.startswith("CandidateBudget"):
        raise ValueError(f"unsupported utility_candidate_budget: {spec.utility_candidate_budget}")
    if spec.layer_normalization_policy not in LAYER_NORMALIZATION_POLICY_VALUES:
        raise ValueError(f"unsupported layer_normalization_policy: {spec.layer_normalization_policy}")
    if spec.no_damage_policy not in NO_DAMAGE_POLICY_VALUES:
        raise ValueError(f"unsupported no_damage_policy: {spec.no_damage_policy}")
    if spec.atom_preprocessing_policy not in ATOM_PREPROCESSING_POLICY_VALUES:
        raise ValueError(f"unsupported atom_preprocessing_policy: {spec.atom_preprocessing_policy}")
    if spec.duplicate_suppression_policy not in DUPLICATE_SUPPRESSION_POLICY_VALUES:
        raise ValueError(f"unsupported duplicate_suppression_policy: {spec.duplicate_suppression_policy}")
    if spec.topology_used_in_construction and spec.construction_policy != "FamilyBalancedRepair":
        raise ValueError("topology_used_in_construction requires FamilyBalancedRepair construction")


def validate_residual_gain_contract(gain_policy: str, *, model_facing: bool) -> None:
    if gain_policy == "optimized" and not model_facing:
        raise ValueError("optimized per-sample residual gain must be model-facing in Experiment 12")
    if gain_policy not in {"fixed", "optimized"}:
        raise ValueError("gain_policy must be fixed or optimized")


def _beam_width(value: str) -> int:
    if value == "Beam4Path":
        return 4
    if value == "Beam8Path":
        return 8
    raise ValueError(f"unsupported path_search_policy: {value}")


def _candidate_budget(value: str) -> int:
    if not value.startswith("CandidateBudget"):
        raise ValueError(f"unsupported utility_candidate_budget: {value}")
    return int(value.removeprefix("CandidateBudget"))


def budget_for_spec(spec: ComponentRowSpec) -> dict[str, Any]:
    budget = RuntimeInterfaceSpec(
        addressing_scheme="flat_categorical",
        residual_layer_count=D,
        dictionary_scope="per_residual_layer",
        parameters={"width": W},
    ).budget(base_dictionary_size=BASE_DICTIONARY_SIZE, scalar_outputs=spec.scalar_outputs)
    payload = budget.as_dict()
    payload["head_outputs_formula"] = _head_formula(spec)
    return payload


def run_component_ladder(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    metadata_path: Path = DEFAULT_METADATA,
    backend: BackendPreference = "auto",
    smoke: bool = False,
    corpus_sample_fraction: float = 1.0,
    resume: bool = False,
    row_ids: set[str] | None = None,
    max_utility_candidates: int | None = None,
    chunk_size: int = 256,
    dataset: Era2CurveDataset | None = None,
    write_report: bool = True,
    report_path: Path = REPORT_PATH,
    report_image_dir: Path = REPORT_IMAGE_DIR,
    progress: Callable[[str], None] | None = None,
) -> dict[str, str]:
    if smoke and corpus_sample_fraction != 1.0:
        raise ValueError("smoke cannot be combined with corpus_sample_fraction other than 1.0")
    if not (0.0 < float(corpus_sample_fraction) <= 1.0):
        raise ValueError("corpus_sample_fraction must be in (0, 1]")
    specs = [spec for spec in default_component_specs() if row_ids is None or spec.row_id in row_ids]
    if max_utility_candidates is not None:
        specs = [
            ComponentRowSpec(
                **{
                    **asdict(spec),
                    "max_utility_candidates": int(max_utility_candidates),
                    "utility_candidate_budget": f"CandidateBudget{int(max_utility_candidates)}",
                }
            )
            for spec in specs
        ]
    if not specs:
        raise ValueError("no Experiment 12 rows selected")
    for spec in specs:
        validate_component_spec(spec)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not resume:
        _reset_run_logs(output_dir)
    status = _initial_status(specs, smoke=smoke, corpus_sample_fraction=corpus_sample_fraction)
    _write_status(output_dir, status)
    _event(output_dir, status, "run_start", f"rows={len(specs)} backend={backend} smoke={smoke}")

    if dataset is None:
        _log(progress, f"experiment12: dataset_load_start metadata={metadata_path} control_point_count={CONTROL_POINT_COUNT}")
        dataset = load_presetshare_curve_dataset(
            metadata_path,
            resolution=CONTROL_POINT_COUNT,
            x_grid_mode="inclusive",
            progress=progress,
        )
    dataset = _subset_dataset(dataset, smoke=smoke, corpus_sample_fraction=corpus_sample_fraction)
    _log(progress, "experiment12: dataset_ready " + " ".join(f"{k}={v}" for k, v in dataset.manifest_fields().items()))

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, spec in enumerate(specs, start=1):
        row_dir = output_dir / "rows" / spec.row_id
        if resume and (row_dir / "summary.csv").exists():
            row = _read_one_csv(row_dir / "summary.csv")
            rows.append(row)
            _mark_row_complete(status, spec.row_id, index, row)
            _event(output_dir, status, "row_skipped", f"{index}/{len(specs)} already completed", row_id=spec.row_id)
            continue
        started = time.perf_counter()
        try:
            status["current_row_id"] = spec.row_id
            status["current_row_number"] = index
            status["current_task_id"] = spec.row_id
            status["current_task_number"] = index
            status["current_task_phase"] = "starting"
            status["current_task_percent"] = 0.0
            status["rows"][spec.row_id] = {"status": "running", "started_at_utc": _now()}
            _set_overall_progress(status)
            _write_status(output_dir, status)
            _event(output_dir, status, "row_start", f"{index}/{len(specs)} {spec.row_id}", row_id=spec.row_id)

            def row_progress(message: str) -> None:
                status["current_task_phase"] = _readable_phase(message)
                status["current_task_percent"] = _progress_percent(message)
                _write_status(output_dir, status)
                _log(progress, f"experiment12: {spec.row_id}: {message}")

            row = _run_row(
                spec,
                dataset,
                row_dir,
                backend=backend,
                chunk_size=chunk_size,
                progress=row_progress,
            )
            row["row_elapsed_seconds"] = time.perf_counter() - started
            row["row_number"] = index
            row["row_count"] = len(specs)
            write_summary_csv(row_dir / "summary.csv", row)
            rows.append(row)
            _mark_row_complete(status, spec.row_id, index, row)
            _event(
                output_dir,
                status,
                "row_complete",
                f"{index}/{len(specs)} validation_p95={row.get('validation_p95_rmse')} elapsed={row['row_elapsed_seconds']:.2f}s",
                row_id=spec.row_id,
            )
            _log(progress, f"experiment12: [{index}/{len(specs)}] {spec.row_id} validation_p95={row.get('validation_p95_rmse')}")
        except Exception as exc:
            failures.append({"row_id": spec.row_id, "error": str(exc)})
            status["rows"][spec.row_id] = {"status": "failed", "failed_at_utc": _now(), "error": str(exc)}
            _set_overall_progress(status)
            _write_status(output_dir, status)
            _event(output_dir, status, "row_failed", str(exc), row_id=spec.row_id)
            raise

    status["current_row_id"] = ""
    status["current_row_number"] = ""
    status["current_task_id"] = ""
    status["current_task_number"] = ""
    status["current_task_phase"] = "complete"
    status["current_task_percent"] = 100.0
    status["completed_at_utc"] = _now()
    _set_overall_progress(status)
    _write_status(output_dir, status)

    result = analyze_component_ladder(
        output_dir=output_dir,
        rows=rows,
        failures=failures,
        write_report=write_report,
        report_path=report_path,
        report_image_dir=report_image_dir,
    )
    _event(output_dir, status, "run_complete", f"report={result.get('report', '')}")
    return result


def analyze_component_ladder(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    rows: list[dict[str, Any]] | None = None,
    failures: list[dict[str, Any]] | None = None,
    write_report: bool = True,
    report_path: Path = REPORT_PATH,
    report_image_dir: Path = REPORT_IMAGE_DIR,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    current_row_ids = {spec.row_id for spec in default_component_specs()}
    if rows is None:
        rows = [row for row in _load_row_summaries(output_dir) if str(row.get("row_id", "")) in current_row_ids]
    else:
        rows = [row for row in rows if str(row.get("row_id", "")) in current_row_ids]
    failures = [] if failures is None else failures
    rows = sorted(rows, key=lambda row: int(float(row.get("row_number", 0) or 0)))
    summary_path = output_dir / "summary.csv"
    deltas_path = output_dir / "component_deltas.csv"
    budget_path = output_dir / "budget_accounting.csv"
    scalar_path = output_dir / "scalar_usage.csv"
    usage_path = output_dir / "atom_usage_diagnostics.csv"
    screening_path = output_dir / "screening_results.csv"
    failures_path = output_dir / "failures.csv"
    _write_csv(summary_path, rows)
    _write_csv(deltas_path, _component_deltas(rows))
    _write_csv(budget_path, [_budget_row(row) for row in rows])
    _write_csv(scalar_path, [_scalar_row(row) for row in rows])
    _write_csv(usage_path, [_usage_row(row) for row in rows])
    _write_csv(screening_path, [_screening_row(row) for row in rows])
    _write_csv(failures_path, failures, fieldnames=["row_id", "error"])
    if write_report:
        _write_plots(report_image_dir, rows)
        _write_report(report_path, rows)
    return {
        "output_dir": str(output_dir),
        "summary": str(summary_path),
        "component_deltas": str(deltas_path),
        "budget_accounting": str(budget_path),
        "scalar_usage": str(scalar_path),
        "atom_usage_diagnostics": str(usage_path),
        "screening_results": str(screening_path),
        "failures": str(failures_path),
        "report": str(report_path) if write_report else "",
        "report_image_dir": str(report_image_dir) if write_report else "",
    }


def status_text(run_dir: Path) -> str:
    path = Path(run_dir) / "run_status.json"
    if not path.exists():
        return f"missing status file: {path}"
    status = _read_json_with_retry(path)
    total = int(status.get("row_count", 0))
    rows = status.get("rows", {})
    completed = sum(1 for row in rows.values() if row.get("status") == "completed")
    failed = sum(1 for row in rows.values() if row.get("status") == "failed")
    current = status.get("current_task_id") or status.get("current_row_id") or ""
    phase = status.get("current_task_phase") or "waiting"
    percent = status.get("current_task_percent", 0.0)
    lines = [
        f"run_id={status.get('run_id', Path(run_dir).name)} experiment=12 elapsed={_format_duration(_elapsed_since(status.get('started_at_utc', '')))}",
        f"Overall: {completed}/{total} rows complete ({_numeric_percent(completed, total):.1f}%) failed={failed}",
        f"Current: {current or 'complete'} {float(percent):.1f}% - {phase}",
    ]
    if status.get("completed_at_utc"):
        lines.append(f"completed_at_utc={status['completed_at_utc']}")
    return "\n".join(lines)


def _run_row(
    spec: ComponentRowSpec,
    dataset: Era2CurveDataset,
    row_dir: Path,
    *,
    backend: BackendPreference,
    chunk_size: int,
    progress: Callable[[str], None] | None,
) -> dict[str, Any]:
    row_dir.mkdir(parents=True, exist_ok=True)
    flags = TopologyFlags(topology_used_in_construction=spec.topology_used_in_construction)
    contract = validate_topology_contract(flags)
    construction_started = time.perf_counter()
    _log(progress, "construction_start")
    assets = _construct_assets(spec, dataset.train_curves, dataset.topology[dataset.train_indices], backend=backend, chunk_size=chunk_size, progress=progress)
    construction_time = time.perf_counter() - construction_started
    _log(progress, f"construction_complete elapsed={construction_time:.2f}s")

    train_encoding, train_reconstructed, train_raw_reconstructed, train_encoding_time = _encode_decode(
        spec,
        dataset.train_curves,
        assets,
        backend=backend,
        chunk_size=chunk_size,
        progress=progress,
        progress_label="train",
    )
    validation_encoding, validation_reconstructed, validation_raw_reconstructed, validation_encoding_time = _encode_decode(
        spec,
        dataset.validation_curves,
        assets,
        backend=backend,
        chunk_size=chunk_size,
        progress=progress,
        progress_label="validation",
    )

    schema = validation_encoding.target_schema(spec)
    budget = budget_for_spec(spec)
    manifest = _manifest(spec, assets, dataset, budget, flags, construction_time, validation_encoding_time)
    summary = {
        **manifest,
        **_prefix("train", reconstruction_summary(dataset.train_curves, train_reconstructed)),
        **_prefix("validation", reconstruction_summary(dataset.validation_curves, validation_reconstructed)),
        **_usage_summary(validation_encoding, widths=assets.residual_widths()),
        **_scalar_summary(validation_encoding, spec),
        **_prefix("train", _overshoot_summary(train_raw_reconstructed)),
        **_prefix("validation", _overshoot_summary(validation_raw_reconstructed)),
        **_asset_diagnostics(assets),
        "topology_contract_pass": contract.passed,
        "runtime_contract_valid": contract.passed,
        "train_encoding_time": train_encoding_time,
        "validation_encoding_time": validation_encoding_time,
        "backend_preference": backend,
    }
    write_json(row_dir / "manifest.json", manifest)
    write_json(row_dir / "targets_schema.json", schema)
    write_json(row_dir / "topology_contract.json", contract.as_dict())
    return summary


def _construct_assets(
    spec: ComponentRowSpec,
    targets: np.ndarray,
    topology: np.ndarray,
    *,
    backend: BackendPreference,
    chunk_size: int,
    progress: Callable[[str], None] | None,
) -> ReconstructionAssets:
    phase_count = _phase_candidate_count(spec, targets.shape[1])
    base = _select_farthest_atoms(targets, width=BASE_DICTIONARY_SIZE, include_zero=False, topology=None)
    base_choice = best_alignment(
        targets,
        base,
        phase_policy="fft_lattice",
        gain_policy="fixed",
        backend=backend,
        chunk_size=chunk_size,
        phase_candidate_count=phase_count,
    )
    prefix = base_choice.values.copy()
    layers = []
    for residual_layer in range(D):
        if residual_layer == 0 or residual_layer + 1 == D or (residual_layer + 1) % 4 == 0:
            _log(progress, f"construction: residual layer {residual_layer + 1}/{D}")
        residual = targets - prefix
        atoms = _select_layer_atoms(residual, topology, spec=spec, backend=backend, chunk_size=chunk_size)
        layers.append(atoms)
        choice = best_alignment(
            residual,
            atoms,
            phase_policy="fft_lattice",
            gain_policy=spec.residual_gain_policy,
            backend=backend,
            chunk_size=chunk_size,
            phase_candidate_count=phase_count,
        )
        prefix = _apply_layer_state_policy(prefix, choice.values, spec)
    return ReconstructionAssets(
        base_dictionary=base,
        residual_layer_dictionaries=layers,
        dictionary_scope="per_residual_layer",
        metadata={"construction_policy": spec.construction_policy, "reserved_atom": "NoOpAtom", "active_atoms_per_layer": NO_OP_ACTIVE_ATOMS},
    )


def _select_layer_atoms(
    residual: np.ndarray,
    topology: np.ndarray,
    *,
    spec: ComponentRowSpec,
    backend: BackendPreference,
    chunk_size: int,
) -> np.ndarray:
    atoms = _select_repair_atoms(
        residual,
        width=W,
        topology=topology if spec.construction_policy == "FamilyBalancedRepair" else None,
        spec=spec,
        backend=backend,
        chunk_size=chunk_size,
    )
    if spec.construction_policy == "TuneAtomsAfterUse":
        atoms = _tune_atoms_after_use(residual, atoms, spec=spec, chunk_size=chunk_size)
    return atoms


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


def _select_repair_atoms(
    residual: np.ndarray,
    *,
    width: int,
    topology: np.ndarray | None,
    spec: ComponentRowSpec,
    backend: BackendPreference,
    chunk_size: int,
) -> np.ndarray:
    matrix = np.asarray(residual, dtype=np.float32)
    atoms = [np.zeros(matrix.shape[1], dtype=np.float32)]
    current_loss = np.mean(matrix * matrix, axis=1)
    selected: set[int] = set()
    phase_count = _phase_candidate_count(spec, matrix.shape[1])
    while len(atoms) < int(width):
        role = _construction_slot_role(spec.construction_policy, len(atoms))
        pool = _utility_candidate_pool(
            matrix,
            current_loss,
            selected,
            topology=topology,
            limit=spec.max_utility_candidates,
            role=role,
        )
        pool = _filter_duplicate_pool(matrix, pool, atoms, spec)
        if len(pool) == 0:
            atoms.append(np.zeros(matrix.shape[1], dtype=np.float32))
            continue
        candidates = _preprocess_candidates(matrix[pool], matrix, spec)
        losses = alignment_matrix(
            matrix,
            candidates,
            phase_policy="fft_lattice",
            gain_policy=spec.residual_gain_policy,
            backend=backend,
            chunk_size=chunk_size,
            phase_candidate_count=phase_count,
        ).losses
        improvement = np.maximum(current_loss[:, None] - losses, 0.0)
        chosen_local = int(np.argmax(_candidate_scores(improvement, current_loss, losses, role)))
        chosen = int(pool[chosen_local])
        selected.add(chosen)
        atoms.append(candidates[chosen_local].astype(np.float32))
        current_loss = np.minimum(current_loss, losses[:, chosen_local])
    return np.stack(atoms).astype(np.float32)


def _utility_candidate_pool(
    matrix: np.ndarray,
    current_loss: np.ndarray,
    selected: set[int],
    *,
    topology: np.ndarray | None,
    limit: int,
    role: str = "overall",
) -> np.ndarray:
    if role == "common":
        target = -np.abs(current_loss - np.median(current_loss))
    elif role in {"hard", "rescue"}:
        target = current_loss
    elif role == "finish":
        target = -current_loss
    elif role == "shape_cluster":
        target = _shape_cluster_priority(matrix, current_loss)
    else:
        target = current_loss
    if topology is None:
        order = np.argsort(target)[::-1]
        return np.asarray([index for index in order if int(index) not in selected][:limit], dtype=np.int32)
    chosen = []
    per_bucket = max(1, int(np.ceil(limit / len(TOPOLOGY_NAMES))))
    for bucket in range(len(TOPOLOGY_NAMES)):
        members = np.flatnonzero(topology == bucket)
        ordered = members[np.argsort(target[members])[::-1]]
        chosen.extend(int(index) for index in ordered if int(index) not in selected)
        if len(chosen) >= per_bucket * (bucket + 1):
            continue
    return np.asarray(chosen[:limit], dtype=np.int32)


def _construction_slot_role(policy: str, atom_slot: int) -> str:
    active_slot = max(1, int(atom_slot))
    if policy in {"BestOverallRepair", "FamilyBalancedRepair", "TuneAtomsAfterUse"}:
        return "overall"
    if policy == "PathAwareRepair":
        return "path_aware"
    if policy == "FinishMoreLfos":
        return "finish"
    if policy == "CommonCaseRepair":
        return "common"
    if policy == "HardCaseRepair":
        return "hard"
    if policy == "MetricBalancedRepair":
        return "balanced"
    if policy == "ShapeClusterRepair":
        return "shape_cluster"
    if policy == "FinishAndRepair":
        return "finish" if active_slot <= NO_OP_ACTIVE_ATOMS // 2 else "overall"
    if policy == "AlternatingFinishRepair":
        return "finish" if active_slot % 2 == 1 else "overall"
    if policy == "FinishRepairRescue":
        if active_slot <= 2:
            return "finish"
        if active_slot <= 5:
            return "common"
        return "hard"
    raise ValueError(f"unsupported construction_policy: {policy}")


def _candidate_scores(improvement: np.ndarray, current_loss: np.ndarray, losses: np.ndarray, role: str) -> np.ndarray:
    if role == "finish":
        previous_perfect = current_loss <= PERFECT_MAX_ABS_EPS**2
        next_perfect = losses <= PERFECT_MAX_ABS_EPS**2
        return np.sum(next_perfect & ~previous_perfect[:, None], axis=0) * 1_000_000.0 + np.sum(improvement, axis=0)
    if role == "common":
        return np.median(improvement, axis=0) * len(current_loss) + np.sum(np.minimum(improvement, np.median(current_loss)), axis=0)
    if role == "hard":
        threshold = np.quantile(current_loss, 0.90)
        mask = current_loss >= threshold
        return np.sum(improvement[mask], axis=0) if np.any(mask) else np.sum(improvement, axis=0)
    if role == "balanced":
        previous_perfect = current_loss <= PERFECT_MAX_ABS_EPS**2
        next_perfect = losses <= PERFECT_MAX_ABS_EPS**2
        finish = np.sum(next_perfect & ~previous_perfect[:, None], axis=0).astype(np.float64)
        median = np.median(improvement, axis=0)
        capped = np.sum(np.minimum(improvement, np.quantile(current_loss, 0.75)), axis=0)
        hard = _candidate_scores(improvement, current_loss, losses, "hard")
        return _normalize_score(finish) + _normalize_score(median) + _normalize_score(capped) + 0.5 * _normalize_score(hard)
    if role == "path_aware":
        broad = np.sum(improvement, axis=0)
        upper_quartile = np.quantile(improvement, 0.75, axis=0)
        return broad + len(current_loss) * upper_quartile
    if role == "shape_cluster":
        return np.sum(improvement, axis=0)
    return np.sum(improvement, axis=0)


def _normalize_score(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    span = float(np.max(array) - np.min(array)) if len(array) else 0.0
    if span <= 1e-15:
        return np.zeros_like(array, dtype=np.float64)
    return (array - np.min(array)) / span


def _shape_cluster_priority(matrix: np.ndarray, current_loss: np.ndarray) -> np.ndarray:
    centered = matrix - np.mean(matrix, axis=1, keepdims=True)
    energy = np.sqrt(np.mean(centered * centered, axis=1))
    return current_loss + 0.05 * energy


def _preprocess_candidates(candidates: np.ndarray, residual_matrix: np.ndarray, spec: ComponentRowSpec) -> np.ndarray:
    matrix = np.asarray(candidates, dtype=np.float32).copy()
    if spec.atom_preprocessing_policy == "RawAtoms":
        return matrix
    if spec.atom_preprocessing_policy == "CenteredEnergyNormalizedAtoms":
        matrix = matrix - np.mean(matrix, axis=1, keepdims=True)
    if spec.atom_preprocessing_policy in {"EnergyNormalizedAtoms", "CenteredEnergyNormalizedAtoms"}:
        target_rms = float(np.median(np.sqrt(np.mean(residual_matrix * residual_matrix, axis=1))))
        rms = np.sqrt(np.mean(matrix * matrix, axis=1, keepdims=True))
        matrix = np.divide(matrix, rms, out=np.zeros_like(matrix), where=rms > 1e-8) * target_rms
    return matrix.astype(np.float32)


def _filter_duplicate_pool(matrix: np.ndarray, pool: np.ndarray, atoms: list[np.ndarray], spec: ComponentRowSpec) -> np.ndarray:
    if spec.duplicate_suppression_policy == "DuplicateSuppressionOff" or len(atoms) <= 1:
        return pool
    kept = [int(index) for index in pool if not _is_phase_scale_duplicate(matrix[int(index)], atoms[1:])]
    return np.asarray(kept, dtype=np.int32)


def _is_phase_scale_duplicate(candidate: np.ndarray, atoms: list[np.ndarray]) -> bool:
    cand = np.asarray(candidate, dtype=np.float32)
    cand = cand - float(np.mean(cand))
    cand_norm = float(np.sqrt(np.sum(cand * cand)))
    if cand_norm <= 1e-8:
        return False
    cand = cand / cand_norm
    for atom in atoms:
        other = np.asarray(atom, dtype=np.float32)
        other = other - float(np.mean(other))
        other_norm = float(np.sqrt(np.sum(other * other)))
        if other_norm <= 1e-8:
            continue
        other = other / other_norm
        best = max(abs(float(np.dot(cand, np.roll(other, shift)))) for shift in range(len(cand)))
        if best >= 0.995:
            return True
    return False


def _tune_atoms_after_use(residual: np.ndarray, atoms: np.ndarray, *, spec: ComponentRowSpec, chunk_size: int) -> np.ndarray:
    matrix = np.asarray(residual, dtype=np.float32)
    tuned = np.asarray(atoms, dtype=np.float32).copy()
    phase_count = _phase_candidate_count(spec, matrix.shape[1])
    choice = best_alignment(
        matrix,
        tuned,
        phase_policy="fft_lattice",
        gain_policy=spec.residual_gain_policy,
        backend="numpy",
        chunk_size=chunk_size,
        phase_candidate_count=phase_count,
    )
    for atom_index in range(1, len(tuned)):
        members = np.flatnonzero(choice.indices == atom_index)
        if len(members) >= 2:
            tuned[atom_index] = np.mean(matrix[members], axis=0).astype(np.float32)
    tuned[0] = 0.0
    return tuned.astype(np.float32)


def _encode_decode(
    spec: ComponentRowSpec,
    targets: np.ndarray,
    assets: ReconstructionAssets,
    *,
    backend: BackendPreference,
    chunk_size: int,
    progress: Callable[[str], None] | None,
    progress_label: str,
) -> tuple[ComponentEncoding, np.ndarray, np.ndarray, float]:
    started = time.perf_counter()
    if spec.path_policy == "greedy":
        encoding, reconstructed = _encode_greedy(spec, targets, assets, backend=backend, chunk_size=chunk_size, progress=progress, progress_label=progress_label)
    else:
        encoding, reconstructed = _encode_beam(spec, targets, assets, backend=backend, chunk_size=chunk_size, progress=progress, progress_label=progress_label)
    raw = reconstructed.astype(np.float32)
    return encoding, np.clip(raw, 0.0, 1.0).astype(np.float32), raw, time.perf_counter() - started


def _encode_greedy(
    spec: ComponentRowSpec,
    targets: np.ndarray,
    assets: ReconstructionAssets,
    *,
    backend: BackendPreference,
    chunk_size: int,
    progress: Callable[[str], None] | None,
    progress_label: str,
) -> tuple[ComponentEncoding, np.ndarray]:
    phase_count = _phase_candidate_count(spec, targets.shape[1])
    base = best_alignment(targets, assets.base_dictionary, phase_policy="fft_lattice", gain_policy="fixed", backend=backend, chunk_size=chunk_size, phase_candidate_count=phase_count)
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
            phase_policy="fft_lattice",
            gain_policy=spec.residual_gain_policy,
            backend=backend,
            chunk_size=chunk_size,
            phase_candidate_count=phase_count,
        )
        next_prefix = _apply_layer_state_policy(prefix, choice.values, spec)
        choice_indices = choice.indices.copy()
        choice_phases = choice.phases.copy()
        choice_gains = choice.gains.copy()
        previous_loss = np.mean((_decoder_view(prefix, spec) - targets) ** 2, axis=1)
        next_loss = np.mean((_decoder_view(next_prefix, spec) - targets) ** 2, axis=1)
        force_no_op = _force_no_op_mask(spec, residual_layer, targets, prefix, previous_loss, next_loss)
        if np.any(force_no_op):
            next_prefix[force_no_op] = prefix[force_no_op]
            choice_indices[force_no_op] = NO_OP_ATOM_INDEX
            choice_phases[force_no_op] = 0.0
            choice_gains[force_no_op] = 0.0
        indices.append(choice_indices)
        phases.append(choice_phases)
        gains.append(choice_gains)
        prefix = next_prefix
    return ComponentEncoding(base.indices, base.phases, base.gains, indices, phases, gains), prefix


def _encode_beam(
    spec: ComponentRowSpec,
    targets: np.ndarray,
    assets: ReconstructionAssets,
    *,
    backend: BackendPreference,
    chunk_size: int,
    progress: Callable[[str], None] | None,
    progress_label: str,
) -> tuple[ComponentEncoding, np.ndarray]:
    rows = len(targets)
    out = _empty_encoding(rows)
    reconstructed = np.empty_like(targets, dtype=np.float32)
    for start in range(0, rows, max(1, int(chunk_size))):
        stop = min(start + max(1, int(chunk_size)), rows)
        local_encoding, local_reconstructed = _encode_beam_batch(spec, targets[start:stop], assets, backend=backend, chunk_size=max(1, int(chunk_size)))
        _scatter_encoding(out, local_encoding, np.arange(start, stop))
        reconstructed[start:stop] = local_reconstructed
        if start == 0 or stop == rows:
            _log(progress, f"{progress_label} encoding: beam batch {stop}/{rows}")
    return out, reconstructed


def _encode_beam_batch(
    spec: ComponentRowSpec,
    targets: np.ndarray,
    assets: ReconstructionAssets,
    *,
    backend: BackendPreference,
    chunk_size: int,
) -> tuple[ComponentEncoding, np.ndarray]:
    phase_count = _phase_candidate_count(spec, targets.shape[1])
    base_matrix = alignment_matrix(targets, assets.base_dictionary, phase_policy="fft_lattice", gain_policy="fixed", backend=backend, chunk_size=chunk_size, phase_candidate_count=phase_count)
    beam = min(max(1, spec.beam_width), base_matrix.losses.shape[1])
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
            phase_policy="fft_lattice",
            gain_policy=spec.residual_gain_policy,
            backend=backend,
            chunk_size=chunk_size,
            phase_candidate_count=phase_count,
        )
        shifted = circular_shift(
            dictionary[np.broadcast_to(np.arange(len(dictionary)), (b * current_beam, len(dictionary))).reshape(-1)],
            matrix.phases.reshape(-1),
        ).reshape(b, current_beam, len(dictionary), targets.shape[1])
        additions = shifted * matrix.gains.reshape(b, current_beam, len(dictionary), 1)
        candidate_state = _apply_layer_state_policy(prefix[:, :, None, :], additions, spec)
        candidate_recon = _decoder_view(candidate_state, spec)
        mse = np.mean((targets[:, None, None, :] - candidate_recon) ** 2, axis=3)
        previous = np.mean((targets[:, None, :] - _decoder_view(prefix, spec)) ** 2, axis=2)
        mse = _apply_overshoot_penalty(mse, candidate_state, spec)
        if spec.no_damage_policy in {"LateLayerNoDamage", "LateLayerNoDamageAndPerfectLocking"} and residual_layer >= D // 2:
            no_damage = mse > (previous[:, :, None] - NO_DAMAGE_MIN_IMPROVEMENT)
            no_damage[:, :, NO_OP_ATOM_INDEX] = False
            mse[no_damage] = np.inf
        if spec.no_damage_policy in {"PerfectLocking", "LateLayerNoDamageAndPerfectLocking"}:
            locked = np.max(np.abs(_decoder_view(prefix, spec) - targets[:, None, :]), axis=2) <= PERFECT_MAX_ABS_EPS
            locked_mask = np.broadcast_to(locked[:, :, None], mse.shape).copy()
            locked_mask[:, :, NO_OP_ATOM_INDEX] = False
            mse[locked_mask] = np.inf
        mse[:, :, 0] = np.minimum(mse[:, :, 0], previous)
        candidate_state[:, :, 0, :] = prefix
        matrix.phases[:, 0] = 0.0
        matrix.gains[:, 0] = 0.0
        flat = mse.reshape(b, -1)
        next_beam = min(spec.beam_width, flat.shape[1])
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
    encoding = ComponentEncoding(
        base_index=base_paths[:, 0],
        base_phase=base_phase_paths[:, 0] % 1.0,
        base_gain=base_gain_paths[:, 0],
        residual_layer_indices=[index_paths[:, 0, layer] for layer in range(D)],
        residual_layer_phases=[phase_paths[:, 0, layer] % 1.0 for layer in range(D)],
        residual_layer_gains=[gain_paths[:, 0, layer] for layer in range(D)],
    )
    return encoding, prefix[:, 0, :].astype(np.float32)


def _apply_layer_state_policy(prefix: np.ndarray, addition: np.ndarray, spec: ComponentRowSpec) -> np.ndarray:
    if spec.layer_normalization_policy == "BoundedResidualStep":
        state = _bounded_residual_step(prefix, addition, lower=0.0, upper=1.0)
    else:
        state = np.asarray(prefix, dtype=np.float32) + np.asarray(addition, dtype=np.float32)
    policy = spec.layer_normalization_policy
    if policy in {"FinalClipOnly", "OvershootPenaltyNoClip", "BoundedResidualStep"}:
        return state.astype(np.float32)
    if policy == "LayerClip0To1":
        return np.clip(state, 0.0, 1.0).astype(np.float32)
    if policy == "LayerClipNeg0p1To1p1":
        return np.clip(state, -0.1, 1.1).astype(np.float32)
    if policy == "LayerClipNeg1To1":
        return np.clip(state, -1.0, 1.0).astype(np.float32)
    if policy == "LayerSoftClip0To1":
        return _soft_clip(state, 0.0, 1.0).astype(np.float32)
    if policy == "LayerSoftClipNeg0p1To1p1":
        return _soft_clip(state, -0.1, 1.1).astype(np.float32)
    if policy == "LayerCenterPreserveClip":
        clipped = np.clip(state, 0.0, 1.0)
        delta = np.mean(state, axis=-1, keepdims=True) - np.mean(clipped, axis=-1, keepdims=True)
        return np.clip(clipped + delta, 0.0, 1.0).astype(np.float32)
    raise ValueError(f"unsupported layer_normalization_policy: {policy}")


def _decoder_view(state: np.ndarray, spec: ComponentRowSpec) -> np.ndarray:
    if spec.layer_normalization_policy == "OvershootPenaltyNoClip":
        return np.asarray(state, dtype=np.float32)
    return np.clip(state, 0.0, 1.0).astype(np.float32)


def _apply_overshoot_penalty(mse: np.ndarray, state: np.ndarray, spec: ComponentRowSpec) -> np.ndarray:
    if spec.layer_normalization_policy != "OvershootPenaltyNoClip":
        return mse
    below = np.maximum(0.0, -state)
    above = np.maximum(0.0, state - 1.0)
    penalty = np.mean(below * below + above * above, axis=-1)
    return mse + 0.25 * penalty


def _soft_clip(state: np.ndarray, lower: float, upper: float) -> np.ndarray:
    width = upper - lower
    mid = lower + width / 2.0
    return lower + width / (1.0 + np.exp(-8.0 * (np.asarray(state, dtype=np.float32) - mid) / width))


def _bounded_residual_step(prefix: np.ndarray, addition: np.ndarray, *, lower: float, upper: float) -> np.ndarray:
    prefix_array = np.asarray(prefix, dtype=np.float32)
    addition_array = np.asarray(addition, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        positive_limit = np.divide(upper - prefix_array, addition_array, out=np.ones_like(addition_array), where=addition_array > 0.0)
        negative_limit = np.divide(lower - prefix_array, addition_array, out=np.ones_like(addition_array), where=addition_array < 0.0)
    limit = np.where(addition_array > 0.0, positive_limit, np.where(addition_array < 0.0, negative_limit, 1.0))
    alpha = np.min(np.clip(limit, 0.0, 1.0), axis=-1, keepdims=True)
    return (prefix_array + alpha * addition_array).astype(np.float32)


def _force_no_op_mask(
    spec: ComponentRowSpec,
    residual_layer: int,
    targets: np.ndarray,
    prefix: np.ndarray,
    previous_loss: np.ndarray,
    next_loss: np.ndarray,
) -> np.ndarray:
    mask = np.zeros(len(targets), dtype=bool)
    if spec.no_damage_policy in {"LateLayerNoDamage", "LateLayerNoDamageAndPerfectLocking"} and residual_layer >= D // 2:
        mask |= next_loss > (previous_loss - NO_DAMAGE_MIN_IMPROVEMENT)
    if spec.no_damage_policy in {"PerfectLocking", "LateLayerNoDamageAndPerfectLocking"}:
        max_abs = np.max(np.abs(_decoder_view(prefix, spec) - targets), axis=1)
        mask |= max_abs <= PERFECT_MAX_ABS_EPS
    return mask


def _manifest(
    spec: ComponentRowSpec,
    assets: ReconstructionAssets,
    dataset: Era2CurveDataset,
    budget: dict[str, Any],
    flags: TopologyFlags,
    construction_time: float,
    encoding_time: float,
) -> dict[str, Any]:
    return {
        "experiment_id": "experiment_12",
        "row_id": spec.row_id,
        "components": list(spec.components),
        "description": spec.description,
        "screening_variable": spec.screening_variable,
        "screening_value": spec.screening_value,
        "scalar_schema": spec.scalar_schema,
        "oracle_construction_id": spec.construction_policy,
        "runtime_interface_id": "flat_categorical_per_residual_layer",
        "decoder_policy_id": spec.layer_normalization_policy,
        "base_dictionary_size": BASE_DICTIONARY_SIZE,
        "W": W,
        "D": D,
        "reserved_atom": "NoOpAtom",
        "reserved_atom_index": NO_OP_ATOM_INDEX,
        "active_atoms_per_layer": NO_OP_ACTIVE_ATOMS,
        "scalar_families": spec.scalar_families,
        "scalar_outputs": spec.scalar_outputs,
        "categorical_outputs": budget["categorical_outputs"],
        "continuous_outputs": budget["continuous_outputs"],
        "residual_atom_selection_outputs": budget["residual_atom_selection_outputs"],
        "head_outputs_formula": budget["head_outputs_formula"],
        "head_outputs_actual": budget["head_outputs_actual"],
        "lfo_control_point_count": dataset.resolution,
        "subdivision_count": dataset.resolution - 1,
        "dictionary_scope": assets.dictionary_scope,
        "codebook_storage_count": assets.codebook_storage_count,
        "oracle_construction_time": construction_time,
        "oracle_encoding_time": encoding_time,
        "phase_enabled": spec.phase_enabled,
        "phase_alignment_policy": "fft_lattice" if spec.phase_enabled else "disabled",
        "oracle_phase_candidate_count": dataset.resolution if spec.phase_enabled else 1,
        "residual_gain_enabled": spec.residual_gain_enabled,
        "residual_gain_policy": spec.residual_gain_policy,
        "residual_gain_model_facing": spec.residual_gain_enabled,
        "path_policy": spec.path_policy,
        "path_search_policy": spec.path_search_policy,
        "beam_width": spec.beam_width,
        "construction_policy": spec.construction_policy,
        "utility_candidate_budget": spec.utility_candidate_budget,
        "max_utility_candidates": spec.max_utility_candidates,
        "layer_normalization_policy": spec.layer_normalization_policy,
        "no_damage_policy": spec.no_damage_policy,
        "atom_preprocessing_policy": spec.atom_preprocessing_policy,
        "duplicate_suppression_policy": spec.duplicate_suppression_policy,
        "fixed_x_grid_note": "LFO x-grid geometry is decoder-owned and adds zero model prediction head outputs.",
        **flags.as_dict(),
        **dataset.manifest_fields(),
    }


def _usage_summary(encoding: ComponentEncoding, *, widths: list[int]) -> dict[str, Any]:
    usage = flat_atom_usage(
        encoding.index_arrays(),
        residual_layer_count=D,
        widths_by_residual_layer=widths,
    )
    dead = [value for key, value in usage.items() if key.endswith("_dead_atom_rate")]
    dominant = [value for key, value in usage.items() if key.endswith("_dominant_atom_share")]
    entropy = [value for key, value in usage.items() if key.endswith("_atom_usage_entropy")]
    usage["residual_layer_dead_atom_rate_median"] = float(np.median(dead)) if dead else 0.0
    usage["residual_layer_dominant_atom_share_median"] = float(np.median(dominant)) if dominant else 0.0
    usage["residual_layer_usage_entropy_median"] = float(np.median(entropy)) if entropy else 0.0
    no_op_rates = []
    for residual_layer, values in enumerate(encoding.residual_layer_indices, start=1):
        rate = float(np.mean(np.asarray(values, dtype=np.int32) == NO_OP_ATOM_INDEX)) if len(values) else 0.0
        usage[f"residual_layer_{residual_layer}_no_op_usage_rate"] = rate
        no_op_rates.append(rate)
    usage["residual_layer_no_op_usage_rate_median"] = float(np.median(no_op_rates)) if no_op_rates else 0.0
    return usage


def _overshoot_summary(raw_reconstructed: np.ndarray) -> dict[str, float]:
    raw = np.asarray(raw_reconstructed, dtype=np.float32)
    outside = (raw < 0.0) | (raw > 1.0)
    amount = np.maximum(0.0, -raw) + np.maximum(0.0, raw - 1.0)
    return {
        "overshoot_rate_before_final_clip": float(np.mean(outside)),
        "overshoot_abs_p95_before_final_clip": float(np.quantile(amount, 0.95)),
    }


def _asset_diagnostics(assets: ReconstructionAssets) -> dict[str, float]:
    duplicate_rates = []
    for layer in assets.residual_layer_dictionaries:
        active = [layer[index] for index in range(1, len(layer))]
        pairs = 0
        duplicates = 0
        for left in range(len(active)):
            for right in range(left + 1, len(active)):
                pairs += 1
                if _is_phase_scale_duplicate(active[left], [active[right]]):
                    duplicates += 1
        duplicate_rates.append(float(duplicates / pairs) if pairs else 0.0)
    return {"duplicate_atom_rate": float(np.median(duplicate_rates)) if duplicate_rates else 0.0}


def _scalar_summary(encoding: ComponentEncoding, spec: ComponentRowSpec) -> dict[str, Any]:
    residual_phases = np.concatenate(encoding.residual_layer_phases) if encoding.residual_layer_phases else np.asarray([], dtype=np.float32)
    residual_gains = np.concatenate(encoding.residual_layer_gains) if encoding.residual_layer_gains else np.asarray([], dtype=np.float32)
    return {
        "base_phase_abs_median": float(np.median(np.abs(encoding.base_phase))) if spec.phase_enabled else 0.0,
        "residual_phase_abs_median": float(np.median(np.abs(residual_phases))) if spec.phase_enabled and len(residual_phases) else 0.0,
        "residual_gain_median": float(np.median(residual_gains)) if spec.residual_gain_enabled and len(residual_gains) else 0.0,
        "residual_gain_abs_p95": float(np.quantile(np.abs(residual_gains), 0.95)) if spec.residual_gain_enabled and len(residual_gains) else 0.0,
        "residual_gain_nonzero_rate": float(np.mean(np.abs(residual_gains) > 1e-8)) if spec.residual_gain_enabled and len(residual_gains) else 0.0,
    }


def _component_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baselines = {
        str(row.get("scalar_schema", "")): row
        for row in rows
        if row.get("screening_variable") == "construction_policy" and row.get("screening_value") == "BestOverallRepair"
    }
    result = []
    for row in rows:
        baseline = baselines.get(str(row.get("scalar_schema", "")))
        baseline_p95 = _float(baseline.get("validation_p95_rmse")) if baseline else None
        baseline_median = _float(baseline.get("validation_median_rmse")) if baseline else None
        p95 = _float(row.get("validation_p95_rmse"))
        median = _float(row.get("validation_median_rmse"))
        result.append(
            {
                "row_id": row.get("row_id", ""),
                "screening_variable": row.get("screening_variable", ""),
                "screening_value": row.get("screening_value", ""),
                "scalar_schema": row.get("scalar_schema", ""),
                "components": row.get("components", ""),
                "head_outputs_actual": row.get("head_outputs_actual", ""),
                "validation_p95_rmse": row.get("validation_p95_rmse", ""),
                "validation_median_rmse": row.get("validation_median_rmse", ""),
                "p95_delta_vs_scalar_default": "" if p95 is None or baseline_p95 is None else p95 - baseline_p95,
                "median_delta_vs_scalar_default": "" if median is None or baseline_median is None else median - baseline_median,
            }
        )
    return result


def _budget_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_id": row.get("row_id", ""),
        "screening_variable": row.get("screening_variable", ""),
        "screening_value": row.get("screening_value", ""),
        "scalar_schema": row.get("scalar_schema", ""),
        "components": row.get("components", ""),
        "W": row.get("W", ""),
        "D": row.get("D", ""),
        "reserved_atom": row.get("reserved_atom", ""),
        "active_atoms_per_layer": row.get("active_atoms_per_layer", ""),
        "scalar_families": row.get("scalar_families", ""),
        "scalar_outputs": row.get("scalar_outputs", ""),
        "categorical_outputs": row.get("categorical_outputs", ""),
        "head_outputs_formula": row.get("head_outputs_formula", ""),
        "head_outputs_actual": row.get("head_outputs_actual", ""),
    }


def _scalar_row(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "row_id",
        "screening_variable",
        "screening_value",
        "scalar_schema",
        "phase_enabled",
        "residual_gain_enabled",
        "base_phase_abs_median",
        "residual_phase_abs_median",
        "residual_gain_median",
        "residual_gain_abs_p95",
        "residual_gain_nonzero_rate",
    ]
    return {key: row.get(key, "") for key in keys}


def _usage_row(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "row_id",
        "screening_variable",
        "screening_value",
        "scalar_schema",
        "residual_layer_dead_atom_rate_median",
        "residual_layer_dominant_atom_share_median",
        "residual_layer_usage_entropy_median",
        "residual_layer_no_op_usage_rate_median",
        "duplicate_atom_rate",
        "validation_overshoot_rate_before_final_clip",
    ]
    return {key: row.get(key, "") for key in keys}


def _screening_row(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "row_id",
        "screening_variable",
        "screening_value",
        "scalar_schema",
        "path_search_policy",
        "construction_policy",
        "utility_candidate_budget",
        "layer_normalization_policy",
        "no_damage_policy",
        "atom_preprocessing_policy",
        "duplicate_suppression_policy",
        "head_outputs_actual",
        "validation_median_rmse",
        "validation_strict_perfect_lfo_rate",
        "validation_p95_rmse",
        "validation_node_max_error_p95",
        "oracle_construction_time",
        "validation_encoding_time",
        "residual_layer_no_op_usage_rate_median",
        "residual_layer_usage_entropy_median",
        "duplicate_atom_rate",
        "validation_overshoot_rate_before_final_clip",
    ]
    return {key: row.get(key, "") for key in keys}


def _write_plots(image_dir: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    image_dir.mkdir(parents=True, exist_ok=True)
    ordered = [row for row in rows if _float(row.get("validation_p95_rmse")) is not None]
    if not ordered:
        return
    _bar_plot(image_dir / "experiment12_validation_p95_by_row.png", ordered, "validation_p95_rmse", "validation P95 RMSE", "Experiment 12 validation P95 by row", plt)
    _bar_plot(image_dir / "experiment12_validation_median_by_row.png", ordered, "validation_median_rmse", "validation median RMSE", "Experiment 12 validation median by row", plt)
    _delta_plot(image_dir / "experiment12_delta_vs_indices_only.png", ordered, "x12_c0_indices_only", "validation_p95_rmse", "P95 RMSE delta vs indices-only baseline", "Experiment 12 P95 delta vs indices-only", plt)
    _scatter_plot(image_dir / "experiment12_p95_vs_head_outputs.png", ordered, "head_outputs_actual", "validation_p95_rmse", "model prediction head outputs", "validation P95 RMSE", "Experiment 12 P95 vs model prediction head budget", plt)
    _bar_plot(image_dir / "experiment12_scalar_usage.png", ordered, "residual_gain_abs_p95", "residual gain abs P95", "Experiment 12 residual gain usage", plt)
    _bar_plot(image_dir / "experiment12_atom_usage.png", ordered, "residual_layer_dead_atom_rate_median", "median residual-layer dead atom rate", "Experiment 12 atom usage", plt)
    _bar_plot(image_dir / "experiment12_runtime_by_row.png", ordered, "row_elapsed_seconds", "row elapsed seconds", "Experiment 12 runtime by row", plt)
    for variable in _screening_variables():
        group = [row for row in rows if row.get("screening_variable") == variable]
        if group:
            _variable_panel_plot(
                image_dir / f"experiment12_{variable}_co_primary.png",
                group,
                variable,
                [
                    ("validation_median_rmse", "median RMSE", "lower better"),
                    ("validation_strict_perfect_lfo_rate", "strict perfect rate", "higher better"),
                    ("validation_p95_rmse", "P95 RMSE", "lower better"),
                    ("validation_node_max_error_p95", "node max P95", "lower better"),
                ],
                f"Experiment 12 {variable} co-primary metrics",
                plt,
            )
            _variable_panel_plot(
                image_dir / f"experiment12_{variable}_diagnostics.png",
                group,
                variable,
                [
                    ("oracle_construction_time", "construction seconds", "lower faster"),
                    ("validation_encoding_time", "validation encoding seconds", "lower faster"),
                    ("residual_layer_no_op_usage_rate_median", "median no-op usage", "diagnostic"),
                    ("validation_overshoot_rate_before_final_clip", "overshoot rate before final clip", "lower cleaner"),
                ],
                f"Experiment 12 {variable} runtime and diagnostics",
                plt,
            )


def _write_report(report_path: Path, rows: list[dict[str, Any]]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_report_text(rows), encoding="utf-8")


def _report_text(rows: list[dict[str, Any]]) -> str:
    default_indices = _report_row(rows, "construction_policy", "BestOverallRepair", "IndicesOnly")
    default_phase_gain = _report_row(rows, "construction_policy", "BestOverallRepair", "PhaseAndResidualGain")
    common_case = _report_row(rows, "construction_policy", "CommonCaseRepair", "PhaseAndResidualGain")
    finish_rescue = _report_row(rows, "construction_policy", "FinishRepairRescue", "PhaseAndResidualGain")
    layer_clip = _report_row(rows, "layer_normalization_policy", "LayerClip0To1", "PhaseAndResidualGain")
    center_clip = _report_row(rows, "layer_normalization_policy", "LayerCenterPreserveClip", "PhaseAndResidualGain")
    beam4 = _report_row(rows, "path_search_policy", "Beam4Path", "PhaseAndResidualGain")
    beam8 = _report_row(rows, "path_search_policy", "Beam8Path", "PhaseAndResidualGain")
    lines = [
        "# Experiment 12: Fixed-W8D16 Screening Grid",
        "",
        "## Main Findings",
        "",
        f"The main result is that `PhaseAndResidualGain` is the dominant quality unlock for fixed `W=8,D=16`. Under the default construction policy, median RMSE moves from `{_metric(default_indices, 'validation_median_rmse')}` to `{_metric(default_phase_gain, 'validation_median_rmse')}`, validation P95 moves from `{_metric(default_indices, 'validation_p95_rmse')}` to `{_metric(default_phase_gain, 'validation_p95_rmse')}`, and node-max P95 moves from `{_metric(default_indices, 'validation_node_max_error_p95')}` to `{_metric(default_phase_gain, 'validation_node_max_error_p95')}`. That gain costs model prediction head budget: `IndicesOnly` uses `160` heads, while `PhaseAndResidualGain` uses `193`.",
        "",
        f"Construction policy is the most important process-like variable in the run. `CommonCaseRepair` is the median and strict-perfect outlier: with `PhaseAndResidualGain`, it reaches median RMSE `{_metric(common_case, 'validation_median_rmse')}` and strict perfect-LFO rate `{_metric(common_case, 'validation_strict_perfect_lfo_rate')}`, but its P95 is `{_metric(common_case, 'validation_p95_rmse')}`. `FinishRepairRescue` is the cleaner balanced construction candidate: median `{_metric(finish_rescue, 'validation_median_rmse')}`, strict perfect rate `{_metric(finish_rescue, 'validation_strict_perfect_lfo_rate')}`, P95 `{_metric(finish_rescue, 'validation_p95_rmse')}`, and node-max P95 `{_metric(finish_rescue, 'validation_node_max_error_p95')}`.",
        "",
        f"End-of-layer normalization is a real free decoder-policy lever. `LayerClip0To1` has the best validation P95 in the run at `{_metric(layer_clip, 'validation_p95_rmse')}`, while `LayerCenterPreserveClip` is essentially tied on P95 at `{_metric(center_clip, 'validation_p95_rmse')}` and has the best node-max P95 among the layer-normalization rows at `{_metric(center_clip, 'validation_node_max_error_p95')}`. The soft-clip variants, bounded residual step, and overshoot-penalty/no-clip variant are weaker in this screen.",
        "",
        "`no_damage_policy` and duplicate suppression are mostly flat. They do not move quality enough to justify treating them as primary Experiment 13 axes unless the grid has spare room. Duplicate suppression is especially weak here: quality is identical to the default row while construction time is higher.",
        "",
        f"The run contains `{len(rows)}` rows. Every row keeps `W=8`, `D=16`, `control_point_count=97`, flat-categorical per-residual-layer addressing, and one required `NoOpAtom` per residual layer. This is still a screening run, not an automatic winner selection: median RMSE, strict perfect-LFO rate, P95 RMSE, and node-max P95 disagree in meaningful ways.",
        "",
        "## Why This Happens",
        "",
        "The scalar result is expected. Residual atoms need phase and scale invariance: a useful residual shape may be shifted in cycle phase or appear at a different amplitude. `IndicesOnly` can only choose an atom slot, so it often needs later layers to compensate for a phase or amplitude mismatch. `PhaseAndResidualGain` gives the decoder the missing alignment degrees of freedom directly.",
        "",
        "`NoOpAtom` changes how atom usage should be read. High no-op usage can mean a layer has stopped doing useful repair, but it is also the safety valve that prevents a residual layer from damaging an already-good reconstruction. The no-op atom is therefore not dead capacity in the usual sense; it is a required stopping action inside the residual ladder.",
        "",
        "The construction-policy split is a finish-vs-repair tradeoff. `CommonCaseRepair` spends atoms on residuals that many LFOs share, so it strongly improves the median and creates many exact reconstructions. It leaves some hard cases under-repaired, which is why its P95 stays high. `FinishRepairRescue` mixes finishing behavior with broader repair and later hard-case rescue, so it gives up some perfect-rate upside for a much better tail.",
        "",
        "Layer clipping helps because residual additions can overshoot the legal LFO y range before the final decoder clip. Hard clipping after each layer can stop overshoot from propagating through later residual choices. This is a decoder/free policy: it changes deterministic reconstruction behavior and adds zero model prediction head outputs. It should not be confused with oracle/offline construction work or with deployed runtime inputs.",
        "",
        "## Experiment 13 Candidate Read",
        "",
        "This section is manual selection guidance, not an automatic ranking. The right Experiment 13 grid should preserve candidates that win different co-primary metrics.",
        "",
        "- `path_search_policy`: keep both `Beam4Path` and `Beam8Path` unless grid size must shrink. `Beam8Path` is modestly better on P95 (`" + _metric(beam8, "validation_p95_rmse") + "` vs `" + _metric(beam4, "validation_p95_rmse") + "`) but costs more encoding time.",
        "- `construction_policy`: shortlist `FinishRepairRescue`, `CommonCaseRepair`, and `FamilyBalancedRepair` or `ShapeClusterRepair`. `FinishRepairRescue` is the balanced choice; `CommonCaseRepair` is the median/perfect-rate stress test.",
        "- `utility_candidate_budget`: shortlist `CandidateBudget48`, `CandidateBudget24`, and `CandidateBudget12`. `CandidateBudget8` is cheap, but under `PhaseAndResidualGain` it is less compelling on tail quality.",
        "- `layer_normalization_policy`: shortlist `LayerClip0To1`, `LayerCenterPreserveClip`, and `LayerClipNeg0p1To1p1`. Treat soft clips, `BoundedResidualStep`, and `OvershootPenaltyNoClip` as weak unless a later run gives them a different role.",
        "- `no_damage_policy`: if keeping three values, use `NoDamageOff`, `LateLayerNoDamage`, and `LateLayerNoDamageAndPerfectLocking`. The variable looks low-impact in this run.",
        "- `atom_preprocessing_policy`: shortlist `EnergyNormalizedAtoms`, `RawAtoms`, and `CenteredEnergyNormalizedAtoms`. Keep the warning that centered normalization hurts `IndicesOnly` badly.",
        "- `duplicate_suppression_policy`: keep both only if Experiment 13 budget allows. Current quality metrics are identical, while duplicate suppression costs more construction time.",
        "",
    ]
    lines.extend(_independent_variable_chapters())
    lines.extend(
        [
        "## Global Plot Notes",
        "",
        "Lower is better for validation P95, validation median, max-point error, overshoot, and runtime. Higher is better for strict perfect-LFO rate.",
        "",
        "### Validation P95 By Row",
        "",
        "![Validation P95](./images/experiment_12/experiment12_validation_p95_by_row.png)",
        "",
        "The x-axis is the screened row; the y-axis is validation P95 RMSE, where lower is better. The visible split is that most `PhaseAndResidualGain` rows sit far below their `IndicesOnly` partners. The best rows are mostly layer-normalization and balanced-construction variants, which supports carrying clipping and construction-policy candidates into Experiment 13.",
        "",
        "### P95 Delta Vs Indices-Only Baseline",
        "",
        "![P95 delta vs indices-only](./images/experiment_12/experiment12_delta_vs_indices_only.png)",
        "",
        "The x-axis is the screened row; the y-axis is P95 RMSE minus the default `IndicesOnly` baseline, so negative is better. Almost every row improves on the baseline, which says the fixed `W8D16` contract has enough room for free process improvements. The few positive bars are the warning cases: finish-only construction can make the tail worse even when it is trying to complete more LFOs.",
        "",
        "### Validation Median By Row",
        "",
        "![Validation median](./images/experiment_12/experiment12_validation_median_by_row.png)",
        "",
        "The x-axis is the screened row; the y-axis is validation median RMSE, where lower is better. The plot has a small cluster near zero plus a broader band of ordinary rows. `CommonCaseRepair` creates the clearest near-zero median bar, while finish-only and soft-clip rows remain visibly high. This is why median remains co-primary instead of being folded into P95.",
        "",
        "### P95 Vs Model Prediction Head Budget",
        "",
        "![P95 vs head outputs](./images/experiment_12/experiment12_p95_vs_head_outputs.png)",
        "",
        "The x-axis is deployed model prediction head budget; the y-axis is validation P95 RMSE. The plot should be read as two vertical clusters, not as individual labels: `160`-head `IndicesOnly` rows and `193`-head `PhaseAndResidualGain` rows. Most of the best tail rows are in the `193`-head cluster, but the vertical spread inside each cluster proves that process and decoder policies matter even when head budget is fixed.",
        "",
        "### Residual Gain Usage",
        "",
        "![Scalar usage](./images/experiment_12/experiment12_scalar_usage.png)",
        "",
        "The x-axis is the screened row; the y-axis is residual-gain absolute P95. Higher is not automatically better here: it means the optimized residual scalar is being used more strongly. Most `PhaseAndResidualGain` rows form a moderate band, while a few construction/normalization rows spike close to the gain bounds. Those spikes are diagnostics for aggressive correction or overshoot compensation, not quality wins by themselves.",
        "",
        "### Atom Usage",
        "",
        "![Atom usage](./images/experiment_12/experiment12_atom_usage.png)",
        "",
        "The x-axis is the screened row; the y-axis is median residual-layer dead-atom rate. Lower means more dictionary slots are used, but this is diagnostic rather than a direct objective. The tallest spikes line up with policies that collapse much of the residual ladder into no-op or unused active atoms, especially finish-heavy and soft/bounded normalization variants. That pattern helps explain why some policies look safe but do not repair the tail well.",
        "",
        "### Runtime",
        "",
        "![Runtime](./images/experiment_12/experiment12_runtime_by_row.png)",
        "",
        "The x-axis is the screened row; the y-axis is row elapsed seconds, where lower is faster. Most rows sit in a broad middle band, but a few construction-heavy rows stand out as clear runtime outliers. This is oracle construction and encoding runtime on the current implementation, not deployed model runtime. It matters for experiment velocity and for sizing Experiment 13, but not for the model prediction head budget.",
        "",
        "## Grouped Evidence Tables",
        "",
        "Co-primary metrics: `validation_median_rmse`, `validation_strict_perfect_lfo_rate`, `validation_p95_rmse`, and `validation_node_max_error_p95`. The tables are grouped by screened variable and show both scalar contexts side by side.",
        "",
        ]
    )
    for variable in [
        "path_search_policy",
        "construction_policy",
        "utility_candidate_budget",
        "layer_normalization_policy",
        "no_damage_policy",
        "atom_preprocessing_policy",
        "duplicate_suppression_policy",
    ]:
        group = [row for row in rows if row.get("screening_variable") == variable]
        if not group:
            continue
        lines.extend(_screening_table(variable, group))
    lines.extend(
        [
            "",
            "## Fixed Contract",
            "",
            "| Variable | Fixed Value |",
            "|---|---|",
            "| `base_dictionary_size` | `32` |",
            "| `residual_width` | `8` |",
            "| `reserved_atom` | `NoOpAtom` |",
            "| `active_atoms_per_layer` | `7` |",
            "| `residual_depth` | `16` |",
            "| `control_point_count` | `97` |",
            "| `runtime_interface` | `FlatCategoricalPerResidualLayer` |",
            "| `dictionary_scope` | `PerResidualLayer` |",
            "| `runtime_topology` | `None` |",
            "",
            "## Screening Variables",
            "",
            "| Variable | Values |",
            "|---|---|",
            f"| `path_search_policy` | `{_join_values(PATH_SEARCH_POLICY_VALUES)}` |",
            f"| `construction_policy` | `{_join_values(CONSTRUCTION_POLICY_VALUES)}` |",
            f"| `utility_candidate_budget` | `{_join_values(UTILITY_CANDIDATE_BUDGET_VALUES)}` |",
            f"| `layer_normalization_policy` | `{_join_values(LAYER_NORMALIZATION_POLICY_VALUES)}` |",
            f"| `no_damage_policy` | `{_join_values(NO_DAMAGE_POLICY_VALUES)}` |",
            f"| `atom_preprocessing_policy` | `{_join_values(ATOM_PREPROCESSING_POLICY_VALUES)}` |",
            f"| `duplicate_suppression_policy` | `{_join_values(DUPLICATE_SUPPRESSION_POLICY_VALUES)}` |",
            "",
            "## Method Notes",
            "",
            "- `W=8` means eight atom choices per residual layer.",
            "- `D=16` means sixteen residual layers.",
            "- `control_point_count=97` is fixed decoder geometry.",
            "- The indices-only baseline has `head_outputs = 32 + 16 * 8 = 160`.",
            "- `PhaseAndResidualGain` has `head_outputs = 32 + 16 * 8 + 17 phase_scalars + 16 residual_gain_scalars = 193`.",
            "- Every residual layer reserves `Atom0 = NoOpAtom`, leaving seven active repair atoms.",
            "- PascalCase is used for variable values in reports and artifacts; variable field names remain implementation-friendly.",
            "- Offline/oracle construction may use corpus residuals to build atoms. Deployed runtime still uses flat categorical per-residual-layer atom selection and does not receive topology or corpus metadata.",
            "- Decoder/free policies such as layer clipping change reconstruction deterministically and add zero model prediction head outputs.",
            "",
            "## Run And Artifact Notes",
            "",
            "Full run command:",
            "",
            "```powershell",
            r"conda run --no-capture-output -n py312 python .\research\experiments\lfo_representation\era2\code\experiment12_component_ladder.py --mkl-threading-layer SEQUENTIAL --native-threads 1 run --async --backend xpu --metadata .\datasets\presetshare\raw\presetshare_vital_metadata.csv --corpus-sample-fraction 1.0 --monitor-refresh-seconds 15",
            "```",
            "",
            "Regenerate this report from completed artifacts:",
            "",
            "```powershell",
            r"conda run --no-capture-output -n py312 python .\research\experiments\lfo_representation\era2\code\experiment12_component_ladder.py --mkl-threading-layer SEQUENTIAL --native-threads 1 analyze --run-dir .\research\experiments\lfo_representation\era2\artifacts\experiment_12\component_ladder",
            "```",
            "",
            f"- Completed rows: `{len(rows)}/72`.",
            "- CSV artifacts live under `research/experiments/lfo_representation/era2/artifacts/experiment_12/component_ladder/`.",
            "- Report images live under `research/experiments/lfo_representation/era2/reports/images/experiment_12/`.",
            "- XPU acceleration was added for optimized phase/gain lattice alignment during the run work. Treat that as workflow/runtime context only; it is not a model-quality variable.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def _independent_variable_chapters() -> list[str]:
    reads = {
        "path_search_policy": "This family asks whether the decoder should keep a wider path beam while choosing atom sequences. The per-family plots show `Beam8Path` buys a small P95 improvement over `Beam4Path`, but the diagnostic panel shows the expected encoding-time cost. It is worth keeping both only if Experiment 13 can afford the extra rows.",
        "construction_policy": "This is the most important process-like family. The co-primary plot shows why there is no single automatic winner: `CommonCaseRepair` dominates median and strict-perfect behavior, while `FinishRepairRescue` gives the better balanced tail and node-max result. This family should get real width in Experiment 13.",
        "utility_candidate_budget": "This family tests how many candidate residuals the offline construction policy considers before choosing a repair atom. The plots show diminishing returns rather than a clean monotonic curve. `CandidateBudget48` is the best quality candidate under `PhaseAndResidualGain`, but `CandidateBudget24` and `CandidateBudget12` remain useful cost controls.",
        "layer_normalization_policy": "This family tests decoder/free end-of-layer state policies. The metric plot shows hard clipping is genuinely useful for the tail: `LayerClip0To1` and `LayerCenterPreserveClip` are the clean candidates. The diagnostic plot separates those wins from policies that merely suppress overshoot while leaving reconstruction quality worse.",
        "no_damage_policy": "This family tests whether late layers should be prevented from making an already-good reconstruction worse. The family plots are mostly flat, which means the required `NoOpAtom` already handles much of the safety behavior. Keep this axis small in Experiment 13.",
        "atom_preprocessing_policy": "This family tests whether residual atoms should be normalized before being put into layer dictionaries. `EnergyNormalizedAtoms` is a plausible keeper because it is competitive under `PhaseAndResidualGain`; `CenteredEnergyNormalizedAtoms` is riskier because the `IndicesOnly` plot shows a clear degradation.",
        "duplicate_suppression_policy": "This family tests whether phase/scale-near-duplicate atoms should be removed during construction. The quality plot is essentially unchanged, while the diagnostic plot shows extra construction cost. Keep both only if Experiment 13 has room; otherwise this is a lower-priority axis.",
    }
    lines = ["## Independent Variable Chapters", ""]
    for variable in _screening_variables():
        title = _screening_title(variable)
        lines.extend(
            [
                f"### {title}",
                "",
                reads[variable],
                "",
                f"![{title} co-primary metrics](./images/experiment_12/experiment12_{variable}_co_primary.png)",
                "",
                f"![{title} runtime and diagnostics](./images/experiment_12/experiment12_{variable}_diagnostics.png)",
                "",
            ]
        )
    return lines


def _report_row(rows: list[dict[str, Any]], variable: str, value: str, scalar_schema: str) -> dict[str, Any] | None:
    return next(
        (
            row
            for row in rows
            if row.get("screening_variable") == variable
            and row.get("screening_value") == value
            and row.get("scalar_schema") == scalar_schema
        ),
        None,
    )


def _metric(row: dict[str, Any] | None, key: str) -> str:
    if row is None:
        return "n/a"
    return _fmt(row.get(key))


def _join_values(values: tuple[str, ...]) -> str:
    return "`, `".join(values)


def _screening_table(variable: str, rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        f"### `{variable}`",
        "",
        "| Value | ScalarSchema | Median RMSE | Perfect Rate | P95 RMSE | Node Max P95 | Construct s | Encode s | NoOp Median | Overshoot Rate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ordered = sorted(rows, key=lambda row: (str(row.get("screening_value", "")), str(row.get("scalar_schema", ""))))
    for row in ordered:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.get('screening_value', '')}`",
                    f"`{row.get('scalar_schema', '')}`",
                    _fmt(row.get("validation_median_rmse")),
                    _fmt(row.get("validation_strict_perfect_lfo_rate")),
                    _fmt(row.get("validation_p95_rmse")),
                    _fmt(row.get("validation_node_max_error_p95")),
                    _fmt(row.get("oracle_construction_time")),
                    _fmt(row.get("validation_encoding_time")),
                    _fmt(row.get("residual_layer_no_op_usage_rate_median")),
                    _fmt(row.get("validation_overshoot_rate_before_final_clip")),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def _screening_variables() -> list[str]:
    return [
        "path_search_policy",
        "construction_policy",
        "utility_candidate_budget",
        "layer_normalization_policy",
        "no_damage_policy",
        "atom_preprocessing_policy",
        "duplicate_suppression_policy",
    ]


def _screening_value_order(variable: str) -> list[str]:
    values_by_variable = {
        "path_search_policy": PATH_SEARCH_POLICY_VALUES,
        "construction_policy": CONSTRUCTION_POLICY_VALUES,
        "utility_candidate_budget": UTILITY_CANDIDATE_BUDGET_VALUES,
        "layer_normalization_policy": LAYER_NORMALIZATION_POLICY_VALUES,
        "no_damage_policy": NO_DAMAGE_POLICY_VALUES,
        "atom_preprocessing_policy": ATOM_PREPROCESSING_POLICY_VALUES,
        "duplicate_suppression_policy": DUPLICATE_SUPPRESSION_POLICY_VALUES,
    }
    return list(values_by_variable.get(variable, ()))


def _screening_title(variable: str) -> str:
    return " ".join(part.capitalize() for part in variable.split("_"))


def _variable_panel_plot(
    path: Path,
    rows: list[dict[str, Any]],
    variable: str,
    panels: list[tuple[str, str, str]],
    title: str,
    plt: Any,
) -> None:
    values = [value for value in _screening_value_order(variable) if any(row.get("screening_value") == value for row in rows)]
    extras = sorted({str(row.get("screening_value", "")) for row in rows} - set(values))
    values.extend(extras)
    if not values:
        return
    scalar_schemas = ["IndicesOnly", "PhaseAndResidualGain"]
    x = np.arange(len(values), dtype=np.float32)
    width = 0.38
    figure_width = max(9.5, 1.15 * len(values))
    figure, axes = plt.subplots(2, 2, figsize=(figure_width, 7.4), squeeze=False)
    figure.suptitle(title)
    for axis, (metric, ylabel, note) in zip(axes.reshape(-1), panels):
        for schema_index, scalar_schema in enumerate(scalar_schemas):
            offset = (schema_index - 0.5) * width
            data = []
            for value in values:
                row = next((item for item in rows if item.get("screening_value") == value and item.get("scalar_schema") == scalar_schema), None)
                data.append(_float(row.get(metric)) if row else np.nan)
            axis.bar(x + offset, data, width=width, label=scalar_schema)
        axis.set_title(f"{ylabel} ({note})", fontsize=9)
        axis.set_xticks(x)
        axis.set_xticklabels(values, rotation=45, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.25)
    axes[0][0].legend(fontsize=8)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _bar_plot(path: Path, rows: list[dict[str, Any]], metric: str, ylabel: str, title: str, plt: Any) -> None:
    labels = [str(row["row_id"]).replace("_", "\n") for row in rows]
    values = [_float(row.get(metric)) or 0.0 for row in rows]
    colors = ["#4C78A8" if not _truthy(row.get("topology_used_in_construction")) else "#59A14F" for row in rows]
    plt.figure(figsize=(max(10.0, 0.75 * len(labels)), 5.4))
    plt.bar(range(len(values)), values, color=colors)
    plt.xticks(range(len(labels)), labels, rotation=0, ha="center", fontsize=8)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _delta_plot(path: Path, rows: list[dict[str, Any]], baseline_id: str, metric: str, ylabel: str, title: str, plt: Any) -> None:
    baseline = _row_by_id(rows, baseline_id)
    anchor = _float(baseline.get(metric)) if baseline else None
    if anchor is None:
        return
    labels = [str(row["row_id"]).replace("_", "\n") for row in rows]
    values = [(_float(row.get(metric)) or 0.0) - anchor for row in rows]
    colors = ["#2CA02C" if value < 0 else "#D62728" for value in values]
    plt.figure(figsize=(max(10.0, 0.75 * len(labels)), 5.4))
    plt.axhline(0.0, color="#111827", linewidth=1)
    plt.bar(range(len(values)), values, color=colors)
    plt.xticks(range(len(labels)), labels, rotation=0, ha="center", fontsize=8)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _scatter_plot(path: Path, rows: list[dict[str, Any]], x_key: str, y_key: str, xlabel: str, ylabel: str, title: str, plt: Any) -> None:
    x = [_float(row.get(x_key)) or 0.0 for row in rows]
    y = [_float(row.get(y_key)) or 0.0 for row in rows]
    plt.figure(figsize=(8.4, 5.2))
    plt.scatter(x, y, color="#4C78A8")
    label_indices = set(range(len(rows)))
    if len(rows) > 20:
        by_y = sorted(range(len(rows)), key=lambda index: y[index])
        label_indices = set(by_y[:6] + by_y[-4:])
    for index, (row, x_value, y_value) in enumerate(zip(rows, x, y)):
        if index not in label_indices:
            continue
        plt.annotate(str(row["row_id"]).replace("x12_", ""), (x_value, y_value), fontsize=7, xytext=(4, 3), textcoords="offset points")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _head_formula(spec: ComponentRowSpec) -> str:
    parts = ["32 + 16 * 8"]
    if spec.phase_enabled:
        parts.append("17 phase_scalars")
    if spec.residual_gain_enabled:
        parts.append("16 residual_gain_scalars")
    return " + ".join(parts)


def _phase_candidate_count(spec: ComponentRowSpec, resolution: int) -> int:
    return int(resolution) if spec.phase_enabled else 1


def _subset_dataset(dataset: Era2CurveDataset, *, smoke: bool, corpus_sample_fraction: float) -> Era2CurveDataset:
    if smoke:
        return dataset.subset(train_count=SMOKE_TRAIN_COUNT, validation_count=SMOKE_VALIDATION_COUNT)
    if float(corpus_sample_fraction) < 1.0:
        return dataset.subset(
            train_count=max(1, int(len(dataset.train_indices) * float(corpus_sample_fraction))),
            validation_count=max(1, int(len(dataset.validation_indices) * float(corpus_sample_fraction))),
        )
    return dataset


def _empty_encoding(row_count: int) -> ComponentEncoding:
    return ComponentEncoding(
        base_index=np.zeros(row_count, dtype=np.int32),
        base_phase=np.zeros(row_count, dtype=np.float32),
        base_gain=np.ones(row_count, dtype=np.float32),
        residual_layer_indices=[np.zeros(row_count, dtype=np.int32) for _ in range(D)],
        residual_layer_phases=[np.zeros(row_count, dtype=np.float32) for _ in range(D)],
        residual_layer_gains=[np.zeros(row_count, dtype=np.float32) for _ in range(D)],
    )


def _scatter_encoding(target: ComponentEncoding, source: ComponentEncoding, indices: np.ndarray) -> None:
    target.base_index[indices] = source.base_index
    target.base_phase[indices] = source.base_phase
    target.base_gain[indices] = source.base_gain
    for layer in range(D):
        target.residual_layer_indices[layer][indices] = source.residual_layer_indices[layer]
        target.residual_layer_phases[layer][indices] = source.residual_layer_phases[layer]
        target.residual_layer_gains[layer][indices] = source.residual_layer_gains[layer]


def _squared_distance_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.mean((left[:, None, :] - right[None, :, :]) ** 2, axis=2)


def _prefix(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def _row_by_id(rows: list[dict[str, Any]], row_id: str) -> dict[str, Any] | None:
    return next((row for row in rows if row.get("row_id") == row_id), None)


def _load_row_summaries(output_dir: Path) -> list[dict[str, Any]]:
    rows = []
    rows_dir = Path(output_dir) / "rows"
    if not rows_dir.exists():
        return rows
    for path in sorted(rows_dir.glob("*/summary.csv")):
        rows.append(_read_one_csv(path))
    return rows


def _read_one_csv(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return next(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _initial_status(specs: list[ComponentRowSpec], *, smoke: bool, corpus_sample_fraction: float) -> dict[str, Any]:
    return {
        "run_id": "experiment_12_component_ladder",
        "experiment_id": "experiment_12",
        "smoke": smoke,
        "corpus_sample_fraction_requested": float(corpus_sample_fraction),
        "row_count": len(specs),
        "row_order": [spec.row_id for spec in specs],
        "rows": {},
        "overall_tasks_completed": 0,
        "overall_tasks_total": len(specs),
        "overall_tasks_percent": 0.0,
        "current_row_id": "",
        "current_row_number": "",
        "current_task_id": "",
        "current_task_number": "",
        "current_task_phase": "created",
        "current_task_percent": 0.0,
        "started_at_utc": _now(),
        "completed_at_utc": "",
    }


def _reset_run_logs(output_dir: Path) -> None:
    for name in ("events.jsonl", "run_status.json"):
        path = output_dir / name
        if path.exists():
            path.write_text("", encoding="utf-8")


def _mark_row_complete(status: dict[str, Any], row_id: str, index: int, row: dict[str, Any]) -> None:
    status["rows"][row_id] = {
        "status": "completed",
        "completed_at_utc": _now(),
        "head_outputs_actual": row.get("head_outputs_actual", ""),
        "validation_p95_rmse": row.get("validation_p95_rmse", ""),
        "row_elapsed_seconds": row.get("row_elapsed_seconds", ""),
    }
    status["current_task_id"] = row_id
    status["current_task_number"] = index
    status["current_task_phase"] = "complete"
    status["current_task_percent"] = 100.0
    _set_overall_progress(status)


def _set_overall_progress(status: dict[str, Any]) -> None:
    total = int(status.get("row_count", 0))
    completed = sum(1 for row in status.get("rows", {}).values() if row.get("status") == "completed")
    status["overall_tasks_completed"] = completed
    status["overall_tasks_total"] = total
    status["overall_tasks_percent"] = _numeric_percent(completed, total)


def _write_status(output_dir: Path, status: dict[str, Any]) -> None:
    write_json(Path(output_dir) / "run_status.json", status)


def _event(output_dir: Path, status: dict[str, Any], event: str, message: str, *, row_id: str = "") -> None:
    payload = {
        "timestamp_utc": _now(),
        "run_id": status.get("run_id", ""),
        "row_id": row_id,
        "event": event,
        "message": message,
    }
    with (Path(output_dir) / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    _write_status(output_dir, status)


def _read_json_with_retry(path: Path, *, attempts: int = 5, delay_seconds: float = 0.05) -> dict[str, Any]:
    last_error: Exception | None = None
    for _ in range(max(1, attempts)):
        try:
            text = path.read_text(encoding="utf-8")
            if text.strip():
                return json.loads(text)
            last_error = ValueError(f"empty JSON file: {path}")
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            time.sleep(delay_seconds)
    raise RuntimeError(f"could not read JSON status from {path}: {last_error}")


def _progress_percent(message: str) -> float:
    if message.startswith("construction_start"):
        return 0.0
    if message.startswith("construction: residual layer"):
        parts = message.rsplit(" ", 1)[-1].split("/")
        if len(parts) == 2:
            return round(75.0 * int(parts[0]) / max(1, int(parts[1])), 1)
    if message.startswith("construction_complete"):
        return 76.0
    if message.startswith("train encoding"):
        return 88.0
    if message.startswith("validation encoding"):
        return 98.0
    return 50.0


def _readable_phase(message: str) -> str:
    return " ".join(str(message).replace("_", " ").split())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _elapsed_since(value: str) -> float:
    try:
        started = datetime.fromisoformat(str(value))
    except ValueError:
        return 0.0
    return max(0.0, (datetime.now(timezone.utc) - started).total_seconds())


def _format_duration(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def _numeric_percent(done: int, total: int) -> float:
    return round(100.0 * done / total, 1) if total else 0.0


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any) -> str:
    number = _float(value)
    return "n/a" if number is None else f"{number:.4f}"


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _log(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)

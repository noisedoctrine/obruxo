"""Freeze a small, measured correction-learning check before the full matched trial."""

import copy
import json
from math import prod
from pathlib import Path
from time import perf_counter, time

import torch

from obruxo_data.hashing import canonical_sha256, sha256_bytes
from obruxo_data.midi import Performance, TempoMap

from . import category_scale_refresh as refresh
from .audit import write_json
from .benchmark import MemoryMonitor
from .conditioned_basic import COUNTS, restore_conditioned
from .conditioned_bridge import read
from .conditioned_short_audit import CELLS, cached_features, feature_bank
from .features import PITCH_SIZE, TIMBRE_SHAPE, tensor_digest
from .native_fit import append_durable, digest
from .native_fit_storage import target_tensor_bytes
from .paired_native_audit import verify
from .paired_native_fit import run_plan
from .paired_native_step import reciprocal_updates

LIMITS = {"wall_seconds": 10800, "process_tree_bytes": 4 * 1024**3, "archive_bytes": 2 * 1024**3,
          "replay_bytes": 512 * 1024**2, "replay_records": 12, "optimizer_attempts": 49,
          "training_attempts": 4352, "qualification_attempts": 432, "native_executions": 2}


def select_recipes(parent):
    original = set(parent["training_population"]["original_state_ids"])
    recipes = sorted((row for row in parent["training"] if row["state_id"] in original), key=lambda row: row["state_id"])
    if len(original) != 32 or len(recipes) != 32:
        raise ValueError("short fit requires the intact original 32-state training panel")
    selected = []
    for unison in range(3):
        for filter_type in range(2):
            group = [row for row in recipes if row["target"]["categorical"] == [unison, filter_type]]
            if len(group) < 2:
                raise ValueError("short fit needs two original states in every category combination")
            selected.extend(group[:2])
    return selected, recipes


def audit_cells(training, regression):
    cells = []
    for index, recipe in enumerate(training):
        for input_view, playback in CELLS:
            cells.append({"label": f"short-{len(cells):03d}", "recipe": recipe, "input_view": input_view,
                          "playback_view": playback, "training_index": index})
    selected = {row["state_id"] for row in training}
    for recipe in regression:
        if recipe["state_id"] not in selected:
            cells.append({"label": f"short-{len(cells):03d}", "recipe": recipe, "input_view": 0, "playback_view": 0, "training_index": None})
    return cells


def storage_bound(training, cells, tail_seconds):
    def frames(recipe, view):
        performance = Performance.from_dict(recipe["performances"][view])
        return TempoMap.from_performance(performance).render_frame_count(performance.end_tick, tail_seconds, 44100)
    source = [frames(recipe, view) for recipe in training for view in range(3)]
    audit = [frames(cell["recipe"], cell["playback_view"]) for cell in cells]
    records = sum(target_tensor_bytes(count) for count in source) + len(source) * 4 * (prod(TIMBRE_SHAPE) + PITCH_SIZE)
    values = {"training_source_target_and_feature_tensor_bytes": records,
              "audit_source_audio_current_and_previous_bytes": 2 * 8 * sum(audit),
              "candidate_waveform_bytes": 3 * 8 * sum(audit),
              "largest_transient_waveform_bytes": 8 * max(source + audit),
              "record_and_waveform_serialization_allowance_bytes": len(training) * 128 * 1024 + len(audit) * 5 * 64 * 1024,
              "checkpoint_export_prefix_interruption_reserve_bytes": 64 * 1024**2,
              "native_measurement_ledger_and_publication_reserve_bytes": 128 * 1024**2,
              "archive_guard_bytes": 32 * 1024**2}
    if records > LIMITS["replay_bytes"]:
        raise ValueError("short fit source replay exceeds its declared budget")
    total = sum(values.values())
    if total > LIMITS["archive_bytes"]:
        raise ValueError("short fit retained audio exceeds its declared archive budget")
    return {"format": "conditioned_short_fit_archive_bound_v1", **values, "estimated_archive_upper_bound_bytes": total,
            "training_source_frames": source, "audit_playback_frames": audit, "candidate_waveforms_maximum": 3 * len(audit)}


def make_plan(pool, parent, bridge, cache_path, retrieval_path):
    proof, bridge_proof, bridge_plan = read(parent / "verification.json"), read(bridge / "summary.json"), read(bridge / "plan.json")
    prior = proof["plan"]
    if (not proof["verified"] or not proof["offline_training_replay_exact"] or not proof["complete"] or not proof["all_native_accepted"]
            or not bridge_proof["initialization_and_gradient_access_passed"] or not bridge_proof["parent_artifacts_unchanged"]
            or bridge_proof["new_native_requests"] != 0 or bridge_proof["new_optimizer_updates"] != 0
            or canonical_sha256(bridge_plan) != bridge_proof["plan_id"] or bridge_plan["parent_plan_id"] != proof["plan_id"]
            or bridge_plan["input_files"]["parent/verification.json"] != digest(parent / "verification.json")):
        raise ValueError("short fit requires the verified matched parent and qualified correction bridge")
    if any(digest(parent / name) != value for name, value in proof["evidence_sha256"].items()):
        raise ValueError("short fit verified parent evidence changed")
    for name, value in bridge_plan["source"].items():
        if sha256_bytes(Path(__file__).with_name(name).read_bytes().replace(b"\r\n", b"\n")) != value:
            raise ValueError("short fit changed the qualified bridge implementation")
    initial_path = bridge / "fit-000/model.pt"
    if (digest(initial_path) != bridge_proof["model_file_sha256"] or digest(cache_path) != prior["input_cache_sha256"]
            or digest(retrieval_path) != prior["retrieval_file_sha256"] or pool.pool_id != prior["pool_id"]):
        raise ValueError("short fit initializer, cache, retrieval or training pool changed")
    saved = torch.load(initial_path, map_location="cpu", weights_only=True)
    model, _frontend, codec = restore_conditioned(saved)
    if tensor_digest(model.state_dict()) != bridge_proof["model_sha256"] or codec.to_dict() != prior["codec"]:
        raise ValueError("short fit initializer tensor or codec identity changed")
    training, regression = select_recipes(prior)
    train_ids = {state for state, assignment in pool.manifest.by_state.items() if assignment.split == "train"}
    if not {row["state_id"] for row in regression}.issubset(train_ids):
        raise ValueError("short fit cannot use a held-out state")
    bank = feature_bank(cache_path)
    for recipe in regression:
        for view in range(3):
            cached_features(bank, recipe, view)
    cells = audit_cells(training, regression)
    updates = reciprocal_updates(len(training), seed=0) + reciprocal_updates(len(training), seed=1)
    plan = copy.deepcopy(prior)
    for key in ("training_population", "validation_panels"):
        plan.pop(key, None)
    names = ("conditioned_short_fit.py", "conditioned_short_audit.py", "conditioned_fit_policy.py", "paired_native_fit.py", "paired_native_audit.py",
             "paired_native_step.py", "paired_native_training.py", "paired_native_evidence.py", "paired_performance_objective.py", "local_execution.py")
    source = {**prior["source"], **bridge_plan["source"], **{name: sha256_bytes(Path(__file__).with_name(name).read_bytes().replace(b"\r\n", b"\n")) for name in names}}
    script = Path(__file__).parents[1] / "reports/scripts/run_conditioned_short_fit.py"
    source["reports/scripts/run_conditioned_short_fit.py"] = sha256_bytes(script.read_bytes().replace(b"\r\n", b"\n"))
    return {**plan, "format": "conditioned_basic_short_native_fit_v1", "training": training, "validation": [], "updates": updates,
            "fit_audit_cells": cells, "regression_state_ids": [row["state_id"] for row in regression],
            "selection": "first_two_lexicographic_original_states_per_unison_filter_combination", "epoch_shuffle_seeds": [0, 1],
            "model_budget": COUNTS, "conditioning_training_policy": saved["conditioning_training_policy"],
            "initial_file_sha256": digest(initial_path), "initial_model_sha256": tensor_digest(model.state_dict()),
            "parent_plan_id": proof["plan_id"], "parent_verification_sha256": digest(parent / "verification.json"),
            "bridge_plan_id": bridge_proof["plan_id"], "bridge_summary_sha256": digest(bridge / "summary.json"), "source": source,
            "optimizer": {**prior["optimizer"], "learning_rate": .001, "updates_per_arm": 48}, "limits": LIMITS,
            "planned_maximum_requests": {"training_targets": 36, "gradient_probes": 4224, "fit_audit": 416, "validation_audit": 0, "total": 4676},
            "storage_bound": storage_bound(training, cells, codec.contract.tail_seconds),
            "gates": {"six_input_playback_groups": "sound_at_most_0.9_initial_and_at_most_parameter_mrstft_strictly_lower_than_initial",
                      "original_32": "mean_sound_and_mrstft_at_most_initial_mrstft_p90_at_most_1.1_initial",
                      "every_correction_group_after_update_one": True, "frozen_weights_and_absent_gradients_each_update": True},
            "development_scoring_allowed": False, "model_admission_allowed": False}


def execution_history(output, plan):
    path = output / "native-executions.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    if len(rows) > plan["limits"]["native_executions"] or any(row["ordinal"] != index + 1 or row["plan_id"] != canonical_sha256(plan)
            or row["resume"] != (index > 0) for index, row in enumerate(rows)):
        raise ValueError("short fit native execution history changed")
    return rows


def charge_execution(output, plan, *, resume):
    if read(output / "plan.json") != plan:
        raise ValueError("short fit frozen plan changed")
    rows = execution_history(output, plan)
    if len(rows) >= plan["limits"]["native_executions"]:
        raise ValueError("short fit native execution allocation exhausted")
    if bool(rows) != resume or (output / "run.json").exists() != resume:
        raise ValueError("short fit recovery must be explicit and follow an attempted run")
    if resume:
        summary = read(output / "summary.json")
        events = [json.loads(line) for line in (output / "renders.jsonl").read_text(encoding="utf-8").splitlines()]
        failed = [row for row in events if row["event"] == "outcome" and row["status"] != "accepted"]
        if (summary["complete"] or not summary["failure"] or not summary["within_resource_limits"] or not failed
                or events[-1] != failed[-1] or any(row.get("error_type") != "RenderTimeout" or "qa" in row for row in failed)):
            raise ValueError("short fit can only recover a terminal known native timeout within the original limits")
    row = {"ordinal": len(rows) + 1, "plan_id": canonical_sha256(plan), "resume": resume, "started_unix": time()}
    append_durable(output / "native-executions.jsonl", row)
    return row


def run_short(pool, parent, bridge, cache_path, retrieval_path, output, *, mode="prepare"):
    started, monitor = perf_counter(), MemoryMonitor()
    monitor.start()
    with refresh.cpu_settings():
        try:
            plan = make_plan(pool, parent, bridge, cache_path, retrieval_path)
        finally:
            monitor.close()
        preparation = {"elapsed_seconds": perf_counter() - started, "peak_process_tree_rss_bytes": monitor.peak_bytes,
                       "rss_sampling_errors": monitor.errors, "native_requests": 0, "optimizer_updates": 0}
        if (preparation["elapsed_seconds"] >= 600 or monitor.peak_bytes is None or monitor.peak_bytes > 4 * 1024**3 or monitor.errors):
            raise ValueError("short fit preparation exceeded ten minutes or 4 GiB")
        if mode == "verify":
            return run_verification(output, pool, bridge, cache_path, plan)
        if mode not in ("prepare", "execute", "resume"):
            raise ValueError("unknown short fit execution mode")
        if mode != "prepare":
            charge_execution(output, plan, resume=mode == "resume")
        result = run_plan(pool, bridge / "fit-000/model.pt", retrieval_path, output, plan, execute=mode != "prepare",
                          resume=mode == "resume", input_validation=preparation, input_cache_path=cache_path)
        if mode == "prepare":
            write_json(output / "preparation.json", preparation)
        return {**result, "input_validation": preparation}


def run_verification(output, pool, bridge, cache_path, plan):
    path = output / "verification.json"
    if path.exists():
        raise ValueError("preserve the existing short fit verification")
    started, monitor = perf_counter(), MemoryMonitor()
    monitor.start()
    try:
        if not execution_history(output, plan):
            raise ValueError("short fit native execution history is missing")
        result = verify(output, pool, bridge, None, None, cache_path, None, None, expected_plan=plan)
    finally:
        monitor.close()
    within = (perf_counter() - started < 1200 and monitor.peak_bytes is not None and 0 < monitor.peak_bytes <= 4 * 1024**3 and not monitor.errors)
    result["verification_resources"] = {"elapsed_seconds": perf_counter() - started, "peak_process_tree_rss_bytes": monitor.peak_bytes,
                                        "rss_sampling_errors": monitor.errors, "within_resource_limits": within}
    result["verified"] = bool(within)
    write_json(path, result)
    if not within:
        raise ValueError("short fit verification exceeded twenty minutes or 4 GiB")
    return {key: result[key] for key in ("verified", "matched_comparison_passed", "offline_training_replay_exact", "comparison_waveforms_rescored", "verification_resources")}

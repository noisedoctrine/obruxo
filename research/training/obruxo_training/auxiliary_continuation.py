"""Continue a verified timeout prefix under a separately declared total wall window."""

from contextlib import contextmanager
import copy
from datetime import datetime
from pathlib import Path
import shutil
from time import perf_counter, time

import psutil
import torch

from obruxo_data.hashing import canonical_sha256, sha256_bytes

from . import basic_direct_fit as fit
from . import category_scale_refresh as refresh
from .audit import write_json
from .basic_direct_fit_audit import run_verification
from .basic_direct_fit_data import Inputs
from .benchmark import MemoryMonitor
from .checkpoints import atomic_torch_save
from .conditioned_bridge import read
from .native_fit import append_durable, digest
from .paired_native_audit import equal_tree

TOTAL_WALL_SECONDS = 24 * 3600
LEDGERS = ("renders.jsonl", "updates.jsonl", "probes.jsonl", "finals.jsonl")


def manifest(source):
    result = {}
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(source.resolve()):
            raise ValueError("continuation source must contain ordinary local evidence files")
        result[path.relative_to(source).as_posix()] = digest(path)
    return result


def require_terminal(source):
    import json
    launch = json.loads((source / "launch.json").read_text(encoding="utf-8-sig"))
    created = datetime.fromisoformat(launch["CreationDate"].replace("Z", "+00:00")).timestamp()
    try:
        process = psutil.Process(launch["ProcessId"])
        if process.is_running() and abs(process.create_time() - created) < .1:
            raise ValueError("the original fit is still live; do not duplicate it")
    except psutil.NoSuchProcess:
        pass
    service = read(source / "service-001-completion.json")
    if service["exit_code"] == 0 or not service["failure"]:
        raise ValueError("continuation requires a terminal failed original execution")


def require_timeout(source, plan):
    require_terminal(source)
    summary, executions = read(source / "summary.json"), fit.lines(source / "executions.jsonl")
    if (summary["plan_id"] != canonical_sha256(plan) or summary["complete"] or not summary["failure"] or summary["result"] is not None
            or not plan["limits"]["wall_seconds"] < summary["elapsed_seconds"] < TOTAL_WALL_SECONDS
            or not summary["peak_process_tree_rss_bytes"] or summary["peak_process_tree_rss_bytes"] > plan["limits"]["process_tree_bytes"]
            or summary["rss_sampling_errors"] or len(executions) != 1 or executions[0]["ordinal"] != 1 or executions[0]["resume"]
            or executions[0]["plan_id"] != canonical_sha256(plan)):
        raise ValueError("only the known timeout with an expired original wall window may continue")
    events = fit.lines(source / "renders.jsonl")
    attempts, outcomes = {}, {}
    for row in events:
        destination = attempts if row["event"] == "attempt" else outcomes
        key = row["phase"], row["attempt"]
        if row["event"] not in ("attempt", "outcome") or key in destination:
            raise ValueError("original native event ledger changed")
        destination[key] = row
    if attempts.keys() != outcomes.keys():
        raise ValueError("an unobserved native request cannot be recovered as a known timeout")
    failures = [row for row in outcomes.values() if row["status"] != "accepted"]
    if len(failures) != 1 or events[-1] != failures[0] or failures[0].get("error_type") != "RenderTimeout" or "qa" in failures[0]:
        raise ValueError("continuation requires exactly one terminal known native timeout")
    for key, outcome in outcomes.items():
        if any(outcome[field] != attempts[key][field] for field in ("phase", "label", "request_id", "source_sha256", "controlled_sha256")):
            raise ValueError("original native outcome differs from its charged request")
        if outcome["status"] == "accepted" and outcome["renderer_identity"] != plan["expected_renderer_identity"]:
            raise ValueError("original renderer identity changed")
    return summary, failures[0]


def make_plan(source, parent, bridge, features, source_plan_sha256):
    if digest(source / "plan.json") != source_plan_sha256:
        raise ValueError("name the exact frozen original plan")
    original = read(source / "plan.json")
    summary, failure = require_timeout(source, original)
    if (digest(bridge / "model.pt") != original["initial_file_sha256"] or digest(features) != original["input_cache_sha256"]
            or digest(parent / "fit_cases.json") != original["input_files"]["parent/fit_cases.json"]):
        raise ValueError("continuation initializer, features or source comparisons changed")
    root = Path(__file__).parents[1]
    for name, value in original["source"].items():
        path = root / name if "/" in name else Path(__file__).with_name(name)
        if sha256_bytes(path.read_bytes().replace(b"\r\n", b"\n")) != value:
            raise ValueError("the frozen fitting implementation changed before recovery")
    saved = torch.load(source / "checkpoint.pt", map_location="cpu", weights_only=True)
    charges = fit.lines(source / "updates.jsonl")
    fit.require_charges(original, charges, saved["trace"])
    if (saved["plan_id"] != canonical_sha256(original) or saved["step"] != summary["committed_updates_per_arm"]
            or not 0 < saved["step"] < len(original["updates"]) or len(charges) != saved["step"] + 1
            or len(charges) + len(original["updates"]) - saved["step"] > original["limits"]["optimizer_attempts"]):
        raise ValueError("the saved prefix cannot finish within the unchanged update-attempt cap")
    files = manifest(source)
    source_bytes = sum((source / name).stat().st_size for name in files)
    archive = original["limits"]["archive_bytes"] - source_bytes
    storage = {**original["storage_bound"], "continuation_provenance_reserve_bytes": 16 * 1024**2,
               "estimated_archive_upper_bound_bytes": original["storage_bound"]["estimated_archive_upper_bound_bytes"] + 16 * 1024**2}
    if storage["estimated_archive_upper_bound_bytes"] > archive:
        raise ValueError("original plus continued evidence exceeds the unchanged aggregate archive cap")
    names = ("auxiliary_continuation.py", "reports/scripts/continue_auxiliary_fit.py")
    code = {name: sha256_bytes((root / name if "/" in name else Path(__file__).with_name(name)).read_bytes().replace(b"\r\n", b"\n")) for name in names}
    return {**copy.deepcopy(original), "format": "continued_basic_auxiliary_after_wall_timeout_v1",
            "limits": {**original["limits"], "wall_seconds": TOTAL_WALL_SECONDS, "archive_bytes": archive},
            "source": {**original["source"], **code}, "storage_bound": storage,
            "operational_continuation": {"original_plan_id": canonical_sha256(original), "original_plan_file_sha256": source_plan_sha256,
                "original_files": files, "original_archive_bytes": source_bytes, "original_budget_passed": False,
                "original_wall_seconds": original["limits"]["wall_seconds"], "total_wall_seconds": TOTAL_WALL_SECONDS,
                "total_archive_bytes": original["limits"]["archive_bytes"], "original_checkpoint_step": saved["step"],
                "original_update_attempts_per_arm": len(charges), "original_native_attempts": sum(summary["render_attempts"].values()),
                "terminal_timeout_request_id": failure["request_id"], "new_epochs": 0, "objective_or_schedule_changed": False}}


def rebind_checkpoint(saved, plan_id):
    rebound = copy.deepcopy(saved)
    rebound["plan_id"] = plan_id
    if not equal_tree({key: value for key, value in saved.items() if key != "plan_id"},
                      {key: value for key, value in rebound.items() if key != "plan_id"}):
        raise ValueError("checkpoint migration changed numerical training state")
    return rebound


def migrate(source, output, plan):
    if source.resolve() == output.resolve() or source.resolve().parent != output.resolve().parent:
        raise ValueError("continuation must use a distinct sibling evidence directory")
    plan_id = canonical_sha256(plan)
    for folder in ("source_audio", "comparison_audio"):
        if (source / folder).exists():
            shutil.copytree(source / folder, output / folder)
    for name in LEDGERS:
        if (source / name).exists():
            shutil.copy2(source / name, output / name)
    saved = torch.load(source / "checkpoint.pt", map_location="cpu", weights_only=True)
    rebound = rebind_checkpoint(saved, plan_id)
    atomic_torch_save(output / "checkpoint.pt", rebound)
    atomic_torch_save(output / "starting-checkpoint.pt", rebound)
    prefix = torch.load(source / "prefix-checkpoint.pt", map_location="cpu", weights_only=True)
    atomic_torch_save(output / "prefix-checkpoint.pt", rebind_checkpoint(prefix, plan_id))
    proof = read(source / "prefix-verification.json")
    write_json(output / "prefix-verification.json", {**proof, "plan_id": plan_id, "original_plan_id": proof["plan_id"],
               "checkpoint_file_sha256": digest(output / "prefix-checkpoint.pt")})
    run = read(source / "run.json")
    write_json(output / "run.json", {**run, "plan_id": plan_id, "original_plan_id": run["plan_id"]})
    original_execution = fit.lines(source / "executions.jsonl")[0]
    append_durable(output / "executions.jsonl", {**original_execution, "plan_id": plan_id,
                   "original_plan_id": original_execution["plan_id"], "metadata_rebound_from_original_execution": True})
    write_json(output / "original-budget-miss.json", read(source / "summary.json"))
    write_json(output / "migration.json", {"plan_id": plan_id, "original_checkpoint_sha256": digest(source / "checkpoint.pt"),
               "starting_checkpoint_sha256": digest(output / "starting-checkpoint.pt"), "numerical_state_unchanged": True})


def validate_migration(source, output, plan):
    policy = plan["operational_continuation"]
    if manifest(source) != policy["original_files"] or read(output / "plan.json") != plan:
        raise ValueError("immutable original evidence or the continuation plan changed")
    current = torch.load(output / "starting-checkpoint.pt", map_location="cpu", weights_only=True)
    original = torch.load(source / "checkpoint.pt", map_location="cpu", weights_only=True)
    if not equal_tree(current, rebind_checkpoint(original, canonical_sha256(plan))):
        raise ValueError("continued starting model, Adam, RNG or trace differs from the original checkpoint")
    prefix = torch.load(source / "prefix-checkpoint.pt", map_location="cpu", weights_only=True)
    if not equal_tree(torch.load(output / "prefix-checkpoint.pt", map_location="cpu", weights_only=True), rebind_checkpoint(prefix, canonical_sha256(plan))):
        raise ValueError("continued prefix differs from the original early recovery check")
    migration = read(output / "migration.json")
    if (migration["plan_id"] != canonical_sha256(plan) or migration["original_checkpoint_sha256"] != digest(source / "checkpoint.pt")
            or migration["starting_checkpoint_sha256"] != digest(output / "starting-checkpoint.pt")
            or read(output / "original-budget-miss.json") != read(source / "summary.json")):
        raise ValueError("continuation migration or original budget-miss record changed")
    for name in LEDGERS:
        if (source / name).exists() and not (output / name).read_bytes().startswith((source / name).read_bytes()):
            raise ValueError("continued native or update ledger lost its original byte prefix")
    run = read(source / "run.json")
    expected = {**run, "plan_id": canonical_sha256(plan), "original_plan_id": run["plan_id"]}
    if read(output / "run.json") != expected:
        raise ValueError("continuation restarted the original clock or changed run provenance")
    execution = fit.lines(source / "executions.jsonl")[0]
    expected_execution = {**execution, "plan_id": canonical_sha256(plan), "original_plan_id": execution["plan_id"],
                          "metadata_rebound_from_original_execution": True}
    if fit.lines(output / "executions.jsonl")[0] != expected_execution:
        raise ValueError("continuation lost the original native execution charge")
    if sum(path.stat().st_size for path in output.rglob("*") if path.is_file()) + policy["original_archive_bytes"] > policy["total_archive_bytes"]:
        raise ValueError("combined original and continuation evidence exceeds its archive cap")


@contextmanager
def continuation_charge(source, output, plan):
    # This operator runs alone in its own process. Reuse the frozen numerical runner;
    # replace only its entry policy, restoring the module immediately afterward.
    original = fit.charge_execution
    def charge(inputs, destination, resume):
        if destination != output or inputs.plan != plan or not resume:
            raise ValueError("continuation entry differs from its frozen destination and plan")
        validate_migration(source, output, plan)
        if time() - read(output / "run.json")["started_unix"] >= TOTAL_WALL_SECONDS:
            raise ValueError("the original-clock continuation window has expired")
        rows = fit.lines(output / "executions.jsonl")
        if len(rows) != 1:
            raise ValueError("both allocated native executions have already been spent")
        append_durable(output / "executions.jsonl", {"ordinal": 2, "plan_id": canonical_sha256(plan), "resume": True,
                       "started_unix": time(), "original_budget_passed": False})
    fit.charge_execution = charge
    try:
        yield
    finally:
        fit.charge_execution = original


def run(source, parent, bridge, features, output, source_plan_sha256, *, mode="prepare"):
    with refresh.cpu_settings():
        if mode in ("prepare", "execute") and time() - read(source / "run.json")["started_unix"] >= TOTAL_WALL_SECONDS:
            raise ValueError("the original-clock continuation window has expired")
        started, monitor = perf_counter(), MemoryMonitor()
        monitor.start()
        try:
            plan = make_plan(source, parent, bridge, features, source_plan_sha256)
            inputs = Inputs(parent, bridge, features, plan)
            inputs.validate()
            if mode == "prepare":
                original_inputs = Inputs(parent, bridge, features, read(source / "plan.json"))
                saved = torch.load(source / "checkpoint.pt", map_location="cpu", weights_only=True)
                fit.replay_training(original_inputs, fit.SoundStore(source, original_inputs), saved, read(source / "prefix-verification.json"))
                output.mkdir(parents=True, exist_ok=False)
                write_json(output / "plan.json", plan)
                migrate(source, output, plan)
                validate_migration(source, output, plan)
        finally:
            monitor.close()
        resources = {"elapsed_seconds": perf_counter() - started, "peak_process_tree_rss_bytes": monitor.peak_bytes, "rss_sampling_errors": monitor.errors}
        if resources["elapsed_seconds"] >= 600 or not monitor.peak_bytes or monitor.peak_bytes > 4 * 1024**3 or monitor.errors:
            raise ValueError("continuation preparation exceeded ten minutes or 4 GiB")
        if mode == "prepare":
            write_json(output / "preparation.json", {**resources, "original_committed_prefix_replay_exact": True,
                       "replayed_updates_per_arm": plan["operational_continuation"]["original_checkpoint_step"], "native_requests": 0,
                       "experimental_training_updates": 0, "numerical_state_unchanged": True})
            return {"plan_id": canonical_sha256(plan), "preparation": resources, "original_checkpoint_replay_exact": True}
        validate_migration(source, output, plan)
        if mode == "execute":
            try:
                with continuation_charge(source, output, plan):
                    fit.execute(inputs, output, resume=True)
            finally:
                if (output / "summary.json").exists():
                    report = read(output / "summary.json")
                    write_json(output / "summary.json", {**report, "original_budget_passed": False,
                               "original_plan_id": plan["operational_continuation"]["original_plan_id"]})
            return read(output / "summary.json")
        if mode == "verify":
            final = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
            start = torch.load(output / "starting-checkpoint.pt", map_location="cpu", weights_only=True)
            if final["trace"][:start["step"]] != start["trace"]:
                raise ValueError("continued training changed the original committed trace")
            fit.replay_training(inputs, fit.SoundStore(output, inputs), start, read(output / "prefix-verification.json"))
            result = run_verification(inputs, output)
            proof = read(output / "verification.json")
            proof.update(original_budget_passed=False, original_committed_prefix_replay_exact=True,
                         original_ledger_prefixes_preserved=True, original_source_evidence_unchanged=True,
                         original_plan_id=plan["operational_continuation"]["original_plan_id"])
            proof["evidence_sha256"].update({name: digest(output / name) for name in ("starting-checkpoint.pt", "migration.json", "original-budget-miss.json")})
            write_json(output / "verification.json", proof)
            return {**result, "original_budget_passed": False, "original_committed_prefix_replay_exact": True}
        raise ValueError("unknown continuation mode")

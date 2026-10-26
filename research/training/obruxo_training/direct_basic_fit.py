"""Checkpointed direct-control fitting and exact offline trajectory replay."""

import copy
import os
import shutil
from time import time

import psutil
import torch

from obruxo_data.hashing import canonical_sha256

from .audit import write_json
from .benchmark import MemoryMonitor
from .checkpoints import atomic_torch_save
from .codec import PilotTarget, load_codec
from .conditioned_bridge import read
from .direct_basic_control import DirectPatch, SPEC, quality, step
from .direct_basic_store import DirectStore
from .native_fit import append_durable
from .paired_native_audit import equal_tree
from .paired_native_fit import PairedSession
from .unison_sound_fit import lines


def checkpoint(output, plan):
    path = output / "checkpoint.pt"
    if not path.exists():
        return None
    saved = torch.load(path, map_location="cpu", weights_only=True)
    rows = lines(output / "measurements.jsonl")
    cursor = saved["measurement_cursor"]
    if (saved["plan_id"] != canonical_sha256(plan) or cursor > len(rows)
            or canonical_sha256(rows[:cursor]) != saved["measurement_prefix_sha256"]):
        raise ValueError("direct checkpoint plan or committed measurements changed")
    return saved


def charge_update(output, plan, case, index):
    attempts = lines(output / "attempts.jsonl")
    if sum(row["case"] == case for row in attempts) >= plan["limits"]["optimizer_attempts_per_case"]:
        raise RuntimeError("direct update attempt allocation exhausted")
    append_durable(output / "attempts.jsonl", {"case": case, "step": index, "ordinal": len(attempts), "started_unix": time()})


def reconstruct(plan, store, saved=None, *, output=None):
    """Rebuild from initialization; cached costs replay committed work without rendering."""
    prior = {(row["case"], row["result"]["step"]): row for row in saved["trace"]} if saved else {}
    if saved and len(prior) != len(saved["trace"]):
        raise ValueError("duplicate direct checkpoint step")
    if saved:
        order = [row["case"] for row in saved["trace"]]
        if order != sorted(order) or sorted(set(order)) != list(range(len(saved["states"]))) or len(saved["states"]) > len(plan["cases"]):
            raise ValueError("direct checkpoint is not an ordered prefix")
        for case in set(order):
            steps = [row["result"]["step"] for row in saved["trace"] if row["case"] == case]
            if steps != list(range(len(steps))) or len(steps) > SPEC["updates"]:
                raise ValueError("direct checkpoint is not a contiguous update prefix")
    trace, states, results, used_prior = [], {}, [], set()
    for case, descriptor in enumerate(plan["cases"]):
        patch = DirectPatch(store.codec, PilotTarget.from_dict(descriptor["initial"]))
        initial = store.comparisons(case, patch.target(), "initial")
        for index in range(SPEC["updates"]):
            key = case, index
            replay = key in prior
            if not replay and output is None:
                raise ValueError("offline direct checkpoint is incomplete")
            store.offline = replay or output is None
            if not replay:
                store.check_limits()
                charge_update(output, plan, case, index)
            result = step(patch, index, lambda candidate, role: store.evaluate(case, candidate, role))
            row = {"case": case, "result": result}
            if replay:
                if row != prior[key]:
                    raise ValueError("direct optimizer trajectory differs from its native measurements")
                used_prior.add(key)
            trace.append(row)
            states[str(case)] = patch.snapshot()
            last_saved = saved and str(case) in saved["states"] and index == max(r["result"]["step"] for r in saved["trace"] if r["case"] == case)
            if last_saved and not equal_tree(states[str(case)], saved["states"][str(case)]):
                raise ValueError("direct controls, categories or Adam differ from the cold checkpoint")
            if not replay:
                rows = list(store.rows.values())
                atomic_torch_save(output / "checkpoint.pt", {"plan_id": canonical_sha256(plan), "trace": trace, "states": copy.deepcopy(states),
                                                            "measurement_cursor": len(rows), "measurement_prefix_sha256": canonical_sha256(rows)})
                write_json(output / "progress.json", {"case": case, "training_index": descriptor["training_index"], "completed_updates": index + 1,
                                                       "accepted_continuous_updates": sum(r["case"] == case and r["result"]["continuous"]["accepted"] for r in trace),
                                                       "joint_loss": result["continuous"]["after_loss"], "consecutive_refusals": patch.refusals,
                                                       "native_measurements": len(store.rows)})
            if patch.refusals >= SPEC["refusal_limit"]:
                break
        store.offline = output is None
        final = store.comparisons(case, patch.target(), "final")
        result = quality([r["scores"] for r in initial], [r["scores"] for r in final], [r["scores"] for r in descriptor["parameter"]])
        results.append({"case": case, "training_index": descriptor["training_index"], "updates": index + 1,
                        "stopped_after_refusals": patch.refusals >= SPEC["refusal_limit"], "initial": initial, "final": final,
                        "final_target": patch.target().to_dict(), "quality": result})
    if used_prior != prior.keys():
        raise ValueError("direct checkpoint has unreachable or reordered updates")
    return {"cases": results, "pass": all(row["quality"]["pass"] for row in results), "trace": trace, "states": states}


def charge_execution(output, plan, resume):
    attempts = lines(output / "executions.jsonl")
    if bool(attempts) != resume or len(attempts) >= plan["limits"]["native_executions"] or (output / "run.json").exists() != resume:
        raise ValueError("direct execution or recovery allocation changed")
    if resume:
        last = attempts[-1]
        try:
            live = psutil.Process(last["pid"])
            if abs(live.create_time() - last["process_created_unix"]) < .01:
                raise ValueError("the preceding direct execution is still live")
        except psutil.NoSuchProcess:
            pass
        previous = read(output / "summary.json")
        events = lines(output / "renders.jsonl")
        failures = [row for row in events if row["event"] == "outcome" and row["status"] == "error"]
        if (previous["complete"] or not previous["within_resource_limits"] or not previous["failure"] or not failures
                or events[-1] != failures[-1] or any(row.get("error_type") != "RenderTimeout" for row in failures)):
            raise ValueError("direct recovery requires a terminal known native timeout")
        archive = output / "interruption"
        archive.mkdir()
        for name in ("summary.json", "checkpoint.pt", "checkpoint.previous.pt", "measurements.jsonl", "renders.jsonl", "attempts.jsonl"):
            if (output / name).exists():
                shutil.copy2(output / name, archive / name)
    else:
        write_json(output / "run.json", {"plan_id": canonical_sha256(plan), "started_unix": time()})
    append_durable(output / "executions.jsonl", {"plan_id": canonical_sha256(plan), "ordinal": len(attempts), "resume": resume,
                                               "pid": os.getpid(), "process_created_unix": psutil.Process().create_time(), "started_unix": time()})


def execute(plan, fit, output, *, resume=False):
    charge_execution(output, plan, resume)
    monitor = MemoryMonitor()
    monitor.start()
    codec = load_codec(plan["codec"])
    session = PairedSession(output, plan, codec, monitor)
    store = DirectStore(output, plan, codec, fit, session)
    failure, result = None, None
    try:
        result = reconstruct(plan, store, checkpoint(output, plan), output=output)
        write_json(output / "result.json", {key: value for key, value in result.items() if key not in ("trace", "states")})
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        monitor.close()
        elapsed = time() - session.started
        archive_bytes = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
        attempts = sum(row["event"] == "attempt" for row in session.events)
        resources = bool(monitor.peak_bytes and monitor.peak_bytes <= plan["limits"]["process_tree_bytes"] and not monitor.errors
                         and elapsed <= plan["limits"]["wall_seconds"] and archive_bytes + 1024**2 <= plan["limits"]["archive_bytes"]
                         and attempts <= plan["maximum_native_requests"])
        report = {"plan_id": canonical_sha256(plan), "complete": failure is None and result is not None, "failure": failure,
                  "elapsed_seconds": elapsed, "peak_process_tree_rss_bytes": monitor.peak_bytes, "rss_sampling_errors": monitor.errors,
                  "archive_bytes_before_summary": archive_bytes, "within_resource_limits": resources, "native_attempts": attempts,
                  "committed_updates": len(checkpoint(output, plan)["trace"]) if (output / "checkpoint.pt").exists() else 0,
                  "direct_quality_passed_before_independent_verification": bool(result and result["pass"]),
                  "inverse_training_updates": 0, "qualified_for_release": False, "test_split_used": False}
        write_json(output / "summary.json", report)
    return report

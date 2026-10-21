"""Matched short fit on the known training pairs; no native renders or helper promotion."""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
from time import time

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


@dataclass(frozen=True)
class FitConfig:
    seed: int = 0
    steps: int = 1000
    batch_size: int = 8
    learning_rate: float = .001
    checkpoint_every: int = 100


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_inputs(parent, repaired):
    import torch
    from obruxo_data.hashing import canonical_sha256
    from obruxo_training.paired_spectral import require_aligned_pairs
    plan = json.loads((parent / "plan.json").read_text())
    preflight = json.loads((repaired / "summary.json").read_text())
    if (not preflight["complete"] or not preflight["preflight_passed"]
            or preflight["plan"]["parent_plan_id"] != canonical_sha256(plan)
            or preflight["plan"]["parent_records_sha256"] != digest(parent / "records.json")):
        raise ValueError("fitting check requires matching completed preflight evidence")
    index = {row["recipe_id"]: row for row in json.loads((parent / "records.json").read_text())}
    records, identities = {"uniform": [], "events": []}, []
    for case, selected in zip(preflight["cases"], preflight["plan"]["pairs"], strict=True):
        if (case["group"], case["pair_id"]) != (selected["group"], selected["pair_id"]):
            raise ValueError("preflight pair order changed")
        files = []
        for recipe, saved in zip(selected["recipes"], case["files"], strict=True):
            row = index[canonical_sha256(recipe)]
            paths = {"uniform": (parent / "records" / row["file"], row["sha256"]),
                     "events": (repaired / saved["file"], saved["sha256"])}
            pair_files = {}
            for arm, (path, expected) in paths.items():
                if digest(path) != expected:
                    raise ValueError("training feature bytes changed")
                record = torch.load(path, map_location="cpu", weights_only=True)
                if record["recipe_id"] != canonical_sha256(recipe) or record["target"] != recipe["target"]:
                    raise ValueError("training record recipe changed")
                records[arm].append(record)
                pair_files[arm] = expected
            old, new = records["uniform"][-1], records["events"][-1]
            if old["source_state_id"] != new["source_state_id"] or not torch.equal(old["performance"], new["performance"]):
                raise ValueError("matched arms have different ancestry or MIDI")
            files.append(pair_files)
        for arm in records:
            require_aligned_pairs(records[arm][-2:])
        identities.append({"group": case["group"], "pair_id": case["pair_id"], "files": files})
    if len(identities) != 12:
        raise ValueError("expected all 12 frozen training pairs")
    return records, identities, plan["codec"], preflight


def make_plan(parent, repaired, identities, codec, preflight):
    from obruxo_data.hashing import sha256_bytes
    from obruxo_training.paired_spectral import pair_spec
    paths = [Path(__file__)]
    paths += [RESEARCH / "training/obruxo_training" / name for name in
              ("temporal_proxy.py", "temporal_proxy_screen.py", "paired_spectral.py", "spectral.py", "spectral_controls.py", "event_frames.py")]
    return {"format": "matched_event_response_fit_v1", "config": asdict(FitConfig()), "codec": codec, "pairs": identities,
            "parent_plan_id": preflight["plan"]["parent_plan_id"], "parent_records_sha256": digest(parent / "records.json"),
            "preflight_sha256": digest(repaired / "summary.json"), "objective": pair_spec(),
            "source": {path.name: sha256_bytes(path.read_bytes().replace(b"\r\n", b"\n")) for path in paths},
            "arms": ["uniform", "events"], "evaluation": "same_repaired_training_measurements_for_both_arms_no_generalization_claim",
            "limits": {"optimizer_updates_per_arm": 1000, "total_optimizer_updates": 2000, "native_requests": 0,
                       "wall_seconds": 1800, "process_tree_bytes": 8 * 1024**3, "archive_bytes": 256 * 1024**2},
            "gates": {"attack_decay_release_mean_vs_no_change": .8, "attack_decay_release_cases_beating_no_change": 2,
                      "detune_mean_vs_no_change": 1., "events_absolute_error_vs_uniform": 1.1},
            "qualified_for_inverse_training": False, "test_split_used": False}


def score(helper, records, adapter, identities):
    import torch
    from obruxo_training.codec import PilotTarget
    from obruxo_training.paired_spectral import paired_change_loss
    from obruxo_training.spectral import spectral_loss
    from obruxo_training.temporal_proxy import collate_frames, hard_controls
    results = []
    with torch.no_grad():
        for index, identity in enumerate(identities):
            pair = records[2 * index:2 * index + 2]
            target, times = collate_frames(pair)
            controls = hard_controls(adapter, [PilotTarget.from_dict(row["target"]) for row in pair])
            predicted = helper(controls, torch.stack([row["performance"] for row in pair]), times)
            unchanged = {"magnitudes": tuple(value[:1].expand_as(value) for value in target["magnitudes"]),
                         "envelope": target["envelope"][:1].expand_as(target["envelope"])}
            results.append({"group": identity["group"], "pair_id": identity["pair_id"],
                            "response_error": float(paired_change_loss(predicted, target).item()),
                            "no_change_error": float(paired_change_loss(unchanged, target).item()),
                            "absolute_error": float(spectral_loss(predicted, target)["loss"].mean())})
    return results


def summarize(results, gates):
    groups = {}
    for group in ("attack", "decay", "release", "detune"):
        arms = {}
        for arm, rows in results.items():
            selected = [row for row in rows if row["group"] == group]
            if len(selected) != 3:
                raise ValueError("incomplete response fit scores")
            arms[arm] = {"response_error": sum(row["response_error"] for row in selected) / 3,
                         "no_change_error": sum(row["no_change_error"] for row in selected) / 3,
                         "absolute_error": sum(row["absolute_error"] for row in selected) / 3,
                         "beats_no_change": sum(row["response_error"] < row["no_change_error"] for row in selected)}
        event = arms["events"]
        ratio = gates["detune_mean_vs_no_change"] if group == "detune" else gates["attack_decay_release_mean_vs_no_change"]
        passed = event["response_error"] <= ratio * event["no_change_error"]
        if group == "detune":
            passed = event["response_error"] < event["no_change_error"]
        if group != "detune":
            passed = passed and event["beats_no_change"] >= gates["attack_decay_release_cases_beating_no_change"]
        groups[group] = {"arms": arms, "response_fit_pass": passed}
    event_absolute = sum(row["absolute_error"] for row in results["events"]) / 12
    uniform_absolute = sum(row["absolute_error"] for row in results["uniform"]) / 12
    absolute_pass = event_absolute <= gates["events_absolute_error_vs_uniform"] * uniform_absolute
    return {"groups": groups, "absolute_error": {"events": event_absolute, "uniform": uniform_absolute}, "absolute_pass": absolute_pass,
            "fit_gate_passed": absolute_pass and all(row["response_fit_pass"] for row in groups.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--repaired", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    import torch
    from obruxo_data.hashing import canonical_sha256
    from obruxo_training.audit import write_json
    from obruxo_training.benchmark import MemoryMonitor
    from obruxo_training.spectral_controls import BasicControlAdapter
    from obruxo_training.codec import load_codec
    from obruxo_training.temporal_proxy_screen import fit_temporal_records
    from obruxo_training.trainer import atomic_torch_save
    torch.set_num_threads(2)
    records, identities, codec, preflight = load_inputs(args.parent, args.repaired)
    plan = make_plan(args.parent, args.repaired, identities, codec, preflight)
    if not args.execute:
        if args.resume:
            raise ValueError("resume requires execute")
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output / "plan.json", plan)
        print(json.dumps({"plan_id": canonical_sha256(plan), "pairs": len(identities), "updates_per_arm": 1000}))
        return
    if json.loads((args.output / "plan.json").read_text()) != plan:
        raise ValueError("frozen response-fit plan changed")
    run_path = args.output / "run.json"
    if args.resume != run_path.exists():
        raise ValueError("use resume only for an existing run")
    if not args.resume:
        write_json(run_path, {"started_unix": time(), "plan_id": canonical_sha256(plan)})
    started = json.loads(run_path.read_text())["started_unix"]
    monitor = MemoryMonitor()
    monitor.start()
    update_path = args.output / "updates.jsonl"
    charges = [json.loads(line) for line in update_path.read_text().splitlines()] if update_path.exists() else []
    attempted = {arm: sum(row["arm"] == arm for row in charges) for arm in plan["arms"]}
    def check_limits():
        if time() - started > plan["limits"]["wall_seconds"] or (monitor.peak_bytes or 0) > plan["limits"]["process_tree_bytes"]:
            raise RuntimeError("response-fit time or memory cap exceeded")
        if sum(path.stat().st_size for path in args.output.rglob("*") if path.is_file()) > plan["limits"]["archive_bytes"]:
            raise RuntimeError("response-fit archive cap exceeded")
    results, arm_seconds, failure = {}, {}, None
    try:
        adapter = BasicControlAdapter(load_codec(codec))
        for arm in plan["arms"]:
            arm_started = time()
            path = args.output / f"{arm}.pt"
            saved = torch.load(path, map_location="cpu", weights_only=True) if path.exists() else None
            def checkpoint(value):
                atomic_torch_save(path, value)
                print(json.dumps({"arm": arm, "step": value["step"], **value["trace"][-1]}), flush=True)
            def charge_update():
                check_limits()
                if attempted[arm] >= plan["limits"]["optimizer_updates_per_arm"]:
                    raise RuntimeError("response-fit update-attempt cap exhausted; lost updates are not free retries")
                attempted[arm] += 1
                with update_path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(json.dumps({"arm": arm, "attempt": attempted[arm]}) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            helper, _ = fit_temporal_records(records[arm], adapter, FitConfig(), {"plan_id": canonical_sha256(plan), "arm": arm},
                                            pairs=[(i, i + 1) for i in range(0, 24, 2)], saved=saved, checkpoint=checkpoint, check_limits=charge_update)
            results[arm] = score(helper, records["events"], adapter, identities)
            arm_seconds[arm] = time() - arm_started
            write_json(args.output / f"{arm}-scores.json", results[arm])
        check_limits()
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        monitor.close()
        complete = len(results) == 2 and failure is None
        elapsed = time() - started
        archive_bytes = sum(path.stat().st_size for path in args.output.rglob("*") if path.is_file())
        resources_ok = (monitor.errors == 0 and monitor.peak_bytes is not None and monitor.peak_bytes <= plan["limits"]["process_tree_bytes"]
                        and elapsed <= plan["limits"]["wall_seconds"] and archive_bytes + 1024**2 <= plan["limits"]["archive_bytes"])
        report = summarize(results, plan["gates"]) if complete else {}
        write_json(args.output / "summary.json", {"complete": complete, "failure": failure, "plan_id": canonical_sha256(plan),
                   "result": report, "cases": results, "within_resource_limits": resources_ok,
                   "elapsed_seconds": elapsed, "peak_process_tree_rss_bytes": monitor.peak_bytes, "rss_sampling_errors": monitor.errors,
                   "optimizer_update_attempts": attempted, "archive_bytes_before_summary": archive_bytes,
                   "arm_seconds": arm_seconds,
                   "native_requests": 0, "qualified_for_inverse_training": False, "test_split_used": False,
                   "claim": "training_fit_diagnosis_not_model_qualification"})
    print(json.dumps(report), flush=True)
    if not resources_ok:
        raise RuntimeError("response fit exceeded resource limits")


if __name__ == "__main__":
    main()

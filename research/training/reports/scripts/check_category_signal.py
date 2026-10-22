"""Freeze and execute a bounded observation-only category-gradient diagnosis."""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
from time import time

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_plan(args):
    import torch
    report = json.loads(args.parent_audit.read_text())
    if not all(report[name] for name in ("complete", "offline_training_replay_exact", "cached_targets_and_inputs_rescored", "hard_audit_patches_reconstructed")):
        raise ValueError("parent run has not passed the required evidence audit")
    for name, expected in report["evidence_sha256"].items():
        if digest(args.parent / name) != expected:
            raise ValueError(f"parent evidence changed: {name}")
    paths = [RESEARCH / "training/obruxo_training/category_signal.py", Path(__file__), Path(__file__).with_name("verify_native_inverse_fit.py")]
    return {"format": "basic_category_signal_check_v1", "parent_plan_id": report["plan_id"],
            "parent_audit_sha256": digest(args.parent_audit), "parent_evidence_sha256": report["evidence_sha256"],
            "initial_sha256": digest(args.initial), "retrieval_sha256": digest(args.retrieval),
            "qualification_sha256": digest(args.qualification), "pool_sha256": digest(args.pool),
            "calibration": {"source": "32_training_reference_features_and_initial_weights_only", "rows": 32,
                            "formula": "max(1,RMS(scores-per_row_class_mean))", "accumulation": "float64", "heads": [3, 2]},
            "diagnostic": {"points": 128, "noise": "same_actual_Gumbel_draws", "forward_patch": "held_at_recorded_anchor",
                           "baseline_gradient_rtol": 1e-5, "baseline_gradient_atol": 1e-30,
                           "minimum_median_gradient_gain": 10., "better_alternative_tolerance": 1e-7},
            "source": {path.name: hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for path in paths},
            "gumbel_implementation_sha256": hashlib.sha256(inspect.getsource(torch.nn.functional.gumbel_softmax).encode()).hexdigest(),
            "limits": {"wall_seconds": 300, "process_tree_bytes": 8 * 1024**3, "archive_bytes": 32 * 1024**2},
            "native_requests": 0, "new_fitting_updates": 0, "replayed_optimizer_updates": 256,
            "new_evaluation_data": False, "device": "cpu", "cpu_threads": 2, "precision": "float32"}


def direction_summary(observer):
    import numpy as np
    heads = []
    for index, name in enumerate(("osc_1_unison_voices", "filter_1_on")):
        rows = [point["heads"][index] for point in observer.points]
        original = np.array([row["original_gradient_l1"] for row in rows])
        scaled = np.array([row["calibrated_gradient_l1"] for row in rows])
        responsive = np.array([max(row["conditional_losses"]) - min(row["conditional_losses"]) > 1e-7 for row in rows])
        finite = bool(np.isfinite(original).all() and np.isfinite(scaled).all())
        before, after = float(np.median(original)), float(np.median(scaled))
        zeros = [int(((values == 0) & responsive).sum()) for values in (original, scaled)]
        heads.append({"name": name, "points": len(rows), "fixed_scale": observer.scales[index],
            "original_gradient_median": before, "calibrated_gradient_median": after,
            "median_gain": after / before if before else None,
            "original_responsive_zero_steps": zeros[0], "calibrated_responsive_zero_steps": zeros[1],
            "better_native_alternative_steps": sum(row["better_alternative"] for row in rows),
            "raw_margin_median": float(np.median([row["raw_top_two_margin"] for row in rows])),
            "all_finite": finite, "pass": bool(len(rows) == 128 and finite and after > 0 and after >= 10 * before and zeros[1] <= zeros[0])})
    return heads


def execute(args, plan):
    import torch
    from obruxo_data.hashing import canonical_sha256
    from obruxo_training.audit import write_json
    from obruxo_training.benchmark import MemoryMonitor
    from obruxo_training.category_signal import SignalObserver
    from obruxo_training.codec import load_codec
    from obruxo_training.pilot import PilotPool
    from verify_native_inverse_fit import verify
    torch.set_num_threads(2)
    if json.loads((args.output / "plan.json").read_text()) != plan or (args.output / "run.json").exists():
        raise ValueError("plan changed or diagnostic already attempted")
    started = time()
    write_json(args.output / "run.json", {"plan_id": canonical_sha256(plan), "started_unix": started})
    monitor = MemoryMonitor()
    monitor.start()

    def check_limits():
        if time() - started > plan["limits"]["wall_seconds"] or (monitor.peak_bytes or 0) > plan["limits"]["process_tree_bytes"]:
            raise RuntimeError("category diagnosis resource budget exhausted")

    parent = json.loads(args.parent_audit.read_text())
    observer = SignalObserver(load_codec(parent["plan"]["codec"]), check_limits=check_limits)
    failure, replay_exact = None, False
    try:
        result = verify(args.parent, PilotPool.load(args.pool), args.initial, args.retrieval, args.qualification, observe_step=observer)
        if result != parent:
            raise ValueError("observation changed the published historical replay/audit result")
        replay_exact = True
        check_limits()
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        monitor.close()
        elapsed = time() - started
        heads = direction_summary(observer) if observer.points else []
        resources = (elapsed <= plan["limits"]["wall_seconds"] and monitor.errors == 0
                     and 0 < (monitor.peak_bytes or 0) <= plan["limits"]["process_tree_bytes"])
        summary = {"plan": plan, "plan_id": canonical_sha256(plan), "failure": failure,
            "fixed_scales": observer.scales, "initial_training_inputs": observer.initial,
            "points": observer.points, "heads": heads, "historical_replay_unchanged": replay_exact,
            "actual_replayed_optimizer_updates": 2 * len(observer.points),
            "elapsed_seconds": elapsed, "peak_process_tree_rss_bytes": monitor.peak_bytes,
            "rss_sampling_errors": monitor.errors, "within_resource_limits": resources,
            "passed": bool(replay_exact and len(observer.initial) == 32 and heads and all(head["pass"] for head in heads) and resources),
            "claim": "fixed_point_derivative_diagnosis_not_new_sound_quality", "model_promoted": False}
        archive = sum(path.stat().st_size for path in args.output.rglob("*") if path.is_file())
        estimated = archive + len(json.dumps(summary, indent=2, sort_keys=True).encode()) + 1024
        summary["estimated_archive_bytes_with_margin"] = estimated
        if estimated > plan["limits"]["archive_bytes"]:
            summary.update(within_resource_limits=False, passed=False, failure="category diagnosis archive budget exceeded")
        write_json(args.output / "summary.json", summary)
    return {"passed": summary["passed"], "fixed_scales": observer.scales, "heads": heads}


def main():
    from obruxo_data.hashing import canonical_sha256
    from obruxo_training.audit import write_json
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "parent-audit", "pool", "initial", "retrieval", "qualification", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = make_plan(args)
    if not args.execute:
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output / "plan.json", plan)
        print(json.dumps({"plan_id": canonical_sha256(plan), "native_requests": 0, "replayed_optimizer_updates": 256}))
        return
    print(json.dumps(execute(args, plan)))


if __name__ == "__main__":
    main()

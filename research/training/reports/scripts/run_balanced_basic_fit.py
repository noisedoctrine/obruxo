"""Prepare, check, execute or audit the balanced three-sound comparison."""

import argparse
import json
import sys
from pathlib import Path

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


def main():
    from obruxo_training.balanced_basic_audit import software_check, verify
    from obruxo_training.balanced_basic_fit import (
        DEFAULT_CPU_THREADS,
        execute,
        load_inputs,
        prepare,
    )
    from obruxo_training.batching import BatchConfig
    from obruxo_training.category_scale_refresh import cpu_settings
    from obruxo_training.local_execution import active_computation
    from obruxo_training.pilot import PilotPool
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "bridge", "features", "control", "data", "pool", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--mode", choices=("prepare", "check", "execute", "resume", "verify"), default="prepare")
    parser.add_argument("--model-batch-size", type=int)
    parser.add_argument("--spectral-batch-size", type=int)
    parser.add_argument("--pitch-batch-size", type=int)
    parser.add_argument("--cpu-threads", type=int, help="deterministic CPU thread count; frozen in the prepared plan")
    args = parser.parse_args()
    overrides = {name: getattr(args, name + "_batch_size") for name in ("model", "spectral", "pitch") if getattr(args, name + "_batch_size") is not None}
    paths = {key: getattr(args, key) for key in ("parent", "bridge", "features", "control", "data")}
    if args.mode == "prepare":
        cpu_threads = DEFAULT_CPU_THREADS if args.cpu_threads is None else args.cpu_threads
    else:
        saved_plan = json.loads((args.output / "plan.json").read_text())
        cpu_threads = saved_plan.get("cpu_threads", 2)
        if args.cpu_threads is not None and args.cpu_threads != cpu_threads:
            raise ValueError("CPU thread count is frozen in the plan; prepare a new run to change it")
    with active_computation(), cpu_settings(cpu_threads):
        pool = PilotPool.load(args.pool)
        if args.mode == "prepare":
            plan = prepare(paths, pool, args.output, batching=BatchConfig(**overrides), cpu_threads=cpu_threads)
            result = {"updates": len(plan["updates"]), "evaluation_cells": len(plan["evaluation_indices"]),
                      "cpu_threads": cpu_threads, "maximum_native_requests": 8000}
        else:
            inputs = load_inputs(args.output, pool)
            if any(BatchConfig.from_plan(inputs.plan).to_dict()[name] != value for name, value in overrides.items()):
                raise ValueError("execution batch sizes are frozen in the plan; prepare a new run to change them")
            if args.mode == "check":
                result = software_check(inputs, args.output)
            elif args.mode == "verify":
                result = verify(inputs, args.output)
            else:
                result = execute(inputs, args.output, resume=args.mode == "resume")
    print(json.dumps({key: value for key, value in result.items() if key not in ("evidence_sha256", "result")}), flush=True)


if __name__ == "__main__":
    main()

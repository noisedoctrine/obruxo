"""Prepare, check, execute or audit the balanced three-sound comparison."""

import argparse
import json
import sys
from pathlib import Path

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


def main():
    from obruxo_training.balanced_basic_audit import software_check, verify
    from obruxo_training.balanced_basic_fit import execute, load_inputs, prepare
    from obruxo_training.category_scale_refresh import cpu_settings
    from obruxo_training.local_execution import active_computation
    from obruxo_training.pilot import PilotPool
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "bridge", "features", "control", "data", "pool", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--mode", choices=("prepare", "check", "execute", "resume", "verify"), default="prepare")
    args = parser.parse_args()
    paths = {key: getattr(args, key) for key in ("parent", "bridge", "features", "control", "data")}
    with active_computation(), cpu_settings():
        pool = PilotPool.load(args.pool)
        if args.mode == "prepare":
            plan = prepare(paths, pool, args.output)
            result = {"updates": len(plan["updates"]), "evaluation_cells": len(plan["evaluation_indices"]), "maximum_native_requests": 8000}
        else:
            inputs = load_inputs(args.output, pool)
            if args.mode == "check":
                result = software_check(inputs, args.output)
            elif args.mode == "verify":
                result = verify(inputs, args.output)
            else:
                result = execute(inputs, args.output, resume=args.mode == "resume")
    print(json.dumps({key: value for key, value in result.items() if key not in ("evidence_sha256", "result")}), flush=True)


if __name__ == "__main__":
    main()

"""Run the shared three-sound reconstruction comparison, locally."""

import argparse
import json
import sys
from pathlib import Path

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


def main():
    from obruxo_training.category_scale_refresh import cpu_settings
    from obruxo_training.local_execution import active_computation
    from obruxo_training.pilot import PilotPool
    from obruxo_training.shared_auxiliary_run import (
        execute,
        load_inputs,
        prepare,
        software_check,
        verify,
    )
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "bridge", "features", "pool", "fit", "auxiliary", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--mode", choices=("prepare", "check", "execute", "resume", "verify"), default="prepare")
    args = parser.parse_args()
    with active_computation(), cpu_settings():
        pool = PilotPool.load(args.pool)
        if args.mode == "prepare":
            plan = prepare(args.parent, args.bridge, args.features, pool, args.fit, args.auxiliary, args.output)
            result = {"updates_per_arm": len(plan["updates"]), "maximum_native_requests": plan["maximum_native_requests"]}
        else:
            inputs, rows, all_rows = load_inputs(args.parent, args.bridge, args.features, pool, args.fit, args.auxiliary, args.output)
            if args.mode == "check":
                result = software_check(inputs, rows, args.output)
            elif args.mode == "verify":
                result = verify(inputs, rows, all_rows, args.output)
            else:
                result = execute(inputs, rows, args.output, resume=args.mode == "resume")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

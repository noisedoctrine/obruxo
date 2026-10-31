"""Prepare, render or audit the small balanced training-performance bank."""

import argparse
import json
import sys
from pathlib import Path

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


def main():
    from obruxo_training.audit import write_json
    from obruxo_training.balanced_basic_data import (
        collect,
        make_plan,
        validate_plan,
        verify,
    )
    from obruxo_training.category_scale_refresh import cpu_settings
    from obruxo_training.conditioned_bridge import read
    from obruxo_training.local_execution import active_computation
    from obruxo_training.pilot import PilotPool
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fit", "bridge", "pool", "reserved", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--mode", choices=("prepare", "collect", "resume", "verify"), default="prepare")
    args = parser.parse_args()
    with active_computation(), cpu_settings():
        pool = PilotPool.load(args.pool)
        if args.mode == "prepare":
            plan = make_plan(args.fit, args.bridge, pool, args.reserved)
            args.output.mkdir(parents=True, exist_ok=False)
            write_json(args.output / "plan.json", plan)
            result = {"source_recordings": 18, "planned_native_requests": 24}
        else:
            plan = read(args.output / "plan.json")
            validate_plan(plan, pool)
            result = verify(plan, pool, args.bridge, args.output) if args.mode == "verify" else collect(plan, pool, args.bridge, args.output, resume=args.mode == "resume")
    print(json.dumps({key: value for key, value in result.items() if key != "evidence_sha256"}), flush=True)


if __name__ == "__main__":
    main()

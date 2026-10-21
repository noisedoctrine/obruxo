"""Freeze or run the local varied-input initialization comparison."""

import argparse
import json
from pathlib import Path
import sys

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


if __name__ == "__main__":
    from obruxo_training.pilot import PilotPool
    from obruxo_training.varied_bootstrap import make_plan, run
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pool", "qualification", "initial-root", "initial-audit", "input-root", "input-audit", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.resume and not args.execute:
        parser.error("resume requires execute")
    pool = PilotPool.load(args.pool)
    plan = make_plan(pool, args.qualification, args.initial_root, args.initial_audit, args.input_root, args.input_audit)
    result = run(pool, plan, args.initial_root, args.output, execute=args.execute, resume=args.resume)
    print(json.dumps({key: value for key, value in result.items() if key not in {"training", "audits"}}), flush=True)

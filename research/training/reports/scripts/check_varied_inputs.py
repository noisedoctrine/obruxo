"""Freeze or run a bounded native check of varied basic training inputs."""

import argparse
import json
from pathlib import Path
import sys

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


if __name__ == "__main__":
    from obruxo_training.pilot import PilotPool
    from obruxo_training.varied_input_check import make_plan, run_check
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pool", "qualification", "bootstrap-root", "bootstrap-audit", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.resume and not args.execute:
        parser.error("resume requires execute")
    plan = make_plan(PilotPool.load(args.pool), args.qualification, args.bootstrap_root, args.bootstrap_audit)
    print(json.dumps(run_check(plan, args.output, execute=args.execute, resume=args.resume)), flush=True)

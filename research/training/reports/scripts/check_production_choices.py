"""Freeze, execute or independently reconstruct the zero-render production-choice check."""

import argparse
import json
from pathlib import Path
import sys

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


if __name__ == "__main__":
    from obruxo_training.pilot import PilotPool
    from obruxo_training.production_choice_check import run_check, run_validation
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pool", "bootstrap-root", "bootstrap-audit", "qualification", "input-cache", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    inputs = (PilotPool.load(args.pool), args.bootstrap_root, args.bootstrap_audit, args.qualification, args.input_cache)
    result = (run_validation(args.output, *inputs) if args.verify else
              run_check(*inputs, args.output, execute=args.execute))
    print(json.dumps(result), flush=True)
    if args.execute and not result["passed"]:
        raise SystemExit(1)

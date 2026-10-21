"""Freeze or run the corrected-render 404-state initialization refresh."""

import argparse
import json
from pathlib import Path
import sys

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


if __name__ == "__main__":
    from obruxo_training.basic_bootstrap_full import run_full
    from obruxo_training.pilot import PilotPool
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pool", "qualification", "preflight-root", "preflight-audit", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.resume and not args.execute:
        parser.error("resume requires execute")
    result = run_full(PilotPool.load(args.pool), args.qualification, args.preflight_root, args.preflight_audit, args.output,
                      execute=args.execute, resume=args.resume)
    print(json.dumps({key: value for key, value in result.items() if key not in {"training", "training_audit"}}), flush=True)

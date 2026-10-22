"""Freeze or run a zero-render category check after the full basic initialization passes."""

import argparse
import json
from pathlib import Path
import sys

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


if __name__ == "__main__":
    from obruxo_training.category_scale_refresh import run_check
    from obruxo_training.pilot import PilotPool
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pool", "bootstrap-root", "bootstrap-audit", "qualification", "input-cache", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_check(PilotPool.load(args.pool), args.bootstrap_root, args.bootstrap_audit,
                               args.qualification, args.input_cache, args.output, execute=args.execute)), flush=True)

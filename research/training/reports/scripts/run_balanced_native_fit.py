"""Freeze or run the matched original/wider-input native sound fit."""

import argparse
import json
from pathlib import Path
import sys

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


if __name__ == "__main__":
    from obruxo_training.balanced_native_fit import run_fit
    from obruxo_training.pilot import PilotPool
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pool", "initial", "retrieval", "qualification", "input-cache", "category-calibration", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.resume and not args.execute:
        parser.error("--resume requires --execute")
    result = run_fit(PilotPool.load(args.pool), args.initial, args.retrieval, args.qualification, args.input_cache,
                     args.category_calibration, args.output, execute=args.execute, resume=args.resume)
    print(json.dumps(result), flush=True)

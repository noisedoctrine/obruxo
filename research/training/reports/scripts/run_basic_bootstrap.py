"""Freeze or run the eight-source corrected-render initialization preflight."""

import argparse
import json
from pathlib import Path
import sys

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


def main():
    from obruxo_training.basic_bootstrap import run_preflight
    from obruxo_training.pilot import PilotPool
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.resume and not args.execute:
        parser.error("resume requires execute")
    result = run_preflight(PilotPool.load(args.pool), args.qualification, args.output, execute=args.execute, resume=args.resume)
    print(json.dumps({key: value for key, value in result.items() if key not in {"training", "training_audit"}}), flush=True)


if __name__ == "__main__":
    main()

"""Freeze or execute the costed matched native spectral inverse-training trial."""

import argparse
import json
from pathlib import Path
import sys

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


def main():
    from obruxo_training.native_fit import run_fit
    from obruxo_training.pilot import PilotPool
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.resume and not args.execute:
        parser.error("--resume requires --execute")
    result = run_fit(PilotPool.load(args.pool), args.initial, args.retrieval, args.qualification, args.output,
                     execute=args.execute, resume=args.resume)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

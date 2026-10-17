"""Qualify combined sources, all registered assets and newly activated families."""
import argparse
import json
from pathlib import Path
import sys

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


def main():
    from obruxo_training.combined_check import check_combined
    from obruxo_training.pilot import PilotPool
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check_combined(PilotPool.load(args.pool), args.output, progress=lambda row: print(json.dumps(row), flush=True))
    print(json.dumps({"passed": result["passed"], "gates": result["gates"]}), flush=True)
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

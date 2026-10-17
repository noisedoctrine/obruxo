"""Measure native response for every registered connection destination or an explicit subset."""
import argparse
import json
from pathlib import Path
import sys

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


def main():
    from obruxo_training.graph_native import check_destinations
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--destinations", nargs="+")
    parser.add_argument("--overrides", type=Path)
    args = parser.parse_args()
    result = check_destinations(json.loads(args.asset_bank.read_text()), args.output, destinations=args.destinations,
        overrides=json.loads(args.overrides.read_text()) if args.overrides else None, progress=lambda row: print(json.dumps(row), flush=True))
    print(json.dumps({"passed": result["passed"], "gates": result["gates"], "responses_observed": result["responses_observed"],
                      "unresolved_destinations": result["unresolved_destinations"]}), flush=True)
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Run bounded native acceptance checks for all declared routing vocabulary."""
import argparse
import json
from pathlib import Path
import sys

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


def main():
    from obruxo_training.routing_check import check_routing
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe", action="store_true", help="Check empty mapping and feedback edge cases before the full matrix")
    args = parser.parse_args()
    result = check_routing(args.output, probe=args.probe, progress=lambda event: print(json.dumps(event), flush=True))
    print(json.dumps({"passed": result["passed"], "gates": result["gates"]}), flush=True)
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

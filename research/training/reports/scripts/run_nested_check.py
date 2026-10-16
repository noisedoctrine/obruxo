"""Qualify custom curve effects, seed precision and numbered-slot isolation."""
import argparse
import json
from pathlib import Path
import sys

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


def main():
    from obruxo_training.nested_check import check_nested
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check_nested(json.loads(args.asset_bank.read_text()), args.output, progress=lambda row: print(json.dumps(row), flush=True))
    print(json.dumps({"passed": result["passed"], "gates": result["gates"]}), flush=True)
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

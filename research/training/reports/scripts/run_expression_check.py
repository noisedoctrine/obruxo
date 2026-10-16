"""Run the bounded native expressive-MIDI compatibility and timing qualification."""
import argparse
import json
from pathlib import Path
import sys

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


def main():
    from obruxo_training.expression_check import check_expression
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check_expression(args.output, progress=lambda row: print(json.dumps(row), flush=True))
    print(json.dumps({"passed": result["passed"], "gates": result["gates"]}), flush=True)
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

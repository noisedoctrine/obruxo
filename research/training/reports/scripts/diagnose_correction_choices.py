"""Inspect the verified correction fit without rendering or training."""

import argparse
import json
from pathlib import Path
import sys

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]

if __name__ == "__main__":
    from obruxo_training.correction_choice_diagnosis import run
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "initial", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.initial, args.output, execute=args.execute)), flush=True)

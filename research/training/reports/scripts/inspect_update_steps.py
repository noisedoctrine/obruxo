"""Freeze or execute the saved sound-only and auxiliary update diagnosis."""

import argparse
import json
from pathlib import Path
import sys

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]

if __name__ == "__main__":
    from obruxo_training.update_diagnosis import run
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, nargs=2, required=True)
    for name in ("parent", "bridge", "features", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.sources, args.parent, args.bridge, args.features, args.output, execute=args.execute)), flush=True)

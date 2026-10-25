"""Freeze or run a no-render inspection of the audited compact sound fit."""

import argparse
import json
from pathlib import Path
import sys

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]

if __name__ == "__main__":
    from obruxo_training.direct_fit_gaps import run
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "parent", "bridge", "features", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.parent, args.bridge, args.features, args.output, execute=args.execute)), flush=True)

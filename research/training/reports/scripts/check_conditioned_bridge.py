"""Freeze or qualify the zero-output basic conditioning bridge without rendering."""

import argparse
import json
from pathlib import Path
import sys

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]

if __name__ == "__main__":
    from obruxo_training.conditioned_bridge import run_bridge
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "initial", "features", "direction", "pool", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_bridge(args.parent, args.initial, args.features, args.direction, args.pool, args.output, execute=args.execute)), flush=True)

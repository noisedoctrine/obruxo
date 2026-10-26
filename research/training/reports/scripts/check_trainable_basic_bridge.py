"""Qualify trainable basic audio features at unchanged initial native controls."""

import argparse
import json
from pathlib import Path
import sys

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]

if __name__ == "__main__":
    from obruxo_training.local_execution import active_computation
    from obruxo_training.trainable_basic_bridge import run
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "initial", "features", "direction", "pool", "recent", "baseline-parent", "failed-fit", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    with active_computation():
        print(json.dumps(run(args.parent, args.initial, args.features, args.direction, args.pool, args.recent,
                             args.baseline_parent, args.failed_fit, args.output, execute=args.execute)), flush=True)

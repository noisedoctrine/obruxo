"""Freeze, execute, recover or independently verify the small local correction fit."""

import argparse
import json
from pathlib import Path
import sys

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]

if __name__ == "__main__":
    from obruxo_training.conditioned_short_fit import run_short
    from obruxo_training.local_execution import active_computation
    from obruxo_training.pilot import PilotPool
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pool", "parent", "bridge", "features", "retrieval", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("mode", choices=("prepare", "execute", "resume", "verify"))
    args = parser.parse_args()
    with active_computation():
        print(json.dumps(run_short(PilotPool.load(args.pool), args.parent, args.bridge, args.features, args.retrieval, args.output, mode=args.mode)), flush=True)

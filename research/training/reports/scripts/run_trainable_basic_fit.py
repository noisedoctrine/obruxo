"""Run matched native sound training with trainable basic audio features."""

import argparse
import json
from pathlib import Path
import sys

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]

if __name__ == "__main__":
    from obruxo_training.local_execution import active_computation
    from obruxo_training.pilot import PilotPool
    from obruxo_training.trainable_basic_fit import run
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "bridge", "features", "pool", "prior", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--mode", choices=("prepare", "execute", "resume", "verify"), default="prepare")
    args = parser.parse_args()
    with active_computation():
        print(json.dumps(run(args.parent, args.bridge, args.features, PilotPool.load(args.pool), args.prior, args.output, mode=args.mode)), flush=True)

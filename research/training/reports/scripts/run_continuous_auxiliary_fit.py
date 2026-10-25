"""Prepare, execute, recover or independently audit the continuous-auxiliary fit."""

import argparse
import json
from pathlib import Path
import sys

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]

if __name__ == "__main__":
    from obruxo_training.continuous_auxiliary_fit import run
    from obruxo_training.local_execution import active_computation
    from obruxo_training.pilot import PilotPool
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "bridge", "recent", "features", "pool", "prior-fit", "diagnosis", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--mode", choices=("prepare", "execute", "resume", "verify"), default="prepare")
    args = parser.parse_args()
    with active_computation():
        print(json.dumps(run(args.parent, args.bridge, args.recent, args.features, PilotPool.load(args.pool), args.prior_fit,
                             args.diagnosis, args.output, mode=args.mode)), flush=True)

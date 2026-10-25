"""Prepare, render or independently audit protected-update Vital comparisons."""

import argparse
import json
from pathlib import Path
import sys

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]

if __name__ == "__main__":
    from obruxo_training.local_execution import active_computation
    from obruxo_training.protected_native_check import run
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs=2, type=Path, required=True)
    for name in ("parent", "bridge", "features", "diagnosis", "screen", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--mode", choices=("prepare", "execute", "verify"), default="prepare")
    parser.add_argument("--continue-from", type=Path)
    args = parser.parse_args()
    with active_computation():
        print(json.dumps(run(args.sources, args.parent, args.bridge, args.features, args.diagnosis, args.screen, args.output, mode=args.mode, continue_from=args.continue_from)), flush=True)

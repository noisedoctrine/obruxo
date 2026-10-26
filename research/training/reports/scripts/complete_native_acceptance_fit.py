"""Complete final comparisons from a terminal fully trained checkpoint."""

import argparse
import json
from pathlib import Path
import sys

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]

if __name__ == "__main__":
    from obruxo_training.final_fit_completion import run
    from obruxo_training.local_execution import active_computation
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "parent", "bridge", "features", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--mode", choices=("prepare", "execute", "verify"), default="prepare")
    args = parser.parse_args()
    with active_computation():
        print(json.dumps(run(args.source, args.parent, args.bridge, args.features, args.output, mode=args.mode)), flush=True)

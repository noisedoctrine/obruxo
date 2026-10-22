"""Freeze or run the zero-source-render combined-input initialization fit."""

import argparse
import json
from pathlib import Path
import sys

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


if __name__ == "__main__":
    import torch
    from obruxo_training.mixed_bootstrap import run
    from obruxo_training.pilot import PilotPool
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pool", "qualification", "cache-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.resume and not args.execute:
        parser.error("resume requires execute")
    torch.set_num_threads(2)
    result = run(PilotPool.load(args.pool), args.qualification, args.cache_root, args.output, execute=args.execute, resume=args.resume)
    print(json.dumps({key: value for key, value in result.items() if key not in {"training", "audits"}}), flush=True)

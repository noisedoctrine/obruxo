"""Combine verified original and wider-input feature banks without new rendering."""

import argparse
import json
from pathlib import Path
import sys

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


if __name__ == "__main__":
    import torch
    from obruxo_training.mixed_features import prepare_cache
    from obruxo_training.pilot import PilotPool
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pool", "original-root", "original-audit", "wide-root", "wide-audit", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = prepare_cache(PilotPool.load(args.pool), args.original_root, args.original_audit, args.wide_root, args.wide_audit, args.output)
    print(json.dumps(result), flush=True)

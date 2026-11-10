"""Prepare, execute, explicitly recover or verify the compact direct/rehearsal fit."""

import argparse
import json
from pathlib import Path
import sys

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]

if __name__ == "__main__":
    from obruxo_training.basic_direct_fit import run
    from obruxo_training.local_execution import active_computation
    from obruxo_training.pilot import PilotPool
    from obruxo_training.batching import BatchConfig
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "bridge", "recent", "features", "pool", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--mode", choices=("prepare", "execute", "resume", "verify"), default="prepare")
    parser.add_argument("--model-batch-size", type=int)
    parser.add_argument("--spectral-batch-size", type=int)
    parser.add_argument("--pitch-batch-size", type=int)
    args = parser.parse_args()
    saved = {} if args.mode == "prepare" else json.loads((args.output / "plan.json").read_text(encoding="utf-8")).get("batching", {})
    overrides = {name: getattr(args, name + "_batch_size") for name in ("model", "spectral", "pitch") if getattr(args, name + "_batch_size") is not None}
    if args.mode != "prepare" and any(saved.get(name, BatchConfig().to_dict()[name]) != value for name, value in overrides.items()):
        raise ValueError("execution batch sizes are frozen in the plan; prepare a new run to change them")
    with active_computation():
        print(json.dumps(run(args.parent, args.bridge, args.recent, args.features, PilotPool.load(args.pool), args.output, mode=args.mode,
                             batching=BatchConfig(**{**saved, **overrides}))), flush=True)

"""Freeze, execute, resume or verify the original/wider paired native sound fit."""

import argparse
import json
from pathlib import Path
import sys

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


if __name__ == "__main__":
    from obruxo_training.paired_native_audit import run_verification
    from obruxo_training.paired_native_fit import run_fit
    from obruxo_training.pilot import PilotPool
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pool", "bootstrap-root", "bootstrap-audit", "qualification", "input-cache", "choice-proof", "direction-source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--verify", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--input-and-transfer", action="store_true", help="Score every prediction under its input and reciprocal performances")
    args = parser.parse_args()
    if args.resume and not args.execute:
        parser.error("--resume requires --execute")
    inputs = (PilotPool.load(args.pool), args.bootstrap_root, args.bootstrap_audit, args.qualification, args.input_cache, args.choice_proof, args.direction_source)
    result = (run_verification(args.output, *inputs) if args.verify else run_fit(*inputs, args.output, execute=args.execute, resume=args.resume,
                                                                           input_and_transfer=args.input_and_transfer))
    print(json.dumps(result), flush=True)

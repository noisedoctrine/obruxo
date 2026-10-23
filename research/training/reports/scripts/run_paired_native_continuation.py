"""Audit, declare, resume or verify the bounded local paired-fit continuation."""

import argparse
import json
from pathlib import Path
import sys

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


if __name__ == "__main__":
    from obruxo_training.paired_interrupted_audit import audit_interrupted
    from obruxo_training.local_execution import active_computation
    from obruxo_training.paired_native_continuation import freeze_continuation, run_continuation, verify_continuation
    from obruxo_training.pilot import PilotPool
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("audit-interrupted", "declare", "resume", "verify"))
    for name in ("pool", "bootstrap-root", "bootstrap-audit", "qualification", "input-cache", "choice-proof", "direction-source", "source", "declaration", "prefix-audit", "scratch"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    pool = PilotPool.load(args.pool)
    inputs = (args.bootstrap_root, args.bootstrap_audit, args.qualification, args.input_cache, args.choice_proof, args.direction_source)
    if args.mode == "audit-interrupted":
        with active_computation():
            result = audit_interrupted(args.source, args.prefix_audit, args.scratch, pool, *inputs)
    elif args.mode == "declare":
        result = freeze_continuation(args.source, args.declaration, args.prefix_audit)
    else:
        operation = run_continuation if args.mode == "resume" else verify_continuation
        result = operation(args.source, args.declaration, pool, *inputs)
    print(json.dumps(result), flush=True)

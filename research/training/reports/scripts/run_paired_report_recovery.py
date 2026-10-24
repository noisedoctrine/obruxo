"""Freeze, check, finish or verify saved final comparisons without more training."""

import argparse
import json
from pathlib import Path
import sys

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


if __name__ == "__main__":
    from obruxo_training.paired_report_recovery import check_cached_fit, freeze_report_recovery, run_saved_comparisons, verify_report_recovery
    from obruxo_training.pilot import PilotPool
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "check", "resume", "verify"))
    for name in ("source", "declaration", "scratch", "pool", "bootstrap-root", "bootstrap-audit", "qualification", "input-cache", "choice-proof", "direction-source"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "freeze":
        result = freeze_report_recovery(args.source, args.declaration)
    elif args.mode == "check":
        result = check_cached_fit(args.source, args.declaration, args.bootstrap_root)
    else:
        inputs = (PilotPool.load(args.pool), args.bootstrap_root, args.bootstrap_audit, args.qualification, args.input_cache, args.choice_proof, args.direction_source)
        result = (run_saved_comparisons(args.source, args.declaration, *inputs) if args.mode == "resume" else
                  verify_report_recovery(args.source, args.declaration, args.scratch, *inputs))
    print(json.dumps(result), flush=True)

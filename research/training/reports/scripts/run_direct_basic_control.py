"""Prepare, run or independently replay the bounded three-sound component check."""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


def main():
    from obruxo_data.hashing import canonical_sha256
    from obruxo_training.audit import write_json
    from obruxo_training.benchmark import MemoryMonitor
    from obruxo_training.category_scale_refresh import cpu_settings
    from obruxo_training.conditioned_bridge import read
    from obruxo_training.direct_basic_audit import verify
    from obruxo_training.direct_basic_fit import execute
    from obruxo_training.direct_basic_plan import make_plan, validate
    from obruxo_training.local_execution import active_computation
    from obruxo_training.pilot import PilotPool
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fit", "bridge", "features", "pool", "feedback", "software", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--mode", choices=("prepare", "execute", "resume", "verify"), default="prepare")
    args = parser.parse_args()
    with active_computation(), cpu_settings():
        if args.mode == "prepare":
            started, monitor = perf_counter(), MemoryMonitor()
            monitor.start()
            try:
                plan = make_plan(args.fit, args.bridge, args.features, PilotPool.load(args.pool), args.feedback, args.software)
            finally:
                monitor.close()
            resources = {"elapsed_seconds": perf_counter() - started, "peak_process_tree_rss_bytes": monitor.peak_bytes, "rss_sampling_errors": monitor.errors}
            if resources["elapsed_seconds"] > 600 or not monitor.peak_bytes or monitor.peak_bytes > 4 * 1024**3 or monitor.errors:
                raise ValueError("direct preparation exceeded ten minutes or 4 GiB")
            args.output.mkdir(parents=True, exist_ok=False)
            write_json(args.output / "plan.json", plan)
            write_json(args.output / "preparation.json", {**resources, "native_requests": 0, "experimental_updates": 0})
            result = {"plan_id": canonical_sha256(plan), "preparation": resources, "maximum_native_requests": plan["maximum_native_requests"]}
        else:
            plan = read(args.output / "plan.json")
            validate(plan, args.fit, args.bridge, args.features, args.feedback, args.software)
            if PilotPool.load(args.pool).pool_id != plan["pool_id"]:
                raise ValueError("frozen direct training pool changed")
            result = verify(plan, args.fit, args.output) if args.mode == "verify" else execute(plan, args.fit, args.output, resume=args.mode == "resume")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

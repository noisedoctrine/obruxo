"""Reserve new basic confirmation families; never render or score a candidate."""

import argparse
from pathlib import Path
import subprocess
import sys
from time import perf_counter

REPOSITORY = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(REPOSITORY / "research/training"), str(REPOSITORY / "research/data_generation")]


def main():
    from obruxo_training.audit import write_json
    from obruxo_training.benchmark import MemoryMonitor
    from obruxo_training.confirmation_reservation import reserve
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pool", "development-fit", "pending-fit", "features", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    implementation = subprocess.check_output(["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"], text=True).strip()
    monitor, started = MemoryMonitor(), perf_counter()
    monitor.start()
    try:
        summary = reserve(args.pool, args.development_fit, args.pending_fit, args.features, args.output, implementation_commit=implementation)
    finally:
        monitor.close()
    elapsed = perf_counter() - started
    within = bool(elapsed < 600 and monitor.peak_bytes and monitor.peak_bytes < 4 * 1024**3 and not monitor.errors)
    summary.update(elapsed_seconds=elapsed, peak_process_tree_rss_bytes=monitor.peak_bytes, rss_sampling_errors=monitor.errors, within_resource_limits=within)
    write_json(args.output / "summary.json", summary)
    print(summary)
    return 0 if within else 1


if __name__ == "__main__":
    raise SystemExit(main())

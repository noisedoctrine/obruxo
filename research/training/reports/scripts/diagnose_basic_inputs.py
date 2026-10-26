"""Inspect verified saved predictions and prepare anonymous native-audio comparisons."""

import argparse
from pathlib import Path
import subprocess
import sys
from time import perf_counter

REPOSITORY = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(REPOSITORY / "research/training"), str(REPOSITORY / "research/data_generation")]


def main():
    from obruxo_training.audit import write_json
    from obruxo_training.basic_input_diagnosis import prepare
    from obruxo_training.benchmark import MemoryMonitor
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    implementation = subprocess.check_output(["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"], text=True).strip()
    monitor, started = MemoryMonitor(), perf_counter()
    monitor.start()
    try:
        result = prepare(args.fit, args.output, implementation_commit=implementation)
    finally:
        monitor.close()
    elapsed = perf_counter() - started
    size = sum(path.stat().st_size for path in args.output.rglob("*") if path.is_file())
    within = bool(elapsed < 300 and monitor.peak_bytes and monitor.peak_bytes < 4 * 1024**3 and not monitor.errors and size < 128 * 1024**2)
    result.update(elapsed_seconds=elapsed, peak_process_tree_rss_bytes=monitor.peak_bytes, rss_sampling_errors=monitor.errors,
                  archive_bytes=size, within_resource_limits=within)
    write_json(args.output / "summary.json", result)
    print(result)
    return 0 if within else 1


if __name__ == "__main__":
    raise SystemExit(main())

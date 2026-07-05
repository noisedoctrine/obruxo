from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

_RUNTIME_ENV = {}


def _pop_runtime_env_args(argv: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    value_flags = {
        "--mkl-threading-layer": "MKL_THREADING_LAYER",
        "--openblas-threads": "OPENBLAS_NUM_THREADS",
        "--omp-threads": "OMP_NUM_THREADS",
        "--mkl-threads": "MKL_NUM_THREADS",
    }
    index = 1
    while index < len(argv):
        arg = argv[index]
        if arg == "--native-threads":
            value = _require_flag_value(argv, index)
            env.update({"OPENBLAS_NUM_THREADS": value, "OMP_NUM_THREADS": value, "MKL_NUM_THREADS": value})
            del argv[index : index + 2]
            continue
        if arg.startswith("--native-threads="):
            value = arg.split("=", 1)[1]
            env.update({"OPENBLAS_NUM_THREADS": value, "OMP_NUM_THREADS": value, "MKL_NUM_THREADS": value})
            del argv[index]
            continue
        matched = False
        for flag, name in value_flags.items():
            if arg == flag:
                env[name] = _require_flag_value(argv, index)
                del argv[index : index + 2]
                matched = True
                break
            if arg.startswith(f"{flag}="):
                env[name] = arg.split("=", 1)[1]
                del argv[index]
                matched = True
                break
        if not matched:
            index += 1
    return env


def _require_flag_value(argv: list[str], index: int) -> str:
    if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
        raise SystemExit(f"{argv[index]} requires a value")
    return argv[index + 1]


_RUNTIME_ENV = _pop_runtime_env_args(sys.argv)
os.environ.update(_RUNTIME_ENV)

from lfo_era2.deviation_audit import DEFAULT_OUTPUT_DIR, run_deviation_audit  # noqa: E402
from lfo_era2.runner import DEFAULT_METADATA  # noqa: E402


def main() -> None:
    args = _parser().parse_args()
    row_ids = set(args.rows.split(",")) if args.rows else None
    result = run_deviation_audit(
        output_dir=args.output_dir,
        metadata_path=args.metadata,
        backend=args.backend,
        corpus_sample_fraction=args.corpus_sample_fraction,
        row_ids=row_ids,
        max_utility_candidates=args.max_utility_candidates,
        chunk_size=args.chunk_size,
        write_report=not args.no_report,
        progress=lambda message: print(message, flush=True),
    )
    print("Wrote Experiment 11 W8D16 deviation audit", flush=True)
    for key, value in result.items():
        print(f"{key}={value}", flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Experiment 11 W8D16 deviation audit.")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--backend", choices=["auto", "numpy", "xpu"], default="auto")
    parser.add_argument("--corpus-sample-fraction", type=float, default=1.0)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--max-utility-candidates", type=int, default=None)
    parser.add_argument("--no-report", action="store_true", help="Write CSV artifacts only; do not update canonical report or plots.")
    parser.add_argument(
        "--rows",
        default="",
        help="Optional comma-separated row ids for a partial diagnostic run.",
    )
    return parser


if __name__ == "__main__":
    main()

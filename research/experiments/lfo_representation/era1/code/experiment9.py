#!/usr/bin/env python3
"""Dedicated entrypoint for Experiment 9."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACTS = EXPERIMENT_DIR / "artifacts"


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("command", choices=("run", "status", "analysis"))
    root.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    root.add_argument("--beam-width", type=int, default=4)
    root.add_argument("--batch-size", type=int)
    root.add_argument("--train-stage-batch-size", type=int)
    root.add_argument("--align-batch-size", type=int)
    root.add_argument("--cache-every", type=int)
    root.add_argument("--max-shapes", type=int)
    root.add_argument("--seed", type=int, default=7267)
    root.add_argument("--align-device", choices=("auto", "cpu", "xpu"), default="xpu")
    return root


def _environment(artifacts: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.setdefault("MPLCONFIGDIR", str(artifacts / "mpl"))
    environment.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
    return environment


def main() -> None:
    args = parser().parse_args()
    catalog_path = args.artifacts / "lfo_catalog.csv"
    codebook_path = args.artifacts / "stock_codebook.json"
    output_dir = args.artifacts / "additive_finalization_9_screen"

    if args.command == "run":
        if not catalog_path.exists():
            raise SystemExit(f"Missing {catalog_path}; run catalog generation first")
        if not codebook_path.exists():
            raise SystemExit(f"Missing {codebook_path}; run codebook generation first")
        command = [
            sys.executable,
            "-m",
            "lfo_experiment.experiment7_worker",
            "9",
            str(catalog_path),
            str(codebook_path),
            str(output_dir),
            "--beam-width",
            str(args.beam_width),
            "--align-device",
            args.align_device,
        ]
        if args.batch_size is not None:
            command.extend(["--batch-size", str(args.batch_size)])
        if args.train_stage_batch_size is not None:
            command.extend(["--train-stage-batch-size", str(args.train_stage_batch_size)])
        if args.align_batch_size is not None:
            command.extend(["--align-batch-size", str(args.align_batch_size)])
        if args.cache_every is not None:
            command.extend(["--cache-every", str(args.cache_every)])
        if args.max_shapes is not None:
            command.extend(["--max-shapes", str(args.max_shapes)])
        if args.seed is not None:
            command.extend(["--seed", str(args.seed)])
        subprocess.run(command, check=True, env=_environment(args.artifacts), cwd=EXPERIMENT_DIR)
        return

    os.environ.setdefault("MPLCONFIGDIR", str(args.artifacts / "mpl"))
    from lfo_experiment.experiment7 import experiment7_status, run_experiment7_analysis

    if args.command == "status":
        print(experiment7_status(output_dir, "9"))
        return

    result = run_experiment7_analysis(output_dir, experiment="9")
    subprocess.run(
        [sys.executable, str(EXPERIMENT_DIR / "generate_experiment9_report.py")],
        check=True,
        env=_environment(args.artifacts),
        cwd=EXPERIMENT_DIR,
    )
    print(f"Wrote Experiment 9 screen analytics to {result['analytics_dir']}")


if __name__ == "__main__":
    main()

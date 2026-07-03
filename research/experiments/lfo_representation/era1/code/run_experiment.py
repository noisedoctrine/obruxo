#!/usr/bin/env python3
"""Command-line entry point for the LFO representation experiment."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys

from lfo_experiment.catalog import build_catalog, build_provisional_codebook


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
DEFAULT_METADATA = REPO_ROOT / "datasets" / "presetshare" / "raw" / "presetshare_vital_metadata.csv"
DEFAULT_ARTIFACTS = EXPERIMENT_DIR / "artifacts"


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument(
        "command",
        choices=(
            "catalog",
            "codebook",
            "benchmark",
            "experiment2",
            "experiment3",
            "experiment4",
            "experiment5",
            "experiment6",
            "experiment6_analysis",
            "experiment6_status",
            "experiment6_device_benchmark",
            "experiment7a",
            "experiment7a_analysis",
            "experiment7a_status",
            "experiment7b",
            "experiment7b_analysis",
            "experiment7b_status",
            "experiment8_screen",
            "experiment8_screen_analysis",
            "experiment8_screen_status",
            "all",
        ),
    )
    root.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    root.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    root.add_argument("--limit", type=int, help="limit presets during catalog smoke tests")
    root.add_argument("--max-shapes", type=int, help="sample at most this many shapes")
    root.add_argument("--resolution", type=int, default=1024)
    root.add_argument("--beam-width", type=int)
    root.add_argument("--batch-size", type=int, help="Experiment 7 eval batch size override")
    root.add_argument("--train-stage-batch-size", type=int, help="Experiment 7 XPU training-stage scoring chunk size")
    root.add_argument("--align-batch-size", type=int, help="Experiment 7 torch alignment chunk size")
    root.add_argument("--cache-every", type=int, help="Experiment 7 intra-run cache cadence; 0 disables")
    root.add_argument("--seed", type=int, help="experiment-specific deterministic seed")
    root.add_argument("--config", type=Path, help="Experiment 7B selected policy config")
    root.add_argument("--parallel", type=int, default=1, help="number of Experiment 6 candidates to run concurrently")
    root.add_argument(
        "--align-device",
        choices=("auto", "cpu", "xpu"),
        default="auto",
        help="alignment backend; auto uses local Intel XPU when available, otherwise CPU",
    )
    root.add_argument(
        "--quick",
        action="store_true",
        help="run a small Experiment 2 matrix for smoke testing",
    )
    root.add_argument(
        "--include-inactive",
        action="store_true",
        help="benchmark routes even when their amount is zero or they are bypassed",
    )
    root.add_argument(
        "--background",
        action="store_true",
        help="launch Experiment 5 in a detached background process and return immediately",
    )
    return root


def launch_experiment5_background(args: argparse.Namespace) -> int:
    catalog_path = args.artifacts / "lfo_catalog.csv"
    experiment4_dir = args.artifacts / "phase_factorized_residual"
    if not catalog_path.exists():
        raise SystemExit(f"Missing {catalog_path}; run the catalog stage first")
    if not (experiment4_dir / "codebooks").exists():
        raise SystemExit(f"Missing Experiment 4 codebooks under {experiment4_dir}")
    output_dir = args.artifacts / "phase_alignment_oracle"
    output_dir.mkdir(parents=True, exist_ok=True)
    marker_path = output_dir / "COMPLETED_EXCPERIMENT_5.txt"
    if marker_path.exists():
        marker_path.unlink()
    stdout_path = output_dir / "experiment5_background_stdout.log"
    stderr_path = output_dir / "experiment5_background_stderr.log"
    command = [
        sys.executable,
        "-u",
        "-m",
        "lfo_experiment.experiment5_worker",
        str(catalog_path),
        str(experiment4_dir),
        str(output_dir),
    ]
    if args.quick:
        command.append("--quick")
    environment = os.environ.copy()
    environment.setdefault("MPLCONFIGDIR", str(args.artifacts / "mpl"))
    environment.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP
    stdout_handle = stdout_path.open("w", encoding="utf-8")
    stderr_handle = stderr_path.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            command,
            cwd=EXPERIMENT_DIR,
            stdout=stdout_handle,
            stderr=stderr_handle,
            env=environment,
            creationflags=flags,
        )
    finally:
        stdout_handle.close()
        stderr_handle.close()
    print(f"Experiment 5 background PID: {process.pid}")
    print(f"Completion marker: {marker_path}")
    print(f"stdout log: {stdout_path}")
    print(f"stderr log: {stderr_path}")
    return process.pid


def main() -> None:
    args = parser().parse_args()
    catalog_path = args.artifacts / "lfo_catalog.csv"
    codebook_path = args.artifacts / "stock_codebook.json"
    if args.beam_width is not None:
        beam_width = args.beam_width
    elif args.command in {"experiment7b", "experiment8_screen"}:
        beam_width = 4
    elif args.command == "experiment4":
        beam_width = 64
    else:
        beam_width = 32

    if args.command in {"catalog", "all"}:
        build_catalog(args.metadata, catalog_path, limit=args.limit)
    if args.command in {"codebook", "all"}:
        if not catalog_path.exists():
            raise SystemExit(f"Missing {catalog_path}; run the catalog stage first")
        build_provisional_codebook(catalog_path, codebook_path)
    if args.command in {"benchmark", "all"}:
        missing = [path for path in (catalog_path, codebook_path) if not path.exists()]
        if missing:
            raise SystemExit(f"Missing prerequisite artifacts: {', '.join(map(str, missing))}")
        os.environ.setdefault("MPLCONFIGDIR", str(args.artifacts / "mpl"))
        from lfo_experiment.benchmark import run_oracle_benchmark

        run_oracle_benchmark(
            catalog_path,
            codebook_path,
            args.artifacts,
            resolution=args.resolution,
            max_shapes=args.max_shapes,
            active_only=not args.include_inactive,
        )
    if args.command == "experiment2":
        needs_catalog = not catalog_path.exists()
        if not needs_catalog:
            with catalog_path.open(encoding="utf-8", newline="") as handle:
                needs_catalog = "author_id" not in next(csv.reader(handle))
        if needs_catalog:
            print("Regenerating the LFO catalog with author metadata", flush=True)
            build_catalog(args.metadata, catalog_path, limit=args.limit)
        if not codebook_path.exists():
            build_provisional_codebook(catalog_path, codebook_path)

        os.environ.setdefault("MPLCONFIGDIR", str(args.artifacts / "mpl"))
        from lfo_experiment.experiment2 import run_experiment2

        keyword_args = {
            "baseline_path": args.artifacts / "oracle_summary.csv",
            "resolution": args.resolution,
            "beam_width": beam_width,
        }
        if args.quick:
            keyword_args.update(
                base_widths=(15, 16),
                residual_widths=(4,),
                depths=(1, 2),
                beam_width=min(beam_width, 8),
            )
        run_experiment2(
            catalog_path,
            codebook_path,
            args.artifacts / "stacked_residual",
            **keyword_args,
        )
    if args.command == "experiment3":
        needs_catalog = not catalog_path.exists()
        if not needs_catalog:
            with catalog_path.open(encoding="utf-8", newline="") as handle:
                needs_catalog = "author_id" not in next(csv.reader(handle))
        if needs_catalog:
            print("Regenerating the LFO catalog with author metadata", flush=True)
            build_catalog(args.metadata, catalog_path, limit=args.limit)
        if not codebook_path.exists():
            build_provisional_codebook(catalog_path, codebook_path)

        os.environ.setdefault("MPLCONFIGDIR", str(args.artifacts / "mpl"))
        from lfo_experiment.experiment3 import run_experiment3

        run_experiment3(
            catalog_path,
            codebook_path,
            args.artifacts / "frequency_first_residual",
            resolution=args.resolution,
            beam_width=beam_width,
            quick=args.quick,
            experiment2_summary=args.artifacts / "stacked_residual" / "stacked_summary.csv",
        )
    if args.command == "experiment4":
        needs_catalog = not catalog_path.exists()
        if not needs_catalog:
            with catalog_path.open(encoding="utf-8", newline="") as handle:
                needs_catalog = "author_id" not in next(csv.reader(handle))
        if needs_catalog:
            print("Regenerating the LFO catalog with author metadata", flush=True)
            build_catalog(args.metadata, catalog_path, limit=args.limit)
        if not codebook_path.exists():
            build_provisional_codebook(catalog_path, codebook_path)

        os.environ.setdefault("MPLCONFIGDIR", str(args.artifacts / "mpl"))
        from lfo_experiment.experiment4 import run_experiment4

        run_experiment4(
            catalog_path,
            codebook_path,
            args.artifacts / "phase_factorized_residual",
            resolution=args.resolution,
            beam_width=beam_width,
            quick=args.quick,
        )
    if args.command == "experiment5":
        if args.background:
            launch_experiment5_background(args)
            return
        if not catalog_path.exists():
            raise SystemExit(f"Missing {catalog_path}; run the catalog stage first")
        experiment4_dir = args.artifacts / "phase_factorized_residual"
        if not (experiment4_dir / "codebooks").exists():
            raise SystemExit(f"Missing Experiment 4 codebooks under {experiment4_dir}")
        command = [
            sys.executable, "-m", "lfo_experiment.experiment5_worker",
            str(catalog_path), str(experiment4_dir),
            str(args.artifacts / "phase_alignment_oracle"),
        ]
        if args.quick:
            command.append("--quick")
        environment = os.environ.copy()
        environment.setdefault("MPLCONFIGDIR", str(args.artifacts / "mpl"))
        # NumPy's threaded MKL and PyTorch XPU otherwise load competing Intel
        # OpenMP runtimes on Windows. CPU reference work is intentionally
        # sequential; XPU performs the batched production alignment.
        environment.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
        subprocess.run(command, check=True, env=environment)
    if args.command == "experiment6":
        if not catalog_path.exists():
            raise SystemExit(f"Missing {catalog_path}; run the catalog stage first")
        if not codebook_path.exists():
            raise SystemExit(f"Missing {codebook_path}; run the codebook stage first")
        experiment4_dir = args.artifacts / "phase_factorized_residual"
        if not (experiment4_dir / "codebooks").exists():
            raise SystemExit(f"Missing Experiment 4 codebooks under {experiment4_dir}")
        command = [
            sys.executable,
            "-m",
            "lfo_experiment.experiment6_worker",
            str(catalog_path),
            str(codebook_path),
            str(experiment4_dir),
            str(args.artifacts / "codebook_selection"),
            "--beam-width",
            str(beam_width),
        ]
        if args.quick:
            command.append("--quick")
        if args.max_shapes is not None:
            command.extend(["--max-shapes", str(args.max_shapes)])
        if args.parallel is not None:
            command.extend(["--parallel", str(max(1, args.parallel))])
        command.extend(["--align-device", args.align_device])
        environment = os.environ.copy()
        environment.setdefault("MPLCONFIGDIR", str(args.artifacts / "mpl"))
        environment.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
        subprocess.run(command, check=True, env=environment, cwd=EXPERIMENT_DIR)
    if args.command == "experiment6_status":
        from lfo_experiment.experiment6 import experiment6_status

        print(experiment6_status(args.artifacts / "codebook_selection"))
    if args.command == "experiment6_analysis":
        os.environ.setdefault("MPLCONFIGDIR", str(args.artifacts / "mpl"))
        from lfo_experiment.experiment6_analysis import run_experiment6_analysis

        output_dir = args.artifacts / "codebook_selection"
        result = run_experiment6_analysis(output_dir)
        print(f"Wrote Experiment 6 analytics to {result['analytics_dir']}")
    if args.command == "experiment6_device_benchmark":
        os.environ.setdefault("MPLCONFIGDIR", str(args.artifacts / "mpl"))
        os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
        from lfo_experiment.experiment6_device_benchmark import run_device_benchmark

        path = run_device_benchmark(
            args.artifacts / "codebook_selection" / "device_benchmark",
            repeats=3 if args.quick else 5,
            parallel=max(1, args.parallel),
        )
        print(f"Wrote device benchmark to {path}")
    if args.command == "experiment7a":
        if not catalog_path.exists():
            raise SystemExit(f"Missing {catalog_path}; run the catalog stage first")
        if not codebook_path.exists():
            raise SystemExit(f"Missing {codebook_path}; run the codebook stage first")
        command = [
            sys.executable,
            "-m",
            "lfo_experiment.experiment7_worker",
            "7A",
            str(catalog_path),
            str(codebook_path),
            str(args.artifacts / "additive_finalization_7a"),
            "--beam-width",
            str(beam_width),
            "--align-device",
            args.align_device,
        ]
        if args.quick:
            command.append("--quick")
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
        environment = os.environ.copy()
        environment.setdefault("MPLCONFIGDIR", str(args.artifacts / "mpl"))
        environment.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
        subprocess.run(command, check=True, env=environment, cwd=EXPERIMENT_DIR)
    if args.command == "experiment7b":
        if not catalog_path.exists():
            raise SystemExit(f"Missing {catalog_path}; run the catalog stage first")
        if not codebook_path.exists():
            raise SystemExit(f"Missing {codebook_path}; run the codebook stage first")
        if args.config is None:
            raise SystemExit(
                "Experiment 7B requires --config selected after reviewing Experiment 7A. "
                "Run experiment7a_analysis and choose one of "
                "artifacts/additive_finalization_7a/candidate_7b_configs/*.json, "
                "or create your own config."
            )
        command = [
            sys.executable,
            "-m",
            "lfo_experiment.experiment7_worker",
            "7B",
            str(catalog_path),
            str(codebook_path),
            str(args.artifacts / "additive_finalization_7b"),
            "--beam-width",
            str(beam_width),
            "--align-device",
            args.align_device,
        ]
        command.extend(["--config", str(args.config.resolve())])
        if args.quick:
            command.append("--quick")
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
        environment = os.environ.copy()
        environment.setdefault("MPLCONFIGDIR", str(args.artifacts / "mpl"))
        environment.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
        subprocess.run(command, check=True, env=environment, cwd=EXPERIMENT_DIR)
    if args.command == "experiment8_screen":
        if not catalog_path.exists():
            raise SystemExit(f"Missing {catalog_path}; run the catalog stage first")
        if not codebook_path.exists():
            raise SystemExit(f"Missing {codebook_path}; run the codebook stage first")
        command = [
            sys.executable,
            "-m",
            "lfo_experiment.experiment7_worker",
            "8",
            str(catalog_path),
            str(codebook_path),
            str(args.artifacts / "additive_finalization_8_screen"),
            "--beam-width",
            str(beam_width),
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
        environment = os.environ.copy()
        environment.setdefault("MPLCONFIGDIR", str(args.artifacts / "mpl"))
        environment.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
        subprocess.run(command, check=True, env=environment, cwd=EXPERIMENT_DIR)
    if args.command == "experiment7a_status":
        os.environ.setdefault("MPLCONFIGDIR", str(args.artifacts / "mpl"))
        from lfo_experiment.experiment7 import experiment7_status

        print(experiment7_status(args.artifacts / "additive_finalization_7a", "7A"))
    if args.command == "experiment7b_status":
        os.environ.setdefault("MPLCONFIGDIR", str(args.artifacts / "mpl"))
        from lfo_experiment.experiment7 import experiment7_status

        print(experiment7_status(args.artifacts / "additive_finalization_7b", "7B"))
    if args.command == "experiment8_screen_status":
        os.environ.setdefault("MPLCONFIGDIR", str(args.artifacts / "mpl"))
        from lfo_experiment.experiment7 import experiment7_status

        print(experiment7_status(args.artifacts / "additive_finalization_8_screen", "8"))
    if args.command == "experiment7a_analysis":
        os.environ.setdefault("MPLCONFIGDIR", str(args.artifacts / "mpl"))
        from lfo_experiment.experiment7 import run_experiment7_analysis

        result = run_experiment7_analysis(args.artifacts / "additive_finalization_7a", experiment="7A")
        print(f"Wrote Experiment 7A analytics to {result['analytics_dir']}")
    if args.command == "experiment7b_analysis":
        os.environ.setdefault("MPLCONFIGDIR", str(args.artifacts / "mpl"))
        from lfo_experiment.experiment7 import run_experiment7_analysis

        result = run_experiment7_analysis(args.artifacts / "additive_finalization_7b", experiment="7B")
        print(f"Wrote Experiment 7B analytics to {result['analytics_dir']}")
    if args.command == "experiment8_screen_analysis":
        os.environ.setdefault("MPLCONFIGDIR", str(args.artifacts / "mpl"))
        from lfo_experiment.experiment7 import run_experiment7_analysis

        result = run_experiment7_analysis(args.artifacts / "additive_finalization_8_screen", experiment="8")
        print(f"Wrote Experiment 8 screen analytics to {result['analytics_dir']}")


if __name__ == "__main__":
    main()

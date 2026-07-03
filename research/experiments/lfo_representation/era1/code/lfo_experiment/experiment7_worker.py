"""Process entrypoint for Experiment 7.

PyTorch is imported first so XPU/OpenMP initialization mirrors the safer
Experiment 5/6 worker layout on Windows.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path

import torch  # noqa: F401  # Deliberately first for XPU initialization order.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("7A", "7B", "8", "9"))
    parser.add_argument("catalog", type=Path)
    parser.add_argument("codebook", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--beam-width", type=int, default=32)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--train-stage-batch-size", type=int)
    parser.add_argument("--align-batch-size", type=int)
    parser.add_argument("--cache-every", type=int)
    parser.add_argument("--max-shapes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--align-device", choices=("auto", "cpu", "xpu"))
    args = parser.parse_args()
    align_device = args.align_device if args.align_device is not None else ("xpu" if args.experiment in {"8", "9"} else "auto")
    os.environ["LFO_ALIGN_DEVICE"] = align_device
    if args.train_stage_batch_size is not None:
        os.environ["LFO_TRAIN_STAGE_BATCH_SIZE"] = str(args.train_stage_batch_size)
    if args.align_batch_size is not None:
        os.environ["LFO_TORCH_ALIGN_BATCH_SIZE"] = str(args.align_batch_size)
    if args.cache_every is not None:
        os.environ["LFO_CACHE_EVERY"] = str(args.cache_every)

    from .experiment7 import SEED, run_experiment7a, run_experiment7b, run_experiment8_screen, run_experiment9_screen

    if args.experiment == "7A":
        run_experiment7a(
            args.catalog,
            args.codebook,
            args.output,
            quick=args.quick,
            beam_width=args.beam_width,
            batch_size=args.batch_size,
            max_shapes=args.max_shapes,
            seed=args.seed if args.seed is not None else SEED,
        )
    elif args.experiment == "7B":
        run_experiment7b(
            args.catalog,
            args.codebook,
            args.output,
            config_path=args.config,
            quick=args.quick,
            beam_width=args.beam_width,
            batch_size=args.batch_size,
            max_shapes=args.max_shapes,
            seed=args.seed if args.seed is not None else 7267,
        )
    elif args.experiment == "8":
        run_experiment8_screen(
            args.catalog,
            args.codebook,
            args.output,
            beam_width=args.beam_width,
            batch_size=args.batch_size,
            max_shapes=args.max_shapes,
            seed=args.seed if args.seed is not None else 7267,
        )
    else:
        run_experiment9_screen(
            args.catalog,
            args.codebook,
            args.output,
            beam_width=args.beam_width,
            batch_size=args.batch_size,
            max_shapes=args.max_shapes,
            seed=args.seed if args.seed is not None else 7267,
        )
    marker_path = args.output / f"COMPLETED_EXPERIMENT_{args.experiment}.txt"
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    marker_path.write_text(f"{timestamp}\nstatus=success\n", encoding="utf-8")


if __name__ == "__main__":
    main()

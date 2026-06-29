"""Process entrypoint for Experiment 6.

Import PyTorch first so XPU initialization happens before NumPy/SciPy-heavy
modules.  This mirrors Experiment 5's safer Windows/XPU process layout.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path

import torch  # Deliberately first for XPU/OpenMP initialization order.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path)
    parser.add_argument("codebook", type=Path)
    parser.add_argument("experiment4", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument("--finalist-beam-width", type=int, default=128)
    parser.add_argument("--max-shapes", type=int)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--align-device", choices=("auto", "cpu", "xpu"), default="auto")
    args = parser.parse_args()
    os.environ["LFO_ALIGN_DEVICE"] = args.align_device

    from .experiment6 import run_experiment6

    run_experiment6(
        args.catalog,
        args.codebook,
        args.experiment4,
        args.output,
        quick=args.quick,
        beam_width=args.beam_width,
        finalist_beam_width=args.finalist_beam_width,
        max_shapes=args.max_shapes,
        parallel=args.parallel,
    )
    marker_path = args.output / "COMPLETED_EXCPERIMENT_6.txt"
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    marker_path.write_text(f"{timestamp}\n", encoding="utf-8")


if __name__ == "__main__":
    main()


"""XPU-first process entrypoint for Experiment 5.

PyTorch XPU and NumPy/SciPy ship different Intel OpenMP runtimes on Windows.
Importing XPU first in this dedicated process avoids unsafe duplicate-runtime flags.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import torch  # Deliberately first: initialize the XPU runtime before NumPy/SciPy.


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("catalog", type=Path)
    parser.add_argument("experiment4", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    from .experiment5 import run_experiment5

    run_experiment5(args.catalog, args.experiment4, args.output, quick=args.quick)
    marker_path = args.output / "COMPLETED_EXCPERIMENT_5.txt"
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    marker_path.write_text(f"{timestamp}\n", encoding="utf-8")


if __name__ == "__main__":
    main()

"""Command-line entry point for the Era 2 LFO framework."""

from __future__ import annotations

import argparse
from pathlib import Path

from .flat import run_flat_smoke


ERA2_ROOT = Path(__file__).resolve().parents[2]


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    subcommands = root.add_subparsers(dest="command", required=True)

    smoke = subcommands.add_parser("smoke-flat", help="run the topology-free flat-categorical smoke path")
    smoke.add_argument("--output-dir", type=Path, default=ERA2_ROOT / "artifacts" / "smoke_flat")
    smoke.add_argument("--D", type=int, default=3, help="residual-layer count")
    smoke.add_argument("--W", type=int, default=4, help="flat atom choices per residual layer")
    smoke.add_argument("--base-dictionary-size", type=int, default=32)
    smoke.add_argument("--resolution", type=int, default=64)
    smoke.add_argument("--phase-bins", type=int, default=1)
    smoke.add_argument("--backend", choices=("auto", "numpy", "xpu"), default="auto")
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "smoke-flat":
        result = run_flat_smoke(
            args.output_dir,
            residual_layer_count=args.D,
            width=args.W,
            base_dictionary_size=args.base_dictionary_size,
            resolution=args.resolution,
            phase_bins=args.phase_bins,
            backend=args.backend,
        )
        print(f"Wrote smoke artifacts to {result['output_dir']}")
        print(f"head_outputs_actual={result['manifest']['head_outputs_actual']}")
        print(f"topology_contract_pass={result['topology_contract']['passed']}")


if __name__ == "__main__":
    main()


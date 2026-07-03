"""Command-line entry point for the Era 2 LFO framework."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from .analytics import analyze_run
from .flat import run_flat_smoke
from .grid_ceiling import (
    DEFAULT_ATOM_GRID_POINTS,
    DEFAULT_DENSE_POINTS,
    DEFAULT_OUTPUT_DIR,
    parse_grid_points,
    run_grid_ceiling_audit,
)
from .runner import DEFAULT_METADATA, run_experiment10_screen, status_text


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

    run_screen = subcommands.add_parser("run-screen", help="run an Era 2 experiment screen")
    run_screen.add_argument("--screen", choices=("experiment10",), default="experiment10")
    run_screen.add_argument("--profile", choices=("quick", "screen", "extended"), default="quick")
    run_screen.add_argument("--backend", choices=("auto", "numpy", "xpu"), default="auto")
    run_screen.add_argument("--run-dir", type=Path)
    run_screen.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    run_screen.add_argument("--resume", action="store_true")
    run_screen.add_argument("--rerun-failed", action="store_true")
    run_screen.add_argument("--no-analyze", action="store_true")
    run_screen.add_argument("--no-monitor", action="store_true", help="do not print live status while running")

    status = subcommands.add_parser("status", help="print run status")
    status.add_argument("--run-dir", type=Path, required=True)
    status.add_argument("--watch", type=float, help="refresh interval in seconds")

    analyze = subcommands.add_parser("analyze", help="regenerate run analytics")
    analyze.add_argument("--run-dir", type=Path, required=True)

    grid_ceiling = subcommands.add_parser("grid-ceiling", help="estimate best-case fixed atom-grid reconstruction loss")
    grid_ceiling.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    grid_ceiling.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    grid_ceiling.add_argument(
        "--atom-grid-points",
        default=",".join(str(value) for value in DEFAULT_ATOM_GRID_POINTS),
        help="comma-separated point-grid sizes",
    )
    grid_ceiling.add_argument("--dense-points", type=int, default=DEFAULT_DENSE_POINTS)
    grid_ceiling.add_argument("--metadata-limit", type=int)
    grid_ceiling.add_argument("--include-inactive", action="store_true")
    grid_ceiling.add_argument("--chunk-size", type=int, default=512)
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
    elif args.command == "run-screen":
        def monitor(run_dir: Path) -> None:
            print("")
            print(status_text(run_dir), flush=True)

        result = run_experiment10_screen(
            profile=args.profile,
            backend=args.backend,
            run_dir=args.run_dir,
            resume=args.resume,
            rerun_failed=args.rerun_failed,
            metadata_path=args.metadata,
            analyze=not args.no_analyze,
            monitor=None if args.no_monitor else monitor,
        )
        print(f"Wrote run artifacts to {result['run_dir']}")
        if result["analytics"]:
            print(f"Wrote analytics to {result['analytics']['analytics_dir']}")
    elif args.command == "status":
        while True:
            print(status_text(args.run_dir))
            if args.watch is None:
                break
            time.sleep(max(1.0, float(args.watch)))
    elif args.command == "analyze":
        result = analyze_run(args.run_dir)
        print(f"Wrote analytics to {result['analytics_dir']}")
    elif args.command == "grid-ceiling":
        result = run_grid_ceiling_audit(
            metadata_path=args.metadata,
            output_dir=args.output_dir,
            atom_grid_points=parse_grid_points(args.atom_grid_points),
            dense_points=args.dense_points,
            metadata_limit=args.metadata_limit,
            active_only=not args.include_inactive,
            chunk_size=args.chunk_size,
            progress=lambda message: print(message, flush=True),
        )
        print(f"Wrote grid ceiling audit to {result['output_dir']}")
        print(f"summary={result['summary']}")
        print(f"report={result['report']}")


if __name__ == "__main__":
    main()

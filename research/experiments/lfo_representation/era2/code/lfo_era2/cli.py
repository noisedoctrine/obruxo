"""Command-line entry point for the Era 2 LFO framework."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time
from typing import Callable

from .analytics import analyze_run
from .flat import run_flat_smoke
from .processed_corpus import (
    DEFAULT_CORPUS_DIR,
    DEFAULT_DENSE_RESOLUTION,
    build_lfo_corpus,
)
from .runner import DEFAULT_METADATA, DEFAULT_RUN_ROOT, run_experiment11_screen, status_text


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
    run_screen.add_argument("--screen", choices=("experiment11",), default="experiment11")
    run_screen.add_argument("--profile", choices=("quick", "screen", "extended"), default="quick")
    run_screen.add_argument("--backend", choices=("auto", "numpy", "xpu"), default="auto")
    run_screen.add_argument("--run-dir", type=Path)
    run_screen.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    run_screen.add_argument("--resume", action="store_true")
    run_screen.add_argument("--rerun-failed", action="store_true")
    run_screen.add_argument("--no-analyze", action="store_true")
    run_screen.add_argument("--async", dest="async_run", action="store_true", help="launch the run in the background and open the monitor")
    run_screen.add_argument("--no-monitor", action="store_true", help="do not print progress or open a monitor")
    run_screen.add_argument("--no-monitor-window", action="store_true", help="do not open a separate status monitor window")
    run_screen.add_argument("--monitor-refresh-seconds", type=int, default=30)

    status = subcommands.add_parser("status", help="print run status")
    status.add_argument("--run-dir", type=Path, required=True)
    status.add_argument("--watch", type=float, help="refresh interval in seconds")

    analyze = subcommands.add_parser("analyze", help="regenerate run analytics")
    analyze.add_argument("--run-dir", type=Path, required=True)

    build_corpus = subcommands.add_parser("build-lfo-corpus", help="build the processed PresetShare LFO corpus")
    build_corpus.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    build_corpus.add_argument("--output-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    build_corpus.add_argument("--dense-resolution", type=int, default=DEFAULT_DENSE_RESOLUTION)
    build_corpus.add_argument("--force", action="store_true")
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
        run_dir = _resolve_cli_run_dir(args.run_dir)

        def progress(message: str) -> None:
            print(f"experiment11 [{time.strftime('%H:%M:%S')}]: {message}", flush=True)

        if args.async_run:
            result = _launch_async_run_screen(args, run_dir, progress)
            print(f"Started async Experiment 11 run: {result['run_dir']}")
            print(f"runner_pid={result['runner_pid']}")
            print(f"stdout_log={result['stdout_log']}")
            print(f"stderr_log={result['stderr_log']}")
            if result["monitor_started"]:
                print(f"monitor_refresh_seconds={result['monitor_refresh_seconds']}")
            return

        result = run_experiment11_screen(
            profile=args.profile,
            backend=args.backend,
            run_dir=run_dir,
            resume=args.resume,
            rerun_failed=args.rerun_failed,
            metadata_path=args.metadata,
            analyze=not args.no_analyze,
            progress=None if args.no_monitor else progress,
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
    elif args.command == "build-lfo-corpus":
        result = build_lfo_corpus(
            metadata_path=args.metadata,
            output_dir=args.output_dir,
            dense_resolution=args.dense_resolution,
            force=args.force,
            progress=lambda message: print(message, flush=True),
        )
        print(f"Wrote processed LFO corpus to {result['output_dir']}")


def _resolve_cli_run_dir(run_dir: Path | None) -> Path:
    if run_dir is not None:
        return Path(run_dir)
    return DEFAULT_RUN_ROOT / f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"


def _open_monitor_window(run_dir: Path, refresh_seconds: int, progress: Callable[[str], None]) -> bool:
    if sys.platform != "win32":
        progress("monitor_window skipped reason=not_windows")
        return False
    script = ERA2_ROOT / "code" / "monitor_era2_run.ps1"
    if not script.exists():
        progress(f"monitor_window skipped reason=missing_script path={script}")
        return False
    command = [
        "powershell.exe",
        "-NoProfile",
        "-NoExit",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script),
        "-RunDir",
        str(run_dir),
        "-RefreshSeconds",
        str(max(1, int(refresh_seconds))),
        "-PythonExe",
        sys.executable,
    ]
    try:
        subprocess.Popen(command, cwd=ERA2_ROOT, creationflags=subprocess.CREATE_NEW_CONSOLE)
        progress(f"monitor_window_opened run_dir={run_dir} refresh_seconds={max(1, int(refresh_seconds))}")
        return True
    except Exception as exc:
        progress(f"monitor_window_failed error={exc}")
        return False


def _launch_async_run_screen(args: argparse.Namespace, run_dir: Path, progress: Callable[[str], None]) -> dict[str, object]:
    log_dir = DEFAULT_RUN_ROOT / "launcher_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = log_dir / f"{run_dir.name}_stdout.log"
    stderr_log = log_dir / f"{run_dir.name}_stderr.log"
    command = _async_runner_command(args, run_dir)
    progress(f"async_runner_start run_dir={run_dir}")
    with stdout_log.open("w", encoding="utf-8") as stdout, stderr_log.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            command,
            cwd=Path.cwd(),
            stdout=stdout,
            stderr=stderr,
            env=None,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    monitor_started = False
    if not args.no_monitor and not args.no_monitor_window:
        monitor_started = _open_monitor_window(run_dir, args.monitor_refresh_seconds, progress)
    return {
        "run_dir": str(run_dir),
        "runner_pid": process.pid,
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "monitor_started": monitor_started,
        "monitor_refresh_seconds": max(1, int(args.monitor_refresh_seconds)),
    }


def _async_runner_command(args: argparse.Namespace, run_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(ERA2_ROOT / "code" / "run_era2.py"),
        "run-screen",
        "--screen",
        args.screen,
        "--profile",
        args.profile,
        "--backend",
        args.backend,
        "--run-dir",
        str(run_dir),
        "--metadata",
        str(args.metadata),
        "--no-monitor-window",
    ]
    if args.resume:
        command.append("--resume")
    if args.rerun_failed:
        command.append("--rerun-failed")
    if args.no_analyze:
        command.append("--no-analyze")
    if args.no_monitor:
        command.append("--no-monitor")
    return command


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


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


os.environ.update(_pop_runtime_env_args(sys.argv))

from lfo_era2.component_ladder import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    analyze_component_ladder,
    run_component_ladder,
    status_text,
)
from lfo_era2.runner import DEFAULT_METADATA  # noqa: E402


def main() -> None:
    args = _parser().parse_args()
    if args.command == "run":
        if args.async_run:
            result = _launch_async(args)
            print(f"Started async Experiment 12 screening run: {result['output_dir']}", flush=True)
            print(f"runner_pid={result['runner_pid']}", flush=True)
            print(f"stdout_log={result['stdout_log']}", flush=True)
            print(f"stderr_log={result['stderr_log']}", flush=True)
            if result["monitor_started"]:
                print(f"monitor_refresh_seconds={result['monitor_refresh_seconds']}", flush=True)
            return
        row_ids = set(args.rows.split(",")) if args.rows else None
        result = run_component_ladder(
            output_dir=args.output_dir,
            metadata_path=args.metadata,
            backend=args.backend,
            smoke=args.smoke,
            corpus_sample_fraction=args.corpus_sample_fraction,
            resume=args.resume,
            row_ids=row_ids,
            max_utility_candidates=args.max_utility_candidates,
            chunk_size=args.chunk_size,
            write_report=not args.no_report,
            progress=lambda message: print(message, flush=True),
        )
        _print_result(result)
    elif args.command == "analyze":
        result = analyze_component_ladder(output_dir=args.run_dir, write_report=not args.no_report)
        _print_result(result)
    elif args.command == "status":
        print(status_text(args.run_dir), flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Experiment 12 fixed-W8D16 screening grid.")
    subcommands = parser.add_subparsers(dest="command", required=True)

    run = subcommands.add_parser("run", help="run the Experiment 12 screening grid")
    run.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    run.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    run.add_argument("--backend", choices=["auto", "numpy", "xpu"], default="auto")
    run.add_argument("--smoke", action="store_true")
    run.add_argument("--corpus-sample-fraction", type=float, default=1.0)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--chunk-size", type=int, default=256)
    run.add_argument("--max-utility-candidates", type=int, default=None)
    run.add_argument("--rows", default="", help="optional comma-separated row ids for partial diagnostic runs")
    run.add_argument("--no-report", action="store_true")
    run.add_argument("--async", dest="async_run", action="store_true")
    run.add_argument("--monitor-refresh-seconds", type=int, default=30)
    run.add_argument("--no-monitor-window", action="store_true")

    analyze = subcommands.add_parser("analyze", help="regenerate Experiment 12 screening analytics and report")
    analyze.add_argument("--run-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    analyze.add_argument("--no-report", action="store_true")

    status = subcommands.add_parser("status", help="print Experiment 12 run status")
    status.add_argument("--run-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def _launch_async(args: argparse.Namespace) -> dict[str, object]:
    log_dir = DEFAULT_OUTPUT_DIR.parent / "launcher_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = log_dir / f"{args.output_dir.name}_stdout.log"
    stderr_log = log_dir / f"{args.output_dir.name}_stderr.log"
    command = _async_command(args)
    with stdout_log.open("w", encoding="utf-8") as stdout, stderr_log.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            command,
            cwd=Path.cwd(),
            stdout=stdout,
            stderr=stderr,
            env=os.environ.copy(),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    monitor_started = False
    if not args.no_monitor_window:
        monitor_started = _open_monitor(args.output_dir, args.monitor_refresh_seconds)
    return {
        "output_dir": str(args.output_dir),
        "runner_pid": process.pid,
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "monitor_started": monitor_started,
        "monitor_refresh_seconds": max(1, int(args.monitor_refresh_seconds)),
    }


def _async_command(args: argparse.Namespace) -> list[str]:
    command = [sys.executable, str(Path(__file__).resolve()), "run"]
    command.extend(["--output-dir", str(args.output_dir)])
    command.extend(["--metadata", str(args.metadata)])
    command.extend(["--backend", args.backend])
    command.extend(["--corpus-sample-fraction", str(args.corpus_sample_fraction)])
    command.extend(["--chunk-size", str(args.chunk_size)])
    if args.smoke:
        command.append("--smoke")
    if args.resume:
        command.append("--resume")
    if args.max_utility_candidates is not None:
        command.extend(["--max-utility-candidates", str(args.max_utility_candidates)])
    if args.rows:
        command.extend(["--rows", args.rows])
    if args.no_report:
        command.append("--no-report")
    return command


def _open_monitor(run_dir: Path, refresh_seconds: int) -> bool:
    if sys.platform != "win32":
        return False
    script = Path(__file__).resolve()
    command = (
        f"$Host.UI.RawUI.WindowTitle = 'LFO Era 2 Experiment 12 Monitor'; "
        "while ($true) { "
        "Clear-Host; "
        "Write-Host 'LFO Era 2 Experiment 12 Monitor' -ForegroundColor Cyan; "
        "Write-Host ('Updated: ' + (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')); "
        f"Write-Host 'RunDir: {run_dir}'; "
        "Write-Host ('-' * 72) -ForegroundColor DarkGray; "
        f"& '{sys.executable}' '{script}' status --run-dir '{run_dir}'; "
        "Write-Host ''; Write-Host ('-' * 72) -ForegroundColor DarkGray; "
        "Write-Host 'Recent events' -ForegroundColor DarkCyan; "
        f"if (Test-Path '{run_dir}\\events.jsonl') {{ Get-Content '{run_dir}\\events.jsonl' -Tail 6 }} else {{ Write-Host 'No events yet.' }}; "
        f"Start-Sleep -Seconds {max(1, int(refresh_seconds))}; "
        "}"
    )
    try:
        subprocess.Popen(
            ["powershell.exe", "-NoProfile", "-NoExit", "-ExecutionPolicy", "Bypass", "-Command", command],
            cwd=Path.cwd(),
            creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
        )
        return True
    except Exception:
        return False


def _print_result(result: dict[str, str]) -> None:
    print("Wrote Experiment 12 screening-grid artifacts", flush=True)
    for key, value in result.items():
        print(f"{key}={value}", flush=True)


if __name__ == "__main__":
    main()

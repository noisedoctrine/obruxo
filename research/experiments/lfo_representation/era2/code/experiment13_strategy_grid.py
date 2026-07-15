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

from lfo_era2.strategy_grid import (  # noqa: E402
    DEFAULT_METADATA,
    DEFAULT_OUTPUT_DIR,
    Experiment13Error,
    analyze_strategy_grid,
    override_epsilon,
    run_13a,
    run_13b,
    run_13b_pilot,
    select_epsilon,
    status_text,
)


def main() -> None:
    args = _parser().parse_args()
    try:
        if args.command in {"run-13a", "run-13b"} and args.async_run:
            result = _launch_async(args)
            print(f"Started async Experiment 13 {args.command}: {result['output_dir']}", flush=True)
            print(f"runner_pid={result['runner_pid']}", flush=True)
            print(f"stdout_log={result['stdout_log']}", flush=True)
            print(f"stderr_log={result['stderr_log']}", flush=True)
            if result["monitor_started"]:
                print(f"monitor_refresh_seconds={result['monitor_refresh_seconds']}", flush=True)
            return
        if args.command == "run-13a":
            run_13a(
                output_dir=args.output_dir,
                metadata_path=args.metadata,
                backend=args.backend,
                smoke=args.smoke,
                corpus_sample_fraction=args.corpus_sample_fraction,
                resume=args.resume,
                row_ids=_row_ids(args.rows),
                chunk_size=args.chunk_size,
                progress=_progress,
            )
        elif args.command == "select-epsilon":
            selection = select_epsilon(run_dir=args.run_dir)
            print(f"selected_epsilon={selection.selected_epsilon}", flush=True)
        elif args.command == "run-13b-pilot":
            run_13b_pilot(
                output_dir=args.output_dir,
                epsilon_selection_path=args.epsilon_selection,
                candidate_epsilons=args.candidate_epsilons,
                row_ids=_row_ids(args.rows),
                metadata_path=args.metadata,
                backend=args.backend,
                chunk_size=args.chunk_size,
                progress=_progress,
            )
        elif args.command == "override-epsilon":
            selection = override_epsilon(
                run_dir=args.run_dir,
                selected_epsilon=args.selected_epsilon,
                rationale=args.rationale,
            )
            print(f"selected_epsilon={selection.selected_epsilon}", flush=True)
        elif args.command == "run-13b":
            run_13b(
                output_dir=args.output_dir,
                epsilon_selection_path=args.epsilon_selection,
                metadata_path=args.metadata,
                backend=args.backend,
                smoke=args.smoke,
                corpus_sample_fraction=args.corpus_sample_fraction,
                resume=args.resume,
                row_ids=_row_ids(args.rows),
                chunk_size=args.chunk_size,
                progress=_progress,
            )
        elif args.command == "analyze":
            result = analyze_strategy_grid(run_dir=args.run_dir)
            for key, value in result.items():
                print(f"{key}={value}", flush=True)
        elif args.command == "status":
            print(status_text(args.run_dir), flush=True)
    except (Experiment13Error, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2) from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Experiment 13 fixed-W8D16 strategy grid.")
    subcommands = parser.add_subparsers(dest="command", required=True)

    run_a = subcommands.add_parser("run-13a", help="run the 90-row Experiment 13A phase")
    _add_run_arguments(run_a, include_selection=False)

    select = subcommands.add_parser("select-epsilon", help="validate completed 13A calibration inputs and select epsilon")
    select.add_argument("--run-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    pilot = subcommands.add_parser("run-13b-pilot", help="run the restricted Experiment 13B pilot")
    pilot.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    pilot.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    pilot.add_argument("--backend", choices=["auto", "numpy", "xpu"], default="auto")
    pilot.add_argument("--epsilon-selection", type=Path, default=None)
    pilot.add_argument("--candidate-epsilons", nargs="+", type=float, required=True)
    pilot.add_argument("--rows", default="", help="optional comma-separated 13B row ids; only pilot policies are accepted")
    pilot.add_argument("--chunk-size", type=int, default=256)

    override = subcommands.add_parser("override-epsilon", help="record an explicit pilot-backed epsilon decision")
    override.add_argument("--run-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    override.add_argument("--selected-epsilon", type=float, required=True)
    override.add_argument("--rationale", required=True)

    run_b = subcommands.add_parser("run-13b", help="run the gated 90-row Experiment 13B phase")
    _add_run_arguments(run_b, include_selection=True)

    analyze = subcommands.add_parser("analyze", help="validate complete phases and generate Experiment 13 outputs")
    analyze.add_argument("--run-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    status = subcommands.add_parser("status", help="print Experiment 13 phase and gate status")
    status.add_argument("--run-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def _add_run_arguments(parser: argparse.ArgumentParser, *, include_selection: bool) -> None:
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--backend", choices=["auto", "numpy", "xpu"], default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--corpus-sample-fraction", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rows", default="", help="optional comma-separated row ids for partial diagnostic runs")
    parser.add_argument("--async", dest="async_run", action="store_true")
    parser.add_argument("--monitor-refresh-seconds", type=int, default=30)
    parser.add_argument("--no-monitor-window", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=256)
    if include_selection:
        parser.add_argument("--epsilon-selection", type=Path, default=None)


def _row_ids(value: str) -> set[str] | None:
    rows = {item.strip() for item in value.split(",") if item.strip()}
    return rows or None


def _launch_async(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    log_dir = output_dir.parent / "launcher_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = log_dir / f"{output_dir.name}_{args.command}_stdout.log"
    stderr_log = log_dir / f"{output_dir.name}_{args.command}_stderr.log"
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
        monitor_started = _open_monitor(output_dir, args.monitor_refresh_seconds)
    return {
        "output_dir": str(output_dir),
        "runner_pid": process.pid,
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "monitor_started": monitor_started,
        "monitor_refresh_seconds": max(1, int(args.monitor_refresh_seconds)),
    }


def _async_command(args: argparse.Namespace) -> list[str]:
    command = [sys.executable, str(Path(__file__).resolve()), args.command]
    command.extend(["--output-dir", str(args.output_dir)])
    command.extend(["--metadata", str(args.metadata)])
    command.extend(["--backend", args.backend])
    command.extend(["--corpus-sample-fraction", str(args.corpus_sample_fraction)])
    command.extend(["--chunk-size", str(args.chunk_size)])
    if args.smoke:
        command.append("--smoke")
    if args.resume:
        command.append("--resume")
    if args.rows:
        command.extend(["--rows", args.rows])
    if getattr(args, "epsilon_selection", None) is not None:
        command.extend(["--epsilon-selection", str(args.epsilon_selection)])
    return command


def _open_monitor(run_dir: Path, refresh_seconds: int) -> bool:
    if sys.platform != "win32":
        return False
    script = Path(__file__).resolve()
    command = (
        "$Host.UI.RawUI.WindowTitle = 'LFO Era 2 Experiment 13 Monitor'; "
        "while ($true) { "
        "Clear-Host; "
        "Write-Host 'LFO Era 2 Experiment 13 Monitor' -ForegroundColor Cyan; "
        "Write-Host ('Updated: ' + (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')); "
        f"Write-Host 'RunDir: {run_dir}'; "
        "Write-Host ('-' * 72) -ForegroundColor DarkGray; "
        f"& '{sys.executable}' '{script}' status --run-dir '{run_dir}'; "
        "Write-Host ''; Write-Host ('-' * 72) -ForegroundColor DarkGray; "
        "Write-Host 'Recent events' -ForegroundColor DarkCyan; "
        f"if (Test-Path '{run_dir}\\run_events.jsonl') {{ Get-Content '{run_dir}\\run_events.jsonl' -Tail 8 }} else {{ Write-Host 'No events yet.' }}; "
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


def _progress(message: str) -> None:
    print(message, flush=True)


if __name__ == "__main__":
    main()

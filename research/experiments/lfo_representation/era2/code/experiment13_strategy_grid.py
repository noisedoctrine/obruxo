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
    analyze_scaling_ablation,
    analyze_strategy_grid,
    override_epsilon,
    run_13a,
    run_13b,
    run_13b_pilot,
    request_cancel,
    select_epsilon,
    status_text,
    verify_equivalence,
)
from lfo_era2.strategy_grid_report import analyze_13a_strategy_grid, analyze_partial_strategy_grid  # noqa: E402
from lfo_era2.strategy_grid_execution import KeepAwakeError, scoped_system_required  # noqa: E402
from lfo_era2.strategy_grid_thresholds import replay_strict_perfect_thresholds  # noqa: E402


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
        if args.command in {"run-13a", "run-13b", "run-13b-pilot", "verify-equivalence"}:
            with scoped_system_required(strict=True):
                _execute(args)
        else:
            _execute(args)
    except (Experiment13Error, KeepAwakeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2) from exc


def _execute(args: argparse.Namespace) -> None:
        if args.command == "run-13a":
            run_13a(
                output_dir=args.output_dir,
                metadata_path=args.metadata,
                backend=args.backend,
                smoke=args.smoke,
                corpus_sample_fraction=args.corpus_sample_fraction,
                train_sample_fraction=args.train_sample_fraction,
                validation_sample_fraction=args.validation_sample_fraction,
                sample_seed=args.sample_seed,
                cache_dir=args.cache_dir,
                rebuild_cache=args.rebuild_cache,
                verify_optimized_kernels=args.verify_optimized_kernels,
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
                cache_dir=args.cache_dir,
                rebuild_cache=args.rebuild_cache,
                verify_optimized_kernels=args.verify_optimized_kernels,
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
                train_sample_fraction=args.train_sample_fraction,
                validation_sample_fraction=args.validation_sample_fraction,
                sample_seed=args.sample_seed,
                cache_dir=args.cache_dir,
                rebuild_cache=args.rebuild_cache,
                verify_optimized_kernels=args.verify_optimized_kernels,
                resume=args.resume,
                row_ids=_row_ids(args.rows),
                chunk_size=args.chunk_size,
                progress=_progress,
            )
        elif args.command == "analyze":
            result = analyze_strategy_grid(run_dir=args.run_dir)
            for key, value in result.items():
                print(f"{key}={value}", flush=True)
        elif args.command == "analyze-partial":
            result = analyze_partial_strategy_grid(
                run_dir=args.run_dir,
                analysis_output_dir=args.analysis_output_dir,
                report_path=args.report_path,
                html_report_path=args.html_report_path,
                image_dir=args.image_dir,
            )
            for key, value in result.items():
                print(f"{key}={value}", flush=True)
        elif args.command == "analyze-13a":
            result = analyze_13a_strategy_grid(
                run_dir=args.run_dir,
                analysis_output_dir=args.analysis_output_dir,
                report_path=args.report_path,
                html_report_path=args.html_report_path,
                image_dir=args.image_dir,
                scaling_baseline_run=args.scaling_baseline_run,
                strict_thresholds_path=args.strict_thresholds_path,
            )
            for key, value in result.items():
                print(f"{key}={value}", flush=True)
        elif args.command == "replay-strict-thresholds":
            result = replay_strict_perfect_thresholds(
                run_dir=args.run_dir,
                output_dir=args.output_dir,
                metadata_path=args.metadata,
                cache_dir=args.cache_dir,
                backend=args.backend,
                chunk_size=args.chunk_size,
                progress=_progress,
            )
            for key, value in result.items():
                print(f"{key}={value}", flush=True)
        elif args.command == "status":
            print(status_text(args.run_dir), flush=True)
        elif args.command == "monitor":
            started = _open_monitor(args.run_dir, args.monitor_refresh_seconds)
            if not started:
                raise Experiment13Error("monitor window could not be started on this platform")
            print(f"monitor_started=true refresh_seconds={args.monitor_refresh_seconds}", flush=True)
        elif args.command == "cancel":
            payload = request_cancel(args.run_dir, reason=args.reason)
            print(f"cancel_requested_at_utc={payload['requested_at_utc']}", flush=True)
        elif args.command == "verify-equivalence":
            result = verify_equivalence(
                baseline_run=args.baseline_run, output_dir=args.output_dir,
                metadata_path=args.metadata, cache_dir=args.cache_dir,
                backend=args.backend, row_ids=_row_ids(args.rows),
                chunk_size=args.chunk_size, progress=_progress,
            )
            print(f"passed={result['passed']} row_count={result['row_count']}", flush=True)
        elif args.command == "analyze-scaling":
            result = analyze_scaling_ablation(
                full_run=args.full_run, sampled_run=args.sampled_run, output_dir=args.output_dir,
            )
            for key, value in result.items():
                print(f"{key}={value}", flush=True)


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
    _add_cache_arguments(pilot)

    override = subcommands.add_parser("override-epsilon", help="record an explicit pilot-backed epsilon decision")
    override.add_argument("--run-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    override.add_argument("--selected-epsilon", type=float, required=True)
    override.add_argument("--rationale", required=True)

    run_b = subcommands.add_parser(
        "run-13b", help="run the gated 45-row LayerClip0To1-only Experiment 13B phase"
    )
    _add_run_arguments(run_b, include_selection=True)

    analyze = subcommands.add_parser("analyze", help="validate complete phases and generate Experiment 13 outputs")
    analyze.add_argument("--run-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    partial = subcommands.add_parser("analyze-partial", help="generate a provisional report from completed 13A row shards")
    partial.add_argument("--run-dir", type=Path, required=True)
    partial.add_argument("--analysis-output-dir", type=Path, required=True)
    partial.add_argument("--report-path", type=Path, required=True)
    partial.add_argument("--html-report-path", type=Path)
    partial.add_argument("--image-dir", type=Path, required=True)

    complete_a = subcommands.add_parser("analyze-13a", help="generate the complete Experiment 13A report")
    complete_a.add_argument("--run-dir", type=Path, required=True)
    complete_a.add_argument("--analysis-output-dir", type=Path, required=True)
    complete_a.add_argument("--report-path", type=Path, required=True)
    complete_a.add_argument("--html-report-path", type=Path)
    complete_a.add_argument("--image-dir", type=Path, required=True)
    complete_a.add_argument("--scaling-baseline-run", type=Path)
    complete_a.add_argument("--strict-thresholds-path", type=Path)

    thresholds = subcommands.add_parser(
        "replay-strict-thresholds",
        help="replay saved 13A codebooks on validation and calculate strict-perfect threshold sensitivity",
    )
    thresholds.add_argument("--run-dir", type=Path, required=True)
    thresholds.add_argument("--output-dir", type=Path, required=True)
    thresholds.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    thresholds.add_argument("--cache-dir", type=Path, default=None)
    thresholds.add_argument("--backend", choices=["auto", "numpy", "xpu"], default="auto")
    thresholds.add_argument("--chunk-size", type=int, default=256)

    status = subcommands.add_parser("status", help="print Experiment 13 phase and gate status")
    status.add_argument("--run-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    monitor = subcommands.add_parser("monitor", help="open the Windows live monitor for an existing run")
    monitor.add_argument("--run-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    monitor.add_argument("--monitor-refresh-seconds", type=int, default=30)

    cancel = subcommands.add_parser("cancel", help="request cancellation at the next safe checkpoint")
    cancel.add_argument("--run-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    cancel.add_argument("--reason", required=True)

    verify = subcommands.add_parser("verify-equivalence", help="compare optimized rows with a legacy baseline")
    verify.add_argument("--baseline-run", type=Path, required=True)
    verify.add_argument("--output-dir", type=Path, required=True)
    verify.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    verify.add_argument("--backend", choices=["auto", "numpy", "xpu"], default="auto")
    verify.add_argument("--rows", required=True)
    verify.add_argument("--chunk-size", type=int, default=256)
    verify.add_argument("--cache-dir", type=Path, default=None)

    scaling = subcommands.add_parser("analyze-scaling", help="compare matched full- and sampled-training rows")
    scaling.add_argument("--full-run", type=Path, required=True)
    scaling.add_argument("--sampled-run", type=Path, required=True)
    scaling.add_argument("--output-dir", type=Path, required=True)
    return parser


def _add_run_arguments(parser: argparse.ArgumentParser, *, include_selection: bool) -> None:
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--backend", choices=["auto", "numpy", "xpu"], default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--corpus-sample-fraction", type=float, default=None)
    parser.add_argument("--train-sample-fraction", type=float, default=1.0)
    parser.add_argument("--validation-sample-fraction", type=float, default=1.0)
    parser.add_argument("--sample-seed", type=int, default=13)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rows", default="", help="optional comma-separated row ids for partial diagnostic runs")
    parser.add_argument("--async", dest="async_run", action="store_true")
    parser.add_argument("--monitor-refresh-seconds", type=int, default=30)
    parser.add_argument("--no-monitor-window", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=256)
    _add_cache_arguments(parser)
    if include_selection:
        parser.add_argument("--epsilon-selection", type=Path, default=None)


def _add_cache_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--verify-optimized-kernels", choices=["off", "first-use"], default="off")


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
    if args.corpus_sample_fraction is not None:
        command.extend(["--corpus-sample-fraction", str(args.corpus_sample_fraction)])
    command.extend(["--train-sample-fraction", str(args.train_sample_fraction)])
    command.extend(["--validation-sample-fraction", str(args.validation_sample_fraction)])
    command.extend(["--sample-seed", str(args.sample_seed)])
    command.extend(["--chunk-size", str(args.chunk_size)])
    if args.cache_dir is not None:
        command.extend(["--cache-dir", str(args.cache_dir)])
    if args.rebuild_cache:
        command.append("--rebuild-cache")
    command.extend(["--verify-optimized-kernels", args.verify_optimized_kernels])
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

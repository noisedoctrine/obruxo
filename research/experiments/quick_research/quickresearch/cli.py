from __future__ import annotations

import argparse
import importlib
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

from .common import BudgetExpired, Context, OUTPUTS, REPO, WORKSPACE, checked_output, code_hash, digest, file_hash, read_json, status, write_json, write_text
from .source_audit import PINNED_REVISION

COMMANDS = {"identity": ("identity", "run"), "prepare-fixtures": ("fixtures", "run"),
            "measurement": ("measurement", "run"), "encoder": ("encoder", "run"),
            "source-audit": ("source_audit", "run"), "source-review": ("source_audit", "review")}
SUCCESS = {"complete", "complete_with_failures", "evidence_collected_review_required", "review_recorded", "review_partial"}


def stop_worker(child) -> None:
    if child.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"], capture_output=True, timeout=5,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except (OSError, subprocess.TimeoutExpired):
            pass
        if child.poll() is None:
            child.kill()
    else:
        os.killpg(child.pid, signal.SIGKILL)
    child.wait()


def budget_seconds(value: str) -> int:
    seconds = int(value)
    if not 1 <= seconds <= 1800:
        raise argparse.ArgumentTypeError("budget must be 1–1800 seconds; it cannot exceed 30 minutes")
    return seconds


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Issue #38: opt-in audits and small experiments. --help starts no work.")
    commands = root.add_subparsers(dest="command", required=True)
    for name in COMMANDS:
        command = commands.add_parser(name)
        command.add_argument("--output", required=True, help=f"private generated output directory beneath {OUTPUTS}")
        command.add_argument("--budget-seconds", type=budget_seconds, default=1800)
        command.add_argument("--resume", action="store_true", help="reuse verified caches with an identical configuration and implementation")
        if name == "identity":
            command.add_argument("--corpus", required=True)
            command.add_argument("--seed", type=int, default=0)
        elif name == "prepare-fixtures":
            command.add_argument("--identity", required=True, help="A's identity_manifest.json")
            command.add_argument("--metadata", required=True, help="CSV with exact type labels and preset_file paths relative to the CSV")
        elif name in ("measurement", "encoder"):
            command.add_argument("--fixtures" if name == "measurement" else "--measurement", required=True)
            command.add_argument("--renderer-config", default=str(REPO / "research/data_generation/configs/renderer.yaml"))
            command.add_argument("--plugin", help="installed Vital VST3; defaults to reviewed renderer configuration")
            if name == "encoder":
                command.add_argument("--checkpoint", default=str(REPO / "research/modelling/basic_pitch/artifacts/basic_pitch_icassp_2022.pt"))
                command.add_argument("--device", choices=("auto", "cpu", "xpu"), default="auto")
        elif name == "source-audit":
            command.add_argument("--source", required=True, help="existing local Vital git checkout; never fetched or changed")
            command.add_argument("--revision", default=PINNED_REVISION)
            command.add_argument("--max-hits", type=int, choices=range(1, 101), default=20, metavar="1..100")
        elif name == "source-review":
            command.add_argument("--packet", required=True)
            command.add_argument("--answers", required=True)
    snapshot = commands.add_parser("status", help="read an existing task status without resuming it")
    snapshot.add_argument("--output", required=True)
    return root


def execute(configuration: dict) -> int:
    args = argparse.Namespace(**configuration)
    context = Context(Path(args.output), max(.1, args.budget_seconds - min(10, args.budget_seconds / 5)))
    try:
        module_name, function_name = COMMANDS[args.command]
        result = getattr(importlib.import_module(f"quickresearch.{module_name}"), function_name)(context, args)
        print(json.dumps(result), flush=True)
        return 0 if result["state"] in SUCCESS else 2
    except BudgetExpired as error:
        status(context.output, "budget_exhausted", message=str(error))
        write_text(context.output / "INCOMPLETE.md", "# Incomplete execution\n\nBudget reached. Cached artifacts are retained; no complete research finding is established.\n")
        return 2
    except Exception as error:
        status(context.output, "failed", error_type=type(error).__name__, message=str(error))
        write_text(context.output / "FAILURE.md", f"# Task failed\n\n{type(error).__name__}: {error}\n\nInspect local status and logs. Partial artifacts are not a complete finding.\n")
        print(f"{type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1


def supervise(configuration: dict) -> int:
    started = time.monotonic()
    output = checked_output(Path(configuration["output"]))
    output.mkdir(parents=True, exist_ok=True)
    configuration["output"] = str(output)
    for key in ("corpus", "identity", "metadata", "fixtures", "measurement", "renderer_config", "plugin", "checkpoint", "source", "packet", "answers"):
        if configuration.get(key):
            configuration[key] = str(Path(configuration[key]).resolve())
    configuration["implementation_hash"] = code_hash(WORKSPACE, REPO / "research/data_generation/obruxo_data",
                                                      REPO / "research/modelling/basic_pitch/obruxo_basic_pitch")
    inputs = {key: file_hash(Path(value)) for key, value in configuration.items()
              if key in ("identity", "metadata", "fixtures", "renderer_config", "checkpoint", "packet", "answers") and Path(value).is_file()}
    identity = {key: value for key, value in configuration.items() if key not in ("resume", "budget_seconds")}
    fingerprint = digest({"configuration": identity, "input_files": inputs})
    invocation = output / "invocation.json"
    if invocation.exists():
        if not configuration["resume"]:
            raise ValueError("output already has a task; use --resume or a new output directory")
        if read_json(invocation)["fingerprint"] != fingerprint:
            raise ValueError("configuration, input file, or implementation changed; use a new output directory")
    elif any(output.iterdir()):
        raise ValueError("new output directory must be empty")
    lock = output / "run.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise ValueError("run.lock exists: another process may be running; inspect status before removing a stale lock") from error
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(str(os.getpid()))
        write_json(invocation, {"fingerprint": fingerprint, "configuration": configuration, "input_file_hashes": inputs})
        worker_config = output / "worker_config.json"
        write_json(worker_config, configuration)
        status(output, "starting", command=configuration["command"], budget_seconds=configuration["budget_seconds"], supervisor_pid=os.getpid())
        environment = dict(os.environ)
        environment.update({name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
        environment["MKL_THREADING_LAYER"] = "SEQUENTIAL"
        environment["PYTHONUNBUFFERED"] = "1"
        print(f"Starting {configuration['command']}; maximum {configuration['budget_seconds']} seconds; output {output}", flush=True)
        with (output / "worker.log").open("a", encoding="utf-8") as log:
            child = subprocess.Popen([sys.executable, str(WORKSPACE / "run.py"), "_worker", str(worker_config)], cwd=REPO,
                                     env=environment, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=os.name != "nt",
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            try:
                remaining = max(.01, configuration["budget_seconds"] - (time.monotonic() - started))
                exit_code = child.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                stop_worker(child)  # Also terminates source-inspection subprocesses belonging to this worker.
                status(output, "deadline_terminated", message="worker terminated at deadline; partial coverage only")
                write_text(output / "INCOMPLETE.md", "# Deadline reached\n\nThe supervised worker was terminated. Existing reports may be partial or from an earlier invocation. "
                           "Consult status.json and manifests; no completed matrix or source interpretation is implied.\n")
                exit_code = 2
            except BaseException:
                stop_worker(child)
                status(output, "interrupted", message="supervisor interrupted; worker terminated")
                raise
        latest = read_json(output / "status.json")
        if latest["state"] in ("starting", "running"):
            status(output, "failed", message=f"worker exited without a final outcome (code {exit_code})")
        current_state = read_json(output / "status.json")["state"]
        status(output, current_state, invocation_wall_seconds=time.monotonic() - started)
        print(json.dumps(read_json(output / "status.json")), flush=True)
        return exit_code
    finally:
        lock.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "_worker":
        return execute(read_json(Path(arguments[1])))
    args = parser().parse_args(arguments)
    try:
        if args.command == "status":
            print(json.dumps(read_json(checked_output(Path(args.output)) / "status.json"), indent=2))
            return 0
        return supervise(vars(args))
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

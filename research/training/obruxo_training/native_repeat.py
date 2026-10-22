"""Supervise one costed repeat after an external process interruption."""

from collections import Counter
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import sys
from time import sleep, time

from obruxo_data.hashing import canonical_sha256, sha256_bytes

from .audit import write_json
from .benchmark import MemoryMonitor


def digest(path):
    return sha256_bytes(Path(path).read_bytes())


def archive_bytes(path):
    return sum(item.stat().st_size for item in Path(path).rglob("*") if item.is_file())


def interrupted_history(source):
    import torch
    source = Path(source)
    raw = (source / "renders.jsonl").read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("preserve and diagnose a partially written native ledger before repeating")
    rows = [json.loads(line) for line in raw.splitlines()]
    attempts = {(row["phase"], row["attempt"]): row for row in rows if row["event"] == "attempt"}
    outcomes = {(row["phase"], row["attempt"]): row for row in rows if row["event"] == "outcome"}
    if (len(attempts) != sum(row["event"] == "attempt" for row in rows) or not outcomes.keys() <= attempts.keys()
            or len(outcomes) != sum(row["event"] == "outcome" for row in rows)):
        raise ValueError("interrupted native ledger contains duplicate or unrelated receipts")
    saved = torch.load(source / "checkpoint.pt", map_location="cpu", weights_only=True)
    plan = json.loads((source / "plan.json").read_text())
    if saved["plan_id"] != canonical_sha256(plan) or saved["step"] != len(saved["trace"]):
        raise ValueError("interrupted checkpoint does not belong to its frozen plan")
    return {"native_attempts": len(attempts), "native_outcomes": len(outcomes),
            "outcome_statuses": dict(Counter(row["status"] for row in outcomes.values())),
            "unobserved_attempts": [row for key, row in attempts.items() if key not in outcomes],
            "committed_updates_per_arm": saved["step"],
            "charged_optimizer_attempts": len((source / "updates.jsonl").read_text().splitlines()),
            "archive_bytes": archive_bytes(source), "qualified_for_release": False,
            "evidence_sha256": {name: digest(source / name) for name in
                ("plan.json", "run.json", "checkpoint.pt", "renders.jsonl", "updates.jsonl", "measurements.jsonl")}}


def prepare_repeat(source, output, command, *, working_directory, pythonpath):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or output == source or source in output.parents or output in source.parents:
        raise ValueError("repeat evidence needs a distinct new directory")
    training = json.loads((source / "plan.json").read_text())
    driver = Path(__file__).parents[1] / "reports/scripts/run_balanced_native_fit.py"
    flags = {"--pool", "--initial", "--retrieval", "--qualification", "--input-cache", "--category-calibration", "--output"}
    if (len(command) != 17 or Path(command[0]).resolve() != Path(sys.executable).resolve()
            or Path(command[1]).resolve() != driver.resolve() or command[-1] != "--execute" or set(command[2:-1:2]) != flags):
        raise ValueError("repeat must invoke the existing frozen native fit exactly once")
    inputs = dict(zip(command[2:-1:2], command[3:-1:2], strict=True))
    if Path(inputs["--output"]).resolve() != output:
        raise ValueError("native child output differs from its repeat evidence directory")
    expected_inputs = {"--initial": training["initial_file_sha256"], "--retrieval": training["retrieval_file_sha256"],
                       "--qualification": training["qualification_file_sha256"], "--input-cache": training["input_cache_sha256"],
                       "--category-calibration": training["category_training"]["diagnosis_file_sha256"]}
    if any(digest(inputs[flag]) != expected for flag, expected in expected_inputs.items()):
        raise ValueError("repeat inputs differ from the frozen comparison")
    history = interrupted_history(source)
    if (training["format"] != "matched_native_inverse_fit_v4" or (source / "summary.json").exists()
            or history["outcome_statuses"] != {"accepted": history["native_outcomes"]}
            or len(history["unobserved_attempts"]) != 1 or history["archive_bytes"] > 1024**3):
        raise ValueError("this repeat requires one externally interrupted run with otherwise accepted observations")
    new_attempts = sum(training["limits"][name] for name in ("training_attempts", "qualification_attempts"))
    reserve = 64 * 1024**2
    if history["archive_bytes"] + training["storage_bound"]["estimated_archive_upper_bound_bytes"] + reserve > 5 * 1024**3:
        raise ValueError("retained and repeated evidence exceeds the aggregate archive cap")
    wrapper = Path(__file__).parents[1] / "reports/scripts/supervise_native_repeat.py"
    windows = wrapper.with_name("start_native_repeat.ps1")
    files = [Path(__file__), Path(__file__).with_name("benchmark.py"), wrapper, windows]
    plan = {"format": "externally_interrupted_native_repeat_v1", "source": str(source), "output": str(output),
            "training_plan_id": canonical_sha256(training), "training_plan_file_sha256": digest(source / "plan.json"),
            "prior_run": history, "command": list(command), "working_directory": str(Path(working_directory).resolve()),
            "pythonpath": str(pythonpath), "started_unix": json.loads((source / "run.json").read_text())["started_unix"],
            "limits": {"native_attempts": history["native_attempts"] + new_attempts,
                       "optimizer_attempts_per_arm": history["charged_optimizer_attempts"] + training["limits"]["optimizer_attempts"],
                       "wall_seconds": training["limits"]["wall_seconds"], "process_tree_bytes": training["limits"]["process_tree_bytes"],
                       "aggregate_archive_bytes": 5 * 1024**3},
            "source_sha256_lf": {path.name: sha256_bytes(path.read_bytes().replace(b"\r\n", b"\n")) for path in files},
            "model_selection": "final_only_unchanged_training_and_quality_gates", "automatic_repeats": 0, "test_split_used": False}
    output.mkdir(parents=True)
    (output / "plan.json").write_bytes((source / "plan.json").read_bytes())
    write_json(output / "repeat-plan.json", {"plan_id": canonical_sha256(plan), "plan": plan})
    return plan


class AttemptCounter:
    def __init__(self, path, *, event="attempt"):
        self.path, self.offset, self.pending, self.attempts, self.event = Path(path), 0, b"", 0, event

    def poll(self):
        if not self.path.exists():
            return self.attempts
        if self.path.stat().st_size < self.offset:
            raise ValueError("supervised append-only ledger was truncated")
        with self.path.open("rb") as handle:
            handle.seek(self.offset)
            raw = handle.read()
            self.offset = handle.tell()
        parts = (self.pending + raw).split(b"\n")
        self.pending = parts.pop()
        self.attempts += sum(self.event is None or json.loads(line).get("event") == self.event for line in parts if line)
        return self.attempts


def resource_usage(plan, monitor, native, optimizer):
    return {"native_attempts_including_interrupted_run": plan["prior_run"]["native_attempts"] + native.poll(),
            "optimizer_attempts_per_arm_including_interrupted_run": plan["prior_run"]["charged_optimizer_attempts"] + optimizer.poll(),
            "elapsed_seconds_including_interrupted_run": time() - plan["started_unix"],
            "peak_process_tree_rss_bytes": monitor.peak_bytes, "rss_sampling_errors": monitor.errors,
            "aggregate_archive_bytes": archive_bytes(plan["source"]) + archive_bytes(plan["output"])}


def require_budget(usage, limits):
    if (usage["native_attempts_including_interrupted_run"] > limits["native_attempts"]
            or usage["optimizer_attempts_per_arm_including_interrupted_run"] > limits["optimizer_attempts_per_arm"]
            or usage["elapsed_seconds_including_interrupted_run"] >= limits["wall_seconds"]
            or (usage["peak_process_tree_rss_bytes"] or 0) > limits["process_tree_bytes"] or usage["rss_sampling_errors"]
            or usage["aggregate_archive_bytes"] + 1024**2 > limits["aggregate_archive_bytes"]):
        raise RuntimeError("aggregate repeat resource limit reached")


def supervise_repeat(path):
    import psutil
    path = Path(path).resolve()
    saved = json.loads(path.read_text())
    plan, output, process = saved["plan"], path.parent, None
    source = Path(plan["source"])
    if (saved["plan_id"] != canonical_sha256(plan) or str(output) != plan["output"] or (output / "supervisor-run.json").exists()
            or digest(output / "plan.json") != plan["training_plan_file_sha256"]):
        raise ValueError("repeat plan changed or this one allowed repeat already started")
    for name, expected in plan["prior_run"]["evidence_sha256"].items():
        if digest(source / name) != expected:
            raise ValueError("interrupted evidence changed")
    directory = Path(__file__).parent
    for name, expected in plan["source_sha256_lf"].items():
        current = directory / name if (directory / name).exists() else directory.parent / "reports/scripts" / name
        if sha256_bytes(current.read_bytes().replace(b"\r\n", b"\n")) != expected:
            raise ValueError("frozen repeat supervisor source changed")
    for candidate in psutil.process_iter(["pid", "cmdline"]):
        command = candidate.info["cmdline"] or []
        if any("run_balanced_native_fit.py" in argument for argument in command) and str(output) in command:
            raise ValueError("a live process already owns this repeat")
    limits, monitor, failure = plan["limits"], MemoryMonitor(), None
    counter = AttemptCounter(output / "renders.jsonl")
    optimizer = AttemptCounter(output / "updates.jsonl", event=None)
    monitor.start()
    try:
        require_budget(resource_usage(plan, monitor, counter, optimizer), limits)
        with ExitStack() as handles:
            stdout = handles.enter_context((output / "native.stdout.log").open("xb"))
            stderr = handles.enter_context((output / "native.stderr.log").open("xb"))
            environment = {**os.environ, "PYTHONPATH": plan["pythonpath"]}
            process = subprocess.Popen(plan["command"], cwd=plan["working_directory"], env=environment, stdout=stdout, stderr=stderr,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            write_json(output / "supervisor-run.json", {"plan_id": saved["plan_id"], "pid": os.getpid(), "child_pid": process.pid, "started_unix": time()})
            while process.poll() is None:
                require_budget(resource_usage(plan, monitor, counter, optimizer), limits)
                sleep(5)
            if process.returncode != 0:
                raise RuntimeError(f"native child exited with code {process.returncode}")
            completed = json.loads((output / "summary.json").read_text())
            if (not completed["complete"] or completed["failure"] is not None or completed["plan_id"] != plan["training_plan_id"]
                    or completed["committed_updates_per_arm"] != 256):
                raise ValueError("native child did not complete the frozen comparison")
            require_budget(resource_usage(plan, monitor, counter, optimizer), limits)
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        if process is not None and process.poll() is None:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, check=False)
            else:
                process.terminate()
            process.wait(timeout=20)
        monitor.close()
        write_json(output / "supervisor-summary.json", {"plan_id": saved["plan_id"], "failure": failure,
                   "child_exit_code": process.returncode if process is not None else None,
                   **resource_usage(plan, monitor, counter, optimizer), "qualified_for_release": False})

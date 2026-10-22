import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from obruxo_data.hashing import canonical_sha256
from obruxo_training.native_repeat import AttemptCounter, digest, prepare_repeat, require_budget
from obruxo_training import native_repeat


def interrupted(tmp_path):
    source, output = tmp_path / "interrupted", tmp_path / "repeat"
    source.mkdir()
    files = {flag: tmp_path / name for flag, name in (("--pool", "pool.json"), ("--initial", "initial.pt"),
             ("--retrieval", "retrieval.pt"), ("--qualification", "qualification.json"), ("--input-cache", "cache.pt"),
             ("--category-calibration", "calibration.json"))}
    for path in files.values():
        path.write_bytes(path.name.encode())
    plan = {"format": "matched_native_inverse_fit_v4", "initial_file_sha256": digest(files["--initial"]),
            "retrieval_file_sha256": digest(files["--retrieval"]), "qualification_file_sha256": digest(files["--qualification"]),
            "input_cache_sha256": digest(files["--input-cache"]),
            "category_training": {"diagnosis_file_sha256": digest(files["--category-calibration"])},
            "limits": {"training_attempts": 5840, "qualification_attempts": 1280, "optimizer_attempts": 260,
                       "wall_seconds": 21600, "process_tree_bytes": 8 * 1024**3},
            "storage_bound": {"estimated_archive_upper_bound_bytes": 2724438911}}
    (source / "plan.json").write_text(json.dumps(plan))
    (source / "run.json").write_text(json.dumps({"started_unix": 1000.}))
    torch.save({"plan_id": canonical_sha256(plan), "step": 1, "trace": [{"step": 1}]}, source / "checkpoint.pt")
    rows = [{"phase": "training", "attempt": 0, "event": "attempt"},
            {"phase": "training", "attempt": 0, "event": "outcome", "status": "accepted"},
            {"phase": "training", "attempt": 1, "event": "attempt"}]
    (source / "renders.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (source / "updates.jsonl").write_text('{}\n{}\n')
    (source / "measurements.jsonl").write_text('{}\n')
    driver = Path(__file__).parents[1] / "reports/scripts/run_balanced_native_fit.py"
    command = [sys.executable, str(driver), *[value for flag, path in files.items() for value in (flag, str(path))], "--output", str(output), "--execute"]
    return source, output, files, command


def test_repeat_counts_interrupted_compute_without_changing_the_training_plan(tmp_path):
    source, output, _, command = interrupted(tmp_path)
    before = {path.name: path.read_bytes() for path in source.iterdir()}
    result = prepare_repeat(source, output, command, working_directory=tmp_path, pythonpath="fixture")
    assert result["limits"]["native_attempts"] == 7122 and result["limits"]["optimizer_attempts_per_arm"] == 262
    assert result["started_unix"] == 1000. and result["automatic_repeats"] == 0
    assert result["prior_run"]["native_outcomes"] == 1 and len(result["prior_run"]["unobserved_attempts"]) == 1
    assert (output / "plan.json").read_bytes() == before["plan.json"]
    assert all(path.read_bytes() == before[path.name] for path in source.iterdir())


@pytest.mark.parametrize("change", ("different_input", "different_output", "extra_command", "rejected_audio"))
def test_repeat_rejects_changed_inputs_commands_or_audio_failures(tmp_path, change):
    source, output, files, command = interrupted(tmp_path)
    if change == "different_input":
        files["--initial"].write_bytes(b"changed")
    elif change == "different_output":
        command[-2] = str(source)
    elif change == "extra_command":
        command.append("--resume")
    else:
        raw = (source / "renders.jsonl").read_text().replace('"accepted"', '"rejected"')
        (source / "renders.jsonl").write_text(raw)
    with pytest.raises(ValueError):
        prepare_repeat(source, output, command, working_directory=tmp_path, pythonpath="fixture")
    assert not output.exists()


def test_incremental_attempt_counter_waits_for_complete_lines(tmp_path):
    path = tmp_path / "events.jsonl"
    counter = AttemptCounter(path)
    assert counter.poll() == 0
    path.write_bytes(b'{"event":"attempt"}\n{"event":"out')
    assert counter.poll() == 1
    with path.open("ab") as handle:
        handle.write(b'come"}\n{"event":"attempt"}\n')
    assert counter.poll() == 2 and counter.poll() == 2 and counter.pending == b""
    path.write_bytes(b"short")
    with pytest.raises(ValueError, match="truncated"):
        counter.poll()


@pytest.mark.parametrize("changed", ("native", "optimizer", "wall", "memory", "archive", "sampling"))
def test_aggregate_caps_include_the_interrupted_run(changed):
    limits = {"native_attempts": 7803, "optimizer_attempts_per_arm": 290, "wall_seconds": 21600,
              "process_tree_bytes": 8 * 1024**3, "aggregate_archive_bytes": 5 * 1024**3}
    usage = {"native_attempts_including_interrupted_run": 7000, "optimizer_attempts_per_arm_including_interrupted_run": 286,
             "elapsed_seconds_including_interrupted_run": 18000, "peak_process_tree_rss_bytes": 2 * 1024**3,
             "aggregate_archive_bytes": 4 * 1024**3, "rss_sampling_errors": 0}
    require_budget(usage, limits)
    key, cap = {"native": ("native_attempts_including_interrupted_run", "native_attempts"),
                "optimizer": ("optimizer_attempts_per_arm_including_interrupted_run", "optimizer_attempts_per_arm"),
                "wall": ("elapsed_seconds_including_interrupted_run", "wall_seconds"),
                "memory": ("peak_process_tree_rss_bytes", "process_tree_bytes"),
                "archive": ("aggregate_archive_bytes", "aggregate_archive_bytes"), "sampling": ("rss_sampling_errors", None)}[changed]
    usage[key] = limits[cap] + 1 if cap else 1
    with pytest.raises(RuntimeError, match="resource limit"):
        require_budget(usage, limits)


@pytest.mark.parametrize("complete", (True, False))
def test_supervisor_records_clean_completion_and_rejects_incomplete_children(tmp_path, monkeypatch, complete):
    import psutil
    source, output, _, command = interrupted(tmp_path)
    plan = prepare_repeat(source, output, command, working_directory=tmp_path, pythonpath="fixture")
    monkeypatch.setattr(native_repeat, "time", lambda: 1001.)
    monkeypatch.setattr(psutil, "process_iter", lambda *_args: [])
    monkeypatch.setattr(native_repeat, "MemoryMonitor", lambda: SimpleNamespace(peak_bytes=4096, errors=0, start=lambda: None, close=lambda: None))

    class Child:
        pid, returncode = 9999999, 0

        def __init__(self, *_args, **_kwargs):
            (output / "summary.json").write_text(json.dumps({"complete": complete, "failure": None,
                "plan_id": plan["training_plan_id"], "committed_updates_per_arm": 256}))

        def poll(self):
            return self.returncode

    monkeypatch.setattr(native_repeat.subprocess, "Popen", Child)
    if complete:
        native_repeat.supervise_repeat(output / "repeat-plan.json")
    else:
        with pytest.raises(ValueError, match="did not complete"):
            native_repeat.supervise_repeat(output / "repeat-plan.json")
    saved = json.loads((output / "supervisor-summary.json").read_text())
    assert (saved["failure"] is None) is complete
    assert saved["native_attempts_including_interrupted_run"] == 2 and saved["qualified_for_release"] is False


@pytest.mark.parametrize("new_work", (False, True))
def test_prelaunch_failures_are_retained_but_cannot_hide_native_work(tmp_path, new_work):
    source, output, _, command = interrupted(tmp_path)
    prior = tmp_path / "prelaunch"
    prior.mkdir()
    (prior / "plan.json").write_bytes((source / "plan.json").read_bytes())
    (prior / "supervisor-summary.json").write_text(json.dumps({"failure": "environment setup failed",
        "native_attempts_including_interrupted_run": 2, "optimizer_attempts_per_arm_including_interrupted_run": 2}))
    if new_work:
        (prior / "run.json").write_text('{}')
        with pytest.raises(ValueError, match="new training or rendering"):
            prepare_repeat(source, output, command, working_directory=tmp_path, pythonpath="fixture", prelaunches=[prior])
    else:
        plan = prepare_repeat(source, output, command, working_directory=tmp_path, pythonpath="fixture", prelaunches=[prior])
        assert len(plan["prelaunches"]) == 1 and plan["prelaunches"][0]["archive_bytes"] > 0

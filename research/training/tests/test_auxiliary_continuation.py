import copy
from datetime import datetime, timezone
import json
import shutil
from time import time

import psutil
import pytest
import torch

from obruxo_data.hashing import canonical_sha256
from obruxo_training import auxiliary_continuation as recovery
from obruxo_training import basic_direct_fit as fit
from obruxo_training.audit import write_json
from obruxo_training.basic_direct_fit_audit import verify
from obruxo_training.basic_direct_fit_data import storage_bound
from obruxo_training.native_fit import digest
from obruxo_training.paired_native_audit import equal_tree
from test_basic_direct_fit import FakeSession, completed  # noqa: F401


@pytest.fixture(scope="module")
def stopped(completed, tmp_path_factory):  # noqa: F811
    inputs, full = completed
    source = tmp_path_factory.mktemp("known-timeout-source")
    parent = inputs.parent
    write_json(parent / "fit_cases.json", inputs.cases)
    features = source.parent / "fixture-features.pt"
    features.write_bytes(b"fixture identity; inputs are provided by the simulation")
    plan = {**copy.deepcopy(inputs.plan), "source": {}, "initial_file_sha256": digest(inputs.initial_path),
            "input_cache_sha256": digest(features), "input_files": {"parent/fit_cases.json": digest(parent / "fit_cases.json")},
            "storage_bound": storage_bound(inputs.plan["fit_audit_cells"], inputs.codec.contract.tail_seconds)}
    plan_id = canonical_sha256(plan)
    write_json(source / "plan.json", plan)
    saved = torch.load(full / "prefix-checkpoint.pt", map_location="cpu", weights_only=True)
    saved = recovery.rebind_checkpoint(saved, plan_id)
    torch.save(saved, source / "checkpoint.pt")
    torch.save(saved, source / "prefix-checkpoint.pt")
    prefix = json.loads((full / "prefix-verification.json").read_text(encoding="utf-8"))
    write_json(source / "prefix-verification.json", {**prefix, "plan_id": plan_id, "checkpoint_file_sha256": digest(source / "prefix-checkpoint.pt")})
    probes = [row for row in fit.lines(full / "probes.jsonl") if row["query"]["step"] <= 2]
    next_probe = next(row for row in fit.lines(full / "probes.jsonl") if row["query"]["step"] == 3)
    events = [row for row in fit.lines(full / "renders.jsonl") if row["phase"] == "training" and int(row["label"].split(":")[1]) <= 2]
    next_attempt = next(row for row in fit.lines(full / "renders.jsonl") if row["event"] == "attempt" and row["label"] == next_probe["receipt"]["label"])
    failed = {key: value for key, value in next_probe["receipt"].items() if key not in ("qa", "retained_audio")}
    failed.update(status="error", error_type="RenderTimeout", wall_seconds=18001.)
    events += [next_attempt, failed]
    for name, rows in (("probes.jsonl", probes), ("renders.jsonl", events), ("updates.jsonl", fit.lines(full / "updates.jsonl")[:3])):
        (source / name).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    for folder in ("source_audio", "comparison_audio"):
        shutil.copytree(full / folder, source / folder)
    original_clock = time() - 18001
    write_json(source / "run.json", {"plan_id": plan_id, "started_unix": original_clock})
    (source / "executions.jsonl").write_text(json.dumps({"ordinal": 1, "plan_id": plan_id, "resume": False, "started_unix": original_clock}) + "\n", encoding="utf-8")
    write_json(source / "launch.json", {"ProcessId": 2147483647, "CreationDate": "2000-01-01T00:00:00Z"})
    write_json(source / "service-001-completion.json", {"exit_code": 1, "failure": "simulated terminal timeout"})
    write_json(source / "summary.json", {"plan_id": plan_id, "complete": False, "failure": "RuntimeError: direct sound probe was not accepted",
        "result": None, "elapsed_seconds": 18001., "peak_process_tree_rss_bytes": 1024**3, "rss_sampling_errors": 0,
        "committed_updates_per_arm": 2, "render_attempts": {"training": len(events) // 2, "qualification": 0}, "within_resource_limits": False})
    return inputs, full, source, features


def prepare(stopped, output):
    inputs, _full, original_source, features = stopped
    source = output.parent / "source"
    shutil.copytree(original_source, source)
    plan = recovery.make_plan(source, inputs.parent, inputs.initial_path.parent, features, digest(source / "plan.json"))
    output.mkdir()
    write_json(output / "plan.json", plan)
    recovery.migrate(source, output, plan)
    recovery.validate_migration(source, output, plan)
    current = copy.copy(inputs)
    current.plan = plan
    return current, source, plan


def test_continuation_changes_only_operational_metadata_and_preserves_total_caps(stopped, tmp_path):
    inputs, source, plan = prepare(stopped, tmp_path / "continued")
    original = recovery.read(source / "plan.json")
    for name, value in original.items():
        if name not in ("format", "source", "limits", "storage_bound"):
            assert plan[name] == value
    assert plan["limits"]["wall_seconds"] == 86400
    assert plan["limits"]["archive_bytes"] + plan["operational_continuation"]["original_archive_bytes"] == original["limits"]["archive_bytes"]
    for name in ("optimizer_attempts", "native_executions", "training_attempts", "qualification_attempts"):
        assert plan["limits"][name] == original["limits"][name]
    assert plan["operational_continuation"]["new_epochs"] == 0
    assert not plan["operational_continuation"]["original_budget_passed"]
    assert inputs.plan["updates"] == original["updates"]


def test_known_timeout_recovers_identical_models_adam_rng_and_keeps_all_charges(stopped, tmp_path, monkeypatch):
    inputs, source, plan = prepare(stopped, tmp_path / "continued")
    old_files = recovery.manifest(source)
    output = tmp_path / "continued"
    original_policy = fit.charge_execution
    monkeypatch.setattr(fit, "PairedSession", FakeSession)
    with recovery.continuation_charge(source, output, plan):
        result = fit.execute(inputs, output, resume=True)
    assert fit.charge_execution is original_policy and result["complete"] and result["within_resource_limits"]
    recovery.validate_migration(source, output, plan)
    audit = verify(inputs, output)
    assert audit["offline_training_replay_exact"] and len(audit["recovered_timeouts"]) == 1
    current = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
    full = torch.load(stopped[1] / "checkpoint.pt", map_location="cpu", weights_only=True)
    for name in ("models", "optimizers", "rng"):
        assert equal_tree(current[name], full[name])
    assert len(fit.lines(output / "updates.jsonl")) == 5 and len(fit.lines(output / "executions.jsonl")) == 2
    assert recovery.manifest(source) == old_files
    assert recovery.read(output / "run.json")["started_unix"] == recovery.read(source / "run.json")["started_unix"]
    with pytest.raises(ValueError, match="already been spent"):
        with recovery.continuation_charge(source, output, plan):
            fit.charge_execution(inputs, output, True)
    assert fit.charge_execution is original_policy


@pytest.mark.parametrize("kind", ["starting_state", "ledger"])
def test_migrated_state_or_old_ledger_prefix_cannot_be_changed(stopped, tmp_path, kind):
    _inputs, source, plan = prepare(stopped, tmp_path / "continued")
    output = tmp_path / "continued"
    if kind == "starting_state":
        saved = torch.load(output / "starting-checkpoint.pt", map_location="cpu", weights_only=True)
        saved["models"]["spectral"]["correction.2.bias"][0] += .1
        torch.save(saved, output / "starting-checkpoint.pt")
    else:
        path = output / "probes.jsonl"
        path.write_bytes(path.read_bytes().split(b"\n", 1)[1])
    with pytest.raises(ValueError, match="differs|prefix"):
        recovery.validate_migration(source, output, plan)


def test_live_original_is_refused_before_any_recovery(stopped, tmp_path):
    source = tmp_path / "live-original"
    shutil.copytree(stopped[2], source)
    current = psutil.Process()
    write_json(source / "launch.json", {"ProcessId": current.pid, "CreationDate": datetime.fromtimestamp(current.create_time(), timezone.utc).isoformat()})
    with pytest.raises(ValueError, match="still live"):
        recovery.require_terminal(source)


def test_original_audio_rejection_cannot_be_relabelled_as_a_clock_timeout(stopped, tmp_path):
    source = tmp_path / "audio-rejection"
    shutil.copytree(stopped[2], source)
    rows = fit.lines(source / "renders.jsonl")
    rows[-1].update(error_type="AudioQAError", qa={"finite": False})
    (source / "renders.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="known native timeout"):
        recovery.require_timeout(source, recovery.read(source / "plan.json"))


def test_expired_total_window_refuses_entry_without_spending_an_execution(stopped, tmp_path, monkeypatch):
    inputs, source, plan = prepare(stopped, tmp_path / "continued")
    output = tmp_path / "continued"
    start = recovery.read(output / "run.json")["started_unix"]
    monkeypatch.setattr(recovery, "time", lambda: start + recovery.TOTAL_WALL_SECONDS + 1)
    original_policy = fit.charge_execution
    with pytest.raises(ValueError, match="window has expired"):
        with recovery.continuation_charge(source, output, plan):
            fit.charge_execution(inputs, output, True)
    assert fit.charge_execution is original_policy and len(fit.lines(output / "executions.jsonl")) == 1

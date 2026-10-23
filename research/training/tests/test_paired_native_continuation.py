import copy
from contextlib import contextmanager
import json
from pathlib import Path
from time import time

import pytest

from obruxo_training import paired_native_continuation as continuation


def policy():
    plan = {"limits": {"wall_seconds": 21600, "archive_bytes": 4 * 1024**3, "process_tree_bytes": 8 * 1024**3,
                       "training_attempts": 5840, "qualification_attempts": 1280, "optimizer_attempts": 132}}
    started = time() - 30000
    manifest = {"limits": continuation.continuation_limits(plan), "deadline_unix": started + 86400,
                "original_started_unix": started, "numerical_plan_id": "frozen", "operational_archive_reserve_bytes": 1024**2}
    return plan, manifest


def test_scoped_budget_preserves_numerical_plan_and_original_clock(monkeypatch):
    plan, manifest = policy()
    before = copy.deepcopy(plan)
    class Session:
        def __init__(self, value):
            self.plan, self.started = value, manifest["original_started_unix"]
    class Records:
        def __init__(self, value):
            self.plan = value
    monkeypatch.setattr(continuation.fit, "PairedSession", Session)
    monkeypatch.setattr(continuation.fit, "FitRecords", Records)
    with continuation.operational_budget(plan, manifest):
        session = continuation.fit.PairedSession(plan)
        assert session.started == manifest["original_started_unix"]
        assert session.plan["limits"]["wall_seconds"] == 86400
        assert session.plan["limits"]["archive_bytes"] == 5 * 1024**3 - 1024**2
        records = continuation.fit.FitRecords(plan)
        assert records.plan["limits"] == session.plan["limits"]
        for key in ("training_attempts", "qualification_attempts", "optimizer_attempts", "process_tree_bytes"):
            assert session.plan["limits"][key] == before["limits"][key]
    assert continuation.fit.PairedSession is Session
    assert continuation.fit.FitRecords is Records
    assert plan == before


@pytest.mark.parametrize("change", ["deadline", "native_limit"])
def test_rejects_changed_or_expired_declaration(change):
    plan, manifest = policy()
    if change == "deadline":
        manifest["deadline_unix"] = time() - 1
    else:
        manifest["limits"]["training_attempts"] += 1
    with pytest.raises(ValueError, match="limits changed|deadline expired"):
        with continuation.operational_budget(plan, manifest):
            pytest.fail("an invalid continuation entered native execution")


def completed_summary():
    return {"elapsed_seconds": 30000, "rss_sampling_errors": 0, "peak_process_tree_rss_bytes": 1024**3,
            "archive_bytes_before_summary": 1024**3, "render_attempts": {"training": 5500, "qualification": 800},
            "complete": True, "failure": None, "fit_result": {"pass": True}, "validation_result": {"pass": True},
            "native_receipts_resolved": True, "incumbent_artifacts_unchanged": True, "within_resource_limits": False}


def test_continuation_never_relabels_the_original_budget_pass():
    _, manifest = policy()
    result = continuation.continuation_decision(completed_summary(), manifest, optimizer_attempts=129, archive_overhead_bytes=1000)
    assert result["matched_comparison_passed_under_continuation"]
    assert not result["original_budget_met"]
    assert not result["qualified_for_release"]


@pytest.mark.parametrize("failure", ["ram", "wall", "native", "archive", "optimizer", "quality", "incomplete"])
def test_unchanged_gates_and_new_resource_ceiling_still_reject(failure):
    _, manifest = policy()
    summary, charges = completed_summary(), 129
    if failure == "ram":
        summary["peak_process_tree_rss_bytes"] = 9 * 1024**3
    elif failure == "wall":
        summary["elapsed_seconds"] = 86400
    elif failure == "native":
        summary["render_attempts"]["training"] = 5841
    elif failure == "archive":
        summary["archive_bytes_before_summary"] = 5 * 1024**3
    elif failure == "optimizer":
        charges = 133
    elif failure == "quality":
        summary["fit_result"]["pass"] = False
    else:
        summary.update(complete=False, failure="RenderTimeout")
    result = continuation.continuation_decision(summary, manifest, optimizer_attempts=charges, archive_overhead_bytes=1000)
    assert not result["matched_comparison_passed_under_continuation"]


@pytest.mark.parametrize("interrupted", [False, True])
def test_continuation_records_completion_and_restores_scopes(tmp_path, monkeypatch, interrupted):
    plan, manifest = policy()
    source, declaration = tmp_path / "source", tmp_path / "declaration"
    source.mkdir()
    declaration.mkdir()
    (source / "plan.json").write_text(json.dumps(plan))
    (source / "run.json").write_text(json.dumps({"started_unix": manifest["original_started_unix"]}))
    (source / "updates.jsonl").write_text("{}\n" * 50)
    manifest.update(maximum_explicit_launches=3, operational_source_sha256=continuation.digest(Path(continuation.__file__)),
                    numerical_plan_file_sha256=continuation.digest(source / "plan.json"),
                    execution_helper_sha256=continuation.digest(Path(continuation.__file__).with_name("local_execution.py")),
                    original_evidence_sha256={"run.json": continuation.digest(source / "run.json")})
    (declaration / "manifest.json").write_text(json.dumps(manifest))
    class Session:
        def __init__(self, value):
            self.plan, self.started = value, manifest["original_started_unix"]
    @contextmanager
    def awake():
        report = {"requested": True, "released": False}
        try:
            yield report
        finally:
            report["released"] = True
    def run_fit(*_inputs, **kwargs):
        assert kwargs == {"execute": True, "resume": True}
        session = continuation.fit.PairedSession(plan)
        assert session.plan["limits"]["wall_seconds"] == 86400
        summary = completed_summary()
        if interrupted:
            summary.update(complete=False, failure="RenderTimeout")
        (source / "summary.json").write_text(json.dumps(summary))
        (source / "updates.jsonl").write_text("{}\n" * 129)
        if interrupted:
            raise RuntimeError("RenderTimeout")
    monkeypatch.setattr(continuation.fit, "PairedSession", Session)
    monkeypatch.setattr(continuation.fit, "run_fit", run_fit)
    monkeypatch.setattr(continuation, "active_computation", awake)
    if interrupted:
        with pytest.raises(RuntimeError, match="RenderTimeout"):
            continuation.run_continuation(source, declaration, object())
    else:
        assert continuation.run_continuation(source, declaration, object())["matched_comparison_passed_under_continuation"]
    result = json.loads((declaration / "completion-001.json").read_text())
    assert result["active_computation_request"]["released"]
    assert result["complete"] != interrupted
    assert continuation.fit.PairedSession is Session
    assert json.loads((source / "plan.json").read_text()) == plan
    assert json.loads((source / "run.json").read_text())["started_unix"] == manifest["original_started_unix"]

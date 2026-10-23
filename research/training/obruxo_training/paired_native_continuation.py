"""Declare a bounded operational continuation without rewriting the numerical plan."""

from contextlib import contextmanager
import json
from pathlib import Path
import shutil
from time import time

from .audit import write_json
from . import paired_native_fit as fit
from .native_fit import digest
from .local_execution import active_computation


def continuation_limits(plan):
    return {**plan["limits"], "wall_seconds": 24 * 3600, "archive_bytes": 5 * 1024**3}


def freeze_continuation(source, output, prefix_audit):
    source, output, prefix_audit = map(Path, (source, output, prefix_audit))
    if output.exists():
        raise ValueError("preserve the existing continuation declaration")
    history = json.loads((source / "summary.json").read_text())
    audit = json.loads(prefix_audit.read_text())
    plan = json.loads((source / "plan.json").read_text())
    started = json.loads((source / "run.json").read_text())["started_unix"]
    if (history["complete"] or history["within_resource_limits"] or not audit["offline_training_replay_exact"]
            or audit["scope"] != "committed_checkpoint_projection_only" or audit["full_interrupted_run_verified"]
            or audit["replayed_updates"] != history["committed_updates_per_arm"] or audit["plan_id"] != history["plan_id"]):
        raise ValueError("continuation requires an audited interrupted checkpoint and an acknowledged original budget failure")
    if any(digest(source / name) != value for name, value in audit["original_evidence_sha256"].items()):
        raise ValueError("interrupted evidence changed after checkpoint replay")
    audit_source = Path(__file__).with_name("paired_interrupted_audit.py")
    if digest(audit_source) != audit["auditor_source_sha256"]:
        raise ValueError("interrupted checkpoint auditor changed after replay")
    limits = continuation_limits(plan)
    if time() >= started + limits["wall_seconds"]:
        raise ValueError("the declared continuation window is already expired")
    charges = len((source / "updates.jsonl").read_text().splitlines())
    events = [json.loads(line) for line in (source / "renders.jsonl").read_text().splitlines()]
    attempts = {(row["phase"], row["attempt"]): row for row in events if row["event"] == "attempt"}
    outcomes = {(row["phase"], row["attempt"]): row for row in events if row["event"] == "outcome"}
    failures = [row for row in outcomes.values() if row["status"] != "accepted"]
    if (attempts.keys() != outcomes.keys() or len(failures) != 1 or failures[0].get("error_type") != "RenderTimeout"
            or "qa" in failures[0] or charges + len(plan["updates"]) - audit["replayed_updates"] > limits["optimizer_attempts"]
            or history["elapsed_seconds"] < plan["limits"]["wall_seconds"]
            or history["rss_sampling_errors"] or not 0 < history["peak_process_tree_rss_bytes"] <= limits["process_tree_bytes"]):
        raise ValueError("only the declared observed timeout may enter this operational continuation")
    manifest = {"format": "paired_native_operational_continuation_v1", "numerical_plan_id": history["plan_id"],
                "declared_unix": time(), "original_started_unix": started, "deadline_unix": started + limits["wall_seconds"],
                "limits": limits, "maximum_explicit_launches": 3, "committed_updates_before_continuation": audit["replayed_updates"],
                "charged_optimizer_attempts_before_continuation": charges, "native_attempts_before_continuation": len(attempts),
                "original_deadline_failed": True, "original_evidence_sha256": audit["original_evidence_sha256"],
                "prefix_audit_sha256": digest(prefix_audit), "numerical_plan_file_sha256": digest(source / "plan.json"),
                "auditor_source_sha256": digest(audit_source),
                "execution_helper_sha256": digest(Path(__file__).with_name("local_execution.py")),
                "operational_source_sha256": digest(Path(__file__)), "operational_archive_reserve_bytes": 1024**2,
                "qualified_for_release": False}
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(prefix_audit, output / "interrupted-prefix-verification.json")
    write_json(output / "manifest.json", manifest)
    return manifest


@contextmanager
def operational_budget(plan, manifest):
    """Scope the session's two operational exceptions to this isolated invocation."""
    limits = continuation_limits(plan)
    if (manifest["limits"] != limits or manifest["deadline_unix"] != manifest["original_started_unix"] + limits["wall_seconds"]
            or time() >= manifest["deadline_unix"]):
        raise ValueError("continuation limits changed or its original-clock deadline expired")
    original, original_records = fit.PairedSession, fit.FitRecords
    scoped_limits = {**limits, "archive_bytes": limits["archive_bytes"] - manifest["operational_archive_reserve_bytes"]}
    def apply_limits(instance):
        if instance.plan != plan:
            raise ValueError("continuation changed its numerical plan")
        instance.plan = {**instance.plan, "limits": scoped_limits}
    class ContinuedSession(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.started != manifest["original_started_unix"]:
                raise ValueError("continuation changed its original start time")
            apply_limits(self)
    class ContinuedRecords(original_records):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            apply_limits(self)
    fit.PairedSession, fit.FitRecords = ContinuedSession, ContinuedRecords
    try:
        yield
    finally:
        fit.PairedSession, fit.FitRecords = original, original_records


def continuation_decision(summary, manifest, *, optimizer_attempts, archive_overhead_bytes):
    limits = manifest["limits"]
    counts = summary["render_attempts"]
    resources = (summary["elapsed_seconds"] < limits["wall_seconds"] and summary["rss_sampling_errors"] == 0
                 and 0 < summary["peak_process_tree_rss_bytes"] <= limits["process_tree_bytes"]
                 and summary["archive_bytes_before_summary"] + archive_overhead_bytes <= limits["archive_bytes"]
                 and optimizer_attempts <= limits["optimizer_attempts"]
                 and counts["training"] <= limits["training_attempts"] and counts["qualification"] <= limits["qualification_attempts"])
    sound = bool(summary["complete"] and summary["failure"] is None and summary["fit_result"] and summary["fit_result"]["pass"]
                 and summary["validation_result"] and summary["validation_result"]["pass"])
    passed = bool(resources and sound and summary["native_receipts_resolved"] and summary["incumbent_artifacts_unchanged"])
    return {"numerical_plan_id": manifest["numerical_plan_id"], "complete": summary["complete"],
            "original_budget_met": summary["within_resource_limits"], "continuation_budget_met": resources,
            "sound_quality_gates_passed": sound, "matched_comparison_passed_under_continuation": passed,
            "charged_optimizer_attempts": optimizer_attempts, "qualified_for_release": False}


def run_continuation(source, declaration, pool, *inputs):
    source, declaration = map(Path, (source, declaration))
    manifest = json.loads((declaration / "manifest.json").read_text())
    plan = json.loads((source / "plan.json").read_text())
    if (digest(Path(__file__)) != manifest["operational_source_sha256"]
            or digest(Path(__file__).with_name("local_execution.py")) != manifest["execution_helper_sha256"]
            or digest(source / "plan.json") != manifest["numerical_plan_file_sha256"]):
        raise ValueError("continuation source or frozen numerical plan changed")
    launches = sorted(declaration.glob("launch-*.json"))
    if len(launches) >= manifest["maximum_explicit_launches"]:
        raise ValueError("no declared continuation launch remains")
    if launches and not (declaration / f"completion-{len(launches):03d}.json").exists():
        raise ValueError("previous continuation is live or needs explicit terminal-process investigation")
    if not launches and any(digest(source / name) != value for name, value in manifest["original_evidence_sha256"].items()):
        raise ValueError("preserve the declared interrupted evidence")
    ordinal = len(launches) + 1
    launch = declaration / f"launch-{ordinal:03d}.json"
    with launch.open("x", encoding="utf-8") as handle:
        json.dump({"started_unix": time(), "manifest_sha256": digest(declaration / "manifest.json")}, handle)
    failure, awake = None, {}
    try:
        with operational_budget(plan, manifest), active_computation() as awake:
            fit.run_fit(pool, *inputs, source, execute=True, resume=True)
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        summary = json.loads((source / "summary.json").read_text())
        charges = len((source / "updates.jsonl").read_text().splitlines())
        overhead = sum(path.stat().st_size for path in declaration.rglob("*") if path.is_file()) + 65536
        decision = continuation_decision(summary, manifest, optimizer_attempts=charges, archive_overhead_bytes=overhead)
        write_json(declaration / f"completion-{ordinal:03d}.json", {"failure": failure, "active_computation_request": awake, **decision})
    return decision


def verify_continuation(source, declaration, pool, *inputs):
    import torch
    from .paired_interrupted_audit import anchor_external_rng
    from .paired_native_audit import run_verification
    source, declaration = map(Path, (source, declaration))
    target = declaration / "verification.json"
    if target.exists():
        raise ValueError("preserve the previous independent continuation audit")
    manifest = json.loads((declaration / "manifest.json").read_text())
    if (digest(Path(__file__)) != manifest["operational_source_sha256"]
            or digest(declaration / "interrupted-prefix-verification.json") != manifest["prefix_audit_sha256"]
            or digest(Path(__file__).with_name("paired_interrupted_audit.py")) != manifest["auditor_source_sha256"]
            or digest(Path(__file__).with_name("local_execution.py")) != manifest["execution_helper_sha256"]):
        raise ValueError("continuation source or interrupted checkpoint proof changed")
    for name, expected in manifest["original_evidence_sha256"].items():
        path = source / name if name in ("plan.json", "run.json", "prefix-checkpoint.pt", "prefix-verification.json") else source / "interruptions/001" / name
        if digest(path) != expected:
            raise ValueError("the original failure was not preserved before continuation")
    with active_computation() as awake:
        anchor = anchor_external_rng(source, torch.load(source / "checkpoint.pt", map_location="cpu", weights_only=True))
        run_verification(source, pool, *inputs)
    proof = json.loads((source / "verification.json").read_text())
    if not proof["verified"] or not proof["offline_training_replay_exact"]:
        raise ValueError("the complete original numerical protocol did not replay")
    charges = len((source / "updates.jsonl").read_text().splitlines())
    overhead = sum(path.stat().st_size for path in declaration.rglob("*") if path.is_file()) + 65536
    decision = continuation_decision(proof, manifest, optimizer_attempts=charges, archive_overhead_bytes=overhead)
    report = {**decision, "independent_audit_complete": True, "original_verification_sha256": digest(source / "verification.json"),
              "manifest_sha256": digest(declaration / "manifest.json"), "original_failure_preserved": True,
              "offline_training_replay_exact": True, "comparison_waveforms_rescored": proof["comparison_waveforms_rescored"]}
    report["external_rng_anchor"] = anchor
    report["active_computation_request"] = awake
    write_json(target, report)
    return report

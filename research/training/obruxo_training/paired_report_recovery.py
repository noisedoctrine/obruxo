"""Reuse final comparisons and audit their declared operational identity explicitly."""

import copy
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from time import perf_counter

import torch

from obruxo_data.hashing import canonical_sha256

from . import category_scale_refresh as refresh
from .audit import write_json
from .benchmark import MemoryMonitor
from . import paired_native_fit as fit
from .features import tensor_digest
from .local_execution import active_computation
from .native_fit import audit_case, digest, predict_candidates, record_features
from .paired_native_continuation import run_continuation, verify_continuation
from .trainer import restore_model


def scoped_plan(plan, manifest):
    limits = {**manifest["limits"], "archive_bytes": manifest["limits"]["archive_bytes"] - manifest["operational_archive_reserve_bytes"]}
    return {**plan, "limits": limits}


def normalized_audit(saved, plan, manifest):
    expected = canonical_sha256(scoped_plan(plan, manifest))
    if saved["contract"]["plan_id"] != expected:
        raise ValueError("cached audit does not belong to the declared operational plan")
    return {**saved, "contract": {**saved["contract"], "plan_id": canonical_sha256(plan)}}


def recovery_sources():
    root = Path(__file__).parent
    return {**{name: root / name for name in ("atomic_io.py", "audit.py", "paired_report_recovery.py")},
            "reports/scripts/run_paired_report_recovery.py": root.parent / "reports/scripts/run_paired_report_recovery.py"}


def require_recovery(source, declaration):
    recovery = json.loads((declaration / "report-recovery.json").read_text())
    if (digest(declaration / "manifest.json") != recovery["operational_manifest_sha256"]
            or digest(source / "checkpoint.pt") != recovery["checkpoint_sha256"]
            or any(digest(source / name) != value for name, value in recovery["immutable_training_evidence"].items())
            or any(digest(path) != recovery["source_sha256"][name] for name, path in recovery_sources().items())):
        raise ValueError("the frozen report-only recovery or final checkpoint changed")
    return recovery


def freeze_report_recovery(source, declaration):
    source, declaration = Path(source), Path(declaration)
    target = declaration / "report-recovery.json"
    if target.exists():
        raise ValueError("preserve the previous report recovery declaration")
    plan = json.loads((source / "plan.json").read_text())
    manifest = json.loads((declaration / "manifest.json").read_text())
    summary = json.loads((source / "summary.json").read_text())
    saved = torch.load(source / "checkpoint.pt", map_location="cpu", weights_only=True)
    if (saved["step"] != len(plan["updates"]) or len(saved["trace"]) != saved["step"] or summary["complete"]
            or not summary["failure"].startswith("PermissionError:") or not summary["native_receipts_resolved"]):
        raise ValueError("report recovery requires finished training and a metadata-only failure")
    caches = [path for path in (source / "audits").glob("fit-*.pt") if not path.stem.endswith(".previous")]
    cached = len(caches)
    remaining = (3 * len(plan["training"]) - cached) * 4 + len(plan["validation"]) * 14
    if summary["render_attempts"]["qualification"] + remaining > manifest["limits"]["qualification_attempts"]:
        raise ValueError("remaining comparisons do not fit the unchanged qualification cap")
    backup = source / "report_recovery"
    backup.mkdir(exist_ok=False)
    partial = source / "fit_cases.json.partial"
    shutil.copy2(partial, backup / partial.name)
    write_json(backup / "partial-evidence.json", {partial.name: digest(partial)})
    report = {"format": "paired_final_report_recovery_v1", "operational_manifest_sha256": digest(declaration / "manifest.json"),
              "numerical_plan_id": canonical_sha256(plan), "operational_audit_id": canonical_sha256(scoped_plan(plan, manifest)),
              "checkpoint_sha256": digest(source / "checkpoint.pt"), "trained_updates": saved["step"],
              "charged_optimizer_attempts": len((source / "updates.jsonl").read_text().splitlines()),
              "cached_comparisons": cached, "qualification_attempts_spent": summary["render_attempts"]["qualification"],
              "maximum_additional_qualification_attempts": remaining, "new_training_updates_allowed": 0,
              "numeric_plan_and_quality_gates_unchanged": True, "preserved_partial_sha256": digest(partial),
              "immutable_training_evidence": {name: digest(source / name) for name in ("checkpoint.pt", "updates.jsonl", "measurements.jsonl", "plan.json", "run.json")},
              "source_sha256": {name: digest(path) for name, path in recovery_sources().items()}}
    write_json(target, report)
    return report


@contextmanager
def report_only_execution():
    original_session, original_fit = fit.PairedSession, fit.fit_pairs
    class ReportSession(original_session):
        def render(self, target, performance, phase, label):
            if phase != "qualification":
                raise RuntimeError("report-only recovery cannot render new training data or gradient probes")
            return super().render(target, performance, phase, label)
    def final_only_fit(models, records, session, output, plan, saved=None, **kwargs):
        if saved is None or saved["step"] != len(plan["updates"]) or len(saved["trace"]) != saved["step"]:
            raise ValueError("report-only recovery refuses an earlier checkpoint or new training")
        return original_fit(models, records, session, output, plan, saved, **kwargs)
    fit.PairedSession, fit.fit_pairs = ReportSession, final_only_fit
    try:
        yield
    finally:
        fit.PairedSession, fit.fit_pairs = original_session, original_fit


def run_saved_comparisons(source, declaration, pool, *inputs):
    source, declaration = Path(source), Path(declaration)
    recovery = require_recovery(source, declaration)
    proof = json.loads((declaration / "cached-fit-recovery.json").read_text())
    if proof["cached_comparisons_reused"] != recovery["cached_comparisons"] or not proof["training_evidence_unchanged"]:
        raise ValueError("cached comparison reuse has not been qualified")
    with report_only_execution():
        result = run_continuation(source, declaration, pool, *inputs)
    require_recovery(source, declaration)
    return result


def check_cached_fit(source, declaration, bootstrap_root):
    source, declaration, bootstrap_root = map(Path, (source, declaration, bootstrap_root))
    target = declaration / "cached-fit-recovery.json"
    if target.exists():
        raise ValueError("preserve the previous cached-fit recovery proof")
    require_recovery(source, declaration)
    plan = json.loads((source / "plan.json").read_text())
    manifest = json.loads((declaration / "manifest.json").read_text())
    saved = torch.load(source / "checkpoint.pt", map_location="cpu", weights_only=True)
    if saved["step"] != len(plan["updates"]):
        raise ValueError("report-only recovery requires the final trained checkpoint")
    files = sorted(path for path in (source / "audits").glob("fit-*.pt") if not path.stem.endswith(".previous"))
    before = {name: digest(source / name) for name in ("checkpoint.pt", "renders.jsonl", "updates.jsonl", "measurements.jsonl")}
    started, monitor = perf_counter(), MemoryMonitor()
    monitor.start()
    try:
        with active_computation(), refresh.cpu_settings():
            initial, _frontend, codec = restore_model(torch.load(bootstrap_root / "fit-000/model.pt", map_location="cpu", weights_only=True))
            models = {"initial": initial.eval()}
            for name, state in saved["models"].items():
                model = copy.deepcopy(initial)
                model.load_state_dict(state)
                models[name] = model.eval()
            identities = {name: tensor_digest(model.state_dict()) for name, model in models.items()}
            class CachedSession:
                def __init__(self):
                    self.plan, self.codec = scoped_plan(plan, manifest), codec
                def render(self, *_args, **_kwargs):
                    raise RuntimeError("cached comparison recovery attempted a new native render")
            records, session = {}, CachedSession()
            for path in files:
                _prefix, index, view = path.stem.split("-")
                index, view = int(index), int(view)
                if index not in records:
                    records[index] = torch.load(source / "records" / f"{index:03d}.pt", map_location="cpu", weights_only=True)
                record, recipe = records[index], plan["training"][index]
                def provider(_audio):
                    return predict_candidates(models, record_features(record, view), codec)
                audit_case(session, source, path.stem, recipe, view, provider, identities, expected_audio=record["audio_sha256"][view])
    finally:
        monitor.close()
    elapsed = perf_counter() - started
    unchanged = before == {name: digest(source / name) for name in before}
    if not unchanged or elapsed >= 600 or monitor.peak_bytes is None or monitor.peak_bytes > 4 * 1024**3 or monitor.errors:
        raise RuntimeError("cached comparison recovery changed training evidence or exceeded its ten-minute/4-GiB allocation")
    report = {"cached_comparisons_reused": len(files), "final_trained_updates": saved["step"], "new_native_requests": 0, "new_optimizer_updates": 0,
              "training_evidence_unchanged": unchanged, "evidence_sha256": before, "elapsed_seconds": elapsed,
              "peak_process_tree_rss_bytes": monitor.peak_bytes, "rss_sampling_errors": monitor.errors, "qualified_for_release": False}
    write_json(target, report)
    return report


def verify_report_recovery(source, declaration, scratch, pool, *inputs):
    source, declaration, scratch = map(Path, (source, declaration, scratch))
    manifest = json.loads((declaration / "manifest.json").read_text())
    plan = json.loads((source / "plan.json").read_text())
    target = declaration / "identity-verification.json"
    if target.exists():
        raise ValueError("preserve the previous identity-bridge verification")
    require_recovery(source, declaration)
    before = {path.relative_to(source).as_posix(): digest(path) for path in (source / "audits").glob("*.pt") if not path.stem.endswith(".previous")}
    source_bytes = sum(path.stat().st_size for path in source.rglob("*") if path.is_file())
    extra_bound = sum((source / name).stat().st_size + 16384 for name in before) + 16 * 1024**2
    operational_bytes = sum(path.stat().st_size for path in declaration.rglob("*") if path.is_file())
    if source_bytes + extra_bound + operational_bytes > manifest["limits"]["archive_bytes"]:
        raise ValueError("temporary identity projection exceeds the unchanged evidence allowance")
    projected = {}
    started, monitor = perf_counter(), MemoryMonitor()
    monitor.start()
    try:
        with active_computation(), refresh.cpu_settings(), TemporaryDirectory(prefix="paired-identity-", dir=scratch) as name:
            root = Path(name).resolve()
            if root.parent != scratch.resolve():
                raise ValueError("identity projection cleanup must stay in the declared scratch directory")
            for path in source.rglob("*"):
                if not path.is_file():
                    continue
                relative = path.relative_to(source)
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                key = relative.as_posix()
                if key in before:
                    audit = torch.load(path, map_location="cpu", weights_only=True)
                    torch.save(normalized_audit(audit, plan, manifest), destination)
                    projected[key] = {"original_sha256": before[key], "projected_sha256": digest(destination)}
                else:
                    destination.hardlink_to(path)
            result = verify_continuation(root, declaration, pool, *inputs)
            proof = declaration / "numerical-verification.json"
            if proof.exists():
                raise ValueError("preserve the previous complete numerical verification")
            shutil.copy2(root / "verification.json", proof)
    finally:
        monitor.close()
    elapsed = perf_counter() - started
    if elapsed >= 1200 or monitor.peak_bytes is None or monitor.peak_bytes > 4 * 1024**3 or monitor.errors:
        raise RuntimeError("complete identity-bridge verification exceeds twenty minutes or four GiB")
    if any(digest(source / key) != value for key, value in before.items()):
        raise RuntimeError("identity projection changed the original cached comparisons")
    require_recovery(source, declaration)
    report = {**result, "original_cache_files_unchanged": True, "numeric_plan_id": canonical_sha256(plan),
              "operational_plan_id": canonical_sha256(scoped_plan(plan, manifest)), "projection": projected,
              "projection_changes": ["audit.contract.plan_id"], "numerical_verification_sha256": digest(proof),
              "identity_bridge_source_sha256": digest(Path(__file__)),
              "identity_verification_resources": {"elapsed_seconds": elapsed, "peak_process_tree_rss_bytes": monitor.peak_bytes,
                                                  "rss_sampling_errors": monitor.errors, "temporary_extra_bytes_upper_bound": extra_bound}}
    write_json(target, report)
    return report

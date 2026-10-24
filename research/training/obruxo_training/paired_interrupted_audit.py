"""Replay a committed checkpoint through an explicitly delimited ledger projection."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import torch

from . import category_scale_refresh as refresh
from .audit import write_json
from .benchmark import MemoryMonitor
from .native_fit import digest
from .paired_native_audit import equal_tree, verify
from .paired_native_fit import make_plan
from .trainer import capture_rng, restore_rng


def anchor_external_rng(source, saved):
    evidence = json.loads((source / "prefix-verification.json").read_text())
    path = source / "prefix-checkpoint.pt"
    if digest(path) != evidence["checkpoint_file_sha256"]:
        raise ValueError("the previously recorded early RNG witness changed")
    witness = torch.load(path, map_location="cpu", weights_only=True)
    if (not evidence["offline_training_replay_exact"] or evidence["plan_id"] != saved["plan_id"] or witness["plan_id"] != saved["plan_id"]
            or witness["step"] != evidence["replayed_updates"] or witness["trace"] != saved["trace"][:witness["step"]]
            or any(not equal_tree(witness["rng"][key], saved["rng"][key]) for key in ("python", "numpy"))):
        raise ValueError("external RNG streams differ from their immutable early witness")
    current = capture_rng("cpu")
    restore_rng({**current, **{key: witness["rng"][key] for key in ("python", "numpy")}}, "cpu")
    return {"checkpoint": path.name, "checkpoint_step": witness["step"], "checkpoint_file_sha256": digest(path),
            "external_streams": ["python", "numpy"], "unchanged_at_saved_checkpoint": True, "torch_replayed_from_seed": True}


def project_committed_prefix(source, destination, saved):
    step = saved["step"]
    if [row["attempt"] for row in saved["trace"]] != list(range(step)):
        raise ValueError("this projection requires an uninterrupted committed prefix")
    raw = (source / "renders.jsonl").read_bytes().splitlines(keepends=True)
    events = [json.loads(line) for line in raw]
    positions = [index for index, row in enumerate(events) if row["event"] == "outcome" and row["status"] == "accepted"
                 and row["label"].startswith(f"gradient:{step - 1}:")]
    if not positions:
        raise ValueError("the final committed update has no accepted native measurements")
    stop = max(positions) + 1
    prefix = events[:stop]
    if any(row["status"] != "accepted" for row in prefix if row["event"] == "outcome"):
        raise ValueError("the projected prefix contains a failed native request")
    (destination / "renders.jsonl").write_bytes(b"".join(raw[:stop]))
    charges = (source / "updates.jsonl").read_bytes().splitlines(keepends=True)
    if [json.loads(line)["attempt"] for line in charges[:step]] != list(range(step)):
        raise ValueError("committed optimizer charges changed")
    (destination / "updates.jsonl").write_bytes(b"".join(charges[:step]))
    measurements = (source / "measurements.jsonl").read_bytes().splitlines(keepends=True)
    (destination / "measurements.jsonl").write_bytes(b"".join(line for line in measurements if json.loads(line)["optimizer_attempt"] < step))
    labels = {row["label"] for row in prefix if row["event"] == "outcome"}
    rows = json.loads((source / "records.json").read_text())
    rows = {index: row for index, row in rows.items() if all(f"source:{index}:{view}" in labels for view in range(3))}
    write_json(destination / "records.json", rows)
    (destination / "records").mkdir()
    for row in rows.values():
        (destination / "records" / row["file"]).hardlink_to(source / "records" / row["file"])
    for name in ("plan.json", "run.json", "checkpoint.pt"):
        (destination / name).hardlink_to(source / name)
    return {"included_render_ledger_rows": stop, "excluded_render_ledger_rows": len(events) - stop,
            "included_optimizer_charges": step, "excluded_optimizer_charges": len(charges) - step,
            "source_record_indices": sorted(map(int, rows)),
            "projected_sha256": {name: digest(destination / name) for name in ("renders.jsonl", "updates.jsonl", "measurements.jsonl", "records.json")}}


def audit_interrupted(source, output, scratch, pool, *inputs):
    source, output, scratch = map(Path, (source, output, scratch))
    if output.exists():
        raise ValueError("preserve the previous interrupted-prefix audit")
    names = ("plan.json", "run.json", "summary.json", "checkpoint.pt", "records.json", "renders.jsonl", "updates.jsonl", "measurements.jsonl",
             "prefix-checkpoint.pt", "prefix-verification.json")
    before = {name: digest(source / name) for name in names}
    history = json.loads((source / "summary.json").read_text())
    if history["complete"] or history["failure"] is None:
        raise ValueError("the source is not an interrupted paired fit")
    started, monitor = perf_counter(), MemoryMonitor()
    monitor.start()
    try:
        with refresh.cpu_settings():
            plan = make_plan(pool, *inputs)
            if plan != json.loads((source / "plan.json").read_text()):
                raise ValueError("the interrupted fit's frozen inputs or source changed")
            saved = torch.load(source / "checkpoint.pt", map_location="cpu", weights_only=True)
            with TemporaryDirectory(prefix="paired-prefix-", dir=scratch) as name:
                if Path(name).resolve().parent != scratch.resolve():
                    raise ValueError("projection cleanup must stay inside its declared scratch directory")
                projection = project_committed_prefix(source, Path(name), saved)
                anchor = anchor_external_rng(source, saved)
                result = verify(Path(name), pool, inputs[0], None, None, None, None, None, prefix_plan=plan)
    finally:
        monitor.close()
    elapsed = perf_counter() - started
    unchanged = before == {name: digest(source / name) for name in names}
    within = elapsed < 1200 and monitor.peak_bytes is not None and monitor.peak_bytes <= 4 * 1024**3 and not monitor.errors
    if not unchanged or not within:
        raise RuntimeError("interrupted-prefix audit changed evidence or exceeded its limits")
    report = {**result, "scope": "committed_checkpoint_projection_only", "full_interrupted_run_verified": False,
              "auditor_source_sha256": digest(Path(__file__)),
              "uncommitted_attempt_admitted": False, "original_run_complete": False,
              "original_within_resource_limits": history["within_resource_limits"], "original_evidence_sha256": before,
              "original_evidence_unchanged": unchanged, "projection": projection,
              "external_rng_anchor": anchor,
              "verification_resources": {"elapsed_seconds": elapsed, "peak_process_tree_rss_bytes": monitor.peak_bytes,
                                         "rss_sampling_errors": monitor.errors, "within_resource_limits": within}}
    write_json(output, report)
    return report

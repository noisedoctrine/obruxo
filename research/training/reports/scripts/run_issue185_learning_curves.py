"""Continue the corrected Issue 183 Residual MLP runs through 8,000 updates."""

# ruff: noqa: E402
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
from time import perf_counter
import xml.etree.ElementTree as ET

import numpy as np
import torch

TRAINING = Path(__file__).resolve().parents[2]
RESEARCH = TRAINING.parent
WORKTREE = TRAINING.parents[1]
MAIN_ROOT = WORKTREE.parent.parent if WORKTREE.parent.name == "tmp" else WORKTREE
sys.path[:0] = [str(RESEARCH / "data_generation"), str(TRAINING)]

from obruxo_data.hashing import canonical_sha256
from obruxo_training.trainer import atomic_torch_save

BASE_RUNNER_PATH = TRAINING / "reports/scripts/run_issue183_flow_matching.py"
BASE_SPEC = importlib.util.spec_from_file_location("issue183_runner_for_issue185", BASE_RUNNER_PATH)
BASE = importlib.util.module_from_spec(BASE_SPEC)
BASE_SPEC.loader.exec_module(BASE)

SOURCE = MAIN_ROOT / "research/training/outputs/issue183-flow-matching-corrected-v2-20261127"
VALIDATION_SOURCE = MAIN_ROOT / "research/training/outputs/issue173-model-selection-active-20261126-v4"
SOURCE_REPORT = TRAINING / "reports/issue183_flow_matching"
ARMS = ("residual_mlp-direct", "residual_mlp-flow")
MILESTONES = (500, 1_000, 2_000, 4_000, 8_000)
RECOVERY_EVERY = 100
FLOW_STEPS = 8
METRICS = (
    "spectral_loss", "octave_band_energy_error_db", "monophonic_octave_energy_error_db",
    "attack_envelope_error_db", "decay_envelope_error_db", "sustain_envelope_error_db",
    "release_envelope_error_db",
)
METRIC_LABELS = {
    "spectral_loss": "Spectral loss",
    "octave_band_energy_error_db": "General octave error (dB)",
    "monophonic_octave_energy_error_db": "MIDI-centered octave error (dB)",
    "attack_envelope_error_db": "Attack envelope error (dB)",
    "decay_envelope_error_db": "Decay envelope error (dB)",
    "sustain_envelope_error_db": "Sustain envelope error (dB)",
    "release_envelope_error_db": "Release envelope error (dB)",
}
COLORS = {"residual_mlp-direct": "#276EF1", "residual_mlp-flow": "#D55E00"}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Progress:
    def __init__(self, output):
        self.output = Path(output)
        self.last_write = 0.0
        self.status = {"phase": "Starting", "arm": "", "completed": 0, "total": 1,
                       "arm_index": 0, "arm_count": len(ARMS), "state": "running", "message": None}

    def update(self, *, phase=None, arm=None, completed=None, total=None, arm_index=None,
               state="running", message=None, force=False):
        for key, value in (("phase", phase), ("arm", arm), ("completed", completed),
                           ("total", total), ("arm_index", arm_index),
                           ("state", state), ("message", message)):
            if value is not None:
                self.status[key] = value
        now = perf_counter()
        if force or now - self.last_write >= 2 or state != "running":
            self.status["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            write_json(self.output / "status.json", self.status)
            if force or state != "running":
                append_jsonl(self.output / "events.jsonl", {
                    "updated_at": self.status["updated_at"], "phase": self.status["phase"],
                    "arm": self.status["arm"], "completed": self.status["completed"],
                    "total": self.status["total"], "state": state, "message": message})
            self.last_write = now


def load_source(output, device, threads, progress):
    frozen = BASE._load_json(SOURCE / "plan.json")
    plan, source_plan_id = frozen.get("plan"), frozen.get("id")
    if (not isinstance(plan, dict) or plan.get("format") != "issue183_direct_vs_flow_matching_v2"
            or plan.get("training", {}).get("updates_per_arm") != 500
            or plan.get("training", {}).get("seed") != 183):
        raise ValueError("source run must be the corrected Issue 183 seed 183, 500-update experiment")

    source_report = BASE._load_json(SOURCE_REPORT / "summary.json")
    expected_hashes = source_report.get("source_checkpoint_sha256", {})
    if set(ARMS) - set(expected_hashes):
        raise ValueError("Issue 183 report is missing a Residual MLP checkpoint hash")
    verified = BASE.verify_completed_run(SOURCE, VALIDATION_SOURCE, expected_arms=ARMS)
    for arm in ARMS:
        if verified["checkpoint_sha256"][arm] != expected_hashes[arm]:
            raise ValueError(f"checkpoint hash does not match the published Issue 183 result: {arm}")

    progress.update(phase="Verifying frozen source data", force=True)
    _, _, train_rows, validation_rows, receipts = BASE.load_sources(
        BASE.TRAINING_SOURCE, VALIDATION_SOURCE, output, progress)
    receipt_path = SOURCE / "data-receipts.jsonl"
    receipt_hash = sha256_file(receipt_path)
    if receipt_hash != frozen.get("data_receipts_sha256"):
        raise ValueError("source cache receipts differ from the frozen Issue 183 plan")

    source_states = {}
    for arm in ARMS:
        saved = torch.load(SOURCE / f"{arm}.pt", map_location="cpu", weights_only=True)
        optimizer_steps = {state["step"].item() for state in saved["optimizer"]["state"].values()}
        if (saved.get("step") != 500 or saved.get("plan_id") != source_plan_id
                or saved.get("arm") != arm or saved.get("seed") != 183 or optimizer_steps != {500.0}):
            raise ValueError(f"source checkpoint is not a resumable 500-update state: {arm}")
        source_states[arm] = saved

    for arm in ARMS:
        expected = source_report["evaluation"][arm]
        if arm.endswith("-flow"):
            expected = expected["8"]
        expected = expected.get("metrics", expected)
        restored = BASE.load_model("residual_mlp", arm.rsplit("-", 1)[1], BASE.SurrogateCodec(
            BASE.PilotCodec(), expanded=True).width).to(device)
        restored.load_state_dict(source_states[arm]["model"])
        restored.eval()
        observed, _ = BASE.evaluate_model(
            restored, arm.rsplit("-", 1)[1], validation_rows,
            VALIDATION_SOURCE / "validation-pairs", BASE.SurrogateCodec(BASE.PilotCodec(), expanded=True),
            device, flow_steps=FLOW_STEPS, seed=183 + 90_000,
            progress=progress, phase=f"Validating published update 500 · {arm}")
        for metric in METRICS:
            if not math.isclose(observed[metric], expected[metric], rel_tol=0, abs_tol=1e-10):
                raise ValueError(f"checkpoint does not reproduce published update-500 {metric}: {arm}")
        del restored
        if device.type == "xpu":
            torch.xpu.empty_cache()

    receipts_output = output / "data-receipts.jsonl"
    if receipts_output.exists():
        if sha256_file(receipts_output) != receipt_hash:
            raise ValueError("existing continuation receipts do not match the source experiment")
    else:
        shutil.copyfile(receipt_path, receipts_output)
    continuation = {
        "format": "issue185_direct_vs_flow_mlp_continuation_v1",
        "issue": 185,
        "source": str(SOURCE.relative_to(MAIN_ROOT)).replace("\\", "/"),
        "source_plan_id": source_plan_id,
        "source_checkpoint_sha256": {arm: verified["checkpoint_sha256"][arm] for arm in ARMS},
        "source_receipts_sha256": receipt_hash,
        "training": {"seed": 183, "source_updates": 500, "target_updates": 8_000,
                     "optimizer": "restored AdamW", "learning_rate": 0.001,
                     "batch_pairs": 4, "recovery_every_updates": RECOVERY_EVERY},
        "evaluation": {"source_validation_plan_id": plan["sources"]["issue173_validation_plan_id"],
                       "validation_pair_count": len(validation_rows),
                       "validation_clip_count": 2 * len(validation_rows),
                       "validation_patch_roots": len({row["state_id"] for row in validation_rows}),
                       "validation_midi_performances": len({row["midi_id"] for row in validation_rows}),
                       "flow_euler_steps": FLOW_STEPS, "milestones": list(MILESTONES),
                       "metrics": list(METRICS), "paired_interval": "10,000 bootstrap replicates over patch roots"},
        "hardware": {"torch": str(torch.__version__), "numpy": np.__version__,
                     "device": str(device), "device_name": torch.xpu.get_device_name(device)
                     if device.type == "xpu" else "CPU", "threads": threads},
    }
    continuation["runner_sha256"] = sha256_file(__file__)
    plan_id = canonical_sha256(continuation)
    plan_path = output / "plan.json"
    frozen_continuation = {"id": plan_id, "plan": continuation}
    if plan_path.exists():
        saved_plan = BASE._load_json(plan_path)
        if saved_plan != frozen_continuation:
            migrate_frozen_plan(output, saved_plan, frozen_continuation,
                                continuation["source_checkpoint_sha256"])
    else:
        write_json(plan_path, frozen_continuation)
    if len(receipts) != 2176 or len(train_rows) != 2046 or len(validation_rows) != 128:
        raise ValueError("frozen data coverage differs from Issue 183")
    return plan_id, continuation, train_rows, validation_rows, source_states


def save_state(path, *, plan_id, arm, step, model, optimizer, order, cursor,
               shuffle, noise_rng, objective_total, objective_count, source_state_sha256):
    atomic_torch_save(path, {
        "format": "issue185_continuation_checkpoint_v1", "plan_id": plan_id,
        "source_checkpoint_sha256": source_state_sha256, "arm": arm, "step": step,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "order": order, "cursor": cursor, "shuffle_state": shuffle.get_state(),
        "noise_state": noise_rng.get_state(), "objective_total": objective_total,
        "objective_count": objective_count,
    })


def restore_state(saved, *, plan_id, arm, source_state_sha256, model, optimizer, shuffle, noise_rng, device):
    if (saved.get("format") != "issue185_continuation_checkpoint_v1"
            or saved.get("plan_id") != plan_id or saved.get("arm") != arm
            or saved.get("source_checkpoint_sha256") != source_state_sha256
            or saved.get("step", 0) < 500 or saved.get("step", 0) > 8_000
            or saved["step"] % RECOVERY_EVERY != 0):
        raise ValueError(f"invalid or incompatible continuation checkpoint: {arm}")
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor) and key != "step":
                state[key] = value.to(device)
    shuffle.set_state(saved["shuffle_state"])
    noise_rng.set_state(saved["noise_state"])
    if len(saved["order"]) < BASE.BATCH_PAIRS or not 0 <= saved["cursor"] <= len(saved["order"]):
        raise ValueError(f"invalid saved training permutation: {arm}")
    return saved["step"], saved["order"], saved["cursor"], saved["objective_total"], saved["objective_count"]


def validate_checkpoint_metadata(checkpoint, arm, plan_id, source_hash, row_count):
    if checkpoint is None:
        return None
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    step, order, cursor = saved.get("step"), saved.get("order"), saved.get("cursor")
    if (saved.get("format") != "issue185_continuation_checkpoint_v1"
            or saved.get("plan_id") != plan_id or saved.get("arm") != arm
            or saved.get("source_checkpoint_sha256") != source_hash
            or not isinstance(step, int) or step < 500 or step > 8_000
            or step % RECOVERY_EVERY or not isinstance(order, list) or len(order) != row_count
            or sorted(order) != list(range(row_count)) or not isinstance(cursor, int)
            or not 0 <= cursor <= row_count):
        raise ValueError(f"invalid continuation recovery state: {arm}")
    return saved


def migrate_frozen_plan(output, existing, replacement, source_hashes):
    old_plan, new_plan = existing.get("plan"), replacement["plan"]
    if not isinstance(old_plan, dict) or existing.get("id") == replacement["id"]:
        raise ValueError("existing continuation plan is invalid")
    comparable = dict(old_plan)
    comparable["runner_sha256"] = new_plan["runner_sha256"]
    if comparable != new_plan:
        raise ValueError("existing continuation plan differs beyond the corrected runner revision")
    old_id, new_id = existing["id"], replacement["id"]
    paths = [output / "recovery" / f"{arm}.pt" for arm in ARMS]
    paths.extend(output / "checkpoints" / f"{arm}-update-{step}.pt"
                 for arm in ARMS for step in MILESTONES)
    for path in paths:
        if not path.is_file():
            continue
        saved = torch.load(path, map_location="cpu", weights_only=True)
        arm = saved.get("arm")
        if saved.get("plan_id") == new_id:
            continue
        if (saved.get("plan_id") != old_id or arm not in source_hashes
                or saved.get("source_checkpoint_sha256") != source_hashes[arm]):
            raise ValueError(f"cannot migrate checkpoint under a changed continuation plan: {path.name}")
        saved["plan_id"] = new_id
        atomic_torch_save(path, saved)
    checkpoints_path = output / "checkpoints.json"
    if checkpoints_path.exists():
        saved_results = BASE._load_json(checkpoints_path)
        if saved_results.get("continuation") != old_plan:
            raise ValueError("saved milestone results differ from the old frozen continuation plan")
        saved_results["continuation"] = new_plan
        write_json(checkpoints_path, saved_results)
    manifest_path = output / "final-manifest.json"
    if manifest_path.exists():
        manifest = BASE._load_json(manifest_path)
        manifest["plan_id"] = new_id
        manifest["milestone_checkpoint_sha256"] = {
            arm: {str(step): sha256_file(output / "checkpoints" / f"{arm}-update-{step}.pt")
                  for step in MILESTONES
                  if (output / "checkpoints" / f"{arm}-update-{step}.pt").is_file()}
            for arm in ARMS}
        manifest["recovery_checkpoint_sha256"] = {
            arm: sha256_file(output / "recovery" / f"{arm}.pt")
            for arm in ARMS if (output / "recovery" / f"{arm}.pt").is_file()}
        write_json(manifest_path, manifest)
    write_json(output / "plan.json", replacement)


def add_source_milestone_500(output, plan_id, continuation, source_states, checkpoints):
    source_summary = BASE._load_json(SOURCE_REPORT / "summary.json")
    source_cases = BASE._load_json(SOURCE_REPORT / "per-case-metrics.json")
    for arm in ARMS:
        if "500" in checkpoints[arm]:
            continue
        mode = arm.rsplit("-", 1)[1]
        expected = source_summary["evaluation"][arm]
        cases = source_cases[arm]
        if mode == "flow":
            expected = expected[str(FLOW_STEPS)]
            cases = cases[str(FLOW_STEPS)]
        expected = expected.get("metrics", expected)
        metrics = BASE.aggregate_cases(cases)
        for metric in METRICS:
            if not math.isclose(metrics[metric], expected[metric], rel_tol=0, abs_tol=1e-10):
                raise ValueError(f"source report does not reproduce its update-500 per-case metrics: {arm}")
        if (len(cases) != 256 or len({case["state_id"] for case in cases}) != 64
                or len({case["midi_id"] for case in cases}) != 16):
            raise ValueError(f"source report has incomplete update-500 validation coverage: {arm}")
        checkpoints[arm]["500"] = {"metrics": metrics, "cases": cases,
                                    "mean_training_objective": None,
                                    "training_updates_since_last_milestone": 0, "segment_seconds": 0.0}
        source = source_states[arm]
        milestone = {"format": "issue185_continuation_checkpoint_v1", "plan_id": plan_id,
                     "source_checkpoint_sha256": continuation["source_checkpoint_sha256"][arm],
                     "arm": arm, "step": 500, "model": source["model"],
                     "optimizer": source["optimizer"], "order": source["order"],
                     "cursor": source["cursor"], "shuffle_state": source["shuffle_state"],
                     "noise_state": source["noise_state"], "objective_total": 0.0,
                     "objective_count": 0}
        atomic_torch_save(output / "checkpoints" / f"{arm}-update-500.pt", milestone)
    write_json(output / "checkpoints.json", checkpoints)


def evaluate_arm(model, arm, rows, pair_dir, codec, device, progress, checkpoint_step):
    mode = arm.rsplit("-", 1)[1]
    metrics, cases = BASE.evaluate_model(
        model, mode, rows, pair_dir, codec, device, flow_steps=FLOW_STEPS,
        seed=183 + 90_000, progress=progress,
        phase=f"Evaluating update {checkpoint_step:,} · {arm}")
    return {"metrics": metrics, "cases": cases}


def write_curve_files(output, result):
    rows = []
    for step in MILESTONES:
        for arm in ARMS:
            metrics = result["checkpoints"][arm][str(step)]["metrics"]
            rows.append({"updates": step, "arm": arm, **{metric: metrics[metric] for metric in METRICS}})
    with (output / "learning-curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["updates", "arm", *METRICS])
        writer.writeheader()
        writer.writerows(rows)


def _svg_text(value):
    return (str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def write_chart(output, result):
    width, height = 800, 430
    left, right, top, bottom = 90, 730, 48, 354
    points = {arm: [(step, result["checkpoints"][arm][str(step)]["metrics"]["spectral_loss"])
                    for step in MILESTONES] for arm in ARMS}
    values = [value for series in points.values() for _, value in series]
    low, high = min(values), max(values)
    padding = max((high - low) * 0.18, 0.01)
    low, high = low - padding, high + padding
    x = lambda step: left + (step - MILESTONES[0]) / (MILESTONES[-1] - MILESTONES[0]) * (right - left)
    y = lambda value: bottom - (value - low) / (high - low) * (bottom - top)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
             '<title id="title">Residual MLP validation spectral loss</title>',
             '<desc id="desc">Direct prediction and eight-step flow matching at five cumulative update counts.</desc>',
             '<rect width="800" height="430" fill="#fff"/>',
             '<text x="90" y="26" font-family="Segoe UI, sans-serif" font-size="17" font-weight="bold">Validation spectral loss (lower is better)</text>']
    for index in range(5):
        fraction = index / 4
        tick = high - fraction * (high - low)
        tick_y = y(tick)
        parts.append(f'<line x1="{left}" y1="{tick_y:.1f}" x2="{right}" y2="{tick_y:.1f}" stroke="#e4e8ee"/>')
        parts.append(f'<text x="{left - 12}" y="{tick_y + 4:.1f}" text-anchor="end" font-family="Segoe UI, sans-serif" font-size="12" fill="#343b45">{tick:.3f}</text>')
    parts.extend([f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#343b45"/>',
                  f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#343b45"/>'])
    for step in MILESTONES:
        step_x = x(step)
        parts.extend([f'<line x1="{step_x:.1f}" y1="{bottom}" x2="{step_x:.1f}" y2="{bottom + 5}" stroke="#343b45"/>',
                      f'<text x="{step_x:.1f}" y="{bottom + 25}" text-anchor="middle" font-family="Segoe UI, sans-serif" font-size="11" fill="#343b45">{step:,}</text>'])
    parts.append(f'<text x="{(left + right) / 2}" y="{bottom + 55}" text-anchor="middle" font-family="Segoe UI, sans-serif" font-size="13" fill="#343b45">Cumulative optimizer updates</text>')
    for index, arm in enumerate(ARMS):
        series = points[arm]
        coordinates = " ".join(f"{x(step):.1f},{y(value):.1f}" for step, value in series)
        color = COLORS[arm]
        parts.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="3"/>')
        for step, value in series:
            parts.append(f'<circle cx="{x(step):.1f}" cy="{y(value):.1f}" r="4.5" fill="{color}"/>')
        legend_y = 68 + index * 25
        label = "Direct prediction" if arm.endswith("direct") else "Flow matching · Euler 8"
        parts.extend([f'<line x1="{right - 210}" y1="{legend_y}" x2="{right - 185}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>',
                      f'<text x="{right - 178}" y="{legend_y + 4}" font-family="Segoe UI, sans-serif" font-size="12" fill="{color}">{_svg_text(label)}</text>'])
    parts.append("</svg>")
    path = output / "validation-loss-curves.svg"
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    ET.parse(path)


def paired_effect_at_checkpoint(direct_cases, flow_cases, metric, seed):
    return BASE.paired_effect(direct_cases, flow_cases, metric, seed=seed)


def summarize_checkpoints(output, plan_id, continuation, checkpoints):
    result = {"plan_id": plan_id, "source_checkpoint_sha256": continuation["source_checkpoint_sha256"],
              "validation_cases": continuation["evaluation"]["validation_clip_count"],
              "validation_patch_roots": continuation["evaluation"]["validation_patch_roots"],
              "validation_midi_performances": continuation["evaluation"]["validation_midi_performances"],
              "checkpoints": checkpoints, "paired_effects": {}}
    for step in MILESTONES:
        direct_cases = checkpoints[ARMS[0]][str(step)]["cases"]
        flow_cases = checkpoints[ARMS[1]][str(step)]["cases"]
        result["paired_effects"][str(step)] = {
            metric: paired_effect_at_checkpoint(direct_cases, flow_cases, metric, seed=18_500 + step)
            for metric in METRICS}
    return result


def _fmt(value, precision=4):
    return "—" if value is None else f"{value:.{precision}f}"


def write_report(output, result, continuation):
    spectral = result["checkpoints"]
    start_gap = spectral[ARMS[1]]["500"]["metrics"]["spectral_loss"] - spectral[ARMS[0]]["500"]["metrics"]["spectral_loss"]
    final_gap = spectral[ARMS[1]]["8000"]["metrics"]["spectral_loss"] - spectral[ARMS[0]]["8000"]["metrics"]["spectral_loss"]
    if start_gap > 0 and final_gap > 0:
        relation = "narrowed" if final_gap < start_gap else "widened" if final_gap > start_gap else "persisted"
    elif start_gap < 0 and final_gap < 0:
        relation = "narrowed" if abs(final_gap) < abs(start_gap) else "widened" if abs(final_gap) > abs(start_gap) else "persisted"
    else:
        relation = "reversed sign"

    lines = [
        "# Issue 185: direct and flow MLP learning curves", "",
        (f"We extended the same Residual MLP models from 500 to 8,000 cumulative optimizer updates. "
         f"The eight-step flow-minus-direct spectral-loss gap {relation}, from {start_gap:+.4f} at update 500 "
         f"to {final_gap:+.4f} at update 8,000. This one-seed result measures learning on the frozen validation panel."), "",
        (f"The panel contains {result['validation_cases']} clips, {result['validation_patch_roots']} patch roots, "
         f"and {result['validation_midi_performances']} distinct MIDI performances. Lower error is better. "
         "Flow results use the same fixed Gaussian noise and eight-step Euler sampler at every checkpoint."), "",
        "![Shared validation spectral-loss curves for direct prediction and eight-step flow matching.](validation-loss-curves.svg)", "",
        "| Updates | Direct spectral loss | Flow spectral loss | Flow − direct | Direct octave error (dB) | Flow octave error (dB) | Direct attack / decay / sustain / release (dB) | Flow attack / decay / sustain / release (dB) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for step in MILESTONES:
        direct = spectral[ARMS[0]][str(step)]["metrics"]
        flow = spectral[ARMS[1]][str(step)]["metrics"]
        gap = result["paired_effects"][str(step)]["spectral_loss"]["flow_minus_direct"]
        direct_envelope = " / ".join(_fmt(direct[key], 2) for key in METRICS[3:])
        flow_envelope = " / ".join(_fmt(flow[key], 2) for key in METRICS[3:])
        lines.append(f"| {step:,} | {_fmt(direct['spectral_loss'])} | {_fmt(flow['spectral_loss'])} | {gap:+.4f} | "
                     f"{_fmt(direct['octave_band_energy_error_db'], 2)} | {_fmt(flow['octave_band_energy_error_db'], 2)} | "
                     f"{direct_envelope} | {flow_envelope} |")

    lines.extend(["", f"The lowest observed validation spectral loss was {_fmt(min(
        item['metrics']['spectral_loss'] for item in spectral[ARMS[0]].values()))} for direct prediction and "
        f"{_fmt(min(item['metrics']['spectral_loss'] for item in spectral[ARMS[1]].values()))} for flow matching. "
        "The table separates those observed best checkpoints from each arm's update-8,000 result. Five checkpoints "
        "do not establish convergence.", "",
        "## Paired uncertainty", "",
        "Intervals resample the 64 validation patch roots, keeping clips from each root together. They do not "
        "resample MIDI performances or training seeds. This is an exploratory single-seed comparison on the "
        "existing panel.", "",
        "| Updates | Metric | Flow − direct (95% patch-root interval) |", "|---:|---|---:|"])
    for step in MILESTONES:
        for metric in ("spectral_loss", "octave_band_energy_error_db"):
            effect = result["paired_effects"][str(step)][metric]
            interval = effect.get("ci95")
            if interval:
                lines.append(f"| {step:,} | {METRIC_LABELS[metric]} | {effect['flow_minus_direct']:+.4f} "
                             f"[{interval[0]:+.4f}, {interval[1]:+.4f}] |")
    lines.extend(["", "## Execution and reproducibility", "",
                  "The two original checkpoints, optimizer states, training permutation, cursor, shuffle RNG, and flow-noise RNG were resumed from the corrected Issue 183 run. No architecture, objective, optimizer setting, data, or seed changed. The original cache receipts were re-verified and retained in [data-receipts.jsonl](data-receipts.jsonl).", "",
                  f"Source checkpoint SHA-256 values and frozen run identity are recorded in [plan.json](plan.json). "
                  f"All milestone metrics and paired intervals are in [summary.json](summary.json), with the "
                  f"shared graph data in [learning-curves.csv](learning-curves.csv).", "",
                  "No Vital controls are predicted or edited. The models condition on the existing 21-control codec "
                  "and known MIDI, then predict a stereo magnitude spectrogram.", "",
                  "Unrun conditions include additional seeds, additional architectures, reserved-test evaluation, "
                  "listening ratings, and downstream patch reconstruction. The conclusion applies to these two "
                  "models on the frozen validation panel only.", ""])
    report_path = output / "REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


def save_results(output, plan_id, continuation, checkpoints):
    result = summarize_checkpoints(output, plan_id, continuation, checkpoints)
    write_json(output / "summary.json", result)
    write_curve_files(output, result)
    write_chart(output, result)
    if all(str(step) in checkpoints[arm] for step in MILESTONES for arm in ARMS):
        write_report(output, result, continuation)
    return result


def fit_arm(output, arm, arm_index, plan_id, source_state, source_hash, train_rows,
            validation_rows, codec, device, progress, checkpoints):
    output = Path(output)
    state_dir = output / "recovery"
    checkpoint_dir = output / "checkpoints"
    state_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)
    recovery_path = state_dir / f"{arm}.pt"
    torch.manual_seed(183)
    if device.type == "xpu":
        torch.xpu.manual_seed_all(183)
    model = BASE.load_model("residual_mlp", arm.rsplit("-", 1)[1], codec.width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    shuffle = torch.Generator(device="cpu").manual_seed(183)
    noise_rng = BASE.cpu_generator(183 + 1000)
    source_step = 500
    order = torch.randperm(len(train_rows), generator=shuffle).tolist()
    cursor = 0
    objective_total = 0.0
    objective_count = 0
    saved = validate_checkpoint_metadata(recovery_path if recovery_path.exists() else None,
                                         arm, plan_id, source_hash, len(train_rows))
    if saved is None:
        saved = source_state
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and key != "step":
                    state[key] = value.to(device)
        order = saved["order"]
        cursor = saved["cursor"]
        shuffle.set_state(saved["shuffle_state"])
        noise_rng.set_state(saved["noise_state"])
        step = source_step
    else:
        step, order, cursor, objective_total, objective_count = restore_state(
            saved, plan_id=plan_id, arm=arm, source_state_sha256=source_hash,
            model=model, optimizer=optimizer, shuffle=shuffle, noise_rng=noise_rng, device=device)
    cache = BASE._pair_loader(BASE.TRAINING_SOURCE / "pairs")

    while step < MILESTONES[-1]:
        next_milestone = min(target for target in MILESTONES if target > step)
        next_recovery = min(((step // RECOVERY_EVERY) + 1) * RECOVERY_EVERY, next_milestone)
        stop_at = min(next_recovery, next_milestone)
        window_started = perf_counter()
        model.train()
        for update in range(step + 1, stop_at + 1):
            if cursor + BASE.BATCH_PAIRS > len(order):
                order = torch.randperm(len(train_rows), generator=shuffle).tolist()
                cursor = 0
            selected = [train_rows[position] for position in order[cursor:cursor + BASE.BATCH_PAIRS]]
            cursor += BASE.BATCH_PAIRS
            batch, _ = BASE.collate_pairs(selected, BASE.TRAINING_SOURCE / "pairs", codec, device, cache)
            optimizer.zero_grad(set_to_none=True)
            loss = BASE._training_objective(model, arm.rsplit("-", 1)[1], batch, noise_rng)
            if not torch.isfinite(loss):
                raise ValueError(f"nonfinite training objective at {arm} update {update}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            objective_total += float(loss.detach())
            objective_count += 1
        step = stop_at
        if device.type == "xpu":
            torch.xpu.synchronize(device)
        progress.update(phase=f"Training {arm}", arm=arm, arm_index=arm_index,
                        completed=step, total=8_000, force=True)
        if step in MILESTONES:
            model.eval()
            result = evaluate_arm(model, arm, validation_rows, VALIDATION_SOURCE / "validation-pairs",
                                  codec, device, progress, step)
            trace = {"metrics": result["metrics"], "cases": result["cases"],
                     "mean_training_objective": objective_total / max(1, objective_count),
                     "training_updates_since_last_milestone": objective_count,
                     "segment_seconds": perf_counter() - window_started}
            checkpoints[arm][str(step)] = trace
            milestone_path = checkpoint_dir / f"{arm}-update-{step}.pt"
            save_state(milestone_path, plan_id=plan_id, arm=arm, step=step, model=model,
                       optimizer=optimizer, order=order, cursor=cursor, shuffle=shuffle,
                       noise_rng=noise_rng, objective_total=objective_total, objective_count=objective_count,
                       source_state_sha256=source_hash)
            objective_total = 0.0
            objective_count = 0
        save_state(recovery_path, plan_id=plan_id, arm=arm, step=step, model=model,
                   optimizer=optimizer, order=order, cursor=cursor, shuffle=shuffle,
                   noise_rng=noise_rng, objective_total=objective_total, objective_count=objective_count,
                   source_state_sha256=source_hash)
        write_json(output / "checkpoints.json", checkpoints)
        if (step in MILESTONES
                and all(all(str(milestone) in checkpoints[current] for milestone in MILESTONES)
                        for current in ARMS)):
            save_results(output, plan_id, checkpoints["continuation"], checkpoints)


def load_existing_checkpoints(output):
    path = output / "checkpoints.json"
    if not path.exists():
        return {arm: {} for arm in ARMS} | {"continuation": None}
    saved = BASE._load_json(path)
    if set(saved) != set(ARMS) | {"continuation"} or saved["continuation"] is None:
        raise ValueError("existing checkpoint results have an invalid schema")
    for arm in ARMS:
        if not set(saved[arm]).issubset({str(step) for step in MILESTONES}):
            raise ValueError(f"existing checkpoint result has an unexpected milestone: {arm}")
        for step, result in saved[arm].items():
            if set(result) != {"metrics", "cases", "mean_training_objective",
                               "training_updates_since_last_milestone", "segment_seconds"}:
                raise ValueError(f"existing checkpoint metrics are incomplete: {arm} update {step}")
    return saved


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    if device.type == "xpu" and not torch.xpu.is_available():
        raise RuntimeError("the configured PyTorch XPU is unavailable")
    progress = Progress(args.output)
    plan_id, continuation, train_rows, validation_rows, source_states = load_source(
        args.output, device, args.threads, progress)
    if args.preflight_only:
        progress.update(phase="Preflight complete", state="preflight_complete",
                        message="Source checkpoints, data receipts, sampling state and update-500 scores verified",
                        force=True)
        print(json.dumps({"plan_id": plan_id, "source_checkpoint_sha256": continuation[
            "source_checkpoint_sha256"], "training_pairs": len(train_rows),
            "validation_pairs": len(validation_rows)}, indent=2), flush=True)
        return
    checkpoints = load_existing_checkpoints(args.output)
    if checkpoints["continuation"] not in (None, continuation):
        raise ValueError("existing continuation metrics belong to another frozen run")
    checkpoints["continuation"] = continuation
    write_json(args.output / "checkpoints.json", checkpoints)
    add_source_milestone_500(args.output, plan_id, continuation, source_states, checkpoints)
    codec = BASE.SurrogateCodec(BASE.PilotCodec(), expanded=True)
    for arm_index, arm in enumerate(ARMS, 1):
        progress.update(phase=f"Training {arm}", arm=arm, arm_index=arm_index,
                        completed=500, total=8_000, force=True)
        fit_arm(args.output, arm, arm_index, plan_id, source_states[arm],
                continuation["source_checkpoint_sha256"][arm], train_rows, validation_rows,
                codec, device, progress, checkpoints)
    result = save_results(args.output, plan_id, continuation, checkpoints)
    source_checkpoint_sha256 = continuation["source_checkpoint_sha256"]
    write_json(args.output / "final-manifest.json", {
        "plan_id": plan_id, "source_checkpoint_sha256": source_checkpoint_sha256,
        "milestone_checkpoint_sha256": {
            arm: {str(step): sha256_file(args.output / "checkpoints" / f"{arm}-update-{step}.pt")
                  for step in MILESTONES} for arm in ARMS},
        "recovery_checkpoint_sha256": {
            arm: sha256_file(args.output / "recovery" / f"{arm}.pt") for arm in ARMS},
        "validation_cases": result["validation_cases"],
        "validation_patch_roots": result["validation_patch_roots"],
        "validation_midi_performances": result["validation_midi_performances"],
    })
    progress.update(phase="Complete", arm="", arm_index=len(ARMS), completed=8_000,
                    total=8_000, state="complete", message="Both learning curves and the report are saved",
                    force=True)
    print(json.dumps({"report": str(args.output / "REPORT.md"),
                      "summary": str(args.output / "summary.json")}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--validation-source", type=Path, default=VALIDATION_SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("xpu", "cpu"), default="xpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--preflight-only", action="store_true",
                         help="verify the frozen inputs and existing models without updating weights")
    args = parser.parse_args()
    if args.source.resolve() != SOURCE.resolve() or args.validation_source.resolve() != VALIDATION_SOURCE.resolve():
        parser.error("the source and validation run identities are frozen for this experiment")
    if args.threads < 1:
        parser.error("threads must be positive")
    try:
        run(args)
    except KeyboardInterrupt:
        Progress(args.output).update(phase="Paused", state="paused",
                                     message="Worker interrupted; verified recovery checkpoints are retained",
                                     force=True)
        return 130
    except BaseException as error:
        Progress(args.output).update(phase="Failed", state="failed",
                                     message=f"{type(error).__name__}: {error}", force=True)
        raise


if __name__ == "__main__":
    main()

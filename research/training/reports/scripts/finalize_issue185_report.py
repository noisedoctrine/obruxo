"""Verify and publish the completed Issue 185 learning-curve artifacts."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

TRAINING = Path(__file__).resolve().parents[2]
REPORTS = TRAINING / "reports"
WORKTREE = TRAINING.parents[1]
MAIN_ROOT = WORKTREE.parent.parent if WORKTREE.parent.name == "tmp" else WORKTREE
DEFAULT_RUN = MAIN_ROOT / "research/training/outputs/issue185-direct-flow-learning-curves-20261127"
DEFAULT_REPORT = REPORTS / "issue185_direct_flow_mlp"
ARMS = ("residual_mlp-direct", "residual_mlp-flow")
MILESTONES = (500, 1_000, 2_000, 4_000, 8_000)
METRICS = (
    "spectral_loss", "octave_band_energy_error_db", "monophonic_octave_energy_error_db",
    "attack_envelope_error_db", "decay_envelope_error_db", "sustain_envelope_error_db",
    "release_envelope_error_db",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_run(run):
    plan_file = read_json(run / "plan.json")
    plan = plan_file.get("plan")
    if not isinstance(plan, dict) or plan.get("format") != "issue185_direct_vs_flow_mlp_continuation_v1":
        raise ValueError("run does not contain a frozen Issue 185 continuation plan")
    if plan.get("training", {}).get("target_updates") != 8_000:
        raise ValueError("run has the wrong cumulative update target")
    if plan.get("runner_sha256") != sha256_file(TRAINING / "reports/scripts/run_issue185_learning_curves.py"):
        raise ValueError("continuation runner changed after the experiment plan was frozen")
    receipt_path = REPORTS / "issue183_flow_matching/data-receipts.jsonl"
    receipt_sha = hashlib.sha256(receipt_path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    source_receipt_sha = plan.get("source_receipts_sha256")
    local_receipt = run / "data-receipts.jsonl"
    if (not local_receipt.is_file()
            or hashlib.sha256(local_receipt.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != source_receipt_sha
            or sha256_file(local_receipt) != receipt_sha):
        raise ValueError("continuation cache receipts differ from the canonical Issue 183 receipts")

    saved_checkpoints = read_json(run / "checkpoints.json")
    if saved_checkpoints.get("continuation") != plan:
        raise ValueError("checkpoint metrics reference a different frozen run plan")
    for arm in ARMS:
        arm_results = saved_checkpoints.get(arm, {})
        if set(arm_results) != {str(step) for step in MILESTONES}:
            raise ValueError(f"one or more evaluated milestones are missing: {arm}")
        for step in MILESTONES:
            result = arm_results[str(step)]
            if not set(METRICS).issubset(result["metrics"]) or len(result["cases"]) != 256:
                raise ValueError(f"validation metrics or cases are incomplete: {arm} at update {step}")
            patch_roots = {case["state_id"] for case in result["cases"]}
            performances = {case["midi_id"] for case in result["cases"]}
            if len(patch_roots) != 64 or len(performances) != 16:
                raise ValueError(f"validation case identity coverage changed: {arm} at update {step}")

    manifest_path = run / "final-manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("plan_id") != plan_file.get("id"):
        raise ValueError("final checkpoint manifest belongs to another run")
    if manifest.get("source_checkpoint_sha256") != plan.get("source_checkpoint_sha256"):
        raise ValueError("published source checkpoint hashes differ from the frozen plan")
    expected_output_hashes = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for step in MILESTONES:
            checkpoint = run / "checkpoints" / f"{arm}-update-{step}.pt"
            checkpoint_state = __import__("torch").load(checkpoint, map_location="cpu", weights_only=True)
            if (checkpoint_state.get("plan_id") != plan_file["id"]
                    or checkpoint_state.get("arm") != arm or checkpoint_state.get("step") != step
                    or checkpoint_state.get("source_checkpoint_sha256") != plan["source_checkpoint_sha256"][arm]):
                raise ValueError(f"saved milestone checkpoint does not match the run plan: {arm} at {step}")
            digest = sha256_file(checkpoint)
            expected_output_hashes[arm][str(step)] = digest
            if manifest.get("milestone_checkpoint_sha256", {}).get(arm, {}).get(str(step)) != digest:
                raise ValueError(f"milestone checkpoint hash does not match the final manifest: {arm} at {step}")

    svg_path = run / "validation-loss-curves.svg"
    ET.parse(svg_path)
    with (run / "learning-curves.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if (len(rows) != 10 or {row["arm"] for row in rows} != set(ARMS)
            or {int(row["updates"]) for row in rows} != set(MILESTONES)):
        raise ValueError("learning-curve CSV must contain both arms at all five checkpoints")
    return plan_file, saved_checkpoints, manifest, expected_output_hashes


def publish(run, report_dir):
    run, report_dir = Path(run).resolve(), Path(report_dir).resolve()
    plan_file, saved_checkpoints, manifest, output_hashes = verify_run(run)
    report_dir.mkdir(parents=True, exist_ok=True)
    report = (run / "REPORT.md").read_text(encoding="utf-8")
    report = report.replace("[data-receipts.jsonl](data-receipts.jsonl)",
                            "[Source #183 cache receipts](../issue183_flow_matching/data-receipts.jsonl)")
    if "Source #183 cache receipts" not in report:
        raise ValueError("the report does not link the tracked source receipts")
    final_direct = saved_checkpoints[ARMS[0]]["8000"]["metrics"]["spectral_loss"]
    final_flow = saved_checkpoints[ARMS[1]]["8000"]["metrics"]["spectral_loss"]
    if final_direct < final_flow:
        decision = "Direct prediction remains better-supported for this panel; another seed is needed before generalizing."
    elif final_flow < final_direct:
        decision = "Flow matching is better-supported for this panel at update 8,000; another seed is needed before generalizing."
    else:
        decision = "The methods tie on this panel at update 8,000; another seed is needed before generalizing."
    intro_marker = "This one-seed result measures learning on the frozen validation panel."
    if intro_marker not in report:
        raise ValueError("the report is missing its single-seed interpretation")
    report = report.replace(intro_marker, decision)
    table_start = report.index("| Updates | Direct spectral loss")
    table_end = report.index("\n\n", table_start)
    table = [
        "| Updates | Direct spectral loss | Flow spectral loss | Flow − direct | Direct octave error (dB) | Flow octave error (dB) | Direct MIDI-centered octave error (dB) | Flow MIDI-centered octave error (dB) | Direct attack / decay / sustain / release (dB) | Flow attack / decay / sustain / release (dB) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for step in MILESTONES:
        direct = saved_checkpoints[ARMS[0]][str(step)]["metrics"]
        flow = saved_checkpoints[ARMS[1]][str(step)]["metrics"]
        direct_envelope = " / ".join(f"{direct[key]:.2f}" for key in METRICS[3:])
        flow_envelope = " / ".join(f"{flow[key]:.2f}" for key in METRICS[3:])
        table.append(
            f"| {step:,} | {direct['spectral_loss']:.4f} | {flow['spectral_loss']:.4f} | "
            f"{flow['spectral_loss'] - direct['spectral_loss']:+.4f} | "
            f"{direct['octave_band_energy_error_db']:.2f} | {flow['octave_band_energy_error_db']:.2f} | "
            f"{direct['monophonic_octave_energy_error_db']:.2f} | "
            f"{flow['monophonic_octave_energy_error_db']:.2f} | {direct_envelope} | {flow_envelope} |"
        )
    report = report[:table_start] + "\n".join(table) + report[table_end:]
    best_by_arm = {}
    for arm in ARMS:
        best_step = min(MILESTONES, key=lambda step: saved_checkpoints[arm][str(step)]["metrics"]["spectral_loss"])
        best_by_arm[arm] = (best_step, saved_checkpoints[arm][str(best_step)]["metrics"]["spectral_loss"])
    paragraph_start = report.index("The lowest observed validation spectral loss was")
    paragraph_end = report.index("Five checkpoints do not establish convergence.", paragraph_start)
    paragraph_end += len("Five checkpoints do not establish convergence.")
    best_paragraph = (
        f"Direct prediction's lowest observed validation loss was {best_by_arm[ARMS[0]][1]:.4f} "
        f"at update {best_by_arm[ARMS[0]][0]:,}. Flow matching's lowest observed loss was "
        f"{best_by_arm[ARMS[1]][1]:.4f} at update {best_by_arm[ARMS[1]][0]:,}. "
        "Each update-8,000 value is reported separately in the comparison table. "
        "Five checkpoints do not establish convergence."
    )
    report = report[:paragraph_start] + best_paragraph + report[paragraph_end:]
    for name in ("learning-curves.csv", "validation-loss-curves.svg", "summary.json", "plan.json"):
        shutil.copy2(run / name, report_dir / name)
    (report_dir / "REPORT.md").write_text(report, encoding="utf-8")
    source_manifest = {
        "plan_id": plan_file["id"],
        "source_plan_id": plan_file["plan"]["source_plan_id"],
        "source_checkpoint_sha256": plan_file["plan"]["source_checkpoint_sha256"],
        "source_receipts_sha256": plan_file["plan"]["source_receipts_sha256"],
        "source_receipts_path": "../issue183_flow_matching/data-receipts.jsonl",
        "milestone_checkpoint_sha256": output_hashes,
        "checkpoint_storage": "research/training/outputs/issue185-direct-flow-learning-curves-20261127/checkpoints/",
        "validation_cases": manifest["validation_cases"],
        "validation_patch_roots": manifest["validation_patch_roots"],
        "validation_midi_performances": manifest["validation_midi_performances"],
        "milestones": list(MILESTONES),
    }
    (report_dir / "checkpoint-hashes.json").write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_dir / "REPORT.md"),
                      "summary": str(report_dir / "summary.json"),
                      "checkpoint_manifest": str(report_dir / "checkpoint-hashes.json"),
                      "run": str(run)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    publish(args.run, args.report_dir)


if __name__ == "__main__":
    main()

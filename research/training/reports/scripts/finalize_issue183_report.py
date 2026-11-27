"""Build a report from one frozen training run and its matching evaluation."""

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

TRAINING = Path(__file__).resolve().parents[2]
RESEARCH = TRAINING.parent
sys.path[:0] = [str(RESEARCH / "data_generation"), str(TRAINING)]

import run_issue183_flow_matching as pilot

BACKBONES = ("residual_mlp", "transformer", "conformer")
METRICS = (
    "spectral_loss", "octave_band_energy_error_db", "monophonic_octave_energy_error_db",
    "attack_envelope_error_db", "decay_envelope_error_db", "sustain_envelope_error_db",
    "release_envelope_error_db",
)


def _public_metadata(value):
    if isinstance(value, dict):
        return {key: _public_metadata(item) for key, item in value.items() if key != "device_properties"}
    if isinstance(value, list):
        return [_public_metadata(item) for item in value]
    return value


def _require_identity(document, label, plan_id, checkpoint_hashes, *, exact_hashes=False):
    source_id = document.get("source_plan_id", document.get("plan_id"))
    if source_id != plan_id:
        raise ValueError(f"{label} belongs to a different frozen training plan")
    hashes = document.get("source_checkpoint_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError(f"{label} is missing source checkpoint hashes")
    if exact_hashes and hashes != checkpoint_hashes:
        raise ValueError(f"{label} checkpoint hashes do not match the completed run")
    if any(checkpoint_hashes.get(arm) != digest for arm, digest in hashes.items()):
        raise ValueError(f"{label} uses a different checkpoint")


def _validate_listening(source, validation_source, listening, plan_id, checkpoint_hashes):
    if (not listening.get("available") or listening.get("experiment_plan_id") != plan_id
            or listening.get("source_checkpoint_sha256") != checkpoint_hashes):
        raise ValueError("listening previews are missing or belong to a different experiment")
    case_ids = listening.get("source_case_ids")
    if not isinstance(case_ids, list) or len(case_ids) != 10 or len(set(case_ids)) != 10:
        raise ValueError("listening metadata must identify ten distinct frozen source cases")
    if pilot.canonical_sha256(case_ids) != listening.get("source_case_ids_sha256"):
        raise ValueError("listening case identity digest does not match the case list")
    validation_source = validation_source.resolve()
    validation_document = pilot._load_json(validation_source / "plan.json")
    validation_plan = validation_document.get("plan", validation_document)
    expected_validation_id = (validation_document.get("id")
                              or validation_plan.get("training", {}).get("pilot_plan_id"))
    if listening.get("source_plan_id") != expected_validation_id:
        raise ValueError("listening previews use a different validation source plan")
    manifest = validation_source.parent / "issue173-listening-test-20261127" / "manifest.json"
    if not manifest.is_file() or pilot.sha256_file(manifest) != listening.get("source_manifest_sha256"):
        raise ValueError("listening source manifest is missing or has changed")
    manifest_cases = [row["case_id"] for row in pilot._load_json(manifest)["cases"]]
    if manifest_cases != case_ids:
        raise ValueError("listening examples do not match the validated source manifest")
    receipts = listening.get("asset_receipts")
    if not receipts:
        raise ValueError("listening metadata is missing asset receipts")
    packet = pilot._load_json(source / "listening" / "packet.json")
    if (packet.get("experiment_plan_id") != plan_id or packet.get("source_plan_id") != expected_validation_id
            or packet.get("source_manifest_sha256") != listening.get("source_manifest_sha256")
            or packet.get("source_case_ids") != case_ids
            or packet.get("source_case_ids_sha256") != listening.get("source_case_ids_sha256")
            or packet.get("source_checkpoint_sha256") != checkpoint_hashes
            or packet.get("asset_receipts") != receipts):
        raise ValueError("listening packet provenance does not match the run summary")
    expected_assets = {}
    for case in packet.get("cases", []):
        for key, relative_path in case.get("audio", {}).items():
            role = "target_audition" if key == "target" else "listening_preview"
            expected_assets[(case["case_id"], (Path("listening") / relative_path).as_posix())] = role
        for relative_path in case.get("exact_outputs", {}).values():
            expected_assets[(case["case_id"], (Path("listening") / relative_path).as_posix())] = "predicted_spectrum"
    if len(expected_assets) != len(receipts):
        raise ValueError("listening asset receipts do not cover the exported packet")
    asset_root = (source / "listening").resolve()
    source_audio_dir = manifest.parent / "audio"
    seen_assets = set()
    for receipt in receipts:
        case_id, relative = receipt.get("case_id"), receipt.get("path")
        key = (case_id, relative)
        if key in seen_assets or expected_assets.get(key) != receipt.get("role"):
            raise ValueError("listening asset receipts do not match packet cases and roles")
        seen_assets.add(key)
        asset = (source / receipt["path"]).resolve()
        if asset != asset_root and asset_root not in asset.parents:
            raise ValueError("listening asset receipt points outside the listening directory")
        if (not asset.is_file() or asset.stat().st_size != receipt.get("bytes")
                or pilot.sha256_file(asset) != receipt.get("sha256")):
            raise ValueError("listening asset receipt does not match a retained preview asset")
        if receipt["role"] == "target_audition":
            example = next(row["example"] for row in packet["cases"] if row["case_id"] == case_id)
            source_audio = source_audio_dir / f"sample-{example}-target.wav"
            if not source_audio.is_file() or pilot.sha256_file(source_audio) != receipt.get("source_sha256"):
                raise ValueError("listening target receipt does not match its frozen source audio")


def bootstrap_direct_comparisons(per_case, draws=20_000, seed=183172):
    rng = np.random.default_rng(seed)
    result = {"format": "issue183_direct_backbone_comparison_v2",
              "difference": "residual_mlp minus comparison backbone", "bootstrap_unit": "patch root",
              "bootstrap_draws": draws, "seed": seed, "comparisons": {}}
    for other in ("transformer", "conformer"):
        left_arm, right_arm = "residual_mlp-direct", f"{other}-direct"
        result["comparisons"][other] = {}
        for metric in METRICS:
            grouped = []
            for arm in (left_arm, right_arm):
                by_root = {}
                for row in per_case[arm]:
                    value = row.get(metric)
                    if row.get("state_id") is not None and value is not None:
                        by_root.setdefault(row["state_id"], []).append(float(value))
                grouped.append({root: float(np.mean(values)) for root, values in by_root.items()})
            roots = sorted(grouped[0].keys() & grouped[1].keys())
            delta = np.asarray([grouped[0][root] - grouped[1][root] for root in roots])
            samples = rng.choice(delta, size=(draws, len(delta)), replace=True).mean(axis=1)
            result["comparisons"][other][metric] = {
                "patch_roots": len(roots), "mean_difference": float(delta.mean()),
                "ci95": [float(np.quantile(samples, .025)), float(np.quantile(samples, .975))],
            }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", type=Path, required=True,
                        help="completed training run whose plan, checkpoints and learning curves are retained")
    parser.add_argument("--evaluation-output", type=Path,
                        help="matching corrected evaluation-only output; defaults to --run-output")
    parser.add_argument("--validation-source", type=Path, default=pilot.VALIDATION_SOURCE)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()
    training_source = args.run_output.resolve()
    evaluation_source = (args.evaluation_output or args.run_output).resolve()
    destination = args.report_output.resolve()
    if destination in {training_source, evaluation_source}:
        raise ValueError("report output must be separate from the source run directories")

    verified = pilot.verify_completed_run(training_source, args.validation_source)
    summary = pilot._load_json(evaluation_source / "summary.json")
    _require_identity(summary, "corrected evaluation", verified["plan_id"], verified["checkpoint_sha256"],
                      exact_hashes=True)
    training = pilot._load_json(training_source / "training-summary.json")
    _require_identity(training, "training summary", verified["plan_id"], verified["checkpoint_sha256"],
                      exact_hashes=True)
    run_summary = pilot._load_json(training_source / "summary.json")
    _require_identity(run_summary, "training run summary", verified["plan_id"], verified["checkpoint_sha256"],
                      exact_hashes=True)
    listening = run_summary.get("listening", {})
    _validate_listening(training_source, args.validation_source, listening, verified["plan_id"],
                        verified["checkpoint_sha256"])
    summary["listening"] = listening
    summary["training"] = training["arms"]
    summary.setdefault("hardware", training.get("hardware", verified["frozen"].get("hardware", {})))
    noise_source = next((source / "noise-sensitivity.json" for source in (evaluation_source, training_source)
                         if (source / "noise-sensitivity.json").is_file()), None)
    if noise_source:
        noise = pilot._load_json(noise_source)
        _require_identity(noise, "sampler noise sensitivity", verified["plan_id"],
                          verified["checkpoint_sha256"])
        summary["noise_sensitivity"] = {key: value for key, value in noise.items()
                                        if key != "per_case_by_seed"}
    duration_path = training_source / "duration-estimate.json"
    duration = pilot._load_json(duration_path)
    _require_identity(duration, "duration estimate", verified["plan_id"], verified["checkpoint_sha256"],
                      exact_hashes=True)
    required_duration = ("updates_per_second", "estimated_total_seconds", "measured_after_updates")
    if any(duration.get(key) is None for key in required_duration):
        raise ValueError("duration estimate is missing measured throughput or complete-run estimate")
    summary["throughput"] = {key: duration[key] for key in (
        "source_plan_id", "arm", "measured_after_updates", "updates_per_second", "elapsed_train_seconds",
        "estimated_training_seconds", "estimated_validation_seconds", "estimated_training_probe_seconds",
        "estimated_total_seconds", "estimate_scope", "assumption")}
    training_probe_noise_source = training_source / "training-probe-noise-sensitivity.json"
    if training_probe_noise_source.is_file():
        training_probe_noise = pilot._load_json(training_probe_noise_source)
        _require_identity(training_probe_noise, "training probe noise results", verified["plan_id"],
                          verified["checkpoint_sha256"])
        summary["training_probe_noise_sensitivity"] = training_probe_noise
    per_case = pilot._load_json(evaluation_source / "per-case-metrics.json")
    direct_comparisons = bootstrap_direct_comparisons(per_case)

    destination.mkdir(parents=True, exist_ok=True)
    summary = _public_metadata(summary)
    pilot.write_report(destination, summary, _public_metadata(verified["plan"]), training["arms"])
    pilot.write_json(destination / "summary.json", summary)
    pilot.write_json(destination / "per-case-metrics.json", per_case)
    pilot.write_json(destination / "direct-backbone-comparison.json", direct_comparisons)
    artifact_sources = {
        "plan.json": training_source,
        "learning-curves.csv": training_source,
        "validation-loss-curves.svg": training_source,
        "data-receipts.jsonl": training_source,
        "training-summary.json": training_source,
        "duration-estimate.json": training_source,
    }
    for name, source in artifact_sources.items():
        if name in {"plan.json", "training-summary.json", "duration-estimate.json"}:
            pilot.write_json(destination / name, _public_metadata(pilot._load_json(source / name)))
        else:
            shutil.copy2(source / name, destination / name)
    if noise_source:
        shutil.copy2(noise_source, destination / "noise-sensitivity.json")
    if training_probe_noise_source.is_file():
        shutil.copy2(training_probe_noise_source, destination / "training-probe-noise-sensitivity.json")
    print(f"Wrote report and corrected evaluation artifacts to {destination}")


if __name__ == "__main__":
    main()

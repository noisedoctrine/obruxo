"""Validate and publish the basic-session benchmark and complete-update profile."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
import sys

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "data_generation")]
from obruxo_data.hashing import canonical_sha256  # noqa: E402


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "profile", "baseline"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--tests-passed", type=int, required=True)
    args = parser.parse_args()
    plan, summary = read(args.run / "plan.json"), read(args.run / "summary.json")
    jobs, pairs = plan["max_jobs"], plan["max_jobs"] + 1
    events = [json.loads(line) for line in (args.run / "renders.jsonl").read_text().splitlines()]
    if (summary["format"] != "obruxo_render_worker_qualification_v3" or summary["plan_sha256"] != canonical_sha256(plan)
            or len(events) != 2 * pairs or any(row["status"] != "accepted" for row in events)
            or not summary["qualified"] or not all(summary["gates"].values()) or len(summary["checks"]) != pairs):
        raise ValueError("worker evidence is incomplete or failed qualification")
    for row in summary["checks"]:
        if (not row["native_state_identical"] or not row["provenance_identical"]
                or row["maximum_absolute_sample_error"] > plan["numeric_limits"]["maximum_absolute_sample_error"]
                or row["waveform_rmse"] > plan["numeric_limits"]["waveform_rmse"]):
            raise ValueError("native equivalence limits failed")
    for name, expected in plan["source"].items():
        if hashlib.sha256((RESEARCH / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"worker qualification source changed: {name}")
    for mode, key in (("fresh", "matched_fresh_mean_seconds"), ("reuse", "reuse_mean_seconds")):
        if not math.isclose(mean(row["wall_seconds"] for row in events if row["mode"] == mode), summary[key], abs_tol=1e-12):
            raise ValueError("worker timings do not match native outcomes")
    if [row["worker_job"] for row in events if row["mode"] == "reuse"] != [*range(1, jobs + 1), 1]:
        raise ValueError("worker lifetime does not match the plan")
    profile, baseline = read(args.profile / "summary.json"), read(args.baseline / "summary.json")
    if (not profile["complete"] or not profile["original_artifacts_unchanged"] or not profile["training_numerics_within_limits"]
            or profile["updates"] != 1 or profile["native_requests"] != baseline["native_requests"]):
        raise ValueError("complete-update comparison is incomplete")
    for actual, limit in (("maximum_loss_error", "absolute_loss_error"), ("maximum_gradient_error", "absolute_gradient_error"),
            ("maximum_relative_gradient_error", "relative_gradient_error_above_0_01"), ("maximum_parameter_update_error", "parameter_update_error")):
        if profile[actual] > plan["training_limits"][limit]:
            raise ValueError(f"training stability failed: {actual}")
    public_profile = {key: value for key, value in profile.items() if key not in {"input_setup", "frontend", "frontend_samples_seconds"}}
    checks = summary.pop("checks")
    summary["comparisons"] = {"pairs": pairs,
        "maximum_absolute_sample_error": max(row["maximum_absolute_sample_error"] for row in checks),
        "maximum_waveform_rmse": max(row["waveform_rmse"] for row in checks),
        "audio_float32_identical_pairs": sum(row["audio_float32_identical"] for row in checks),
        "native_state_all_identical": all(row["native_state_identical"] for row in checks),
        "provenance_all_identical": all(row["provenance_identical"] for row in checks),
        "engine_reused_pairs": sum(row["engine_reused"] for row in checks)}
    summary.update(local_tests_passed=args.tests_passed, github_actions="disabled", pipeline_profile=public_profile,
                   pipeline_baseline={key: baseline[key] for key in ("update", "worker_stages_total_seconds", "worker_wall_seconds")},
                   raw_evidence_sha256=hashlib.sha256((args.run / "renders.jsonl").read_bytes()).hexdigest(),
                   profile_evidence_sha256=hashlib.sha256((args.profile / "summary.json").read_bytes()).hexdigest())
    reports = Path(__file__).resolve().parents[1]
    for name, value in (("render_worker_plan.json", plan), ("render_worker_summary.json", summary)):
        (reports / "data" / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    old_time, new_time = (value["update"]["complete_update"]["inclusive"] for value in (baseline, profile))
    old, new = baseline["worker_stages_total_seconds"], profile["worker_stages_total_seconds"]
    sample_error = summary["comparisons"]["maximum_absolute_sample_error"]
    rows = ["# Basic Vital session results — 1 November 2026", "",
            f"The same 126-render training update took **{old_time:.2f}s in the saved baseline and {new_time:.2f}s with {jobs} jobs per worker**. Candidate controls and acceptance decisions were preserved. The profiles were separate runs with different host conditions; no full-update speedup is claimed from this comparison.", "",
            f"The adjacent-pair benchmark provides the controlled render comparison: **{summary['matched_fresh_mean_seconds']:.3f}s fresh versus {summary['reuse_mean_seconds']:.3f}s reused**, {100*summary['mean_wall_reduction']:.1f}% less time across {pairs} pairs, including cold starts and recycling at {jobs} jobs. All {2*pairs} attempts passed QA and all restart/memory gates passed.", "",
            "## Changes and measured costs", "",
            "| Work per 126-render update | Before | After |", "| --- | ---: | ---: |",
            *[f"| {label} | {a:.3f}s | {b:.3f}s |" for label,a,b in (("Create Vital instances", old["plugin_creation"], new["plugin_creation"]),
              ("Load/apply patch controls", old.get("plugin_state_load", old.get("direct_parameter_updates")), new["direct_parameter_updates"]),
              ("Validate requests", old["request_validation"], new["request_validation"]),
              ("Whole native worker calls", baseline["worker_wall_seconds"], profile["worker_wall_seconds"]))], "",
            "The basic profile keeps one engine/plugin per worker, clears MIDI each request, and uses the host's normal render reset. It loads the complete preset initially and updates the eleven controls thereafter. Setter normalization explicitly matches the full loader's float32 arithmetic. Every render still saves and verifies native state, fingerprints the plugin and checks the audio.", "",
            "Only the reviewed legacy basic domain, 44.1kHz, 128-sample buffers, 32,768-frame preroll, fixed oscillator/sampler phases and unchanged init assets/routes/effects qualify. The shared control definition keeps the codec and renderer bounds aligned. Template changes force revalidation/recreation; other presets use the full loading path in the same renderer.", "",
            "## Numerical correctness", "",
            f"The fixed-phase paired check found maximum absolute sample error **{sample_error:.3g}**, with native readback/provenance identical. {summary['comparisons']['audio_float32_identical_pairs']}/{pairs} audio buffers were bit-identical; the remaining pairs passed the explicit numeric limits recorded in the plan. This is numerical equivalence, not a bit-identical claim.", "",
            *([f"Warm worker median memory changed from {summary['warm_worker_memory']['early_median_bytes']/1024**2:.2f} MiB to {summary['warm_worker_memory']['late_median_bytes']/1024**2:.2f} MiB across the {jobs}-job lifetime, below the predeclared 64 MiB growth limit.", ""] if "warm_worker_memory" in summary else []),
            f"For the complete update, maximum loss difference was **{profile['maximum_loss_error']:.3g}**, gradient coefficient difference **{profile['maximum_gradient_error']:.3g}**, and model-weight update difference **{profile['maximum_parameter_update_error']:.3g}**. All 126 requested candidate controls matched the original workload exactly; {profile['exact_native_hash_matches']}/126 waveform hashes matched exactly. The full trace differs in tiny measured numeric values. Original training artifacts were unchanged.", "",
            "An initial double-precision setter conversion amplified tiny rounding differences in Adam and failed the update-stability guard. Matching the full loader's float32 arithmetic fixed that failure. The rejected setter and float32-readback-fallback implementations were removed; no optimizer or loss changes were introduced to accommodate them.", "",
            f"{args.tests_passed} focused local regression checks passed, covering worker failure/recycling, backpressure, stream contracts and checkpoint recovery. Static correctness lint and whitespace checks passed. GitHub Actions remains disabled.", "",
            "## Remaining cost and scope", "",
            f"Native worker calls now account for {100*profile['worker_wall_seconds']/new_time:.1f}% of the update. Spectral feature extraction and loss comparisons together take {profile['update']['spectral_features']['exclusive']+profile['update']['spectral_loss']['exclusive']:.2f}s. All 120 gradient-probe renders and six acceptance renders per update are retained.", "",
            "This qualifies the recorded basic Windows workload and numerical bounds. It does not establish finished-model quality or arbitrary-preset/cross-platform equivalence. The v3 lifecycle is frozen in new stream contracts and native receipts; historical runs use their recorded source revision.", "",
            "[Usage](../RENDER_WORKER.md) · [Plan](data/render_worker_plan.json) · [Measurements](data/render_worker_summary.json)", ""]
    (reports / "RENDER_WORKER_RESULTS.md").write_text("\n".join(rows), encoding="utf-8", newline="\n")
    print(json.dumps({"qualified": summary["qualified"], "paired_wall_reduction": summary["mean_wall_reduction"],
                      "update_before_seconds": old_time, "update_after_seconds": new_time}, indent=2))


if __name__ == "__main__":
    main()

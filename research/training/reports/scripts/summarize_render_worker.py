"""Validate and publish the current paired worker benchmark and stage profile."""

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
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--tests-passed", type=int, required=True)
    args = parser.parse_args()
    plan, summary = read(args.run / "plan.json"), read(args.run / "summary.json")
    events = [json.loads(line) for line in (args.run / "renders.jsonl").read_text().splitlines()]
    if (summary["format"] != "obruxo_render_worker_qualification_v2" or summary["plan_sha256"] != canonical_sha256(plan)
            or len(events) != 66 or any(row["status"] != "accepted" for row in events)
            or not summary["qualified"] or not all(summary["gates"].values()) or len(summary["checks"]) != 33
            or not all(row["audio_float32_identical"] and row["native_state_identical"] and row["provenance_identical"]
                       for row in summary["checks"])):
        raise ValueError("worker evidence is incomplete or failed qualification")
    for name, expected in plan["source"].items():
        if hashlib.sha256((RESEARCH / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"worker qualification source changed: {name}")
    for mode, key in (("fresh", "matched_fresh_mean_seconds"), ("reuse", "reuse_mean_seconds")):
        if not math.isclose(mean(row["wall_seconds"] for row in events if row["mode"] == mode), summary[key], abs_tol=1e-12):
            raise ValueError("worker timings do not match native outcomes")
    if [row["worker_job"] for row in events if row["mode"] == "reuse"] != [*range(1, 33), 1]:
        raise ValueError("worker lifetime does not match the plan")
    profile_events = read(args.profile / "renders.json")
    if (not read(args.profile / "result.json")["complete"] or len(profile_events) != 12
            or not all(row["status"] == "accepted" and row["exact_source_match"] for row in profile_events)):
        raise ValueError("profile did not reproduce all saved source waveforms")
    profile = {}
    for mode in ("fresh", "reused"):
        stages = [read(path)["stages"] for path in args.profile.glob(f"{mode}-*.json")]
        if len(stages) != 6:
            raise ValueError("profile needs six complete stage measurements per mode")
        profile[mode] = {"wall_seconds": mean(row["wall_seconds"] for row in profile_events if row["mode"] == mode),
                         "backend_seconds": mean(row["render_seconds"] for row in profile_events if row["mode"] == mode),
                         "stages": {key: mean(row.get(key, 0.) for row in stages) for key in sorted(set().union(*stages))}}
    summary.update(local_tests_passed=args.tests_passed, github_actions="disabled", profile=profile,
                   raw_evidence_sha256=hashlib.sha256((args.run / "renders.jsonl").read_bytes()).hexdigest(),
                   profile_evidence_sha256=hashlib.sha256((args.profile / "renders.json").read_bytes()).hexdigest())
    reports = Path(__file__).resolve().parents[1]
    for name, value in (("render_worker_plan.json", plan), ("render_worker_summary.json", summary)):
        (reports / "data" / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    stages = profile["reused"]["stages"]
    rows = ["# Render-worker results — 31 October 2026", "",
            f"Mean whole-request time fell from **{summary['matched_fresh_mean_seconds']:.3f}s to {summary['reuse_mean_seconds']:.3f}s ({100 * summary['mean_wall_reduction']:.1f}% less time)** in 33 adjacent pairs with alternating execution order. Cold starts and recycling are included. All 66 attempts passed QA, and all 33 pairs matched float32 audio, state readback and provenance exactly.", "",
            f"The check covers {len(plan['families'])} basic unison/filter combinations, held notes and chords, A–B–A ordering, fixed oscillator/sampler phases, 32,768 frames of preroll and recycling after 32 jobs. Peak sampled process-tree memory was {summary['peak_process_tree_rss_bytes'] / 2**20:.1f} MiB, with no sampling errors or surviving children.", "",
            "The implementation reuses the Python process, renderer setup and immutable state template. Each clip gets a fresh engine/plugin and full state readback. Reusing the plugin itself failed exact reproducibility in a six-request diagnostic (largest absolute sample difference 8.74e-10); that implementation was removed. No tolerance was relaxed.", "",
            "## Remaining cost", "",
            f"A separate instrumented run reproduced all 12 saved source clips exactly. Its means were {profile['fresh']['wall_seconds']:.3f}s fresh and {profile['reused']['wall_seconds']:.3f}s reused. The table is time per reused request; the engine render is included in the audio-render total.", "",
            "| Work | Seconds |", "| --- | ---: |",
            *[f"| {label} | {stages[key]:.3f} |" for label, key in (("Create plugin", "plugin_creation"),
              ("Save/load plugin state", "plugin_state_save_load"), ("Validate request", "request_validation"),
              ("Convert state", "state_conversion"), ("Audio rendering, preroll and copying", "audio_render_including_preroll"),
              ("Native engine render alone (included above)", "nested_native_engine_render"))], "",
            "Plugin creation and state handling remain the main render costs. Actual audio synthesis is already faster than real time. The earlier saved-update profile measured about 0.36s of network work per update; this change does not alter the network or its objective. A full training run has not been repeated, so the paired render speedup is not a measured whole-training speedup.", "",
            "## Integration and validation", "",
            f"New online streams and NativeSession collections/fits use one renderer per worker, recycled after 32 jobs by default. The duplicate one-shot worker implementation was removed; fresh correctness checks use the same worker with a one-job lifetime. {args.tests_passed} focused regression tests passed, covering rejection, crash/deadline/cancellation recovery, cleanup, backpressure and optimizer/RNG recovery. Static correctness checks passed. GitHub Actions remains disabled.", "",
            "The new lifecycle is recorded in stream contracts and native receipts. Historical runs use their recorded source revision; they cannot silently resume under this policy. This qualifies the recorded local Windows basic-control cases, not other native versions/platforms, arbitrary patches or model quality. Random-phase training remains stochastic.", "",
            "[Usage](../RENDER_WORKER.md) · [Plan](data/render_worker_plan.json) · [Measurements](data/render_worker_summary.json)", ""]
    (reports / "RENDER_WORKER_RESULTS.md").write_text("\n".join(rows), encoding="utf-8", newline="\n")
    print(json.dumps({key: summary[key] for key in ("qualified", "gates", "mean_wall_reduction", "accepted")}, indent=2))


if __name__ == "__main__":
    main()

"""Publish matched native latency evidence without keeping audio or patch payloads."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--tests-passed", type=int, required=True)
    args = parser.parse_args()
    before, after = [json.loads(path.read_text()) for path in (args.before, args.after)]
    if (before["panel_sha256"] != after["panel_sha256"] or len(before["records"]) != 5 or len(after["records"]) != 5
            or not all(run["training_only"] and not run["final_test_used"] for run in (before, after))):
        raise ValueError("latency comparison requires the same five training-only cases")
    cases = []
    for left, right in zip(before["records"], after["records"], strict=True):
        if (any(left[key] != right[key] for key in ("family", "state_id", "request_id", "audio_seconds", "provenance"))
                or left["status"] != "accepted" or right["status"] != "accepted"
                or left["qa"]["audio_float32_sha256"] != right["qa"]["audio_float32_sha256"]
                or left["renderer_qa"]["native_state"] != right["renderer_qa"]["native_state"]):
            raise ValueError("native requests, state readback, provenance, or audio differ")
        phases = [row["renderer_qa"]["phase_seconds"] for row in (left, right)]
        codec_seconds = [row["juce_memory_block_decode"] + row["juce_memory_block_encode"] for row in phases]
        cases.append({"family": left["family"], "request_id": left["request_id"], "audio_seconds": left["audio_seconds"],
                      "before_wall_seconds": left["wall_seconds"], "after_wall_seconds": right["wall_seconds"],
                      "before_codec_seconds": codec_seconds[0], "after_codec_seconds": codec_seconds[1],
                      "before_synthesis_seconds": phases[0]["render"], "after_synthesis_seconds": phases[1]["render"],
                      "native_state_identical": True, "audio_float32_identical": True})
    before_wall = sum(row["before_wall_seconds"] for row in cases)
    after_wall = sum(row["after_wall_seconds"] for row in cases)
    summary = {"format": "obruxo_vital_latency_comparison_v1", "cases": cases, "training_only": True, "final_test_used": False,
               "native_attempts": 10, "accepted": 10, "audio_files_written": 0, "process_mode": "fresh_process_and_plugin_per_render",
               "before_wall_seconds": before_wall, "after_wall_seconds": after_wall,
               "wall_reduction_percent": 100 * (1 - after_wall / before_wall),
               "before_codec_mean_seconds": mean(row["before_codec_seconds"] for row in cases),
               "after_codec_mean_seconds": mean(row["after_codec_seconds"] for row in cases),
               "renderer_identity": {key: value for key, value in before["records"][0]["provenance"].items() if key != "request_id"},
               "source_sha256": {"before": before["renderer_source_sha256"], "after": after["renderer_source_sha256"]},
               "raw_evidence_sha256": {label: hashlib.sha256(path.read_bytes()).hexdigest() for label, path in (("before", args.before), ("after", args.after))},
               "panel_sha256": before["panel_sha256"], "local_tests_passed": args.tests_passed, "github_actions": "disabled",
               "limits": "Five fixed training cases, one cold process per case, serial before/after Windows CPU timing; not a scaling or quality claim."}
    reports = Path(__file__).resolve().parents[1]
    (reports / "data/vital_render_latency.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    rows = ["# Vital render latency", "",
            "Vital already renders offline. The measured delay was dominated by work around synthesis, including repeated process/plugin startup and Python conversion of JUCE patch-state data.", "",
            "The state encoder/decoder now uses standard-library base64 and byte translation tables, reversing bit order to preserve JUCE's exact format. It keeps declared-size padding, truncation and invalid-character rejection. Native state readback, per-render plugin hashing, QA and process timeouts still run.", "",
            f"Across the same five training patches, total fresh-process render time fell from **{before_wall:.2f}s to {after_wall:.2f}s ({summary['wall_reduction_percent']:.1f}% less wall time)**. Mean patch-state conversion fell from **{summary['before_codec_mean_seconds']:.3f}s to {summary['after_codec_mean_seconds']:.4f}s**. Every requested patch, read-back native state, renderer provenance and float32 audio hash matched before/after exactly.", "",
            "| Family | Audio duration | Native synthesis after | Whole call before | Whole call after |", "|---|---:|---:|---:|---:|"]
    rows += [f"| {row['family']} | {row['audio_seconds']:.1f}s | {row['after_synthesis_seconds']:.3f}s | {row['before_wall_seconds']:.2f}s | {row['after_wall_seconds']:.2f}s |" for row in cases]
    rows += ["", f"All 10 native attempts passed. The full data-generation suite plus training stream checks passed **{args.tests_passed} local tests**. Dedicated compatibility tests cover every single-byte value, bit boundaries, payloads up to 256 KiB, short/excess data and malformed encodings. GitHub Actions remains disabled.", "",
             "This is a five-case local Windows CPU measurement, performed serially before/after after other test jobs finished. Individual calls vary, and this does not establish broad throughput or model quality. The small profiler used to find the bottleneck is excluded from these unprofiled phase measurements. No final-test states or audio archive were used.", "",
             "The remaining substantial work is process/plugin startup and native state loading. Reusing an isolated worker is a separate follow-up: it needs crash/timeout recovery, bounded lifetime and tests for patch-to-patch contamination before becoming the default. Increasing batch or queue size alone cannot remove this cost.", "",
             "[Aggregate evidence](data/vital_render_latency.json) · [Measurement script](scripts/benchmark_vital_latency.py) · [Summarizer](scripts/summarize_vital_latency.py)", ""]
    (reports / "VITAL_RENDER_LATENCY.md").write_text("\n".join(rows))
    print(json.dumps({key: summary[key] for key in ("wall_reduction_percent", "before_codec_mean_seconds", "after_codec_mean_seconds", "accepted")}, indent=2))


if __name__ == "__main__":
    main()

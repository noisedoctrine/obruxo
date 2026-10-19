"""Publish aggregate and per-case diagnostics without preset, weight or audio payloads."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean


def summarize(source):
    raw = source.read_bytes()
    result = json.loads(raw)
    plan, events = result["plan"], result["events"]
    if not result["complete"] or result["accepted_requests"] != 70 or not result["within_resource_limits"]:
        raise ValueError("do not publish incomplete native checks as a completed diagnostic")
    if [row["request_id"] for row in events] != [job["request_id"] for job in plan["jobs"]]:
        raise ValueError("completed requests differ from the frozen plan")
    repeats = [row for row in events if row["method"] == "source_repeat"]
    if len(repeats) != 10 or not all(row["repeat_exact"] for row in repeats):
        raise ValueError("phase-controlled repeats failed")
    indexed = {(row["case"], row["role"], row["method"]): row for row in events}
    if len(indexed) != 70:
        raise ValueError("duplicate requests in the diagnostic")
    cases = [case["case"] for case in plan["cases"]]
    methods = ("prediction", *plan["groups"])
    case_scores, aggregates = [], {}
    for case in cases:
        scores = {method: {role: indexed[case, role, method]["scores"] for role in ("reference", "transfer")} for method in methods}
        primary = {method: mean(scores[method][role]["mrstft"] for role in ("reference", "transfer")) for method in methods}
        case_scores.append({"case": case, "scores": scores, "primary": primary,
                            "delta_from_prediction": {method: value - primary["prediction"] for method, value in primary.items()}})
    for method in methods:
        aggregates[method] = {"primary_mean": mean(case["primary"][method] for case in case_scores),
                              "improved_cases": sum(case["delta_from_prediction"][method] < 0 for case in case_scores),
                              "planned_cases": len(cases)}
        for metric in ("mrstft", "rms_relative", "envelope_relative_rmse", "stereo_energy_error"):
            aggregates[method][metric] = {role: mean(indexed[case, role, method]["scores"][metric] for case in cases)
                                         for role in ("reference", "transfer")}
    return {"source_summary_sha256": hashlib.sha256(raw).hexdigest(), "plan": plan,
            "completed_requests": 70, "accepted_requests": 70, "exact_source_repeat_pairs": 10,
            "elapsed_seconds": result["elapsed_seconds"], "peak_process_tree_rss_bytes": result["peak_process_tree_rss_bytes"],
            "rss_sampling_errors": result["rss_sampling_errors"], "within_resource_limits": True,
            "native_render_seconds": sum(row["render_seconds"] for row in events),
            "audio_seconds": sum(row["qa"]["actual_frames"] / 44100 for row in events),
            "aggregates": aggregates, "case_scores": case_scores,
            "render_receipts": [{key: row[key] for key in ("case", "method", "role", "request_id", "status")}
                                | {"audio_sha256": row["qa"]["audio_float32_sha256"]} for row in events],
            "quality_claim": "five_development_cases_oracle_diagnosis_only", "model_promotion": False,
            "listening_status": "clips_prepared_listener_assessment_pending", "training_updates": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.source)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"completed_requests": result["completed_requests"], "aggregates": result["aggregates"]}))


if __name__ == "__main__":
    main()

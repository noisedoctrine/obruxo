"""Publish completed A1 evidence, including failed qualification, without training tensors."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


def summarize(directory):
    from obruxo_data.hashing import canonical_sha256
    from obruxo_training.temporal_proxy_screen import qualification_summary
    names = ("summary.json", "plan.json", "records.json", "cases.json", "renders.jsonl")
    raw = {name: (directory / name).read_bytes() for name in names}
    summary, plan, records, cases = [json.loads(raw[name]) for name in names[:4]]
    events = [json.loads(line) for line in raw["renders.jsonl"].splitlines()]
    if (not summary["complete"] or summary["failure"] is not None or canonical_sha256(plan) != summary["plan_id"]
            or len(records) != len(plan["training_recipes"]) or len(cases) != 2 * plan["config"]["validation_states"]):
        raise ValueError("incomplete or mismatched helper evidence")
    result = qualification_summary(cases, plan)
    for key, value in result.items():
        if value != summary["result"][key]:
            raise ValueError("qualification summary disagrees with saved cases")
    outcomes = [event for event in events if event["event"] == "outcome"]
    native = [event for event in outcomes if event["phase"] == "qualification"]
    if len(native) != 20 * plan["config"]["validation_states"] or any(row["status"] != "accepted" for row in native):
        raise ValueError("missing accepted qualification requests")
    repeat_count = sum(case["rows"]["repeat"]["repeat_exact"] for case in cases)
    if repeat_count != len(cases):
        raise ValueError("non-exact qualification repeat")
    failed = [row["receipt"] for row in records if not row["accepted"]]
    failures = Counter(row.get("message", ",".join(code for code in row.get("diagnostic_codes", []) if code.startswith("audio."))) for row in failed)
    compact = []
    for case in cases:
        rows, start = case["rows"], case["rows"]["start"]
        guided = {method: {"native_delta": rows[method]["native_mrstft"] - start["native_mrstft"],
                           "spectral_delta": rows[method]["native_spectral"] - start["native_spectral"],
                           "helper_delta": rows[method]["helper_error"] - start["helper_error"],
                           "category_changed": rows[method]["guide"]["category_changed"]}
                  for method in ("continuous", "joint")}
        compact.append({"case": case["case"], "role": case["role"], "state_id": case["state_id"],
                        "performance_id": case["performance_id"], "start_native_error": start["native_mrstft"],
                        "guided": guided, "repeat_exact": rows["repeat"]["repeat_exact"]})
    artifact = directory / "helper.pt"
    with artifact.open("rb") as handle:
        artifact_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"source_hashes": {name: hashlib.sha256(value).hexdigest() for name, value in raw.items()},
            "helper_sha256": artifact_hash, "plan_id": summary["plan_id"], "source_modules": plan["source"],
            "config": plan["config"], "limits": plan["limits"], "model": plan["model"],
            "complete": True, "qualified_for_inverse_training": result["qualified_for_inverse_training"],
            "result": result, "cases": compact, "exact_repeat_pairs": repeat_count,
            "training_accepted": sum(row["accepted"] for row in records), "training_failures": dict(failures),
            "retained_archive_bytes": sum(path.stat().st_size for path in directory.rglob("*") if path.is_file()),
            **{key: summary[key] for key in ("render_attempts", "elapsed_seconds", "peak_process_tree_rss_bytes",
                                           "rss_sampling_errors", "within_resource_limits", "renderer_identity",
                                           "test_split_used", "training_audio_archived")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = summarize(args.source)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False, sort_keys=True) + "\n")
    print(json.dumps({"qualified": value["qualified_for_inverse_training"], "exact_repeats": value["exact_repeat_pairs"],
                      "training_accepted": value["training_accepted"]}))


if __name__ == "__main__":
    main()

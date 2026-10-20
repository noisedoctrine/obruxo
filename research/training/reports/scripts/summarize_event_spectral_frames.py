"""Publish verified event-frame preflight evidence without its feature tensors."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


def summarize(directory):
    from obruxo_data.hashing import canonical_sha256
    names = ("plan.json", "summary.json", "cases.json", "renders.jsonl")
    raw = {name: (directory / name).read_bytes() for name in names}
    plan, summary, cases = [json.loads(raw[name]) for name in names[:3]]
    events = [json.loads(line) for line in raw["renders.jsonl"].splitlines()]
    outcomes = [event for event in events if event["event"] == "outcome"]
    if (not summary["complete"] or not summary["preflight_passed"] or not summary["within_resource_limits"]
            or summary["plan_id"] != canonical_sha256(plan) or summary["plan"] != plan or summary["cases"] != cases
            or sum(event["event"] == "attempt" for event in events) != 36 or len(outcomes) != 36
            or any(row["status"] != "accepted" for row in outcomes) or len(cases) != 12
            or summary["optimizer_updates"] != 0 or summary["qualified_for_inverse_training"]):
        raise ValueError("incomplete or mismatched event-frame evidence")
    if [(row["group"], row["pair_id"]) for row in cases] != [(row["group"], row["pair_id"]) for row in plan["pairs"]]:
        raise ValueError("selected pair membership changed")
    for row in cases:
        hashes = [event["qa"]["audio_float32_sha256"] for event in outcomes if event["label"].startswith(f"{row['pair_id']}:")]
        if hashes != row["audio_sha256"] or hashes[0] != hashes[2] or not row["full_envelopes_exact"] or not row["pair_times_exact"]:
            raise ValueError("repeat or alignment failed")
        if any(value["sampled_mean_change"] <= 1e-7 for value in row["coverage"][:3] if value["full_mean_change"] > 1e-7):
            raise ValueError("known spectral response lost")
        for record in row["files"]:
            path = directory / record["file"]
            if path.stat().st_size != record["serialized_bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError("retained feature bytes changed")
    retained = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
    if retained > plan["limits"]["archive_bytes"] or summary["projected_1024_record_archive_bytes"] > plan["gates"]["projected_1024_record_archive_bytes"]:
        raise ValueError("archive sizing gate failed")
    return {**summary, "source_hashes": {name: hashlib.sha256(value).hexdigest() for name, value in raw.items()},
            "exact_repeat_pairs": 12, "retained_archive_bytes": retained}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = summarize(args.source)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({key: value[key] for key in ("preflight_passed", "native_requests", "exact_repeat_pairs", "retained_archive_bytes")}))


if __name__ == "__main__":
    main()

"""Verify and publish the completed bounded time-coverage diagnosis."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


def summarize(directory):
    from obruxo_data.hashing import canonical_sha256
    raw = {name: (directory / name).read_bytes() for name in ("plan.json", "summary.json", "cases.json", "renders.jsonl")}
    plan, summary, cases = [json.loads(raw[name]) for name in ("plan.json", "summary.json", "cases.json")]
    events = [json.loads(line) for line in raw["renders.jsonl"].splitlines()]
    attempts = [row for row in events if row["event"] == "attempt"]
    outcomes = [row for row in events if row["event"] == "outcome"]
    if (not summary["complete"] or summary["failure"] is not None or not summary["within_resource_limits"]
            or summary["plan_id"] != canonical_sha256(plan) or summary["cases"] != cases
            or summary["native_requests"] != 36 or len(attempts) != 36 or len(outcomes) != 36
            or any(row["status"] != "accepted" for row in outcomes)):
        raise ValueError("incomplete or mismatched native coverage evidence")
    if len(cases) != 12 or [(row["group"], row["pair_id"]) for row in cases] != [(row["group"], row["pair_id"]) for row in plan["pairs"]]:
        raise ValueError("selected pair membership changed")
    for row in cases:
        hashes = [event["qa"]["audio_float32_sha256"] for event in outcomes if event["label"].startswith(f"{row['group']}:{row['pair_id']}:")]
        if (hashes != row["audio_sha256"] or hashes[0] != hashes[2] or not row["repeat_exact"] or not row["stored_frames_exact"]):
            raise ValueError("native repeat or stored-frame check failed")
    groups = {}
    for group in ("attack", "decay", "release", "detune"):
        selected = [row for row in cases if row["group"] == group]
        groups[group] = {representation: {"helper_mean": sum(row[representation]["helper"] for row in selected) / len(selected),
                                         "no_change_mean": sum(row[representation]["no_change"] for row in selected) / len(selected),
                                         "beats_no_change": sum(row[representation]["helper"] < row[representation]["no_change"] for row in selected)}
                         for representation in ("full", "sampled")}
    return {**summary, "plan": plan, "groups": groups, "exact_repeat_pairs": 12,
            "source_hashes": {name: hashlib.sha256(value).hexdigest() for name, value in raw.items()},
            "qualified_for_inverse_training": False, "qualification_changed": False,
            "retained_archive_bytes": sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = summarize(args.source)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({key: value[key] for key in ("groups", "elapsed_seconds", "peak_process_tree_rss_bytes", "retained_archive_bytes")}))


if __name__ == "__main__":
    main()

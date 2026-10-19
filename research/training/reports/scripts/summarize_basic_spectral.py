"""Publish compact A0 receipts without native audio or preset contents."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.source.read_bytes()
    source = json.loads(raw)
    rows = source["events"]
    if (not source["complete"] or not source["within_resource_limits"] or source["failure"] is not None
            or len(rows) != 12 or any(row["status"] != "accepted" for row in rows)
            or sum(row.get("repeat_exact", False) for row in rows) != 6):
        raise ValueError("A0 run is incomplete or failed")
    names = ("voices_index", "filter_index", "repeat", "source_sha256", "request_id", "status", "target_bytes",
             "measurement_seconds", "spectral_shapes", "repeat_exact", "identity_loss", "gain_loss", "pan_loss", "timing_loss")
    output = {"source_summary_sha256": hashlib.sha256(raw).hexdigest(), "plan": source["plan"],
              "events": [{**{key: row[key] for key in names if key in row},
                          "audio_sha256": row["qa"]["audio_float32_sha256"]} for row in rows],
              **{key: source[key] for key in ("complete", "elapsed_seconds", "peak_process_tree_rss_bytes", "rss_sampling_errors",
                                             "within_resource_limits", "profiles", "profile_scope", "renderer_identity")}}
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(output, indent=2, allow_nan=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

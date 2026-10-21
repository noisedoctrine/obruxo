"""Post-run training-pair diagnosis; no renders, optimizer updates or qualification changes."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


def diagnose(directory):
    import torch
    from obruxo_data.hashing import canonical_sha256
    from obruxo_training.codec import PilotTarget
    from obruxo_training.paired_spectral import complete_pairs, paired_change_loss, require_aligned_pairs
    from obruxo_training.temporal_proxy import collate_frames, hard_controls
    from obruxo_training.temporal_proxy_data import RecordCache
    from obruxo_training.temporal_proxy_screen import load_temporal_proxy

    started = perf_counter()
    torch.set_num_threads(2)
    plan = json.loads((directory / "plan.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    if not plan["config"].get("paired_changes") or not summary["complete"] or canonical_sha256(plan) != summary["plan_id"]:
        raise ValueError("diagnosis requires a completed, matching paired run")
    artifact = directory / "helper.pt"
    saved = torch.load(artifact, map_location="cpu", weights_only=True)
    if saved["plan_id"] != summary["plan_id"]:
        raise ValueError("helper belongs to a different run")
    helper, adapter = load_temporal_proxy(artifact, require_qualified=False)
    cache = RecordCache(directory, json.loads((directory / "records.json").read_text()))
    names = ("level", "pan", "detune", "cutoff", "resonance", "attack", "decay", "sustain", "release")
    groups = defaultdict(list)
    with torch.no_grad():
        for first, second in complete_pairs(cache):
            records = [cache[first], cache[second]]
            require_aligned_pairs(records)
            targets = [PilotTarget.from_dict(record["target"]) for record in records]
            category_changes = [i for i, (a, b) in enumerate(zip(targets[0].categorical, targets[1].categorical, strict=True)) if a != b]
            if category_changes:
                if len(category_changes) != 1:
                    raise ValueError("unexpected multiple category changes")
                group = ("unison", "filter_enable")[category_changes[0]]
            else:
                changed = [i for i, (a, b) in enumerate(zip(targets[0].continuous, targets[1].continuous, strict=True)) if a != b]
                if len(changed) != 1:
                    raise ValueError("expected an isolated continuous change")
                group = names[changed[0]]
            measured, times = collate_frames(records)
            predicted = helper(hard_controls(adapter, targets), torch.stack([row["performance"] for row in records]), times)
            unchanged = {"magnitudes": tuple(value[:1].expand_as(value) for value in measured["magnitudes"]),
                         "envelope": measured["envelope"][:1].expand_as(measured["envelope"])}
            error = float(paired_change_loss(predicted, measured).item())
            zero_response = float(paired_change_loss(unchanged, measured).item())
            groups[group].append({"pair_id": records[0]["pair_id"], "error": error, "zero_response_error": zero_response})
    report = {name: {"pairs": len(rows), "mean_change_error": sum(row["error"] for row in rows) / len(rows),
                     "mean_zero_response_error": sum(row["zero_response_error"] for row in rows) / len(rows),
                     "beats_zero_response": sum(row["error"] < row["zero_response_error"] for row in rows)}
              for name, rows in groups.items()}
    with artifact.open("rb") as handle:
        helper_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"claim": "posthoc_training_fit_diagnosis_not_validation_or_qualification", "plan_id": summary["plan_id"],
            "helper_sha256": helper_hash, "records_sha256": hashlib.sha256((directory / "records.json").read_bytes()).hexdigest(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "groups": report, "pairs": dict(groups),
            "native_requests": 0, "optimizer_updates": 0, "elapsed_seconds": perf_counter() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = diagnose(args.source)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"groups": value["groups"], "elapsed_seconds": value["elapsed_seconds"]}))


if __name__ == "__main__":
    main()

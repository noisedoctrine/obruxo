"""Summarize existing development evidence; no rendering, training or promotion."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def summarize(data):
    names = ("state_coverage_summary.json", "fresh_validation_summary.json")
    raw = {name: (data / name).read_bytes() for name in names}
    original, added = [json.loads(raw[name]) for name in names]
    methods = ("expanded_16000", "expanded_seed0_16000")
    panels = []
    for label, count, method in zip(("original_development", "additional_development"), (21, 23), methods, strict=True):
        metrics = {}
        for candidate in (method, "fixed_template", "train_retrieval"):
            if count == 21:
                scores = original["validation"]["aggregate_metrics"][candidate]
                if any(scores[role]["scored"] != count or scores[role]["failed_or_unscorable"] for role in ("reference", "transfer")):
                    raise ValueError("original development panel is incomplete")
                reference, transfer = [scores[role]["mean_mrstft"] for role in ("reference", "transfer")]
                level = {role: scores[role]["mean_rms_relative"] for role in ("reference", "transfer")}
            else:
                scores = added["metrics"][candidate]
                if scores["planned"] != count or scores["scored"] != count:
                    raise ValueError("additional development panel is incomplete")
                reference, transfer = scores["reference_mean"], scores["transfer_mean"]
                level = {role: scores[f"{role}_rms_relative"] for role in ("reference", "transfer")}
            if not all(math.isfinite(value) and value >= 0 for value in (reference, transfer, *level.values())):
                raise ValueError("invalid historical metric")
            metrics[candidate] = {"primary_mean": (reference + transfer) / 2, "reference_mean": reference,
                                  "transfer_mean": transfer, "rms_relative": level}
        panels.append({"panel": label, "cases": count, "candidate": method, "metrics": metrics})

    cases = added["cases"]
    scores = added["metrics"][methods[1]]["state_scores"]
    if len(cases) != 23 or len(scores) != 23 or len({case["state_id"] for case in cases}) != 23:
        raise ValueError("case selection requires the complete 23-state panel")
    if any(case["case"] != index for index, case in enumerate(cases)) or not all(math.isfinite(score) for score in scores):
        raise ValueError("invalid case order or score")
    ordered = sorted(range(len(cases)), key=lambda index: (scores[index], cases[index]["state_id"]))
    # Nearest ranks for minimum, lower quartile, median, upper quartile, maximum.
    ranks = (0, 5, 11, 17, 22)
    selected = [{**cases[ordered[rank]], "rank_1_based": rank + 1, "historical_primary_error": scores[ordered[rank]]} for rank in ranks]
    return {"policy": "basic_reconstruction_development_audit_v1", "training_updates": 0, "new_renders": 0,
            "test_split_evaluated": False, "promotion": "none", "validation_status": "both_panels_already_used_for_development",
            "source_sha256": {name: hashlib.sha256(content).hexdigest() for name, content in raw.items()},
            "panels": panels,
            "original_parameter_diagnostics": original["parameter_diagnostics"][methods[0]],
            "additional_parameter_diagnostics": added["metrics"][methods[1]]["parameter_diagnostics"],
            "diagnostic_selection": {"candidate": methods[1], "policy": "nearest_rank_five_number_summary_v1", "cases": selected},
            "limits": ["spectral error is not calibrated perceptual similarity", "parameter errors do not establish causes of sound errors",
                       "selected cases are diagnostics, not a new validation panel", "no listening or intervention result yet"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"panels": [panel["cases"] for panel in result["panels"]], "diagnostic_cases": len(result["diagnostic_selection"]["cases"]), "new_renders": 0}))


if __name__ == "__main__":
    main()

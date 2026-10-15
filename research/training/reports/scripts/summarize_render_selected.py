"""Publish checked aggregate evidence for direct-render target selection and learning."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]
from obruxo_data.hashing import canonical_sha256  # noqa: E402
from obruxo_training.candidate_data import dataset_digest  # noqa: E402
from obruxo_training.evaluate import curve_statistics  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--tests-passed", type=int, required=True)
    args = parser.parse_args()
    plan = json.loads((args.run / "plan.json").read_text())
    summary = json.loads((args.run / "summary.json").read_text())
    comparison = json.loads((args.run / "comparison/summary.json").read_text())
    data = torch.load(args.run / "data/data.pt", map_location="cpu", weights_only=True)
    if (summary["plan_id"] != canonical_sha256(plan) or data["sha256"] != dataset_digest(data)
            or summary["collection"]["data_sha256"] != data["sha256"]
            or [row["state_id"] for row in comparison["cases"]] != plan["validation_states"]):
        raise ValueError("incomplete or changed direct-render evidence")
    for name, digest in plan["source"].items():
        if hashlib.sha256((RESEARCH / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"experiment source changed: {name}")
    pair = curve_statistics(comparison["cases"], ["render_selected"], baseline_name="control")
    selected = [row for row in data["cases"] if row["selected"] is not None]
    gains = [(np.mean(row["scores"][row["selected"]]) - np.mean(row["scores"]["model"])) / np.mean(row["scores"]["model"])
             for row in selected]
    summary.update(paired_render_selected_vs_control=pair,
                   confirmed_target_summary={"cases": len(selected), "selected_methods": dict(Counter(row["selected"] for row in selected)),
                                             "mean_relative_native_change": float(np.mean(gains)) if gains else None},
                   native_outcomes={"training": dict(Counter(row["status"] for row in data["history"])),
                                    "validation_attempted": comparison["renders_attempted"], "validation_accepted": comparison["renders_accepted"]},
                   artifacts_sha256={name: hashlib.sha256((args.run / name).read_bytes()).hexdigest()
                                     for name in ("plan.json", "data/data.pt", "control/checkpoint.pt", "render_selected/checkpoint.pt", "comparison/summary.json")},
                   verification={"full_local_tests_passed": args.tests_passed, "github_actions": "disabled"})
    reports = Path(__file__).resolve().parents[1]
    for name, value in (("render_selected_plan.json", plan), ("render_selected_data_plan.json", data["plan"]), ("render_selected_summary.json", summary)):
        (reports / "data" / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    models = comparison["comparison"]["checkpoints"]
    values = pair["checkpoints"]["render_selected"]["against_control"]
    control_mean = models["control"]["mean_reference_transfer_mrstft"]
    candidate_mean = models["render_selected"]["mean_reference_transfer_mrstft"]
    change = 100 * (candidate_mean / control_mean - 1)
    lines = ["# Direct-render selection and learning", "",
             f"Vital confirmed alternative patch guesses for **{len(selected)}/{plan['config']['training_states']} training recordings**. Learning from those extra targets changed mean held-out native error by **{change:+.2f}%** relative to the matched source-supervised control. This fixed single-seed authored-alpha comparison does not promote a release model.", "",
             "[Protocol](../RENDER_SELECTED_TRAINING.md) · [Frozen screen](data/render_selected_plan.json) · [Collection plan](data/render_selected_data_plan.json) · [Aggregate evidence](data/render_selected_summary.json)", "",
             "## What was compared", "",
             "Thirty-two training patches were selected outside the source inverse checkpoint's retained training states. Proposals used its ordinary prediction, training-only audio retrieval, one categorical alternative and a bounded continuous change. The source patch label and known notes never entered the proposer. Actual training MIDI was used only to render and score proposals.", "",
             "The best proposal and ordinary prediction each received an independent confirmation render. An extra target was retained only when the candidate's worse draw beat the ordinary prediction's better draw by at least 5%. Ground-truth parameter supervision remained an anchor in both training arms.", "",
             f"Both arms received identical warm-start weights, optimizer initialization, old/new feature minibatches and draw counts for {plan['config']['steps']} updates. The sole objective difference was confirmed-target weight zero versus {plan['config']['candidate_weight']}. Validation used ordinary single predictions, not evaluator best-of-four.", "",
             "| Model | Mean reference/transfer MR-STFT |",
             "|---|---:|", f"| Matched source-supervised control | {control_mean:.4f} |", f"| Additional confirmed targets | {candidate_mean:.4f} |", "",
             f"Candidate-minus-control paired mean: **{values['mean_delta']:+.4f}**; descriptive 95% state-bootstrap interval: **{values['ci95']}**. Candidate wins: **{values['wins']}/{values['planned']}** states; ties: {values['ties']}. Both performances of a patch stay in the same resampling unit. The validation patches are reused development cases, not a fresh final test.", "",
             "## Native evidence and compute", "",
             f"Training selection accepted {summary['collection']['accepted']}/{summary['collection']['attempted']} native attempts, with {summary['collection']['accepted_sources']} accepted sources. Validation accepted {comparison['renders_accepted']}/{comparison['renders_attempted']} renders. Failed and unscored cases retain their planned denominators. Confirmed target methods: {summary['confirmed_target_summary']['selected_methods']}.", "",
             f"Selection took {summary['collection']['elapsed_seconds_this_segment'] / 60:.2f} CPU minutes after source loading. The matched learning arms took {sum(arm['elapsed_seconds_this_segment'] for arm in summary['arms'].values()) / 60:.2f} minutes; native validation took {comparison['elapsed_seconds'] / 60:.2f} minutes after checkpoint loading. These segments exclude plan/pool and repeated contract verification, and some work overlapped the local software suite. They are not a dedicated throughput benchmark.", "",
             f"Retained feature storage including the original bank was {summary['collection']['feature_bytes_including_source'] / 2**20:.2f} MiB. Sampled selection peak RSS was {summary['collection']['peak_process_tree_rss_bytes'] / 2**30:.2f} GiB. Execution used local CPU float32 and four threads; no paid compute or audio archive was used.", "",
             f"The full local training/renderer suite passes **{args.tests_passed} tests**. New checks cover legal proposals, training isolation, independent confirmation, QA rejection, feature/cache integrity, exact optimizer recovery, matched minibatches, portable audio-only exports and render-attempt caps including recovery. GitHub Actions stays disabled.", "",
             "## Remaining plan", "",
             "The next independent capability is deployed audio-only candidate ranking, evaluated separately from oracle best-of-four. A conditioning comparison, broader performance/control coverage, measured data scaling and release gates remain open in the completion ledger. A successful training-case alternative does not itself establish improved inverse generalization, and a small development comparison does not complete the finished-model plan. Final test states remain reserved.", ""]
    (reports / "RENDER_SELECTED_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"relative_native_error_change_percent": change, "paired": values, "confirmed_targets": len(selected)}, indent=2))


if __name__ == "__main__":
    main()

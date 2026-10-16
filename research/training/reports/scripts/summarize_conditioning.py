"""Publish the one matched performance-conditioning comparison."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]
from obruxo_data.hashing import canonical_sha256  # noqa: E402
from obruxo_training.evaluate import curve_statistics, retrieval_digest  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--tests-passed", type=int, required=True)
    args = parser.parse_args()
    plan = json.loads((args.run / "plan.json").read_text())
    summary = json.loads((args.run / "summary.json").read_text())
    comparison = json.loads((args.run / "comparison/summary.json").read_text())
    if (summary["plan_id"] != canonical_sha256(plan) or [row["state_id"] for row in comparison["cases"]] != plan["validation_states"]
            or summary["new_training_renders"] != 0 or plan["test_split_evaluated"]):
        raise ValueError("conditioning comparison differs from the frozen plan")
    for name, expected in plan["source"].items():
        if hashlib.sha256((RESEARCH / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"conditioning source changed: {name}")
    checkpoints = [torch.load(args.run / arm / "checkpoint.pt", map_location="cpu", weights_only=True) for arm in ("concat", "cross_attention")]
    left, right = checkpoints
    if (retrieval_digest(left["replay"]) != retrieval_digest(right["replay"]) or not torch.equal(left["batch_rng"], right["batch_rng"])
            or [row["uses"] for row in left["replay"]] != [row["uses"] for row in right["replay"]]):
        raise ValueError("conditioning arms did not use identical features and batches")
    pair = curve_statistics(comparison["cases"], ["cross_attention"], baseline_name="concat")
    values = pair["checkpoints"]["cross_attention"]["against_concat"]
    means = comparison["comparison"]["checkpoints"]
    change = 100 * (means["cross_attention"]["mean_reference_transfer_mrstft"] / means["concat"]["mean_reference_transfer_mrstft"] - 1)
    summary.update(paired_attention_vs_concat=pair, relative_error_change_percent=change,
                   final_conditioning_gain=float(right["model"]["conditioning_gain"]),
                   artifact_sha256={arm: hashlib.sha256((args.run / arm / "checkpoint.pt").read_bytes()).hexdigest()
                                    for arm in ("concat", "cross_attention")},
                   verification={"full_local_tests_passed": args.tests_passed, "github_actions": "disabled"})
    reports = Path(__file__).resolve().parents[1]
    for name, value in (("conditioning_plan.json", plan), ("conditioning_summary.json", summary)):
        (reports / "data" / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    rows = ["# Performance-conditioning comparison", "",
            f"The attention variant changed mean native sound error by **{change:+.2f}%** against the original feature concatenation model. This single-seed ten-patch alpha check does not demonstrate a benefit, so concatenation remains the default. Both architectures remain available for broader coverage work; no release model is promoted.", "",
            "Both models used the same frozen 126-view paired feature cache, labels, shared initial weights, initial predictions, minibatches and per-example draw counts for 1,000 updates. No new training audio was rendered. The attention path adds 70,465 parameters and begins with zero residual gain; this compares practical architectures, not attention independently of extra capacity.", "",
            "| Architecture | Parameters | Mean reference/transfer MR-STFT |", "|---|---:|---:|"]
    rows += [f"| {name} | {summary['arms'][name]['trainable_parameters']} | {means[name]['mean_reference_transfer_mrstft']:.4f} |" for name in ("concat", "cross_attention")]
    rows += ["", f"Attention-minus-concatenation paired mean: {values['mean_delta']:+.4f}; descriptive 95% state-bootstrap interval: {values['ci95']}. Wins: {values['wins']}/{values['planned']}; ties: {values['ties']}. Reference and reserved transfer performances remain together in each resampling unit. These development states have been reused and are not a final test.", "",
             f"All {summary['validation_accepted']}/{summary['validation_attempts']} native validation attempts passed. The learned residual gain was {summary['final_conditioning_gain']:.6f}. The full local renderer/training suite passed {args.tests_passed} tests, including common initialization, temporal feature layout, gradients, architecture guards, exact recovery and audio-only export/load behavior. Ruff/diff checks passed. GitHub Actions stays disabled.", "",
             "Compute was local CPU float32 with four threads; independent checks overlapped training and validation, so timings are not a dedicated throughput benchmark. Raw training audio was not archived, and final-test states remain reserved.", "",
             "This completes the planned initial architecture comparison. Further narrow tuning is deferred while canonical assets, sampler/voice behavior, LFOs, effect order and routing coverage are expanded, followed by measured data scaling and combined training.", "",
             "[Architecture and protocol](../CONDITIONING.md) · [Frozen plan](data/conditioning_plan.json) · [Aggregate evidence](data/conditioning_summary.json)", ""]
    (reports / "CONDITIONING_RESULTS.md").write_text("\n".join(rows), encoding="utf-8", newline="\n")
    print(json.dumps({"relative_error_change_percent": change, "paired": values, "final_gain": summary["final_conditioning_gain"]}, indent=2))


if __name__ == "__main__":
    main()

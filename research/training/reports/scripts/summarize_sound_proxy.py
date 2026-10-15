"""Publish aggregate evidence from a completed, frozen native sound-proxy screen."""

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
from obruxo_training.features import tensor_digest  # noqa: E402
from obruxo_training.proxy_screen import data_digest, summarize  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reports", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--tests-passed", type=int, required=True)
    args = parser.parse_args()
    plan = json.loads((args.run / "plan.json").read_text())
    summary = json.loads((args.run / "summary.json").read_text())
    saved = torch.load(args.run / "checkpoint.pt", map_location="cpu", weights_only=True)
    rows = saved["rows"]
    renders = [json.loads(line) for line in (args.run / "renders.jsonl").read_text().splitlines()]
    if (summary["plan_id"] != canonical_sha256(plan) or not summary["complete_panel"]
            or saved["plan_id"] != summary["plan_id"] or canonical_sha256(rows) != saved["rows_sha256"]
            or data_digest(saved["records"]) != saved["data_sha256"]
            or tensor_digest(saved["training"]["model"]) != summary["model_sha256"]
            or canonical_sha256(renders[:saved["log_count"]]) != saved["log_sha256"]
            or len(rows) != 2 * len(plan["validation_states"]) or len(renders) != summary["render_attempts"]):
        raise ValueError("only complete, matching planned evidence may be published")
    if any(summary[key] != value for key, value in summarize(rows, plan).items()):
        raise ValueError("reported qualification differs from completed checkpoint cases")
    for row in rows:
        path = args.run / f"case-{row['case']:03d}.json"
        if path.exists() and json.loads(path.read_text()) != row:
            raise ValueError("completed case export changed")
        if not path.exists():
            path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for name, digest in plan["source"].items():
        if hashlib.sha256((RESEARCH / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"experiment source changed: {name}")
    by_state, families = {}, {}
    for row in rows:
        scored = "large" in row["candidates"] and "guided" in row["candidates"]
        delta = row["candidates"]["guided"]["mrstft"] - row["candidates"]["large"]["mrstft"] if scored else None
        if delta is not None:
            by_state.setdefault(row["state_id"], []).append(delta)
        family = families.setdefault(row["family"], {"planned_performances": 0, "forward": [], "gradient_deltas": []})
        family["planned_performances"] += 1
        if "forward_error" in row:
            family["forward"].append(row["forward_error"])
        if delta is not None:
            family["gradient_deltas"].append(delta)
    family_results = {}
    for name, values in families.items():
        errors = values["forward"]
        deltas = values["gradient_deltas"]
        family_results[name] = {"planned_performances": values["planned_performances"], "scored_forward": len(errors),
                                "mean_forward_errors": {key: float(np.mean([item[key] for item in errors])) for key in errors[0]} if errors else {},
                                "scored_gradient": len(deltas), "mean_native_gradient_delta": float(np.mean(deltas)) if deltas else None}
    state_deltas = np.array([np.mean(values) for values in by_state.values() if len(values) == 2])
    interval = None
    if len(state_deltas):
        samples = np.random.default_rng(0).integers(len(state_deltas), size=(2000, len(state_deltas)))
        interval = np.quantile(state_deltas[samples].mean(1), [.025, .975]).tolist()
    updates = [row for row in rows if "proxy_update" in row and "guided" in row["candidates"] and "large" in row["candidates"]]
    update_diagnostics = {"scored": len(updates),
                          "lower_predicted_error": sum(row["proxy_update"]["final_proxy_error"] < row["proxy_update"]["initial_proxy_error"] for row in updates),
                          "lower_native_error": sum(row["candidates"]["guided"]["mrstft"] < row["candidates"]["large"]["mrstft"] for row in updates),
                          "higher_native_error": sum(row["candidates"]["guided"]["mrstft"] > row["candidates"]["large"]["mrstft"] for row in updates)}
    summary.update(by_family=family_results,
                   update_diagnostics=update_diagnostics,
                   artifacts_sha256={name: hashlib.sha256((args.run / name).read_bytes()).hexdigest()
                                     for name in ("plan.json", "summary.json", "checkpoint.pt", "proxy.pt", "renders.jsonl")},
                   paired_state_gradient={"planned_states": len(plan["validation_states"]), "complete_states": len(state_deltas),
                                          "mean_delta": float(state_deltas.mean()) if len(state_deltas) else None,
                                          "ci95": interval, "bootstrap_resamples": 2000, "seed": 0, "descriptive_only": True},
                   native_outcomes={stage: dict(Counter(row["status"] for row in renders if row["stage"] == stage))
                                    for stage in ("training", "validation")},
                   rejected_diagnostics=dict(Counter(code for row in renders if row["status"] != "accepted" for code in row["diagnostic_codes"])),
                   render_wall_seconds=sum(row["wall_seconds"] for row in renders),
                   verification={"full_local_tests_passed": args.tests_passed, "github_actions": "disabled"})
    data = args.reports / "data"
    data.mkdir(parents=True, exist_ok=True)
    (data / "sound_proxy_plan.json").write_bytes((args.run / "plan.json").read_bytes())
    (data / "sound_proxy_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    qualified = summary["qualified_for_inverse_training"]
    conclusion = ("The helper passes this bounded feasibility gate. A subsequent matched inverse-training comparison is still required."
                  if qualified else "The helper does not pass the frozen native qualification gate. It is not connected to inverse training.")
    lines = ["# Sound-prediction helper: native feasibility result", "", conclusion, "",
             "The helper predicts fixed sound measurements from patch controls and actual notes. The check asks whether those predictions generalize and whether its suggested changes improve real Vital audio. This is a single-seed screen on the authored alpha vocabulary, not a finished-model quality claim.", "",
             "[Frozen plan](data/sound_proxy_plan.json) · [Aggregate evidence](data/sound_proxy_summary.json) · [Protocol](../AUDIO_PROXY.md)", "",
             "## Held-out prediction", "",
             "Errors below use equal measurement-group weights and training-only coordinate scales. They are helper-regression errors, separate from native MR-STFT sound distance. Lower is better.", "",
             "| Performance | Scored/planned | Helper | Training mean | Nearest training | Five-neighbor interpolation | Gate |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for role, values in summary["forward_errors"].items():
        means = values["mean_errors"]
        numbers = [f"{means[name]:.4f}" if means[name] is not None else "unscored" for name in ("proxy", *plan["baselines"])]
        lines.append(f"| {role} | {values['scored']}/{values['planned']} | " + " | ".join(numbers) + f" | {'pass' if values['passed'] else 'fail'} |")
    ranking, gradient = summary["candidate_ranking"], summary["gradient_probe"]
    lines += ["", "## Advice checked against Vital", "",
              f"Candidate rankings agree with real renders on **{ranking['concordant']}/{ranking['scored']} scored comparisons**; {ranking['noise_or_proxy_ties']} comparisons are ties under the frozen repeat-noise rule, out of {ranking['planned_comparisons']} planned. Missing comparisons stay in the planned denominator.", "",
              f"Helper-guided updates improve real sound beyond source-repeat noise on **{gradient['improvements_beyond_repeat_noise']}/{gradient['scored']} scored performances**, out of {gradient['planned']} planned. Mean guided-minus-starting native MR-STFT change: **{gradient['mean_native_mrstft_delta']}**. Negative means improvement. The paired-state bootstrap interval is **{interval}**; it is descriptive and resamples patches, not individual correlated performances.", "",
              f"The update lowered the helper's predicted error on {update_diagnostics['lower_predicted_error']}/{update_diagnostics['scored']} cases. Real error decreased on {update_diagnostics['lower_native_error']} and increased on {update_diagnostics['higher_native_error']}. Prediction accuracy and useful input gradients are separate requirements; this result diagnoses this helper and bounded update, not the underlying need for audio-based training.", "",
              "| Synthesis family | Forward scored/planned | Helper error | Nearest-training error | Native guided-minus-starting change |",
              "|---|---:|---:|---:|---:|"]
    for name, values in family_results.items():
        means = values["mean_forward_errors"]
        values_to_show = [means.get("proxy"), means.get("nearest_training_patch_and_performance"), values["mean_native_gradient_delta"]]
        numbers = [f"{value:.4f}" if value is not None else "unscored" for value in values_to_show]
        lines.append(f"| {name} | {values['scored_forward']}/{values['planned_performances']} | " + " | ".join(numbers) + " |")
    lines += ["", "## Data, compute and limits", "",
              f"Native QA accepted **{summary['renders_accepted']}/{summary['render_attempts']} attempts**. Outcomes by phase: `{summary['native_outcomes']}`. Rejected diagnostics: `{summary['rejected_diagnostics']}`. Accepted training data contains {summary['training_records']} measurements from {summary['training_states']} patches, covering {summary['training_families']}.", "",
              f"The helper has {summary['trainable_parameters']:,} parameters. Retained measurements occupy {summary['retained_feature_bytes'] / 2**20:.2f} MiB; sampled peak process-tree memory is {(summary['peak_process_tree_rss_bytes'] or 0) / 2**30:.2f} GiB. The CPU run took {summary['elapsed_seconds_this_segment'] / 60:.2f} minutes after pool/plan preparation. Summed native render wall time was {summary['render_wall_seconds'] / 60:.2f} minutes. Four CPU threads, float32, no paid compute and no audio archive were used.", "",
              "Training uses only the four reference performances. Reserved performances change register, velocity, timing and/or note combinations; their input range is broader than this small training bank. Reference and transfer results therefore remain separate. This screen cannot establish performance across the full Vital vocabulary or real recordings. Linux/Colab native rendering remains unqualified.", "",
              f"The complete local training/renderer suite passes **{args.tests_passed} tests**, including fixed measurement behavior, active/inactive input gradients, exact optimizer recovery, QA rejection, native collection recovery, artifact loading, data tampering and qualification gates. Ruff and whitespace checks are recorded separately in the PR. GitHub Actions stays disabled.", "",
              "These are development validation patches, including patches used in earlier screens. The interval does not account for repeated development decisions. The final test reserve is untouched. The inverse model and selected release weights are unchanged. A passed helper gate permits a later inverse-training comparison; it does not establish that comparison's outcome.", "",
              "## Next planned capability", "",
              "Implement bounded candidate improvement checked directly with Vital. This also provides the training plan's fallback route to better audio targets without relying on this helper's gradients. Larger proxy runs, tuning and seed replication remain deferred while that broad direction is screened. Audio-based training remains a requirement; this failed helper/update is not evidence against it.", ""]
    (args.reports / "SOUND_PROXY_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"qualified_for_inverse_training": qualified, "render_attempts": summary["render_attempts"],
                      "report": str(args.reports / "SOUND_PROXY_RESULTS.md")}, indent=2))


if __name__ == "__main__":
    main()

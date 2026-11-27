"""Check Euler-noise sensitivity for the two strongest issue 183 flow arms."""

import argparse
from pathlib import Path
import sys

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING = SCRIPT_DIR.parents[1]
RESEARCH = TRAINING.parent
WORKTREE = TRAINING.parents[1]
MAIN_ROOT = WORKTREE.parent.parent if WORKTREE.parent.name == "tmp" else WORKTREE
sys.path.insert(0, str(SCRIPT_DIR))
sys.path[:0] = [str(RESEARCH / "data_generation"), str(TRAINING)]

import run_issue183_flow_matching as pilot

from obruxo_training.codec import PilotCodec
from obruxo_training.flow_matching import FlowSpectrumModel
from obruxo_training.surrogate_codec import SurrogateCodec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--checkpoint-source", type=Path,
                        help="completed training run whose checkpoints should be evaluated")
    parser.add_argument("--report-file", type=Path,
                        help="optional markdown report to append the sensitivity table to")
    parser.add_argument("--validation-source", type=Path,
                        default=MAIN_ROOT / "research/training/outputs/issue173-model-selection-active-20261126-v4")
    parser.add_argument("--device", choices=("xpu", "cpu"), default="xpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    checkpoint_source = args.checkpoint_source or args.run_output
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "xpu" and not torch.xpu.is_available():
        raise RuntimeError("configured XPU is unavailable")
    validation_frozen = pilot._load_json(args.validation_source / "plan.json")
    source = validation_frozen.get("plan", validation_frozen)
    rows = source["evaluation"]["validation_proxy"]["pair_recipes"]
    if len(rows) != 128:
        raise ValueError("noise sensitivity must reuse the frozen 128-pair validation panel")
    codec = SurrogateCodec(PilotCodec(), expanded=True)
    summary = pilot._load_json(args.run_output / "summary.json")
    finalists = sorted(("residual_mlp", "transformer", "conformer"),
                       key=lambda name: summary["evaluation"][f"{name}-flow"]["4"]["spectral_loss"])[:2]
    verified = pilot.verify_completed_run(checkpoint_source, args.validation_source,
                                          [f"{name}-flow" for name in finalists])
    summary_plan_id = summary.get("source_plan_id", summary.get("plan_id"))
    if summary_plan_id != verified["plan_id"]:
        raise ValueError("evaluation results belong to a different frozen training run")
    if summary.get("source_checkpoint_sha256"):
        for arm, digest in verified["checkpoint_sha256"].items():
            if summary["source_checkpoint_sha256"].get(arm) != digest:
                raise ValueError(f"evaluation results use a different checkpoint: {arm}")
    noise_seeds, step_counts = (90183, 90184, 90185), (4, 8, 16)
    results, per_case = {}, {}
    for backbone in finalists:
        arm = f"{backbone}-flow"
        saved = torch.load(checkpoint_source / f"{arm}.pt", map_location="cpu", weights_only=True)
        model = FlowSpectrumModel(backbone, codec.width).to(device)
        model.load_state_dict(saved["model"])
        model.eval()
        results[arm], per_case[arm] = {}, {}
        for steps in step_counts:
            results[arm][str(steps)], per_case[arm][str(steps)] = {}, {}
            for seed in noise_seeds:
                metrics, cases = pilot.evaluate_model(
                    model, "flow", rows, args.validation_source / "validation-pairs", codec, device,
                    flow_steps=steps, seed=seed)
                results[arm][str(steps)][str(seed)] = metrics
                per_case[arm][str(steps)][str(seed)] = cases
                print(f"completed {arm}, Euler {steps}, noise seed {seed}: spectral loss {metrics['spectral_loss']:.4f}", flush=True)
                if device.type == "xpu":
                    torch.xpu.synchronize(device)
        del model
    fields = ("spectral_loss", "octave_band_energy_error_db", "monophonic_octave_energy_error_db",
              "attack_envelope_error_db",
              "decay_envelope_error_db", "sustain_envelope_error_db", "release_envelope_error_db")
    ranges = {}
    for arm, step_results in results.items():
        ranges[arm] = {}
        for steps, seeds in step_results.items():
            ranges[arm][steps] = {
                metric: {"values": [float(seeds[str(seed)][metric]) for seed in noise_seeds],
                         "mean": float(sum(seeds[str(seed)][metric] for seed in noise_seeds) / len(noise_seeds)),
                         "min": float(min(seeds[str(seed)][metric] for seed in noise_seeds)),
                         "max": float(max(seeds[str(seed)][metric] for seed in noise_seeds))}
                for metric in fields}
    output = {"format": "issue183_flow_noise_sensitivity_v1", "source_plan_id": verified["plan_id"],
              "source_checkpoint_sha256": verified["checkpoint_sha256"],
              "selected_by": "best 4-step validation spectral loss among three flow backbones on the same exploratory holdout",
              "validation_patch_roots": len({row["state_id"] for row in rows}),
              "validation_pairs": len(rows), "validation_clips": 2 * len(rows),
              "midi_performances": len({row["midi_id"] for row in rows}),
              "finalists": finalists, "noise_seeds": list(noise_seeds),
              "euler_steps": list(step_counts), "aggregate_by_seed": results, "seed_ranges": ranges,
              "per_case_by_seed": per_case}
    pilot.write_json(args.run_output / "noise-sensitivity.json", output)
    summary["noise_sensitivity"] = {key: value for key, value in output.items() if key != "per_case_by_seed"}
    pilot.write_json(args.run_output / "summary.json", summary)
    if args.report_file:
        append_report(args.report_file, output)


def append_report(path, output):
    lines = ["", "## Flow sampler noise sensitivity", "",
             "We re-evaluated the two flow arms with the lowest 4-step validation spectral loss, using the same 128 paired validation recipes under three fixed Gaussian initializations and all three Euler step counts. This is a sensitivity check on the same exploratory holdout, not an independent confirmation.", "",
             "| Flow arm | Euler steps | Spectral loss (mean; min–max) | Octave-band energy error dB (mean; min–max) |", "|---|---:|---:|---:|"]
    for arm in output["finalists"]:
        arm_key = f"{arm}-flow"
        for steps in output["euler_steps"]:
            row = output["seed_ranges"][arm_key][str(steps)]
            spectral, octave = row["spectral_loss"], row["octave_band_energy_error_db"]
            lines.append(f"| {arm_key} | {steps} | {spectral['mean']:.4f} ({spectral['min']:.4f}–{spectral['max']:.4f}) | {octave['mean']:.4f} ({octave['min']:.4f}–{octave['max']:.4f}) |")
    lines += ["", "Noise seeds and per-case scores are in [noise-sensitivity.json](noise-sensitivity.json).", ""]
    text = path.read_text(encoding="utf-8")
    marker = "## Execution and reproducibility"
    if marker not in text:
        raise ValueError("expected report section is missing")
    if "## Flow sampler noise sensitivity" not in text:
        text = text.replace(marker, "\n".join(lines) + marker, 1)
        path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

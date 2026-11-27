"""Measure fixed-subset training reconstruction across independent flow noise starts."""

import argparse
from pathlib import Path
import sys

import torch

TRAINING = Path(__file__).resolve().parents[2]
RESEARCH = TRAINING.parent
sys.path[:0] = [str(RESEARCH / "data_generation"), str(TRAINING)]

import run_issue183_flow_matching as pilot

from obruxo_training.codec import PilotCodec
from obruxo_training.flow_matching import FlowSpectrumModel
from obruxo_training.surrogate_codec import SurrogateCodec


def summarize(values):
    return {"values": values, "mean": sum(values) / len(values), "min": min(values), "max": max(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--training-source", type=Path, default=pilot.TRAINING_SOURCE)
    parser.add_argument("--validation-source", type=Path, default=pilot.VALIDATION_SOURCE)
    parser.add_argument("--device", choices=("xpu", "cpu"), default="xpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    source, training_source = args.run_output.resolve(), args.training_source.resolve()
    validation_source = args.validation_source.resolve()
    verified = pilot.verify_completed_run(source, validation_source)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    if device.type == "xpu" and not torch.xpu.is_available():
        raise RuntimeError("configured XPU unavailable")

    scratch = source / "training-probe-noise-evaluation"
    if scratch.exists():
        raise ValueError("training probe scratch directory already exists")
    scratch.mkdir()
    progress = pilot.Progress(scratch, pilot.TRAINING_PROBE_PAIRS)
    progress.status["arm_count"] = len(pilot.BACKBONES)
    _, _, training_rows, _, _ = pilot.load_sources(training_source, validation_source, scratch, progress)
    rows = pilot.fixed_training_probe(training_rows)
    if len(rows) != pilot.TRAINING_PROBE_PAIRS or len({row["state_id"] for row in rows}) != len(rows):
        raise ValueError("fixed training subset must contain one recipe from each of 16 patch roots")

    codec = SurrogateCodec(PilotCodec(), expanded=True)
    output = {"format": "issue183_training_probe_noise_sensitivity_v1",
              "source_plan_id": verified["plan_id"], "noise_seeds": list(pilot.TRAINING_PROBE_NOISE_SEEDS),
              "sampler_steps": 8, "recipes": len(rows), "clips": 2 * len(rows),
              "patch_roots": len({row["state_id"] for row in rows}), "arms": {},
              "source_checkpoint_sha256": {f"{backbone}-flow": verified["checkpoint_sha256"][f"{backbone}-flow"]
                                            for backbone in pilot.BACKBONES}}
    for index, backbone in enumerate(pilot.BACKBONES, 1):
        arm = f"{backbone}-flow"
        saved = torch.load(source / f"{arm}.pt", map_location="cpu", weights_only=True)
        model = FlowSpectrumModel(backbone, codec.width).to(device)
        model.load_state_dict(saved["model"])
        model.eval()
        per_seed = [pilot.evaluate_flow_training_probe(
            model, rows, training_source / "pairs", codec, device, seed=seed,
            progress=progress, phase=f"{arm} · training probe")
            for seed in pilot.TRAINING_PROBE_NOISE_SEEDS]
        fields = ("velocity_mse", "sampled_spectral_loss", "sampled_octave_band_energy_error_db")
        output["arms"][arm] = {"per_seed": per_seed,
                               **{field: summarize([float(item[field]) for item in per_seed]) for field in fields}}
        if device.type == "xpu":
            torch.xpu.synchronize(device)
        del model
        progress.update(phase=f"Completed training probe {arm}", completed=pilot.TRAINING_PROBE_PAIRS,
                        total=pilot.TRAINING_PROBE_PAIRS, arm_index=index, force=True)

    output_path = source / "training-probe-noise-sensitivity.json"
    pilot.write_json(output_path, output)
    progress.update(phase="Training probe noise sensitivity complete", completed=pilot.TRAINING_PROBE_PAIRS,
                    total=pilot.TRAINING_PROBE_PAIRS, arm_index=len(pilot.BACKBONES), state="complete", force=True)
    print(f"Wrote training-subset probe results to {output_path}", flush=True)


if __name__ == "__main__":
    main()

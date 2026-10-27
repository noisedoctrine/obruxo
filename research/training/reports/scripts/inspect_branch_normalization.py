"""Post-hoc read-only audit of the information exposed to the temporary note head."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

REPOSITORY = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(REPOSITORY / "research/training"), str(REPOSITORY / "research/data_generation")]


def main():
    import torch
    from torch.nn import functional as F
    from obruxo_data.hashing import canonical_sha256
    from obruxo_training.audit import write_json
    from obruxo_training.basic_direct import restore_direct
    from obruxo_training.branch_adversary import initialize, note_targets
    from obruxo_training.category_scale_refresh import cpu_settings
    from obruxo_training.conditioned_bridge import read
    from obruxo_training.features import tensor_digest
    from obruxo_training.native_fit import digest
    from obruxo_training.representation_probes import regression_metrics, ridge_predict
    from run_branch_adversary_screen import encode
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("initial", "features", "prior", "candidate", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    plan, summary = read(args.candidate / "plan.json"), read(args.candidate / "summary.json")
    prior = read(args.prior / "summary.json")
    if (summary["plan_id"] != canonical_sha256(plan) or not summary["complete"]
            or any(digest(Path(path)) != value for path, value in plan["inputs_sha256"].items())):
        raise ValueError("normalization inspection requires unchanged completed screen inputs")
    paths = (args.initial, args.features, args.prior / "observations.json", args.prior / "paired_only-encoder.pt",
             args.candidate / "encoder.pt", args.candidate / "plan.json", args.candidate / "summary.json", args.prior / "checkpoint.pt", args.candidate / "checkpoint.pt")
    bound = {str(path): digest(path) for path in paths}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "plan.json", {"format": "posthoc_branch_normalization_inspection_v1", "inputs_sha256": bound,
               "source_sha256_lf": hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
               "changes_to_screen_rules": False, "new_native_requests": 0, "encoder_optimizer_updates": 0})
    started, results = perf_counter(), {}
    with cpu_settings():
        rows = read(args.prior / "observations.json")
        replay = torch.load(args.features, map_location="cpu", weights_only=True)["replay"]
        bank = {(row["state_id"], row["audio_sha256"]): row["features"] for row in replay}
        for row in rows:
            row["features"] = bank[row["state_id"], row["audio_sha256"]]
            if tensor_digest(row["features"]) != row["features_sha256"]:
                raise ValueError("normalization input features changed")
        training = set(plan["training_states"])
        train = torch.tensor([row["state_id"] in training for row in rows])
        target = note_targets(rows).double()
        model, _front, _codec = restore_direct(torch.load(args.initial, map_location="cpu", weights_only=True))
        for name, path, expected in (("paired_only", args.prior / "paired_only-encoder.pt", prior["results"]["paired_only"]["encoder_sha256"]),
                                     ("calibrated", args.candidate / "encoder.pt", summary["result"]["encoder_sha256"])):
            encoder = initialize(model.correction[0].timbre)["encoder"]
            encoder.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["encoder"])
            if tensor_digest(encoder.state_dict()) != expected:
                raise ValueError("normalization inspection changed its encoder")
            values = encode(encoder, rows)
            channels = {"raw": values, "head_layer_normalized": F.layer_norm(values, (768,)),
                        "raw_mean_and_scale": torch.stack((values.mean(1), values.std(1, unbiased=False)), 1)}
            results[name] = {}
            for channel, features in channels.items():
                predicted = ridge_predict(features[train], target[train], features[~train])
                results[name][channel] = {label: regression_metrics(target[~train, index], predicted[:, index], target[train, index].mean().expand(int((~train).sum())))
                                         for index, label in enumerate(("scaled_mean_midi", "scaled_mean_note_seconds"))}
            saved = torch.load((args.prior if name == "paired_only" else args.candidate) / "checkpoint.pt", map_location="cpu", weights_only=True)
            arm = initialize(model.correction[0].timbre)
            arm["head"].load_state_dict(saved["arms"]["paired_only"]["head"] if name == "paired_only" else saved["arm"]["head"])
            with torch.no_grad():
                predicted = arm["head"](values, coefficient=0.).double()
            results[name]["actual_training_head"] = {label: regression_metrics(target[~train, index], predicted[~train, index], target[train, index].mean().expand(int((~train).sum())))
                                                       for index, label in enumerate(("scaled_mean_midi", "scaled_mean_note_seconds"))}
    if any(digest(Path(path)) != value for path, value in bound.items()):
        raise ValueError("normalization inspection changed its source evidence")
    report = {"posthoc": True, "results": results, "elapsed_seconds": perf_counter() - started,
              "encoder_optimizer_updates": 0, "new_native_requests": 0, "changes_to_screen_rules": False,
              "limit": "Information in mean/scale exposes an adversary coverage gap; it does not establish the cause of the screen outcome or a successful repair."}
    write_json(args.output / "summary.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

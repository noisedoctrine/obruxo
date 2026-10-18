"""Audit retained-feature fitting and native loading of the resulting eight predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


def family_metrics(prediction, labels, codec):
    import torch
    values = torch.tensor([target.continuous for target in labels])
    masks = torch.tensor([codec.continuous_mask(target) for target in labels])
    errors = prediction["continuous"] - values
    if codec.cyclic_indices:
        errors[:, codec.cyclic_indices] = (errors[:, codec.cyclic_indices] + .5).remainder(1.) - .5
    continuous = {}
    for group in sorted(set(codec.continuous_groups)):
        indices = [index for index, name in enumerate(codec.continuous_groups) if name == group]
        count = masks[:, indices].sum()
        continuous[group] = {"active_values": int(count),
                             "mae": float((errors[:, indices].abs() * masks[:, indices]).sum() / count) if count else None}
    categories = torch.tensor([target.categorical for target in labels])
    active = torch.tensor([codec.categorical_mask(target) for target in labels]) > 0
    correct = torch.stack([logits.argmax(1) == categories[:, index] for index, logits in enumerate(prediction["categorical"])], dim=1)
    categorical = {}
    for group in sorted(set(codec.categorical_groups)):
        indices = [index for index, name in enumerate(codec.categorical_groups) if name == group]
        categorical[group] = {"active_choices": int(active[:, indices].sum()),
                              "wrong_choices": int((~correct[:, indices] & active[:, indices]).sum())}
    return {"continuous": continuous, "categorical": categorical}


def main():
    import torch
    from obruxo_data.hashing import canonical_json, sha256_bytes
    from obruxo_data.render.base import RenderRequest
    from obruxo_training.audit import write_json
    from obruxo_training.benchmark import MemoryMonitor
    from obruxo_training.codec import PilotTarget
    from obruxo_training.features import tensor_digest
    from obruxo_training.model import batch_features, loss_terms, predicted_targets
    from obruxo_training.pilot import PilotPool, PilotSampler
    from obruxo_training.reproducibility import FULL_PHASE_CONTROL
    from obruxo_training.stream import NativeRendererFactory, RenderWorker
    from obruxo_training.trainer import curve_loss_inputs, load_model

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pool", "checkpoint", "source-checkpoint", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    started, monitor = perf_counter(), MemoryMonitor()
    monitor.start()
    try:
        pool = PilotPool.load(args.pool)
        saved, source = [torch.load(path, weights_only=True) for path in (args.checkpoint, args.source_checkpoint)]
        training = json.loads(args.checkpoint.with_name("summary.json").read_text())
        model, frontend, exported_codec = load_model(args.checkpoint.with_name("model.pt"))
        replay, targets = saved["replay"], [PilotTarget.from_dict(item["target"]) for item in saved["replay"]]
        sampler = PilotSampler(pool, seed=training["config"]["seed"], policy=training["config"]["sampling_policy"])
        source_counts = {name: sum(pool.sampling_group(next(entry for entry in pool.entries if entry.state_id == item["state_id"])) == name
                                   for item in replay) for name in ("authored", "corpus")}
        unchanged = len(replay) == len(source["replay"]) and all(
            all(left[key] == right[key] for key in ("sequence_id", "state_id", "recipe_id", "audio_sha256", "target"))
            and all(torch.equal(left["features"][key], right["features"][key]) for key in left["features"])
            and all(torch.equal(left[key], right[key]) for key in curve_loss_inputs([left], pool.codec))
            for left, right in zip(replay, source["replay"], strict=True))
        with torch.no_grad():
            prediction = model(batch_features(replay, "cpu"))
            guesses = predicted_targets(prediction, pool.codec)
            labels = [pool.codec.supervision_target(target) for target in targets]
            oracle = {"continuous": torch.tensor([target.continuous for target in labels]), "categorical": []}
            for index, (_, options) in enumerate(pool.codec.categorical):
                logits = torch.full((len(labels), len(options)), -100.)
                logits.scatter_(1, torch.tensor([[target.categorical[index]] for target in labels]), 100.)
                oracle["categorical"].append(logits)
            floor = {key: float(value) for key, value in loss_terms(oracle, targets, pool.codec, **curve_loss_inputs(replay, pool.codec)).items()}
        plan = {"qualification": "mixed_retained_feature_overfit_and_native_predictions_v1", "pool_id": pool.pool_id,
                "codec_id": pool.codec.codec_id, "source_checkpoint_sha256": sha256_bytes(args.source_checkpoint.read_bytes()),
                "checkpoint_sha256": sha256_bytes(args.checkpoint.read_bytes()), "training_config": training["config"],
                "examples": len(replay), "source_counts": source_counts, "native_requests": 2 * len(replay),
                "diagnostic_phase_control": FULL_PHASE_CONTROL, "training_phase_control": "original_requested_values_unchanged",
                "source_sha256_lf": {path.relative_to(RESEARCH).as_posix(): hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                                     for path in (Path(__file__), RESEARCH / "training/obruxo_training/trainer.py",
                                                  RESEARCH / "training/obruxo_training/model.py")},
                "fits_assets_or_decoders": False, "final_test_audio": False, "model_promotion": False}
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output / "plan.json", plan)
        factory, events, checks = NativeRendererFactory(renderer_id=pool.codec.contract.renderer_id), [], []
        for item, guess in zip(replay, guesses, strict=True):
            recipe = sampler.recipe_at(item["sequence_id"])
            original = recipe.render_request()
            document = json.loads(pool.codec.decode(guess))
            document["settings"].update(FULL_PHASE_CONTROL)
            request = RenderRequest(preset_json=canonical_json(document), performance=original.performance,
                                    tail_seconds=original.tail_seconds, renderer_id=original.renderer_id)
            pair = []
            for role in ("prediction", "repeat"):
                with RenderWorker(renderer_factory=factory, max_jobs=1) as worker:
                    audio, metadata = worker.render(request, timeout=60.)
                pair.append(metadata)
                events.append({"sequence_id": item["sequence_id"], "role": role, "request_id": request.request_id, **metadata})
                write_json(args.output / "native_events.json", events)
                del audio
            accepted = all(row["status"] == "accepted" for row in pair)
            hashes = [row.get("qa", {}).get("audio_float32_sha256") for row in pair]
            checks.append({"sequence_id": item["sequence_id"], "accepted": accepted,
                           "repeat_exact": hashes[0] is not None and hashes[0] == hashes[1]})
            print(json.dumps({"stage": "native_prediction", **checks[-1]}), flush=True)
        gates = {"strict_overfit": training["tiny_overfit_gate"], "retained_features_and_labels_unchanged": unchanged,
                 "no_training_renders": training["totals"]["render_attempts"] == training["feature_seconds_this_segment"] == 0,
                 "both_training_sources": all(source_counts.values()),
                 "training_reference_recipes_only": all(sampler.recipe_at(item["sequence_id"]).split == "train"
                     and sampler.recipe_at(item["sequence_id"]).performance_role == "reference" for item in replay),
                 "frozen_decoder_and_assets": saved["run_contract"]["codec"] == source["run_contract"]["codec"] == pool.codec.to_dict(),
                 "frozen_frontend": tensor_digest(saved["basic_pitch"]) == tensor_digest(source["basic_pitch"]) == frontend.frozen_digest,
                 "export_weights_match": tensor_digest(saved["model"]) == tensor_digest(model.state_dict())
                     and exported_codec.to_dict() == pool.codec.to_dict(),
                 "native_predictions_accepted": all(row["accepted"] for row in checks),
                 "native_prediction_repeats_exact": all(row["repeat_exact"] for row in checks),
                 "native_full_readback": all(row.get("renderer_qa", {}).get("native_state", {}).get("verified_loaded_scalars") == 775
                     and len(row.get("renderer_qa", {}).get("native_state", {}).get("nested_families_verified", [])) == 6 for row in events),
                 "within_memory_budget": monitor.peak_bytes is not None and training["peak_process_tree_rss_bytes"] is not None
                     and monitor.errors == training["rss_sampling_errors"] == 0
                     and max(monitor.peak_bytes, training["peak_process_tree_rss_bytes"]) <= 8 * 1024**3}
        summary = {"plan": plan, "gates": gates, "all_gates_pass": all(gates.values()), "training": training,
                   "perfect_label_curve_floor": floor, "fit_by_family": family_metrics(prediction, labels, pool.codec),
                   "native_predictions": checks,
                   "peak_process_tree_rss_bytes": monitor.peak_bytes, "elapsed_seconds": perf_counter() - started,
                   "quality_claim": "eight_training_examples_only_not_validation_or_release_evidence"}
        write_json(args.output / "summary.json", summary)
        print(json.dumps({"gates": gates, "all_gates_pass": summary["all_gates_pass"]}), flush=True)
        return 0 if summary["all_gates_pass"] else 2
    finally:
        monitor.close()


if __name__ == "__main__":
    raise SystemExit(main())

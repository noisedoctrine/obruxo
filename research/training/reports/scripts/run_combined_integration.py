"""Verify combined on-demand features, broad exposure and exact checkpoint recovery."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import sys

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


def main():
    import torch
    from obruxo_data.hashing import canonical_sha256
    from obruxo_training.audit import write_json
    from obruxo_training.combined_codec import CombinedCodec
    from obruxo_training.features import tensor_digest
    from obruxo_training.model import batch_features
    from obruxo_training.performances import bank_for
    from obruxo_training.pilot import PilotPool, PilotSampler
    from obruxo_training.stream import NativeRendererFactory, StreamConfig
    from obruxo_training.trainer import TrainConfig, load_model, train_pilot, validate_replay
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--native-qualification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    pool = PilotPool.load(args.pool)
    codec = pool.codec
    native_plan = json.loads((args.native_qualification / "plan.json").read_text())
    native = json.loads((args.native_qualification / "summary.json").read_text())
    if (not isinstance(codec, CombinedCodec) or not native["passed"] or native["plan_sha256"] != canonical_sha256(native_plan)
            or native_plan["pool_id"] != pool.pool_id or native_plan["codec_id"] != codec.codec_id
            or any(hashlib.sha256((RESEARCH / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() != digest
                   for name, digest in native_plan["source_sha256_lf"].items())):
        raise ValueError("native combined qualification does not match this pool and runtime")
    config = TrainConfig(steps=12, replay_capacity=64, replay_mib=8, batch_size=16, render_budget=128,
                         checkpoint_every=6, cpu_threads=4, sampling_policy="state_round_robin")
    stream = StreamConfig(prefetch_clips=1, timeout_seconds=60, worker_jobs=4)
    sampler = PilotSampler(pool, policy=config.sampling_policy)
    references = {entry.performance_id: entry.name for entry in bank_for(codec.contract.renderer_id) if entry.role == "reference"}
    prospective = []
    for index in range(64):
        recipe = sampler.recipe_at(index)
        prospective.append((recipe.state_id, recipe.performance_id))
    if len({state for state, _ in prospective}) != 64 or {performance for _, performance in prospective} != set(references):
        raise ValueError("the frozen first 64 recipes must cover distinct states and every reference performance")
    files = [Path(__file__), *(RESEARCH / "training/obruxo_training").glob("*.py"),
             *(RESEARCH / name for name in native_plan["source_sha256_lf"] if name.startswith("data_generation/"))]

    def sources():
        return {path.relative_to(RESEARCH).as_posix(): hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for path in sorted(set(files))}

    plan = {"integration": "combined_on_demand_features_and_exact_recovery_v1", "source_sha256_lf": sources(),
            "pool_id": pool.pool_id, "codec_id": codec.codec_id, "training_contract": codec.contract.to_dict(),
            "native_qualification_sha256": canonical_sha256(native), "training": asdict(config), "stream": stream.to_dict(),
            "states_by_split": pool.summary()["states_by_split"], "reference_performances": references,
            "initial_64_distinct_training_states": True, "native_training_attempt_budget": 128, "recovery_snapshot_step": 6,
            "claim": "feature_ingestion_broad_exposure_and_recovery_not_reconstruction_quality", "final_test_audio": False, "model_promotion": False}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "plan.json", plan)
    factory = NativeRendererFactory(renderer_id=codec.contract.renderer_id)

    def snapshot(step):
        if step == 6:
            shutil.copyfile(args.output / "training/checkpoint.pt", args.output / "recovery-step6.pt")

    trained = train_pilot(pool, args.output / "training", config=config, stream_config=stream, renderer_factory=factory,
                          after_step=snapshot, progress=lambda row: print(json.dumps(row), flush=True))
    recovered = train_pilot(pool, args.output / "recovered", config=config, stream_config=stream, renderer_factory=factory,
                            resume=args.output / "recovery-step6.pt", progress=lambda row: print(json.dumps({"recovery": True, **row}), flush=True))
    saved = torch.load(args.output / "training/checkpoint.pt", weights_only=True)
    restored = torch.load(args.output / "recovered/checkpoint.pt", weights_only=True)
    validate_replay(saved["replay"], sampler, saved["cursor"], config)
    counts = Counter(sampler.recipe_at(item["sequence_id"]).performance_id for item in saved["replay"])
    model, _, restored_codec = load_model(args.output / "training/model.pt")

    def equal(left, right):
        if torch.is_tensor(left):
            return torch.is_tensor(right) and torch.equal(left, right)
        if isinstance(left, dict):
            return isinstance(right, dict) and left.keys() == right.keys() and all(equal(left[key], right[key]) for key in left)
        if isinstance(left, (list, tuple)):
            return type(left) is type(right) and len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right, strict=True))
        return left == right

    peak = max(trained["peak_process_tree_rss_bytes"] or 0, recovered["peak_process_tree_rss_bytes"] or 0)
    gates = {"native_profile_qualified": native["passed"], "training_complete": trained["complete"] and recovered["complete"],
             "all_reference_performances_seen": set(counts) == set(references), "distinct_training_states": trained["replay_distinct_states"] == 64,
             "audio_only_inputs": set(batch_features(saved["replay"])) == {"timbre", "pitch"}, "original_labels_and_recipes_verified": True,
             "frozen_frontend_unchanged": trained["frozen_weights_and_buffers_unchanged"] and recovered["frozen_weights_and_buffers_unchanged"],
             "exact_recovery": all(equal(saved[key], restored[key]) for key in ("model", "optimizer", "rng", "batch_rng", "replay", "cursor")),
             "recovery_without_new_renders": recovered["fresh_accepted_this_segment"] == 0 and recovered["totals"]["render_attempts"] == trained["totals"]["render_attempts"],
             "export_restored": tensor_digest(model.state_dict()) == trained["model_state_sha256"] and restored_codec.codec_id == codec.codec_id,
             "source_unchanged": sources() == plan["source_sha256_lf"],
             "memory_bounded": peak > 0 and peak <= 8 * 1024**3 and trained["rss_sampling_errors"] == recovered["rss_sampling_errors"] == 0}
    result = {"plan_sha256": canonical_sha256(plan), "passed": all(gates.values()), "gates": gates, "training": trained, "recovery": recovered,
              "retained_performances": {references[key]: count for key, count in counts.items()}, "peak_process_tree_rss_bytes": peak,
              "final_test_audio": False, "model_promotion": False}
    write_json(args.output / "summary.json", result)
    print(json.dumps({"passed": result["passed"], "gates": gates}), flush=True)
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

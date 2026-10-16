"""One matched concatenation versus cross-attention comparison on existing paired features."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


def main():
    import torch
    from obruxo_data.hashing import canonical_sha256
    from obruxo_training.audit import write_json
    from obruxo_training.evaluate import evaluate_comparison, load_evaluation_checkpoint, retrieval_digest
    from obruxo_training.features import tensor_digest
    from obruxo_training.model import PilotInverse, batch_features, model_spec
    from obruxo_training.proxy_screen import balanced_entries, source_identity
    from obruxo_training.stream import NativeRendererFactory, StreamConfig
    from obruxo_training.trainer import TrainConfig, train_pilot
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--seed-replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plugin-path")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--train-only", action="store_true", help="Commit matched training arms; defer the same frozen native comparison")
    args = parser.parse_args()
    from obruxo_training.pilot import PilotPool
    pool = PilotPool.load(args.pool)
    cache = torch.load(args.seed_replay, map_location="cpu", weights_only=True)
    if cache.get("format") != "obruxo_paired_features_v1" or len(cache["replay"]) != 126:
        raise ValueError("this fixed conditioning screen requires the qualified 126-view paired alpha cache")
    config = TrainConfig(mode="online", steps=1000, replay_capacity=128, replay_mib=8, batch_size=16, learning_rate=.001,
                         seed=0, checkpoint_every=500, render_budget=128, cpu_threads=4, sampling_policy="paired_views", fresh_examples=126)
    configs = {name: replace(config, conditioning=name) for name in ("concat", "cross_attention")}
    panel = [entry.state_id for entry in balanced_entries(pool, "validation", 10)]
    torch.set_num_threads(4)
    torch.manual_seed(0)
    baseline = PilotInverse(pool.codec)
    torch.manual_seed(0)
    attention = PilotInverse(pool.codec, "cross_attention")
    if any(not torch.equal(value, attention.state_dict()[name]) for name, value in baseline.state_dict().items()):
        raise ValueError("conditioning models do not share identical common initialization")
    features = batch_features(cache["replay"][:2])
    with torch.no_grad():
        left, right = baseline(features), attention(features)
    if not torch.equal(left["continuous"], right["continuous"]) or any(not torch.equal(a, b) for a, b in zip(left["categorical"], right["categorical"], strict=True)):
        raise ValueError("zero-start conditioning changed initial predictions")
    plan = {"screen": "paired_alpha_conditioning_v1", "pool_id": pool.pool_id, "source": source_identity(),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "cache_sha256": hashlib.sha256(args.seed_replay.read_bytes()).hexdigest(), "validation_states": panel,
            "arms": {name: asdict(value) for name, value in configs.items()},
            "model_specs": {name: model_spec(pool.codec, name) for name in configs},
            "common_initial_weights_sha256": tensor_digest(baseline.state_dict()), "initial_predictions_equal": True,
            "new_training_renders": 0, "validation_maximum_attempts": 132, "planned_validation_attempts": 120,
            "comparison": "ordinary_top1_same_codec_and_frozen_frontend", "test_split_evaluated": False,
            "limit": "single_seed_small_authored_alpha_not_a_release_promotion"}
    if args.resume:
        if json.loads((args.output / "plan.json").read_text()) != plan:
            raise ValueError("conditioning screen plan changed; cannot resume")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output / "plan.json", plan)
    factory, stream = NativeRendererFactory(args.plugin_path), StreamConfig(prefetch_clips=1, timeout_seconds=30.)
    results, checkpoints = {}, {}
    for name, arm_config in configs.items():
        output = args.output / name
        checkpoint = output / "checkpoint.pt"
        if args.resume and (output / "summary.json").exists():
            saved, _, _, _ = load_evaluation_checkpoint(checkpoint, pool)
            result = json.loads((output / "summary.json").read_text())
            if (saved["step"] != arm_config.steps or TrainConfig(**saved["run_contract"]["training"]) != arm_config
                    or saved["run_contract"]["replay_seed"]["checkpoint_sha256"] != plan["cache_sha256"]
                    or result["model_state_sha256"] != tensor_digest(saved["model"])):
                raise ValueError("completed conditioning arm differs from the frozen plan")
        else:
            recover = checkpoint if args.resume and checkpoint.exists() else None
            result = train_pilot(pool, output, config=arm_config, stream_config=stream, renderer_factory=factory,
                                 resume=recover, seed_replay=None if recover else args.seed_replay,
                                 progress=lambda event: print(json.dumps(event), flush=True))
        if result["totals"]["render_attempts"] != 0:
            raise ValueError("cached conditioning comparison performed new training renders")
        results[name], checkpoints[name] = result, checkpoint
    saved = [torch.load(path, map_location="cpu", weights_only=True) for path in checkpoints.values()]
    if (retrieval_digest(saved[0]["replay"]) != retrieval_digest(saved[1]["replay"])
            or not torch.equal(saved[0]["batch_rng"], saved[1]["batch_rng"])
            or [item["uses"] for item in saved[0]["replay"]] != [item["uses"] for item in saved[1]["replay"]]):
        raise ValueError("conditioning arms used different minibatches")
    if args.train_only:
        write_json(args.output / "status.json", {"complete": False, "stage": "training_complete_validation_pending"})
        return
    def progress(event):
        if event["method"] == "source_repeat":
            print(json.dumps(event), flush=True)
    comparison = evaluate_comparison(pool, checkpoints, checkpoints["concat"], args.output / "comparison", count=10,
                                     validation_states=panel, allow_architecture_comparison=True, renderer_factory=factory,
                                     timeout=30., render_attempt_limit=132, resume=args.resume and (args.output / "comparison/progress.json").exists(),
                                     progress=progress)
    summary = {"plan_id": canonical_sha256(plan), "arms": results, "comparison": comparison["comparison"],
               "validation_attempts": comparison["renders_attempted"], "validation_accepted": comparison["renders_accepted"],
               "new_training_renders": 0, "final_test_used": False, "release_model_promoted": False}
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "status.json", {"complete": True})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

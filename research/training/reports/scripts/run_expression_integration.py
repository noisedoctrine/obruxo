"""Check native expressive feature collection and updates without a quality experiment."""
import argparse
from collections import Counter
from dataclasses import asdict
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
    from obruxo_training.expression_codec import ExpressionCodec
    from obruxo_training.features import tensor_digest
    from obruxo_training.model import batch_features
    from obruxo_training.performances import bank_for
    from obruxo_training.pilot import PilotPool, PilotSampler
    from obruxo_training.stream import NativeRendererFactory, StreamConfig
    from obruxo_training.trainer import TrainConfig, load_model, train_pilot, validate_replay
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-bank", type=Path, required=True)
    parser.add_argument("--lfo-bank", type=Path, required=True)
    parser.add_argument("--native-qualification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    codec = ExpressionCodec(json.loads(args.asset_bank.read_text()), json.loads(args.lfo_bank.read_text()))
    native_plan = json.loads((args.native_qualification / "plan.json").read_text())
    native = json.loads((args.native_qualification / "summary.json").read_text())
    if (not native["passed"] or native["plan_sha256"] != canonical_sha256(native_plan) or native_plan["renderer_id"] != codec.contract.renderer_id
            or any(hashlib.sha256((RESEARCH / name).read_bytes()).hexdigest() != digest for name, digest in native_plan["source_sha256"].items())):
        raise ValueError("native expression qualification does not match this runtime")
    pool = PilotPool.create(32, codec=codec)
    config = TrainConfig(steps=3, replay_capacity=16, replay_mib=8, batch_size=4, render_budget=32,
                         checkpoint_every=1, cpu_threads=4, sampling_policy="performance_round_robin")
    stream = StreamConfig(prefetch_clips=1, timeout_seconds=30, worker_jobs=4)
    files = [Path(__file__), *(RESEARCH / "training/obruxo_training").glob("*.py"),
             *(RESEARCH / name for name in native_plan["source_sha256"] if name.startswith("data_generation/"))]

    def sources():
        return {path.relative_to(RESEARCH).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(set(files))}

    references = [entry for entry in bank_for(codec.contract.renderer_id) if entry.role == "reference"]
    plan = {"integration": "expressive_native_feature_and_update_v1", "source_sha256": sources(), "pool_id": pool.pool_id, "codec_id": codec.codec_id,
            "training_contract": codec.contract.to_dict(), "native_qualification_sha256": canonical_sha256(native),
            "training": asdict(config), "stream": stream.to_dict(), "native_attempt_budget": 32,
            "reference_performances": {entry.performance_id: entry.name for entry in references},
            "states_by_split": pool.summary()["states_by_split"], "final_test_audio": False, "model_promotion": False,
            "claim": "feature_ingestion_and_updates_not_reconstruction_quality_or_combined_control_coverage"}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "plan.json", plan)
    pool.save(args.output / "pool.json")
    trained = train_pilot(pool, args.output / "training", config=config, stream_config=stream,
                          renderer_factory=NativeRendererFactory(renderer_id=codec.contract.renderer_id),
                          progress=lambda row: print(json.dumps(row), flush=True))
    saved = torch.load(args.output / "training/checkpoint.pt", weights_only=True)
    sampler = PilotSampler(pool, policy=config.sampling_policy)
    validate_replay(saved["replay"], sampler, saved["cursor"], config)
    counts = Counter(item["performance_id"] for item in saved["replay"])
    inputs_audio_only = set(batch_features(saved["replay"])) == {"timbre", "pitch"}
    model, _, restored = load_model(args.output / "training/model.pt")
    gates = {"native_profile_qualified": native["passed"], "training_complete": trained["complete"],
             "all_reference_performances_seen": set(counts) == set(plan["reference_performances"]),
             "audio_only_inputs": inputs_audio_only, "original_labels_and_recipes_verified": True,
             "frozen_frontend_unchanged": trained["frozen_weights_and_buffers_unchanged"],
             "export_restored": tensor_digest(model.state_dict()) == trained["model_state_sha256"] and restored.codec_id == codec.codec_id,
             "source_unchanged": sources() == plan["source_sha256"],
             "memory_bounded": trained["rss_sampling_errors"] == 0 and trained["peak_process_tree_rss_bytes"] is not None
                                and trained["peak_process_tree_rss_bytes"] <= 8 * 1024**3}
    summary = {"plan_sha256": canonical_sha256(plan), "passed": all(gates.values()), "gates": gates, "training": trained,
               "retained_performances": {plan["reference_performances"][key]: count for key, count in counts.items()},
               "final_test_audio": False, "model_promotion": False}
    write_json(args.output / "summary.json", summary)
    print(json.dumps({"passed": summary["passed"], "gates": gates}), flush=True)
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

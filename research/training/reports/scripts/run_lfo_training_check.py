"""Fixed local integration of audio-only learning for eight Era2 curve outputs."""
import argparse
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
    from obruxo_training.codec_check import check_codec_audio
    from obruxo_training.evaluate import evaluate_pilot
    from obruxo_training.features import tensor_digest
    from obruxo_training.lfo_bank import sample_lfo
    from obruxo_training.lfo_codec import LfoCurveCodec
    from obruxo_training.lfo_curriculum import authored_lfos
    from obruxo_training.pilot import PilotPool
    from obruxo_training.stream import NativeRendererFactory, StreamConfig
    from obruxo_training.trainer import TrainConfig, load_model, train_pilot
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-bank", type=Path, required=True)
    parser.add_argument("--lfo-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-replay", type=Path, help="Reuse verified same-pool features with a fresh model, optimizer and RNG")
    args = parser.parse_args()
    torch.set_num_threads(4)
    codec = LfoCurveCodec(json.loads(args.asset_bank.read_text()), json.loads(args.lfo_bank.read_text()))
    pool = PilotPool.create(256, codec=codec)
    validation = {}
    for entry in pool.entries:
        if pool.manifest.by_state[entry.state_id].split == "validation":
            validation.setdefault(codec.family(entry.target), entry.state_id)
    if set(validation) != set(codec.to_dict()["families"]):
        raise ValueError("frozen validation roots do not cover every numbered LFO")
    root_index = {dict(pool.manifest.variants)[entry.state_id]: index for index, entry in enumerate(pool.entries)}
    fitted = codec.lfo_bank.to_dict()["bank"]
    source_parents = {entry["state_id"]: entry["parent_state_id"] for entry in fitted["split_manifest"]["variants"]}
    for member in fitted["fitted_rows"]:
        index = root_index[source_parents[member["state_id"]]]
        shape = authored_lfos(index)[member["slot"] - 1]
        if canonical_sha256(sample_lfo(shape).tolist()) != member["curve_sha256"]:
            raise ValueError("authored curve generator differs from the bank's fitted source curves")
    config = TrainConfig(steps=1000, replay_capacity=64, replay_mib=8, batch_size=16, render_budget=128,
                         checkpoint_every=250, cpu_threads=4, sampling_policy="state_round_robin")
    stream = StreamConfig(prefetch_clips=1, timeout_seconds=30, worker_jobs=4)
    files = [Path(__file__), *(RESEARCH / "training/obruxo_training").glob("*.py")]

    def sources():
        return {path.relative_to(RESEARCH).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(files)}

    plan = {"integration": "eight_lfo_audio_only_training_v1", "source_sha256": sources(), "pool_id": pool.pool_id,
            "codec_id": codec.codec_id, "factory_bank_id": codec.base.bank["bank_id"], "lfo_bank_id": codec.lfo_bank.bank_id,
            "fitted_curve_generator_matches": True, "training": asdict(config), "stream": stream.to_dict(),
            "states_by_split": pool.summary()["states_by_split"], "root_manifest_id": pool.manifest.parent.manifest_id,
            "validation_families": sorted(validation), "validation_states": [validation[name] for name in sorted(validation)],
            "native_attempt_budget": {"serialized_target_codec": 24, "training": 128, "validation": 96},
            "outputs": {"continuous": len(codec.continuous), "categorical_heads": len(codec.categorical), "lfo_slots": 8},
            "fixed_connection_slots": 16, "learned_connection_structure": False, "final_test_audio": False, "model_promotion": False}
    if args.seed_replay is not None:
        plan["seed_replay_checkpoint_sha256"] = hashlib.sha256(args.seed_replay.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "plan.json", plan)
    pool.save(args.output / "pool.json")
    qualified = check_codec_audio(pool, args.output / "codec", timeout=30,
                                  progress=lambda done, total, status: print(json.dumps({"stage": "codec", "done": done, "total": total, "status": status}), flush=True))
    if not qualified["passed"]:
        raise RuntimeError("serialized LFO target/native correctness failed")
    factory = NativeRendererFactory(renderer_id=codec.contract.renderer_id)
    trained = train_pilot(pool, args.output / "training", config=config, stream_config=stream, renderer_factory=factory, seed_replay=args.seed_replay,
                          progress=lambda event: print(json.dumps(event), flush=True))
    model, _, restored_codec = load_model(args.output / "training/model.pt")
    restored = tensor_digest(model.state_dict()) == trained["model_state_sha256"] and restored_codec.codec_id == codec.codec_id
    evaluated = evaluate_pilot(pool, args.output / "training/checkpoint.pt", args.output / "validation", count=8,
                               renderer_factory=factory, timeout=30, validation_states=plan["validation_states"],
                               progress=lambda event: print(json.dumps(event), flush=True) if event["method"] == "source_repeat" else None)
    gates = {"serialized_targets_exact": qualified["passed"], "fitted_curve_generator_matches": True,
             "training_complete": trained["complete"], "frozen_frontend_unchanged": trained["frozen_weights_and_buffers_unchanged"],
             "inference_bundle_restored": restored, "native_validation_complete": evaluated["renders_attempted"] == evaluated["renders_accepted"] == 96,
             "source_unchanged": sources() == plan["source_sha256"],
             "memory_bounded": trained["rss_sampling_errors"] == 0 and trained["peak_process_tree_rss_bytes"] is not None
                                and trained["peak_process_tree_rss_bytes"] <= 8 * 1024**3}
    if args.seed_replay is not None:
        gates["seeded_features_match_plan"] = trained["replay_seed"]["checkpoint_sha256"] == plan["seed_replay_checkpoint_sha256"]
    summary = {"plan_sha256": canonical_sha256(plan), "pipeline_check_passed": all(gates.values()), "gates": gates, "training": trained,
               "native_validation": {name: evaluated[name] for name in ("aggregates", "renders_attempted", "renders_accepted")},
               "final_test_audio": False, "model_promotion": False, "learned_connection_structure": False}
    write_json(args.output / "summary.json", summary)
    print(json.dumps({"pipeline_check_passed": summary["pipeline_check_passed"], "gates": gates}), flush=True)
    return 0 if summary["pipeline_check_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

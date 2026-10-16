"""Fixed local integration of learned factory assets, with no release-model promotion."""
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
    from obruxo_training.asset_codec import FactoryAssetCodec
    from obruxo_training.audit import write_json
    from obruxo_training.codec_check import check_codec_audio
    from obruxo_training.evaluate import evaluate_pilot
    from obruxo_training.features import tensor_digest
    from obruxo_training.pilot import PilotPool
    from obruxo_training.stream import NativeRendererFactory, StreamConfig
    from obruxo_training.trainer import TrainConfig, load_model, train_pilot
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    codec = FactoryAssetCodec(json.loads(args.bank.read_text(encoding="utf-8")))
    pool = PilotPool.create(256, codec=codec)
    validation = {}
    for entry in pool.entries:
        if pool.manifest.by_state[entry.state_id].split == "validation":
            validation.setdefault(codec.family(entry.target), entry.state_id)
    if set(validation) != set(codec.to_dict()["families"]):
        raise ValueError("frozen validation roots do not cover every declared factory family")
    config = TrainConfig(steps=1000, replay_capacity=64, replay_mib=8, batch_size=16, render_budget=128,
                         checkpoint_every=250, cpu_threads=4, sampling_policy="state_round_robin")
    stream = StreamConfig(prefetch_clips=1, timeout_seconds=30, worker_jobs=4)
    sources = list((RESEARCH / "training/obruxo_training").glob("*.py")) + [Path(__file__)]

    def source_hashes():
        return {str(path.relative_to(RESEARCH)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(sources)}

    plan = {"experiment": "factory_asset_integration_v1", "bank_id": codec.bank["bank_id"], "pool_id": pool.pool_id,
            "codec_id": codec.codec_id, "source_sha256": source_hashes(), "training": asdict(config), "stream": stream.to_dict(),
            "root_manifest_id": pool.manifest.parent.manifest_id, "states_by_split": pool.summary()["states_by_split"],
            "validation_families": sorted(validation), "validation_states": [validation[name] for name in sorted(validation)],
            "native_attempt_budget": {"codec": 18, "training": 128, "validation": 60}, "final_test_used": False,
            "learned_assets": {"wavetable_choices": len(codec._waves), "sample_choices": len(codec._samples)}, "model_promotion": False}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "plan.json", plan)
    pool.save(args.output / "pool.json")
    qualification = check_codec_audio(pool, args.output / "codec", timeout=30,
                                       progress=lambda done, total, status: print(json.dumps({"stage": "codec", "done": done, "total": total, "status": status}), flush=True))
    if not qualification["passed"]:
        raise RuntimeError("factory codec failed exact fixed-phase native qualification")
    factory = NativeRendererFactory(renderer_id=codec.contract.renderer_id)
    trained = train_pilot(pool, args.output / "training", config=config, stream_config=stream, renderer_factory=factory,
                          progress=lambda event: print(json.dumps(event), flush=True))
    model, _, restored_codec = load_model(args.output / "training/model.pt")
    restored = tensor_digest(model.state_dict()) == trained["model_state_sha256"] and restored_codec.codec_id == codec.codec_id
    evaluated = evaluate_pilot(pool, args.output / "training/checkpoint.pt", args.output / "validation", count=6,
                               renderer_factory=factory, timeout=30, validation_states=plan["validation_states"],
                               progress=lambda event: print(json.dumps(event), flush=True) if event["method"] == "source_repeat" else None)
    gates = {"factory_bank_qualified": codec.bank["qualification"]["qualified"], "codec_exact": qualification["passed"],
             "training_complete": trained["complete"], "frozen_frontend_unchanged": trained["frozen_weights_and_buffers_unchanged"],
             "inference_bundle_restored": restored, "native_validation_complete": evaluated["renders_attempted"] == evaluated["renders_accepted"] == 60,
             "source_unchanged": plan["source_sha256"] == source_hashes(),
             "memory_bounded": trained["rss_sampling_errors"] == 0 and trained["peak_process_tree_rss_bytes"] is not None
                                and trained["peak_process_tree_rss_bytes"] <= 8 * 1024**3}
    summary = {"plan_sha256": canonical_sha256(plan), "pipeline_check_passed": all(gates.values()), "gates": gates,
               "training": trained, "native_validation": {name: evaluated[name] for name in ("aggregates", "renders_attempted", "renders_accepted")},
               "bank_id": codec.bank["bank_id"], "final_test_used": False, "model_promotion": False}
    write_json(args.output / "summary.json", summary)
    print(json.dumps({"pipeline_check_passed": summary["pipeline_check_passed"], "gates": gates}), flush=True)
    return 0 if summary["pipeline_check_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

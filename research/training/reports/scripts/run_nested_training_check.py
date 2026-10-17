"""Fixed original-audio CPU training for all custom-curve slots and seed outputs."""
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
    from obruxo_training.lfo_curriculum import authored_lfos
    from obruxo_training.nested_codec import FAMILIES, NestedStateCodec
    from obruxo_training.pilot import PilotPool
    from obruxo_training.stream import NativeRendererFactory, StreamConfig
    from obruxo_training.trainer import TrainConfig, load_model, train_pilot
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-bank", type=Path, required=True)
    parser.add_argument("--lfo-bank", type=Path, required=True)
    parser.add_argument("--native-qualification", type=Path, required=True, help="Completed native check directory for this profile")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    codec = NestedStateCodec(json.loads(args.asset_bank.read_text()), json.loads(args.lfo_bank.read_text()))
    native_plan = json.loads((args.native_qualification / "plan.json").read_text())
    native_summary = json.loads((args.native_qualification / "summary.json").read_text())
    if (not native_summary["passed"] or native_summary["plan_sha256"] != canonical_sha256(native_plan)
            or native_plan["factory_bank_id"] != codec.base.bank["bank_id"] or native_plan["renderer_id"] != codec.contract.renderer_id
            or any(hashlib.sha256((RESEARCH / name).read_bytes()).hexdigest() != digest for name, digest in native_plan["source_sha256"].items())):
        raise ValueError("completed native nested-state qualification does not match this runtime")
    pool = PilotPool.create(256, codec=codec)
    validation = {}
    for entry in pool.entries:
        if pool.manifest.by_state[entry.state_id].split == "validation":
            validation.setdefault(codec.family(entry.target), entry.state_id)
    if set(validation) != set(FAMILIES):
        raise ValueError("frozen validation roots do not cover every custom-curve mode and seed slot")
    root_index = {dict(pool.manifest.variants)[entry.state_id]: index for index, entry in enumerate(pool.entries)}
    fitted = codec.lfo_bank.to_dict()["bank"]
    source_parents = {entry["state_id"]: entry["parent_state_id"] for entry in fitted["split_manifest"]["variants"]}
    for member in fitted["fitted_rows"]:
        shape = authored_lfos(root_index[source_parents[member["state_id"]]])[member["slot"] - 1]
        if canonical_sha256(sample_lfo(shape).tolist()) != member["curve_sha256"]:
            raise ValueError("authored LFOs differ from the fixed bank's fitted sources")
    config = TrainConfig(steps=400, replay_capacity=64, replay_mib=8, batch_size=16, render_budget=128,
                         checkpoint_every=100, cpu_threads=4, sampling_policy="state_round_robin")
    stream = StreamConfig(prefetch_clips=1, timeout_seconds=30, worker_jobs=4)
    files = [Path(__file__), *(RESEARCH / "training/obruxo_training").glob("*.py"),
             *(RESEARCH / name for name in native_plan["source_sha256"] if name.startswith("data_generation/"))]

    def sources():
        return {path.relative_to(RESEARCH).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(set(files))}

    plan = {"integration": "all_nested_state_audio_only_training_v1", "source_sha256": sources(), "pool_id": pool.pool_id,
            "codec_id": codec.codec_id, "factory_bank_id": codec.base.bank["bank_id"], "lfo_bank_id": codec.lfo_bank.bank_id,
            "native_qualification_sha256": canonical_sha256(native_summary), "renderer_id": codec.contract.renderer_id,
            "fitted_curve_generator_matches": True, "training": asdict(config), "stream": stream.to_dict(),
            "states_by_split": pool.summary()["states_by_split"], "root_manifest_id": pool.manifest.parent.manifest_id,
            "validation_families": sorted(validation), "validation_states": [validation[name] for name in sorted(validation)],
            "native_attempt_budget": {"serialized_target_codec": 39, "training": 128, "validation": 156},
            "outputs": {"continuous": len(codec.continuous), "categorical_heads": len(codec.categorical), "custom_warps": 3, "seeds_uint32": 3},
            "final_test_audio": False, "model_promotion": False}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "plan.json", plan)
    pool.save(args.output / "pool.json")
    qualified = check_codec_audio(pool, args.output / "codec", timeout=30,
                                  progress=lambda done, total, status: print(json.dumps({"stage": "codec", "done": done, "total": total, "status": status}), flush=True))
    if not qualified["passed"]:
        raise RuntimeError("serialized nested target/native correctness failed")
    factory = NativeRendererFactory(renderer_id=codec.contract.renderer_id)
    trained = train_pilot(pool, args.output / "training", config=config, stream_config=stream, renderer_factory=factory,
                          progress=lambda event: print(json.dumps(event), flush=True))
    model, _, restored_codec = load_model(args.output / "training/model.pt")
    restored = tensor_digest(model.state_dict()) == trained["model_state_sha256"] and restored_codec.codec_id == codec.codec_id
    evaluated = evaluate_pilot(pool, args.output / "training/checkpoint.pt", args.output / "validation", count=13,
                               renderer_factory=factory, timeout=30, validation_states=plan["validation_states"],
                               progress=lambda event: print(json.dumps(event), flush=True) if event["method"] == "source_repeat" else None)
    gates = {"native_profile_qualified": native_summary["passed"], "serialized_targets_exact": qualified["passed"],
             "fitted_curve_generator_matches": True, "training_complete": trained["complete"],
             "frozen_frontend_unchanged": trained["frozen_weights_and_buffers_unchanged"], "inference_bundle_restored": restored,
             "native_validation_complete": evaluated["renders_attempted"] == evaluated["renders_accepted"] == 156,
             "source_unchanged": sources() == plan["source_sha256"],
             "memory_bounded": trained["rss_sampling_errors"] == 0 and trained["peak_process_tree_rss_bytes"] is not None
                                and trained["peak_process_tree_rss_bytes"] <= 8 * 1024**3}
    summary = {"plan_sha256": canonical_sha256(plan), "pipeline_check_passed": all(gates.values()), "gates": gates, "training": trained,
               "native_validation": {name: evaluated[name] for name in ("aggregates", "renders_attempted", "renders_accepted")},
               "final_test_audio": False, "model_promotion": False}
    write_json(args.output / "summary.json", summary)
    print(json.dumps({"pipeline_check_passed": summary["pipeline_check_passed"], "gates": gates}), flush=True)
    return 0 if summary["pipeline_check_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""One fixed integration run for full scalar learning; not a release-quality comparison."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


def main():
    import numpy as np
    import torch
    from obruxo_data.hashing import canonical_sha256
    from obruxo_data.render.base import RenderRequest
    from obruxo_training.audit import write_json
    from obruxo_training.codec_check import check_codec_audio
    from obruxo_training.evaluate import evaluate_pilot
    from obruxo_training.features import tensor_digest
    from obruxo_training.full_scalar import FullScalarCodec
    from obruxo_training.performances import initial_bank
    from obruxo_training.pilot import PilotPool
    from obruxo_training.reproducibility import fixed_oscillator_phases
    from obruxo_training.stream import NativeRendererFactory, StreamConfig, render_isolated
    from obruxo_training.trainer import TrainConfig, load_model, train_pilot

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    codec = FullScalarCodec()
    pool = PilotPool.create(256, codec=codec)
    by_family = {}
    for entry in pool.entries:
        if pool.manifest.by_state[entry.state_id].split == "validation":
            by_family.setdefault(codec.family(entry.target), entry.state_id)
    if set(by_family) != set(codec.to_dict()["families"]):
        raise ValueError("frozen validation population must contain all five authored scalar families")
    config = TrainConfig(steps=400, replay_capacity=32, replay_mib=8, batch_size=16, render_budget=64,
                         checkpoint_every=100, cpu_threads=4, sampling_policy="state_round_robin")
    stream = StreamConfig(prefetch_clips=1, timeout_seconds=30, worker_jobs=4)
    sources = list((RESEARCH / "training/obruxo_training").glob("*.py")) + [Path(__file__)]
    sources += [RESEARCH / name for name in ("data_generation/obruxo_data/render/base.py", "data_generation/obruxo_data/render/vita.py",
                                           "data_generation/obruxo_data/render/full_state.py", "data_generation/obruxo_data/vital/full_contract.py",
                                           "data_generation/obruxo_data/vital/atlas.py")]

    def source_hashes():
        return {str(path.relative_to(RESEARCH)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(sources)}

    plan = {"experiment": "full_scalar_integration_v1", "pool_id": pool.pool_id, "codec_id": codec.codec_id,
            "split_policy": "complete_variants_of_frozen_parent_v1", "root_manifest_id": pool.manifest.parent.manifest_id,
            "states_by_split": pool.summary()["states_by_split"],
            "training": asdict(config), "stream": stream.to_dict(), "source_sha256": source_hashes(),
            "validation_states": [by_family[name] for name in sorted(by_family)], "validation_families": sorted(by_family),
            "native_attempt_budget": {"codec": 15, "active_order_equivalence": 2, "training": 64, "validation": 50},
            "final_test_used": False, "model_promotion": False, "purpose": "codec/training/export/native-evaluation integration"}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "plan.json", plan)
    pool.save(args.output / "pool.json")
    factory = NativeRendererFactory(renderer_id=codec.contract.renderer_id)
    qualification = check_codec_audio(pool, args.output / "codec", timeout=30,
                                       progress=lambda done, total, status: print(json.dumps({"stage": "codec", "done": done, "total": total, "status": status}), flush=True))
    if not qualification["passed"]:
        raise RuntimeError("full scalar codec native qualification failed")
    effect = next(entry for entry in pool.entries if pool.manifest.by_state[entry.state_id].split == "train" and codec.family(entry.target) == "effects")
    order_renders, order_audio = [], []
    for target in (effect.target, codec.supervision_target(effect.target)):
        request = RenderRequest(preset_json=fixed_oscillator_phases(codec.decode(target)).decode(), performance=initial_bank()[0].performance(),
                                renderer_id=codec.contract.renderer_id)
        audio, metadata = render_isolated(request, timeout=30)
        order_renders.append(metadata)
        order_audio.append(audio)
    order_exact = all(audio is not None for audio in order_audio) and np.array_equal(*order_audio)
    write_json(args.output / "active-order.json", {"audio_exact": order_exact, "renders": order_renders})
    del order_audio
    if not order_exact:
        raise RuntimeError("disabled-effect order normalization changed native audio")
    trained = train_pilot(pool, args.output / "training", config=config, stream_config=stream, renderer_factory=factory,
                          progress=lambda event: print(json.dumps(event), flush=True))
    restored, _, restored_codec = load_model(args.output / "training/model.pt")
    exported = tensor_digest(restored.state_dict()) == trained["model_state_sha256"] and restored_codec.codec_id == codec.codec_id
    evaluated = evaluate_pilot(pool, args.output / "training/checkpoint.pt", args.output / "validation", count=5,
                               renderer_factory=factory, timeout=30, validation_states=plan["validation_states"],
                               progress=lambda event: print(json.dumps(event), flush=True) if event["method"] == "source_repeat" else None)
    gates = {"codec_exact": qualification["passed"], "active_effect_order_equivalent": order_exact,
             "training_complete": trained["complete"], "frozen_frontend_unchanged": trained["frozen_weights_and_buffers_unchanged"],
             "inference_bundle_restored": exported, "native_validation_complete": evaluated["renders_attempted"] == evaluated["renders_accepted"] == 50,
             "memory_bounded": trained["rss_sampling_errors"] == 0 and trained["peak_process_tree_rss_bytes"] is not None
                                and trained["peak_process_tree_rss_bytes"] <= 8 * 1024**3,
             "source_unchanged": plan["source_sha256"] == source_hashes()}
    observed = {name: set() for name in codec.schema.scalar_names}
    for entry in pool.entries:
        if pool.manifest.by_state[entry.state_id].split == "train":
            for name, value in codec._values(entry.target).items():
                observed[name].add(value)
    varied = [name for name, values in observed.items() if len(values) > 1]
    summary = {"plan_sha256": canonical_sha256(plan), "pipeline_check_passed": all(gates.values()), "gates": gates,
               "scalar_fields_represented": 775, "authored_training_controls_varied": sorted(varied),
               "training": trained, "native_validation": {key: evaluated[key] for key in ("aggregates", "renders_attempted", "renders_accepted")},
               "final_test_used": False, "model_promotion": False}
    write_json(args.output / "summary.json", summary)
    print(json.dumps({"pipeline_check_passed": summary["pipeline_check_passed"], "gates": gates}), flush=True)
    return 0 if summary["pipeline_check_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

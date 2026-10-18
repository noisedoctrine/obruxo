"""Train a bounded original-corpus batch on demand and verify exact feature-based recovery."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


def main():
    import torch
    from obruxo_data.hashing import canonical_json, canonical_sha256
    from obruxo_training.audit import write_json
    from obruxo_training.benchmark import MemoryMonitor
    from obruxo_training.corpus_pool import CorpusPool
    from obruxo_training.pilot import PilotPool, PilotSampler
    from obruxo_training.performance_eligibility import attack_seconds, longest_held_seconds
    from obruxo_training.performances import bank_for
    from obruxo_training.stream import NativeRendererFactory, StreamConfig
    from obruxo_training.trainer import TrainConfig, load_model, train_pilot, validate_replay
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-run", type=Path, help="Resume an interrupted integration's training/checkpoint.pt")
    args = parser.parse_args()
    torch.set_num_threads(2)
    monitor = MemoryMonitor()
    monitor.start()
    try:
        pool = PilotPool.load(args.pool)
        if not isinstance(pool, CorpusPool):
            raise ValueError("integration requires an original corpus source pool")
        config = TrainConfig(steps=3, replay_capacity=8, batch_size=8, render_budget=32, checkpoint_every=1,
                             cpu_threads=2, sampling_policy="state_round_robin")
        stream = StreamConfig(prefetch_clips=1, worker_jobs=1)
        longest = max(longest_held_seconds(entry.performance()) for entry in bank_for(pool.codec.contract.renderer_id) if entry.role == "reference")
        attacks = [attack_seconds(pool.codec, entry.target) for entry in pool.entries]
        observation = {"training_states": len(attacks), "longest_reference_held_seconds": longest,
                       "states_without_attack_compatible_reference": sum(value > longest for value in attacks),
                       "maximum_requested_attack_seconds": max(attacks), "membership_changed": False}
        files = [Path(__file__), *(RESEARCH / "training/obruxo_training").glob("*.py"),
                 *(RESEARCH / "data_generation/obruxo_data/render").glob("*.py"), RESEARCH / "data_generation/obruxo_data/vital/full_contract.py"]

        def sources():
            return {path.relative_to(RESEARCH).as_posix(): hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for path in files}

        plan = {"integration": "original_corpus_on_demand_training_and_recovery_v1", "source_sha256_lf": sources(),
                "pool": pool.summary(), "training_contract": pool.codec.contract.to_dict(), "retained_examples": 8,
                "attempt_budget": 32, "updates": 3, "stream": stream.to_dict(), "sampling_policy": config.sampling_policy,
                "phase_policy": "original_requested_values_retained_recovery_uses_saved_features_not_regenerated_audio",
                "observation_coverage": observation, "rss_limit_bytes": 8 * 1024**3, "final_test_audio": False, "model_promotion": False}
        prefix, checkpoint = [], None
        if args.resume_run:
            previous = json.loads((args.resume_run / "plan.json").read_text())
            if previous["pool"]["pool_id"] != pool.pool_id or any(value != sources().get(name) for name, value in previous["source_sha256_lf"].items()
                    if name != Path(__file__).relative_to(RESEARCH).as_posix()):
                raise ValueError("interrupted integration differs from the current source pool or runtime")
            checkpoint = args.resume_run / "training/checkpoint.pt"
            prefix = [json.loads(line) for line in (args.resume_run / "training/renders.jsonl").read_text().splitlines()]
            plan["continuation"] = {"previous_plan_sha256": canonical_sha256(previous), "acknowledged_events": len(prefix),
                                    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output / "plan.json", plan)

        def snapshot(step):
            if step == 1:
                shutil.copyfile(args.output / "training/checkpoint.pt", args.output / "step1.pt")

        factory = NativeRendererFactory(renderer_id=pool.codec.contract.renderer_id)
        complete = train_pilot(pool, args.output / "training", config=config, stream_config=stream, renderer_factory=factory,
                               resume=checkpoint, after_step=snapshot, progress=lambda row: print(json.dumps(row), flush=True))
        recovered = train_pilot(pool, args.output / "recovered", config=config, stream_config=stream, renderer_factory=factory,
                                resume=args.output / "step1.pt")
        left, right = [torch.load(args.output / name / "checkpoint.pt", weights_only=True) for name in ("training", "recovered")]
        sampler = PilotSampler(pool, policy=config.sampling_policy)
        validate_replay(left["replay"], sampler, left["cursor"], config)
        events = prefix + [json.loads(line) for line in (args.output / "training/renders.jsonl").read_text().splitlines()]
        accepted = {row["sequence_id"]: row for row in events if row["status"] == "accepted"}
        indices = {entry.state_id: index for index, entry in enumerate(pool.entries)}
        checks = []
        for item in left["replay"]:
            index = indices[item["state_id"]]
            entry = pool.entries[index]
            raw = pool.source_preset(entry.target, index)
            recipe = sampler.recipe_at(item["sequence_id"])
            request = recipe.render_request()
            native = accepted[item["sequence_id"]]["renderer_qa"]["native_state"]
            checks.append({"state_id": item["state_id"], "raw_hash_verified": hashlib.sha256(raw).hexdigest() == entry.raw_sha256 == recipe.raw_sha256,
                           "original_document": request.preset_json == canonical_json(json.loads(raw)),
                           "native_requested_hash": native["requested_preset_sha256"] == hashlib.sha256(request.preset_json.encode()).hexdigest(),
                           "full_readback": native["verified_loaded_scalars"] == 775 and len(native["nested_families_verified"]) == 6,
                           "training_only": recipe.split == "train" and recipe.performance_role == "reference"})

        def equal(a, b):
            if torch.is_tensor(a):
                return torch.is_tensor(b) and torch.equal(a, b)
            if isinstance(a, dict):
                return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[key], b[key]) for key in a)
            if isinstance(a, (tuple, list)):
                return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b, strict=True))
            return a == b

        _, _, exported = load_model(args.output / "training/model.pt")
        gates = {"complete": complete["complete"] and recovered["complete"],
                 "eight_distinct_training_states": complete["replay_distinct_states"] == 8,
                 "original_sources_and_full_readback": len(checks) == 8 and all(all(value for key, value in row.items() if key != "state_id") for row in checks),
                 "audio_only_inputs": all(set(item["features"]) == {"timbre", "pitch"} for item in left["replay"]),
                 "exact_recovery": all(equal(left[key], right[key]) for key in ("model", "optimizer", "rng", "batch_rng", "replay", "cursor")),
                 "no_recovery_renders": recovered["fresh_accepted_this_segment"] == 0,
                 "all_examples_used_each_update": all(item["uses"] == config.steps for item in left["replay"]),
                 "exported_codec": exported.codec_id == pool.codec.codec_id and str(pool.corpus_root) not in json.dumps(exported.to_dict()),
                 "frozen_frontend": complete["frozen_weights_and_buffers_unchanged"] and recovered["frozen_weights_and_buffers_unchanged"],
                 "source_unchanged": sources() == plan["source_sha256_lf"]}
    finally:
        monitor.close()
    gates["memory_bounded"] = monitor.errors == 0 and monitor.peak_bytes is not None and monitor.peak_bytes <= plan["rss_limit_bytes"]
    summary = {"plan_sha256": canonical_sha256(plan), "passed": all(gates.values()), "gates": gates,
               "training": complete, "recovery": recovered, "source_checks": checks, "attempts": len(events),
               "unadmitted_attempts": [{key: row[key] for key in ("sequence_id", "state_id", "status", "error_type", "message", "diagnostic_codes") if key in row}
                                       for row in events if row["status"] != "accepted"], "peak_process_tree_rss_bytes": monitor.peak_bytes,
               "claim": "bounded_original_corpus_training_integration_not_population_admission_or_model_quality",
               "final_test_audio": False, "model_promotion": False}
    write_json(args.output / "summary.json", summary)
    print(json.dumps({"passed": summary["passed"], "gates": gates, "attempts": len(events)}), flush=True)
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

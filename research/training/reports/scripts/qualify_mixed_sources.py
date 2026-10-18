"""Verify balanced original-source training, inherited authored coverage and held-out ingestion."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


def equal(a, b):
    import torch
    if torch.is_tensor(a):
        return torch.is_tensor(b) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[key], b[key]) for key in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def main():
    import torch
    from obruxo_data.hashing import canonical_json, canonical_sha256
    from obruxo_data.render.base import RenderRequest
    from obruxo_training.audit import write_json
    from obruxo_training.benchmark import MemoryMonitor
    from obruxo_training.features import tensor_digest
    from obruxo_training.mixed_pool import MixedPool
    from obruxo_training.pilot import PilotPool, PilotSampler
    from obruxo_training.reproducibility import FULL_PHASE_CONTROL
    from obruxo_training.stream import NativeRendererFactory, RenderWorker, StreamConfig
    from obruxo_training.trainer import TrainConfig, load_model, train_pilot, validate_replay
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    monitor, started = MemoryMonitor(), perf_counter()
    monitor.start()
    try:
        print(json.dumps({"stage": "load_mixed_source_metadata"}), flush=True)
        pool = PilotPool.load(args.pool)
        if not isinstance(pool, MixedPool):
            raise ValueError("qualification requires a mixed source pool")
        load_seconds = perf_counter() - started
        coverage = json.loads(args.coverage.read_text())
        if coverage["pool_id"] != pool.authored.pool_id or coverage["codec_id"] != pool.authored.codec.codec_id:
            raise ValueError("authored coverage receipt differs from the mixture's source artifact")
        by_state = {entry.state_id: entry for entry in pool.entries}
        mask_checks = []
        methods = ("continuous_mask", "categorical_mask", "active_lfos", "active_warps", "active_seeds", "active_route_maps")
        for entry in pool.entries:
            if entry.source == "authored" and pool.manifest.by_state[entry.state_id].split == "train":
                original = pool.authored.entries[entry.source_index]
                mask_checks.append(entry.target == original.target and all(tuple(getattr(pool.codec, method)(entry.target)) ==
                    tuple(getattr(pool.authored.codec, method)(original.target)) for method in methods))
        sampler = PilotSampler(pool, policy="source_balanced")
        sampler.recipe_at(0)
        order = sampler._order
        schedule = {"epoch_attempts": sampler.epoch_size, "authored_attempts": sum(pool.sampling_group(entry) == "authored" for entry, _ in order),
                    "corpus_attempts": sum(pool.sampling_group(entry) == "corpus" for entry, _ in order),
                    "all_training_states_visited": {entry.state_id for entry, _ in order} == {
                        entry.state_id for entry in pool.entries if pool.manifest.by_state[entry.state_id].split == "train"},
                    "sources_alternate": all(pool.sampling_group(entry) == ("authored" if index % 2 == 0 else "corpus") for index, (entry, _) in enumerate(order))}
        files = [Path(__file__), *(RESEARCH / "training/obruxo_training").glob("*.py"),
                 *(RESEARCH / "data_generation/obruxo_data/render").glob("*.py"), RESEARCH / "data_generation/obruxo_data/vital/full_contract.py"]

        def sources():
            return {path.relative_to(RESEARCH).as_posix(): hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for path in files}

        plan = {"qualification": "mixed_authored_corpus_training_and_validation_ingestion_v1", "source_sha256_lf": sources(),
                "pool": pool.summary(), "source_schedule": schedule, "training_examples": 8, "training_attempt_budget": 32,
                "updates": 3, "validation_positive_pairs_per_source": 2, "validation_candidates_per_source": 4,
                "validation_attempt_budget": 16, "validation_diagnostic_phase_control": FULL_PHASE_CONTROL,
                "training_phase_control": "original_requested_values_unchanged", "rss_limit_bytes": 8 * 1024**3,
                "authored_coverage_receipt_sha256": canonical_sha256(coverage), "authored_training_masks_checked": len(mask_checks),
                "metadata_load_seconds": load_seconds, "final_test_audio": False, "fits_assets_or_decoders": False, "model_promotion": False}
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output / "plan.json", plan)
        config = TrainConfig(steps=3, replay_capacity=8, batch_size=8, render_budget=32, checkpoint_every=1, cpu_threads=2,
                             sampling_policy="source_balanced")
        stream, factory = StreamConfig(prefetch_clips=1), NativeRendererFactory(renderer_id=pool.codec.contract.renderer_id)

        def snapshot(step):
            if step == 1:
                shutil.copyfile(args.output / "training/checkpoint.pt", args.output / "step1.pt")

        complete = train_pilot(pool, args.output / "training", config=config, stream_config=stream, renderer_factory=factory,
                               after_step=snapshot, progress=lambda row: print(json.dumps(row), flush=True))
        recovered = train_pilot(pool, args.output / "recovered", config=config, stream_config=stream, renderer_factory=factory, resume=args.output / "step1.pt")
        left, right = [torch.load(args.output / name / "checkpoint.pt", weights_only=True) for name in ("training", "recovered")]
        validate_replay(left["replay"], PilotSampler(pool, policy=config.sampling_policy), left["cursor"], config)
        events = [json.loads(line) for line in (args.output / "training/renders.jsonl").read_text().splitlines()]
        replay_sources = {name: sum(pool.sampling_group(by_state[item["state_id"]]) == name for item in left["replay"]) for name in ("authored", "corpus")}
        training_checks = []
        accepted = {row["sequence_id"]: row for row in events if row["status"] == "accepted"}
        for item in left["replay"]:
            recipe = sampler.recipe_at(item["sequence_id"])
            native = accepted[item["sequence_id"]]["renderer_qa"]["native_state"]
            training_checks.append(recipe.raw_sha256 == by_state[item["state_id"]].raw_sha256
                and native["requested_preset_sha256"] == hashlib.sha256(recipe.render_request().preset_json.encode()).hexdigest()
                and native["verified_loaded_scalars"] == 775 and len(native["nested_families_verified"]) == 6
                and recipe.split == "train" and recipe.performance_role == "reference")
        _, frontend, exported = load_model(args.output / "training/model.pt")
        validation_sampler = PilotSampler(pool, split="validation", policy="attack_compatible")
        validation_sampler.recipe_at(0)
        val_events, val_checks, chosen = [], [], {name: 0 for name in ("authored", "corpus")}
        candidates = {name: 0 for name in chosen}
        for sequence, (entry, _) in enumerate(validation_sampler._order):
            group = pool.sampling_group(entry)
            if chosen[group] == 2 or candidates[group] == 4:
                continue
            candidates[group] += 1
            recipe = validation_sampler.recipe_at(sequence)
            original = recipe.render_request()
            document = json.loads(original.preset_json)
            document["settings"].update(FULL_PHASE_CONTROL)
            request = RenderRequest(preset_json=canonical_json(document), performance=original.performance,
                                    tail_seconds=original.tail_seconds, renderer_id=original.renderer_id)
            pair, audio_hashes = [], []
            for role in ("original", "repeat"):
                with RenderWorker(renderer_factory=factory, max_jobs=1) as worker:
                    audio, metadata = worker.render(request, timeout=60.)
                val_events.append({"state_id": entry.state_id, "source": group, "role": role, "request_id": request.request_id, **metadata})
                pair.append(metadata)
                audio_hashes.append(metadata.get("qa", {}).get("audio_float32_sha256"))
                write_json(args.output / "validation_events.json", val_events)
                if metadata["status"] != "accepted":
                    break
                if role == "original":
                    feature_hash = tensor_digest(frontend(audio))
                del audio
            valid = len(pair) == 2 and all(row["status"] == "accepted" for row in pair)
            if valid:
                chosen[group] += 1
                val_checks.append({"state_id": entry.state_id, "source": group, "repeat_exact": len(set(audio_hashes)) == 1,
                                   "feature_sha256": feature_hash, "held_out": recipe.split == "validation",
                                   "not_in_training": entry.state_id not in {row["state_id"] for row in left["replay"]},
                                   "evaluation_group": pool.evaluation_group(entry.state_id)})
            print(json.dumps({"stage": "validation_source", "source": group, "state_id": entry.state_id, "accepted_pair": valid}), flush=True)
            if all(count == 2 for count in chosen.values()):
                break
        frontend.require_frozen()
        gates = {"balanced_attempt_schedule": schedule["sources_alternate"] and schedule["all_training_states_visited"]
                    and schedule["authored_attempts"] == schedule["corpus_attempts"],
                 "authored_supervision_preserved": len(mask_checks) == coverage["audited_train_states"] and all(mask_checks),
                 "complete": complete["complete"] and recovered["complete"], "both_sources_consumed": all(replay_sources.values()),
                 "eight_distinct_training_sources": complete["replay_distinct_states"] == 8,
                 "original_sources_and_full_readback": len(training_checks) == 8 and all(training_checks),
                 "audio_only_inputs": all(set(item["features"]) == {"timbre", "pitch"} for item in left["replay"]),
                 "exact_recovery": all(equal(left[key], right[key]) for key in ("model", "optimizer", "rng", "batch_rng", "replay", "cursor")),
                 "no_recovery_renders": recovered["fresh_accepted_this_segment"] == 0,
                 "validation_sources_sealed_from_training": len(val_checks) == 4 and all(row["held_out"] and row["not_in_training"] and row["repeat_exact"] for row in val_checks),
                 "validation_attempt_budget": len(val_events) <= 16, "exported_codec": exported.codec_id == pool.codec.codec_id,
                 "frozen_frontend": complete["frozen_weights_and_buffers_unchanged"] and recovered["frozen_weights_and_buffers_unchanged"],
                 "source_unchanged": sources() == plan["source_sha256_lf"]}
    finally:
        monitor.close()
    gates["memory_bounded"] = monitor.errors == 0 and monitor.peak_bytes is not None and monitor.peak_bytes <= plan["rss_limit_bytes"]
    summary = {"plan_sha256": canonical_sha256(plan), "passed": all(gates.values()), "gates": gates, "training": complete, "recovery": recovered,
               "retained_sources": replay_sources, "training_attempts": len(events), "source_checks": training_checks,
               "unadmitted_training_attempts": [{"sequence_id": row["sequence_id"], "state_id": sampler.recipe_at(row["sequence_id"]).state_id,
                   "source": pool.sampling_group(by_state[sampler.recipe_at(row["sequence_id"]).state_id]),
                   **{key: row[key] for key in ("status", "error_type", "message", "diagnostic_codes") if key in row}}
                   for row in events if row["status"] != "accepted"], "validation_checks": val_checks, "validation_candidates": candidates,
               "unadmitted_validation_attempts": [{key: row[key] for key in ("state_id", "source", "status", "error_type", "message", "diagnostic_codes") if key in row}
                   for row in val_events if row["status"] != "accepted"], "peak_process_tree_rss_bytes": monitor.peak_bytes,
               "elapsed_seconds": perf_counter() - started,
               "claim": "bounded_mixed_training_and_validation_ingestion_not_population_admission_or_model_quality",
               "final_test_audio": False, "fits_assets_or_decoders": False, "model_promotion": False}
    write_json(args.output / "summary.json", summary)
    print(json.dumps({"passed": summary["passed"], "gates": gates}), flush=True)
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

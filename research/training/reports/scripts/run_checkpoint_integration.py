"""Collect native features once, snapshot before update one and prove exact recovery."""
import argparse
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
    from obruxo_training.checkpoints import COLLECTION_FORMAT, INITIAL_FILE, load_training_checkpoint
    from obruxo_training.model import batch_features
    from obruxo_training.pilot import PilotPool, PilotSampler
    from obruxo_training.stream import NativeRendererFactory, StreamConfig
    from obruxo_training.trainer import TrainConfig, train_pilot, validate_replay
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--native-qualification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    pool = PilotPool.load(args.pool)
    native_plan = json.loads((args.native_qualification / "plan.json").read_text())
    native = json.loads((args.native_qualification / "summary.json").read_text())
    if (not native["passed"] or native["plan_sha256"] != canonical_sha256(native_plan)
            or native_plan["pool_id"] != pool.pool_id or native_plan["codec_id"] != pool.codec.codec_id):
        raise ValueError("native source qualification differs from this pool")
    changed = {name for name, expected in native_plan["source_sha256_lf"].items()
               if hashlib.sha256((RESEARCH / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() != expected}
    allowed = {"training/obruxo_training/" + name for name in ("trainer.py", "evaluate.py", "paired_features.py", "pilot.py", "cli.py")}
    if changed - allowed:
        raise ValueError(f"qualified rendering, source or feature implementation changed: {sorted(changed - allowed)}")
    # Pilot/CLI changes add the separately qualified opt-in schedule. This run retains state_round_robin.
    config = TrainConfig(steps=12, replay_capacity=64, replay_mib=8, batch_size=16, render_budget=128,
                         checkpoint_every=6, cpu_threads=4, sampling_policy="state_round_robin")
    stream = StreamConfig(prefetch_clips=1, timeout_seconds=60, worker_jobs=4)
    files = [Path(__file__), *(RESEARCH / "training/obruxo_training").glob("*.py"),
             *(RESEARCH / name for name in native_plan["source_sha256_lf"] if name.startswith("data_generation/"))]

    def sources():
        return {path.relative_to(RESEARCH).as_posix(): hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for path in sorted(set(files))}

    plan = {"integration": "native_collection_checkpoint_recovery_v1", "source_sha256_lf": sources(),
            "pool_id": pool.pool_id, "codec_id": pool.codec.codec_id, "training_contract": pool.codec.contract.to_dict(),
            "native_qualification_sha256": canonical_sha256(native), "qualified_source_differences": sorted(changed),
            "source_difference_scope": "training_storage_and_opt_in_sampler_integration_no_render_source_or_feature_changes",
            "training": asdict(config), "stream": stream.to_dict(), "native_attempt_budget": 128,
            "snapshot": "all_64_retained_features_before_initial_metrics_and_first_optimizer_update",
            "repeat_policy": "exact_retained_features_no_native_rerender_on_recovery",
            "native_collection": "original_requested_phases_not_a_waveform_repeat_comparison",
            "prior_fixed_phase_fresh_process_source_qualification": True, "rss_limit_bytes": 8 * 1024**3,
            "final_test_audio": False, "model_promotion": False}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "plan.json", plan)
    snapshot = args.output / "before-update-one"

    def progress(event):
        print(json.dumps(event), flush=True)
        if event["stage"] == "render" and event["replay_examples"] == 64 and not snapshot.exists():
            snapshot.mkdir()
            # Copy the immutable dependency first, then its recovery pointer, as persistent storage must do.
            for name in (INITIAL_FILE, "checkpoint.pt"):
                shutil.copyfile(args.output / "training" / name, snapshot / name)
            load_training_checkpoint(snapshot / "checkpoint.pt")

    trained = train_pilot(pool, args.output / "training", config=config, stream_config=stream,
                          renderer_factory=NativeRendererFactory(renderer_id=pool.codec.contract.renderer_id), progress=progress)
    before = load_training_checkpoint(snapshot / "checkpoint.pt")
    raw = torch.load(snapshot / "checkpoint.pt", weights_only=True)
    recovered = train_pilot(pool, args.output / "recovered", config=config, stream_config=stream, resume=snapshot / "checkpoint.pt",
                            progress=lambda row: print(json.dumps({"recovery": True, **row}), flush=True))
    left, right = [load_training_checkpoint(args.output / name / "checkpoint.pt") for name in ("training", "recovered")]
    sampler = PilotSampler(pool, policy=config.sampling_policy)
    validate_replay(left["replay"], sampler, left["cursor"], config)

    def equal(a, b):
        if torch.is_tensor(a):
            return torch.is_tensor(b) and torch.equal(a, b)
        if isinstance(a, dict):
            return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[key], b[key]) for key in a)
        if isinstance(a, (list, tuple)):
            return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b, strict=True))
        return a == b

    peak = max(trained["peak_process_tree_rss_bytes"] or 0, recovered["peak_process_tree_rss_bytes"] or 0)
    gates = {"training_complete": trained["complete"] and recovered["complete"],
             "compact_snapshot_before_learning": raw["format"] == COLLECTION_FORMAT and before["step"] == 0
                 and before["initial_metrics"] is None and before["optimizer"]["state"] == {} and len(before["replay"]) == 64,
             "initial_state_written_once": trained["checkpoint_storage"]["initial_writes"] == 1,
             "every_acknowledgement_persisted": trained["checkpoint_storage"]["collection_writes"] == 1 + trained["totals"]["render_attempts"],
             "distinct_training_states": trained["replay_distinct_states"] == 64,
             "audio_only_inputs": set(batch_features(left["replay"])) == {"timbre", "pitch"},
             "exact_recovery": all(equal(left[key], right[key]) for key in ("model", "optimizer", "rng", "batch_rng", "replay", "cursor")),
             "no_new_audio_or_features_on_recovery": recovered["fresh_accepted_this_segment"] == 0 and recovered["feature_seconds_this_segment"] == 0
                 and recovered["totals"]["render_attempts"] == trained["totals"]["render_attempts"],
             "frozen_frontend": trained["frozen_weights_and_buffers_unchanged"] and recovered["frozen_weights_and_buffers_unchanged"],
             "final_checkpoint_self_contained": torch.load(args.output / "training/checkpoint.pt", weights_only=True)["format"] == "obruxo_pilot_training_v1",
             "source_unchanged": sources() == plan["source_sha256_lf"],
             "memory_bounded": peak > 0 and peak <= plan["rss_limit_bytes"] and trained["rss_sampling_errors"] == recovered["rss_sampling_errors"] == 0}
    summary = {"plan_sha256": canonical_sha256(plan), "passed": all(gates.values()), "gates": gates,
               "training": trained, "recovery": recovered, "peak_process_tree_rss_bytes": peak,
               "final_test_audio": False, "model_promotion": False}
    write_json(args.output / "summary.json", summary)
    print(json.dumps({"passed": summary["passed"], "gates": gates}), flush=True)
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Verify original singleton sources reach native rendering, labels, training and recovery."""
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
    from obruxo_data.render.curve_inputs import prepare_curve_inputs
    from obruxo_training.audit import write_json
    from obruxo_training.curve_codec import CurveInputCodec
    from obruxo_training.graph_codec import GraphCoverageCodec
    from obruxo_training.pilot import PilotPool, PilotSampler
    from obruxo_training.stream import NativeRendererFactory, StreamConfig
    from obruxo_training.trainer import TrainConfig, load_model, train_pilot, validate_replay
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--precision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    qualifications = {}
    for name, path in (("singleton", args.native), ("precision", args.precision)):
        plan = json.loads((path / "plan.json").read_text())
        summary = json.loads((path / "summary.json").read_text())
        if (not summary["passed"] or summary["plan_sha256"] != canonical_sha256(plan) or any(
                hashlib.sha256((RESEARCH / relative).read_bytes().replace(b"\r\n", b"\n")).hexdigest() != expected
                for relative, expected in plan["source_sha256_lf"].items())):
            raise ValueError("native curve qualification differs from the runtime")
        qualifications[name] = canonical_sha256(summary)
    pool = PilotPool.load(args.pool)
    if not isinstance(pool.codec, (CurveInputCodec, GraphCoverageCodec)):
        raise ValueError("integration requires an original curve-source or connection-coverage pool")
    graph = isinstance(pool.codec, GraphCoverageCodec)
    config = TrainConfig(steps=3, replay_capacity=16, replay_mib=8, batch_size=16, render_budget=64,
                         checkpoint_every=1, cpu_threads=4, sampling_policy="attack_compatible")
    stream = StreamConfig(prefetch_clips=1, worker_jobs=4)
    files = [Path(__file__), *(RESEARCH / "training/obruxo_training").glob("*.py"),
             *(RESEARCH / "data_generation/obruxo_data/render").glob("*.py"), RESEARCH / "data_generation/obruxo_data/vital/full_contract.py"]

    def sources():
        return {path.relative_to(RESEARCH).as_posix(): hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for path in sorted(set(files))}

    plan = {"integration": "graph_variants_on_demand_training_v1" if graph else "original_singleton_sources_on_demand_training_v1", "source_sha256_lf": sources(),
            "pool_id": pool.pool_id, "codec_id": pool.codec.codec_id, "training_contract": pool.codec.contract.to_dict(),
            "native_qualifications": qualifications, "retained_examples": 16, "attempt_budget": 64, "updates": 3,
            "sampling_policy": config.sampling_policy, "stream": stream.to_dict(), "states_by_split": pool.summary()["states_by_split"],
            "recovery_snapshot_step": 1, "rss_limit_bytes": 8 * 1024**3, "final_test_audio": False, "model_promotion": False}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "plan.json", plan)

    def snapshot(step):
        if step == 1:
            shutil.copyfile(args.output / "training/checkpoint.pt", args.output / "step1.pt")

    factory = NativeRendererFactory(renderer_id=pool.codec.contract.renderer_id)
    full = train_pilot(pool, args.output / "training", config=config, stream_config=stream, renderer_factory=factory, after_step=snapshot,
                       progress=lambda row: print(json.dumps(row), flush=True))
    recovered = train_pilot(pool, args.output / "recovered", config=config, stream_config=stream, renderer_factory=factory, resume=args.output / "step1.pt")
    left, right = [torch.load(args.output / name / "checkpoint.pt", weights_only=True) for name in ("training", "recovered")]
    sampler = PilotSampler(pool, policy=config.sampling_policy)
    validate_replay(left["replay"], sampler, left["cursor"], config)
    events = [json.loads(line) for line in (args.output / "training/renders.jsonl").read_text().splitlines()]
    changes = [change for event in events if event["status"] == "accepted"
               for change in event["renderer_qa"]["native_state"]["curve_changes"]]
    families = {change["path"].split("/")[2] for change in changes if change["singleton_expanded"]}
    event_by_sequence = {event["sequence_id"]: event for event in events}
    source_checks = []
    source_indices = {entry.state_id: index for index, entry in enumerate(pool.entries)}
    variant_counts = {"original": 0, "connection_coverage": 0}
    graph_checks = []
    active_counts = {family: 0 for family in ("lfos", "modulations", "custom_warps")}
    for item in left["replay"]:
        request = sampler.recipe_at(item["sequence_id"]).render_request()
        native = event_by_sequence[item["sequence_id"]]["renderer_qa"]["native_state"]
        applied, ledger = prepare_curve_inputs(json.loads(request.preset_json))
        target = next(entry.target for entry in pool.entries if entry.state_id == item["state_id"])
        if graph:
            variant = source_indices[item["state_id"]] % pool.codec.variants_per_root
            variant_counts["connection_coverage" if variant else "original"] += 1
            if variant:
                routes = applied["settings"]["modulations"]
                graph_checks.append(all(pool.codec.routing.active_routes(pool.codec._routing_target(target)))
                    and set(pool.codec.sources) <= {route["source"] for route in routes})
        flags = {"lfos": pool.codec.active_lfos(target), "modulations": pool.codec.active_route_maps(target), "custom_warps": pool.codec.active_warps(target)}
        supervised = []
        for change in ledger:
            if not change["singleton_expanded"]:
                continue
            _, _, family, position, *_ = change["path"].split("/")
            enabled = bool(flags[family][int(position)])
            active_counts[family] += enabled
            supervised.append(enabled)
        source_checks.append({"sequence_id": item["sequence_id"], "original_source_exact":
            native["requested_preset_sha256"] == hashlib.sha256(request.preset_json.encode()).hexdigest(),
            "applied_source_exact": native["applied_preset_sha256"] == hashlib.sha256(canonical_json(applied).encode()).hexdigest(),
            "ledger_exact": native["curve_changes"] == ledger, "all_singletons_supervised": all(supervised)})

    def equal(a, b):
        if torch.is_tensor(a):
            return torch.is_tensor(b) and torch.equal(a, b)
        if isinstance(a, dict):
            return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[key], b[key]) for key in a)
        if isinstance(a, (tuple, list)):
            return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b, strict=True))
        return a == b

    _, _, exported = load_model(args.output / "training/model.pt")
    peak = max(full["peak_process_tree_rss_bytes"] or 0, recovered["peak_process_tree_rss_bytes"] or 0)
    gates = {"complete": full["complete"] and recovered["complete"], "all_curve_families_consumed": families == {"lfos", "modulations", "custom_warps"},
             "distinct_training_states": full["replay_distinct_states"] == 16,
             "original_raw_sources_and_labels": all(row["original_source_exact"] and row["applied_source_exact"] and row["ledger_exact"] for row in source_checks),
             "all_singleton_families_actively_sampled": all(active_counts.values()) and all(row["all_singletons_supervised"] for row in source_checks)
                 and all(item["uses"] == config.steps for item in left["replay"]),
             "audio_only_inputs": all(set(item["features"]) == {"timbre", "pitch"} for item in left["replay"]),
             "exact_recovery": all(equal(left[key], right[key]) for key in ("model", "optimizer", "rng", "batch_rng", "replay", "cursor")),
             "no_recovery_renders": recovered["fresh_accepted_this_segment"] == 0,
             "exported_codec": exported.codec_id == pool.codec.codec_id,
             "frozen_frontend": full["frozen_weights_and_buffers_unchanged"] and recovered["frozen_weights_and_buffers_unchanged"],
             "source_unchanged": sources() == plan["source_sha256_lf"],
             "memory_bounded": peak > 0 and peak <= plan["rss_limit_bytes"] and full["rss_sampling_errors"] == recovered["rss_sampling_errors"] == 0}
    if graph:
        gates.update(original_and_connection_variants_consumed=all(variant_counts.values()),
                     consumed_graphs_keep_all_slots_and_sources=bool(graph_checks) and all(graph_checks))
    summary = {"plan_sha256": canonical_sha256(plan), "passed": all(gates.values()), "gates": gates, "training": full, "recovery": recovered,
               "singleton_families_consumed": sorted(families), "active_singletons_by_family": active_counts,
               "source_checks": source_checks, "peak_process_tree_rss_bytes": peak,
               "final_test_audio": False, "model_promotion": False}
    if graph:
        summary["retained_variants"] = variant_counts
    write_json(args.output / "summary.json", summary)
    print(json.dumps({"passed": summary["passed"], "gates": gates}), flush=True)
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

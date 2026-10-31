"""Compare fresh and reused workers across fixed-phase patches, performances and recycling."""

import argparse
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
from statistics import mean
import sys
from time import perf_counter

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


def main():
    from obruxo_data.hashing import canonical_sha256
    from obruxo_data.render.base import RenderRequest
    from obruxo_data.render.qa import RepeatabilityTolerance, compare_audio
    from obruxo_training.audit import write_json
    from obruxo_training.benchmark import MemoryMonitor
    from obruxo_training.performances import initial_bank
    from obruxo_training.pilot import PilotPool
    from obruxo_training.reproducibility import FULL_PHASE_CONTROL, fixed_generator_phases
    import numpy as np
    from obruxo_training.stream import NativeRendererFactory, RenderWorker, render_isolated

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plugin-path")
    args = parser.parse_args()
    pool = PilotPool.load(args.pool)
    families = {}
    for entry in pool.entries:
        if pool.manifest.by_state[entry.state_id].split == "train":
            family = pool.codec.family(entry.target) if hasattr(pool.codec, "family") else str(entry.target.categorical)
            families.setdefault(family, entry)
    if len(families) < 2:
        raise ValueError("worker qualification requires contrasting training patches")
    requests, cases = [], []
    for family, entry in families.items():
        for performance in (initial_bank()[0], initial_bank()[3]):
            cases.append({"family": family, "performance": performance.name})
            requests.append(RenderRequest(preset_json=fixed_generator_phases(pool.codec.decode(entry.target)).decode(), performance=performance.performance(),
                                      sample_rate=pool.codec.contract.sample_rate, channels=2, tail_seconds=pool.codec.contract.tail_seconds,
                                      renderer_id=pool.codec.contract.renderer_id, preroll_frames=32768))
    source = {str(path.relative_to(RESEARCH)): hashlib.sha256(path.read_bytes()).hexdigest() for path in
              [RESEARCH / "training/obruxo_training/stream.py", RESEARCH / "data_generation/obruxo_data/render/vita.py",
               RESEARCH / "data_generation/obruxo_data/render/basic_profile.py", Path(__file__)]}
    plan = {"qualification": "basic_native_session_v3", "pool_id": pool.pool_id, "source": source,
            "families": list(families), "source_training_states": [entry.state_id for entry in families.values()], "split": "train",
            "phase_control": FULL_PHASE_CONTROL, "training_pool_changed": False, "cases": cases, "preroll_frames": 32768,
            "state_lineage": "diagnostic_phase_fixed_derivatives_of_train_only_sources",
            "requests": [request.request_id for request in requests], "max_jobs": 32, "reuse_order": [0, 1, 0] + [index % len(requests) for index in range(30)],
            "fresh_attempts": 33, "reuse_attempts": 33, "tolerance": asdict(RepeatabilityTolerance()),
            "numeric_limits": {"maximum_absolute_sample_error": 1e-6, "waveform_rmse": 1e-8},
            "training_limits": {"absolute_loss_error": 1e-6, "absolute_gradient_error": 1e-4,
                                "relative_gradient_error_above_0_01": .001, "parameter_update_error": 1e-5},
            "timing_design": "adjacent_fresh_reuse_pairs_with_alternating_order",
            "minimum_mean_wall_reduction": .10, "rss_limit_bytes": 8 * 1024**3, "final_test_used": False}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "plan.json", plan)
    factory, monitor = NativeRendererFactory(args.plugin_path, renderer_id=pool.codec.contract.renderer_id), MemoryMonitor()
    events, checks = [], []
    children_before = {child.pid for child in mp.active_children()}
    started = perf_counter()
    monitor.start()

    def record(mode, index, request, metadata):
        event = {"mode": mode, "case": index, "request_id": request.request_id, **metadata}
        events.append(event)
        with (args.output / "renders.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(event, allow_nan=False) + "\n")
        print(json.dumps({key: event[key] for key in ("mode", "case", "status", "wall_seconds")}), flush=True)
        if event["status"] != "accepted":
            raise ValueError("worker qualification cannot drop rejected/error attempts")

    try:
        with RenderWorker(renderer_factory=factory, max_jobs=32) as worker:
            for index, selected in enumerate(plan["reuse_order"]):
                request = requests[selected]
                pair = {}
                for mode in (("fresh", "reuse") if index % 2 == 0 else ("reuse", "fresh")):
                    pair[mode] = (worker.render(request, timeout=30.) if mode == "reuse" else
                                  render_isolated(request, renderer_factory=factory, timeout=30.))
                    record(mode, index, request, pair[mode][1])
                audio, metadata = pair["reuse"]
                reference, expected = pair["fresh"]
                comparison = compare_audio(reference, audio)
                checks.append({"case": index, **cases[selected], **comparison,
                               "maximum_absolute_sample_error": float(np.max(np.abs(reference - audio))),
                               "direct_controls": metadata["renderer_qa"].get("basic_session", {}).get("direct_controls", False),
                               "engine_reused": metadata["renderer_qa"].get("basic_session", {}).get("engine_reused", False),
                               "audio_float32_identical": expected["qa"]["audio_float32_sha256"] == metadata["qa"]["audio_float32_sha256"],
                               "native_state_identical": expected["renderer_qa"]["native_state"] == metadata["renderer_qa"]["native_state"],
                               "provenance_identical": expected["provenance"] == metadata["provenance"]})
    finally:
        monitor.close()
    fresh = [row for row in events if row["mode"] == "fresh"]
    reused = [row for row in events if row["mode"] == "reuse"]
    matched_fresh_mean = mean(row["wall_seconds"] for row in fresh)
    reuse_mean = mean(row["wall_seconds"] for row in reused)
    reduction = 1 - reuse_mean / matched_fresh_mean
    process_checks = (len({row["worker_pid"] for row in reused[:32]}) == 1
                      and reused[32]["worker_pid"] != reused[0]["worker_pid"]
                      and [row["worker_job"] for row in reused] == [*range(1, 33), 1]
                      and {child.pid for child in mp.active_children()} <= children_before)
    gates = {"native_equivalence": all(row["within_tolerance"]
                                      and row["maximum_absolute_sample_error"] <= plan["numeric_limits"]["maximum_absolute_sample_error"]
                                      and row["waveform_rmse"] <= plan["numeric_limits"]["waveform_rmse"]
                                      and row["native_state_identical"] and row["provenance_identical"] for row in checks),
             "basic_session_used": sum(row["engine_reused"] for row in checks) == 31,
             "bounded_lifetime_and_cleanup": process_checks, "wall_reduction": reduction >= plan["minimum_mean_wall_reduction"],
             "memory": monitor.peak_bytes is not None and monitor.peak_bytes <= plan["rss_limit_bytes"]}
    summary = {"format": "obruxo_render_worker_qualification_v3", "plan_sha256": canonical_sha256(plan), "gates": gates,
               "qualified": all(gates.values()), "accepted": len(events), "attempted": len(events), "checks": checks,
               "matched_fresh_mean_seconds": matched_fresh_mean, "reuse_mean_seconds": reuse_mean, "mean_wall_reduction": reduction,
               "backend_mean_seconds": {mode: mean(row["render_seconds"] for row in events if row["mode"] == mode) for mode in ("fresh", "reuse")},
               "elapsed_seconds": perf_counter() - started, "peak_process_tree_rss_bytes": monitor.peak_bytes,
               "rss_sampling_errors": monitor.errors, "audio_files_written": 0, "final_test_used": False,
               "limit": "Fixed-phase reproducibility on local Windows pinned Vital/DawDreamer, the recorded training cases and 32-job lifetime; random-phase draws are separate, and this is not Linux qualification or a model-quality claim."}
    write_json(args.output / "summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "checks"}, indent=2), flush=True)


if __name__ == "__main__":
    main()

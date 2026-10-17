"""Repeat known silent views and a longer attack-compatible view without changing patches."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


def main():
    from obruxo_data.hashing import canonical_sha256
    from obruxo_data.render.base import RenderRequest
    from obruxo_training.audit import write_json
    from obruxo_training.benchmark import MemoryMonitor
    from obruxo_training.performance_eligibility import attack_seconds, eligible, longest_held_seconds, rule_document
    from obruxo_training.pilot import PilotPool, PilotSampler
    from obruxo_training.reproducibility import FULL_PHASE_CONTROL, fixed_generator_phases
    from obruxo_training.stream import NativeRendererFactory, RenderWorker

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pool = PilotPool.load(args.pool)
    if pool.pool_id != "bfe52afe8c95e2d90df5fd120d60e4f128bd3508ffca4c37d22f968a03128fb5":
        raise ValueError("this regression panel requires the original frozen combined pool")
    args.output.mkdir(parents=True, exist_ok=False)
    files = list((RESEARCH / "training/obruxo_training").glob("*.py")) + [Path(__file__)]
    files += list((RESEARCH / "data_generation/obruxo_data/render").glob("*.py"))

    def sources():
        return {path.relative_to(RESEARCH).as_posix(): hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for path in sorted(files)}

    old = PilotSampler(pool, policy="state_round_robin")
    compatible = PilotSampler(pool, policy="attack_compatible")
    originals = {sequence: old.recipe_at(sequence) for sequence in (6, 15, 45)}
    schedule = [compatible.recipe_at(index) for index in range(compatible.epoch_size)]
    replacement = next(recipe for recipe in schedule if recipe.state_id == originals[45].state_id)
    indices = {entry.state_id: index for index, entry in enumerate(pool.entries)}
    cases = [(f"original_{sequence}", recipe, "rejected") for sequence, recipe in originals.items()]
    cases.append(("attack_compatible_45", replacement, "accepted"))
    plan = {"qualification": "source_view_eligibility_v1", "pool_id": pool.pool_id, "codec_id": pool.codec.codec_id,
            "pool_file_sha256": hashlib.sha256(args.pool.read_bytes()).hexdigest(), "source_sha256_lf": sources(),
            "rule": rule_document(), "sampler_id": compatible.sampler_id, "attempt_budget": len(cases) * 2,
            "phase_control": FULL_PHASE_CONTROL, "worker_jobs": 1, "rss_limit_bytes": 8 * 1024**3,
            "training_contract": pool.codec.contract.to_dict(), "final_test_audio": False, "model_promotion": False,
            "cases": [{"case": name, "state_id": recipe.state_id, "sequence_id": recipe.sequence_id,
                       "performance_id": recipe.performance_id, "expected_status": status} for name, recipe, status in cases]}
    write_json(args.output / "plan.json", plan)
    rows, checks = [], []
    monitor, started = MemoryMonitor(), perf_counter()
    monitor.start()
    try:
        with RenderWorker(renderer_factory=NativeRendererFactory(renderer_id=pool.codec.contract.renderer_id), max_jobs=1) as worker:
            for name, recipe, expected in cases:
                original = recipe.render_request()
                request = RenderRequest(preset_json=fixed_generator_phases(original.preset_json.encode()).decode(),
                    performance=original.performance, tail_seconds=4., renderer_id=pool.codec.contract.renderer_id)
                target = pool.entries[indices[recipe.state_id]].target
                for repeat in range(2):
                    _, metadata = worker.render(request, timeout=60.)
                    rows.append({"case": name, "repeat": repeat, "request_id": request.request_id, **metadata})
                    write_json(args.output / "events.json", rows)
                pair = rows[-2:]
                checks.append({"case": name, "expected_status": expected, "statuses": [row["status"] for row in pair],
                    "expected_qa": all(row["status"] == expected and (expected == "accepted" or "audio.silence" in row["diagnostic_codes"]) for row in pair),
                    "repeat_exact": pair[0].get("qa", {}).get("audio_float32_sha256") is not None and
                                    pair[0]["qa"]["audio_float32_sha256"] == pair[1].get("qa", {}).get("audio_float32_sha256"),
                    "rms": pair[0].get("qa", {}).get("rms"), "attack_seconds": attack_seconds(pool.codec, target),
                    "held_seconds": longest_held_seconds(request.performance), "temporal_eligible": eligible(pool.codec, target, request.performance),
                    "original_labels_exact": pool.codec.encode(original.preset_json.encode()) == target})
                print(json.dumps(checks[-1]), flush=True)
    finally:
        monitor.close()
    gates = {"expected_native_qa": len(rows) == plan["attempt_budget"] and all(row["expected_qa"] for row in checks),
             "fixed_phase_repeats": all(row["repeat_exact"] for row in checks),
             "original_source_labels": all(row["original_labels_exact"] for row in checks),
             "same_preset_longer_view": replacement.render_request().preset_json == originals[45].render_request().preset_json,
             "attack_rule": not checks[2]["temporal_eligible"] and checks[3]["temporal_eligible"],
             "complete_training_schedule": len({recipe.state_id for recipe in schedule}) == compatible.epoch_size and all(
                 recipe.split == "train" and recipe.performance_role == "reference" and eligible(pool.codec, pool.entries[indices[recipe.state_id]].target,
                 recipe.render_request().performance) for recipe in schedule),
             "source_unchanged": sources() == plan["source_sha256_lf"],
             "memory_bounded": monitor.errors == 0 and monitor.peak_bytes is not None and monitor.peak_bytes <= plan["rss_limit_bytes"]}
    summary = {"plan_sha256": canonical_sha256(plan), "passed": all(gates.values()), "gates": gates, "checks": checks,
               "attempted": len(rows), "accepted": sum(row["status"] == "accepted" for row in rows),
               "rejected": sum(row["status"] == "rejected" for row in rows), "training_states_scheduled": len(schedule),
               "elapsed_seconds": perf_counter() - started, "peak_process_rss_bytes": monitor.peak_bytes,
               "audio_files_written": 0, "final_test_audio": False, "model_promotion": False}
    write_json(args.output / "summary.json", summary)
    print(json.dumps({"passed": summary["passed"], "gates": gates}), flush=True)
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

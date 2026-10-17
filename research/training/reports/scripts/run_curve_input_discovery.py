"""Reproduce the pinned native singleton-LFO failure and mapping/warp constant behavior."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


def main():
    from obruxo_data.hashing import canonical_json, canonical_sha256
    from obruxo_data.midi import Performance
    from obruxo_data.render.base import FULL_EXPRESSION_RENDERER_ID, RenderRequest
    from obruxo_data.render.curve_inputs import prepare_curve_inputs
    from obruxo_training.audit import write_json
    from obruxo_training.stream import NativeRendererFactory, RenderWorker
    from run_curve_input_check import cases
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    selected = {"lfo_1_mode_None_y_0.2", "mapping_1_mode_None_y_0.2", "warp_1_mode_13_y_0.2"}
    files = [Path(__file__), Path(__file__).with_name("run_curve_input_check.py"),
             *(RESEARCH / "data_generation/obruxo_data/render").glob("*.py"), RESEARCH / "data_generation/obruxo_data/vital/full_contract.py"]

    def sources():
        return {path.relative_to(RESEARCH).as_posix(): hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for path in sorted(set(files))}

    plan = {"discovery": "native_singleton_expected_failure_controls_v1", "source_sha256_lf": sources(), "attempt_budget": 9,
            "renderer_id": FULL_EXPRESSION_RENDERER_ID, "worker_jobs": 1, "phase_control": "all_four_generator_random_phases_zero",
            "expected": "two_worker_exits_for_active_singleton_lfo_seven_accepted_controls",
            "final_test_audio": False, "model_promotion": False}
    write_json(args.output / "plan.json", plan)
    performance = Performance(bpm=120)
    performance.add_note(pitch=60, velocity=90, start_tick=0, duration_ticks=1440)
    performance.end_tick = 1920
    rows, checks = [], []
    with RenderWorker(renderer_factory=NativeRendererFactory(renderer_id=FULL_EXPRESSION_RENDERER_ID), max_jobs=1) as worker:
        for name, family, source in cases():
            if name not in selected:
                continue
            original = deepcopy(source)
            settings = original["settings"]
            curves = settings["lfos"] + settings["custom_warps"] + [route["line_mapping"] for route in settings["modulations"] if "line_mapping" in route]
            shape = next(shape for shape in curves if shape["num_points"] == 1)
            shape.update(points=[.375, .25], powers=[0.], smooth=False)
            projected, _ = prepare_curve_inputs(original)
            group = []
            for role, document in (("original", original), ("constant_reference", projected), ("repeat", original)):
                request = RenderRequest(preset_json=canonical_json(document), performance=performance, tail_seconds=4., renderer_id=FULL_EXPRESSION_RENDERER_ID)
                _, metadata = worker.render(request, timeout=60.)
                row = {"family": family, "role": role, "request_id": request.request_id, **metadata}
                rows.append(row)
                group.append(row)
                write_json(args.output / "events.json", rows)
            expected = ([row["status"] for row in group] == ["error", "accepted", "error"] and
                all(group[index].get("error_type") == "WorkerExited" for index in (0, 2))) if family == "lfo" else (
                all(row["status"] == "accepted" for row in group) and len({row["qa"]["audio_float32_sha256"] for row in group}) == 1)
            checks.append({"family": family, "statuses": [row["status"] for row in group], "expected_outcomes": expected})
            print(json.dumps(checks[-1]), flush=True)
    gates = {"all_attempts_accounted": len(rows) == 9, "expected_failure_and_positive_controls": all(row["expected_outcomes"] for row in checks),
             "source_unchanged": sources() == plan["source_sha256_lf"]}
    summary = {"plan_sha256": canonical_sha256(plan), "passed": all(gates.values()), "gates": gates, "checks": checks,
               "attempted": len(rows), "accepted": sum(row["status"] == "accepted" for row in rows), "expected_errors": sum(row["status"] == "error" for row in rows),
               "audio_files_written": 0, "final_test_audio": False, "model_promotion": False}
    write_json(args.output / "summary.json", summary)
    print(json.dumps({"passed": summary["passed"], "gates": gates}), flush=True)
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

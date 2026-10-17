"""Check decimal coordinates/powers on multi-point curves against exact native float32 inputs."""
from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


def main():
    from obruxo_data.hashing import canonical_json, canonical_sha256
    from obruxo_data.midi import Performance
    from obruxo_data.render.base import FULL_CURVE_RENDERER_ID, FULL_EXPRESSION_RENDERER_ID, RenderRequest
    from obruxo_data.render.curve_inputs import prepare_curve_inputs
    from obruxo_training.audit import write_json
    from obruxo_training.stream import NativeRendererFactory, RenderWorker
    from run_curve_input_check import cases
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    selected = {"lfo_8_mode_None_y_0.2", "mapping_64_mode_None_y_0.2", *(f"warp_1_mode_{mode}_y_0.2" for mode in (13, 14, 15))}
    fixtures = []
    for name, family, original in cases():
        if name not in selected:
            continue
        for smooth in (False, True):
            document = deepcopy(original)
            settings = document["settings"]
            curves = settings["lfos"] + settings["custom_warps"] + [route["line_mapping"] for route in settings["modulations"] if "line_mapping" in route]
            shape = next(shape for shape in curves if shape["num_points"] == 1)
            shape.update(num_points=3, points=[0., .1, .37, .8, 1., .2], powers=[.37, -.37, 0.], smooth=smooth)
            fixtures.append((name + f"_smooth_{smooth}", document))
    files = [Path(__file__), Path(__file__).with_name("run_curve_input_check.py"),
             *(RESEARCH / "data_generation/obruxo_data/render").glob("*.py"), RESEARCH / "data_generation/obruxo_data/vital/full_contract.py"]

    def sources():
        return {path.relative_to(RESEARCH).as_posix(): hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for path in sorted(set(files))}

    plan = {"qualification": "native_multi_point_decimal_curve_import_v1", "source_sha256_lf": sources(),
            "cases": [name for name, _ in fixtures], "attempt_budget": 30, "worker_jobs": 1,
            "phase_control": "three_oscillator_and_sampler_random_phases_zero", "final_test_audio": False, "model_promotion": False}
    write_json(args.output / "plan.json", plan)
    performance = Performance(bpm=120)
    performance.add_note(pitch=60, velocity=90, start_tick=0, duration_ticks=1440)
    performance.end_tick = 1920
    rows, checks, started = [], [], perf_counter()
    with RenderWorker(renderer_factory=NativeRendererFactory(renderer_id=FULL_CURVE_RENDERER_ID), max_jobs=1) as new_worker, \
         RenderWorker(renderer_factory=NativeRendererFactory(renderer_id=FULL_EXPRESSION_RENDERER_ID), max_jobs=1) as old_worker:
        for name, original in fixtures:
            applied, changes = prepare_curve_inputs(original)
            pair = []
            for role, document, worker, renderer in (("decimal", original, new_worker, FULL_CURVE_RENDERER_ID),
                    ("native_float32", applied, old_worker, FULL_EXPRESSION_RENDERER_ID), ("repeat", original, new_worker, FULL_CURVE_RENDERER_ID)):
                request = RenderRequest(preset_json=canonical_json(document), performance=performance, tail_seconds=4., renderer_id=renderer)
                _, metadata = worker.render(request, timeout=60.)
                row = {"case": name, "role": role, "request_id": request.request_id, **metadata}
                rows.append(row)
                pair.append(row)
                write_json(args.output / "events.json", rows)
            hashes = [row.get("qa", {}).get("audio_float32_sha256") for row in pair]
            checks.append({"case": name, "accepted": all(row["status"] == "accepted" for row in pair),
                "exact_audio_equivalence": None not in hashes and len(set(hashes)) == 1,
                "precision_only": bool(changes) and all(not change["singleton_expanded"] for change in changes),
                "ledger_exact": pair[0].get("renderer_qa", {}).get("native_state", {}).get("curve_changes") == changes})
            print(json.dumps(checks[-1]), flush=True)
    gates = {"all_attempts_accounted": len(rows) == plan["attempt_budget"], "all_accepted": all(row["accepted"] for row in checks),
             "exact_audio_equivalence_and_repeats": all(row["exact_audio_equivalence"] for row in checks),
             "precision_only_and_exact_ledger": all(row["precision_only"] and row["ledger_exact"] for row in checks),
             "source_unchanged": sources() == plan["source_sha256_lf"]}
    summary = {"plan_sha256": canonical_sha256(plan), "passed": all(gates.values()), "gates": gates, "checks": checks,
               "attempted": len(rows), "elapsed_seconds": perf_counter() - started, "audio_files_written": 0,
               "final_test_audio": False, "model_promotion": False}
    write_json(args.output / "summary.json", summary)
    print(json.dumps({"passed": summary["passed"], "gates": gates}), flush=True)
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

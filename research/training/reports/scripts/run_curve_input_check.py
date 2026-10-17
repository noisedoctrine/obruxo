"""Qualify constant singleton import and exact phase-controlled native equivalence."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


def cases():
    from obruxo_data.vital.full_contract import FullPresetContract
    from obruxo_training.reproducibility import FULL_PHASE_CONTROL
    positions = [("lfo", slot, None) for slot in range(1, 9)]
    positions += [("mapping", slot, None) for slot in (1, 32, 64)]
    positions += [("warp", slot, mode) for slot in range(1, 4) for mode in (13, 14, 15)]
    for ordinal, (family, slot, mode) in enumerate(positions):
        for level in (.2, .8):
            document = FullPresetContract().template()
            settings = document["settings"]
            settings.update(FULL_PHASE_CONTROL, volume=2500., sample_on=0., osc_1_level=.15, osc_1_destination=4.)
            settings["modulations"][0] = {"source": f"lfo_{slot}" if family == "lfo" else "lfo_1", "destination": "osc_1_pan"}
            settings["modulation_1_amount"] = .4
            shape = {"name": "OBRUXO singleton fixture", "num_points": 1, "points": [(0., .37, 1.)[ordinal % 3], level],
                     "powers": [(-.37, .37)[ordinal % 2]], "smooth": bool(ordinal % 2)}
            if family == "lfo":
                settings["lfos"][slot - 1] = shape
            elif family == "mapping":
                settings["modulations"][0] = {"source": "", "destination": ""}
                settings["modulations"][slot - 1] = {"source": "lfo_1", "destination": "osc_1_pan", "line_mapping": shape}
                settings[f"modulation_{slot}_amount"] = .4
            else:
                settings.update({f"osc_{number}_on": float(number == slot) for number in range(1, 4)})
                settings.update({f"osc_{slot}_level": .15, f"osc_{slot}_destination": 4.,
                                 f"osc_{slot}_spectral_morph_type": float(mode), f"osc_{slot}_spectral_morph_amount": .5})
                settings["modulations"][0] = {"source": "", "destination": ""}
                settings["custom_warps"][slot - 1] = shape
            yield f"{family}_{slot}_mode_{mode}_y_{level}", family, document


def main():
    from obruxo_data.hashing import canonical_json, canonical_sha256
    from obruxo_data.midi import Performance
    from obruxo_data.render.base import FULL_CURVE_RENDERER_ID, FULL_EXPRESSION_RENDERER_ID, RenderRequest
    from obruxo_data.render.curve_inputs import curve_input_policy, float32, prepare_curve_inputs
    from obruxo_training.audit import write_json
    from obruxo_training.benchmark import MemoryMonitor
    from obruxo_training.stream import NativeRendererFactory, RenderWorker
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    fixtures = list(cases())
    files = [Path(__file__), *(RESEARCH / "data_generation/obruxo_data/render").glob("*.py"),
             RESEARCH / "data_generation/obruxo_data/vital/full_contract.py", RESEARCH / "training/obruxo_training/stream.py",
             RESEARCH / "training/obruxo_training/reproducibility.py"]

    def sources():
        return {path.relative_to(RESEARCH).as_posix(): hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for path in sorted(set(files))}

    plan = {"qualification": "native_curve_input_constant_semantics_v1", "policy": curve_input_policy(),
            "source_sha256_lf": sources(), "renderer_id": FULL_CURVE_RENDERER_ID, "baseline_renderer_id": FULL_EXPRESSION_RENDERER_ID,
            "cases": [name for name, *_ in fixtures], "attempt_budget": sum(3 if family == "lfo" else 4 for _, family, _ in fixtures),
            "worker_jobs": 1, "phase_control": "all_three_oscillator_and_sampler_random_phases_zero",
            "lfo_reference": "explicit_two_point_constant_legacy_profile_original_singleton_crashes_native_worker",
            "mapping_warp_reference": "both_legacy_native_singleton_and_explicit_two_point_constant",
            "required_audio": "finite_exact_hash_equivalence_silence_only_allowed_for_constant_custom_warps",
            "rss_limit_bytes": 8 * 1024**3, "final_test_audio": False, "model_promotion": False}
    write_json(args.output / "plan.json", plan)
    performance = Performance(bpm=120)
    performance.add_note(pitch=60, velocity=90, start_tick=0, duration_ticks=1440)
    performance.end_tick = 1920
    rows, checks = [], []
    monitor, started = MemoryMonitor(), perf_counter()
    monitor.start()
    try:
        with RenderWorker(renderer_factory=NativeRendererFactory(renderer_id=FULL_CURVE_RENDERER_ID), max_jobs=1) as new_worker, \
             RenderWorker(renderer_factory=NativeRendererFactory(renderer_id=FULL_EXPRESSION_RENDERER_ID), max_jobs=1) as old_worker:
            for name, family, document in fixtures:
                applied, changes = prepare_curve_inputs(document)
                variants = [("single", document, new_worker, FULL_CURVE_RENDERER_ID),
                            ("constant_reference", applied, old_worker, FULL_EXPRESSION_RENDERER_ID),
                            ("repeat", document, new_worker, FULL_CURVE_RENDERER_ID)]
                if family != "lfo":
                    legacy = deepcopy(document)
                    shape = next(shape for shape in (legacy["settings"]["custom_warps"] if family == "warp" else
                        [route["line_mapping"] for route in legacy["settings"]["modulations"] if "line_mapping" in route]) if shape["num_points"] == 1)
                    shape["points"] = [float32(value) for value in shape["points"]]
                    shape["powers"] = [float32(value) for value in shape["powers"]]
                    variants.append(("legacy_single", legacy, old_worker, FULL_EXPRESSION_RENDERER_ID))
                pair = []
                for role, raw, worker, renderer in variants:
                    request = RenderRequest(preset_json=canonical_json(raw), performance=performance, tail_seconds=4., renderer_id=renderer)
                    _, metadata = worker.render(request, timeout=60.)
                    row = {"case": name, "role": role, "request_id": request.request_id, **metadata}
                    rows.append(row)
                    pair.append(row)
                    write_json(args.output / "events.json", rows)
                hashes = [row.get("qa", {}).get("audio_float32_sha256") for row in pair]
                statuses = [row["status"] for row in pair]
                valid = all(row.get("qa", {}).get("finite") is True and (row["status"] == "accepted" or
                    (family == "warp" and row["status"] == "rejected" and set(row["diagnostic_codes"]) <= {"audio.silence", "vital.native.curve_input_bridge"})) for row in pair)
                native = pair[0].get("renderer_qa", {}).get("native_state", {})
                checks.append({"case": name, "family": family, "statuses": statuses, "audio_qa_as_declared": valid,
                    "exact_audio_equivalence": None not in hashes and len(set(hashes)) == 1,
                    "applied_structure_verified": set(native.get("nested_families_verified", [])) == {
                        "lfos", "custom_warps", "random_values", "modulations", "wavetables", "sample"}
                        and native.get("verified_loaded_scalars") == 775 and native.get("modulation_slots_verified") == 64,
                    "change_ledger_exact": native.get("curve_changes") == changes,
                    "source_preserved": native.get("requested_preset_sha256") == hashlib.sha256(canonical_json(document).encode()).hexdigest(),
                    "rms": pair[0].get("qa", {}).get("rms")})
                print(json.dumps(checks[-1]), flush=True)
    finally:
        monitor.close()
    gates = {"all_attempts_accounted": len(rows) == plan["attempt_budget"],
             "finite_audio_and_declared_qa": all(row["audio_qa_as_declared"] for row in checks),
             "exact_native_equivalence_and_repeats": all(row["exact_audio_equivalence"] for row in checks),
             "exact_applied_state_and_change_ledger": all(row["applied_structure_verified"] and row["change_ledger_exact"] for row in checks),
             "original_source_preserved": all(row["source_preserved"] for row in checks),
             "source_unchanged": sources() == plan["source_sha256_lf"],
             "memory_bounded": monitor.errors == 0 and monitor.peak_bytes is not None and monitor.peak_bytes <= plan["rss_limit_bytes"]}
    summary = {"plan_sha256": canonical_sha256(plan), "passed": all(gates.values()), "gates": gates, "checks": checks,
               "attempted": len(rows), "accepted": sum(row["status"] == "accepted" for row in rows),
               "rejected": sum(row["status"] == "rejected" for row in rows), "errors": sum(row["status"] == "error" for row in rows),
               "elapsed_seconds": perf_counter() - started, "peak_process_rss_bytes": monitor.peak_bytes,
               "audio_files_written": 0, "final_test_audio": False, "model_promotion": False}
    write_json(args.output / "summary.json", summary)
    print(json.dumps({"passed": summary["passed"], "gates": gates}), flush=True)
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

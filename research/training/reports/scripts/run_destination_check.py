"""Check native audio routing for all oscillator and sampler destinations."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


def main():
    from obruxo_data.hashing import canonical_json, canonical_sha256
    from obruxo_data.render.base import FULL_EXPRESSION_RENDERER_ID, RenderRequest
    from obruxo_training.audit import write_json
    from obruxo_training.benchmark import MemoryMonitor
    from obruxo_training.full_scalar import EFFECTS, FullScalarCodec
    from obruxo_training.performances import bank_for
    from obruxo_training.reproducibility import FULL_PHASE_CONTROL
    from obruxo_training.stream import NativeRendererFactory, RenderWorker
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    codec = FullScalarCodec()
    base = codec.schema.template()
    base["settings"].update(FULL_PHASE_CONTROL)
    base["settings"].update(volume=2500., osc_1_level=.15, osc_2_level=.15, osc_3_level=.15, sample_level=.15,
        chorus_dry_wet=.1, distortion_mix=.1, reverb_dry_wet=.1, delay_dry_wet=.1, flanger_dry_wet=.1,
        phaser_dry_wet=.1, compressor_mix=.1, compressor_low_gain=-9., compressor_band_gain=-9., compressor_high_gain=-9.)
    reference = next(entry for entry in bank_for(FULL_EXPRESSION_RENDERER_ID) if entry.role == "reference")
    files = [Path(__file__), RESEARCH / "training/obruxo_training/full_scalar.py",
             *(RESEARCH / "data_generation/obruxo_data" / name for name in ("render/vita.py", "render/base.py", "render/full_state.py",
               "render/midi_transport.py", "vital/full_contract.py"))]

    def sources():
        return {path.relative_to(RESEARCH).as_posix(): hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for path in files}

    plan = {"qualification": "native_source_audio_destination_domains_v1", "renderer_id": FULL_EXPRESSION_RENDERER_ID,
            "source_sha256_lf": sources(), "phase_control": FULL_PHASE_CONTROL, "performance_id": reference.performance_id,
            "sources": ["osc_1", "osc_2", "osc_3", "sample"], "all_effects_on_destinations": list(range(15)),
            "all_effects_off_destinations": list(range(5, 15)), "expected_audible": list(range(5)),
            "expected_exact_zero": list(range(5, 15)), "attempt_budget": 100, "worker_jobs": 4,
            "tail_seconds": 4., "rss_limit_bytes": 8 * 1024**3, "final_test_audio": False}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "plan.json", plan)
    rows, monitor, started = [], MemoryMonitor(), perf_counter()
    monitor.start()
    try:
        with RenderWorker(renderer_factory=NativeRendererFactory(renderer_id=FULL_EXPRESSION_RENDERER_ID), max_jobs=4) as worker:
            for source in plan["sources"]:
                for enabled in (True, False):
                    for destination in range(15) if enabled else range(5, 15):
                        document = deepcopy(base)
                        settings = document["settings"]
                        settings.update({f"osc_{slot}_on": float(source == f"osc_{slot}") for slot in range(1, 4)})
                        settings["sample_on"] = float(source == "sample")
                        settings.update({effect + "_on": float(enabled) for effect in EFFECTS})
                        settings[source + "_destination"] = float(destination)
                        settings.update(codec._values(codec._target(settings)))
                        request = RenderRequest(preset_json=canonical_json(document), performance=reference.performance(), tail_seconds=4., renderer_id=FULL_EXPRESSION_RENDERER_ID)
                        _, metadata = worker.render(request, timeout=60.)
                        row = {"source": source, "effects_enabled": enabled, "destination": destination,
                               "status": metadata["status"], "peak": metadata.get("qa", {}).get("peak"), "rms": metadata.get("qa", {}).get("rms"),
                               "state_verified": metadata.get("provenance", {}).get("settings", {}).get("verified_loaded_scalars") == 775}
                        rows.append(row)
                        write_json(args.output / "checks.json", rows)
                        print(json.dumps(row), flush=True)
    finally:
        monitor.close()
    gates = {"all_attempts_accounted": len(rows) == 100, "all_scalar_states_verified": all(row["state_verified"] for row in rows),
             "standard_destinations_audible": all(row["status"] == "accepted" for row in rows if row["destination"] < 5),
             "inactive_destinations_exact_zero": all(row["status"] == "rejected" and row["peak"] == row["rms"] == 0. for row in rows if row["destination"] >= 5),
             "source_unchanged": sources() == plan["source_sha256_lf"],
             "memory_bounded": monitor.errors == 0 and monitor.peak_bytes is not None and monitor.peak_bytes <= plan["rss_limit_bytes"]}
    result = {"plan_sha256": canonical_sha256(plan), "passed": all(gates.values()), "gates": gates, "checks": rows,
              "accepted": sum(row["status"] == "accepted" for row in rows), "expected_zero_rejections": sum(row["status"] == "rejected" for row in rows),
              "elapsed_seconds": perf_counter() - started, "peak_process_rss_bytes": monitor.peak_bytes, "memory_sampling_errors": monitor.errors,
              "audio_files_written": 0, "final_test_audio": False}
    write_json(args.output / "summary.json", result)
    print(json.dumps({"passed": result["passed"], "gates": gates}), flush=True)
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

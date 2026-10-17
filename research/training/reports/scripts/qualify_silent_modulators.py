"""Check silent output routes, active FM sources and disabled-asset guards in native Vital."""
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
    import numpy as np
    from obruxo_data.hashing import canonical_json, canonical_sha256
    from obruxo_data.midi import Performance
    from obruxo_data.render.base import FULL_CURVE_RENDERER_ID, RenderRequest
    from obruxo_training.asset_codec import FactoryAssetCodec
    from obruxo_training.audit import write_json
    from obruxo_training.benchmark import MemoryMonitor
    from obruxo_training.reproducibility import FULL_PHASE_CONTROL
    from obruxo_training.stream import NativeRendererFactory, RenderWorker
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bank = json.loads(args.asset_bank.read_text())
    assets = FactoryAssetCodec(bank)
    if len(assets._samples) < 3:
        raise ValueError("FM sample qualification needs two distinct non-init samples")
    scalar, base = assets.scalar, assets.scalar.schema.template()
    settings = base["settings"]
    settings.update(FULL_PHASE_CONTROL, volume=2500., osc_1_on=1., osc_2_on=1., osc_3_on=0., sample_on=0.,
                    osc_1_destination=4., osc_2_destination=9., osc_1_level=.15, osc_2_level=.15,
                    osc_1_distortion_type=7., osc_1_distortion_amount=.4, osc_2_transpose=0.)
    settings["wavetables"][1] = deepcopy(assets._waves[1])
    cases = {}
    for name, values in (("osc_on", {}), ("osc_off", {"osc_2_on": 0.}), ("osc_ratio", {"osc_2_transpose": 12.}),
                         ("osc_off_ratio", {"osc_2_on": 0., "osc_2_transpose": 12.}), ("osc_direct", {"osc_2_destination": 4.})):
        document = deepcopy(base)
        document["settings"].update(values)
        cases[name] = document
    for name, enabled, choice in (("sample_off_a", 0., 1), ("sample_off_b", 0., 2), ("sample_on_a", 1., 1), ("sample_on_b", 1., 2)):
        document = deepcopy(base)
        document["settings"].update(osc_1_distortion_type=9., osc_2_on=0., sample_on=enabled, sample_level=0., sample_loop=1., sample_keytrack=1.)
        document["settings"]["sample"] = deepcopy(assets._samples[choice])
        cases[name] = document
    cases["osc_repeat"] = deepcopy(cases["osc_on"])
    performance = Performance(bpm=120)
    performance.add_note(pitch=60, velocity=90, start_tick=240, duration_ticks=1440)
    performance.end_tick = 2400
    files = [Path(__file__), *(RESEARCH / "training/obruxo_training").glob("*.py"),
             *(RESEARCH / "data_generation/obruxo_data/render").glob("*.py"), RESEARCH / "data_generation/obruxo_data/vital/full_contract.py"]

    def sources():
        return {path.relative_to(RESEARCH).as_posix(): hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for path in files}

    for document in cases.values():
        document["settings"].update(scalar._values(scalar._target(document["settings"])))
    plan = {"qualification": "silent_generator_fm_source_and_disabled_asset_guards_v1", "source_sha256_lf": sources(),
            "bank_id": bank["bank_id"], "cases": list(cases), "request_state_sha256": {name: canonical_sha256(doc) for name, doc in cases.items()},
            "phase_control": FULL_PHASE_CONTROL, "worker_jobs": 1, "performance": performance.to_dict(), "attempt_budget": len(cases),
            "fixture_scalars": "prospectively_projected_to_native_float32_grid", "rss_limit_bytes": 8 * 1024**3,
            "final_test_audio": False, "model_promotion": False}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "plan.json", plan)
    buffers, rows, monitor, started = {}, [], MemoryMonitor(), perf_counter()
    monitor.start()
    try:
        with RenderWorker(renderer_factory=NativeRendererFactory(renderer_id=FULL_CURVE_RENDERER_ID), max_jobs=1) as worker:
            for name, document in cases.items():
                request = RenderRequest(preset_json=canonical_json(document), performance=performance, tail_seconds=4., renderer_id=FULL_CURVE_RENDERER_ID)
                buffers[name], metadata = worker.render(request, timeout=60.)
                rows.append({"case": name, "request_id": request.request_id, **metadata})
                write_json(args.output / "events.json", rows)
                print(json.dumps({"case": name, "status": metadata["status"]}), flush=True)
    finally:
        monitor.close()

    def difference(left, right):
        if buffers[left] is None or buffers[right] is None:
            return None
        return float(np.sqrt(np.mean((buffers[left].astype(np.float64) - buffers[right]) ** 2)) /
                     max(1e-12, np.sqrt(np.mean(buffers[left].astype(np.float64) ** 2))))

    pairs = (("osc_on", "osc_off"), ("osc_on", "osc_ratio"), ("osc_off", "osc_off_ratio"), ("osc_on", "osc_direct"),
             ("sample_off_a", "sample_off_b"), ("sample_on_a", "sample_on_b"), ("osc_on", "osc_repeat"))
    values = {left + "__" + right: difference(left, right) for left, right in pairs}
    changed = ("osc_on__osc_off", "osc_on__osc_ratio", "osc_on__osc_direct", "sample_on_a__sample_on_b")
    exact = ("osc_off__osc_off_ratio", "sample_off_a__sample_off_b", "osc_on__osc_repeat")
    gates = {"all_native_accepted": len(rows) == len(cases) and all(row["status"] == "accepted" for row in rows),
             "active_modulators_and_direct_route_matter": all(values[name] is not None and values[name] > 1e-5 for name in changed),
             "disabled_assets_and_fixed_phase_repeat_exact": all(values[name] == 0. for name in exact),
             "source_unchanged": sources() == plan["source_sha256_lf"],
             "memory_bounded": monitor.errors == 0 and monitor.peak_bytes is not None and monitor.peak_bytes <= plan["rss_limit_bytes"]}
    summary = {"plan_sha256": canonical_sha256(plan), "passed": all(gates.values()), "gates": gates,
               "relative_rms_changes": values, "attempted": len(rows), "peak_process_rss_bytes": monitor.peak_bytes,
               "elapsed_seconds": perf_counter() - started, "final_test_audio": False, "model_promotion": False}
    write_json(args.output / "summary.json", summary)
    print(json.dumps({"passed": summary["passed"], "gates": gates, "relative_rms_changes": values}), flush=True)
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

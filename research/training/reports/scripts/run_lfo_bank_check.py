"""Fixed train-only Era2 fitting and native curve reconstruction diagnostic."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


def authored_population():
    from obruxo_data.hashing import canonical_json
    from obruxo_training.full_scalar import FullScalarCodec
    from obruxo_training.identity import PresetIdentity
    from obruxo_training.pilot import PilotPool
    from obruxo_training.splits import FrozenVariantManifest
    pool = PilotPool.create(256, codec=FullScalarCodec())
    parents = dict(pool.manifest.variants)
    members, variants = [], []
    for index, entry in enumerate(pool.entries):
        document = json.loads(pool.codec.decode(entry.target))
        rng = np.random.default_rng(index)
        for slot, shape in enumerate(document["settings"]["lfos"], 1):
            count = 3 + (index + slot) % 6
            x = np.linspace(0, 1, count, dtype=np.float32)
            y = rng.uniform(.05, .95, count).astype(np.float32)
            shape.update(name=f"Authored {index} slot {slot}", num_points=count,
                         points=np.column_stack((x, y)).reshape(-1).tolist(),
                         powers=rng.uniform(-4, 4, count).astype(np.float32).tolist(), smooth=bool((index + slot) % 2))
        raw = canonical_json(document).encode()
        state = PresetIdentity.from_bytes(raw).state
        members.append((state, raw))
        variants.append((state, parents[entry.state_id]))
    return members, FrozenVariantManifest(pool.manifest.parent, tuple(variants))


def main():
    import torch
    from obruxo_data.hashing import canonical_json, canonical_sha256
    from obruxo_data.render.base import FULL_PRESET_RENDERER_ID, RenderRequest
    from obruxo_data.vital.full_contract import FullPresetContract
    from obruxo_training.audit import write_json
    from obruxo_training.benchmark import MemoryMonitor
    from obruxo_training.lfo_bank import LfoBank, LfoShape, curve_to_lfo, fit_lfo_bank, runtime_sources, sample_lfo, sample_shape
    from obruxo_training.evaluate import audio_distance
    from obruxo_training.performances import initial_bank
    from obruxo_training.reproducibility import OSCILLATOR_PHASE_CONTROL
    from obruxo_training.stream import NativeRendererFactory, RenderWorker
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    members, manifest = authored_population()
    train = [raw for state, raw in members if manifest.by_state[state].split == "train"]
    chosen = {split: next((state, raw) for state, raw in members if manifest.by_state[state].split == split) for split in ("train", "validation")}
    panel = [{"slot": slot, "state_id": chosen[split][0], "split": split} for slot in range(1, 9)
             for split in ("train" if slot % 2 else "validation",)]
    files = [Path(__file__), *(RESEARCH / "training/obruxo_training").glob("*.py"),
             *(RESEARCH / "data_generation/obruxo_data/render").glob("*.py"), RESEARCH / "data_generation/obruxo_data/vital/full_contract.py"]

    def sources():
        return {path.relative_to(RESEARCH).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(files)}

    plan = {"check": "train_only_lfo_bank_native_v1", "maximum_curves": 64, "source_sha256": sources(), "runtime_sources": runtime_sources(),
            "authored_point_precision": "float32_before_state_identity_and_fitting",
            "root_manifest_id": manifest.parent.manifest_id, "manifest_id": manifest.manifest_id,
            "states_by_split": {split: sum(entry.split == split for entry in manifest.assignments) for split in ("train", "validation", "test")},
            "panel": panel, "performances": [initial_bank()[index].to_dict() for index in (0, 4)],
            "native_attempt_budget": 80, "timeout_seconds": 30, "worker_jobs": 4, "rss_limit_bytes": 8 * 1024**3,
            "phase_control": {**OSCILLATOR_PHASE_CONTROL, "sample_random_phase": 0.}, "final_test_audio": False,
            "model_training": False, "quality_threshold_calibrated": False}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "plan.json", plan)
    write_json(args.output / "status.json", {"complete": False})
    monitor, started = MemoryMonitor(), perf_counter()
    monitor.start()
    events, checks = [], []
    try:
        bank = fit_lfo_bank(train, manifest, maximum_curves=plan["maximum_curves"], progress=lambda value: print(value, flush=True))
        write_json(args.output / "bank.json", bank.to_dict())
        restored = LfoBank(json.loads((args.output / "bank.json").read_text()))
        factory = NativeRendererFactory(renderer_id=FULL_PRESET_RENDERER_ID)
        with RenderWorker(renderer_factory=factory, max_jobs=4) as worker:
            for case in panel:
                slot, split = case["slot"], case["split"]
                shape = json.loads(chosen[split][1])["settings"]["lfos"][slot - 1]
                values = sample_lfo(shape)
                codes, expected = bank.encode(values)
                decoded = restored.decode(codes)[0]
                grid, rebuilt = curve_to_lfo(values), curve_to_lfo(decoded)
                dense = sample_shape(LfoShape.from_json(shape), 4097)
                grid_dense = sample_shape(LfoShape.from_json(grid), 4097)
                row = {**case, "decoder_matches_encoder_exactly": np.array_equal(decoded, expected[0]),
                       "sampled_curve_rmse": float(np.sqrt(np.mean((values - decoded)**2))),
                       "original_to_grid_dense_curve_rmse": float(np.sqrt(np.mean((dense - grid_dense)**2))),
                       "code": codes[0].to_dict(), "performances": {}}
                for performance in (initial_bank()[0], initial_bank()[4]):
                    audio = {}
                    for name, selected_shape in (("original", shape), ("original_repeat", shape), ("grid", grid), ("code", rebuilt), ("code_repeat", rebuilt)):
                        document = FullPresetContract().template()
                        settings = document["settings"]
                        settings.update(plan["phase_control"])
                        settings.update(osc_1_level=.12, modulation_1_amount=.08)
                        settings[f"lfo_{slot}_frequency"], settings[f"lfo_{slot}_sync"] = 0., 0.
                        settings["lfos"][slot - 1] = deepcopy(selected_shape)
                        settings["modulations"][0] = {"source": f"lfo_{slot}", "destination": "osc_1_level"}
                        request = RenderRequest(preset_json=canonical_json(document), performance=performance.performance(), renderer_id=FULL_PRESET_RENDERER_ID)
                        audio[name], metadata = worker.render(request, timeout=30)
                        events.append({**case, "performance": performance.name, "kind": name, "request_id": request.request_id, **metadata})
                        write_json(args.output / "events.json", events)
                    valid = all(value is not None for value in audio.values())
                    row["performances"][performance.name] = {"all_accepted": valid,
                        "original_repeat_exact": valid and np.array_equal(audio["original"], audio["original_repeat"]),
                        "code_repeat_exact": valid and np.array_equal(audio["code"], audio["code_repeat"]),
                        "distances": {f"{left}_to_{right}": audio_distance(audio[left], audio[right]) for left, right in
                                      (("original", "grid"), ("grid", "code"), ("original", "code"))} if valid else {}}
                checks.append(row)
                write_json(args.output / "checks.json", checks)
                print(json.dumps({"slot": slot, "split": split, "completed": len(checks)}), flush=True)
    finally:
        monitor.close()
    gates = {"all_train_only": all(manifest.by_state[row["state_id"]].split == "train" for row in bank.to_dict()["bank"]["fitted_rows"]),
             "restored_bank_identity": restored.bank_id == bank.bank_id,
             "decoder_exact": all(row["decoder_matches_encoder_exactly"] for row in checks),
             "native_accepted": len(events) == 80 and all(row["status"] == "accepted" for row in events),
             "fixed_phase_repeats_exact": all(value["original_repeat_exact"] and value["code_repeat_exact"] for row in checks for value in row["performances"].values()),
             "source_unchanged": plan["source_sha256"] == sources() and plan["runtime_sources"] == runtime_sources(),
             "memory_bounded": monitor.errors == 0 and monitor.peak_bytes is not None and monitor.peak_bytes <= plan["rss_limit_bytes"]}
    content = bank.to_dict()["bank"]
    summary = {"plan_sha256": canonical_sha256(plan), "gates": gates, "passed": all(gates.values()), "bank_id": bank.bank_id,
               "fit": {name: content[name] for name in ("source_presets", "source_curve_rows", "unique_training_curves", "maximum_curves", "fitted_rows")},
               "attempted": len(events), "accepted": sum(row["status"] == "accepted" for row in events), "checks": checks,
               "peak_process_rss_bytes": monitor.peak_bytes, "memory_sampling_errors": monitor.errors,
               "elapsed_seconds": perf_counter() - started, "final_test_audio": False, "audio_files_written": 0,
               "audio_to_code_training": False, "codec_quality_gate": "not_calibrated_or_passed"}
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "status.json", {"complete": True, "passed": summary["passed"]})
    print(json.dumps({"passed": summary["passed"], "gates": gates}), flush=True)
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Native effect, slot isolation and exact uint32 readback checks for nested state."""
from copy import deepcopy
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np

from obruxo_data.hashing import canonical_json, canonical_sha256
from obruxo_data.render.base import FULL_SEED_RENDERER_ID, RenderRequest

from .asset_codec import FactoryAssetCodec
from .audit import write_json
from .benchmark import MemoryMonitor
from .lfo_bank import curve_to_lfo, sample_lfo
from .nested_codec import SEED_MODE, WARP_MODES
from .performances import initial_bank
from .reproducibility import FULL_PHASE_CONTROL
from .stream import NativeRendererFactory, RenderWorker

SEED_EDGES = (0, 1, 4, 16777217, 2147483647, 2147483648, 4294967294, 4294967295)
SILENT_SEEDS = (0, 1, 16777217)


def probe_warp(high=False):
    points = [0., .1, .5, .8, 1., .2] if high else [0., 1., .5, .2, 1., 0.]
    return {"name": "Authored warp check", "num_points": 3, "points": np.asarray(points, dtype=np.float32).tolist(),
            "powers": np.asarray([.25, -.25, 0.], dtype=np.float32).tolist(), "smooth": bool(high)}


def check_nested(asset_bank, output, *, timeout=30., progress=None):
    assets = FactoryAssetCodec(asset_bank)
    scalar, schema = assets.scalar, assets.schema
    wave = next((value for value in assets._waves if value.get("name") == "Basic Shapes"), assets._waves[1])
    base = schema.template()
    base["settings"].update(FULL_PHASE_CONTROL)
    base["settings"].update(volume=3844., osc_1_on=0., osc_2_on=0., osc_3_on=0., sample_on=0.)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    research = Path(__file__).resolve().parents[2]
    files = [Path(__file__), research / "training/obruxo_training/nested_codec.py", research / "training/obruxo_training/stream.py",
             research / "training/obruxo_training/contracts.py", research / "data_generation/obruxo_data/render/vita.py",
             research / "data_generation/obruxo_data/render/full_state.py", research / "data_generation/obruxo_data/render/base.py",
             research / "data_generation/obruxo_data/vital/full_contract.py"]
    sources = {path.relative_to(research).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    plan = {"check": "three_warps_and_uint32_seeds_native_v1", "source_sha256": sources, "factory_bank_id": assets.bank["bank_id"],
            "renderer_id": FULL_SEED_RENDERER_ID, "warp_modes": list(WARP_MODES), "seed_mode": SEED_MODE,
            "seed_edges": list(SEED_EDGES), "attempt_budget": 150, "causal_relative_rms_minimum": .001,
            "expected_silent_seeds": list(SILENT_SEEDS), "silence_scope": "Basic_Shapes_frame128_mode6_amount0.6_only",
            "phase_control": FULL_PHASE_CONTROL, "rss_limit_bytes": 8 * 1024**3, "final_test_audio": False,
            "representation_error": "measured_diagnostic_not_general_curve_fidelity_or_model_quality_qualification"}
    write_json(output / "plan.json", plan)
    events, cases, expected_rejected, monitor, started = [], [], set(), MemoryMonitor(), perf_counter()
    monitor.start()
    try:
        with RenderWorker(renderer_factory=NativeRendererFactory(renderer_id=FULL_SEED_RENDERER_ID), max_jobs=4) as worker:
            def render(case, role, document):
                settings = document["settings"]
                projected = scalar.settings(scalar._target(settings))
                settings.update({name: projected[name] for name in schema.scalar_names})
                schema.validate(document).require_valid()
                request = RenderRequest(preset_json=canonical_json(document), performance=initial_bank()[0].performance(), renderer_id=FULL_SEED_RENDERER_ID)
                audio, metadata = worker.render(request, timeout=timeout)
                events.append({"case": case, "role": role, "request_id": request.request_id, **metadata})
                write_json(output / "events.json", events)
                return audio if metadata["status"] == "accepted" else None

            def source(slot, mode):
                document = deepcopy(base)
                settings = document["settings"]
                settings.update({f"osc_{slot}_on": 1., f"osc_{slot}_level": .15, f"osc_{slot}_wave_frame": 128.,
                                 f"osc_{slot}_spectral_morph_type": float(mode), f"osc_{slot}_spectral_morph_amount": .6})
                settings["wavetables"][slot - 1] = deepcopy(wave)
                return document

            def difference(left, right):
                if left is None or right is None:
                    return None
                return float(np.sqrt(np.mean((left.astype(np.float64) - right) ** 2)) / max(1e-12, np.sqrt(np.mean(left.astype(np.float64) ** 2))))

            def exact(left, right):
                return left is not None and right is not None and np.array_equal(left, right)

            def record(row):
                cases.append(row)
                write_json(output / "checks.json", cases)
                if progress:
                    progress({"case": row["case"], "attempted": len(events), "budget": plan["attempt_budget"], **row})

            for slot in range(1, 4):
                other = slot % 3
                for mode in WARP_MODES:
                    name = f"warp_{slot}_mode_{mode}"
                    document = source(slot, mode)
                    left, right, repeated, representations = [], [], [], []
                    for variant in (False, True):
                        document["settings"]["custom_warps"][slot - 1] = probe_warp(variant)
                        original = render(name, "original_" + str(variant), deepcopy(document))
                        repeat = render(name, "repeat_" + str(variant), deepcopy(document))
                        projected = deepcopy(document)
                        projected["settings"]["custom_warps"][slot - 1] = curve_to_lfo(sample_lfo(probe_warp(variant)))
                        reconstruction = render(name, "projection_" + str(variant), projected)
                        (right if variant else left).append(original)
                        repeated.append(exact(original, repeat))
                        representations.append(difference(original, reconstruction))
                    isolated = deepcopy(document)
                    isolated["settings"]["custom_warps"][other] = probe_warp(False)
                    isolated_audio = render(name, "other_slot_curve", isolated)
                    delta = difference(left[0], right[0])
                    record({"case": name, "repeats_exact": all(repeated), "changed_curve_relative_rms": delta,
                            "causal_effect": delta is not None and delta > plan["causal_relative_rms_minimum"],
                            "other_slot_unchanged": exact(right[0], isolated_audio), "projection_relative_rms": representations})
                for seed in SEED_EDGES:
                    name = f"seed_{slot}_{seed}"
                    document = source(slot, 0)
                    document["settings"]["random_values"][slot - 1]["seed"] = seed
                    original = render(name, "original", deepcopy(document))
                    repeat = render(name, "repeat", deepcopy(document))
                    record({"case": name, "repeats_exact": exact(original, repeat), "claim": "exact_seed_state_with_audible_carrier"})
                name = f"seed_effect_{slot}"
                document = source(slot, SEED_MODE)
                original = render(name, "original_4", deepcopy(document))
                repeat = render(name, "repeat_4", deepcopy(document))
                document["settings"]["random_values"][slot - 1]["seed"] = 123456789
                changed = render(name, "original_123456789", deepcopy(document))
                changed_repeat = render(name, "repeat_123456789", deepcopy(document))
                document["settings"]["random_values"][other]["seed"] = 123456789
                isolated = render(name, "other_slot_seed", document)
                delta = difference(original, changed)
                record({"case": name, "repeats_exact": exact(original, repeat) and exact(changed, changed_repeat),
                        "changed_seed_relative_rms": delta, "causal_effect": delta is not None and delta > plan["causal_relative_rms_minimum"],
                        "other_slot_unchanged": exact(changed, isolated)})
                for seed in SILENT_SEEDS:
                    name = f"silent_seed_{slot}_{seed}"
                    document = source(slot, SEED_MODE)
                    document["settings"]["random_values"][slot - 1]["seed"] = seed
                    silent = []
                    for role in ("original", "repeat"):
                        render(name, role, deepcopy(document))
                        silent.append(events[-1])
                    valid = all(row["status"] == "rejected" and row.get("diagnostic_codes") == ["audio.silence"]
                                and row.get("qa", {}).get("peak") == 0. and row.get("qa", {}).get("finite") is True
                                and row.get("renderer_qa", {}).get("native_state", {}).get("verified_loaded_scalars") == 775 for row in silent)
                    valid = valid and len({row["qa"]["audio_float32_sha256"] for row in silent}) == 1
                    if valid:
                        expected_rejected.update((name, row["role"]) for row in silent)
                    record({"case": name, "expected_exact_silence": valid, "training_audio_eligible": False})
                document = source(slot, 0)
                original = render(f"inactive_{slot}", "original", deepcopy(document))
                document["settings"]["custom_warps"][slot - 1] = probe_warp(True)
                document["settings"]["random_values"][slot - 1]["seed"] = 0xFFFFFFFF
                changed = render(f"inactive_{slot}", "inactive_changes", document)
                record({"case": f"inactive_{slot}", "inactive_audio_exact": exact(original, changed)})
    finally:
        monitor.close()
    gates = {"native_state_and_qa": len(events) == plan["attempt_budget"] and all(row["status"] == "accepted" or (row["case"], row["role"]) in expected_rejected for row in events),
             "fixed_phase_repeats": all(row.get("repeats_exact", True) for row in cases),
             "causal_effects": all(row.get("causal_effect", True) for row in cases),
             "slot_isolation": all(row.get("other_slot_unchanged", True) for row in cases),
             "inactive_audio_exact": all(row.get("inactive_audio_exact", True) for row in cases),
             "known_silent_states_rejected": len(expected_rejected) == 18 and all(row.get("expected_exact_silence", True) for row in cases),
             "source_unchanged": all(hashlib.sha256((research / name).read_bytes()).hexdigest() == digest for name, digest in sources.items()),
             "memory_bounded": monitor.errors == 0 and monitor.peak_bytes is not None and monitor.peak_bytes <= plan["rss_limit_bytes"]}
    summary = {"plan_sha256": canonical_sha256(plan), "passed": all(gates.values()), "gates": gates, "cases": cases,
               "attempted": len(events), "accepted": sum(row["status"] == "accepted" for row in events),
               "expected_silence_rejections": len(expected_rejected),
               "elapsed_seconds": perf_counter() - started, "peak_process_rss_bytes": monitor.peak_bytes, "memory_sampling_errors": monitor.errors,
               "final_test_audio": False, "audio_files_written": 0}
    write_json(output / "summary.json", summary)
    return summary

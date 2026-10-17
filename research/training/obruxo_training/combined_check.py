"""Bounded native source, asset, family-effect and phase-repeat qualification."""
from copy import deepcopy
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np

from obruxo_data.hashing import canonical_json, canonical_sha256
from obruxo_data.midi import Performance
from obruxo_data.render.base import RenderRequest

from .audit import write_json
from .benchmark import MemoryMonitor
from .combined_codec import CombinedCodec
from .full_scalar import EFFECTS
from .performances import bank_for
from .reproducibility import FULL_PHASE_CONTROL, fixed_generator_phases
from .stream import NativeRendererFactory, RenderWorker


def source_indices(pool):
    codec = pool.codec
    selected = []
    for index, entry in enumerate(pool.entries):
        if pool.manifest.by_state[entry.state_id].split != "train":
            continue
        ordinal = codec.routing._ordinal[codec._root(index)]
        if (ordinal % 4 == 3 and ordinal // 4 < len(codec.base._waves)) or (ordinal % 4 != 3 and ordinal < 12):
            selected.append(index)
    return selected


def family_cases(codec):
    base = codec.schema.template()
    base["settings"].update(FULL_PHASE_CONTROL)
    base["settings"].update(volume=2500., osc_1_level=.15, osc_1_destination=3., sample_on=0.)
    references = [entry.performance() for entry in bank_for(codec.contract.renderer_id) if entry.role == "reference"]
    cases = []
    controls = {"chorus": {"chorus_dry_wet": .3}, "compressor": {"compressor_mix": .7, "compressor_low_gain": 0.,
        "compressor_band_gain": 0., "compressor_high_gain": 0., "compressor_low_upper_threshold": -50.,
        "compressor_band_upper_threshold": -50., "compressor_high_upper_threshold": -50.},
        "delay": {"delay_dry_wet": .3, "delay_feedback": .2, "delay_frequency": 3., "delay_sync": 0.},
        "distortion": {"distortion_drive": 6., "distortion_mix": .5}, "eq": {"eq_band_gain": 6., "eq_band_cutoff": 69.},
        "filter_fx": {"filter_fx_cutoff": 69., "filter_fx_mix": .6}, "flanger": {"flanger_dry_wet": .3},
        "phaser": {"phaser_dry_wet": .5}, "reverb": {"reverb_dry_wet": .3, "reverb_decay_time": -.5}}
    for effect in EFFECTS:
        changed = deepcopy(base)
        changed["settings"].update(controls[effect], **{effect + "_on": 1.})
        cases.append((effect, deepcopy(base), changed, references[0]))
    sampler = deepcopy(base)
    sampler["settings"].update(osc_1_on=0., sample_on=1., sample_level=.15, sample_destination=3., sample_keytrack=1.)
    changed = deepcopy(sampler)
    changed["settings"]["sample"] = deepcopy(codec.base._samples[1])
    cases.append(("sample_content", sampler, changed, references[0]))
    mono = deepcopy(base)
    mono["settings"].update(voice_override=1., polyphony=1.)
    cases.append(("voice_polyphony", deepcopy(base), mono, references[3]))
    glide = Performance(bpm=120)
    glide.add_note(pitch=60, velocity=96, start_tick=0, duration_ticks=960)
    glide.add_note(pitch=72, velocity=96, start_tick=480, duration_ticks=960)
    glide.end_tick = 1920
    before = deepcopy(mono)
    before["settings"].update(legato=1., portamento_time=-10.)
    after = deepcopy(before)
    after["settings"].update(portamento_time=-4., portamento_force=1.)
    cases.append(("portamento", before, after, glide))
    for name, source, control, low, high in (("lfo_playback", "lfo_1", "lfo_1_frequency", -1., 1.),
            ("random_playback", "random_1", "random_1_style", 0., 3.), ("secondary_envelope", "env_2", "env_2_attack", .1, .6)):
        before = deepcopy(base)
        before["settings"]["modulations"][0] = {"source": source, "destination": "osc_1_pan"}
        before["settings"].update(modulation_1_amount=.2, lfo_1_sync=0., random_1_sync=0., random_1_frequency=2., env_2_sustain=1.)
        before["settings"][control] = low
        after = deepcopy(before)
        after["settings"][control] = high
        cases.append((name, before, after, references[0]))
    return cases


def check_combined(pool, output, *, progress=None):
    codec = pool.codec
    if not isinstance(codec, CombinedCodec) or len(pool.entries) != 256:
        raise ValueError("combined native qualification requires the frozen 256-state combined pool")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    research = Path(__file__).resolve().parents[2]
    files = list((research / "training/obruxo_training").glob("*.py"))
    files += [research / "data_generation/obruxo_data" / name for name in ("render/base.py", "render/vita.py", "render/capabilities.py",
              "render/midi_transport.py", "render/full_state.py", "midi/events.py", "midi/validation.py", "vital/full_contract.py")]

    def sources():
        return {path.relative_to(research).as_posix(): hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for path in sorted(files)}

    indices, families = source_indices(pool), family_cases(codec)
    asset_offset = len(codec.base.scalar.categorical)
    choices = [set() for _ in range(4)]
    for index in indices:
        target = pool.entries[index].target
        flags = codec.categorical_mask(target)[asset_offset:asset_offset + 4]
        for seen, choice, enabled in zip(choices, target.categorical[asset_offset:asset_offset + 4], flags, strict=True):
            if enabled:
                seen.add(choice)
    asset_sizes = [len(codec.base._waves)] * 3 + [len(codec.base._samples)]
    assets_covered = all(seen == set(range(size)) for seen, size in zip(choices, asset_sizes, strict=True))
    if not assets_covered:
        raise ValueError("native source fixtures must cover every registered choice in every active slot")
    plan = {"qualification": "combined_source_and_family_native_v1", "pool_id": pool.pool_id, "codec_id": codec.codec_id,
            "source_sha256_lf": sources(), "training_contract": codec.contract.to_dict(), "source_indices": indices,
            "family_cases": [name for name, *_ in families], "attempt_budget": len(indices) * 2 + len(families) * 3,
            "registered_choices_by_slot": asset_sizes,
            "phase_control": FULL_PHASE_CONTROL, "minimum_effect_relative_rms": .005, "worker_jobs": 1,
            "random_generator_policy": "fresh_process_per_request_preserve_all_requested_random_routes",
            "rss_limit_bytes": 8 * 1024**3, "final_test_audio": False, "model_promotion": False}
    write_json(output / "plan.json", plan)
    references = [entry for entry in bank_for(codec.contract.renderer_id) if entry.role == "reference"]
    events, checks, label_checks = [], [], []
    monitor, started = MemoryMonitor(), perf_counter()
    monitor.start()
    try:
        with RenderWorker(renderer_factory=NativeRendererFactory(renderer_id=codec.contract.renderer_id), max_jobs=1) as worker:
            def render(name, role, raw, performance):
                request = RenderRequest(preset_json=fixed_generator_phases(raw).decode(), performance=performance, tail_seconds=4., renderer_id=codec.contract.renderer_id)
                audio, metadata = worker.render(request, timeout=60.)
                events.append({"case": name, "role": role, "request_id": request.request_id, **metadata})
                write_json(output / "events.json", events)
                return audio if metadata["status"] == "accepted" else None

            def record(row):
                checks.append(row)
                write_json(output / "checks.json", checks)
                if progress:
                    progress({**row, "attempted": len(events), "budget": plan["attempt_budget"]})

            for index in indices:
                entry = pool.entries[index]
                raw = codec.source_preset(entry.target, index)
                label_checks.append(codec.encode(raw) == entry.target)
                ordinal = codec.routing._ordinal[codec._root(index)]
                performance = references[(ordinal // 4) % len(references)].performance()
                first = render(f"source_{index}", "source", raw, performance)
                repeat = render(f"source_{index}", "repeat", raw, performance)
                record({"case": f"source_{index}", "family": codec.family(entry.target), "source_labels_exact": label_checks[-1],
                        "repeat_exact": first is not None and repeat is not None and np.array_equal(first, repeat)})
            for name, before, after, performance in families:
                for document in (before, after):
                    document["settings"].update(codec.base.scalar._values(codec.base.scalar._target(document["settings"])))
                first = render(name, "before", canonical_json(before).encode(), performance)
                changed = render(name, "after", canonical_json(after).encode(), performance)
                repeat = render(name, "repeat", canonical_json(after).encode(), performance)
                relative = None if first is None or changed is None else float(np.sqrt(np.mean((first.astype(np.float64) - changed) ** 2)) / max(1e-12, np.sqrt(np.mean(first.astype(np.float64) ** 2))))
                record({"case": name, "relative_rms_change": relative, "effect_present": relative is not None and relative > plan["minimum_effect_relative_rms"],
                        "repeat_exact": changed is not None and repeat is not None and np.array_equal(changed, repeat)})
    finally:
        monitor.close()
    gates = {"all_native_accepted": len(events) == plan["attempt_budget"] and all(row["status"] == "accepted" for row in events),
             "registered_choices_covered": assets_covered,
             "original_source_labels": all(label_checks), "fixed_phase_repeats": all(row["repeat_exact"] for row in checks),
             "family_effects": all(row.get("effect_present", True) for row in checks), "source_unchanged": sources() == plan["source_sha256_lf"],
             "memory_bounded": monitor.errors == 0 and monitor.peak_bytes is not None and monitor.peak_bytes <= plan["rss_limit_bytes"]}
    summary = {"plan_sha256": canonical_sha256(plan), "passed": all(gates.values()), "gates": gates, "checks": checks,
               "attempted": len(events), "accepted": sum(row["status"] == "accepted" for row in events), "elapsed_seconds": perf_counter() - started,
               "peak_process_rss_bytes": monitor.peak_bytes, "memory_sampling_errors": monitor.errors, "audio_files_written": 0,
               "final_test_audio": False, "model_promotion": False}
    write_json(output / "summary.json", summary)
    return summary

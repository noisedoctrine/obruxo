"""Matched native destination responses; readback, repeatability and audibility are separate."""
from copy import deepcopy
import hashlib
from pathlib import Path
import re
from time import perf_counter

import numpy as np

from obruxo_data.hashing import canonical_json, canonical_sha256
from obruxo_data.midi import EventKind, MidiEvent, Performance
from obruxo_data.render.base import FULL_CURVE_RENDERER_ID, RenderRequest
from obruxo_data.vital.full_contract import FullPresetContract

from .asset_codec import FactoryAssetCodec
from .audit import write_json
from .benchmark import MemoryMonitor
from .full_scalar import EFFECTS
from .graph_context import prepare_destination_context
from .reproducibility import FULL_PHASE_CONTROL
from .stream import NativeRendererFactory, RenderWorker


def destination_fixture(destination, assets, overrides=None):
    schema = FullPresetContract(renderer_id=FULL_CURVE_RENDERER_ID)
    document = schema.template()
    settings = document["settings"]
    wave = next((value for value in assets._waves if value.get("name") == "Basic Shapes"), assets._waves[1])
    settings["wavetables"] = [deepcopy(wave) for _ in range(3)]
    settings.update(FULL_PHASE_CONTROL, volume=2500., osc_1_on=1., osc_2_on=0., osc_3_on=0., sample_on=0.,
                    osc_1_level=.15, osc_1_destination=4., osc_1_wave_frame=128., env_1_delay=0., env_1_attack=.1,
                    env_1_sustain=.8, env_1_release=.4)
    settings["modulations"] = [{"source": "", "destination": ""} for _ in range(64)]
    for slot in range(1, 65):
        settings.update({f"modulation_{slot}_amount": 0., f"modulation_{slot}_power": 0.,
                         f"modulation_{slot}_bypass": 0., f"modulation_{slot}_bipolar": 0., f"modulation_{slot}_stereo": 0.})
    oscillator = re.match(r"osc_([123])_", destination)
    if oscillator:
        slot = int(oscillator[1])
        settings.update({f"osc_{number}_on": float(number == slot) for number in range(1, 4)})
        settings[f"osc_{slot}_level"] = .15
    if destination.startswith("sample_"):
        settings.update(osc_1_on=0., sample_on=1., sample_level=.15, sample_destination=4.)
    for effect in EFFECTS:
        if destination.startswith(effect + "_"):
            settings.update({effect + "_on": 1., "osc_1_destination": 3.})
    settings.update(delay_feedback=.15, delay_frequency=3., reverb_decay_time=-2., reverb_delay=.01,
                    compressor_mix=.7, compressor_low_gain=0., compressor_band_gain=0., compressor_high_gain=0.,
                    compressor_low_upper_threshold=-24., compressor_band_upper_threshold=-24., compressor_high_upper_threshold=-24.)
    prepare_destination_context(settings, destination, assets._waves, assets._samples)
    # A downstream modulation path makes envelope/rate/phase targets observable.
    module = re.match(r"(lfo_[1-8]|random_[1-4]|env_[2-6])_", destination)
    if module or destination == "beats_per_minute":
        source = module[1] if module else "lfo_1"
        settings["modulations"][0] = {"source": source, "destination": "osc_1_pan"}
        settings["modulation_1_amount"] = .3
    referenced = re.fullmatch(r"modulation_(\d+)_(?:amount|power)", destination)
    child = int(referenced[1]) - 1 if referenced else None
    if child is not None:
        settings["modulations"][child] = {"source": "lfo_2", "destination": "osc_1_pan"}
        settings[f"modulation_{child + 1}_amount"] = .3
    owner = next(slot for slot in range(63, 0, -1) if slot != child)
    settings["modulations"][owner] = {"source": "mod_wheel", "destination": destination}
    metadata = assets.scalar._parameters[destination]
    position = (settings[destination] - metadata.minimum) / (metadata.maximum - metadata.minimum)
    settings[f"modulation_{owner + 1}_amount"] = .1 if position < .5 else -.1
    if overrides:
        if set(overrides) - schema.scalar_names:
            raise ValueError("native response overrides must name registered scalar controls")
        settings.update(overrides)
    settings.update(assets.scalar._values(assets.scalar._target(settings)))
    schema.validate(document).require_valid()
    return document, owner


def check_destinations(bank, output, *, destinations=None, overrides=None, progress=None):
    assets = FactoryAssetCodec(bank)
    schema = FullPresetContract(renderer_id=FULL_CURVE_RENDERER_ID)
    names = tuple(sorted(schema.destinations)) if destinations is None else tuple(destinations)
    if len(set(names)) != len(names) or not names or set(names) - schema.destinations:
        raise ValueError("response panel requires distinct registered destinations")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    research = Path(__file__).resolve().parents[2]
    files = [Path(__file__), Path(__file__).with_name("graph_context.py"),
             *(research / "data_generation/obruxo_data/render").glob("*.py"), research / "data_generation/obruxo_data/vital/full_contract.py"]

    def sources():
        return {path.relative_to(research).as_posix(): hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for path in sorted(set(files))}

    plan = {"qualification": "native_connection_destination_response_v1", "source_sha256_lf": sources(), "bank_id": bank["bank_id"],
            "destinations": list(names), "destination_inventory_size": len(schema.destinations), "attempt_budget": 3 * len(names),
            "phase_control": FULL_PHASE_CONTROL, "worker_jobs": 1, "overrides": overrides or {},
            "driver": "CC1_127_before_note_on_mod_wheel_source", "minimum_relative_rms_response": 1e-5,
            "claim": "response_in_declared_context_zero_is_unresolved_not_unsupported", "rss_limit_bytes": 8 * 1024**3,
            "final_test_audio": False, "model_promotion": False}
    write_json(output / "plan.json", plan)
    performance = Performance(bpm=120)
    performance.events.append(MidiEvent(0, 1, EventKind.CONTROL_CHANGE, 0, (1, 127)))
    performance.add_note(pitch=60, velocity=90, start_tick=240, duration_ticks=1440)
    performance.end_tick = 2400
    rows, checks = [], []
    monitor, started = MemoryMonitor(), perf_counter()
    monitor.start()
    try:
        with RenderWorker(renderer_factory=NativeRendererFactory(renderer_id=FULL_CURVE_RENDERER_ID), max_jobs=1) as worker:
            for name in names:
                changed, owner = destination_fixture(name, assets, (overrides or {}).get(name))
                before = deepcopy(changed)
                before["settings"][f"modulation_{owner + 1}_bypass"] = 1.
                pair, buffers = [], []
                for role, document in (("bypassed", before), ("connected", changed), ("repeat", changed)):
                    request = RenderRequest(preset_json=canonical_json(document), performance=performance, tail_seconds=4., renderer_id=FULL_CURVE_RENDERER_ID)
                    audio, metadata = worker.render(request, timeout=60.)
                    row = {"destination": name, "role": role, "request_id": request.request_id, **metadata}
                    rows.append(row)
                    pair.append(row)
                    buffers.append(audio)
                    write_json(output / "events.json", rows)
                valid = all(row["status"] == "accepted" for row in pair)
                relative = None if not valid else float(np.sqrt(np.mean((buffers[0].astype(np.float64) - buffers[1]) ** 2)) /
                    max(1e-12, np.sqrt(np.mean(buffers[0].astype(np.float64) ** 2))))
                checks.append({"destination": name, "accepted": valid, "relative_rms_change": relative,
                    "response_observed": relative is not None and relative > plan["minimum_relative_rms_response"],
                    "repeat_exact": valid and pair[1]["qa"]["audio_float32_sha256"] == pair[2]["qa"]["audio_float32_sha256"],
                    "statuses": [row["status"] for row in pair]})
                write_json(output / "checks.json", checks)
                if progress:
                    progress({**checks[-1], "completed_destinations": len(checks), "planned_destinations": len(names)})
    finally:
        monitor.close()
    gates = {"all_attempts_accounted": len(rows) == plan["attempt_budget"], "all_native_accepted": all(row["accepted"] for row in checks),
             "fixed_phase_repeats": all(row["repeat_exact"] for row in checks), "source_unchanged": sources() == plan["source_sha256_lf"],
             "memory_bounded": monitor.errors == 0 and monitor.peak_bytes is not None and monitor.peak_bytes <= plan["rss_limit_bytes"]}
    summary = {"plan_sha256": canonical_sha256(plan), "passed": all(gates.values()), "gates": gates, "checks": checks,
               "destinations_checked": len(checks), "responses_observed": sum(row["response_observed"] for row in checks),
               "unresolved_destinations": [row["destination"] for row in checks if not row["response_observed"]],
               "attempted": len(rows), "elapsed_seconds": perf_counter() - started, "peak_process_rss_bytes": monitor.peak_bytes,
               "audio_files_written": 0, "final_test_audio": False, "model_promotion": False}
    write_json(output / "summary.json", summary)
    return summary

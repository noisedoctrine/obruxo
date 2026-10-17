"""Native notes compatibility, expression timing and MPE isolation qualification."""
from copy import deepcopy
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np

from obruxo_data.hashing import canonical_json, canonical_sha256
from obruxo_data.midi import EventKind, MidiEvent, Performance, TempoMap
from obruxo_data.render.base import FULL_EXPRESSION_RENDERER_ID, FULL_SEED_RENDERER_ID, RenderRequest

from .audit import write_json
from .benchmark import MemoryMonitor
from .expression_bank import expression_bank
from .full_scalar import FullScalarCodec
from .performances import initial_bank
from .reproducibility import FULL_PHASE_CONTROL
from .stream import NativeRendererFactory, RenderWorker


def held_expression(channel=0, release=0):
    value = Performance(bpm=120)
    value.add_note(pitch=69, velocity=96, start_tick=120, duration_ticks=1320, channel=channel, release_velocity=release)
    value.end_tick = 2400
    return value


def boundary_performance():
    value = Performance(ticks_per_beat=88200, bpm=60)
    for index, tick in enumerate((1, 3, 127, 129, 255, 257)):
        value.add_note(pitch=60 + index, velocity=90, start_tick=tick, duration_ticks=8820)
    value.end_tick = 17640
    return value


def legacy_chord_noteoffs(performance):
    """Reproduce the old API's observed double-to-integer floor at 1.4 seconds."""
    timing = TempoMap.from_performance(performance)
    value = Performance(ticks_per_beat=17640, bpm=150)
    for note in performance.note_spans():
        start = timing.tick_to_sample(note.start_tick, 44100)
        end = timing.tick_to_sample(note.end_tick, 44100)
        value.add_note(pitch=note.pitch, velocity=note.velocity, start_tick=start, duration_ticks=end - start - 1)
    value.end_tick = timing.tick_to_sample(performance.end_tick, 44100)
    return value


def expression_specs():
    return (("pitch_bend", EventKind.PITCH_BEND, (4096,), None, 0, 0),
            ("mod_wheel", EventKind.CONTROL_CHANGE, (1, 127), "mod_wheel", 0, 0),
            ("pressure", EventKind.CHANNEL_PRESSURE, (127,), "aftertouch", 0, 0),
            ("slide_2", EventKind.CONTROL_CHANGE, (74, 127), "slide", 1, 1),
            ("slide_16", EventKind.CONTROL_CHANGE, (74, 127), "slide", 15, 1),
            ("pressure_2", EventKind.CHANNEL_PRESSURE, (127,), "aftertouch", 1, 1),
            ("pressure_16", EventKind.CHANNEL_PRESSURE, (127,), "aftertouch", 15, 1),
            ("bend_2", EventKind.PITCH_BEND, (4096,), None, 1, 1),
            ("bend_16", EventKind.PITCH_BEND, (4096,), None, 15, 1))


def relative_rms(left, right):
    if left is None or right is None:
        return None
    return float(np.sqrt(np.mean((left.astype(np.float64) - right) ** 2)) / max(1e-12, np.sqrt(np.mean(left.astype(np.float64) ** 2))))


def channel_amplitudes(audio, frequency, rate=44100):
    if audio is None:
        return None
    segment = audio[int(.8 * rate):int(1.25 * rate)].astype(np.float64)
    phase = np.exp(-2j * np.pi * frequency * np.arange(len(segment)) / rate) * np.hanning(len(segment))
    return np.abs((segment * phase[:, None]).sum(0))


def check_expression(output, *, timeout=30., progress=None):
    scalar = FullScalarCodec()
    base = scalar.schema.template()
    base["settings"].update(FULL_PHASE_CONTROL)
    base["settings"].update(volume=2500., osc_1_level=.15, osc_1_unison_voices=1., env_1_release=.65, pitch_bend_range=2.)
    base["settings"]["modulations"][0] = {"source": "lfo_1", "destination": "osc_1_pan"}
    base["settings"]["modulation_1_amount"] = .02
    note_cases = [(entry.name, entry.performance(), 44100) for entry in initial_bank()]
    note_cases += [(f"sample_boundaries_{rate}", boundary_performance(), rate) for rate in (44100, 48000, 96000)]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    research = Path(__file__).resolve().parents[2]
    paths = [Path(__file__), research / "training/obruxo_training/expression_bank.py", research / "training/obruxo_training/performances.py",
             *(research / "data_generation/obruxo_data" / name for name in ("render/base.py", "render/vita.py", "render/capabilities.py",
               "render/midi_transport.py", "render/full_state.py", "midi/events.py", "midi/validation.py", "vital/full_contract.py"))]
    sources = {path.relative_to(research).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    plan = {"check": "expressive_midi_native_v1", "source_sha256": sources, "renderer_id": FULL_EXPRESSION_RENDERER_ID,
            "comparison_renderer_id": FULL_SEED_RENDERER_ID, "attempt_budget": 93, "worker_jobs": 4, "buffer_size": 128,
            "legacy_rounding_probe": "chord_wide_old_API_repeat_and_one_sample_earlier_SMF_noteoffs",
            "timeout_seconds": timeout, "note_compatibility_cases": [name for name, _, _ in note_cases],
            "expression_cases": [row[0] for row in expression_specs()] + ["release_velocity", "sustain"],
            "minimum_relative_rms_change": .005, "control_response_window_samples": 128, "mpe_unaffected_voice_relative_limit": .03,
            "phase_control": FULL_PHASE_CONTROL, "rss_limit_bytes": 8 * 1024**3, "final_test_audio": False}
    write_json(output / "plan.json", plan)
    events, cases, monitor, started = [], [], MemoryMonitor(), perf_counter()
    monitor.start()
    try:
        with RenderWorker(renderer_factory=NativeRendererFactory(renderer_id=FULL_SEED_RENDERER_ID), max_jobs=4) as old_worker, \
             RenderWorker(renderer_factory=NativeRendererFactory(renderer_id=FULL_EXPRESSION_RENDERER_ID), max_jobs=4) as worker:
            def render(name, role, document, performance, rate=44100, old=False):
                settings = document["settings"]
                projected = scalar.settings(scalar._target(settings))
                settings.update({name: projected[name] for name in scalar.schema.scalar_names})
                request = RenderRequest(preset_json=canonical_json(document), performance=performance, sample_rate=rate,
                                        renderer_id=FULL_SEED_RENDERER_ID if old else FULL_EXPRESSION_RENDERER_ID)
                audio, metadata = (old_worker if old else worker).render(request, timeout=timeout)
                events.append({"case": name, "role": role, "request_id": request.request_id, **metadata})
                write_json(output / "events.json", events)
                return audio if metadata["status"] == "accepted" else None

            def exact(left, right):
                return left is not None and right is not None and np.array_equal(left, right)

            def record(row):
                cases.append(row)
                write_json(output / "checks.json", cases)
                if progress:
                    progress({**row, "attempted": len(events), "budget": plan["attempt_budget"]})

            for name, performance, rate in note_cases:
                old = render(name, "old_notes", deepcopy(base), performance, rate, True)
                current = render(name, "new_notes", deepcopy(base), performance, rate)
                repeated = render(name, "new_repeat", deepcopy(base), performance, rate)
                row = {"case": name, "notes_audio_exact": exact(old, current), "repeat_exact": exact(current, repeated)}
                if name == "chord_wide":
                    shifted = render(name, "legacy_floor_probe", deepcopy(base), legacy_chord_noteoffs(performance))
                    old_repeat = render(name, "old_repeat", deepcopy(base), performance, old=True)
                    row.update(legacy_requested_noteoff_sample=61740, legacy_observed_noteoff_sample=61739,
                               old_repeat_exact=exact(old, old_repeat), one_sample_earlier_matches_old=exact(old, shifted),
                               legacy_float_floor_confirmed=not exact(old, current) and exact(old, shifted) and exact(old, old_repeat))
                row["note_transport_qualified"] = row["notes_audio_exact"] or row.get("legacy_float_floor_confirmed", False)
                record(row)
            specs = [(name, kind, data, source, channel, mpe, held_expression(channel), 480)
                     for name, kind, data, source, channel, mpe in expression_specs()]
            specs.extend((("release_velocity", None, None, "lift", 0, 0, held_expression(), 1440),
                          ("sustain", None, None, None, 0, 0, held_expression(), 1440)))
            for name, kind, data, source, channel, mpe, performance, anchor_tick in specs:
                document = deepcopy(base)
                document["settings"]["mpe_enabled"] = float(mpe)
                if source:
                    document["settings"]["modulations"][63] = {"source": source, "destination": "osc_1_pan"}
                    document["settings"]["modulation_64_amount"] = .25
                changed = performance.clone()
                if name == "release_velocity":
                    changed = held_expression(release=127)
                elif name == "sustain":
                    changed.events.extend((MidiEvent(960, 99, EventKind.CONTROL_CHANGE, 0, (64, 127)),
                                           MidiEvent(1920, 100, EventKind.CONTROL_CHANGE, 0, (64, 0))))
                else:
                    changed.events.append(MidiEvent(480, 99, kind, channel, data))
                original = render(name, "without_expression", deepcopy(document), performance)
                expressed = render(name, "with_expression", deepcopy(document), changed)
                repeated = render(name, "expression_repeat", deepcopy(document), changed)
                delta = relative_rms(original, expressed)
                first = None
                if original is not None and expressed is not None:
                    indices = np.flatnonzero(np.max(np.abs(original - expressed), axis=1) > 1e-7)
                    first = int(indices[0]) if len(indices) else None
                anchor = TempoMap.from_performance(performance).tick_to_sample(anchor_tick, 44100)
                record({"case": name, "repeat_exact": exact(expressed, repeated), "relative_rms_change": delta,
                        "causal_effect": delta is not None and delta > plan["minimum_relative_rms_change"], "first_changed_sample": first,
                        "requested_effect_sample": anchor, "response_within_host_buffer": first is not None and abs(first - anchor) <= 128})
            for name, kind, data, source in (("mpe_slide_isolation", EventKind.CONTROL_CHANGE, (74, 127), "slide"),
                                           ("mpe_pressure_isolation", EventKind.CHANNEL_PRESSURE, (127,), "aftertouch"),
                                           ("mpe_bend_isolation", EventKind.PITCH_BEND, (4096,), None)):
                performance = held_expression(1)
                performance.add_note(pitch=76, velocity=96, start_tick=120, duration_ticks=1320, channel=15)
                changed = performance.clone()
                changed.events.append(MidiEvent(480, 99, kind, 1, data))
                document = deepcopy(base)
                document["settings"]["mpe_enabled"] = 1.
                if source:
                    document["settings"]["modulations"][63] = {"source": source, "destination": "osc_1_pan"}
                    document["settings"]["modulation_64_amount"] = .25
                original = render(name, "without_expression", deepcopy(document), performance)
                expressed = render(name, "with_expression", deepcopy(document), changed)
                repeated = render(name, "expression_repeat", deepcopy(document), changed)
                before, after = channel_amplitudes(original, 440 * 2**(7 / 12)), channel_amplitudes(expressed, 440 * 2**(7 / 12))
                unaffected = float(np.max(np.abs(after - before) / np.maximum(before, 1e-12))) if before is not None and after is not None else None
                delta = relative_rms(original, expressed)
                record({"case": name, "repeat_exact": exact(expressed, repeated), "relative_rms_change": delta,
                        "causal_effect": delta is not None and delta > plan["minimum_relative_rms_change"],
                        "other_voice_relative_amplitude_change": unaffected, "other_voice_preserved": unaffected is not None and unaffected < .03})
            for entry in expression_bank()[8:]:
                document = deepcopy(base)
                document["settings"].update(mpe_enabled=1., osc_2_on=1., osc_3_on=1., osc_1_level=.05, osc_2_level=.05, osc_3_level=.05)
                for slot, source in enumerate(("mod_wheel", "aftertouch", "slide", "lift"), 60):
                    document["settings"]["modulations"][slot] = {"source": source, "destination": f"osc_{slot % 3 + 1}_pan"}
                    document["settings"][f"modulation_{slot + 1}_amount"] = .1
                original = render(entry.name, "source", deepcopy(document), entry.performance())
                repeated = render(entry.name, "repeat", deepcopy(document), entry.performance())
                record({"case": entry.name, "repeat_exact": exact(original, repeated)})
    finally:
        monitor.close()
    gates = {"all_native_accepted": len(events) == plan["attempt_budget"] and all(row["status"] == "accepted" for row in events),
             "note_transport_qualified": all(row.get("note_transport_qualified", True) for row in cases),
             "fixed_phase_repeats": all(row["repeat_exact"] for row in cases),
             "expression_effects": all(row.get("causal_effect", True) for row in cases),
             "native_control_response_window": all(row.get("response_within_host_buffer", True) for row in cases),
             "mpe_voice_isolation": all(row.get("other_voice_preserved", True) for row in cases),
             "source_unchanged": all(hashlib.sha256((research / name).read_bytes()).hexdigest() == digest for name, digest in sources.items()),
             "memory_bounded": monitor.errors == 0 and monitor.peak_bytes is not None and monitor.peak_bytes <= plan["rss_limit_bytes"]}
    summary = {"plan_sha256": canonical_sha256(plan), "passed": all(gates.values()), "gates": gates, "cases": cases,
               "attempted": len(events), "accepted": sum(row["status"] == "accepted" for row in events),
               "elapsed_seconds": perf_counter() - started, "peak_process_rss_bytes": monitor.peak_bytes, "memory_sampling_errors": monitor.errors,
               "final_test_audio": False, "audio_files_written": 0}
    write_json(output / "summary.json", summary)
    return summary

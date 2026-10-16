"""Native routing vocabulary/state acceptance; distinct from learned sound quality."""
from copy import deepcopy
import hashlib
from pathlib import Path
from time import perf_counter

from obruxo_data.hashing import canonical_json, canonical_sha256
from obruxo_data.render.base import FULL_PRESET_RENDERER_ID, RenderRequest

from .audit import write_json
from .benchmark import MemoryMonitor
from .full_scalar import FullScalarCodec
from .lfo_bank import curve_to_lfo
from .performances import initial_bank
from .reproducibility import FULL_PHASE_CONTROL
from .stream import NativeRendererFactory, RenderWorker


def routing_cases(*, probe=False):
    scalar, cases = FullScalarCodec(), []
    schema = scalar.schema
    base = schema.template()
    base["settings"].update(FULL_PHASE_CONTROL)
    base["settings"].update(volume=3844., osc_1_level=.1, lfo_1_frequency=0., lfo_1_sync=0.)

    def add(name, edits, repeats=1):
        document = deepcopy(base)
        settings = document["settings"]
        edits(settings)
        projected = scalar.settings(scalar._target(settings))
        settings.update({name: projected[name] for name in schema.scalar_names})
        schema.validate(document).require_valid()
        cases.append((name, document, repeats))

    def connection(settings, slot, source="lfo_1", destination="osc_1_pan", amount=.001, mapped=False):
        route = {"source": source, "destination": destination}
        if mapped:
            route["line_mapping"] = curve_to_lfo([1. - index / 96 for index in range(97)], name="Native routing map")
        settings["modulations"][slot - 1] = route
        settings[f"modulation_{slot}_amount"] = amount

    def empty_map(settings):
        connection(settings, 64)
        settings["modulations"][0]["line_mapping"] = curve_to_lfo([1. - index / 96 for index in range(97)])
    add("empty_map", empty_map, 2)
    add("self_amount", lambda settings: (connection(settings, 1, destination="modulation_1_amount"), connection(settings, 64)), 2)

    def cycle(settings):
        connection(settings, 1, "lfo_1", "lfo_2_frequency", .01)
        connection(settings, 32, "lfo_2", "lfo_1_frequency", .01)
        connection(settings, 64, "lfo_1", "osc_1_pan", .02, True)
    add("modulator_cycle", cycle, 2)

    def amount_cycle(settings):
        connection(settings, 1, "lfo_1", "modulation_64_amount", .01)
        connection(settings, 64, "lfo_2", "modulation_1_amount", .01)
        connection(settings, 32, "lfo_3", "osc_1_pan", .02, True)
    add("amount_cycle", amount_cycle, 2)
    if probe:
        return cases
    for source in sorted(schema.sources):
        def edit(settings, source=source):
            connection(settings, 1, source)
            if source in schema.scalar_names:
                settings[source] = .5
        add("source_" + source, edit)
    for destination in sorted(schema.destinations):
        add("destination_" + destination, lambda settings, destination=destination: connection(settings, 64, destination=destination))
    for slot in range(1, 65):
        def edit(settings, slot=slot):
            connection(settings, slot, amount=.01, mapped=True)
            settings.update({f"modulation_{slot}_power": .2, f"modulation_{slot}_bipolar": 1., f"modulation_{slot}_stereo": 1.})
        add(f"slot_{slot}", edit, 2)

    def dense(settings):
        for slot in range(1, 65):
            connection(settings, slot, f"lfo_{(slot - 1) % 8 + 1}", amount=.001, mapped=slot % 8 == 0)
            settings[f"modulation_{slot}_bypass"] = float(slot % 5 == 0)
    add("dense_64", dense, 2)
    return cases


def check_routing(output, *, probe=False, timeout=30., progress=None):
    cases = routing_cases(probe=probe)
    research = Path(__file__).resolve().parents[2]
    files = [Path(__file__), research / "training/obruxo_training/routing.py", research / "training/obruxo_training/stream.py",
             research / "data_generation/obruxo_data/render/vita.py", research / "data_generation/obruxo_data/render/full_state.py"]
    sources = {path.relative_to(research).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    requests = [(name, RenderRequest(preset_json=canonical_json(document), performance=initial_bank()[0].performance(), renderer_id=FULL_PRESET_RENDERER_ID), repeats)
                for name, document, repeats in cases]
    plan = {"check": "ordered_routing_native_probe_v1" if probe else "ordered_routing_native_v1", "renderer_id": FULL_PRESET_RENDERER_ID,
            "source_sha256": sources, "case_request_ids": {name: request.request_id for name, request, _ in requests},
            "native_attempt_budget": sum(repeats for _, _, repeats in requests), "timeout_seconds": timeout, "worker_jobs": 4,
            "phase_control": FULL_PHASE_CONTROL, "rss_limit_bytes": 8 * 1024**3, "final_test_audio": False,
            "vocabulary_counts": {"sources": 32, "destinations": 431, "numbered_slots": 64},
            "expected_rejections": {"empty_map": "Vital did not load the requested modulations"},
            "claim": "state_acceptance_and_fixed_phase_repeats_not_all_MIDI_excitation_or_destination_effectiveness"}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", plan)
    monitor, started, events, checks = MemoryMonitor(), perf_counter(), [], []
    monitor.start()
    try:
        with RenderWorker(renderer_factory=NativeRendererFactory(renderer_id=FULL_PRESET_RENDERER_ID), max_jobs=4) as worker:
            for name, request, repeats in requests:
                group = []
                for repeat in range(repeats):
                    _, metadata = worker.render(request, timeout=timeout)
                    row = {"case": name, "repeat": repeat, "request_id": request.request_id, **metadata}
                    events.append(row)
                    group.append(row)
                    write_json(output / "events.json", events)
                accepted = all(row["status"] == "accepted" for row in group)
                expected = plan["expected_rejections"].get(name)
                rejected = bool(expected) and all(row["status"] == "error" and row.get("message") == expected for row in group)
                exact = accepted and repeats > 1 and len({row["qa"]["audio_float32_sha256"] for row in group}) == 1
                checks.append({"case": name, "attempted": repeats, "accepted": accepted, "expected_rejection": rejected,
                               "passed": rejected if expected else accepted and (repeats == 1 or exact), "fixed_phase_repeat_exact": exact})
                write_json(output / "checks.json", checks)
                if progress:
                    progress({"case": name, "accepted": accepted, "done": len(checks), "total": len(cases)})
    finally:
        monitor.close()
    unchanged = all(hashlib.sha256((research / name).read_bytes()).hexdigest() == digest for name, digest in sources.items())
    gates = {"native_state_contract": len(events) == plan["native_attempt_budget"] and all(row["passed"] for row in checks),
             "repeats_exact": all(row["fixed_phase_repeat_exact"] for row in checks if row["attempted"] > 1 and not row["expected_rejection"]), "source_unchanged": unchanged,
             "memory_bounded": monitor.errors == 0 and monitor.peak_bytes is not None and monitor.peak_bytes <= plan["rss_limit_bytes"]}
    summary = {"plan_sha256": canonical_sha256(plan), "passed": all(gates.values()), "gates": gates,
               "attempted": len(events), "accepted": sum(row["status"] == "accepted" for row in events), "checks": checks,
               "expected_rejected": sum(row["attempted"] for row in checks if row["expected_rejection"]),
               "elapsed_seconds": perf_counter() - started, "peak_process_rss_bytes": monitor.peak_bytes, "memory_sampling_errors": monitor.errors,
               "final_test_audio": False, "audio_files_written": 0}
    write_json(output / "summary.json", summary)
    return summary

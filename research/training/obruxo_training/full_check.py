"""Bounded native qualification of the broad preset profile, before broad learning."""
from copy import deepcopy
import base64
import hashlib
from pathlib import Path
import platform
from time import perf_counter

import numpy as np

from obruxo_data.hashing import canonical_json, canonical_sha256
from obruxo_data.render.base import FULL_PRESET_RENDERER_ID, RenderRequest
from obruxo_data.vital.full_contract import FullPresetContract

from .audit import write_json
from .benchmark import MemoryMonitor
from .performances import initial_bank
from .reproducibility import OSCILLATOR_PHASE_CONTROL
from .stream import NativeRendererFactory, RenderWorker, StreamConfig, render_isolated


def full_check_cases():
    """Independent authored diagnostics; no dataset, learned asset bank or test reserve."""
    contract = FullPresetContract()
    base = contract.template()
    base.update(author="OBRUXO", comments="Full-profile authored correctness fixture", preset_name="OBRUXO full check")
    base["settings"].update(OSCILLATOR_PHASE_CONTROL)
    base["settings"].update(osc_1_level=.25, sample_random_phase=0.)
    cases = []

    def add(name, controls=None, edit=None):
        document = deepcopy(base)
        document["settings"].update(controls or {})
        if edit:
            edit(document["settings"])
        contract.validate(document).require_valid()
        cases.append((name, document))

    add("factory")
    add("third_oscillator_and_new_ordinals", {"osc_1_level": .12, "osc_3_on": 1., "osc_3_level": .12,
                                            "osc_3_transpose": -12., "osc_1_stack_style": 12., "osc_1_spectral_morph_type": 16.})

    def curves(settings):
        settings["lfos"][0].update(num_points=3, points=[0., 1., .5, 0., 1., 1.], powers=[.25, -.25, 0.])
    add("lfo_curve", {"lfo_1_sync_type": 6.}, curves)
    def warp(settings):
        settings["custom_warps"][0]["powers"][0] = .25
    add("custom_warp", edit=warp)

    def routes(settings):
        settings["modulations"][0] = {"source": "lfo_1", "destination": "filter_1_cutoff",
                                      "line_mapping": deepcopy(settings["lfos"][0])}
        settings["modulations"][63] = {"source": "lfo_2", "destination": "modulation_1_amount"}
    add("indexed_routes_and_remap", {"filter_1_on": 1., "modulation_1_amount": .1, "modulation_64_amount": .2}, routes)
    add("sampler_mono", {"sample_on": 1., "osc_1_on": 0., "sample_level": .25, "sample_loop": 0.})

    def stereo(settings):
        sample = settings["sample"]
        samples = np.frombuffer(base64.b64decode(sample["samples"]), dtype="<i2").astype(np.int32)
        sample["samples_stereo"] = base64.b64encode(np.clip(-samples, -32768, 32767).astype("<i2").tobytes()).decode()
    add("sampler_stereo", {"sample_on": 1., "osc_1_on": 0., "sample_level": .25, "sample_loop": 0.}, stereo)
    add("voice_and_portamento", {"voice_override": 1., "voice_priority": 1., "legato": 1., "portamento_time": -4.})
    effects = {"chorus_on": 1., "chorus_dry_wet": .2, "distortion_on": 1., "distortion_mix": .2}
    add("effect_order_1", {**effects, "effect_chain_order": 1.})
    add("effect_order_2", {**effects, "effect_chain_order": 2.})

    def seeds(settings):
        for index, value in enumerate(settings["random_values"]):
            value["seed"] = 17 + index
    add("random_seeds", edit=seeds)

    def shepard(settings):
        settings["wavetables"][0]["groups"][0]["components"][0]["type"] = "Shepard Tone Source"
    add("shepard_component", edit=shepard)
    return cases


def check_full_render(output, *, plugin_path=None, timeout=30., progress=None):
    StreamConfig(timeout_seconds=timeout)
    cases = full_check_cases()
    requests = [RenderRequest(preset_json=canonical_json(document), performance=initial_bank()[0].performance(),
                              renderer_id=FULL_PRESET_RENDERER_ID) for _, document in cases]
    research = Path(__file__).resolve().parents[2]
    sources = ["data_generation/obruxo_data/render/base.py", "data_generation/obruxo_data/render/vita.py",
               "data_generation/obruxo_data/render/full_state.py", "data_generation/obruxo_data/vital/full_contract.py",
               "training/obruxo_training/full_check.py", "training/obruxo_training/stream.py"]
    hashes = {name: hashlib.sha256((research / name).read_bytes()).hexdigest() for name in sources}
    contract = FullPresetContract()
    plan = {"qualification": "full_1_5_5_native_profile_v1", "renderer_id": FULL_PRESET_RENDERER_ID,
            "full_contract_sha256": canonical_sha256(contract.spec), "source_sha256": hashes,
            "cases": [{"name": name, "request_id": request.request_id} for (name, _), request in zip(cases, requests, strict=True)],
            "attempt_budget": 3 * len(cases), "timeout_seconds": timeout, "worker_max_jobs": 4,
            "rss_limit_bytes": 8 * 1024**3,
            "phase_control": OSCILLATOR_PHASE_CONTROL, "sample_random_phase": 0., "final_test_used": False,
            "purpose": "authored correctness fixtures; no training or reconstruction-quality claim",
            "reuse_order": list(range(len(cases))) + list(reversed(range(len(cases)))), "audio_archive": False}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", plan)
    factory, monitor = NativeRendererFactory(plugin_path, FULL_PRESET_RENDERER_ID), MemoryMonitor()
    events = []
    monitor.start()
    started = perf_counter()

    def record(case, mode, metadata):
        row = {"case": cases[case][0], "mode": mode, "request_id": requests[case].request_id, **metadata}
        events.append(row)
        write_json(output / "events.json", events)
        if progress:
            progress({"case": row["case"], "mode": mode, "status": metadata["status"]})

    try:
        for index, request in enumerate(requests):
            audio, metadata = render_isolated(request, renderer_factory=factory, timeout=timeout)
            record(index, "fresh", metadata)
        with RenderWorker(renderer_factory=factory, max_jobs=4) as worker:
            for index in plan["reuse_order"]:
                audio, metadata = worker.render(requests[index], timeout=timeout)
                record(index, "reuse", metadata)
    finally:
        monitor.close()
    checks = []
    for name, _ in cases:
        rows = [row for row in events if row["case"] == name]
        accepted = len(rows) == 3 and all(row["status"] == "accepted" for row in rows)
        checks.append({"case": name, "accepted": accepted,
                       "fixed_phase_audio_exact": accepted and len({row["qa"]["audio_float32_sha256"] for row in rows}) == 1,
                       "state_exact": accepted and all(row["renderer_qa"]["native_state"] == rows[0]["renderer_qa"]["native_state"] for row in rows),
                       "provenance_exact": accepted and all(row["provenance"] == rows[0]["provenance"] for row in rows)})
    source_unchanged = hashes == {name: hashlib.sha256((research / name).read_bytes()).hexdigest() for name in sources}
    reused = [row for row in events if row["mode"] == "reuse"]
    jobs_by_pid = {}
    for row in reused:
        jobs_by_pid.setdefault(row.get("worker_pid"), []).append(row.get("worker_job"))
    gates = {"source_unchanged": source_unchanged,
             "renderer_identity_constant": len({canonical_sha256({key: value for key, value in row.get("provenance", {}).items() if key != "request_id"}) for row in events}) == 1,
             "all_cases_exact": all(all(row[key] for key in ("accepted", "fixed_phase_audio_exact", "state_exact", "provenance_exact")) for row in checks),
             "all_775_scalars_verified": all(row.get("renderer_qa", {}).get("native_state", {}).get("verified_loaded_scalars") == 775 for row in events),
             "worker_lifetime_bounded": len(jobs_by_pid) == 6 and all(jobs == [1, 2, 3, 4] for jobs in jobs_by_pid.values()),
             "memory_bounded": monitor.errors == 0 and monitor.peak_bytes is not None and monitor.peak_bytes <= plan["rss_limit_bytes"]}
    qualified = all(gates.values())
    summary = {"plan_sha256": canonical_sha256(plan), "qualified": qualified, "source_unchanged": source_unchanged,
               "attempts": len(events), "accepted": sum(row["status"] == "accepted" for row in events), "checks": checks,
               "peak_process_rss_bytes": monitor.peak_bytes, "memory_sampling_errors": monitor.errors,
               "elapsed_seconds": perf_counter() - started, "platform": platform.platform(), "gates": gates, "final_test_used": False}
    write_json(output / "summary.json", summary)
    return summary

"""Measure proposed protected updates with fresh, phase-fixed Vital audio."""

from pathlib import Path
import shutil
from time import perf_counter, time
from types import SimpleNamespace

import torch

from obruxo_data.hashing import canonical_sha256, sha256_bytes
from obruxo_data.midi import Performance, TempoMap

from .audit import write_json
from .basic_direct import REHEARSAL
from .basic_direct_fit import SoundStore, lines
from .benchmark import MemoryMonitor
from .category_scale_refresh import cpu_settings
from .codec import PilotCodec, PilotTarget
from .conditioned_bridge import read
from .native_fit import append_durable, digest
from .paired_native_evidence import RetainedAuditSession, native_ledger, rescore_waveform
from .paired_native_fit import PairedSession
from .protected_update_check import make_plan as screen_plan
from .unison_fit_data import require_request
from .unison_sound_fit import CostStore
from .update_diagnosis import movement

STEPS = list(range(4, 49, 4))
LIMITS = {"wall_seconds": 900, "process_tree_bytes": 4 * 1024**3, "archive_bytes": 512 * 1024**2,
          "qualification_attempts": 75, "training_attempts": 0}
GATES = {"minimum_protected_points": 9, "minimum_weighted_actual_gain_fraction": .005, "sound_numerical_tolerance": 1e-6,
         "exact_phase_fixed_repeats": 2}


def build_cases(inputs, source, screen, steps):
    saved = torch.load(source / "checkpoint.pt", map_location="cpu", weights_only=True)
    store, jobs = SoundStore(source, inputs), []
    for number in steps:
        recorded, selected = saved["trace"][number - 1], screen["rows"][number - 1]
        if recorded["step"] != number or selected["step"] != number:
            raise ValueError("native check proposal order changed")
        for observation, cells in enumerate(inputs.observations(recorded["update"])):
            estimate, change = recorded["estimates"][observation], selected["changes"][observation]
            before = estimate["anchor"]
            after = {"continuous": [a + d for a, d in zip(before["continuous"], change["continuous_delta"], strict=True)], "categorical": change["after_choices"]}
            if movement(before, after, estimate, inputs.codec) != change:
                raise ValueError("native candidate differs from its screened output movement")
            candidate = inputs.codec.canonical_target(PilotTarget.from_dict(after))
            anchor = inputs.codec.canonical_target(PilotTarget.from_dict(before))
            costs = []
            for index in cells:
                query = {"step": number, "observation": observation, "index": index, "candidate": anchor.to_dict()}
                probe = store.probes[canonical_sha256(query)]
                cell = inputs.plan["fit_audit_cells"][index]
                performance = cell["recipe"]["performances"][cell["playback_view"]]
                require_request(inputs.codec, inputs.plan, anchor, Performance.from_dict(performance), probe["receipt"])
                baseline = inputs.cases[index]["rows"]["initial"]["scores"]["spectral"]
                jobs.append({"label": f"step{number:03d}:observation{observation}:cell{index:03d}:after", "step": number,
                             "observation": observation, "index": index, "candidate": candidate.to_dict(), "before_sound": probe["loss"],
                             "accepted_initial_sound": baseline, "proposal_nonzero": selected["accepted"],
                             "performance": performance,
                             "before_receipt": probe["receipt"]})
                costs.append(probe["loss"])
            if sum(costs) / len(costs) != estimate["value"]:
                raise ValueError("native check before costs differ from the measured gradient anchor")
    return jobs


def make_plan(sources, parent, bridge, features, diagnosis, screen):
    checked = screen_plan(sources, parent, bridge, features, diagnosis)
    result = read(screen / "summary.json")
    if (read(screen / "plan.json") != checked or result["plan_id"] != canonical_sha256(checked) or not result["useful_local_proposals"]
            or not result["within_resource_limits"] or not result["model_adam_rng_replay_exact"] or result["temporary_protected_proposals"] != 48):
        raise ValueError("native check requires the unchanged passing protected-update screen")
    from .basic_direct_fit_data import Inputs
    # Source-only loading: no model forward pass or optimizer application.
    inputs = Inputs.__new__(Inputs)
    inputs.plan, inputs.codec, inputs.cases = read(sources[0] / "plan.json"), PilotCodec(), read(parent / "fit_cases.json")
    inputs.lookup = {(cell["recipe"]["state_id"], cell["input_view"], cell["playback_view"]): i for i, cell in enumerate(inputs.plan["fit_audit_cells"])}
    inputs.references = [inputs.lookup[(row["state_id"], 0, 0)] for row in inputs.plan["reference_manifest"]]
    jobs = build_cases(inputs, sources[0], result, STEPS)
    jobs += [{**jobs[index], "label": label, "repeat_of": jobs[index]["label"]}
             for index, label in ((0, "repeat_first:after"), (-1, "repeat_last:after"))]
    bound = sum(8 * TempoMap.from_performance(Performance.from_dict(job["performance"])).render_frame_count(
        Performance.from_dict(job["performance"]).end_tick, inputs.codec.contract.tail_seconds, 44100) + 16384 for job in jobs) + 16 * 1024**2
    if len(jobs) != 74 or bound > LIMITS["archive_bytes"]:
        raise ValueError("native protection panel exceeds its declared render or evidence allocation")
    root, code = Path(__file__).parents[1], dict(checked["source"])
    for name in ("protected_native_check.py", "temporal_proxy_data.py", "stream.py", "reports/scripts/check_protected_native.py",
                 "../data_generation/obruxo_data/render/vita.py", "../data_generation/obruxo_data/render/base.py", "../data_generation/obruxo_data/render/qa.py"):
        path = root / name if "/" in name else Path(__file__).with_name(name)
        code[name] = sha256_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
    return {"format": "protected_update_native_acceptance_screen_v1", "screen_plan_id": canonical_sha256(checked),
            "screen_summary_sha256": digest(screen / "summary.json"), "source": code, "source_runs": checked["runs"],
            "steps": STEPS, "jobs": jobs, "limits": LIMITS, "gates": GATES, "archive_upper_bound_bytes": bound,
            **{key: inputs.plan[key] for key in ("render_profile", "render_phase_controls", "expected_renderer_identity")},
            "codec_id": inputs.codec.codec_id, "source_plan_sha256": digest(sources[0] / "plan.json"),
            "optimizer_applications": 0, "experimental_training_updates": 0, "test_split_used": False, "qualified_for_release": False}


def decisions(plan, rows):
    actual = {row["label"]: row for row in rows}
    if len(actual) != len(rows) or set(actual) != {job["label"] for job in plan["jobs"]}:
        raise ValueError("native protection coverage is incomplete or duplicated")
    repeats, points = [], []
    for job in plan["jobs"]:
        if "repeat_of" in job:
            same = actual[job["label"]]["receipt"]["qa"]["audio_float32_sha256"] == actual[job["repeat_of"]]["receipt"]["qa"]["audio_float32_sha256"]
            repeats.append(same)
    for number in plan["steps"]:
        observations = []
        for observation in range(4):
            jobs = [job for job in plan["jobs"] if job["step"] == number and job["observation"] == observation and "repeat_of" not in job]
            before = sum(job["before_sound"] for job in jobs) / len(jobs)
            after = sum(actual[job["label"]]["scores"]["spectral"] for job in jobs) / len(jobs)
            observations.append({"before": before, "after": after, "change": after - before, "initial": jobs[0]["accepted_initial_sound"]})
        weights = REHEARSAL["paired_weights"] + REHEARSAL["reference_weights"]
        objective = {when: sum(weight * (row[when] + (max(0., row[when] - row["initial"]) if i > 1 else 0.))
                               for i, (weight, row) in enumerate(zip(weights, observations, strict=True))) for when in ("before", "after")}
        protected = all(row["change"] <= plan["gates"]["sound_numerical_tolerance"] for row in observations)
        useful = objective["after"] < objective["before"] - plan["gates"]["sound_numerical_tolerance"]
        nonzero = all(job["proposal_nonzero"] for job in plan["jobs"] if job["step"] == number)
        points.append({"step": number, "observations": observations, "objective": objective, "all_sounds_protected": protected,
                       "nonzero_proposal": nonzero, "accepted": bool(protected and useful and nonzero)})
    before = sum(row["objective"]["before"] for row in points)
    after = sum(row["objective"]["after"] for row in points)
    gain = (before - after) / before if before else 0.
    accepted = sum(row["accepted"] for row in points)
    qualified = accepted >= plan["gates"]["minimum_protected_points"] and gain >= plan["gates"]["minimum_weighted_actual_gain_fraction"] and len(repeats) == 2 and all(repeats)
    return {"points": points, "accepted_points": accepted, "weighted_actual_gain_fraction": gain, "repeat_exact": repeats,
            "native_acceptance_screen_passed": bool(qualified), "qualified_for_release": False, "test_split_used": False}


def source_store(source):
    return CostStore(source, SimpleNamespace(plan=read(source / "plan.json")))


def rescore_rows(plan, source, output, rows, *, prefix=False):
    jobs = plan["jobs"][:len(rows)] if prefix else plan["jobs"]
    attempts, outcomes, recovered = native_ledger(lines(output / "renders.jsonl"), plan["expected_renderer_identity"])
    if len(outcomes) != len(rows) + len(recovered) or len(attempts) > plan["limits"]["qualification_attempts"]:
        raise ValueError("native protection attempts differ from retained results")
    codec, store = PilotCodec(), source_store(source)
    for job, row in zip(jobs, rows, strict=True):
        receipt = row["receipt"]
        if (row["label"] != job["label"] or receipt["label"] != job["label"] or receipt["phase"] != "qualification"
                or any(outcomes[(receipt["phase"], receipt["attempt"])].get(key) != value for key, value in receipt.items() if key != "retained_audio")):
            raise ValueError("native protection receipt differs from its charged ledger")
        require_request(codec, plan, PilotTarget.from_dict(job["candidate"]), Performance.from_dict(job["performance"]), receipt)
        reference, _metadata = store.source(job["index"])
        if rescore_waveform(output, receipt, reference, codec.contract.tail_seconds) != row["scores"]:
            raise ValueError("native protection sound score cannot be reproduced")
    return attempts, recovered


def continuation_contract(previous, plan, source):
    old, summary, run = read(previous / "plan.json"), read(previous / "summary.json"), read(previous / "run.json")
    rows = lines(previous / "measurements.jsonl")
    shared = ("steps", "jobs", "gates", "render_profile", "render_phase_controls", "expected_renderer_identity")
    differences = {name: {"historical": value, "current": plan["source"].get(name)} for name, value in old["source"].items() if plan["source"].get(name) != value}
    if (summary["complete"] or summary["plan_id"] != canonical_sha256(old) or run["plan_id"] != canonical_sha256(old)
            or summary["failure"] != {"type": "RuntimeError", "message": "helper wall-time budget exhausted"}
            or not 0 < len(rows) < len(old["jobs"]) or any(old[key] != plan[key] for key in shared)
            or time() - run["started_unix"] < old["limits"]["wall_seconds"]):
        raise ValueError("only the unchanged terminal wall-limited native prefix may continue")
    if set(differences) - {"protected_native_check.py", "reports/scripts/check_protected_native.py"}:
        raise ValueError("native continuation changed numerical source beyond its recovery adapter")
    rescore_rows(old, source, previous, rows, prefix=True)
    manifest = {path.relative_to(previous).as_posix(): digest(path) for path in previous.rglob("*") if path.is_file()}
    return {"source_plan_id": canonical_sha256(old), "source_evidence_sha256": manifest, "original_started_unix": run["started_unix"],
            "accepted_results_reused": len(rows), "original_15_minute_budget_passed": False,
            "aggregate_wall_seconds_cap": 1800, "source_archive_bytes": sum(path.stat().st_size for path in previous.rglob("*") if path.is_file()),
            "source_sha256_differences": differences}


def require_prefix(plan, previous):
    contract = plan["operational_continuation"]
    if any(digest(previous / name) != value for name, value in contract["source_evidence_sha256"].items()):
        raise ValueError("immutable native prefix changed")


def migrate_prefix(plan, previous, output):
    require_prefix(plan, previous)
    shutil.copyfile(previous / "renders.jsonl", output / "renders.jsonl")
    shutil.copyfile(previous / "measurements.jsonl", output / "reused_measurements.jsonl")
    (output / "comparison_audio").mkdir()
    for path in (previous / "comparison_audio").glob("*.pt"):
        shutil.copyfile(path, output / "comparison_audio" / path.name)


def execute(plan, source, output):
    if (output / "run.json").exists():
        raise ValueError("preserve attempted native checks; no silent second execution")
    original_start = plan.get("operational_continuation", {}).get("original_started_unix", time())
    write_json(output / "run.json", {"plan_id": canonical_sha256(plan), "started_unix": original_start})
    started, monitor = perf_counter(), MemoryMonitor()
    monitor.start()
    codec, store, rows, failure = PilotCodec(), source_store(source), [], None
    session = RetainedAuditSession(PairedSession(output, plan, codec, monitor), output)
    reused = {row["label"]: row for row in lines(output / "reused_measurements.jsonl")}
    try:
        for job in plan["jobs"]:
            if job["label"] in reused:
                row = reused[job["label"]]
                append_durable(output / "measurements.jsonl", row)
                rows.append(row)
                continue
            candidate, performance = PilotTarget.from_dict(job["candidate"]), Performance.from_dict(job["performance"])
            audio, receipt = session.render(candidate, performance, "qualification", job["label"])
            failures = sum(event["event"] == "outcome" and event["status"] != "accepted" for event in session.events)
            if audio is None and receipt.get("error_type") == "RenderTimeout" and "qa" not in receipt and failures == 1:
                audio, receipt = session.render(candidate, performance, "qualification", job["label"])
            if audio is None:
                raise RuntimeError("native protected proposal was not accepted")
            require_request(codec, plan, candidate, performance, receipt)
            reference, _metadata = store.source(job["index"])
            row = {"label": job["label"], "receipt": receipt, "scores": rescore_waveform(output, receipt, reference, codec.contract.tail_seconds)}
            append_durable(output / "measurements.jsonl", row)
            rows.append(row)
            if sum(path.stat().st_size for path in output.rglob("*") if path.is_file()) + 1024**2 > plan["limits"]["archive_bytes"]:
                raise RuntimeError("native protection archive limit exceeded")
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}
    finally:
        monitor.close()
    resources = {"elapsed_seconds": time() - original_start, "current_execution_seconds": perf_counter() - started,
                 "peak_process_tree_rss_bytes": monitor.peak_bytes, "rss_sampling_errors": monitor.errors}
    within = bool(resources["elapsed_seconds"] < plan["limits"]["wall_seconds"] and monitor.peak_bytes and monitor.peak_bytes <= plan["limits"]["process_tree_bytes"] and not monitor.errors)
    result = {"plan_id": canonical_sha256(plan), "complete": len(rows) == len(plan["jobs"]) and failure is None,
              "failure": failure, "resources": resources, "within_resource_limits": within,
              "result": decisions(plan, rows) if failure is None else None,
              "native_attempts": sum(row["event"] == "attempt" for row in session.events),
              "experimental_training_updates": 0, "optimizer_applications": 0, "qualified_for_release": False, "test_split_used": False}
    write_json(output / "summary.json", result)
    if failure or not within:
        raise RuntimeError(f"native protection check incomplete: {failure or 'resource limit'}")
    return result


def verify(plan, source, output):
    summary, rows = read(output / "summary.json"), lines(output / "measurements.jsonl")
    if not summary["complete"] or summary["failure"] or not summary["within_resource_limits"] or summary["plan_id"] != canonical_sha256(plan):
        raise ValueError("native protection check is incomplete or changed")
    attempts, recovered = rescore_rows(plan, source, output, rows)
    result = decisions(plan, rows)
    if result != summary["result"]:
        raise ValueError("native protection decisions cannot be reproduced")
    evidence = {path.relative_to(output).as_posix(): digest(path) for path in output.rglob("*") if path.is_file() and path.name != "verification.json"}
    proof = {"plan_id": canonical_sha256(plan), "verified": True, "comparison_waveforms_rescored": len(rows), "native_attempts": len(attempts),
             "recovered_timeouts": recovered, "result": result, "evidence_sha256": evidence,
             "optimizer_applications": 0, "experimental_training_updates": 0, "qualified_for_release": False, "test_split_used": False}
    write_json(output / "verification.json", proof)
    return proof


def run(sources, parent, bridge, features, diagnosis, screen, output, *, mode="prepare", continue_from=None):
    with cpu_settings():
        plan = make_plan(sources, parent, bridge, features, diagnosis, screen)
        if continue_from is not None:
            plan["operational_continuation"] = continuation_contract(continue_from, plan, sources[0])
            plan["limits"] = {**plan["limits"], "wall_seconds": 1800}
        if mode == "prepare":
            output.mkdir(parents=True, exist_ok=False)
            write_json(output / "plan.json", plan)
            return {"plan_id": canonical_sha256(plan), "planned_native_requests": len(plan["jobs"]), "archive_upper_bound_bytes": plan["archive_upper_bound_bytes"]}
        if read(output / "plan.json") != plan:
            raise ValueError("native protection plan changed")
        if mode not in ("execute", "verify"):
            raise ValueError("expected prepare, execute or verify")
        if continue_from is not None:
            require_prefix(plan, continue_from)
            if mode == "execute":
                if any((output / name).exists() for name in ("run.json", "renders.jsonl", "reused_measurements.jsonl")):
                    raise ValueError("preserve attempted native continuations; no second migration")
                migrate_prefix(plan, continue_from, output)
        result = execute(plan, sources[0], output) if mode == "execute" else verify(plan, sources[0], output)
        rebuilt = make_plan(sources, parent, bridge, features, diagnosis, screen)
        if continue_from is not None:
            rebuilt["operational_continuation"] = continuation_contract(continue_from, rebuilt, sources[0])
            rebuilt["limits"] = {**rebuilt["limits"], "wall_seconds": 1800}
            if (not (output / "renders.jsonl").read_bytes().startswith((continue_from / "renders.jsonl").read_bytes())
                    or lines(output / "measurements.jsonl")[:plan["operational_continuation"]["accepted_results_reused"]] != lines(continue_from / "measurements.jsonl")):
                raise ValueError("native continuation changed its accepted prefix")
            aggregate = plan["operational_continuation"]["source_archive_bytes"] + sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
            if aggregate > plan["limits"]["archive_bytes"]:
                raise ValueError("aggregate native continuation archive limit failed")
        if rebuilt != plan:
            raise ValueError("native protection source evidence changed")
        return {key: value for key, value in result.items() if key not in ("evidence_sha256",)}

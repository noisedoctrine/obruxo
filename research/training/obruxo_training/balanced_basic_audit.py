"""Frozen quality gates and offline verification for balanced basic sound training."""

from time import time

import numpy as np
import torch
from obruxo_data.hashing import canonical_sha256
from obruxo_data.midi import Performance

from .audit import write_json
from .balanced_basic_fit import LIMITS, SPEC, BalancedStore, train
from .basic_direct import restore_direct
from .benchmark import MemoryMonitor
from .codec import PilotTarget
from .conditioned_bridge import read
from .features import tensor_digest
from .inward_basic import export_inward
from .native_fit import digest
from .paired_native_audit import equal_tree
from .paired_native_evidence import native_ledger, WaveformAudit
from .batching import BatchConfig, predict_cells
from .unison_fit_data import require_request
from .unison_sound_fit import lines

ARMS = ("initial", "control", "balanced")
METRICS = ("spectral", "mrstft")


def results(inputs, store):
    cells, plan = inputs.plan["fit_audit_cells"], inputs.plan

    def score(index, arm):
        return (store.initial(index) if arm == "initial" else store.finals[index, arm])["scores"]

    def means(indices):
        return {arm: {metric: float(np.mean([score(index, arm)[metric] for index in indices])) for metric in METRICS} for arm in ARMS}

    report = {"three_sounds": {}, "six_groups": {}, "grid": {}}
    for state, midi in zip(SPEC["training_indices"], (36, 61, 83), strict=True):
        indices = [i for i in plan["selected_comparisons"] if cells[i]["training_index"] == state]
        summary = means(indices)
        every = all(score(i, "balanced")["spectral"] <= .9 * score(i, "initial")["spectral"]
                    and score(i, "balanced")["mrstft"] < score(i, "initial")["mrstft"] for i in indices)
        held = [i for i in indices if cells[i]["input_view"] == 1]
        beats_parameter = all(score(i, "balanced")["spectral"] <= inputs.parameter_finals[i, "parameter"]["scores"]["spectral"] for i in held)
        report["three_sounds"][str(midi)] = {"means": summary, "comparisons": len(indices),
            "every_comparison_improves": bool(every), "held_input_beats_parameter_control": bool(beats_parameter),
            "learning_prerequisite_passed": bool(every and beats_parameter and summary["balanced"]["spectral"] <= .75 * summary["initial"]["spectral"])}
        indices = [i for i in plan["grid_comparisons"] if cells[i]["training_index"] == state]
        grid = means(indices)
        report["grid"][str(midi)] = {"means": grid, "comparisons": len(indices), "pass": bool(
            grid["balanced"]["spectral"] <= .9 * grid["initial"]["spectral"]
            and grid["balanced"]["spectral"] <= grid["control"]["spectral"] and grid["balanced"]["mrstft"] < grid["initial"]["mrstft"])}
    for view, role in enumerate(plan["training_view_roles"]):
        for transfer in (False, True):
            indices = [i for i, cell in enumerate(cells[:104]) if cell["training_index"] is not None and cell["input_view"] == view
                       and (cell["input_view"] != cell["playback_view"]) == transfer]
            summary = means(indices)
            parameter = {metric: float(np.mean([inputs.parameter_finals[i, "parameter"]["scores"][metric] for i in indices])) for metric in METRICS}
            report["six_groups"][f"{role}_{'transfer' if transfer else 'own'}"] = {"means": summary, "saved_parameter_control": parameter,
                "pass": bool(summary["balanced"]["spectral"] <= .9 * summary["initial"]["spectral"]
                             and summary["balanced"]["spectral"] <= parameter["spectral"]
                             and summary["balanced"]["mrstft"] < summary["initial"]["mrstft"])}
    indices = [i for i, cell in enumerate(cells[:104]) if cell["input_view"] == cell["playback_view"] == 0]
    if len(indices) != 32 or {cells[i]["recipe"]["state_id"] for i in indices} != set(plan["regression_state_ids"]):
        raise ValueError("balanced fit changed the original reference panel")
    reference = means(indices)
    tails = {arm: float(np.quantile([score(i, arm)["mrstft"] for i in indices], .9)) for arm in ARMS}
    earlier = plan["earlier_sound_reference"]
    preserved = all(reference["balanced"][metric] <= earlier["means"][metric] for metric in METRICS) and tails["balanced"] <= earlier["mrstft_p90"]
    report["references"] = {"means": reference, "mrstft_p90": tails, "earlier": earlier, "preserved": bool(preserved)}
    report["selected_means"] = means(plan["selected_comparisons"])
    improved = all(report["selected_means"]["balanced"][metric] <= .95 * report["selected_means"]["control"][metric] for metric in METRICS)
    no_patch_regression = all(group["means"]["balanced"][metric] <= 1.05 * group["means"]["control"][metric]
                              for group in report["three_sounds"].values() for metric in METRICS)
    report["directional_data_benefit"] = bool(improved and no_patch_regression and preserved)
    saved = torch.load(store.output / "checkpoint.pt", map_location="cpu", weights_only=True)
    initial = inputs.model.state_dict()
    branches = {branch: any(not torch.equal(value, saved["model"][name]) for name, value in initial.items()
                           if name.startswith(f"correction.0.{branch}.")) for branch in ("timbre", "pitch")}
    access = {group: any(row["arms"]["balanced"]["gradients"][group] > 0 for row in saved["trace"][1:])
              for group in ("hidden", "continuous", "unison", "filter")}
    accepted = sum(row["arms"]["balanced"]["acceptance"]["accepted"] for row in saved["trace"])
    training_passed = accepted >= 36 and all(branches.values()) and all(access.values())
    report["training_checks"] = {"accepted_updates": accepted, "copied_branches_changed": branches,
                                 "later_gradient_access": access, "pass": bool(training_passed)}
    report["full_reconstruction_gates"] = bool(training_passed and preserved
        and all(row["pass"] for row in report["six_groups"].values()) and all(row["pass"] for row in report["grid"].values())
        and all(row["learning_prerequisite_passed"] for row in report["three_sounds"].values()))
    report["model_admitted"] = False
    return report


def software_check(inputs, output):
    started, monitor = time(), MemoryMonitor()
    monitor.start()

    class AnalyticStore:
        def probe(self, _step, _observation, index, patch):
            values = torch.tensor(patch.continuous, dtype=torch.float32)
            return 1. + float((values - (.2 + .01 * (index % 4))).square().mean()) + .02 * sum(patch.categorical)

    try:
        before = tensor_digest(inputs.model.state_dict())
        for index, cell in enumerate(inputs.plan["fit_audit_cells"][:104]):
            # Historical byte-exact source checks reproduce the original singleton layout.
            prediction = inputs.predict(inputs.model, cell)
            if inputs.adapter.export(prediction["continuous"], prediction["categorical"])[0].to_dict() != inputs.cases[index]["rows"]["initial"]["candidate"]:
                raise ValueError("balanced source bridge changed an old starting prediction")
        model, expected = train(inputs, AnalyticStore(), stop_after=2)
        _model, first = train(inputs, AnalyticStore(), stop_after=1)
        _model, recovered = train(inputs, AnalyticStore(), stop_after=2, saved=first)
        if not equal_tree(expected, recovered) or tensor_digest(inputs.model.state_dict()) != before:
            raise ValueError("balanced software recovery changed model, Adam, trace, RNG or initializer")
        cold, _front, _codec = restore_direct(export_inward(model, inputs.frontend, inputs.codec, inputs.plan["pool_id"]))
        with torch.no_grad():
            warm_rows = predict_cells(inputs, model, inputs.plan["fit_audit_cells"])
            cold_rows = predict_cells(inputs, cold, inputs.plan["fit_audit_cells"])
        for a, b in zip(warm_rows, cold_rows, strict=True):
            if inputs.adapter.export(a["continuous"], a["categorical"]) != inputs.adapter.export(b["continuous"], b["categorical"]):
                raise ValueError("balanced software cold export changed a prediction")
    finally:
        monitor.close()
    elapsed = time() - started
    if elapsed > 600 or not monitor.peak_bytes or monitor.peak_bytes > LIMITS["process_tree_bytes"] or monitor.errors:
        raise ValueError("balanced software check exceeded ten minutes or 4 GiB")
    report = {"plan_id": canonical_sha256(inputs.plan), "passed": True, "old_starting_predictions_unchanged": 104,
        "cold_export_comparisons": 158, "model_adam_trace_rng_recovery_exact": True, "simulated_optimizer_applications": 4,
        "native_requests": 0, "experimental_updates": 0, "elapsed_seconds": elapsed,
        "peak_process_tree_rss_bytes": monitor.peak_bytes, "rss_sampling_errors": monitor.errors}
    write_json(output / "software-check.json", report)
    return report


def verify(inputs, output):
    started, monitor = time(), MemoryMonitor()
    monitor.start()
    try:
        report = verify_evidence(inputs, output)
    finally:
        monitor.close()
    elapsed = time() - started
    if elapsed > 1200 or not monitor.peak_bytes or monitor.peak_bytes > LIMITS["process_tree_bytes"] or monitor.errors:
        raise ValueError("balanced audit exceeded twenty minutes or 4 GiB")
    report["resources"] = {"elapsed_seconds": elapsed, "peak_process_tree_rss_bytes": monitor.peak_bytes, "rss_sampling_errors": monitor.errors}
    write_json(output / "verification.json", report)
    return report


def verify_evidence(inputs, output):
    summary, plan = read(output / "summary.json"), inputs.plan
    plan_id = canonical_sha256(plan)
    if (not summary["complete"] or summary["failure"] or not summary["within_resource_limits"]
            or summary["plan_id"] != plan_id or read(output / "run.json")["plan_id"] != plan_id):
        raise ValueError("balanced fit is incomplete or its frozen identity changed")
    names = {"plan.json", "run.json", "software-check.json", "executions.jsonl", "updates.jsonl", "probes.jsonl", "renders.jsonl",
             "finals.jsonl", "checkpoint.pt", "checkpoint.previous.pt", "prefix-verification.json", "summary.json", "result.json", "diagnostic-balanced.pt"}
    evidence = {path.relative_to(output).as_posix(): digest(path) for path in output.rglob("*") if path.is_file()
                and (path.name in names or path.relative_to(output).parts[0] in ("comparison_audio", "source_audio", "interruption"))}
    store = BalancedStore(output, inputs)
    expected = {(i, arm) for i in plan["evaluation_indices"] for arm in ("control", "balanced")}
    expected |= {(i, "initial") for i in plan["grid_comparisons"]} | {(i, "balanced_repeat") for i in plan["repeat_comparisons"]}
    if set(store.finals) != expected:
        raise ValueError("balanced endpoint coverage is incomplete or includes undeclared comparisons")
    saved = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
    model, replay = train(inputs, store)
    if not equal_tree(saved, replay) or saved["step"] != 48 or store.used_probes != set(store.probes):
        raise ValueError("balanced model, Adam, RNG or complete probe trajectory did not replay exactly")
    cold, _front, _codec = restore_direct(torch.load(output / "diagnostic-balanced.pt", map_location="cpu", weights_only=True))
    if not equal_tree(cold.state_dict(), model.state_dict()):
        raise ValueError("balanced endpoint cold export differs from replay")
    attempts, outcomes, recovered = native_ledger(lines(output / "renders.jsonl"), plan["expected_renderer_identity"])
    rescored = 0
    waveform_audit = WaveformAudit(BatchConfig.from_plan(plan).spectral)
    with torch.no_grad():
        prediction_banks = {id(network): predict_cells(inputs, network, [plan["fit_audit_cells"][index] for index in plan["evaluation_indices"]])
                            for network in (inputs.model, inputs.control_model, model)}
    for position, index in enumerate(plan["evaluation_indices"]):
        cell = plan["fit_audit_cells"][index]
        reference, _source = store.source(index)
        arms = (*ARMS, "balanced_repeat") if index in plan["repeat_comparisons"] else ARMS
        for arm in arms:
            row = store.initial(index, audit=waveform_audit) if arm == "initial" else store.finals[index, arm]
            patch = PilotTarget.from_dict(row["candidate"])
            receipt = row["receipt"]
            require_request(inputs.codec, plan, patch, Performance.from_dict(cell["recipe"]["performances"][cell["playback_view"]]), receipt)
            waveform_audit.add(output, receipt, reference, inputs.codec.contract.tail_seconds, row["scores"])
            if arm == "control" and index < 104:
                original = inputs.control_finals[index, "sound_only"]
                if any(row[key] != original[key] for key in ("candidate", "scores", "receipt")):
                    raise ValueError("reused control endpoint changed")
            elif not (arm == "initial" and index < 104):
                if {key: value for key, value in receipt.items() if key != "retained_audio"} != outcomes[receipt["phase"], receipt["attempt"]]:
                    raise ValueError("comparison receipt differs from its charged native outcome")
            network = inputs.model if arm == "initial" else inputs.control_model if arm == "control" else model
            with torch.no_grad():
                prediction = prediction_banks[id(network)][position]
            if inputs.adapter.export(prediction["continuous"], prediction["categorical"])[0] != patch:
                raise ValueError("comparison patch differs from its declared model")
            rescored += 1
    waveform_audit.finish()
    for index in plan["repeat_comparisons"]:
        if store.finals[index, "balanced"]["receipt"]["qa"]["audio_float32_sha256"] != store.finals[index, "balanced_repeat"]["receipt"]["qa"]["audio_float32_sha256"]:
            raise ValueError("balanced fixed-phase endpoint repeat differs")
    for row in store.probes.values():
        query, receipt = row["query"], row["receipt"]
        arm, observation = query["observation"].rsplit("-", 1)
        cells = inputs.observations(plan["updates"][query["step"] - 1])
        if arm != "balanced" or query["index"] not in cells[int(observation)] or receipt != outcomes[receipt["phase"], receipt["attempt"]]:
            raise ValueError("balanced training receipt is outside its declared update")
        cell = plan["fit_audit_cells"][query["index"]]
        require_request(inputs.codec, plan, PilotTarget.from_dict(query["candidate"]), Performance.from_dict(cell["recipe"]["performances"][cell["playback_view"]]), receipt)
    charges = lines(output / "updates.jsonl")
    steps = [row["step"] for row in charges]
    if (len(charges) > LIMITS["optimizer_attempts"] or [row["attempt"] for row in charges] != list(range(len(charges)))
            or sorted(set(steps)) != list(range(1, 49)) or steps != sorted(steps) or len(charges) - len(set(steps)) > len(recovered)):
        raise ValueError("balanced update charge ledger changed")
    result = results(inputs, store)
    if result != read(output / "result.json") or rescored != 426 or len(attempts) != summary["native_attempts"]:
        raise ValueError("balanced result, evaluation count or native accounting changed")
    return {"plan_id": plan_id, "verified": True, "complete": True, "model_adam_trace_rng_replay_exact": True,
            "comparison_rows_rescored": rescored, "recovered_timeouts": recovered, "evidence_sha256": evidence,
            "result": result, "qualified_for_release": False, "test_split_used": False}

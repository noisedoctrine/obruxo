"""Replay direct/rehearsal updates and independently rescore final native sounds."""

from time import perf_counter

import numpy as np
import torch

from obruxo_data.hashing import canonical_sha256
from obruxo_data.midi import Performance

from .audit import write_json
from .basic_direct import restore_direct
from .basic_direct_fit import SoundStore, lines, probe_label, replay_training, require_charges
from .benchmark import MemoryMonitor
from .codec import PilotTarget
from .conditioned_bridge import read
from .conditioned_short_audit import means_by_state
from .features import tensor_digest
from .native_fit import digest
from .paired_native_evidence import native_ledger, rescore_waveform
from .unison_fit_data import require_request


def decisions(inputs, store, trace):
    cells = inputs.plan["fit_audit_cells"]
    if set(store.finals) != {(index, arm) for index in range(len(cells)) for arm in ("parameter", "spectral")}:
        raise ValueError("direct final comparison coverage is incomplete")
    cases = [{"state_id": cell["recipe"]["state_id"], "input_view": cell["input_view"], "playback_view": cell["playback_view"],
              "training_index": cell["training_index"], "rows": {"initial": store.initial(index),
                **{arm: store.finals[(index, arm)] for arm in ("parameter", "spectral")}}} for index, cell in enumerate(cells)]
    groups = {}
    for view, role in enumerate(inputs.plan["training_view_roles"]):
        for transfer in (False, True):
            selected = [row for row in cases if row["training_index"] is not None and row["input_view"] == view
                        and (row["input_view"] != row["playback_view"]) == transfer]
            means = means_by_state(selected)
            passed = (means["spectral"]["spectral"] <= inputs.plan["gates"]["six_groups_sound_ratio"] * means["initial"]["spectral"]
                      and means["spectral"]["spectral"] <= means["parameter"]["spectral"]
                      and means["spectral"]["mrstft"] < means["initial"]["mrstft"])
            groups[f"{role}_{'transfer' if transfer else 'own'}"] = {"states": len({row["state_id"] for row in selected}),
                "cases": len(selected), "means": means, "pass": bool(passed)}
    original = [row for row in cases if row["input_view"] == row["playback_view"] == 0]
    if {row["state_id"] for row in original} != set(inputs.plan["regression_state_ids"]):
        raise ValueError("direct original preservation panel changed")
    means = means_by_state(original)
    p90 = {arm: float(np.quantile([row["rows"][arm]["scores"]["mrstft"] for row in original], .9)) for arm in ("initial", "parameter", "spectral")}
    regression = {"states": len(original), "means": means, "mrstft_p90": p90,
                  "pass": bool(all(means["spectral"][metric] <= means["initial"][metric] for metric in ("spectral", "mrstft"))
                               and p90["spectral"] <= inputs.plan["gates"]["original_32_mrstft_p90_ratio"] * p90["initial"])}
    access = {arm: {group: any(row["correction_gradients"][arm][group] > 0 for row in trace[1:])
                    for group in ("hidden", "continuous", "unison", "filter")} for arm in ("parameter", "spectral")}
    reached = len(trace) > 1 and all(all(groups.values()) for groups in access.values())
    clamping = {arm: {bound: sum(row["clamping"][arm][obs][bound] for row in trace for obs in range(4))
                      for bound in ("below_zero", "above_one", "controls")} for arm in ("parameter", "spectral")}
    result = {"six_groups": groups, "original_regression": regression, "later_gradient_access": access, "clamping": clamping,
              "pass": bool(all(row["pass"] for row in groups.values()) and regression["pass"] and reached)}
    earlier = inputs.plan.get("earlier_sound_reference")
    if earlier:
        current = {"means": means["spectral"], "mrstft_p90": p90["spectral"]}
        preserved = all(current["means"][metric] <= earlier["means"][metric] for metric in ("spectral", "mrstft")) and current["mrstft_p90"] <= earlier["mrstft_p90"]
        result["earlier_sound_preservation"] = {"current": current, "earlier": earlier, "pass": bool(preserved)}
        result["pass"] = bool(result["pass"] and preserved)
    return result


def verify(inputs, output):
    plan, summary = inputs.plan, read(output / "summary.json")
    plan_id = canonical_sha256(plan)
    if (read(output / "plan.json") != plan or read(output / "run.json")["plan_id"] != plan_id
            or summary["plan_id"] != plan_id or not summary["complete"] or summary["failure"] or not summary["within_resource_limits"]):
        raise ValueError("direct fit is incomplete or frozen evidence changed")
    executions = lines(output / "executions.jsonl")
    if not 1 <= len(executions) <= plan["limits"]["native_executions"] or any(row["ordinal"] != index + 1
            or row["resume"] != (index > 0) or row["plan_id"] != plan_id for index, row in enumerate(executions)):
        raise ValueError("direct native execution history changed")
    attempts, outcomes, recovered = native_ledger(lines(output / "renders.jsonl"), plan["expected_renderer_identity"])
    for phase in ("training", "qualification"):
        if sum(key[0] == phase for key in attempts) > plan["limits"][f"{phase}_attempts"]:
            raise ValueError("direct native attempt budget exceeded")
    store, used, audio_ids = SoundStore(output, inputs), set(), {}
    for case in inputs.cases:
        for receipt in (case["source_receipt"], case["rows"]["initial"]["receipt"]):
            audio_ids[receipt["request_id"]] = receipt["qa"]["audio_float32_sha256"]

    def receipt_check(receipt, phase, label):
        raw = {key: value for key, value in receipt.items() if key != "retained_audio"}
        if (raw != outcomes.get((receipt["phase"], receipt["attempt"])) or receipt["phase"] != phase or receipt["label"] != label):
            raise ValueError("direct sound cites an unrecorded native request")
        for key, row in outcomes.items():
            if row["status"] == "accepted" and all(row[field] == receipt[field] for field in ("phase", "label", "request_id", "source_sha256", "controlled_sha256")):
                if row["qa"]["audio_float32_sha256"] != receipt["qa"]["audio_float32_sha256"]:
                    raise ValueError("identical direct recovery request changed audio")
                used.add(key)
        request, audio = receipt["request_id"], receipt["qa"]["audio_float32_sha256"]
        if request in audio_ids and audio_ids[request] != audio:
            raise ValueError("identical fixed-phase direct request changed audio")
        audio_ids[request] = audio

    saved = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
    prefix = read(output / "prefix-verification.json")
    snapshot = torch.load(output / "prefix-checkpoint.pt", map_location="cpu", weights_only=True)
    charges = lines(output / "updates.jsonl")
    require_charges(plan, charges, saved["trace"])
    if (saved["step"] != len(plan["updates"]) or summary["committed_updates_per_arm"] != saved["step"]
            or prefix["plan_id"] != plan_id or not prefix["offline_training_replay_exact"]
            or prefix["replayed_updates"] != plan["prefix_check_updates"] or snapshot["step"] != prefix["replayed_updates"]
            or digest(output / "prefix-checkpoint.pt") != prefix["checkpoint_file_sha256"] or snapshot["trace"] != saved["trace"][:snapshot["step"]]):
        raise ValueError("direct final checkpoint or prefix evidence changed")
    replay_training(inputs, SoundStore(output, inputs), snapshot)
    models, _optimizers = replay_training(inputs, store, saved, prefix)
    expected = plan.get("expected_parameter_model_sha256")
    if expected and tensor_digest(models["parameter"].state_dict()) != expected:
        raise ValueError("auxiliary comparison changed the matched parameter-control identity")
    expected_prefix = plan.get("expected_parameter_prefix_model_sha256")
    if expected_prefix and tensor_digest(snapshot["models"]["parameter"]) != expected_prefix:
        raise ValueError("auxiliary comparison changed the matched parameter-control prefix")
    if store.used_probes != set(store.probes):
        raise ValueError("direct probe ledger contains unused measurements")
    for row in store.probes.values():
        query = row["query"]
        index = query["index"]
        cells = inputs.observations(plan["updates"][query["step"] - 1])
        if index not in cells[query["observation"]] or not np.isfinite(row["loss"]) or row["loss"] < 0:
            raise ValueError("direct probe source or scalar cost changed")
        cell = plan["fit_audit_cells"][index]
        performance = Performance.from_dict(cell["recipe"]["performances"][cell["playback_view"]])
        candidate = PilotTarget.from_dict(query["candidate"])
        require_request(inputs.codec, plan, candidate, performance, row["receipt"])
        receipt_check(row["receipt"], "training", probe_label(query))
        _source, metadata = store.source(index)
        if metadata != row["source_audio"]:
            raise ValueError("direct probe ground-truth recording changed")
    identities = {}
    compared = 0
    for arm, model in models.items():
        artifact = torch.load(output / f"diagnostic-{arm}.pt", map_location="cpu", weights_only=True)
        cold, _frontend, _codec = restore_direct(artifact)
        if (artifact["training_plan_id"] != plan_id or artifact["training_objective"] != arm
                or tensor_digest(cold.state_dict()) != tensor_digest(model.state_dict())):
            raise ValueError("direct cold export differs from the replayed model")
        identities[arm] = tensor_digest(model.state_dict())
        for index, cell in enumerate(plan["fit_audit_cells"]):
            with torch.no_grad():
                candidate = inputs.target(index, prediction=inputs.predict(model.eval(), cell))
                cold_target = inputs.target(index, prediction=inputs.predict(cold, cell))
            row = store.finals[(index, arm)]
            if candidate != cold_target or row["candidate"] != candidate.to_dict() or row["arm"] != arm or row["origin"] != "new" or row["index"] != index:
                raise ValueError("direct final production prediction changed")
            performance = Performance.from_dict(cell["recipe"]["performances"][cell["playback_view"]])
            require_request(inputs.codec, plan, candidate, performance, row["receipt"])
            receipt_check(row["receipt"], "qualification", f"final:{index:03d}:{arm}")
            source, metadata = store.source(index)
            if metadata != row["source_audio"] or rescore_waveform(output, row["receipt"], source, inputs.codec.contract.tail_seconds) != row["scores"]:
                raise ValueError("direct final sound cannot be independently reconstructed")
            compared += 1
    for index in range(len(plan["fit_audit_cells"])):
        store.initial(index)  # Rescores the accepted initial waveform against the bound source.
        compared += 1
    if used != {key for key, row in outcomes.items() if row["status"] == "accepted"}:
        raise ValueError("direct native ledger has an unexplained accepted request")
    result = decisions(inputs, store, saved["trace"])
    archive = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    if (result != summary["result"] or recovered != summary["recovered_timeouts"] or not summary["native_receipts_resolved"]
            or summary["matched_comparison_passed"] != result["pass"] or archive + 1024**2 > plan["limits"]["archive_bytes"]):
        raise ValueError("direct final decision, recovery or archive changed")
    roots = ("plan.json", "run.json", "executions.jsonl", "summary.json", "checkpoint.pt", "prefix-checkpoint.pt",
             "prefix-verification.json", "renders.jsonl", "updates.jsonl", "probes.jsonl", "finals.jsonl", "diagnostic-parameter.pt", "diagnostic-spectral.pt")
    evidence = {name: digest(output / name) for name in roots}
    for folder in ("source_audio", "comparison_audio", "interruption"):
        evidence.update({path.relative_to(output).as_posix(): digest(path) for path in (output / folder).rglob("*") if path.is_file()})
    return {"plan_id": plan_id, "complete": True, "offline_training_replay_exact": True, "committed_updates_per_arm": saved["step"],
            "comparison_waveforms_rescored": compared, "training_scalar_probes_replayed": len(store.probes), "training_probe_audio_independently_rescored": False,
            "native_receipts_resolved": True, "recovered_timeouts": recovered, "result": result, "matched_comparison_passed": result["pass"],
            "final_model_sha256": identities, "evidence_sha256": evidence, "test_split_used": False, "qualified_for_release": False}


def run_verification(inputs, output):
    path = output / "verification.json"
    if path.exists():
        raise ValueError("preserve the existing direct fit verification")
    started, monitor = perf_counter(), MemoryMonitor()
    monitor.start()
    try:
        result = verify(inputs, output)
    finally:
        monitor.close()
    within = bool(perf_counter() - started < 1200 and monitor.peak_bytes and monitor.peak_bytes <= 4 * 1024**3 and not monitor.errors)
    result["verification_resources"] = {"elapsed_seconds": perf_counter() - started, "peak_process_tree_rss_bytes": monitor.peak_bytes,
                                        "rss_sampling_errors": monitor.errors, "within_resource_limits": within}
    result["verified"] = within
    write_json(path, result)
    if not within:
        raise ValueError("direct verification exceeded twenty minutes or 4 GiB")
    return {key: result[key] for key in ("verified", "matched_comparison_passed", "offline_training_replay_exact", "comparison_waveforms_rescored", "verification_resources")}

"""Independent finite-cost waveform rescoring and exact two-arm training replay."""

from time import perf_counter

import numpy as np
import torch

from obruxo_data.hashing import canonical_sha256
from obruxo_data.midi import Performance

from .audit import write_json
from .benchmark import MemoryMonitor
from .conditioned_bridge import read
from .conditioned_short_audit import means_by_state
from .features import tensor_digest
from .native_fit import digest
from .paired_native_audit import equal_tree
from .paired_native_evidence import native_ledger, rescore_waveform
from .unison_choice import restore_unison
from .unison_fit_data import require_request
from .unison_sound_fit import CostStore, lines, replay_training


def decisions(inputs, store, signal):
    cells = inputs.plan["fit_audit_cells"]
    if set(store.finals) != {(index, arm) for index in range(len(cells)) for arm in ("parameter", "spectral")}:
        raise ValueError("unison final comparison coverage is incomplete")
    cases = [{"state_id": cell["recipe"]["state_id"], "input_view": cell["input_view"], "playback_view": cell["playback_view"],
              "training_index": cell["training_index"], "rows": {"initial": store.initial(index),
                **{arm: store.finals[(index, arm)] for arm in ("parameter", "spectral")}}} for index, cell in enumerate(cells)]
    groups = {}
    for view, role in enumerate(inputs.plan["training_view_roles"]):
        for transfer in (False, True):
            selected = [row for row in cases if row["training_index"] is not None and row["input_view"] == view
                        and (row["input_view"] != row["playback_view"]) == transfer]
            means = means_by_state(selected)
            passed = all(means["spectral"][metric] <= means["initial"][metric] for metric in ("spectral", "mrstft"))
            groups[f"{role}_{'transfer' if transfer else 'own'}"] = {"states": len({row["state_id"] for row in selected}),
                "cases": len(selected), "means": means, "pass": bool(passed)}
    wider = {}
    for view in (1, 2):
        selected = [row for row in cases if row["training_index"] is not None and row["input_view"] == view]
        means = means_by_state(selected)
        ceiling = signal["groups"][str(view)]
        required = ceiling["initial_mean"] - inputs.plan["gates"]["fraction_of_available_gain"] * ceiling["available_gain"]
        wider[str(view)] = {"means": means, "oracle": ceiling, "required_sound_loss_at_most": required,
                            "pass": bool(means["spectral"]["spectral"] <= min(required, means["parameter"]["spectral"]))}
    original = [row for row in cases if row["input_view"] == row["playback_view"] == 0]
    if {row["state_id"] for row in original} != set(inputs.plan["regression_state_ids"]):
        raise ValueError("unison original preservation panel changed")
    means = means_by_state(original)
    p90 = {name: float(np.quantile([row["rows"][name]["scores"]["mrstft"] for row in original], .9)) for name in ("initial", "parameter", "spectral")}
    regression = {"states": len(original), "means": means, "mrstft_p90": p90,
                  "pass": bool(all(means["spectral"][metric] <= means["initial"][metric] for metric in ("spectral", "mrstft"))
                               and p90["spectral"] <= inputs.plan["gates"]["original_32_mrstft_p90_ratio"] * p90["initial"])}
    return {"six_groups": groups, "wider_oracle_recovery": wider, "original_regression": regression,
            "pass": bool(signal["pass"] and all(row["pass"] for row in groups.values()) and all(row["pass"] for row in wider.values()) and regression["pass"])}


def verify(inputs, output):
    plan, summary = inputs.plan, read(output / "summary.json")
    if (read(output / "plan.json") != plan or not summary["complete"] or summary["failure"] is not None
            or summary["plan_id"] != canonical_sha256(plan) or read(output / "run.json")["plan_id"] != canonical_sha256(plan)):
        raise ValueError("unison fit is incomplete or frozen evidence changed")
    executions = lines(output / "executions.jsonl")
    if (not 1 <= len(executions) <= plan["limits"]["native_executions"] or any(row["ordinal"] != index + 1
            or row["resume"] != (index > 0) or row["plan_id"] != canonical_sha256(plan) for index, row in enumerate(executions))):
        raise ValueError("unison execution history changed")
    attempts, outcomes, recovered = native_ledger(lines(output / "renders.jsonl"), plan["expected_renderer_identity"])
    store, used, audio_ids = CostStore(output, inputs), set(), {}
    for case in inputs.cases:
        for receipt in (case["source_receipt"], case["rows"]["initial"]["receipt"]):
            request, audio = receipt["request_id"], receipt["qa"]["audio_float32_sha256"]
            if request in audio_ids and audio_ids[request] != audio:
                raise ValueError("identical retained unison request changed audio")
            audio_ids[request] = audio
    expected_costs = {(index, choice) for index, cell in enumerate(plan["fit_audit_cells"]) if cell["training_index"] is not None for choice in range(3)}
    if set(store.costs) != expected_costs:
        raise ValueError("unison measured choice coverage changed")
    def receipt_check(receipt, phase, label):
        bare = {key: value for key, value in receipt.items() if key != "retained_audio"}
        if bare != outcomes.get((receipt["phase"], receipt["attempt"])) or receipt["phase"] != phase or receipt["label"] != label:
            raise ValueError("unison waveform cites an unrecorded native request")
        for key, row in outcomes.items():
            if row["status"] == "accepted" and all(row[field] == receipt[field] for field in ("phase", "label", "request_id", "source_sha256", "controlled_sha256")):
                if row["qa"]["audio_float32_sha256"] != receipt["qa"]["audio_float32_sha256"]:
                    raise ValueError("identical unison recovery request changed audio")
                used.add(key)
    def check_row(index, candidate, row):
        cell = plan["fit_audit_cells"][index]
        performance = Performance.from_dict(cell["recipe"]["performances"][cell["playback_view"]])
        if row["index"] != index or row["candidate"] != candidate.to_dict():
            raise ValueError("unison measured or final patch changed")
        require_request(inputs.codec, plan, candidate, performance, row["receipt"])
        request, audio = row["receipt"]["request_id"], row["receipt"]["qa"]["audio_float32_sha256"]
        if request in audio_ids and audio_ids[request] != audio:
            raise ValueError("identical fixed-phase unison request changed audio")
        audio_ids[request] = audio
        source, metadata = store.source(index)
        if metadata != row["source_audio"] or rescore_waveform(output, row["receipt"], source, inputs.codec.contract.tail_seconds) != row["scores"]:
            raise ValueError("unison sound cost cannot be independently reconstructed")
    for (index, choice), row in store.costs.items():
        check_row(index, inputs.target(index, choice), row)
        if row["choice"] != choice:
            raise ValueError("unison cost choice changed")
        if row["origin"] == "parent":
            if row != store.initial(index):
                raise ValueError("reused initial unison evidence changed")
        elif row["origin"] == "new":
            receipt_check(row["receipt"], "training", f"cost:{index:03d}:{choice}")
        else:
            raise ValueError("unknown unison cost origin")
    signal = store.signal()
    if signal != summary["signal"] or read(output / "signal.json") != signal:
        raise ValueError("unison oracle signal changed")
    compared = len(store.costs)
    final_ids = {}
    charges = lines(output / "updates.jsonl")
    if signal["pass"]:
        saved = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
        prefix = read(output / "prefix-verification.json")
        snapshot = torch.load(output / "prefix-checkpoint.pt", map_location="cpu", weights_only=True)
        if (prefix["plan_id"] != canonical_sha256(plan) or prefix["replayed_updates"] != plan["prefix_check_updates"]
                or not prefix["offline_training_replay_exact"] or digest(output / "prefix-checkpoint.pt") != prefix["checkpoint_file_sha256"]
                or snapshot["trace"] != saved["trace"][:snapshot["step"]] or snapshot["step"] != prefix["replayed_updates"]
                or saved["step"] != len(plan["updates"]) or summary["committed_updates_per_arm"] != saved["step"]
                or len(charges) != saved["step"] or len(charges) > plan["limits"]["optimizer_attempts"]):
            raise ValueError("unison final checkpoint or prefix evidence changed")
        for ordinal, (charge, row) in enumerate(zip(charges, saved["trace"], strict=True)):
            if charge != {key: row[key] for key in ("attempt", "step", "examples")} or charge["attempt"] != ordinal or charge["step"] != ordinal + 1:
                raise ValueError("unison charged update trace changed")
        replay_training(inputs, store, snapshot)
        models, _optimizers = replay_training(inputs, store, saved, prefix)
        for name, model in models.items():
            exported = torch.load(output / f"diagnostic-{name}.pt", map_location="cpu", weights_only=True)
            cold, front, codec = restore_unison(exported)
            if (tensor_digest(cold.state_dict()) != tensor_digest(model.state_dict()) or codec.to_dict() != inputs.codec.to_dict()
                    or front.describe() != inputs.frontend.describe() or exported["training_plan_id"] != canonical_sha256(plan)
                    or exported["training_objective"] != name or exported["qualified_for_release"]):
                raise ValueError("unison cold export differs from its replayed model")
            final_ids[name] = tensor_digest(cold.state_dict())
        if set(store.finals) != {(index, arm) for index in range(len(plan["fit_audit_cells"])) for arm in models}:
            raise ValueError("unison final fresh comparison coverage changed")
        for (index, arm), row in store.finals.items():
            with torch.no_grad():
                candidate = inputs.target(index, prediction=inputs.predict(models[arm].eval(), plan["fit_audit_cells"][index]))
            check_row(index, candidate, row)
            if row["origin"] != "new":
                raise ValueError("final unison comparison must use a fresh native request")
            receipt_check(row["receipt"], "qualification", f"final:{index:03d}:{arm}")
            compared += 1
        result = decisions(inputs, store, signal)
        compared += sum(cell["training_index"] is None for cell in plan["fit_audit_cells"])
    else:
        if charges or store.finals or summary["committed_updates_per_arm"] != 0:
            raise ValueError("unison fit advanced after a failed signal gate")
        result = None
    interruption = output / "interruption"
    if interruption.exists():
        for name, sha in read(interruption / "evidence.json").items():
            if digest(interruption / name) != sha or (name.endswith(".jsonl") and not (output / name).read_bytes().startswith((interruption / name).read_bytes())):
                raise ValueError("unison interruption evidence changed")
        if (interruption / "checkpoint.pt").exists():
            old = torch.load(interruption / "checkpoint.pt", map_location="cpu", weights_only=True)
            if old["step"] != saved["step"] or not equal_tree(old, saved):
                raise ValueError("unison interruption checkpoint changed")
    archive = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    counts = {phase: sum(key[0] == phase for key in attempts) for phase in ("training", "qualification")}
    within = bool(summary["rss_sampling_errors"] == 0 and 0 < summary["peak_process_tree_rss_bytes"] <= plan["limits"]["process_tree_bytes"]
                  and summary["elapsed_seconds"] <= plan["limits"]["wall_seconds"] and archive + 1024**2 <= plan["limits"]["archive_bytes"]
                  and all(counts[phase] <= plan["limits"][f"{phase}_attempts"] for phase in counts))
    if (used != {key for key, row in outcomes.items() if row["status"] == "accepted"} or result != summary["result"]
            or counts != summary["render_attempts"] or recovered != summary["recovered_timeouts"] or not summary["native_receipts_resolved"]
            or within != summary["within_resource_limits"] or summary["matched_comparison_passed"] != bool(result and result["pass"] and within)):
        raise ValueError("unison request accounting or final quality/resource decision changed")
    return {**summary, "offline_training_replay_exact": True, "comparison_waveforms_rescored": compared,
            "final_model_sha256": final_ids, "native_requests_added": 0, "experimental_training_updates_added": 0,
            "evidence_sha256": {name: digest(output / name) for name in ("plan.json", "summary.json", "costs.jsonl", "renders.jsonl", "executions.jsonl",
                "checkpoint.pt", "updates.jsonl", "finals.jsonl", "prefix-verification.json", "prefix-checkpoint.pt", "diagnostic-parameter.pt", "diagnostic-spectral.pt")
                if (output / name).exists()}}


def run_verification(inputs, output):
    path = output / "verification.json"
    if path.exists():
        raise ValueError("preserve the existing unison fit verification")
    started, monitor = perf_counter(), MemoryMonitor()
    monitor.start()
    try:
        result = verify(inputs, output)
    finally:
        monitor.close()
    elapsed = perf_counter() - started
    within = bool(elapsed < 600 and monitor.peak_bytes and monitor.peak_bytes <= 4 * 1024**3 and not monitor.errors)
    result.update(verified=within, verification_resources={"elapsed_seconds": elapsed, "peak_process_tree_rss_bytes": monitor.peak_bytes,
                                                       "rss_sampling_errors": monitor.errors, "within_resource_limits": within})
    write_json(path, result)
    if not within:
        raise ValueError("unison independent audit exceeded ten minutes or 4 GiB")
    return result

"""Frozen local execution and offline audit of the shared reconstruction comparison."""

import copy
import random
from pathlib import Path
from time import time

import numpy as np
import torch
from obruxo_data.hashing import canonical_sha256, sha256_bytes
from obruxo_data.midi import Performance

from .audit import write_json
from .basic_direct import rehearsal_indices, restore_direct
from .basic_direct_fit import SoundStore, charge_execution
from .basic_direct_fit_data import Inputs
from .benchmark import MemoryMonitor
from .checkpoints import atomic_torch_save
from .codec import PilotTarget
from .conditioned_bridge import read, training_bank
from .features import tensor_digest
from .inward_basic import POLICY, export_inward
from .native_fit import append_durable, digest
from .paired_native_audit import equal_tree
from .paired_native_evidence import native_ledger, rescore_waveform
from .paired_native_fit import PairedSession
from .paired_native_step import reciprocal_updates
from .reproducibility import FULL_PHASE_CONTROL
from .shared_basic_auxiliary import ARMS, SPEC, encode, initialize, step
from .trainable_basic import COUNTS
from .trainable_basic_fit import source_code
from .trainer import capture_rng, restore_rng
from .unison_fit_data import require_request
from .unison_sound_fit import lines

SOURCE_NAMES = ("shared_basic_auxiliary.py", "shared_auxiliary_run.py", "fitted_note_head.py", "branch_adversary.py", "inward_basic.py",
                "reports/scripts/run_shared_basic_auxiliary.py", "reports/SHARED_BASIC_AUXILIARY_CARD.md")
LIMITS = {"wall_seconds": 36000, "process_tree_bytes": 4 * 1024**3, "archive_bytes": 2 * 1024**3,
          "replay_bytes": 512 * 1024**2, "optimizer_attempts": 49, "native_executions": 2,
          "training_attempts": 15000, "qualification_attempts": 360}


def source_identity(prior):
    inherited, _differences = source_code(prior)
    root = Path(__file__).parents[1]
    current = {name: sha256_bytes((root / name if "/" in name else Path(__file__).with_name(name)).read_bytes().replace(b"\r\n", b"\n"))
               for name in SOURCE_NAMES}
    return {**inherited, **current}


def prepare(parent, bridge, cache, pool, fit, auxiliary, output):
    old, proof = read(fit / "plan.json"), read(fit / "verification.json")
    initial, qualification = torch.load(bridge / "model.pt", map_location="cpu", weights_only=True), read(bridge / "summary.json")
    model, _frontend, codec = restore_direct(initial)
    split = read(auxiliary / "plan.json")
    if (not proof["verified"] or not proof["complete"] or not proof["offline_training_replay_exact"]
            or proof["plan_id"] != canonical_sha256(old) or proof["qualified_for_release"] or proof["test_split_used"]
            or proof["committed_updates_per_arm"] != 48 or pool.pool_id != old["pool_id"] or codec.to_dict() != pool.codec.to_dict()
            or digest(cache) != old["input_cache_sha256"] or digest(bridge / "model.pt") != old["initial_file_sha256"]
            or tensor_digest(model.state_dict()) != old["initial_model_sha256"]
            or qualification["model_file_sha256"] != old["initial_file_sha256"]
            or old["render_phase_controls"] != FULL_PHASE_CONTROL or old["render_profile"]["preroll_frames"] != 32768
            or len(split["training_states"]) != 259 or len(split["probe_test_states"]) != 145
            or set(split["training_states"]) & set(split["probe_test_states"])):
        raise ValueError("shared comparison requires the verified fit, unchanged initializer, cache, renderer and auxiliary split")
    bound = {str(path): digest(path) for path in (cache, bridge / "model.pt", bridge / "summary.json", fit / "plan.json",
             fit / "verification.json", fit / "finals.jsonl", auxiliary / "plan.json", auxiliary / "observations.json")}
    if any(digest(fit / name) != proof["evidence_sha256"].get(name) for name in ("plan.json", "finals.jsonl")):
        raise ValueError("prior fit evidence changed")
    for name, expected in old["input_files"].items():
        if name.startswith("parent/"):
            path = parent / name.split("/", 1)[1]
            if digest(path) != expected:
                raise ValueError("qualified parent recordings or comparisons changed")
            bound[str(path)] = expected
    if any(pool.manifest.by_state[state].split != "train" for state in split["training_states"] + split["probe_test_states"]):
        raise ValueError("auxiliary diagnostics must retain the original training-only development population")
    schedule = []
    for seed in range(SPEC["epochs"]):
        for update in reciprocal_updates(3, seed=seed):
            examples = [{**row, "state": SPEC["training_indices"][row["state"]]} for row in update["examples"]]
            schedule.append({"examples": examples, "references": list(rehearsal_indices(len(schedule)))})
    if len(schedule) != SPEC["updates_per_arm"]:
        raise ValueError("unexpected shared fit schedule")
    selected = [index for index, cell in enumerate(old["fit_audit_cells"]) if cell["training_index"] in SPEC["training_indices"]]
    if len(selected) != 21:
        raise ValueError("the declared three sounds must retain all 21 comparisons")
    exposed = set(old["regression_state_ids"]) | {old["training"][index]["state_id"] for index in SPEC["training_indices"]}
    repeats = [index for index in selected if old["fit_audit_cells"][index]["input_view"] == 1]
    if len(repeats) != 6:
        raise ValueError("the fixed-phase repeats must cover both held-input playbacks for all three sounds")
    plan = {**old, "format": "shared_basic_sound_auxiliary_comparison_v1", "source": source_identity(old), "inputs_sha256": bound,
            "updates": schedule, "optimizer": {"learning_rate": SPEC["learning_rate"], "seed": SPEC["seed"], "gradient_norm_cap": 5.},
            "limits": LIMITS, "maximum_native_requests": 15360, "planned_maximum_native_requests": 14620,
            "comparison_spec": SPEC, "inward_policy": POLICY, "model_budget": COUNTS, "auxiliary_split": split["training_states"],
            "diagnostic_split": sorted(set(split["probe_test_states"]) - exposed), "selected_comparisons": selected,
            "repeat_comparisons": repeats, "comparison_fit": str(fit),
            "diagnostic_states_excluded_for_sound_exposure": sorted(set(split["probe_test_states"]) & exposed),
            "parent_fit_plan_id": canonical_sha256(old), "model_admission_allowed": False, "qualified_for_release": False}
    for key in list(plan):
        if key.startswith("expected_parameter_") or key in ("parameter_control_rehearsal_sha256", "require_trainable_input_updates"):
            del plan[key]
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", plan)
    return plan


def load_inputs(parent, bridge, cache, pool, fit, auxiliary, output):
    plan = read(output / "plan.json")
    if (plan["comparison_spec"] != SPEC or plan["limits"] != LIMITS or plan["source"] != source_identity(plan)
            or pool.pool_id != plan["pool_id"] or digest(cache) != plan["input_cache_sha256"]
            or any(digest(Path(path)) != expected for path, expected in plan["inputs_sha256"].items())):
        raise ValueError("frozen comparison code, inputs or allocation changed")
    if str(fit) != plan["comparison_fit"]:
        raise ValueError("matched settings-control fit path changed")
    inputs = Inputs(parent, bridge, cache, plan)
    inputs.comparison_fit = fit
    bank = training_bank(torch.load(cache, map_location="cpu", weights_only=True), pool)
    cached = {(row["state_id"], row["audio_sha256"]): row for row in bank.values()}
    rows = read(auxiliary / "observations.json")
    for row in rows:
        item = cached[row["state_id"], row["audio_sha256"]]
        if tensor_digest(item["features"]) != row["features_sha256"] or item["performance_id"] != row["performance_id"]:
            raise ValueError("auxiliary cached observations changed")
        row["features"] = item["features"]
    training = [row for row in rows if row["state_id"] in plan["auxiliary_split"]]
    if len(rows) != 1212 or len(training) != 777:
        raise ValueError("incomplete auxiliary three-view population")
    return inputs, training, rows


def snapshot(inputs, models, optimizers, trace, calibration):
    return {"plan_id": canonical_sha256(inputs.plan), "step": len(trace), "trace": copy.deepcopy(trace), "calibration": calibration,
            "models": {arm: copy.deepcopy(model.state_dict()) for arm, model in models.items()},
            "optimizers": {arm: copy.deepcopy(optimizer.state_dict()) for arm, optimizer in optimizers.items()}, "rng": capture_rng("cpu")}


def train(inputs, rows, store, *, output=None, saved=None, stop_after=None):
    random.seed(SPEC["seed"])
    np.random.seed(SPEC["seed"])
    torch.manual_seed(SPEC["seed"])
    models, optimizers = initialize(inputs)
    trace, calibration = [], None
    if saved is not None:
        if saved["plan_id"] != canonical_sha256(inputs.plan) or saved["step"] != len(saved["trace"]):
            raise ValueError("shared comparison checkpoint changed")
        for arm, model in models.items():
            model.load_state_dict(saved["models"][arm])
            model.require_frozen()
            optimizers[arm].load_state_dict(saved["optimizers"][arm])
        trace, calibration = copy.deepcopy(saved["trace"]), copy.deepcopy(saved["calibration"])
        restore_rng(saved["rng"], "cpu")
    charges = lines(output / "updates.jsonl") if output else []
    count = len(inputs.plan["updates"]) if stop_after is None else stop_after
    for index in range(len(trace), count):
        if output:
            store.check_limits()
            if len(charges) >= LIMITS["optimizer_attempts"]:
                raise ValueError("shared comparison optimizer attempt cap exhausted")
            charge = {"attempt": len(charges), "step": index + 1}
            append_durable(output / "updates.jsonl", charge)
            charges.append(charge)
        result, calibration = step(inputs, store, models, optimizers, rows, index, calibration)
        trace.append(result)
        if output:
            state = snapshot(inputs, models, optimizers, trace, calibration)
            atomic_torch_save(output / "checkpoint.pt", state)
            if index + 1 == 2 and not (output / "prefix-verification.json").exists():
                _models, replay = train(inputs, rows, SoundStore(output, inputs), stop_after=2)
                if not equal_tree(state, replay):
                    raise ValueError("the first two native updates did not replay exactly")
                restore_rng(state["rng"], "cpu")
                write_json(output / "prefix-verification.json", {"plan_id": state["plan_id"], "replayed_updates_per_arm": 2,
                           "models_adam_calibration_rng_exact": True, "native_requests_added": 0})
            write_json(output / "progress.json", {"step": index + 1, "planned_updates": count,
                       "sound_loss": {arm: result["arms"][arm]["sound_loss"] for arm in ARMS},
                       "accepted_updates": {arm: sum(row["arms"][arm]["acceptance"]["accepted"] for row in trace) for arm in ARMS}})
    return models, snapshot(inputs, models, optimizers, trace, calibration)


def collect(inputs, store, models):
    for index, cell in enumerate(inputs.plan["fit_audit_cells"]):
        store.initial(index)
        for arm, model in models.items():
            with torch.no_grad():
                prediction = inputs.predict(model, cell)
                patch = inputs.adapter.export(prediction["continuous"], prediction["categorical"])[0]
            key = index, arm
            if key not in store.finals:
                row = {**store.render(index, patch, "qualification", f"final:{arm}:{index:03d}"), "arm": arm}
                append_durable(store.output / "finals.jsonl", row)
                store.finals[key] = row
            if index in inputs.plan["repeat_comparisons"]:
                key = index, arm + "_repeat"
                if key not in store.finals:
                    row = {**store.render(index, patch, "qualification", f"repeat:{arm}:{index:03d}"), "arm": arm + "_repeat"}
                    append_durable(store.output / "finals.jsonl", row)
                    store.finals[key] = row
                if store.finals[key]["receipt"]["qa"]["audio_float32_sha256"] != store.finals[index, arm]["receipt"]["qa"]["audio_float32_sha256"]:
                    raise ValueError("fixed-phase endpoint repeat differs")


def reconstruction_result(inputs, store):
    selected, cells = inputs.plan["selected_comparisons"], inputs.plan["fit_audit_cells"]

    def scores(index, arm):
        return store.initial(index)["scores"] if arm == "initial" else store.finals[index, arm]["scores"]

    def means(indices, arm):
        return {metric: float(np.mean([scores(index, arm)[metric] for index in indices])) for metric in ("spectral", "mrstft")}

    result = {"three_sounds": {}, "six_groups": {}, "references": {}}
    finals = {(row["index"], row["arm"]): row for row in lines(inputs.comparison_fit / "finals.jsonl")}
    for training_index, midi in zip(SPEC["training_indices"], (36, 61, 83), strict=True):
        indices = [index for index in selected if cells[index]["training_index"] == training_index]
        group = {"midi": midi, "comparisons": len(indices), "means": {arm: means(indices, arm) for arm in ("initial", *ARMS)}, "arms": {}}
        for arm in ARMS:
            per_cell = all(scores(index, arm)["spectral"] <= .9 * scores(index, "initial")["spectral"]
                           and scores(index, arm)["mrstft"] < scores(index, "initial")["mrstft"] for index in indices)
            held = [index for index in indices if cells[index]["input_view"] == 1]
            beats_control = all(scores(index, arm)["spectral"] <= finals[index, "parameter"]["scores"]["spectral"] for index in held)
            group["arms"][arm] = {"each_comparison_improves": bool(per_cell), "held_input_beats_parameter_control": bool(beats_control),
                "learning_prerequisite_passed": bool(per_cell and beats_control and group["means"][arm]["spectral"] <= .75 * group["means"]["initial"]["spectral"])}
        result["three_sounds"][str(midi)] = group
    for view, role in enumerate(inputs.plan["training_view_roles"]):
        for transfer in (False, True):
            indices = [index for index, cell in enumerate(cells) if cell["training_index"] is not None and cell["input_view"] == view
                       and (cell["input_view"] != cell["playback_view"]) == transfer]
            group = {arm: means(indices, arm) for arm in ("initial", *ARMS)}
            control = {metric: float(np.mean([finals[index, "parameter"]["scores"][metric] for index in indices]))
                       for metric in ("spectral", "mrstft")}
            group["saved_parameter_control"] = control
            group["gates"] = {arm: bool(group[arm]["spectral"] <= .9 * group["initial"]["spectral"]
                and group[arm]["spectral"] <= control["spectral"] and group[arm]["mrstft"] < group["initial"]["mrstft"]) for arm in ARMS}
            result["six_groups"][f"{role}_{'transfer' if transfer else 'own'}"] = group
    references = [index for index, cell in enumerate(cells) if cell["input_view"] == cell["playback_view"] == 0]
    if len(references) != 32 or {cells[index]["recipe"]["state_id"] for index in references} != set(inputs.plan["regression_state_ids"]):
        raise ValueError("reference preservation panel is incomplete")
    for arm in ("initial", *ARMS):
        result["references"][arm] = {"means": means(references, arm), "mrstft_p90": float(np.quantile([scores(i, arm)["mrstft"] for i in references], .9))}
    earlier = inputs.plan["earlier_sound_reference"]
    for arm in ARMS:
        current = result["references"][arm]
        current["preserved"] = bool(all(current["means"][metric] <= earlier["means"][metric] for metric in ("spectral", "mrstft"))
                                    and current["mrstft_p90"] <= earlier["mrstft_p90"])
    result["selected_means"] = {arm: means(selected, arm) for arm in ("initial", *ARMS)}
    gain = all(result["selected_means"]["sound_auxiliary"][metric] <= .95 * result["selected_means"]["sound_only"][metric] for metric in ("spectral", "mrstft"))
    preserved = all(group["means"]["sound_auxiliary"][metric] <= 1.05 * group["means"]["sound_only"][metric]
                    for group in result["three_sounds"].values() for metric in ("spectral", "mrstft"))
    result["directional_auxiliary_benefit"] = bool(gain and preserved and result["references"]["sound_auxiliary"]["preserved"])
    saved = torch.load(store.output / "checkpoint.pt", map_location="cpu", weights_only=True)
    initial = inputs.model.state_dict()
    result["training_checks"] = {}
    for arm in ARMS:
        accepted = sum(row["arms"][arm]["acceptance"]["accepted"] for row in saved["trace"])
        branches = {branch: any(not torch.equal(value, saved["models"][arm][name]) for name, value in initial.items()
                               if name.startswith(f"correction.0.{branch}.")) for branch in ("timbre", "pitch")}
        access = {group: any(row["arms"][arm]["gradients"][group] > 0 for row in saved["trace"][1:])
                  for group in ("hidden", "continuous", "unison", "filter")}
        result["training_checks"][arm] = {"accepted_updates": accepted, "copied_branches_changed": branches,
            "later_gradient_access": access, "pass": bool(accepted >= inputs.plan["gates"]["minimum_native_accepted_updates"]
                                                           and all(branches.values()) and all(access.values()))}
    result["full_reconstruction_gates"] = {arm: bool(all(group["gates"][arm] for group in result["six_groups"].values())
        and result["references"][arm]["preserved"] and result["training_checks"][arm]["pass"]) for arm in ARMS}
    result["model_admitted"] = False
    return result


def execute(inputs, rows, output, *, resume=False):
    check = read(output / "software-check.json")
    if check["plan_id"] != canonical_sha256(inputs.plan) or not check["passed"]:
        raise ValueError("the real-initializer software check must pass before native execution")
    charge_execution(inputs, output, resume)
    monitor = MemoryMonitor()
    monitor.start()
    native = PairedSession(output, inputs.plan, inputs.codec, monitor)
    store = SoundStore(output, inputs, native)
    failure, result = None, None
    try:
        saved = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True) if resume and (output / "checkpoint.pt").exists() else None
        models, _saved = train(inputs, rows, store, output=output, saved=saved)
        collect(inputs, store, models)
        result = reconstruction_result(inputs, store)
        write_json(output / "result.json", result)
        for arm, model in models.items():
            atomic_torch_save(output / f"diagnostic-{arm}.pt", export_inward(model, inputs.frontend, inputs.codec, inputs.plan["pool_id"]))
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        monitor.close()
        elapsed = time() - native.started
        requests = sum(row["event"] == "attempt" for row in native.events)
        archive = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
        within = bool(monitor.peak_bytes and monitor.peak_bytes <= LIMITS["process_tree_bytes"] and not monitor.errors
                      and elapsed <= LIMITS["wall_seconds"] and archive <= LIMITS["archive_bytes"] and requests <= inputs.plan["maximum_native_requests"])
        saved = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True) if (output / "checkpoint.pt").exists() else None
        write_json(output / "summary.json", {"plan_id": canonical_sha256(inputs.plan), "complete": failure is None and result is not None,
            "failure": failure, "within_resource_limits": within, "elapsed_seconds": elapsed, "native_attempts": requests,
            "committed_updates_per_arm": saved["step"] if saved else 0, "peak_process_tree_rss_bytes": monitor.peak_bytes,
            "rss_sampling_errors": monitor.errors, "archive_bytes_before_summary": archive, "qualified_for_release": False, "test_split_used": False})
    return read(output / "summary.json")


def verify(inputs, rows, all_rows, output):
    started, monitor = time(), MemoryMonitor()
    monitor.start()
    try:
        report = verify_evidence(inputs, rows, all_rows, output)
    finally:
        monitor.close()
    elapsed = time() - started
    if elapsed > 1200 or not monitor.peak_bytes or monitor.peak_bytes > LIMITS["process_tree_bytes"] or monitor.errors:
        raise ValueError("offline audit exceeded twenty minutes or 4 GiB")
    report["resources"] = {"elapsed_seconds": elapsed, "peak_process_tree_rss_bytes": monitor.peak_bytes, "rss_sampling_errors": monitor.errors}
    write_json(output / "verification.json", report)
    return report


def verify_evidence(inputs, rows, all_rows, output):
    summary = read(output / "summary.json")
    if not summary["complete"] or summary["failure"] or not summary["within_resource_limits"]:
        raise ValueError("cannot verify an incomplete or out-of-budget comparison")
    scientific = {"plan.json", "software-check.json", "run.json", "executions.jsonl", "renders.jsonl", "probes.jsonl", "updates.jsonl",
                  "checkpoint.pt", "checkpoint.previous.pt", "finals.jsonl", "result.json", "summary.json", "prefix-verification.json",
                  *(f"diagnostic-{arm}.pt" for arm in ARMS)}
    evidence = {path.relative_to(output).as_posix(): digest(path) for path in output.rglob("*") if path.is_file()
                and (path.name in scientific or path.relative_to(output).parts[0] in ("source_audio", "comparison_audio", "interruption"))}
    store = SoundStore(output, inputs)
    saved = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
    models, replay = train(inputs, rows, store)
    if not equal_tree(saved, replay) or saved["step"] != SPEC["updates_per_arm"]:
        raise ValueError("shared model/Adam/calibration/RNG trajectory did not replay exactly")
    for arm in ARMS:
        cold, _frontend, _codec = restore_direct(torch.load(output / f"diagnostic-{arm}.pt", map_location="cpu", weights_only=True))
        if not equal_tree(cold.state_dict(), models[arm].state_dict()):
            raise ValueError("cold endpoint export differs from the reconstructed model")
    expected_finals = {(i, arm) for i in range(len(inputs.plan["fit_audit_cells"])) for arm in ARMS}
    expected_finals |= {(i, arm + "_repeat") for i in inputs.plan["repeat_comparisons"] for arm in ARMS}
    if set(store.finals) != expected_finals or store.used_probes != set(store.probes):
        raise ValueError("endpoint or training-probe evidence coverage changed")
    rescored = 0
    for index, cell in enumerate(inputs.plan["fit_audit_cells"]):
        source, _metadata = store.source(index)
        for arm in ("initial", *ARMS, *(arm + "_repeat" for arm in ARMS if index in inputs.plan["repeat_comparisons"])):
            row = store.initial(index) if arm == "initial" else store.finals[index, arm]
            patch = PilotTarget.from_dict(row["candidate"])
            require_request(inputs.codec, inputs.plan, patch, Performance.from_dict(cell["recipe"]["performances"][cell["playback_view"]]), row["receipt"])
            if rescore_waveform(output, row["receipt"], source, inputs.codec.contract.tail_seconds) != row["scores"]:
                raise ValueError("retained endpoint audio does not reproduce its score")
            if arm != "initial":
                model_arm = arm.removesuffix("_repeat")
                with torch.no_grad():
                    prediction = inputs.predict(models[model_arm], cell)
                if inputs.adapter.export(prediction["continuous"], prediction["categorical"])[0] != patch:
                    raise ValueError("endpoint patch differs from the replayed shared model")
            rescored += 1
    for index in inputs.plan["repeat_comparisons"]:
        for arm in ARMS:
            if store.finals[index, arm]["receipt"]["qa"]["audio_float32_sha256"] != store.finals[index, arm + "_repeat"]["receipt"]["qa"]["audio_float32_sha256"]:
                raise ValueError("saved fixed-phase endpoint repeat differs")
    attempts, outcomes, recovered = native_ledger(lines(output / "renders.jsonl"), inputs.plan["expected_renderer_identity"])
    for row in store.probes.values():
        query, receipt = row["query"], row["receipt"]
        arm, observation = query["observation"].rsplit("-", 1)
        cells = inputs.observations(inputs.plan["updates"][query["step"] - 1])
        if arm not in ARMS or query["index"] not in cells[int(observation)]:
            raise ValueError("training probe is outside the declared update")
        cell = inputs.plan["fit_audit_cells"][query["index"]]
        require_request(inputs.codec, inputs.plan, PilotTarget.from_dict(query["candidate"]),
                        Performance.from_dict(cell["recipe"]["performances"][cell["playback_view"]]), receipt)
        if receipt != outcomes[receipt["phase"], receipt["attempt"]]:
            raise ValueError("training measurement differs from its native receipt")
    charges = lines(output / "updates.jsonl")
    charged_steps = [row["step"] for row in charges]
    if (len(charges) > LIMITS["optimizer_attempts"] or [row["attempt"] for row in charges] != list(range(len(charges)))
            or sorted(set(charged_steps)) != list(range(1, SPEC["updates_per_arm"] + 1))
            or charged_steps != sorted(charged_steps) or len(charges) - len(set(charged_steps)) > len(recovered)):
        raise ValueError("optimizer attempts do not match the completed trajectory and recorded recovery")
    result = reconstruction_result(inputs, store)
    if result != read(output / "result.json") or len(attempts) != summary["native_attempts"]:
        raise ValueError("comparison result or native accounting changed")
    diagnostics = probe_models(inputs, models, all_rows)
    report = {"plan_id": canonical_sha256(inputs.plan), "verified": True, "complete": True,
              "models_adam_calibration_rng_replay_exact": True, "endpoint_waveforms_rescored": rescored,
              "recovered_timeouts": recovered, "evidence_sha256": evidence, "result": result, "secondary_probes": diagnostics,
              "qualified_for_release": False, "test_split_used": False}
    return report


def software_check(inputs, rows, output):
    """Exercise the real initializer with an analytic test cost, adding no native evidence."""
    started, original = time(), tensor_digest(inputs.model.state_dict())
    monitor = MemoryMonitor()
    monitor.start()

    class AnalyticStore:
        def probe(self, _step, _observation, index, patch):
            values = torch.tensor(patch.continuous, dtype=torch.float32)
            return 1. + float((values - (.2 + .01 * (index % 4))).square().mean()) + .02 * sum(patch.categorical)

    try:
        inputs.validate()
        _models, first = train(inputs, rows, AnalyticStore(), stop_after=1)
        if not equal_tree(first["models"][ARMS[0]], first["models"][ARMS[1]]) or not equal_tree(first["optimizers"][ARMS[0]], first["optimizers"][ARMS[1]]):
            raise ValueError("zero-auxiliary first update does not preserve matched model/Adam identity")
        models, expected = train(inputs, rows, AnalyticStore(), stop_after=2)
        _models, recovered = train(inputs, rows, AnalyticStore(), saved=first, stop_after=2)
        if not equal_tree(expected, recovered) or tensor_digest(inputs.model.state_dict()) != original:
            raise ValueError("the real-initializer software recovery changed its trajectory or source")
        exports = 0
        for model in models.values():
            cold, _frontend, _codec = restore_direct(export_inward(model, inputs.frontend, inputs.codec, inputs.plan["pool_id"]))
            for index in inputs.plan["selected_comparisons"]:
                cell = inputs.plan["fit_audit_cells"][index]
                with torch.no_grad():
                    a, b = inputs.predict(model, cell), inputs.predict(cold, cell)
                if inputs.adapter.export(a["continuous"], a["categorical"]) != inputs.adapter.export(b["continuous"], b["categorical"]):
                    raise ValueError("cold exports changed the software-check predictions")
                exports += 1
    finally:
        monitor.close()
    elapsed = time() - started
    if elapsed > 600 or not monitor.peak_bytes or monitor.peak_bytes > LIMITS["process_tree_bytes"] or monitor.errors:
        raise ValueError("software check exceeded ten minutes or 4 GiB")
    report = {"plan_id": canonical_sha256(inputs.plan), "passed": True, "native_requests": 0, "experimental_updates": 0,
              "simulated_optimizer_applications_per_arm": 4, "cold_export_comparisons": exports,
              "models_adam_calibration_rng_recovery_exact": True, "calibration": expected["calibration"],
              "elapsed_seconds": elapsed, "peak_process_tree_rss_bytes": monitor.peak_bytes, "rss_sampling_errors": monitor.errors}
    write_json(output / "software-check.json", report)
    return report


def probe_models(inputs, models, rows):
    from reports.scripts.run_branch_adversary_screen import score
    training = set(inputs.plan["auxiliary_split"])
    diagnostic = set(inputs.plan["diagnostic_split"])
    selected = [row for row in rows if row["state_id"] in training | diagnostic]
    active = torch.tensor([inputs.codec.continuous_mask(PilotTarget(tuple(row["continuous"]), tuple(row["categorical"]))) for row in selected])
    return {arm: score(encode(model.correction[0].timbre, selected), selected, training, active) for arm, model in models.items()}

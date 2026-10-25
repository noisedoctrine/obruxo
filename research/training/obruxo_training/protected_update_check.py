"""Check protected proposals at saved states without advancing a new trajectory."""

from pathlib import Path
from time import perf_counter

import torch
from torch.nn.utils import parameters_to_vector

from obruxo_data.hashing import canonical_sha256, sha256_bytes

from .audit import write_json
from .basic_direct import REHEARSAL
from .basic_direct_fit import SoundStore, initialize, step
from .basic_direct_fit_data import Inputs
from .benchmark import MemoryMonitor
from .category_scale_refresh import cpu_settings
from .codec import PilotTarget
from .conditioned_bridge import read
from .native_fit import digest
from .native_gradient import NativeGradient, measured_loss
from .paired_native_audit import equal_tree
from .protected_updates import SPEC, loss_gradients, select_proposal
from .trainer import capture_rng
from .update_diagnosis import LIMITS, make_plan as diagnosis_plan


def sound_losses(model, inputs, observations, estimates, temperature):
    result = []
    for cells, data in zip(observations, estimates, strict=True):
        prediction = inputs.predict(model, inputs.plan["fit_audit_cells"][cells[0]])
        encoded = inputs.adapter(prediction["continuous"], prediction["categorical"], temperature=temperature)
        estimate = NativeGradient(PilotTarget.from_dict(data["anchor"]), data["value"], tuple(data["continuous"]), tuple(map(tuple, data["categorical"])), data["queries"])
        result.append(measured_loss(prediction["continuous"], (encoded[:, 9:12], encoded[:, 12:]), estimate))
    return result


def inspect_proposals(inputs, source):
    saved = torch.load(source / "checkpoint.pt", map_location="cpu", weights_only=True)
    if saved["plan_id"] != canonical_sha256(inputs.plan) or saved["step"] != len(saved["trace"]) or saved["step"] != len(inputs.plan["updates"]):
        raise ValueError("protected check needs the complete matching trajectory")
    models, optimizers = initialize(inputs)
    store, rows = SoundStore(source, inputs), []
    for index, recorded in enumerate(saved["trace"]):
        model = models["spectral"]
        before = parameters_to_vector(model.correction.parameters()).detach().clone()
        observations = inputs.observations(recorded["update"])
        rng = capture_rng("cpu")
        losses = sound_losses(model, inputs, observations, recorded["estimates"], recorded["temperature"])
        gradients = loss_gradients(losses, model)
        result = step(models, optimizers, inputs, store, index)
        if {"step": index + 1, "update": inputs.plan["updates"][index], **result} != {key: value for key, value in recorded.items() if key != "attempt"}:
            raise ValueError("protected inspection changed historical updates")
        if not equal_tree(rng, capture_rng("cpu")):
            raise ValueError("protected inspection changed randomness")
        proposal = parameters_to_vector(model.correction.parameters()).detach() - before
        incoming = {name: value.clone() for name, value in model.state_dict().items()}
        _candidate, protected = select_proposal(model, before, proposal, gradients, inputs, observations, recorded["estimates"])
        if not equal_tree(incoming, model.state_dict()) or not equal_tree(rng, capture_rng("cpu")):
            raise ValueError("temporary protected proposal changed historical state")
        weights = list(REHEARSAL["paired_weights"] + REHEARSAL["reference_weights"])
        for observation in (2, 3):
            baseline = inputs.cases[observations[observation][0]]["rows"]["initial"]["scores"]["spectral"]
            weights[observation] *= 2 if float(losses[observation].detach()) > float(torch.tensor(baseline)) else 1
        original = float(sum(weight * torch.dot(gradient.double(), proposal.double()) for weight, gradient in zip(weights, gradients, strict=True)))
        protected["weighted_output_estimated_change"] = sum(weight * row["estimated_sound_change"] for weight, row in zip(weights, protected["changes"], strict=True))
        rows.append({"step": index + 1, "original_weight_linear_change": original, **protected})
    if (not all(equal_tree(model.state_dict(), saved["models"][arm]) and equal_tree(optimizers[arm].state_dict(), saved["optimizers"][arm]) for arm, model in models.items())
            or not equal_tree(capture_rng("cpu"), saved["rng"]) or store.used_probes != set(store.probes)):
        raise ValueError("protected inspection failed exact model/Adam/RNG/probe replay")
    original_gain = sum(max(0., -row["original_weight_linear_change"]) for row in rows)
    protected_gain = -sum(row["weighted_output_estimated_change"] for row in rows)
    return {"rows": rows, "model_adam_rng_replay_exact": True, "scalar_probes_reused": len(store.used_probes),
            "replayed_updates_per_arm": saved["step"], "temporary_protected_proposals": len(rows),
            "accepted_proposals": sum(row["accepted"] for row in rows), "original_weight_linear_gain": original_gain,
            "protected_output_estimated_gain": protected_gain, "estimated_gain_fraction": protected_gain / original_gain if original_gain else 0.}


def make_plan(sources, parent, bridge, features, diagnosis):
    prior = diagnosis_plan(sources, parent, bridge, features)
    result = read(diagnosis / "summary.json")
    if read(diagnosis / "plan.json") != prior or result["plan_id"] != canonical_sha256(prior) or not result["within_resource_limits"] or not all(row["model_adam_rng_replay_exact"] for row in result["runs"]):
        raise ValueError("protected check requires the unchanged complete update diagnosis")
    source = dict(prior["source"])
    root = Path(__file__).parents[1]
    for name in ("protected_updates.py", "protected_update_check.py", "reports/scripts/check_protected_updates.py"):
        path = root / name if "/" in name else Path(__file__).with_name(name)
        source[name] = sha256_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
    return {"format": "saved_state_protected_update_check_v1", "diagnosis_plan_id": canonical_sha256(prior), "diagnosis_summary_sha256": digest(diagnosis / "summary.json"),
            "runs": prior["runs"], "source": source, "policy": SPEC, "limits": LIMITS,
            "gates": {"minimum_accepted_proposals": 36, "minimum_estimated_gain_fraction": .25, "all_four_local_estimates_nonincreasing": True},
            "native_requests": 0, "experimental_training_updates": 0, "replayed_optimizer_applications": 96,
            "temporary_protected_proposals": 48, "new_trajectory_qualified": False, "native_sound_acceptance_qualified": False,
            "test_split_used": False, "qualified_for_release": False}


def run(sources, parent, bridge, features, diagnosis, output, *, execute=False):
    with cpu_settings():
        plan = make_plan(sources, parent, bridge, features, diagnosis)
        if not execute:
            output.mkdir(parents=True, exist_ok=False)
            write_json(output / "plan.json", plan)
            return {"plan_id": canonical_sha256(plan), "native_requests": 0, "experimental_training_updates": 0}
        if read(output / "plan.json") != plan or (output / "summary.json").exists():
            raise ValueError("preserve frozen protected-update check and completed results")
        started, monitor = perf_counter(), MemoryMonitor()
        monitor.start()
        try:
            result = inspect_proposals(Inputs(parent, bridge, features, read(sources[0] / "plan.json")), sources[0])
        finally:
            monitor.close()
        resources = {"elapsed_seconds": perf_counter() - started, "peak_process_tree_rss_bytes": monitor.peak_bytes, "rss_sampling_errors": monitor.errors}
        within = bool(resources["elapsed_seconds"] < LIMITS["wall_seconds"] and monitor.peak_bytes and monitor.peak_bytes <= LIMITS["process_tree_bytes"] and not monitor.errors)
        useful = result["accepted_proposals"] >= plan["gates"]["minimum_accepted_proposals"] and result["estimated_gain_fraction"] >= plan["gates"]["minimum_estimated_gain_fraction"]
        result.update(plan_id=canonical_sha256(plan), resources=resources, within_resource_limits=within, useful_local_proposals=useful,
                      native_requests=0, experimental_training_updates=0, new_trajectory_qualified=False, native_sound_acceptance_qualified=False,
                      test_split_used=False, qualified_for_release=False)
        write_json(output / "summary.json", result)
        if not within or sum(path.stat().st_size for path in output.rglob("*") if path.is_file()) > LIMITS["archive_bytes"] or make_plan(sources, parent, bridge, features, diagnosis) != plan:
            raise ValueError("protected check resource limits or source preservation failed")
        return {key: value for key, value in result.items() if key != "rows"}

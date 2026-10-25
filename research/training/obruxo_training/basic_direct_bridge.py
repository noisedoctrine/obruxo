"""Qualify direct basic controls and bind real reference sounds for rehearsal."""

import copy
import json
from pathlib import Path
from time import perf_counter

import torch

from obruxo_data.hashing import canonical_sha256, sha256_bytes
from obruxo_data.midi import Performance
from obruxo_data.render.qa import audio_float32_sha256

from . import category_scale_refresh as refresh
from .audit import write_json
from .basic_direct import COUNTS, REHEARSAL, SPEC, BasicDirect, export_direct, reference_loss, rehearsal_indices, restore_direct
from .benchmark import MemoryMonitor
from .checkpoints import atomic_torch_save
from .codec import PilotTarget
from .conditioned_bridge import inputs as original_inputs, read, training_bank
from .features import tensor_digest
from .inference_profile import artifact_rendering, prediction_preset
from .model import batch_features
from .native_fit import digest
from .native_gradient import NativeGradient, measured_loss
from .native_inverse import native_reconstruction_loss
from .paired_native_audit import equal_tree
from .pilot import PilotPool
from .production_controls import ProductionControlAdapter
from .production_direction import raw_prediction
from .reproducibility import FULL_PHASE_CONTROL
from .trainer import capture_rng, restore_model, restore_rng
from .unison_choice import SCALES
from .unison_fit_data import require_request

LIMITS = {"wall_seconds": 300, "process_tree_bytes": 4 * 1024**3, "archive_bytes": 64 * 1024**2}


def reference_manifest(recent, baseline_parent):
    proof, plan = read(recent / "verification.json"), read(recent / "plan.json")
    if (not proof["verified"] or not proof["offline_training_replay_exact"] or not proof["complete"]
            or not proof["native_receipts_resolved"] or proof["comparison_waveforms_rescored"] != 480
            or proof["committed_updates_per_arm"] != 192 or proof["matched_comparison_passed"] or proof["test_split_used"]):
        raise ValueError("direct bridge requires the audited unison fit and its preservation failure")
    files = {f"recent/{name}": digest(recent / name) for name in proof["evidence_sha256"]}
    if {name: files[f"recent/{name}"] for name in proof["evidence_sha256"]} != proof["evidence_sha256"]:
        raise ValueError("audited unison evidence changed")
    cases = read(baseline_parent / "fit_cases.json")
    if digest(baseline_parent / "fit_cases.json") != plan["input_files"]["parent/fit_cases.json"]:
        raise ValueError("accepted initial reference predictions changed")
    files["baseline_parent/fit_cases.json"] = digest(baseline_parent / "fit_cases.json")
    references = []
    for cell, case in zip(plan["fit_audit_cells"], cases, strict=True):
        if cell["input_view"] != 0 or cell["playback_view"] != 0:
            continue
        recipe, initial = cell["recipe"], case["rows"]["initial"]
        source_sha = recipe["source_audio_sha256"][0]
        source_file = f"source_audio/{source_sha}.pt"
        wave_file = f"comparison_audio/{initial['receipt']['qa']['audio_float32_sha256']}.pt"
        if (initial["receipt"]["retained_audio"]["file"] != wave_file or digest(recent / wave_file) != initial["receipt"]["retained_audio"]["sha256"]):
            raise ValueError("accepted initial reference waveform changed")
        files[f"recent/{source_file}"] = digest(recent / source_file)
        files[f"recent/{wave_file}"] = digest(recent / wave_file)
        references.append({"state_id": recipe["state_id"], "recipe_id": recipe["source_recipe_ids"][0],
            "features_sha256": recipe["source_feature_sha256"][0], "source_audio_sha256": source_sha,
            "source_file": source_file, "initial_wave_file": wave_file, "initial_scores": initial["scores"],
            "initial_patch": initial["candidate"], "performance": recipe["performances"][0], "source_target": recipe["target"],
            "source_receipt": case["source_receipt"], "initial_receipt": initial["receipt"]})
    references.sort(key=lambda row: row["state_id"])
    if len(references) != 32 or {row["state_id"] for row in references} != set(plan["regression_state_ids"]):
        raise ValueError("direct rehearsal requires exactly the original 32 training recordings")
    files["recent/verification.json"] = digest(recent / "verification.json")
    return references, files


def make_plan(parent, initial, cache, direction, pool_path, recent, baseline_parent):
    prior = original_inputs(parent, initial, cache, direction, pool_path)
    prior.pop("new_optimizer_updates")
    if read(direction / "plan.json")["adapter"]["fixed_scales"] != list(SCALES):
        raise ValueError("direct bridge changed its qualified category calibration")
    references, files = reference_manifest(recent, baseline_parent)
    recent_plan = read(recent / "plan.json")
    native = read(direction / "verification.json")
    if (recent_plan["expected_renderer_identity"] != native["renderer_identity"]
            or recent_plan["input_cache_sha256"] != prior["input_files"]["features"]):
        raise ValueError("direct rehearsal and gradient evidence use different renderers or recordings")
    names = ("basic_direct.py", "basic_direct_bridge.py", "unison_choice.py", "unison_fit_data.py", "paired_native_audit.py", "native_inverse.py")
    source = {**prior["source"], **{name: sha256_bytes(Path(__file__).with_name(name).read_bytes().replace(b"\r\n", b"\n")) for name in names}}
    script = Path(__file__).parents[1] / "reports/scripts/check_basic_direct_bridge.py"
    source["reports/scripts/check_basic_direct_bridge.py"] = sha256_bytes(script.read_bytes().replace(b"\r\n", b"\n"))
    return {**prior, "format": "basic_direct_reference_bridge_v1", "source": source, "limits": LIMITS, "model_budget": COUNTS,
            "model_spec": SPEC, "initialization": "zero_direct_control_and_score_offsets_v1", "reference_files": files,
            "expected_renderer_identity": native["renderer_identity"], "render_profile": recent_plan["render_profile"],
            "render_phase_controls": recent_plan["render_phase_controls"],
            "reference_manifest": references, "rehearsal": REHEARSAL, "new_native_requests": 0, "experimental_training_updates": 0,
            "simulation": {"optimizer": "Adam", "learning_rate": .001, "steps": 2, "replayed_steps": 1,
                           "observations_per_step": 4, "maximum_software_cost_evaluations": 266}}


def gradients(model):
    model.require_frozen()
    groups = dict.fromkeys(("hidden", "continuous", "unison", "filter"), 0.)
    for name, parameter in model.correction.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise ValueError("missing or nonfinite direct basic gradient")
        if name.startswith("0."):
            groups["hidden"] += float(parameter.grad.double().abs().sum())
        else:
            for group, start, end in (("continuous", 0, 9), ("unison", 9, 12), ("filter", 12, 14)):
                groups[group] += float(parameter.grad[start:end].double().abs().sum())
    return groups


def gradient_access(model, features, direction):
    before, rng = tensor_digest(model.state_dict()), torch.get_rng_state().clone()
    model.zero_grad(set_to_none=True)
    inputs = batch_features([{"features": features}])
    if raw_prediction(model.base(inputs)) != direction["raw_prediction"]:
        raise ValueError("native gradient has a stale base prediction")
    prediction = model(inputs)
    adapter = ProductionControlAdapter(scales=SPEC["adapter_scales"])
    if adapter.export(prediction["continuous"], prediction["categorical"])[0].to_dict() != direction["production_patch"]:
        raise ValueError("native gradient has a stale direct-control anchor")
    saved = direction["estimate"]
    estimate = NativeGradient(PilotTarget.from_dict(saved["anchor"]), saved["value"], tuple(saved["continuous"]),
                              tuple(tuple(row) for row in saved["categorical"]), saved["queries"])
    encoded = adapter(prediction["continuous"], prediction["categorical"], temperature=1., sample=True)
    choices = (encoded[:, 9:12], encoded[:, 12:])
    bias = model.correction[2].bias
    continuous = []
    for index, coefficient in enumerate(estimate.continuous):
        if abs(coefficient) <= 1e-7:
            continuous.append(None)
            continue
        value = torch.autograd.grad(prediction["continuous"][0, index] * coefficient, bias, retain_graph=True)[0]
        norm = float(value.double().abs().sum())
        if not torch.isfinite(value).all() or norm <= 0:
            raise ValueError("responding continuous sound cost cannot reach its direct correction")
        continuous.append(norm)
    categories = []
    for head, costs in enumerate(estimate.categorical):
        if max(costs) - min(costs) <= 1e-7:
            categories.append(None)
            continue
        component = (choices[head] * choices[head].new_tensor([costs])).sum()
        value = torch.autograd.grad(component, bias, retain_graph=True)[0]
        start, end = (9, 12) if head == 0 else (12, 14)
        norm = float(value[start:end].double().abs().sum())
        if not torch.isfinite(value).all() or norm <= 0:
            raise ValueError("responding category sound cost cannot reach its direct correction")
        categories.append(norm)
    measured_loss(prediction["continuous"], choices, estimate).backward()
    groups = gradients(model)
    if groups["hidden"] != 0 or tensor_digest(model.state_dict()) != before or not torch.equal(rng, torch.get_rng_state()):
        raise ValueError("direct initialization gradient path changed weights, RNG or the hidden layer")
    return {"continuous_bias_gradient_l1": continuous, "category_bias_gradient_l1": categories, "gradient_groups": groups,
            "frozen_gradients_absent": True, "weights_and_rng_unchanged": True}


@torch.no_grad()
def preserve(base, model, cold, codec, bank, *, initial):
    adapter = ProductionControlAdapter(codec, scales=SPEC["adapter_scales"])
    old = ProductionControlAdapter(codec, scales=SCALES)
    rows = []
    for recipe, row in sorted(bank.items()):
        features = batch_features([{"features": row["features"]}])
        baseline, prediction, restored = [network(features) for network in (base, model, cold)]
        if not torch.equal(prediction["continuous"], restored["continuous"]) or any(
                not torch.equal(a, b) for a, b in zip(prediction["categorical"], restored["categorical"], strict=True)):
            raise ValueError("cold direct export changed predictions")
        if not torch.isfinite(prediction["continuous"]).all() or (prediction["continuous"] < 0).any() or (prediction["continuous"] > 1).any():
            raise ValueError("direct control correction leaves the legal normalized interval")
        if initial:
            if not torch.equal(baseline["continuous"], prediction["continuous"]):
                raise ValueError("direct initialization changed continuous values")
            for head, scale in enumerate(SCALES):
                for temperature in (1., .5):
                    if not torch.equal((baseline["categorical"][head] / scale / temperature).softmax(-1),
                                       (prediction["categorical"][head] / temperature).softmax(-1)):
                        raise ValueError("direct initialization changed calibrated probabilities")
        targets = [current.export(value["continuous"], value["categorical"])[0]
                   for current, value in ((old, baseline), (adapter, prediction), (adapter, restored))]
        presets = [prediction_preset(network, codec, target) for network, target in zip((base, model, cold), targets, strict=True)]
        if presets[1] != presets[2] or (initial and (presets[0] != presets[1] or targets[0] != targets[1])):
            raise ValueError("starting or cold direct preset changed")
        rows.append({"recipe_id": recipe, "state_id": row["state_id"], "features_sha256": tensor_digest(row["features"]),
                     "audio_sha256": row["audio_sha256"], "preset_sha256": sha256_bytes(presets[1])})
    return rows


def simulate(model, frontend, codec, pool_id, features, output):
    original_rng = capture_rng("cpu")
    trial = copy.deepcopy(model).train()
    trial.zero_grad(set_to_none=True)
    optimizer = torch.optim.Adam(trial.correction.parameters(), lr=.001)
    adapter = ProductionControlAdapter(codec, scales=SPEC["adapter_scales"])
    calls = 0
    def cost(index, candidate):
        nonlocal calls
        calls += 1
        return (sum((value - (.2 + .05 * index))**2 for value in candidate.continuous)
                + .1 * (candidate.categorical[0] - (index + 1) % 3)**2 + .15 * (candidate.categorical[1] - index % 2)**2)
    baselines = []
    with torch.no_grad():
        for index in (2, 3):
            value = trial(batch_features([{"features": features[index]}]))
            baselines.append(cost(index, adapter.export(value["continuous"], value["categorical"])[0]))
    def step(network, current):
        current.zero_grad(set_to_none=True)
        losses = []
        for index, values in enumerate(features):
            prediction = network(batch_features([{"features": values}]))
            loss, _estimates = native_reconstruction_loss(prediction, adapter, lambda _i, candidate: cost(index, candidate))
            losses.append(loss if index < 2 else reference_loss(loss, baselines[index - 2]))
        loss = sum(weight * value for weight, value in zip((.25, .5, .125, .125), losses, strict=True))
        loss.backward()
        groups = gradients(network)
        current.step()
        network.require_frozen()
        return {"loss": float(loss.detach()), "gradient_groups": groups}
    try:
        first = step(trial, optimizer)
        atomic_torch_save(output / "simulation-checkpoint.pt", {"artifact": export_direct(trial, frontend, codec, pool_id),
            "optimizer": copy.deepcopy(optimizer.state_dict()), "rng": capture_rng("cpu")})
        second = step(trial, optimizer)
        expected_rng = capture_rng("cpu")
        snapshot = torch.load(output / "simulation-checkpoint.pt", map_location="cpu", weights_only=True)
        restored, _front, _codec = restore_direct(snapshot["artifact"])
        current = torch.optim.Adam(restored.correction.parameters(), lr=.001)
        current.load_state_dict(snapshot["optimizer"])
        restore_rng(snapshot["rng"], "cpu")
        repeated = step(restored.train(), current)
        if (second != repeated or not equal_tree(trial.state_dict(), restored.state_dict())
                or not equal_tree(optimizer.state_dict(), current.state_dict()) or not equal_tree(expected_rng, capture_rng("cpu"))
                or not all(value > 0 for value in second["gradient_groups"].values()) or calls > 266):
            raise ValueError("software direct model/Adam/RNG recovery or later gradient access failed")
        atomic_torch_save(output / "simulation-model.pt", export_direct(trial, frontend, codec, pool_id))
        cold, _front, _codec = restore_direct(torch.load(output / "simulation-model.pt", map_location="cpu", weights_only=True))
        return trial.eval(), cold, {"first": first, "second": second, "model_adam_rng_recovery_exact": True,
                                   "simulated_optimizer_step_applications": 3, "software_cost_evaluations": calls,
                                   "experimental_training_updates": 0, "native_requests": 0}
    finally:
        restore_rng(original_rng, "cpu")


def validate_references(plan, recent, model, codec, bank):
    adapter = ProductionControlAdapter(codec, scales=SPEC["adapter_scales"])
    for reference in plan["reference_manifest"]:
        row = bank[reference["recipe_id"]]
        if (row["state_id"] != reference["state_id"] or row["audio_sha256"] != reference["source_audio_sha256"]
                or tensor_digest(row["features"]) != reference["features_sha256"]):
            raise ValueError("direct reference input differs from its accepted training recording")
        value = torch.load(recent / reference["source_file"], map_location="cpu", weights_only=True)["audio"]
        if value.dtype != torch.float32 or value.ndim != 2 or value.shape[1] != 2:
            raise ValueError("direct reference source is not stereo float32 audio")
        source = value.numpy()
        if audio_float32_sha256(source) != reference["source_audio_sha256"]:
            raise ValueError("direct reference source audio changed")
        with torch.no_grad():
            prediction = model(batch_features([{"features": row["features"]}]))
        target = adapter.export(prediction["continuous"], prediction["categorical"])[0]
        if target.to_dict() != reference["initial_patch"]:
            raise ValueError("direct reference baseline is not the accepted initializer")
        performance = Performance.from_dict(reference["performance"])
        require_request(codec, plan, target, performance, reference["initial_receipt"])
        require_request(codec, plan, PilotTarget.from_dict(reference["source_target"]), performance, reference["source_receipt"])
    counts = [0] * 32
    for step in range(48):
        for index in rehearsal_indices(step):
            counts[index] += 1
    if counts != [3] * 32:
        raise ValueError("reference schedule does not cover the original recordings evenly")
    return {"reference_records": 32, "schedule_check_updates": 48, "visits_per_reference": counts, "accepted_baselines_verified": True}


def run(parent, initial, cache, direction, pool_path, recent, baseline_parent, output, *, execute=False):
    started, monitor = perf_counter(), MemoryMonitor()
    monitor.start()
    try:
        plan = make_plan(parent, initial, cache, direction, pool_path, recent, baseline_parent)
        if not execute:
            output.mkdir(parents=True, exist_ok=False)
            write_json(output / "plan.json", plan)
        else:
            if read(output / "plan.json") != plan or (output / "summary.json").exists():
                raise ValueError("preserve the direct bridge or changed frozen inputs")
            with refresh.cpu_settings():
                saved = torch.load(initial, map_location="cpu", weights_only=True)
                base, frontend, codec = restore_model(saved)
                pool = PilotPool.load(pool_path)
                if (saved["training_pool_id"] != pool.pool_id or codec.to_dict() != pool.codec.to_dict()
                        or artifact_rendering(saved)["render_phase_controls"] != FULL_PHASE_CONTROL
                        or artifact_rendering(saved)["render_profile"]["preroll_frames"] != 32768):
                    raise ValueError("direct bridge changed its accepted scope or fixed-phase profile")
                bank = training_bank(torch.load(cache, map_location="cpu", weights_only=True), pool)
                model = BasicDirect(base, seed=plan["seed"]).eval()
                target = output / "model.pt"
                atomic_torch_save(target, export_direct(model, frontend, codec, pool.pool_id))
                initial_sha = digest(target)
                cold, _front, _codec = restore_direct(torch.load(target, map_location="cpu", weights_only=True))
                identities = preserve(base, model, cold, codec, bank, initial=True)
                reference_check = validate_references(plan, recent, model, codec, bank)
                descriptors, cases = read(direction / "plan.json")["cases"], read(direction / "cases.json")
                if len(cases) != 24 or len(descriptors) != 24:
                    raise ValueError("direct bridge requires all 24 native points")
                native = []
                for descriptor, case in zip(descriptors, cases, strict=True):
                    row = bank[descriptor["input_recipe_id"]]
                    if (any(descriptor[key] != case[key] for key in ("index", "role", "state_id"))
                            or row["state_id"] != descriptor["state_id"] or row["audio_sha256"] != descriptor["input_audio_sha256"]
                            or tensor_digest(row["features"]) != descriptor["input_features_sha256"]):
                        raise ValueError("direct native qualification input changed")
                    native.append({"index": case["index"], "role": case["role"], "state_id": case["state_id"],
                                   **gradient_access(model, row["features"], case["direction"])})
                features = [bank[descriptors[index]["input_recipe_id"]]["features"] for index in (0, 1)]
                features += [bank[plan["reference_manifest"][index]["recipe_id"]]["features"] for index in (0, 16)]
                trial, restored, recovery = simulate(model, frontend, codec, pool.pool_id, features, output)
                simulated = preserve(base, trial, restored, codec, bank, initial=False)
                if digest(target) != initial_sha or tensor_digest(model.state_dict()) != torch.load(target, map_location="cpu", weights_only=True)["model_sha256"]:
                    raise ValueError("direct qualification changed its untrained initializer")
            if make_plan(parent, initial, cache, direction, pool_path, recent, baseline_parent) != plan:
                raise ValueError("direct qualification changed parent evidence")
    finally:
        monitor.close()
    resources = {"elapsed_seconds": perf_counter() - started, "peak_process_tree_rss_bytes": monitor.peak_bytes, "rss_sampling_errors": monitor.errors}
    if resources["elapsed_seconds"] >= 300 or monitor.peak_bytes is None or monitor.peak_bytes > LIMITS["process_tree_bytes"] or monitor.errors:
        raise ValueError("direct qualification exceeded five minutes or 4 GiB")
    report = {"plan_id": canonical_sha256(plan), "resources": resources, "new_native_requests": 0, "experimental_training_updates": 0,
              "simulated_optimizer_step_applications": 3 if execute else 0, "qualified_for_release": False}
    if execute:
        report.update(initialization_gradient_reference_recovery_passed=True, parent_artifacts_unchanged=True, model_budget=COUNTS,
                      initialization_cases=len(identities), simulated_cold_cases=len(simulated), native_gradient_cases=len(native),
                      model_file_sha256=digest(target), model_sha256=tensor_digest(model.state_dict()), reference_check=reference_check,
                      initial_rows=identities, simulated_rows=simulated, native_rows=native, software_recovery=recovery)
    archive = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    if archive + 2 * len(json.dumps(report, indent=2).encode("utf-8")) + 1024**2 > LIMITS["archive_bytes"]:
        raise ValueError("direct qualification exceeded 64 MiB evidence")
    write_json(output / ("summary.json" if execute else "preparation.json"), report)
    return {key: value for key, value in report.items() if key not in ("initial_rows", "simulated_rows", "native_rows")}

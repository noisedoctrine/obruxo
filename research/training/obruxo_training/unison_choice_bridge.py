"""Qualify direct unison corrections against immutable real-render evidence."""

import copy
import json
from pathlib import Path
from time import perf_counter

import torch

from obruxo_data.hashing import canonical_sha256, sha256_bytes

from . import category_scale_refresh as refresh
from .audit import write_json
from .benchmark import MemoryMonitor
from .checkpoints import atomic_torch_save
from .conditioned_bridge import inputs as original_inputs, read, training_bank
from .features import tensor_digest
from .inference_profile import artifact_rendering, model_rendering, prediction_preset
from .model import batch_features
from .native_fit import digest
from .native_gradient import NativeGradient, measured_loss
from .paired_native_audit import equal_tree
from .pilot import PilotPool
from .production_controls import ProductionControlAdapter
from .production_direction import raw_prediction
from .reproducibility import FULL_PHASE_CONTROL
from .trainer import capture_rng, restore_model, restore_rng
from .unison_choice import COUNTS, SCALES, SPEC, UnisonChoice, export_unison, measured_choice_loss, restore_unison

LIMITS = {"wall_seconds": 300, "process_tree_bytes": 4 * 1024**3, "archive_bytes": 64 * 1024**2}


def make_plan(parent, initial, cache, direction, pool_path, diagnosis):
    prior = original_inputs(parent, initial, cache, direction, pool_path)
    prior.pop("new_optimizer_updates")
    if read(direction / "plan.json")["adapter"]["fixed_scales"] != list(SCALES):
        raise ValueError("unison correction changed its qualified score calibration")
    diagnosed = read(diagnosis / "summary.json")
    if (not diagnosed["parent_evidence_unchanged"] or diagnosed["native_requests"] != 0 or diagnosed["optimizer_updates"] != 0
            or len(diagnosed["prediction_rows"]) != 108 or len(diagnosed["measured_choice_rows"]) != 192
            or diagnosed["plan_id"] != canonical_sha256(read(diagnosis / "plan.json"))):
        raise ValueError("unison bridge requires the completed saved category diagnosis")
    source = {**prior["source"], **{name: sha256_bytes(Path(__file__).with_name(name).read_bytes().replace(b"\r\n", b"\n"))
              for name in ("unison_choice.py", "unison_choice_bridge.py", "paired_native_audit.py")}}
    script = Path(__file__).parents[1] / "reports/scripts/check_unison_choice_bridge.py"
    source["reports/scripts/check_unison_choice_bridge.py"] = sha256_bytes(script.read_bytes().replace(b"\r\n", b"\n"))
    return {**prior, "format": "independent_unison_choice_bridge_v1", "source": source, "limits": LIMITS, "model_budget": COUNTS,
            "model_spec": SPEC, "initialization": "zero_final_unison_correction_layer_v1", "original_scales": list(SCALES),
            "diagnosis_files": {name: digest(diagnosis / name) for name in ("plan.json", "summary.json")},
            "synthetic_recovery": {"optimizer": "Adam", "learning_rate": .001, "checkpoint_after_step": 1,
                                   "committed_simulated_steps": 2, "replayed_simulated_steps": 1, "native_requests": 0},
            "new_native_requests": 0, "experimental_training_updates": 0}


@torch.no_grad()
def preserve_predictions(base, model, restored, codec, bank, *, calibrated_initial):
    old = ProductionControlAdapter(codec, scales=SCALES)
    new = ProductionControlAdapter(codec, scales=SPEC["adapter_scales"])
    rows = []
    for recipe, item in sorted(bank.items()):
        features = batch_features([{"features": item["features"]}])
        first, second, third = [network(features) for network in (base, model, restored)]
        for value in (second, third):
            if not torch.equal(value["continuous"], first["continuous"]) or not torch.equal(value["categorical"][1], first["categorical"][1]):
                raise ValueError("unison correction changed continuous or filter predictions")
        if any(not torch.equal(second[key], third[key]) for key in ("continuous",)) or any(
                not torch.equal(a, b) for a, b in zip(second["categorical"], third["categorical"], strict=True)):
            raise ValueError("cold unison export changed a prediction")
        if calibrated_initial:
            for temperature in (1., .5):
                probabilities = (first["categorical"][0] / SCALES[0] / temperature).softmax(-1)
                if not torch.equal(probabilities, (second["categorical"][0] / temperature).softmax(-1)):
                    raise ValueError("unison initialization changed calibrated probabilities")
        predictions = [adapter.export(value["continuous"], value["categorical"])[0]
                       for adapter, value in ((old, first), (new, second), (new, third))]
        presets = [prediction_preset(network, codec, target) for network, target in zip((base, model, restored), predictions, strict=True)]
        if presets[1] != presets[2] or (calibrated_initial and (presets[0] != presets[1] or predictions[0] != predictions[1])):
            raise ValueError("starting or cold restored unison preset changed")
        rows.append({"recipe_id": recipe, "state_id": item["state_id"], "features_sha256": tensor_digest(item["features"]),
                     "audio_sha256": item["audio_sha256"], "preset_sha256": sha256_bytes(presets[1]),
                     "continuous_and_filter_sha256": tensor_digest({"continuous": second["continuous"], "filter": second["categorical"][1]})})
    return rows


def correction_gradients(model):
    model.require_frozen()
    values = {}
    for name, parameter in model.correction.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise ValueError("missing or nonfinite direct unison gradient")
        group = "output" if name.startswith("2.") else "hidden"
        values[group] = values.get(group, 0.) + float(parameter.grad.double().abs().sum())
    return values


def gradient_access(model, features, direction):
    before, rng = tensor_digest(model.state_dict()), torch.get_rng_state().clone()
    model.zero_grad(set_to_none=True)
    inputs = batch_features([{"features": features}])
    if raw_prediction(model.base(inputs)) != direction["raw_prediction"]:
        raise ValueError("cached category gradient is stale for the base predictor")
    prediction = model(inputs)
    adapter = ProductionControlAdapter(scales=SPEC["adapter_scales"])
    if adapter.export(prediction["continuous"], prediction["categorical"])[0].to_dict() != direction["production_patch"]:
        raise ValueError("cached category costs have a different hard anchor")
    saved = direction["estimate"]
    from .codec import PilotTarget
    estimate = NativeGradient(PilotTarget.from_dict(saved["anchor"]), saved["value"], tuple(saved["continuous"]),
                              tuple(tuple(row) for row in saved["categorical"]), saved["queries"])
    checks = []
    for temperature in (1., .5):
        encoded = adapter(prediction["continuous"], prediction["categorical"], temperature=temperature, sample=True)
        loss = measured_loss(prediction["continuous"], (encoded[:, 9:12], encoded[:, 12:]), estimate)
        scores = prediction["categorical"][0]
        observed = torch.autograd.grad(loss, scores, retain_graph=True)[0]
        probability = (scores.detach() / temperature).softmax(-1)
        costs = scores.new_tensor([estimate.categorical[0]])
        expected = probability * (costs - (probability * costs).sum(-1, keepdim=True)) / temperature
        if not torch.allclose(observed, expected, atol=1e-7, rtol=1e-5):
            raise ValueError("direct unison gradient differs from the measured-cost softmax derivative")
        checks.append({"temperature": temperature, "gradient_l1": float(observed.double().abs().sum()),
                       "maximum_derivative_error": float((observed - expected).abs().max())})
    loss.backward()
    groups = correction_gradients(model)
    responding = max(estimate.categorical[0]) - min(estimate.categorical[0]) > 1e-7
    if groups["hidden"] != 0 or (responding and groups["output"] <= 0):
        raise ValueError("direct unison initialization has the wrong gradient path")
    if tensor_digest(model.state_dict()) != before or not torch.equal(rng, torch.get_rng_state()):
        raise ValueError("gradient qualification changed unison weights or RNG")
    return {"responding": responding, "derivative_checks": checks, "gradient_groups": groups,
            "frozen_gradients_absent": True, "weights_and_rng_unchanged": True}


def simulate_recovery(model, frontend, codec, pool_id, features, costs, output):
    before_rng = capture_rng("cpu")
    inputs = batch_features([{"features": features}])
    trial = copy.deepcopy(model).train()
    trial.zero_grad(set_to_none=True)
    optimizer = torch.optim.Adam(trial.correction.parameters(), lr=.001)
    def step(network, current):
        current.zero_grad(set_to_none=True)
        prediction = network(inputs)
        loss = measured_choice_loss(prediction, [costs])
        loss.backward()
        groups = correction_gradients(network)
        current.step()
        network.require_frozen()
        return {"loss": float(loss.detach()), "gradient_groups": groups}
    try:
        first = step(trial, optimizer)
        snapshot = {"artifact": export_unison(trial, frontend, codec, pool_id), "optimizer": copy.deepcopy(optimizer.state_dict()), "rng": capture_rng("cpu")}
        atomic_torch_save(output / "simulation-checkpoint.pt", snapshot)
        second = step(trial, optimizer)
        expected_rng = capture_rng("cpu")
        loaded = torch.load(output / "simulation-checkpoint.pt", map_location="cpu", weights_only=True)
        restored, _front, _codec = restore_unison(loaded["artifact"])
        current = torch.optim.Adam(restored.correction.parameters(), lr=.001)
        current.load_state_dict(loaded["optimizer"])
        restore_rng(loaded["rng"], "cpu")
        repeated = step(restored.train(), current)
        if (second != repeated or not equal_tree(trial.state_dict(), restored.state_dict())
                or not equal_tree(optimizer.state_dict(), current.state_dict()) or not equal_tree(expected_rng, capture_rng("cpu"))
                or not all(second["gradient_groups"][key] > 0 for key in ("hidden", "output"))):
            raise ValueError("simulated unison model/Adam/RNG recovery or later gradient access failed")
        atomic_torch_save(output / "simulation-model.pt", export_unison(trial, frontend, codec, pool_id))
        cold, _front, _codec = restore_unison(torch.load(output / "simulation-model.pt", map_location="cpu", weights_only=True))
        return trial.eval(), cold, {"first": first, "second": second, "model_adam_rng_recovery_exact": True,
                                   "simulated_optimizer_step_applications": 3, "experimental_training_updates": 0}
    finally:
        restore_rng(before_rng, "cpu")


def run(parent, initial, cache, direction, pool_path, diagnosis, output, *, execute=False):
    started, monitor = perf_counter(), MemoryMonitor()
    monitor.start()
    try:
        plan = make_plan(parent, initial, cache, direction, pool_path, diagnosis)
        if not execute:
            output.mkdir(parents=True, exist_ok=False)
            write_json(output / "plan.json", plan)
        else:
            if read(output / "plan.json") != plan or (output / "summary.json").exists():
                raise ValueError("preserve the existing unison qualification or changed frozen input")
            with refresh.cpu_settings():
                saved = torch.load(initial, map_location="cpu", weights_only=True)
                base, frontend, codec = restore_model(saved)
                pool = PilotPool.load(pool_path)
                if (pool.pool_id != saved["training_pool_id"] or codec.to_dict() != pool.codec.to_dict()
                        or artifact_rendering(saved)["render_phase_controls"] != FULL_PHASE_CONTROL
                        or artifact_rendering(saved)["render_profile"]["preroll_frames"] != 32768):
                    raise ValueError("unison bridge changed its accepted scope or fixed-phase profile")
                bank = training_bank(torch.load(cache, map_location="cpu", weights_only=True), pool)
                model = UnisonChoice(base, seed=plan["seed"]).eval()
                artifact = output / "model.pt"
                atomic_torch_save(artifact, export_unison(model, frontend, codec, pool.pool_id))
                artifact_digest = digest(artifact)
                restored, _front, _codec = restore_unison(torch.load(artifact, map_location="cpu", weights_only=True))
                initial_rows = preserve_predictions(base, model, restored, codec, bank, calibrated_initial=True)
                descriptors, cases = read(direction / "plan.json")["cases"], read(direction / "cases.json")
                if len(descriptors) != 24 or len(cases) != 24:
                    raise ValueError("unison qualification requires the 24 native cases")
                gradients = []
                for descriptor, case in zip(descriptors, cases, strict=True):
                    item = bank[descriptor["input_recipe_id"]]
                    if (any(descriptor[key] != case[key] for key in ("index", "role", "state_id"))
                            or item["state_id"] != descriptor["state_id"] or item["audio_sha256"] != descriptor["input_audio_sha256"]
                            or tensor_digest(item["features"]) != descriptor["input_features_sha256"]):
                        raise ValueError("unison native qualification input changed")
                    gradients.append({"index": case["index"], "role": case["role"], "state_id": case["state_id"],
                                      **gradient_access(model, item["features"], case["direction"])})
                selected = next(index for index, row in enumerate(gradients) if row["responding"])
                item = bank[descriptors[selected]["input_recipe_id"]]
                costs = cases[selected]["direction"]["estimate"]["categorical"][0]
                trial, cold, recovery = simulate_recovery(model, frontend, codec, pool.pool_id, item["features"], costs, output)
                simulated_rows = preserve_predictions(base, trial, cold, codec, bank, calibrated_initial=False)
                if (model_rendering(model) != model_rendering(base) or tensor_digest(model.state_dict()) != read_model_digest(artifact)
                        or digest(artifact) != artifact_digest):
                    raise ValueError("unison qualification changed its initial export or profile")
            if make_plan(parent, initial, cache, direction, pool_path, diagnosis) != plan:
                raise ValueError("unison qualification changed immutable parent evidence")
    finally:
        monitor.close()
    resources = {"elapsed_seconds": perf_counter() - started, "peak_process_tree_rss_bytes": monitor.peak_bytes, "rss_sampling_errors": monitor.errors}
    if resources["elapsed_seconds"] >= 300 or monitor.peak_bytes is None or monitor.peak_bytes > LIMITS["process_tree_bytes"] or monitor.errors:
        raise ValueError("unison qualification exceeded five minutes or 4 GiB")
    report = {"plan_id": canonical_sha256(plan), "resources": resources, "new_native_requests": 0, "experimental_training_updates": 0,
              "simulated_optimizer_step_applications": 3 if execute else 0,
              "qualified_for_release": False}
    if execute:
        report.update(initialization_and_gradient_access_passed=True, parent_artifacts_unchanged=True, model_budget=COUNTS,
                      initialization_cases=len(initial_rows), simulated_preservation_cases=len(simulated_rows), native_gradient_cases=len(gradients),
                      model_sha256=tensor_digest(model.state_dict()), model_file_sha256=digest(artifact),
                      initial_rows=initial_rows, simulated_rows=simulated_rows, gradient_rows=gradients, synthetic_recovery=recovery)
    archive = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    if archive + 2 * len(json.dumps(report).encode("utf-8")) + 1024**2 > LIMITS["archive_bytes"]:
        raise ValueError("unison qualification exceeded its evidence ceiling")
    write_json(output / ("summary.json" if execute else "preparation.json"), report)
    return {key: value for key, value in report.items() if key not in ("initial_rows", "simulated_rows", "gradient_rows")}


def read_model_digest(path):
    return torch.load(path, map_location="cpu", weights_only=True)["model_sha256"]

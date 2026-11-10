"""Matched native sound fitting with trainable basic audio representations."""

from pathlib import Path
from time import perf_counter
import math

import torch

from obruxo_data.hashing import canonical_sha256, sha256_bytes

from .audit import write_json
from .basic_direct import REHEARSAL
from .basic_direct_fit import execute, initialize
from .basic_direct_fit_audit import run_verification
from .basic_direct_fit_data import Inputs
from .basic_direct_fit_step import predict_many
from .batching import BatchConfig, compatible_infrastructure_source
from .benchmark import MemoryMonitor
from .category_scale_refresh import cpu_settings
from .codec import PilotTarget
from .conditioned_bridge import read
from .features import tensor_digest
from .model import loss_terms
from .native_acceptance import SPEC as ACCEPTANCE
from .native_fit import digest
from .paired_native_audit import state_signature
from .trainable_basic import COUNTS, SPEC, TrainableBasic
from .trainer import capture_rng, restore_rng

COMPATIBLE = {"basic_direct.py", "basic_direct_fit.py", "basic_direct_fit_audit.py"}


def optimizer_digest(state):
    return canonical_sha256(state_signature(state))


def check_input_learning(inputs, models, optimizers, *, prefix=False):
    if type(inputs.model) is not TrainableBasic or set(models) != {"parameter", "spectral"}:
        raise ValueError("input-learning check requires both matched trainable basic models")
    baseline, result = inputs.model.state_dict(), {}
    for arm, model in models.items():
        model.require_frozen()
        state, branches = model.state_dict(), {}
        for branch in ("timbre", "pitch"):
            keys = [name for name in baseline if name.startswith(f"correction.0.{branch}.")]
            change = sum(float((state[name].double() - baseline[name].double()).abs().sum()) for name in keys)
            if not keys or not change > 0 or not math.isfinite(change):
                raise ValueError("both copied input branches must change before continuation")
            branches[branch] = change
        result[arm] = branches
    key = "expected_parameter_prefix_optimizer_sha256" if prefix else "expected_parameter_optimizer_sha256"
    expected = inputs.plan.get(key)
    if expected and optimizer_digest(optimizers["parameter"].state_dict()) != expected:
        raise ValueError("matched parameter-control Adam state differs from its frozen rehearsal")
    return result


def parameter_rehearsal(inputs):
    """Pin the label control, never using its weights to initialize the candidate."""
    rng = capture_rng("cpu")
    try:
        models, optimizers = initialize(inputs)
        model, optimizer = models["parameter"], optimizers["parameter"]
        losses, prefix = [], None
        weights = REHEARSAL["paired_weights"] + REHEARSAL["reference_weights"]
        for index, update in enumerate(inputs.plan["updates"]):
            optimizer.zero_grad(set_to_none=True)
            rows = []
            cells = [inputs.plan["fit_audit_cells"][indices[0]] for indices in inputs.observations(update)]
            predictions = predict_many(model, [inputs.features[inputs.key(cell)] for cell in cells], batch_size=BatchConfig.from_plan(inputs.plan).model)
            for cell, (prediction, _bounds) in zip(cells, predictions, strict=True):
                target = PilotTarget.from_dict(cell["recipe"]["target"])
                rows.append(loss_terms(prediction, [target], inputs.codec)["loss"])
            loss = sum(weight * value for weight, value in zip(weights, rows, strict=True))
            if not torch.isfinite(loss):
                raise ValueError("nonfinite parameter-control rehearsal")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.correction.parameters(), inputs.plan["optimizer"]["gradient_norm_cap"])
            if not torch.isfinite(norm):
                raise ValueError("nonfinite parameter-control rehearsal gradient")
            optimizer.step()
            model.require_frozen()
            losses.append(float(loss.detach()))
            if index + 1 == inputs.plan["prefix_check_updates"]:
                prefix = {"model_sha256": tensor_digest(model.state_dict()), "optimizer_sha256": optimizer_digest(optimizer.state_dict())}
        if prefix is None:
            raise ValueError("parameter-control rehearsal has no declared prefix")
        return {"software_parameter_control_optimizer_applications": len(losses), "experimental_candidate_updates": 0,
                "native_requests": 0, "losses": losses, "prefix": prefix,
                "model_sha256": tensor_digest(model.state_dict()), "optimizer_sha256": optimizer_digest(optimizer.state_dict())}
    finally:
        restore_rng(rng, "cpu")


def source_code(plan, *, compatible=()):
    root, actual, differences = Path(__file__).parents[1], {}, {}
    for name, expected in plan["source"].items():
        path = root / name if "/" in name else Path(__file__).with_name(name)
        actual[name] = sha256_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
        if actual[name] != expected:
            module = name.removeprefix("obruxo_training/")
            if module not in COMPATIBLE and module not in compatible and name not in compatible and not compatible_infrastructure_source(name, expected, actual[name]):
                raise ValueError(f"qualified numerical prediction, loss or renderer implementation changed: {name}")
            differences[name] = {"historical": expected, "current": actual[name]}
    return actual, differences


def inputs_bound(parent, bridge, cache, pool, prior):
    qualified, bridge_plan = read(bridge / "summary.json"), read(bridge / "plan.json")
    proof, old = read(prior / "verification.json"), read(prior / "plan.json")
    if (not qualified["initialization_gradient_reference_recovery_passed"] or not qualified["parent_artifacts_unchanged"]
            or qualified["model_budget"] != COUNTS or qualified["initialization_cases"] != 1616 or qualified["simulated_cold_cases"] != 1616
            or qualified["native_gradient_cases"] != 24 or qualified["new_native_requests"] or qualified["experimental_training_updates"]
            or qualified["plan_id"] != canonical_sha256(bridge_plan) or digest(bridge / "model.pt") != qualified["model_file_sha256"]
            or not proof["verified"] or not proof["complete"] or not proof["offline_training_replay_exact"]
            or proof["committed_updates_per_arm"] != 48 or proof["comparison_waveforms_rescored"] != 312 or proof["matched_comparison_passed"]
            or proof["test_split_used"] or proof["qualified_for_release"] or bridge_plan["preceding_fit_plan_id"] != proof["plan_id"]
            or digest(prior / "verification.json") != bridge_plan["preceding_fit_verification_sha256"]
            or old["native_acceptance"] != ACCEPTANCE or old.get("continuous_auxiliary") is not None
            or digest(cache) != old["input_cache_sha256"] or pool.pool_id != old["pool_id"]):
        raise ValueError("trainable sound fitting requires its qualified untrained bridge and audited preceding fit")
    files = {f"prior/{name}": value for name, value in proof["evidence_sha256"].items()}
    files.update({name: value for name, value in old["input_files"].items() if name.startswith("parent/")})
    files.update({f"bridge/{name}": digest(bridge / name) for name in ("plan.json", "summary.json", "model.pt")})
    files["prior/verification.json"] = digest(prior / "verification.json")
    locations = {"parent": parent, "prior": prior, "bridge": bridge}
    if any(digest(locations[name.split("/", 1)[0]] / name.split("/", 1)[1]) != value for name, value in files.items()):
        raise ValueError("verified training inputs or preceding evidence changed")
    train = {state for state, assignment in pool.manifest.by_state.items() if assignment.split == "train"}
    if (not set(old["regression_state_ids"]).issubset(train) or any(row["state_id"] not in train for row in old["training"])
            or any(bridge_plan[key] != old[key] for key in ("expected_renderer_identity", "render_profile", "render_phase_controls"))):
        raise ValueError("trainable sound fitting changed the renderer or includes held-out states")
    return old, bridge_plan, qualified, files


def prepare(parent, bridge, cache, pool, prior, output):
    old, bridge_plan, qualified, files = inputs_bound(parent, bridge, cache, pool, prior)
    code, old_differences = source_code(old)
    bridge_code, bridge_differences = source_code(bridge_plan)
    names = ("trainable_basic_fit.py", "reports/scripts/run_trainable_basic_fit.py")
    root = Path(__file__).parents[1]
    code.update({name: sha256_bytes((root / name if "/" in name else Path(__file__).with_name(name)).read_bytes().replace(b"\r\n", b"\n")) for name in names})
    storage = {**old["storage_bound"], "trainable_input_checkpoint_optimizer_allowance_bytes": 64 * 1024**2,
               "estimated_archive_upper_bound_bytes": old["storage_bound"]["estimated_archive_upper_bound_bytes"] + 64 * 1024**2,
               "resident_replay_upper_bound_bytes": old["storage_bound"]["resident_replay_upper_bound_bytes"] + 4 * 1024**2}
    limits = {**old["limits"], "wall_seconds": 21600}
    if storage["estimated_archive_upper_bound_bytes"] > limits["archive_bytes"] or storage["resident_replay_upper_bound_bytes"] > limits["replay_bytes"]:
        raise ValueError("trainable input evidence or replay exceeds the fixed allocation")
    plan = {**{key: value for key, value in old.items() if key != "final_only_completion"},
            "format": "trainable_basic_native_acceptance_fit_v1", "source": {**code, **bridge_code}, "input_files": files,
            "initial_file_sha256": qualified["model_file_sha256"], "initial_model_sha256": qualified["model_sha256"],
            "bridge_plan_id": qualified["plan_id"], "model_budget": COUNTS, "model_spec": SPEC,
            "require_trainable_input_updates": True, "limits": limits, "storage_bound": storage,
            "comparison_parent_plan_id": canonical_sha256(old),
            "compatible_adapter_source_differences": {"preceding_fit": old_differences, "bridge": bridge_differences}}
    inputs = Inputs(parent, bridge, cache, plan)
    inputs.validate()
    rehearsal = parameter_rehearsal(inputs)
    plan.update(expected_parameter_model_sha256=rehearsal["model_sha256"], expected_parameter_optimizer_sha256=rehearsal["optimizer_sha256"],
                expected_parameter_prefix_model_sha256=rehearsal["prefix"]["model_sha256"],
                expected_parameter_prefix_optimizer_sha256=rehearsal["prefix"]["optimizer_sha256"])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "parameter-control-rehearsal.json", rehearsal)
    plan["parameter_control_rehearsal_sha256"] = digest(output / "parameter-control-rehearsal.json")
    write_json(output / "plan.json", plan)
    return plan, rehearsal


def run(parent, bridge, cache, pool, prior, output, *, mode="prepare"):
    started, monitor = perf_counter(), MemoryMonitor()
    monitor.start()
    with cpu_settings():
        try:
            if mode == "prepare":
                plan, rehearsal = prepare(parent, bridge, cache, pool, prior, output)
            else:
                plan = read(output / "plan.json")
                old, bridge_plan, qualified, files = inputs_bound(parent, bridge, cache, pool, prior)
                code, differences = source_code(plan)
                if (differences or code != plan["source"] or files != plan["input_files"]
                        or canonical_sha256(old) != plan["comparison_parent_plan_id"] or qualified["plan_id"] != plan["bridge_plan_id"]
                        or digest(cache) != plan["input_cache_sha256"]
                        or digest(output / "parameter-control-rehearsal.json") != plan["parameter_control_rehearsal_sha256"]):
                    raise ValueError("frozen trainable sound-fit plan, inputs or source changed")
                inputs = Inputs(parent, bridge, cache, plan)
                inputs.validate()
        finally:
            monitor.close()
        preparation = {"elapsed_seconds": perf_counter() - started, "peak_process_tree_rss_bytes": monitor.peak_bytes,
                       "rss_sampling_errors": monitor.errors, "native_requests": 0, "experimental_candidate_updates": 0,
                       "software_parameter_control_optimizer_applications": len(plan["updates"]) if mode == "prepare" else 0}
        if preparation["elapsed_seconds"] >= 600 or not monitor.peak_bytes or monitor.peak_bytes > 4 * 1024**3 or monitor.errors:
            raise ValueError("trainable sound-fit preparation exceeded ten minutes or 4 GiB")
        if mode == "prepare":
            write_json(output / "preparation.json", preparation)
            return {"plan_id": canonical_sha256(plan), "preparation": preparation, "storage_bound": plan["storage_bound"], "control_rehearsal": rehearsal}
        if mode == "verify":
            return run_verification(inputs, output)
        if mode not in ("execute", "resume"):
            raise ValueError("expected prepare, execute, resume or verify")
        return execute(inputs, output, resume=mode == "resume")

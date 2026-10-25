"""Inspect actual control changes from two audited training trajectories."""

from pathlib import Path
from statistics import median
from time import perf_counter

import torch

from obruxo_data.hashing import canonical_sha256, sha256_bytes

from . import category_scale_refresh as refresh
from .audit import write_json
from .basic_direct import REHEARSAL
from .basic_direct_fit import SoundStore, initialize, lines, step
from .basic_direct_fit_data import Inputs
from .benchmark import MemoryMonitor
from .codec import PilotTarget
from .conditioned_bridge import read
from .native_fit import digest
from .native_gradient import native_gradient_spec
from .paired_native_audit import equal_tree
from .trainer import capture_rng

LIMITS = {"wall_seconds": 300, "process_tree_bytes": 4 * 1024**3, "archive_bytes": 64 * 1024**2}
COMPATIBLE_REPLAY_ADAPTERS = {"basic_direct_fit.py", "basic_direct_fit_step.py", "basic_direct_fit_audit.py"}


def controls(prediction):
    return {"continuous": prediction["continuous"].detach()[0].tolist(),
            "categorical": [int(value.detach().argmax(-1)) for value in prediction["categorical"]]}


def movement(before, after, estimate, codec):
    if before != {"continuous": estimate["anchor"]["continuous"], "categorical": estimate["anchor"]["categorical"]}:
        raise ValueError("update inspection reached stale measured controls")
    delta = [b - a for a, b in zip(before["continuous"], after["continuous"], strict=True)]
    mask = codec.continuous_mask(PilotTarget.from_dict(estimate["anchor"]))
    continuous = sum(g * d for g, d in zip(estimate["continuous"], delta, strict=True))
    category = sum(costs[new] - costs[old] for costs, old, new in zip(estimate["categorical"], before["categorical"], after["categorical"], strict=True))
    spacing = native_gradient_spec()["continuous_step"]
    return {"continuous_delta": delta, "maximum_control_change": max(map(abs, delta)),
            "maximum_previously_active_change": max(abs(d) for d, active in zip(delta, mask, strict=True) if active),
            "above_measurement_spacing": max(map(abs, delta)) > spacing + 1e-8,
            "category_heads_changed": sum(a != b for a, b in zip(before["categorical"], after["categorical"], strict=True)),
            "before_choices": before["categorical"], "after_choices": after["categorical"],
            "linear_continuous_change": continuous, "conditional_category_change": category,
            "estimated_sound_change": continuous + category}


def predict_bank(inputs, model):
    result = {}
    with torch.no_grad():
        for cell in inputs.plan["fit_audit_cells"]:
            key = inputs.key(cell)
            if key not in result:
                result[key] = controls(inputs.predict(model, cell))
    return result


def inspect(inputs, source):
    saved = torch.load(source / "checkpoint.pt", map_location="cpu", weights_only=True)
    if saved["plan_id"] != canonical_sha256(inputs.plan) or saved["step"] != len(saved["trace"]) or saved["step"] != len(inputs.plan["updates"]):
        raise ValueError("update inspection requires a complete matching checkpoint")
    finals = {(row["index"], row["arm"]): row for row in lines(source / "finals.jsonl")}
    models, optimizers = initialize(inputs)
    store = SoundStore(source, inputs)
    observations, updates = [], []
    for index, recorded in enumerate(saved["trace"]):
        rng = capture_rng("cpu")
        before = predict_bank(inputs, models["spectral"])
        if not equal_tree(rng, capture_rng("cpu")):
            raise ValueError("prediction inspection changed randomness")
        result = step(models, optimizers, inputs, store, index)
        expected = {"step": index + 1, "update": inputs.plan["updates"][index], **result}
        if expected != {key: value for key, value in recorded.items() if key != "attempt"}:
            raise ValueError("control inspection changed the recorded update trace")
        after = predict_bank(inputs, models["spectral"])
        bank_changes = {key: max(abs(b - a) for a, b in zip(before[key]["continuous"], after[key]["continuous"], strict=True)) for key in before}
        spacing = native_gradient_spec()["continuous_step"]
        current_rows, weighted = [], 0.
        for observation, cells in enumerate(inputs.observations(recorded["update"])):
            cell = inputs.plan["fit_audit_cells"][cells[0]]
            key = inputs.key(cell)
            role = "paired_original" if observation == 0 else "reference" if observation > 1 else inputs.plan["training_view_roles"][cell["input_view"]]
            estimate = recorded["estimates"][observation]
            row = {"step": index + 1, "observation": observation, "role": role, "state_id": key[0], "input_view": key[1],
                   **movement(before[key], after[key], estimate, inputs.codec)}
            weight = (REHEARSAL["paired_weights"] + REHEARSAL["reference_weights"])[observation]
            if observation > 1:
                baseline = inputs.cases[cells[0]]["rows"]["initial"]["scores"]["spectral"]
                active = float(torch.tensor(estimate["value"])) > float(torch.tensor(baseline))
                weight *= 2 if active else 1
            weighted += weight * row["estimated_sound_change"]
            if index + 1 == len(saved["trace"]):
                final = sum(finals[(cell_index, "spectral")]["scores"]["spectral"] for cell_index in cells) / len(cells)
                row.update(final_step_measured_before=estimate["value"], final_step_measured_after=final,
                           final_step_measured_change=final - estimate["value"])
            observations.append(row)
            current_rows.append(row)
        worst = max(bank_changes, key=bank_changes.get)
        updates.append({"step": index + 1, "sampled_maximum_control_change": max(row["maximum_control_change"] for row in current_rows),
                        "bank_maximum_control_change": bank_changes[worst], "bank_inputs_above_spacing": sum(value > spacing + 1e-8 for value in bank_changes.values()),
                        "worst_bank_input": {"state_id": worst[0], "input_view": worst[1]}, "bank_inputs": len(bank_changes),
                        "weighted_native_linear_change": weighted,
                        "reference_estimated_worsening": sum(row["estimated_sound_change"] > 1e-9 for row in current_rows if row["role"] == "reference")})
    if (not all(equal_tree(model.state_dict(), saved["models"][arm]) and equal_tree(optimizers[arm].state_dict(), saved["optimizers"][arm])
                for arm, model in models.items()) or not equal_tree(capture_rng("cpu"), saved["rng"])
            or store.used_probes != set(store.probes)):
        raise ValueError("inspected models, Adam, RNG or native-probe replay differ from the audited run")
    for cell_index, cell in enumerate(inputs.plan["fit_audit_cells"]):
        key = inputs.key(cell)
        target = inputs.codec.canonical_target(PilotTarget(tuple(after[key]["continuous"]), tuple(after[key]["categorical"])))
        if target.to_dict() != finals[(cell_index, "spectral")]["candidate"]:
            raise ValueError("final-step audio differs from the inspected final prediction")
    roles = sorted({row["role"] for row in observations})
    groups = {role: {"observations": sum(row["role"] == role for row in observations),
                    "above_spacing": sum(row["role"] == role and row["above_measurement_spacing"] for row in observations),
                    "estimated_worsening": sum(row["role"] == role and row["estimated_sound_change"] > 1e-9 for row in observations)} for role in roles}
    return {"updates": updates, "observations": observations, "groups": groups,
            "summary": {"steps": len(updates), "sampled_steps_above_spacing": sum(row["sampled_maximum_control_change"] > spacing + 1e-8 for row in updates),
                        "bank_steps_above_spacing": sum(row["bank_maximum_control_change"] > spacing + 1e-8 for row in updates),
                        "median_sampled_maximum_change": median(row["sampled_maximum_control_change"] for row in updates),
                        "maximum_bank_change": max(row["bank_maximum_control_change"] for row in updates),
                        "steps_with_reference_estimated_worsening": sum(row["reference_estimated_worsening"] > 0 for row in updates),
                        "multiple_category_heads_changed_observations": sum(row["category_heads_changed"] > 1 for row in observations)},
            "final_step_measured_observations": [row for row in observations if "final_step_measured_change" in row],
            "model_adam_rng_replay_exact": True, "replayed_updates_per_arm": saved["step"], "scalar_probes_reused": len(store.used_probes)}


def make_plan(sources, parent, bridge, features):
    if len(sources) != 2 or sources[0].resolve() == sources[1].resolve():
        raise ValueError("compare two distinct audited trajectories")
    root = Path(__file__).parents[1]
    current = {}
    runs, plans = [], []
    for source in sources:
        proof, plan = read(source / "verification.json"), read(source / "plan.json")
        if (proof["plan_id"] != canonical_sha256(plan) or not proof["verified"] or not proof["complete"] or not proof["offline_training_replay_exact"] or not proof["native_receipts_resolved"]
                or proof["committed_updates_per_arm"] != 48 or proof["comparison_waveforms_rescored"] != 312
                or proof["test_split_used"] or proof["qualified_for_release"]):
            raise ValueError("update diagnosis requires complete independently audited training trajectories")
        if any(digest(source / name) != value for name, value in proof["evidence_sha256"].items()):
            raise ValueError("audited trajectory evidence changed")
        if (digest(parent / "fit_cases.json") != plan["input_files"]["parent/fit_cases.json"] or digest(features) != plan["input_cache_sha256"]
                or digest(bridge / "model.pt") != plan["initial_file_sha256"]):
            raise ValueError("trajectory source recordings, cache or initializer changed")
        differences = {}
        for name, value in plan["source"].items():
            path = root / name if "/" in name else Path(__file__).with_name(name)
            actual = sha256_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
            current[name] = actual
            if actual != value:
                if name not in COMPATIBLE_REPLAY_ADAPTERS:
                    raise ValueError("historical numerical model, loss or renderer implementation changed")
                differences[name] = {"historical": value, "current": actual}
        runs.append({"plan_id": proof["plan_id"], "verification_sha256": digest(source / "verification.json"),
                     "evidence_sha256": proof["evidence_sha256"], "compatible_replayer_differences": differences,
                     "continuous_auxiliary": plan.get("continuous_auxiliary"), "model_budget": plan["model_budget"]})
        plans.append(plan)
    shared = ("training", "fit_audit_cells", "updates", "optimizer", "reference_manifest", "rehearsal", "temperature", "pool_id",
              "input_cache_sha256", "initial_file_sha256", "model_budget", "model_spec", "render_profile", "render_phase_controls", "expected_renderer_identity")
    if any(plans[0][key] != plans[1][key] for key in shared) or plans[0].get("continuous_auxiliary") is not None or plans[1].get("continuous_auxiliary") is None:
        raise ValueError("compare matched sound-only then continuous-auxiliary trajectories")
    names = ("update_diagnosis.py", "continuous_auxiliary.py", "reports/scripts/inspect_update_steps.py")
    current.update({name: sha256_bytes((root / name if "/" in name else Path(__file__).with_name(name)).read_bytes().replace(b"\r\n", b"\n")) for name in names})
    return {"format": "audited_basic_update_movement_diagnosis_v1", "runs": runs, "source": current, "limits": LIMITS,
            "measurement_spacing": native_gradient_spec()["continuous_step"], "spacing_is_certified_linearity_radius": False,
            "native_requests": 0, "experimental_training_updates": 0, "replayed_optimizer_applications": 192,
            "measured_after_update_audio_available_only_for_final_step": True, "test_split_used": False, "qualified_for_release": False}


def run(sources, parent, bridge, features, output, *, execute=False):
    with refresh.cpu_settings():
        plan = make_plan(sources, parent, bridge, features)
        if not execute:
            output.mkdir(parents=True, exist_ok=False)
            write_json(output / "plan.json", plan)
            return {"plan_id": canonical_sha256(plan), "native_requests": 0, "experimental_training_updates": 0}
        if read(output / "plan.json") != plan or (output / "summary.json").exists():
            raise ValueError("preserve the frozen update diagnosis and any completed result")
        started, monitor = perf_counter(), MemoryMonitor()
        monitor.start()
        try:
            results = [{"plan_id": row["plan_id"], **inspect(Inputs(parent, bridge, features, read(source / "plan.json")), source)}
                       for source, row in zip(sources, plan["runs"], strict=True)]
        finally:
            monitor.close()
        resource = {"elapsed_seconds": perf_counter() - started, "peak_process_tree_rss_bytes": monitor.peak_bytes, "rss_sampling_errors": monitor.errors}
        within = bool(resource["elapsed_seconds"] < LIMITS["wall_seconds"] and monitor.peak_bytes and monitor.peak_bytes <= LIMITS["process_tree_bytes"] and not monitor.errors)
        result = {"plan_id": canonical_sha256(plan), "runs": results, "resources": resource, "within_resource_limits": within,
                  "native_requests": 0, "experimental_training_updates": 0, "replayed_optimizer_applications": sum(row["replayed_updates_per_arm"] * 2 for row in results),
                  "qualified_for_release": False, "test_split_used": False}
        write_json(output / "summary.json", result)
        if not within or sum(path.stat().st_size for path in output.rglob("*") if path.is_file()) > LIMITS["archive_bytes"]:
            raise ValueError("update diagnosis exceeded its resource allocation")
        if make_plan(sources, parent, bridge, features) != plan:
            raise ValueError("update diagnosis changed its source evidence")
        return {"plan_id": result["plan_id"], "runs": [{"plan_id": row["plan_id"], "summary": row["summary"], "groups": row["groups"]} for row in results],
                "resources": resource, "within_resource_limits": within, "native_requests": 0, "experimental_training_updates": 0}

"""Qualify trainable input features without changing any starting patch."""

from pathlib import Path
from time import perf_counter
import json

import torch

from obruxo_data.hashing import canonical_sha256, sha256_bytes

from .audit import write_json
from .basic_direct import export_direct, restore_direct
from .basic_direct_bridge import LIMITS, gradient_access, make_plan as direct_plan, preserve, simulate, validate_references
from .benchmark import MemoryMonitor
from .category_scale_refresh import cpu_settings
from .checkpoints import atomic_torch_save
from .conditioned_bridge import read, training_bank
from .features import tensor_digest
from .inference_profile import artifact_rendering
from .native_fit import digest
from .pilot import PilotPool
from .trainable_basic import COUNTS, SPEC, TrainableBasic, input_gradients
from .trainer import restore_model


def make_plan(parent, initial, cache, direction, pool, recent, baseline_parent, failed_fit):
    prior = direct_plan(parent, initial, cache, direction, pool, recent, baseline_parent)
    proof = read(failed_fit / "verification.json")
    if (not proof["verified"] or not proof["complete"] or not proof["offline_training_replay_exact"]
            or proof["comparison_waveforms_rescored"] != 312 or proof["matched_comparison_passed"]
            or proof["test_split_used"] or proof["qualified_for_release"]):
        raise ValueError("trainable inputs require the audited failed basic fit")
    evidence = {name: digest(failed_fit / name) for name in proof["evidence_sha256"]}
    if evidence != proof["evidence_sha256"]:
        raise ValueError("audited preceding fit evidence changed")
    failed_plan = read(failed_fit / "plan.json")
    if (failed_plan["input_cache_sha256"] != prior["input_files"]["features"]
            or failed_plan["expected_renderer_identity"] != prior["expected_renderer_identity"]
            or failed_plan["render_profile"] != prior["render_profile"]
            or failed_plan["render_phase_controls"] != prior["render_phase_controls"]):
        raise ValueError("trainable inputs changed the recorded features or rendering contract")
    names = ("trainable_basic.py", "trainable_basic_bridge.py", "basic_direct.py", "basic_direct_fit.py", "basic_direct_fit_step.py", "native_acceptance.py")
    code = {name: sha256_bytes(Path(__file__).with_name(name).read_bytes().replace(b"\r\n", b"\n")) for name in names}
    script = "reports/scripts/check_trainable_basic_bridge.py"
    code[script] = sha256_bytes((Path(__file__).parents[1] / script).read_bytes().replace(b"\r\n", b"\n"))
    return {**prior, "format": "trainable_basic_input_bridge_v1", "source": {**prior["source"], **code},
            "model_spec": SPEC, "model_budget": COUNTS, "initialization": SPEC["initialization"],
            "preceding_fit_plan_id": proof["plan_id"], "preceding_fit_evidence_sha256": evidence,
            "preceding_fit_verification_sha256": digest(failed_fit / "verification.json"),
            "new_native_requests": 0, "experimental_training_updates": 0, "model_admission_allowed": False}


def run(parent, initial, cache, direction, pool_path, recent, baseline_parent, failed_fit, output, *, execute=False):
    started, monitor = perf_counter(), MemoryMonitor()
    monitor.start()
    try:
        plan = make_plan(parent, initial, cache, direction, pool_path, recent, baseline_parent, failed_fit)
        if not execute:
            output.mkdir(parents=True, exist_ok=False)
            write_json(output / "plan.json", plan)
        else:
            if read(output / "plan.json") != plan or (output / "summary.json").exists():
                raise ValueError("preserve the attempted bridge or changed frozen inputs")
            with cpu_settings():
                saved = torch.load(initial, map_location="cpu", weights_only=True)
                base, frontend, codec = restore_model(saved)
                pool = PilotPool.load(pool_path)
                if pool.pool_id != saved["training_pool_id"] or codec.to_dict() != pool.codec.to_dict():
                    raise ValueError("trainable input bridge changed its basic pool or codec")
                profile = artifact_rendering(saved)
                if any(profile[key] != plan[key] for key in ("render_profile", "render_phase_controls")):
                    raise ValueError("trainable inputs changed settled rendering or fixed generator phases")
                bank = training_bank(torch.load(cache, map_location="cpu", weights_only=True), pool)
                model = TrainableBasic(base, seed=plan["seed"]).eval()
                target = output / "model.pt"
                atomic_torch_save(target, export_direct(model, frontend, codec, pool.pool_id))
                initial_sha = digest(target)
                cold, _front, _codec = restore_direct(torch.load(target, map_location="cpu", weights_only=True))
                identities = preserve(base, model, cold, codec, bank, initial=True)
                reference_check = validate_references(plan, recent, model, codec, bank)
                descriptors, cases = read(direction / "plan.json")["cases"], read(direction / "cases.json")
                if len(cases) != 24 or len(descriptors) != 24:
                    raise ValueError("trainable input bridge requires all 24 qualified native points")
                native = []
                for descriptor, case in zip(descriptors, cases, strict=True):
                    row = bank[descriptor["input_recipe_id"]]
                    if (any(descriptor[key] != case[key] for key in ("index", "role", "state_id"))
                            or row["state_id"] != descriptor["state_id"] or row["audio_sha256"] != descriptor["input_audio_sha256"]
                            or tensor_digest(row["features"]) != descriptor["input_features_sha256"]):
                        raise ValueError("trainable input native qualification input changed")
                    checked = gradient_access(model, row["features"], case["direction"])
                    gradients = input_gradients(model)
                    if any(gradients.values()):
                        raise ValueError("zero final projection should initially block input-layer gradients")
                    native.append({"index": case["index"], "role": case["role"], "state_id": case["state_id"],
                                   **checked, "initial_input_gradients": gradients})
                features = [bank[descriptors[index]["input_recipe_id"]]["features"] for index in (0, 1)]
                features += [bank[plan["reference_manifest"][index]["recipe_id"]]["features"] for index in (0, 16)]
                trial, restored, recovery = simulate(model, frontend, codec, pool.pool_id, features, output)
                later_gradients = input_gradients(trial)
                if not all(value > 0 for value in later_gradients.values()):
                    raise ValueError("sound-cost derivatives do not reach both trainable audio branches")
                simulated = preserve(base, trial, restored, codec, bank, initial=False)
                if digest(target) != initial_sha or tensor_digest(model.state_dict()) != torch.load(target, map_location="cpu", weights_only=True)["model_sha256"]:
                    raise ValueError("qualification changed the untrained input-model initializer")
            if make_plan(parent, initial, cache, direction, pool_path, recent, baseline_parent, failed_fit) != plan:
                raise ValueError("qualification changed immutable source evidence")
    finally:
        monitor.close()
    resources = {"elapsed_seconds": perf_counter() - started, "peak_process_tree_rss_bytes": monitor.peak_bytes, "rss_sampling_errors": monitor.errors}
    if (resources["elapsed_seconds"] >= LIMITS["wall_seconds"] or not monitor.peak_bytes
            or monitor.peak_bytes > LIMITS["process_tree_bytes"] or monitor.errors):
        raise ValueError("trainable input qualification exceeded five minutes or 4 GiB")
    report = {"plan_id": canonical_sha256(plan), "resources": resources, "new_native_requests": 0, "experimental_training_updates": 0,
              "qualified_for_release": False, "model_admission_allowed": False}
    if execute:
        report.update(initialization_gradient_reference_recovery_passed=True, parent_artifacts_unchanged=True, model_budget=COUNTS,
                      initialization_cases=len(identities), simulated_cold_cases=len(simulated), native_gradient_cases=len(native),
                      model_file_sha256=digest(target), model_sha256=tensor_digest(model.state_dict()), reference_check=reference_check,
                      initial_rows=identities, simulated_rows=simulated, native_rows=native, software_recovery=recovery,
                      later_input_gradient_l1=later_gradients)
    archive = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    if archive + 2 * len(json.dumps(report, indent=2).encode("utf-8")) + 1024**2 > LIMITS["archive_bytes"]:
        raise ValueError("trainable input qualification exceeded 64 MiB evidence")
    write_json(output / ("summary.json" if execute else "preparation.json"), report)
    return {key: value for key, value in report.items() if key not in ("initial_rows", "simulated_rows", "native_rows")}

"""Qualify the frozen-predictor correction at identical, previously measured controls."""

from collections import Counter
import json
from pathlib import Path
from time import perf_counter

import torch

from obruxo_data.hashing import canonical_sha256, sha256_bytes

from . import category_scale_refresh as refresh
from .audit import write_json
from .benchmark import MemoryMonitor
from .checkpoints import atomic_torch_save
from .conditioned_basic import COUNTS, bridge_state, gradient_access, make_conditioned, restore_conditioned
from .features import tensor_digest, validate_features
from .inference_profile import artifact_rendering, model_rendering, prediction_preset
from .model import batch_features
from .native_fit import digest
from .pilot import PilotPool
from .production_controls import ProductionControlAdapter
from .reproducibility import FULL_PHASE_CONTROL
from .trainer import restore_model

LIMITS = {"wall_seconds": 300, "process_tree_bytes": 4 * 1024**3, "archive_bytes": 64 * 1024**2}


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def inputs(parent, initial, cache, direction, pool_path):
    proof = read(parent / "verification.json")
    native = read(direction / "verification.json")
    if (not proof["verified"] or not proof["offline_training_replay_exact"] or not proof["all_native_accepted"] or not proof["complete"]
            or not native["verified"] or not native["qualified_production_direction"] or not native["all_model_gradients_and_local_steps_reconstructed"]):
        raise ValueError("conditioning bridge requires independently verified parents")
    files = {f"parent/{name}": digest(parent / name) for name in proof["evidence_sha256"]}
    if {name: files[f"parent/{name}"] for name in proof["evidence_sha256"]} != proof["evidence_sha256"]:
        raise ValueError("verified parent evidence changed")
    bound = proof["plan"]["native_direction_evidence_sha256"]
    current = {name: digest(direction / name) for name in bound}
    if current != bound or native["evidence_sha256"] != bound:
        raise ValueError("native direction no longer matches the verified parent")
    if (digest(initial) != proof["plan"]["initial_file_sha256"] or digest(initial) != native["plan"]["initial_sha256"]
            or digest(cache) != proof["plan"]["input_cache_sha256"] or digest(cache) != native["plan"]["input_cache_sha256"]):
        raise ValueError("conditioning bridge changed its initializer or cached recordings")
    names = ("model.py", "features.py", "codec.py", "trainer.py", "production_controls.py", "native_gradient.py", "spectral_controls.py",
             "inference_profile.py", "reproducibility.py")
    code = {name: sha256_bytes(Path(__file__).with_name(name).read_bytes().replace(b"\r\n", b"\n")) for name in names}
    if any(value != proof["plan"]["source"][name] for name, value in code.items()):
        raise ValueError("conditioning bridge changed the qualified prediction/control path")
    code.update({name: sha256_bytes(Path(__file__).with_name(name).read_bytes().replace(b"\r\n", b"\n"))
                 for name in ("conditioned_basic.py", "conditioned_bridge.py", "audit.py", "checkpoints.py", "benchmark.py")})
    files.update({f"direction/{name}": value for name, value in current.items()})
    files.update({"parent/verification.json": digest(parent / "verification.json"), "direction/verification.json": digest(direction / "verification.json"),
                  "initial": digest(initial), "features": digest(cache), "pool": digest(pool_path)})
    return {"format": "basic_conditioned_identity_bridge_v1", "parent_plan_id": proof["plan_id"], "direction_plan_id": native["plan_id"],
            "input_files": files, "source": code, "limits": LIMITS, "model_budget": COUNTS,
            "seed": 0, "initialization": "zero_output_projection_gain_one_v1", "new_native_requests": 0, "new_optimizer_updates": 0,
            "runtime": {"torch": str(torch.__version__), "device": "cpu", "precision": "float32", "threads": 2},
            "scheduled_recipes": 1616, "distinct_training_recordings": 1212, "training_states": 404}


def training_bank(cache, pool):
    train = {state for state, assignment in pool.manifest.by_state.items() if assignment.split == "train"}
    bank, observations = {}, {}
    for row in cache["replay"]:
        if row["state_id"] not in train:
            raise ValueError("conditioning bridge contains a held-out input")
        validate_features(row["features"])
        recipe = row["recipe_id"]
        if recipe in bank and (tensor_digest(row["features"]) != tensor_digest(bank[recipe]["features"])
                               or row["audio_sha256"] != bank[recipe]["audio_sha256"] or row["state_id"] != bank[recipe]["state_id"]):
            raise ValueError("duplicate training recipe has different features or audio")
        observation = row["state_id"], row["audio_sha256"]
        feature_hash = tensor_digest(row["features"])
        if observation in observations and observations[observation] != feature_hash:
            raise ValueError("duplicate recording has different features")
        observations[observation] = feature_hash
        bank[recipe] = row
    per_state = Counter(state for state, _audio in observations)
    if (len(cache["replay"]) != 1616 or len(bank) != 1616 or len(observations) != 1212
            or len(per_state) != 404 or any(count != 3 for count in per_state.values())):
        raise ValueError("conditioning bridge requires the full accepted 404-state/three-view cache")
    return bank


@torch.no_grad()
def identical_predictions(base, model, restored, codec, bank, adapter):
    rows = []
    for recipe, item in sorted(bank.items()):
        features = batch_features([{"features": item["features"]}])
        predictions = [network(features) for network in (base, model, restored)]
        tensors = [{"continuous": value["continuous"], **{f"category_{head}": scores for head, scores in enumerate(value["categorical"])}}
                   for value in predictions]
        if any(any(not torch.equal(value[key], tensors[0][key]) for key in value) for value in tensors[1:]):
            raise ValueError("conditioning initialization or cold export changes a starting prediction")
        targets = [adapter.export(value["continuous"], value["categorical"])[0] for value in predictions]
        presets = [prediction_preset(network, codec, target) for network, target in zip((base, model, restored), targets, strict=True)]
        if targets[1:] != [targets[0], targets[0]] or any(value != presets[0] for value in presets[1:]):
            raise ValueError("conditioning bridge changes a decoded preset")
        rows.append({"recipe_id": recipe, "state_id": item["state_id"], "features_sha256": tensor_digest(item["features"]),
                     "audio_sha256": item["audio_sha256"], "raw_prediction_sha256": tensor_digest(tensors[0]), "preset_sha256": sha256_bytes(presets[0])})
    return rows


def native_access(model, adapter, bank, direction):
    plan, cases = read(direction / "plan.json"), read(direction / "cases.json")
    if len(cases) != 24 or len(plan["cases"]) != 24:
        raise ValueError("conditioning bridge requires all 24 qualified native points")
    result = []
    for descriptor, case in zip(plan["cases"], cases, strict=True):
        key = descriptor["index"], descriptor["role"], descriptor["state_id"]
        if key != (case["index"], case["role"], case["state_id"]):
            raise ValueError("native qualification case ordering changed")
        item = bank[descriptor["input_recipe_id"]]
        if (item["state_id"] != descriptor["state_id"] or tensor_digest(item["features"]) != descriptor["input_features_sha256"]
                or item["audio_sha256"] != descriptor["input_audio_sha256"]):
            raise ValueError("native qualification input changed")
        result.append({"index": case["index"], "role": case["role"], "state_id": case["state_id"],
                       **gradient_access(model, item["features"], adapter, case["direction"])})
    if Counter(row["role"] for row in result) != Counter(original_input=12, wider_input=12):
        raise ValueError("native qualification role coverage changed")
    return result


def run_bridge(parent, initial, cache, direction, pool_path, output, *, execute=False):
    parent, initial, cache, direction, pool_path, output = map(Path, (parent, initial, cache, direction, pool_path, output))
    started, monitor = perf_counter(), MemoryMonitor()
    monitor.start()
    try:
        frozen = inputs(parent, initial, cache, direction, pool_path)
        if not execute:
            output.mkdir(parents=True, exist_ok=False)
            write_json(output / "plan.json", frozen)
        else:
            if read(output / "plan.json") != frozen or (output / "summary.json").exists():
                raise ValueError("preserve the previous bridge or changed frozen input")
            with refresh.cpu_settings():
                saved = torch.load(initial, map_location="cpu", weights_only=True)
                base, frontend, codec = restore_model(saved)
                pool = PilotPool.load(pool_path)
                if pool.pool_id != saved["training_pool_id"] or pool.codec.to_dict() != codec.to_dict():
                    raise ValueError("conditioning bridge changed the training population or codec")
                if (artifact_rendering(saved)["render_phase_controls"] != FULL_PHASE_CONTROL
                        or artifact_rendering(saved)["render_profile"]["preroll_frames"] != 32768):
                    raise ValueError("conditioning bridge requires the settled fixed-phase profile")
                bank = training_bank(torch.load(cache, map_location="cpu", weights_only=True), pool)
                model = make_conditioned(base, codec, seed=frozen["seed"])
                artifact = bridge_state(model, frontend, codec, pool.pool_id)
                target = output / "fit-000/model.pt"
                target.parent.mkdir()
                atomic_torch_save(target, artifact)
                model_file_digest = digest(target)
                restored, restored_frontend, restored_codec = restore_conditioned(torch.load(target, map_location="cpu", weights_only=True))
                if (restored_frontend.describe() != frontend.describe() or restored_codec.to_dict() != codec.to_dict()
                        or model_rendering(restored) != model_rendering(base)):
                    raise ValueError("conditioned export changed its frontend, codec or phase/profile contract")
                adapter = ProductionControlAdapter(codec, scales=read(direction / "plan.json")["adapter"]["fixed_scales"])
                predictions = identical_predictions(base, model, restored, codec, bank, adapter)
                gradients = native_access(model, adapter, bank, direction)
                if digest(target) != model_file_digest:
                    raise ValueError("conditioning artifact changed")
            if inputs(parent, initial, cache, direction, pool_path) != frozen:
                raise ValueError("conditioning bridge changed immutable parent inputs")
    finally:
        monitor.close()
    resources = {"elapsed_seconds": perf_counter() - started, "peak_process_tree_rss_bytes": monitor.peak_bytes, "rss_sampling_errors": monitor.errors}
    if (resources["elapsed_seconds"] >= LIMITS["wall_seconds"] or monitor.peak_bytes is None
            or monitor.peak_bytes > LIMITS["process_tree_bytes"] or monitor.errors):
        raise ValueError("conditioning bridge exceeds five minutes or four GiB")
    if not execute:
        report = {"plan_id": canonical_sha256(frozen), "preparation_resources": resources, "new_native_requests": 0, "new_optimizer_updates": 0}
        write_json(output / "preparation.json", report)
        return report
    report = {"plan_id": canonical_sha256(frozen), "initialization_and_gradient_access_passed": True,
              "starting_predictions_and_presets_identical": len(predictions), "distinct_training_recordings": 1212,
              "training_states": 404, "native_gradient_cases": len(gradients),
              "responding_continuous_checks": sum(value is not None for row in gradients for value in row["continuous_projection_gradient_l1"]),
              "responding_category_checks": sum(value is not None for row in gradients for value in row["category_projection_gradient_l1"]),
              "parent_artifacts_unchanged": True, "base_weights_frozen": True, "new_native_requests": 0, "new_optimizer_updates": 0,
              "qualified_for_release": False, "useful_conditioning_training": False, "test_split_used": False, "resources": resources,
              "model_budget": COUNTS, "model_file": "fit-000/model.pt", "model_file_sha256": digest(target),
              "model_sha256": tensor_digest(model.state_dict()), "policy": model.conditioning_training_policy,
              "prediction_rows": predictions, "gradient_rows": gradients}
    before_summary = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    publication_bound = before_summary + 2 * len(json.dumps(report, indent=2, sort_keys=True).encode("utf-8")) + 65536
    if publication_bound > LIMITS["archive_bytes"]:
        raise ValueError("conditioning bridge exceeds its 64-MiB evidence ceiling")
    report["archive_bytes_before_summary"] = before_summary
    report["archive_bound_with_publication_bytes"] = publication_bound
    write_json(output / "summary.json", report)
    return {key: value for key, value in report.items() if key not in ("prediction_rows", "gradient_rows", "policy")}

"""Bind a zero-render category derivative check to refreshed basic initialization."""

from contextlib import contextmanager
from collections import Counter
import copy
import inspect
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import torch

from obruxo_data.hashing import canonical_sha256, sha256_bytes

from .audit import write_json
from .benchmark import MemoryMonitor
from .category_jacobian import JACOBIAN_GATES, check_jacobians
from .category_signal import fit_scales, scaled_logits
from .evaluate import load_evaluation_checkpoint
from .features import tensor_digest
from .inference_profile import prediction_preset
from .model import batch_features, predicted_targets
from .profiled_sampler import profile_metadata
from .temporal_proxy_data import category_balanced_entries
from .trainer import load_model

CALIBRATION = {"states": 32, "observations": 128, "source_selection": "category_balanced_training_states_original_held_original_alternate",
               "fit": "max_one_centered_logits_rms_float64",
               "temperatures": [1., .5], "draws_per_input_and_temperature": 16, "seed": 0}
INITIALIZATION = {"experiment": "settled_basic_full_bootstrap_v1", "records": 404, "states": 404, "updates": 16000}
LIMITS = {"wall_seconds": 300, "process_tree_bytes": 4 * 1024**3, "archive_bytes": 32 * 1024**2,
          "native_requests": 0, "optimizer_updates": 0}
SOURCE_NAMES = ("category_scale_refresh.py", "category_jacobian.py", "category_signal.py", "spectral_controls.py",
                "model.py", "features.py", "reproducibility.py", "profiled_sampler.py", "trainer.py", "evaluate.py",
                "native_fit.py", "category_calibration.py", "input_performances.py", "paired.py", "inference_profile.py",
                "mixed_features.py", "mixed_input_performances.py")


def digest(path):
    return sha256_bytes(Path(path).read_bytes())


@contextmanager
def cpu_settings():
    threads, deterministic = torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()
    with torch.random.fork_rng(devices=[]):
        try:
            torch.set_num_threads(2)
            torch.use_deterministic_algorithms(True)
            yield
        finally:
            torch.set_num_threads(threads)
            torch.use_deterministic_algorithms(deterministic)


def source_identity():
    return {name: sha256_bytes(Path(__file__).with_name(name).read_bytes().replace(b"\r\n", b"\n")) for name in SOURCE_NAMES}


def runtime_identity():
    return {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__, "cpu_threads": 2,
            "gumbel_sha256": sha256_bytes(inspect.getsource(torch.nn.functional.gumbel_softmax).encode())}


def model_identity(model, frontend, codec):
    sizes = [len(options) for _name, options in codec.categorical]
    return {"model_sha256": tensor_digest(model.state_dict()), "frontend_sha256": tensor_digest(frontend.basic_pitch.state_dict()),
            "continuous_heads": len(codec.continuous), "categorical_sizes": sizes, "dense_outputs": len(codec.continuous) + sum(sizes),
            "trainable_parameters": sum(parameter.numel() for parameter in model.parameters())}


def load_inputs(pool, initial_path, retrieval_path, qualification_path, input_cache_path):
    from .native_fit import require_qualification
    from .mixed_features import validate_cache
    from .mixed_input_performances import MixedSampler
    from .stream import StreamConfig
    from .trainer import TrainConfig
    qualification = json.loads(qualification_path.read_text())
    require_qualification(qualification, preroll_frames=32768)
    saved, _, _, _ = load_evaluation_checkpoint(retrieval_path, pool)
    model, frontend, codec = load_model(initial_path)
    initial = torch.load(initial_path, map_location="cpu", weights_only=True)
    expected_profile = profile_metadata(32768)
    if (any(initial.get(key) != value for key, value in expected_profile.items())
            or saved["run_contract"]["training"].get("preroll_frames") != 32768
            or saved["renderer_identity"] != qualification["renderer_identity"]
            or tensor_digest(model.state_dict()) != tensor_digest(saved["model"])
            or tensor_digest(frontend.basic_pitch.state_dict()) != tensor_digest(saved["basic_pitch"])
            or initial["training_pool_id"] != pool.pool_id or codec.to_dict() != pool.codec.to_dict()):
        raise ValueError("category refresh needs matching corrected-render basic model, checkpoint and native qualification")
    training = saved["run_contract"]["training"]
    counts = Counter(item["state_id"] for item in saved["replay"])
    if (len(saved["replay"]) != INITIALIZATION["records"] or saved["step"] != INITIALIZATION["updates"]
            or len(counts) != INITIALIZATION["states"] or set(counts.values()) != {1}
            or training.get("sampling_policy") != "state_round_robin"):
        raise ValueError("category refresh requires the retained accepted full initialization")
    cache = torch.load(input_cache_path, map_location="cpu", weights_only=True)
    sampler, stream = MixedSampler(pool), StreamConfig(prefetch_clips=1)
    config = TrainConfig(replay_capacity=sampler.epoch_size, replay_mib=64, batch_size=4, render_budget=sampler.epoch_size,
                         cpu_threads=2, prediction_scope="basic", preroll_frames=32768, sampling_policy="paired_views", input_performances="mixed")
    validate_cache(cache, pool, sampler, frontend, config, stream)
    if (cache["origins"]["original"]["checkpoint_sha256"] != digest(retrieval_path)
            or cache["renderer_identity"] != qualification["renderer_identity"]):
        raise ValueError("category input recordings belong to another initializer or renderer")
    entries = category_balanced_entries(pool, "train", CALIBRATION["states"])
    by_state = {entry.state_id: [] for entry in entries}
    for item in cache["replay"]:
        if item["state_id"] in by_state:
            by_state[item["state_id"]].append(item)
    selected = [item for entry in entries for item in sorted(by_state[entry.state_id], key=lambda item: item["sequence_id"])]
    if len(selected) != CALIBRATION["observations"] or any(len(rows) != 4 for rows in by_state.values()):
        raise ValueError("category check lacks the full weighted original and wider input distribution")
    inputs = []
    for item in selected:
        recipe = sampler.recipe_at(item["sequence_id"])
        inputs.append({"state_id": item["state_id"], "sequence_id": item["sequence_id"], "recipe_id": item["recipe_id"],
                       "performance_id": recipe.performance_id, "input_view": recipe.input_view,
                       "audio_sha256": item["audio_sha256"], "features_sha256": tensor_digest(item["features"])})
    return model, frontend, codec, selected, inputs


def replay_audit_passed(audit):
    return (audit["model_optimizer_batch_and_global_rng_exact"] and audit["replayed_updates"] == INITIALIZATION["updates"]
            and audit["training_source_receipts_verified"] == INITIALIZATION["records"] and audit["native_renders_added"] == 0
            and audit["all_eighty_audit_requests_reconstructed"]
            and audit["native_source_repeat_pairs_exact"] == 16 and audit["hard_patch_request_exports_reconstructed"] == 8
            and audit["hard_categories_exact"] and audit["cached_vs_fresh_input_predictions_max_normalized_delta"] <= 1e-6)


def make_plan(pool, bootstrap_root, bootstrap_audit_path, qualification_path, input_cache_path):
    summary = json.loads((bootstrap_root / "summary.json").read_text())
    prior_plan = json.loads((bootstrap_root / "plan.json").read_text())
    audit = json.loads(bootstrap_audit_path.read_text())
    cache = torch.load(input_cache_path, map_location="cpu", weights_only=True)
    if (cache["origins"]["original"]["audit_sha256"] != digest(bootstrap_audit_path)
            or cache["origins"]["original"]["checkpoint_sha256"] != digest(bootstrap_root / "fit-000/checkpoint.pt")):
        raise ValueError("category input cache does not refer to the retained accepted initializer")
    if (prior_plan["experiment"] != INITIALIZATION["experiment"] or not summary["complete"] or not summary["preflight_passed"]
            or summary["plan_id"] != canonical_sha256(prior_plan) or audit["plan_id"] != summary["plan_id"]
            or not replay_audit_passed(audit)):
        raise ValueError("category refresh requires the retained accepted initialization and exact replay audit")
    for name, expected in audit["evidence_sha256"].items():
        if digest(bootstrap_root / name) != expected:
            raise ValueError("full bootstrap evidence changed")
    fit_root = sorted(bootstrap_root.glob("fit-*"))[-1]
    with cpu_settings():
        bundle = load_inputs(pool, fit_root / "model.pt", fit_root / "checkpoint.pt", qualification_path, input_cache_path)
        model, frontend, codec, _, inputs = bundle
        inference = model_identity(model, frontend, codec)
    return {"format": "basic_category_jacobian_check_v2", "pool_id": pool.pool_id, "manifest_id": pool.manifest.manifest_id,
            "initial_sha256": digest(fit_root / "model.pt"), "retrieval_sha256": digest(fit_root / "checkpoint.pt"),
            "initial_artifact": (fit_root / "model.pt").relative_to(bootstrap_root).as_posix(),
            "retrieval_artifact": (fit_root / "checkpoint.pt").relative_to(bootstrap_root).as_posix(),
            "qualification_sha256": digest(qualification_path), "bootstrap_audit_sha256": digest(bootstrap_audit_path),
            "input_cache_sha256": digest(input_cache_path),
            "bootstrap_plan_id": summary["plan_id"], "bootstrap_audit_id": canonical_sha256(audit),
            "initialization": INITIALIZATION.copy(),
            "inference": inference,
            "training_inputs": inputs, "calibration": copy.deepcopy(CALIBRATION), "limits": LIMITS.copy(), "derivative_gates": JACOBIAN_GATES.copy(),
            "source": source_identity(), "runtime": runtime_identity(), **profile_metadata(32768),
            "test_split_used": False}, bundle, audit


def mathematical_evidence(model, frontend, codec, items, inputs):
    before = tensor_digest(model.state_dict()), tensor_digest(frontend.basic_pitch.state_dict())
    with torch.no_grad():
        predictions = [model.eval()(batch_features([item])) for item in items]
    continuous = torch.cat([row["continuous"] for row in predictions])
    logits = [torch.cat([row["categorical"][head] for row in predictions]) for head in (0, 1)]
    scales = fit_scales(logits)
    records = []
    for metadata, prediction in zip(inputs, predictions, strict=True):
        original = predicted_targets(prediction, codec)[0]
        scaled = predicted_targets({**prediction, "categorical": scaled_logits(prediction["categorical"], scales)}, codec)[0]
        if original != scaled or prediction_preset(model, codec, original) != prediction_preset(model, codec, scaled):
            raise ValueError("category scaling changed the initial production patch")
        records.append({**metadata, "raw_logits": [row[0].tolist() for row in prediction["categorical"]],
                        "continuous": prediction["continuous"][0].tolist(), "hard_patch_sha256": sha256_bytes(prediction_preset(model, codec, original)),
                        "hard_patch_unchanged": True})
    rng = torch.get_rng_state().clone()
    derivatives = check_jacobians(continuous, logits, scales)
    if not torch.equal(rng, torch.get_rng_state()) or before != (tensor_digest(model.state_dict()), tensor_digest(frontend.basic_pitch.state_dict())):
        raise ValueError("category observation changed model/frontend weights or sampler RNG")
    return {"initial_training_inputs": records, "fixed_scales": list(scales), "derivatives": derivatives,
            "model_frontend_and_rng_unchanged": True}


def run_check(pool, bootstrap_root, bootstrap_audit_path, qualification_path, input_cache_path, output, *, execute=False):
    started, monitor = perf_counter(), MemoryMonitor()
    monitor.start()
    try:
        return _run_check(pool, bootstrap_root, bootstrap_audit_path, qualification_path, input_cache_path, output, execute, started, monitor)
    finally:
        monitor.close()


def _run_check(pool, bootstrap_root, bootstrap_audit_path, qualification_path, input_cache_path, output, execute, started, monitor):
    plan, bundle, audit = make_plan(pool, bootstrap_root, bootstrap_audit_path, qualification_path, input_cache_path)
    if output.exists():
        if json.loads((output / "plan.json").read_text()) != plan:
            raise ValueError("frozen refreshed-category plan changed")
    else:
        output.mkdir(parents=True)
        write_json(output / "plan.json", plan)
    if not execute:
        return {"plan_id": canonical_sha256(plan), "executed": False}
    if (output / "run.json").exists():
        raise ValueError("category check already started; preserve its evidence")
    write_json(output / "run.json", {"plan_id": canonical_sha256(plan)})
    try:
        with cpu_settings():
            model, frontend, codec, items, inputs = bundle
            evidence = mathematical_evidence(model, frontend, codec, items, inputs)
        result = {"plan": plan, "plan_id": canonical_sha256(plan), "bootstrap_audit": audit, **evidence,
                  "elapsed_seconds": perf_counter() - started, "peak_process_tree_rss_bytes": monitor.peak_bytes,
                  "rss_sampling_errors": monitor.errors, "native_requests": 0, "optimizer_updates": 0,
                  "failure": None, "test_split_used": False, "qualified_for_release": False}
        result["within_resource_limits"] = (result["elapsed_seconds"] < LIMITS["wall_seconds"] and monitor.peak_bytes is not None
                                            and monitor.peak_bytes < LIMITS["process_tree_bytes"] and monitor.errors == 0)
        result["evidence_bytes_upper_bound"] = 2 * len(json.dumps(result, sort_keys=True, indent=2).encode()) + sum(
            path.stat().st_size for path in output.rglob("*") if path.is_file()) + 1024
        result["within_resource_limits"] = result["within_resource_limits"] and result["evidence_bytes_upper_bound"] < LIMITS["archive_bytes"]
        result["passed"] = evidence["derivatives"]["passed"] and result["within_resource_limits"]
        write_json(output / "summary.json", result)
        return {key: value for key, value in result.items() if key not in {"plan", "bootstrap_audit", "derivatives", "initial_training_inputs"}}
    except BaseException as error:
        write_json(output / "status.json", {"complete": False, "error_type": type(error).__name__})
        raise


def validate_refresh(path, pool, initial_path, retrieval_path, qualification_path, input_cache_path):
    try:
        return _validate_refresh(path, pool, initial_path, retrieval_path, qualification_path, input_cache_path)
    except (KeyError, TypeError) as error:
        raise ValueError("refreshed category evidence is incomplete or malformed") from error


def _validate_refresh(path, pool, initial_path, retrieval_path, qualification_path, input_cache_path):
    report = json.loads(path.read_text())
    plan = report["plan"]
    if (plan["format"] != "basic_category_jacobian_check_v2" or report["plan_id"] != canonical_sha256(plan)
            or not report["passed"] or report["failure"] is not None or not report["within_resource_limits"]
            or not report["derivatives"]["passed"] or report["qualified_for_release"]
            or report["native_requests"] != 0 or report["optimizer_updates"] != 0 or report["test_split_used"]
            or plan["calibration"] != CALIBRATION or plan["limits"] != LIMITS or plan["source"] != source_identity()
            or plan["initialization"] != INITIALIZATION
            or plan["derivative_gates"] != JACOBIAN_GATES
            or plan["runtime"] != runtime_identity() or plan["pool_id"] != pool.pool_id or plan["manifest_id"] != pool.manifest.manifest_id
            or plan["initial_sha256"] != digest(initial_path) or plan["retrieval_sha256"] != digest(retrieval_path)
            or plan["qualification_sha256"] != digest(qualification_path)
            or plan["input_cache_sha256"] != digest(input_cache_path)
            or any(plan.get(key) != value for key, value in profile_metadata(32768).items())
            or plan["bootstrap_audit_id"] != canonical_sha256(report["bootstrap_audit"])):
        raise ValueError("refreshed category evidence is incomplete, changed or belongs to other artifacts")
    audit = report["bootstrap_audit"]
    if (not replay_audit_passed(audit)
            or audit["plan_id"] != plan["bootstrap_plan_id"]
            or audit["evidence_sha256"].get(plan["initial_artifact"]) != plan["initial_sha256"]
            or audit["evidence_sha256"].get(plan["retrieval_artifact"]) != plan["retrieval_sha256"]):
        raise ValueError("refreshed category evidence lacks its completed bootstrap replay")
    if (not 0 < report["elapsed_seconds"] < LIMITS["wall_seconds"] or report["peak_process_tree_rss_bytes"] is None
            or not 0 < report["peak_process_tree_rss_bytes"] < LIMITS["process_tree_bytes"] or report["rss_sampling_errors"] != 0
            or not 0 < report["evidence_bytes_upper_bound"] < LIMITS["archive_bytes"] or path.stat().st_size > LIMITS["archive_bytes"]):
        raise ValueError("refreshed category check lacks valid resource measurements")
    with cpu_settings():
        model, frontend, codec, items, inputs = load_inputs(pool, initial_path, retrieval_path, qualification_path, input_cache_path)
        if inputs != plan["training_inputs"]:
            raise ValueError("refreshed category training inputs changed")
        if model_identity(model, frontend, codec) != plan["inference"]:
            raise ValueError("refreshed model or frontend evidence changed")
        expected = mathematical_evidence(model, frontend, codec, items, inputs)
    if not expected["derivatives"]["passed"] or any(report[key] != value for key, value in expected.items()):
        raise ValueError("refreshed category constants or derivative evidence cannot be reconstructed")
    return {"format": "fixed_train_only_category_scales_v1", "scales": report["fixed_scales"],
            "input_cache_sha256": plan["input_cache_sha256"],
            "diagnosis_file_sha256": digest(path), "diagnosis_plan_id": report["plan_id"], "diagnosis_format": plan["format"],
            "fit": plan["calibration"], "loss_arms": ["parameter", "spectral"],
            "application": "raw_logits_divided_by_fixed_scales_before_both_training_losses",
            "inference": "raw_argmax_unchanged", "initial_weights_changed": False}

"""Train-only derivative/export qualification of deterministic category choices."""

import copy
import json
from pathlib import Path
from time import perf_counter

import torch

from obruxo_data.hashing import canonical_sha256, sha256_bytes

from . import category_scale_refresh as refresh
from .audit import write_json
from .benchmark import MemoryMonitor
from .category_jacobian import JACOBIAN_GATES
from .category_signal import fit_scales
from .features import tensor_digest
from .inference_profile import prediction_preset
from .model import batch_features, predicted_targets
from .production_controls import ProductionControlAdapter

LIMITS = {"wall_seconds": 600, "process_tree_bytes": 4 * 1024**3, "archive_bytes": 32 * 1024**2,
          "native_requests": 0, "optimizer_updates": 0}
CALIBRATION = {"states": 32, "observations": 128, "distinct_observations": 96,
               "inputs": "original_held_original_alternate", "fit": "max_one_centered_logits_rms_float64",
               "temperatures": [1., .5], "random_draws": 0}


def source_identity():
    names = ("production_controls.py", "production_choice_check.py", "native_inverse.py", "native_gradient.py", "codec.py", "benchmark.py")
    paths = {name: Path(__file__).with_name(name) for name in names}
    paths["reports/scripts/check_production_choices.py"] = Path(__file__).parents[1] / "reports/scripts/check_production_choices.py"
    return {**refresh.source_identity(), **{name: sha256_bytes(path.read_bytes().replace(b"\r\n", b"\n")) for name, path in paths.items()}}


def predictions(model, items):
    frozen = copy.deepcopy(model).eval()
    with torch.no_grad():
        return [frozen(batch_features([item])) for item in items]


def mathematical_evidence(model, frontend, codec, items, inputs):
    if not items or len(items) != len(inputs):
        raise ValueError("production-choice check needs paired input identities and features")
    before = tensor_digest(model.state_dict()), tensor_digest(frontend.basic_pitch.state_dict())
    rng = torch.get_rng_state().clone()
    rows = predictions(model, items)
    scales = fit_scales([torch.cat([row["categorical"][head] for row in rows]) for head in (0, 1)])
    adapter = ProductionControlAdapter(codec, scales=scales)
    records, points = [], []
    for index, (metadata, prediction) in enumerate(zip(inputs, rows, strict=True)):
        continuous = prediction["continuous"]
        original = predicted_targets(prediction, codec)[0]
        if adapter.export(continuous, prediction["categorical"]) != [original]:
            raise ValueError("production export differs from raw model argmax")
        records.append({**metadata, "continuous": continuous[0].tolist(),
                        "raw_logits": [scores[0].tolist() for scores in prediction["categorical"]],
                        "hard_patch_sha256": sha256_bytes(prediction_preset(model, codec, original)), "hard_patch_unchanged": True})
        expected_encoding = adapter(continuous, prediction["categorical"], sample=False)
        for temperature in CALIBRATION["temperatures"]:
            raw = [scores.clone().requires_grad_(True) for scores in prediction["categorical"]]
            encoded = adapter(continuous, raw, temperature=temperature)
            if not torch.equal(encoded.detach(), expected_encoding):
                raise ValueError("training forward patch differs from production")
            heads = []
            for head, (scores, scale, offset) in enumerate(zip(raw, scales, (9, 12), strict=True)):
                selected = encoded[:, offset:offset + scores.shape[1]]
                actual = torch.stack([torch.autograd.grad(selected[0, option], scores, retain_graph=True)[0][0]
                                      for option in range(scores.shape[1])])

                def formula(divisor):
                    probability = (scores.detach() / divisor / temperature).softmax(-1)[0]
                    return (probability.diag() - probability[:, None] * probability[None, :]) / divisor / temperature

                expected, unscaled = formula(scale), formula(1.)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-8)
                if not torch.isfinite(actual).all():
                    raise ValueError("nonfinite production-choice derivative")
                heads.append({"head": head, "hard_choice": int(selected.detach().argmax(-1).item()),
                              "jacobian": actual.tolist(), "original_norm": float(unscaled.double().norm()),
                              "scaled_norm": float(actual.double().norm()), "maximum_formula_error": float((actual - expected).abs().max())})
            points.append({"row": index, "temperature": temperature, "heads": heads, "forward_matches_production": True})
    gates, heads = JACOBIAN_GATES.copy(), []
    for head in (0, 1):
        temperatures = []
        for temperature in CALIBRATION["temperatures"]:
            selected = [point["heads"][head] for point in points if point["temperature"] == temperature]
            original = torch.tensor([row["original_norm"] for row in selected], dtype=torch.float64)
            scaled = torch.tensor([row["scaled_norm"] for row in selected], dtype=torch.float64)
            raw_median, median = float(original.quantile(.5)), float(scaled.quantile(.5))
            fraction = float((scaled >= gates["matrix_norm_floor"]).double().mean())
            gain = gates["weak_median_gain"] if raw_median < gates["weak_raw_median_threshold"] else gates["healthy_median_gain"]
            passed = (bool((scaled > 0).all()) and fraction >= gates["fraction_above_floor"]
                      and median >= gates["median_norm_floor"] and median >= raw_median * gain)
            temperatures.append({"temperature": temperature, "points": len(selected), "original_median_norm": raw_median,
                                 "scaled_median_norm": median, "fraction_above_connectivity_floor": fraction,
                                 "exact_zero_points": int((scaled == 0).sum()), "pass": passed})
        heads.append({"head": head, "temperatures": temperatures, "pass": all(row["pass"] for row in temperatures)})
    if before != (tensor_digest(model.state_dict()), tensor_digest(frontend.basic_pitch.state_dict())) or not torch.equal(rng, torch.get_rng_state()):
        raise ValueError("production-choice observation changed weights or RNG")
    return {"fixed_scales": list(scales), "initial_training_inputs": records, "adapter": adapter.contract(),
            "derivatives": {"points": points, "heads": heads, "gates": gates, "passed": all(row["pass"] for row in heads)},
            "model_frontend_and_rng_unchanged": True, "training_forward_matches_production": True}


def make_plan(pool, bootstrap_root, bootstrap_audit_path, qualification_path, input_cache_path):
    prior, bundle, audit = refresh.make_plan(pool, bootstrap_root, bootstrap_audit_path, qualification_path, input_cache_path)
    choice = {key: value for key, value in ProductionControlAdapter().contract().items() if key != "fixed_scales"}
    return {**prior, "format": "basic_production_choice_check_v1", "source": source_identity(), "limits": LIMITS.copy(), "choice_policy": choice,
            "calibration": copy.deepcopy(CALIBRATION), "derivative_gates": JACOBIAN_GATES.copy(),
            "scope": "derivative_export_contract_only_native_direction_and_model_quality_unqualified"}, bundle, audit


def run_check(pool, bootstrap_root, bootstrap_audit_path, qualification_path, input_cache_path, output, *, execute=False):
    started, monitor = perf_counter(), MemoryMonitor()
    started_here = False
    monitor.start()
    try:
        with refresh.cpu_settings():
            plan, bundle, audit = make_plan(pool, bootstrap_root, bootstrap_audit_path, qualification_path, input_cache_path)
            if output.exists():
                if json.loads((output / "plan.json").read_text()) != plan:
                    raise ValueError("frozen production-choice plan changed")
            else:
                output.mkdir(parents=True)
                write_json(output / "plan.json", plan)
            if not execute:
                preparation = {"plan_id": canonical_sha256(plan), "executed": False, "elapsed_seconds": perf_counter() - started,
                               "peak_process_tree_rss_bytes": monitor.peak_bytes, "rss_sampling_errors": monitor.errors,
                               "native_requests": 0, "optimizer_updates": 0}
                preparation["within_resource_limits"] = (0 < preparation["elapsed_seconds"] < LIMITS["wall_seconds"]
                    and monitor.peak_bytes is not None and 0 < monitor.peak_bytes < LIMITS["process_tree_bytes"] and monitor.errors == 0)
                if (output / "preparation.json").exists():
                    raise ValueError("production-choice preparation already recorded; preserve evidence")
                write_json(output / "preparation.json", preparation)
                if not preparation["within_resource_limits"]:
                    raise ValueError("production-choice preparation exceeded its resource limits")
                return preparation
            if (output / "run.json").exists():
                raise ValueError("production-choice check already started; preserve evidence")
            write_json(output / "run.json", {"plan_id": canonical_sha256(plan)})
            started_here = True
            evidence = mathematical_evidence(*bundle)
        result = {"plan": plan, "plan_id": canonical_sha256(plan), "bootstrap_audit": audit, **evidence,
                  "elapsed_seconds": perf_counter() - started, "peak_process_tree_rss_bytes": monitor.peak_bytes,
                  "rss_sampling_errors": monitor.errors, "native_requests": 0, "optimizer_updates": 0,
                  "failure": None, "test_split_used": False, "qualified_for_release": False, "native_direction_qualified": False}
        bound = 2 * len(json.dumps(result, sort_keys=True, indent=2).encode()) + sum(path.stat().st_size for path in output.rglob("*") if path.is_file()) + 2048
        result["evidence_bytes_upper_bound"] = bound
        result["within_resource_limits"] = (0 < result["elapsed_seconds"] < LIMITS["wall_seconds"] and monitor.peak_bytes is not None
                                            and 0 < monitor.peak_bytes < LIMITS["process_tree_bytes"] and monitor.errors == 0 and bound < LIMITS["archive_bytes"])
        result["passed"] = evidence["derivatives"]["passed"] and result["within_resource_limits"]
        write_json(output / "summary.json", result)
        return {key: value for key, value in result.items() if key not in {"plan", "bootstrap_audit", "derivatives", "initial_training_inputs"}}
    except BaseException as error:
        if started_here:
            write_json(output / "status.json", {"complete": False, "error_type": type(error).__name__})
        raise
    finally:
        monitor.close()


def validate_check(path, pool, bootstrap_root, bootstrap_audit_path, qualification_path, input_cache_path):
    """Rebuild inputs, constants, every matrix and hard export; a pass flag is insufficient."""
    try:
        report = json.loads(path.read_text())
        with refresh.cpu_settings():
            plan, bundle, audit = make_plan(pool, bootstrap_root, bootstrap_audit_path, qualification_path, input_cache_path)
            if (report["plan"] != plan or report["plan_id"] != canonical_sha256(plan) or report["bootstrap_audit"] != audit
                    or not report["passed"] or report["failure"] is not None or not report["within_resource_limits"]
                    or report["native_requests"] != 0 or report["optimizer_updates"] != 0 or report["test_split_used"]
                    or report["qualified_for_release"] or report["native_direction_qualified"]
                    or not 0 < report["elapsed_seconds"] < LIMITS["wall_seconds"]
                    or not 0 < report["peak_process_tree_rss_bytes"] < LIMITS["process_tree_bytes"] or report["rss_sampling_errors"] != 0
                    or not 0 < report["evidence_bytes_upper_bound"] < LIMITS["archive_bytes"] or path.stat().st_size > LIMITS["archive_bytes"]):
                raise ValueError("production-choice evidence is changed, incomplete or outside its scope")
            expected = mathematical_evidence(*bundle)
        if not expected["derivatives"]["passed"] or any(report[key] != value for key, value in expected.items()):
            raise ValueError("production-choice derivative/export evidence cannot be reconstructed")
        return {"format": plan["format"], "plan_id": report["plan_id"], "report_sha256": refresh.digest(path),
                "scales": report["fixed_scales"], "adapter": report["adapter"], "native_direction_qualified": False}
    except (KeyError, TypeError) as error:
        raise ValueError("production-choice evidence is incomplete or malformed") from error


def run_validation(output, *inputs):
    path = output / "verification.json"
    if path.exists():
        raise ValueError("production-choice verification already recorded; preserve evidence")
    started, monitor = perf_counter(), MemoryMonitor()
    monitor.start()
    try:
        result = validate_check(output / "summary.json", *inputs)
    finally:
        monitor.close()
    elapsed = perf_counter() - started
    within = (0 < elapsed < LIMITS["wall_seconds"] and monitor.peak_bytes is not None
              and 0 < monitor.peak_bytes < LIMITS["process_tree_bytes"] and monitor.errors == 0)
    evidence = {**result, "elapsed_seconds": elapsed, "peak_process_tree_rss_bytes": monitor.peak_bytes,
                "rss_sampling_errors": monitor.errors, "within_resource_limits": within, "verified": within,
                "native_requests": 0, "optimizer_updates": 0, "qualified_for_release": False}
    archive_bound = sum(item.stat().st_size for item in output.rglob("*") if item.is_file()) + 2 * len(json.dumps(evidence, indent=2).encode()) + 2048
    evidence["archive_bytes_upper_bound"] = archive_bound
    evidence["within_resource_limits"] = evidence["verified"] = within and archive_bound < LIMITS["archive_bytes"]
    write_json(path, evidence)
    if not evidence["within_resource_limits"]:
        raise ValueError("production-choice verification exceeded its resource limits")
    return evidence

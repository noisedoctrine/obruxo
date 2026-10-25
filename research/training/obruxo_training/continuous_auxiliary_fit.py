"""Fresh matched native fit with the qualified paired continuous-label auxiliary."""

import copy
import math
from pathlib import Path
from time import perf_counter

import torch

from obruxo_data.hashing import canonical_sha256, sha256_bytes

from . import category_scale_refresh as refresh
from .audit import write_json
from .basic_direct_fit import execute
from .basic_direct_fit_audit import run_verification
from .basic_direct_fit_data import Inputs, make_plan as base_plan
from .benchmark import MemoryMonitor
from .conditioned_bridge import read
from .continuous_auxiliary import SPEC
from .features import tensor_digest
from .native_fit import digest


def make_plan(parent, bridge, recent, cache, pool, prior_fit, diagnosis):
    proof, prior = read(prior_fit / "verification.json"), read(prior_fit / "plan.json")
    inspected, inspection = read(diagnosis / "summary.json"), read(diagnosis / "plan.json")
    if (not proof["verified"] or not proof["complete"] or not proof["offline_training_replay_exact"] or not proof["native_receipts_resolved"]
            or proof["comparison_waveforms_rescored"] != 312 or proof["committed_updates_per_arm"] != 48 or proof["matched_comparison_passed"]
            or proof["test_split_used"] or proof["qualified_for_release"]
            or inspected["plan_id"] != canonical_sha256(inspection) or inspection["fit_plan_id"] != proof["plan_id"]
            or inspection["fit_verification_sha256"] != digest(prior_fit / "verification.json")
            or not inspected["within_resource_limits"] or not inspected["alignment"]["model_adam_rng_replay_exact"]
            or inspected["native_requests"] != 0 or inspected["experimental_training_updates"] != 0
            or inspection["auxiliary"] != SPEC or inspected["alignment_counts"]["paired_outputs_positive"] != 96):
        raise ValueError("continuous auxiliary fit requires the audited direct fit and frozen exact-replay diagnosis")
    neural = inspected["alignment"]["neural_steps"]
    if (inspected["alignment"]["offline_replayed_updates_per_arm"] != 48 or len(neural) != 48
            or len(inspected["alignment"]["paired_output_observations"]) != 96
            or any(not math.isfinite(row["gradient_cosine"]) or not math.isfinite(row["auxiliary_to_native_norm_ratio"])
                   or 1 + row["gradient_cosine"] * row["auxiliary_to_native_norm_ratio"] <= 0 for row in neural)):
        raise ValueError("fixed continuous auxiliary lacks all 48 inspected local sound-descent directions")
    plan = base_plan(parent, bridge, recent, cache, pool)
    if any(digest(prior_fit / name) != value for name, value in proof["evidence_sha256"].items()):
        raise ValueError("earlier direct-fit evidence changed")
    if any(inspection["input_files"][f"fit/{name}"] != value for name, value in proof["evidence_sha256"].items()):
        raise ValueError("auxiliary diagnosis used different direct-fit evidence")
    for key in ("input_cache_sha256", "initial_file_sha256", "initial_model_sha256", "pool_id", "training", "fit_audit_cells", "regression_state_ids",
                "render_profile", "render_phase_controls", "expected_renderer_identity", "updates", "optimizer"):
        if plan[key] != prior[key]:
            raise ValueError("auxiliary comparison changed its initialization, observations or matched budget")
    root = Path(__file__).parents[1]
    # The diagnosis ran under its frozen historical fitter. Bind its new helper unchanged.
    if inspection["source"]["continuous_auxiliary.py"] != sha256_bytes(Path(__file__).with_name("continuous_auxiliary.py").read_bytes().replace(b"\r\n", b"\n")):
        raise ValueError("qualified continuous auxiliary contract changed")
    prefix = torch.load(prior_fit / "prefix-checkpoint.pt", map_location="cpu", weights_only=True)
    reference = proof["result"]["original_regression"]
    files = {f"prior_fit/{name}": digest(prior_fit / name) for name in ("verification.json", "plan.json", "prefix-checkpoint.pt")}
    files.update({f"diagnosis/{name}": digest(diagnosis / name) for name in ("summary.json", "plan.json")})
    names = ("continuous_auxiliary.py", "continuous_auxiliary_fit.py", "reports/scripts/run_continuous_auxiliary_fit.py")
    code = {name: sha256_bytes((root / name if "/" in name else Path(__file__).with_name(name)).read_bytes().replace(b"\r\n", b"\n")) for name in names}
    return {**plan, "format": "basic_native_sound_with_continuous_auxiliary_v1", "source": {**plan["source"], **code},
            "input_files": {**plan["input_files"], **files}, "continuous_auxiliary": copy.deepcopy(SPEC),
            "loss": "native_sound_reference_rehearsal_plus_source_active_paired_continuous_mse",
            "prior_fit_plan_id": proof["plan_id"], "diagnosis_plan_id": inspected["plan_id"],
            "expected_parameter_model_sha256": proof["final_model_sha256"]["parameter"],
            "expected_parameter_prefix_model_sha256": tensor_digest(prefix["models"]["parameter"]),
            "earlier_sound_reference": {"plan_id": proof["plan_id"], "model_sha256": proof["final_model_sha256"]["spectral"],
                                        "means": reference["means"]["spectral"], "mrstft_p90": reference["mrstft_p90"]["spectral"]},
            "gates": {**plan["gates"], "earlier_sound_original_32_mean_and_p90_nonregression": True}}


def run(parent, bridge, recent, cache, pool, prior_fit, diagnosis, output, *, mode="prepare"):
    started, monitor = perf_counter(), MemoryMonitor()
    monitor.start()
    with refresh.cpu_settings():
        try:
            plan = make_plan(parent, bridge, recent, cache, pool, prior_fit, diagnosis)
            inputs = Inputs(parent, bridge, cache, plan)
            inputs.validate()
        finally:
            monitor.close()
        preparation = {"elapsed_seconds": perf_counter() - started, "peak_process_tree_rss_bytes": monitor.peak_bytes,
                       "rss_sampling_errors": monitor.errors, "native_requests": 0, "experimental_training_updates": 0}
        if preparation["elapsed_seconds"] >= 600 or not monitor.peak_bytes or monitor.peak_bytes > 4 * 1024**3 or monitor.errors:
            raise ValueError("continuous auxiliary preparation exceeded ten minutes or 4 GiB")
        if mode == "prepare":
            output.mkdir(parents=True, exist_ok=False)
            write_json(output / "plan.json", plan)
            write_json(output / "preparation.json", preparation)
            return {"plan_id": canonical_sha256(plan), "preparation": preparation, "storage_bound": plan["storage_bound"]}
        if mode == "verify":
            return run_verification(inputs, output)
        if mode not in ("execute", "resume"):
            raise ValueError("unknown continuous auxiliary fit mode")
        return execute(inputs, output, resume=mode == "resume")

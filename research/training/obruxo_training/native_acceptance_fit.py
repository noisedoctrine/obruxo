"""Fresh matched sound fitting with measured acceptance and weight rollback."""

import copy
from pathlib import Path
from time import perf_counter

import torch

from obruxo_data.hashing import canonical_sha256, sha256_bytes

from .audit import write_json
from .basic_direct_fit import SoundStore, execute, replay_training
from .basic_direct_fit_audit import run_verification
from .basic_direct_fit_data import Inputs, make_plan as base_plan
from .benchmark import MemoryMonitor
from .category_scale_refresh import cpu_settings
from .conditioned_bridge import read
from .features import tensor_digest
from .native_acceptance import SPEC
from .native_fit import digest
from .protected_native_check import GATES


def metadata_bound(prior, storage, limits):
    sizes = {name: max(map(len, (prior / name).read_bytes().splitlines())) + 1 for name in ("renders.jsonl", "probes.jsonl")}
    measured = 4 * (2 * sum(limits[key] for key in ("training_attempts", "qualification_attempts")) * sizes["renders.jsonl"]
                    + limits["training_attempts"] * sizes["probes.jsonl"])
    original = storage["native_probe_ledger_and_publication_reserve_bytes"]
    extra = max(0, measured - original) + 17 * 1024**2
    result = {**storage, "native_probe_ledger_and_publication_reserve_bytes": max(original, measured),
              "acceptance_trial_trace_allowance_bytes": 16 * 1024**2, "acceptance_bank_and_weight_allowance_bytes": 1024**2,
              "resident_replay_upper_bound_bytes": storage["resident_replay_upper_bound_bytes"] + 1024**2,
              "estimated_archive_upper_bound_bytes": storage["estimated_archive_upper_bound_bytes"] + extra,
              "measured_prior_maximum_json_line_bytes": sizes, "serialized_metadata_safety_factor": 4}
    if result["estimated_archive_upper_bound_bytes"] > limits["archive_bytes"] or result["resident_replay_upper_bound_bytes"] > limits["replay_bytes"]:
        raise ValueError("measured native-acceptance evidence or replay exceeds its allocation")
    return result


def make_plan(parent, bridge, recent, cache, pool, prior_fit, native_check):
    proof, prior = read(prior_fit / "verification.json"), read(prior_fit / "plan.json")
    inspected, native_plan = read(native_check / "verification.json"), read(native_check / "plan.json")
    if (not proof["verified"] or not proof["complete"] or not proof["offline_training_replay_exact"] or not proof["native_receipts_resolved"]
            or proof["comparison_waveforms_rescored"] != 312 or proof["committed_updates_per_arm"] != 48 or proof["matched_comparison_passed"]
            or proof["test_split_used"] or proof["qualified_for_release"] or not inspected["verified"]
            or inspected["plan_id"] != canonical_sha256(native_plan) or inspected["comparison_waveforms_rescored"] != 74
            or inspected["optimizer_applications"] != 0 or inspected["experimental_training_updates"] != 0
            or inspected["test_split_used"] or inspected["qualified_for_release"] or native_plan["gates"] != GATES
            or inspected["result"]["native_acceptance_screen_passed"] or inspected["result"]["repeat_exact"] != [True, True]
            or len(inspected["result"]["points"]) != 12
            or any(row["objective"]["after"] >= row["objective"]["before"] for row in inspected["result"]["points"])):
        raise ValueError("native-accepted fitting requires audited prior learning and the unchanged rejected protection comparison")
    for source, verified in ((prior_fit, proof), (native_check, inspected)):
        if any(digest(source / name) != value for name, value in verified["evidence_sha256"].items()):
            raise ValueError("audited source evidence changed before accepted fitting")
    plan = base_plan(parent, bridge, recent, cache, pool)
    for key in ("input_cache_sha256", "initial_file_sha256", "initial_model_sha256", "pool_id", "training", "fit_audit_cells", "regression_state_ids",
                "render_profile", "render_phase_controls", "expected_renderer_identity", "updates", "optimizer"):
        if plan[key] != prior[key]:
            raise ValueError("native acceptance changed its initializer, inputs or matched update schedule")
    if any(native_plan[key] != prior[key] for key in ("render_profile", "render_phase_controls", "expected_renderer_identity")):
        raise ValueError("native acceptance changed the tested rendering profile")
    root, differences = Path(__file__).parents[1], {}
    for name, value in native_plan["source"].items():
        path = root / name if "/" in name else Path(__file__).with_name(name)
        current = sha256_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
        if current != value:
            if name not in {"basic_direct_fit.py", "basic_direct_fit_audit.py"}:
                raise ValueError("numerical model, sound loss or renderer changed after native comparison")
            differences[name] = {"historical": value, "current": current}
    names = ("native_acceptance.py", "native_acceptance_fit.py", "reports/scripts/run_native_acceptance_fit.py")
    code = {name: sha256_bytes((root / name if "/" in name else Path(__file__).with_name(name)).read_bytes().replace(b"\r\n", b"\n")) for name in names}
    prefix = torch.load(prior_fit / "prefix-checkpoint.pt", map_location="cpu", weights_only=True)
    reference = proof["result"]["original_regression"]
    limits = {**plan["limits"], "wall_seconds": 21600, "training_attempts": 7344}
    files = {f"prior_fit/{name}": digest(prior_fit / name) for name in ("verification.json", "plan.json", "prefix-checkpoint.pt")}
    files.update({f"native_check/{name}": digest(native_check / name) for name in ("verification.json", "plan.json", "summary.json")})
    return {**plan, "format": "basic_actual_native_acceptance_fit_v1", "source": {**plan["source"], **code}, "input_files": {**plan["input_files"], **files},
            "native_acceptance": SPEC, "prior_fit_plan_id": proof["plan_id"], "rejected_protection_check_plan_id": inspected["plan_id"],
            "compatible_adapter_source_differences": differences, "limits": limits, "storage_bound": metadata_bound(prior_fit, plan["storage_bound"], limits),
            "planned_native_requests": {"gradient": 6336, "trial": 864, "qualification": 208, "total": 7408},
            "expected_parameter_model_sha256": proof["final_model_sha256"]["parameter"],
            "expected_parameter_prefix_model_sha256": tensor_digest(prefix["models"]["parameter"]),
            "earlier_sound_reference": {"plan_id": proof["plan_id"], "model_sha256": proof["final_model_sha256"]["spectral"],
                                        "means": reference["means"]["spectral"], "mrstft_p90": reference["mrstft_p90"]["spectral"]},
            "gates": {**plan["gates"], "minimum_native_accepted_updates": 36, "earlier_sound_original_32_mean_and_p90_nonregression": True}}


def compatible_prior(inputs, prior_fit):
    original = copy.copy(inputs)
    original.plan = read(prior_fit / "plan.json")
    store = SoundStore(prior_fit, original)
    saved = torch.load(prior_fit / "checkpoint.pt", map_location="cpu", weights_only=True)
    replay_training(original, store, saved, read(prior_fit / "prefix-verification.json"))
    if store.used_probes != set(store.probes):
        raise ValueError("compatible default replay did not use every prior scalar probe")
    return {"prior_model_adam_rng_replay_exact": True, "replayed_updates_per_arm": saved["step"],
            "replayed_optimizer_applications": 2 * saved["step"], "scalar_probes_reused": len(store.used_probes)}


def run(parent, bridge, recent, cache, pool, prior_fit, native_check, output, *, mode="prepare"):
    started, monitor = perf_counter(), MemoryMonitor()
    monitor.start()
    with cpu_settings():
        try:
            plan = make_plan(parent, bridge, recent, cache, pool, prior_fit, native_check)
            inputs = Inputs(parent, bridge, cache, plan)
            inputs.validate()
            compatibility = compatible_prior(inputs, prior_fit)
        finally:
            monitor.close()
        preparation = {"elapsed_seconds": perf_counter() - started, "peak_process_tree_rss_bytes": monitor.peak_bytes,
                       "rss_sampling_errors": monitor.errors, "native_requests": 0, "experimental_training_updates": 0, **compatibility}
        if preparation["elapsed_seconds"] >= 600 or not monitor.peak_bytes or monitor.peak_bytes > 4 * 1024**3 or monitor.errors:
            raise ValueError("native-acceptance preparation exceeded ten minutes or 4 GiB")
        if mode == "prepare":
            output.mkdir(parents=True, exist_ok=False)
            write_json(output / "plan.json", plan)
            write_json(output / "preparation.json", preparation)
            return {"plan_id": canonical_sha256(plan), "preparation": preparation, "storage_bound": plan["storage_bound"]}
        if read(output / "plan.json") != plan:
            raise ValueError("native-acceptance fitting plan changed")
        if mode == "verify":
            return run_verification(inputs, output)
        if mode not in ("execute", "resume"):
            raise ValueError("unknown native-acceptance fit mode")
        return execute(inputs, output, resume=mode == "resume")

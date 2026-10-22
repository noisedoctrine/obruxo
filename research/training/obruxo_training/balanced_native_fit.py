"""Matched sound fitting with original/wider inputs and independent development panels."""

from pathlib import Path

import torch

from obruxo_data.hashing import sha256_bytes

from .balanced_sound_data import cached_training_records, cross_view_updates, development_panels
from .category_calibration import load_calibration
from .mixed_input_performances import MixedSampler
from .native_fit import digest, make_plan as legacy_plan, run_plan
from .native_fit_storage import storage_bound
from .temporal_proxy_data import category_balanced_entries
from .training_coverage import expand_training


def compose_plan(pool, base, cache, calibration):
    original = category_balanced_entries(pool, "train", 32)
    entries, population = expand_training(pool, original)
    training = cached_training_records(pool, entries, cache, MixedSampler(pool))
    validation, panels = development_panels(pool, category_balanced_entries(pool, "validation", 12))
    updates = cross_view_updates(len(training))
    storage = storage_bound(training, validation, pool.codec.contract.tail_seconds)
    archive_cap = 4 * 1024**3
    if storage["estimated_archive_upper_bound_bytes"] + 32 * 1024**2 > archive_cap:
        raise ValueError("balanced sound-fit evidence exceeds its declared archive cap")
    source, gradients, fit, development = 3 * len(training), 22 * len(updates), 12 * len(training), 14 * len(validation)
    return {**base, "format": "matched_native_inverse_fit_v4", "training": training, "validation": validation,
            "training_population": population, "training_view_roles": ["original_input", "wide_held_input", "wide_alternate_input"],
            "validation_panels": panels, "updates": updates, "storage_bound": storage, "category_training": calibration,
            "input_schedule": "original_to_held_held_to_original_original_to_alternate_alternate_to_original_v1",
            "optimizer": {**base["optimizer"], "updates_per_arm": len(updates)},
            "limits": {**base["limits"], "training_attempts": source + gradients + 16,
                       "qualification_attempts": fit + development + 8, "optimizer_attempts": len(updates) + 4,
                       "wall_seconds": 21600, "archive_bytes": archive_cap},
            "planned_maximum_requests": {"training_targets": source, "gradient_probes": gradients, "fit_audit": fit,
                                         "validation_audit": development, "total": source + gradients + fit + development},
            "gates": {**base["gates"], "retained_original_fit_required": True, "each_validation_panel_required": True}}


def make_plan(pool, initial_path, retrieval_path, qualification_path, cache_path, calibration_path):
    base = legacy_plan(pool, initial_path, retrieval_path, qualification_path)
    calibration = load_calibration(calibration_path, pool, initial_path, qualification_path,
                                   retrieval_path=retrieval_path, input_cache_path=cache_path)
    if (calibration.get("diagnosis_format") != "basic_category_jacobian_check_v2"
            or calibration.get("input_cache_sha256") != digest(cache_path)):
        raise ValueError("balanced sound fit needs category feedback checked against these input recordings")
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    plan = compose_plan(pool, base, cache, calibration)
    names = ("balanced_native_fit.py", "balanced_sound_data.py", "native_fit_storage.py", "training_coverage.py",
             "category_calibration.py", "category_scale_refresh.py", "category_jacobian.py", "category_signal.py",
             "mixed_features.py", "mixed_input_performances.py", "input_performances.py", "profiled_sampler.py", "native_fit_report.py")
    paths = [Path(__file__).with_name(name) for name in names]
    paths += [Path(__file__).parents[1] / "reports/scripts/run_balanced_native_fit.py"]
    paths += [Path(__file__).parents[1] / "reports/scripts/verify_native_inverse_fit.py"]
    plan["source"].update({path.name: sha256_bytes(path.read_bytes().replace(b"\r\n", b"\n")) for path in paths})
    plan["input_cache_sha256"] = digest(cache_path)
    return plan


def run_fit(pool, initial_path, retrieval_path, qualification_path, cache_path, calibration_path, output, *, execute=False, resume=False):
    plan = make_plan(pool, initial_path, retrieval_path, qualification_path, cache_path, calibration_path)
    return run_plan(pool, initial_path, retrieval_path, output, plan, execute=execute, resume=resume)

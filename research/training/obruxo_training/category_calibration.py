"""Validate fixed category-training scales against their completed real-data preflight."""

import inspect
import json
from pathlib import Path

import torch

from obruxo_data.hashing import canonical_sha256, sha256_bytes

from .category_signal import fit_scales


def file_digest(path):
    return sha256_bytes(Path(path).read_bytes())


def load_calibration(path, pool, initial_path, qualification_path):
    report = json.loads(path.read_text())
    plan = report["plan"]
    required = ("passed", "historical_replay_unchanged", "within_resource_limits", "all_saved_derivatives_reconstructed", "actual_training_points_verified")
    if (plan["format"] != "basic_category_signal_check_v1" or report["plan_id"] != canonical_sha256(plan)
            or not all(report[name] for name in required) or report["failure"] is not None
            or report["actual_replayed_optimizer_updates"] != 256 or len(report["points"]) != 128
            or len(report["heads"]) != 2 or not all(head["pass"] for head in report["heads"])):
        raise ValueError("category calibration lacks the required completed derivative preflight")
    encoded_pool = (json.dumps({"pool_id": pool.pool_id, "pool": pool.to_dict()}, ensure_ascii=True,
                              allow_nan=False, sort_keys=True, indent=2) + "\n").encode()
    if (sha256_bytes(encoded_pool) != plan["pool_sha256"] or file_digest(initial_path) != plan["initial_sha256"]
            or file_digest(qualification_path) != plan["qualification_sha256"]):
        raise ValueError("category calibration belongs to different pool, initial weights or qualification")
    source = Path(__file__).with_name("category_signal.py").read_bytes().replace(b"\r\n", b"\n")
    if (sha256_bytes(source) != plan["source"]["category_signal.py"]
            or sha256_bytes(inspect.getsource(torch.nn.functional.gumbel_softmax).encode()) != plan["gumbel_implementation_sha256"]):
        raise ValueError("qualified category scaling or Gumbel implementation changed")
    initial = report["initial_training_inputs"]
    if len(initial) != 32 or not all(row["hard_patch_unchanged"] for row in initial):
        raise ValueError("category calibration lacks unchanged initial hard patches")
    scales = fit_scales([torch.tensor([row["raw_logits"][head] for row in initial], dtype=torch.float32) for head in (0, 1)])
    if list(scales) != report["fixed_scales"]:
        raise ValueError("category calibration scales differ from the frozen training inputs")
    return {"format": "fixed_train_only_category_scales_v1", "scales": list(scales),
            "diagnosis_file_sha256": file_digest(path), "diagnosis_plan_id": report["plan_id"],
            "fit": plan["calibration"], "loss_arms": ["parameter", "spectral"],
            "application": "raw_logits_divided_by_fixed_scales_before_both_training_losses",
            "inference": "raw_argmax_unchanged", "initial_weights_changed": False}

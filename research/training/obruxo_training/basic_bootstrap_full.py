"""Freeze broader basic initialization only after the small corrected-render check passes."""

from dataclasses import asdict, replace
import json
from pathlib import Path

from obruxo_data.hashing import canonical_sha256, sha256_bytes

from . import basic_bootstrap
from .profiled_sampler import training_sampler
from .trainer import TrainConfig


def make_plan(pool, qualification_path, preflight_root, preflight_audit):
    prior = json.loads((preflight_root / "summary.json").read_text())
    audit = json.loads(preflight_audit.read_text())
    prior_plan = json.loads((preflight_root / "plan.json").read_text())
    if (not prior["complete"] or not prior["preflight_passed"] or prior["plan_id"] != canonical_sha256(prior_plan)
            or prior_plan["pool_id"] != pool.pool_id or audit["plan_id"] != prior["plan_id"]
            or not audit["model_optimizer_batch_and_global_rng_exact"] or audit["replayed_updates"] != 800
            or audit["native_renders_added"] != 0 or prior["attempts"] != 88
            or not audit["all_eighty_audit_requests_reconstructed"]):
        raise ValueError("full bootstrap requires the completed small fit and exact replay audit")
    required = {"summary.json", "plan.json", "fit-000/checkpoint.pt", "fit-000/checkpoint.initial.pt", "fit-000/model.pt"}
    if not required <= audit["evidence_sha256"].keys():
        raise ValueError("small bootstrap audit is missing its model/checkpoint evidence")
    for name, expected in audit["evidence_sha256"].items():
        if sha256_bytes((preflight_root / name).read_bytes()) != expected:
            raise ValueError("small bootstrap evidence changed after verification")
    plan = basic_bootstrap.make_plan(pool, qualification_path)
    config = replace(TrainConfig(**plan["config"]), steps=16000, replay_capacity=404, replay_mib=32, batch_size=32,
                     checkpoint_every=250, render_budget=404)
    sampler = training_sampler(pool, config)
    if sampler.epoch_size != 404:
        raise ValueError("this full bootstrap card requires the original 404-state training population")
    recipes = [sampler.recipe_at(index) for index in range(404)]
    if len({recipe.state_id for recipe in recipes}) != 404:
        raise ValueError("full bootstrap contains repeated source states")
    plan.update(experiment="settled_basic_full_bootstrap_v1", config=asdict(config),
                sources=[{"sequence_id": recipe.sequence_id, "state_id": recipe.state_id, "recipe_id": recipe.recipe_id,
                          "request_id": recipe.render_request().request_id, "performance_id": recipe.performance_id} for recipe in recipes],
                limits={"wall_seconds": 7200, "process_tree_bytes": 4 * 1024**3, "archive_bytes": 512 * 1024**2,
                        "training_attempts": 406, "evaluation_attempts": 90, "optimizer_updates": 16000},
                planned_requests={"training": 404, "evaluation": 80, "total": 484},
                recovery="explicit_resume_original_deadline_up_to_two_source_and_ten_evaluation_extra_attempts",
                prior_preflight={"plan_id": prior["plan_id"], "summary_sha256": sha256_bytes((preflight_root / "summary.json").read_bytes()),
                                 "audit_sha256": sha256_bytes(preflight_audit.read_bytes())})
    plan["gates"].pop("trainer_tiny_overfit")
    plan["gates"]["parameter_fit"] = {"continuous_mae_below": .06, "categorical_accuracy": 1., "final_initial_loss_ratio_below": .05}
    plan["source"]["basic_bootstrap_full.py"] = sha256_bytes(Path(__file__).read_bytes().replace(b"\r\n", b"\n"))
    return plan


def run_full(pool, qualification_path, preflight_root, preflight_audit, output, *, execute=False, resume=False):
    plan = make_plan(pool, qualification_path, preflight_root, preflight_audit)
    return basic_bootstrap.run_frozen_plan(pool, plan, output, execute=execute, resume=resume)

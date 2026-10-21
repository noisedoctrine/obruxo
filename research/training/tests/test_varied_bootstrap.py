from dataclasses import asdict
import copy
import json

import pytest
import torch

from obruxo_data.hashing import sha256_bytes
from obruxo_training import varied_bootstrap as bootstrap
from obruxo_training.checkpoints import load_training_checkpoint
from obruxo_training.features import tensor_digest
from obruxo_training.pilot import PilotPool
from obruxo_training.profiled_sampler import training_sampler
from obruxo_training.trainer import TrainConfig, train_pilot
from profile_renderer import ProfileRendererFactory


def test_acoustic_admission_requires_wide_gain_and_original_preservation_in_both_roles():
    plan = {"panels": {name: [0, 1] for name in ("original", "wide")},
            "gates": {"wide_candidate_baseline_ratio_below_both_roles": .9,
                      "original_candidate_baseline_ratio_at_most_both_roles": 1.1, "candidate_template_ratio_below_all_panels_roles": .9}}
    audits = {panel: {"aggregates": {method: {role: {"scored": 2, "planned": 2, "mean_mrstft": score}
                                             for role in ("reference", "transfer")}
                                    for method, score in (("baseline", 1.), ("varied", .8), ("fixed_template", 2.), ("source_repeat", 0.))}}
              for panel in plan["panels"]}
    assert bootstrap.acoustic_gates(plan, audits)["acoustic_gates_passed"]
    for panel, role, value in (("wide", "transfer", .95), ("original", "reference", 1.11)):
        changed = copy.deepcopy(audits)
        changed[panel]["aggregates"]["varied"][role]["mean_mrstft"] = value
        assert not bootstrap.acoustic_gates(plan, changed)["acoustic_gates_passed"]
    changed = copy.deepcopy(audits)
    changed["wide"]["aggregates"]["varied"]["reference"]["scored"] = 1
    with pytest.raises(ValueError, match="incomplete"):
        bootstrap.acoustic_gates(plan, changed)
    for audit in audits.values():
        for method in ("baseline", "varied", "fixed_template"):
            for score in audit["aggregates"][method].values():
                score["mean_mrstft"] = 0.
    assert not bootstrap.acoustic_gates(plan, audits)["acoustic_gates_passed"]  # Zero/zero is not an improvement.


def test_varied_fit_comparison_keeps_cases_distinct_and_resume_adds_no_native_work(tmp_path):
    pool = PilotPool.create(24, seed=0)
    initial_root = tmp_path / "initial"
    initial_config = TrainConfig(steps=2, replay_capacity=2, batch_size=2, checkpoint_every=1, render_budget=2, cpu_threads=2,
                                 prediction_scope="basic", preroll_frames=32768, sampling_policy="state_round_robin")
    initial = train_pilot(pool, initial_root / "fit-000", config=initial_config, renderer_factory=ProfileRendererFactory())
    config = TrainConfig(steps=2, replay_capacity=4, batch_size=4, checkpoint_every=1, render_budget=4, cpu_threads=2,
                         prediction_scope="basic", preroll_frames=32768, sampling_policy="paired_views", input_performances="varied")
    sampler, old_sampler = training_sampler(pool, config), training_sampler(pool, initial_config)
    recipes = [sampler.recipe_at(index) for index in range(4)]
    old_by_state = {old_sampler.recipe_at(index).state_id: old_sampler.recipe_at(index) for index in range(old_sampler.epoch_size)}
    panels = {"original": [old_by_state[recipe.state_id] for recipe in recipes[::2]], "wide": recipes[::2]}
    plan = {"pool_id": pool.pool_id, "config": asdict(config), "renderer_identity": initial["renderer_identity"],
            "initial_model": {"file_sha256": sha256_bytes((initial_root / "fit-000/model.pt").read_bytes()),
                              "model_state_sha256": initial["model_state_sha256"]},
            "sources": [{"recipe_id": recipe.recipe_id} for recipe in recipes],
            "panels": {panel: [{"sequence_id": recipe.sequence_id} for recipe in members] for panel, members in panels.items()},
            "gates": {"wide_candidate_baseline_ratio_below_both_roles": .9, "original_candidate_baseline_ratio_at_most_both_roles": 1.1,
                      "candidate_template_ratio_below_all_panels_roles": .9},
            "limits": {"wall_seconds": 600, "process_tree_bytes": 4 * 1024**3, "archive_bytes": 128 * 1024**2,
                       "training_attempts": 4, "evaluation_attempts": 48}}
    output = tmp_path / "run"
    summary = bootstrap.run(pool, plan, initial_root, output, execute=True, renderer_factory=ProfileRendererFactory())
    assert summary["complete"] and summary["all_outcomes_accepted"] and summary["attempts"] == 52
    assert not summary["passed"] and not summary["qualified_for_release"]  # The fixture ignores controls; no audible gain.
    for panel, audit in summary["audits"].items():
        assert audit["renders_attempted"] == 24 and len({row["state_id"] for row in audit["cases"]}) == 2
        frozen = json.loads((output / f"audit-{panel}/plan.json").read_text())
        assert frozen["training_sequence_ids"] == [row["sequence_id"] for row in plan["panels"][panel]]
    checkpoint = load_training_checkpoint(output / "fit-000/checkpoint.pt")
    start = torch.load(output / "fit-000/checkpoint.initial.pt", weights_only=True)["state"]
    assert tensor_digest(start["model"]) == initial["model_state_sha256"]
    assert checkpoint["run_contract"]["initial_model"]["file_sha256"] == plan["initial_model"]["file_sha256"]
    resumed = bootstrap.run(pool, plan, initial_root, output, execute=True, resume=True, renderer_factory=ProfileRendererFactory())
    assert resumed["attempts"] == 52 and resumed["model_state_sha256"] == summary["model_state_sha256"]
    assert all(audit["renders_attempted_this_segment"] == 0 for audit in resumed["audits"].values())

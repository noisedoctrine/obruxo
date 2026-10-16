"""One fixed direct-render improvement/distillation comparison, without oracle inference."""

from dataclasses import asdict
import json
from pathlib import Path

import torch

from obruxo_data.hashing import canonical_sha256, sha256_bytes

from .audit import write_json
from .candidate_data import CandidateConfig, collect_candidates, training_panel
from .candidate_learning import train_candidates
from .evaluate import evaluate_comparison, load_evaluation_checkpoint
from .features import tensor_digest
from .proxy_screen import balanced_entries, source_identity


def run_candidate_screen(pool, checkpoint, output, *, config=None, renderer_factory=None, resume=False, progress=None):
    config = config or CandidateConfig()
    torch.set_num_threads(config.cpu_threads)
    source, _, _, _ = load_evaluation_checkpoint(checkpoint, pool)
    training = training_panel(pool, source, config.training_states)
    validation = balanced_entries(pool, "validation", config.validation_states)
    plan = {"format": "obruxo_render_selected_screen_v1", "pool_id": pool.pool_id, "source": source_identity(),
            "source_checkpoint_sha256": sha256_bytes(Path(checkpoint).read_bytes()), "config": asdict(config),
            "source_model_sha256": tensor_digest(source["model"]), "training_states": [entry.state_id for entry in training],
            "validation_states": [entry.state_id for entry in validation], "arms": {"control": 0., "render_selected": config.candidate_weight},
            "selection_limit": "single_seed_development_comparison_no_release_promotion",
            "render_budget": {"training_maximum": 7 * config.training_states + 14,
                              "validation_maximum": 12 * config.validation_states + 12,
                              "maximum_total": 7 * config.training_states + 12 * config.validation_states + 26},
            "source_supervision": "retained_as_anchor_in_both_arms", "known_midi": "train_native_scorer_and_evaluator_only",
            "final_test_used": False, "audio_files_written": 0}
    output = Path(output)
    if resume:
        if json.loads((output / "plan.json").read_text()) != plan:
            raise ValueError("candidate screen plan changed; cannot resume")
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "plan.json", plan)
    write_json(output / "status.json", {"complete": False})
    data, collection = collect_candidates(pool, checkpoint, output / "data", config=config, renderer_factory=renderer_factory,
                                          resume=resume and (output / "data/data.pt").exists(), progress=progress)
    arms, paths = {}, {}
    for name, weight in plan["arms"].items():
        arms[name] = train_candidates(pool, source, data, output / name, config=config, auxiliary_weight=weight,
                                     resume=resume and (output / name / "checkpoint.pt").exists(), progress=progress)
        paths[name] = output / name / "checkpoint.pt"
    if arms["control"]["uses"] != arms["render_selected"]["uses"] or arms["control"]["batch_rng_sha256"] != arms["render_selected"]["batch_rng_sha256"]:
        raise ValueError("candidate arms did not use matched initialization and minibatches")
    comparison = evaluate_comparison(pool, paths, checkpoint, output / "comparison", count=config.validation_states,
                                     validation_states=plan["validation_states"], renderer_factory=renderer_factory,
                                     timeout=config.timeout, resume=resume and (output / "comparison/progress.json").exists(), progress=progress,
                                     render_attempt_limit=plan["render_budget"]["validation_maximum"])
    if collection["attempted"] + comparison["renders_attempted"] > plan["render_budget"]["maximum_total"]:
        raise ValueError("candidate screen exceeded its frozen native-attempt budget")
    summary = {"format": plan["format"], "plan_id": canonical_sha256(plan), "collection": collection, "arms": arms,
               "comparison": comparison["comparison"], "validation_renders_attempted": comparison["renders_attempted"],
               "validation_renders_accepted": comparison["renders_accepted"], "audio_files_written": 0,
               "inverse_training_input": "audio_only", "oracle_candidate_selection_used_at_inference": False,
               "final_test_used": False, "release_model_promoted": False}
    write_json(output / "summary.json", summary)
    write_json(output / "status.json", {"complete": True})
    return summary

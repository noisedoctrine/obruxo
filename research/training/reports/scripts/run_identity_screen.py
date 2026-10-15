"""One bounded paired-identity feasibility screen on the authored alpha vocabulary."""

import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import shutil
import sys

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]
CODE = ("training/obruxo_training/paired.py", "training/obruxo_training/model.py", "training/obruxo_training/pilot.py",
        "training/obruxo_training/paired_features.py",
        "training/obruxo_training/trainer.py", "training/obruxo_training/features.py", "training/obruxo_training/evaluate.py",
        "training/obruxo_training/stream.py", "data_generation/obruxo_data/render/vita.py",
        "training/reports/scripts/run_identity_screen.py")


def code_identity():
    return {name: hashlib.sha256((RESEARCH / name).read_bytes()).hexdigest() for name in CODE}


def main():
    import numpy as np
    import torch
    from obruxo_data.hashing import canonical_sha256
    from obruxo_training.audit import write_json
    from obruxo_training.evaluate import evaluate_comparison, retrieval_digest
    from obruxo_training.features import tensor_digest
    from obruxo_training.paired import complete_pairs
    from obruxo_training.paired_features import prepare_paired_features
    from obruxo_training.pilot import PilotPool, PilotSampler
    from obruxo_training.stream import NativeRendererFactory, StreamConfig
    from obruxo_training.trainer import TrainConfig, load_model, train_pilot
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plugin-path")
    parser.add_argument("--recover-checkpoint", type=Path, help="Recover an already rendered paired prefix; model/optimizer state is discarded")
    parser.add_argument("--recover-log", type=Path, help="Full render log belonging to the recovery checkpoint")
    parser.add_argument("--reuse-training", type=Path, help="Reuse completed matched arms under unchanged runtime code and arm settings")
    args = parser.parse_args()
    pool = PilotPool.load(args.pool)
    if pool.codec.generator != "independent_authored_alpha_v1":
        raise ValueError("identity screen requires the qualified alpha vocabulary")
    families = pool.codec.to_dict()["families"]
    panel = []
    for family in families:
        candidates = [entry.state_id for entry in pool.entries if pool.manifest.by_state[entry.state_id].split == "validation"
                      and pool.codec.family(entry.target) == family]
        if len(candidates) < 2:
            raise ValueError("identity screen needs two validation states per synthesis family")
        panel.extend(candidates[:2])
    sampler = PilotSampler(pool, policy="paired_views", seed=0)
    recipes = [sampler.recipe_at(index) for index in range(128)]
    if len({recipe.state_id for recipe in recipes}) != 64:
        raise ValueError("feature bank must contain exactly 64 distinct training states")
    base = TrainConfig(mode="overfit", steps=1, replay_capacity=128, replay_mib=8, batch_size=16, learning_rate=.001,
                       seed=0, checkpoint_every=1, render_budget=128, cpu_threads=4, sampling_policy="paired_views")
    configs = {name: replace(base, mode="online", steps=1000, checkpoint_every=1000, identity_weight=weight)
               for name, weight in (("control", 0.), ("identity", 1.))}
    reuse = args.reuse_training
    previous = json.loads((reuse / "plan.json").read_text()) if reuse else None
    if reuse and (any(previous["code"].get(name) != digest for name, digest in code_identity().items()
                      if name != "training/reports/scripts/run_identity_screen.py")
                  or previous["pool_id"] != pool.pool_id or previous["arms"] != {name: asdict(config) for name, config in configs.items()}
                  or previous["validation_states"] != panel):
        raise ValueError("completed training reuse requires unchanged runtime, pool, panel and arm settings")
    plan = {"version": 1, "screen": "paired_identity_alpha_v1", "pool_id": pool.pool_id,
            "pool_file_sha256": hashlib.sha256(args.pool.read_bytes()).hexdigest(), "code": code_identity(),
            "feature_bank": {"format": "obruxo_paired_features_v1", "states": 64, "attempts": 128, "replay_mib": 8},
            "arms": {name: asdict(config) for name, config in configs.items()},
            "feature_cache": {"policy": "complete_qa_qualified_pairs_only", "planned_states": 64, "attempts": 128,
                              "minimum_qualified_states": base.batch_size // 2, "fresh_target": "number_of_qualified_views",
                              "recovery_checkpoint_sha256": hashlib.sha256(args.recover_checkpoint.read_bytes()).hexdigest() if args.recover_checkpoint else None,
                              "recovery_log_sha256": hashlib.sha256(args.recover_log.read_bytes()).hexdigest() if args.recover_log else None},
            "training_recipe_ids": [recipe.recipe_id for recipe in recipes], "validation_states": panel,
            "validation_policy": "first_two_authored_validation_states_per_family_before_outcomes",
            "render_budget": {"feature_bank": 128, "comparison": 120, "total": 248},
            "reference_transfer_metric": "equal_weight_mean_mrstft_per_state", "identity_diagnostics": True,
            "paired_statistics": {"contrast": "identity_minus_control", "unit": "validation_state", "bootstrap_resamples": 2000,
                                  "seed": 0, "limit": "descriptive_single_seed_screen"},
            "selection_limit": "single_seed_feasibility_only_no_release_promotion", "test_split_evaluated": False}
    if reuse:
        plan["reused_training"] = {"plan_sha256": canonical_sha256(previous),
                                   "artifacts": {name: hashlib.sha256((reuse / name).read_bytes()).hexdigest()
                                                 for name in ("features/features.pt", "control/checkpoint.pt", "identity/checkpoint.pt")}}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "plan.json", plan)
    write_json(args.output / "validation-states.json", panel)
    write_json(args.output / "status.json", {"complete": False})
    factory, stream = NativeRendererFactory(args.plugin_path), StreamConfig(prefetch_clips=1, timeout_seconds=30)

    def progress(event):
        if event["stage"] != "render" or event["sequence_id"] % 16 == 15 or event["status"] != "accepted":
            print(json.dumps(event), flush=True)

    if reuse:
        for directory in ("features", "control", "identity"):
            destination = args.output / directory
            destination.mkdir()
            names = ("features.pt", "summary.json") if directory == "features" else ("checkpoint.pt", "checkpoint.previous.pt", "model.pt", "summary.json")
            for name in names:
                shutil.copyfile(reuse / directory / name, destination / name)
        bank = json.loads((args.output / "features/summary.json").read_text())
    else:
        bank = prepare_paired_features(pool, args.output / "features", states=64, stream_config=stream, renderer_factory=factory,
                                       recover_checkpoint=args.recover_checkpoint, recover_log=args.recover_log, progress=progress)
    source = args.output / "features/features.pt"
    source_saved = torch.load(source, weights_only=True)
    groups = complete_pairs(source_saved["replay"])
    targets = {entry.state_id: entry.target for entry in pool.entries}
    coverage = dict(Counter(pool.codec.family(targets[state]) for state in groups))
    if len(groups) < base.batch_size // 2 or set(coverage) != set(families):
        raise RuntimeError("qualified bank did not cover enough paired states and every synthesis family")
    results, saved, checkpoints = {}, {}, {}
    for name, config in configs.items():
        config = replace(config, fresh_examples=bank["retained_views"])
        results[name] = (json.loads((args.output / name / "summary.json").read_text()) if reuse else
                         train_pilot(pool, args.output / name, config=config, stream_config=stream,
                                     renderer_factory=factory, seed_replay=source, progress=progress))
        checkpoints[name] = args.output / name / "checkpoint.pt"
        saved[name] = torch.load(checkpoints[name], weights_only=True)
        if (not results[name]["complete"] or TrainConfig(**saved[name]["run_contract"]["training"]) != config
                or saved[name]["run_contract"]["replay_seed"]["checkpoint_sha256"] != hashlib.sha256(source.read_bytes()).hexdigest()):
            raise ValueError("completed arm differs from the frozen feature source or training settings")
        model, _, codec = load_model(args.output / name / "model.pt")
        if tensor_digest(model.state_dict()) != results[name]["model_state_sha256"] or codec.codec_id != pool.codec.codec_id:
            raise RuntimeError("exported arm differs from its trained model")
        if results[name]["totals"]["render_attempts"] != 0:
            raise RuntimeError("cached comparison arm unexpectedly rendered new training audio")
    left, right = saved.values()
    if (retrieval_digest(left["replay"]) != retrieval_digest(right["replay"])
            or not torch.equal(left["batch_rng"], right["batch_rng"])
            or [item["uses"] for item in left["replay"]] != [item["uses"] for item in right["replay"]]):
        raise RuntimeError("comparison arms did not consume identical features and minibatches")
    comparison = evaluate_comparison(pool, checkpoints, checkpoints["control"], args.output / "comparison", count=len(panel),
                                     renderer_factory=factory, timeout=30, validation_states=panel, identity_diagnostics=True,
                                     progress=lambda event: print(json.dumps(event), flush=True))
    if comparison["renders_accepted"] != 120 or comparison["renders_attempted"] != 120:
        raise RuntimeError("comparison did not meet the complete planned render denominator")
    if code_identity() != plan["code"]:
        raise RuntimeError("runtime sources changed during the frozen screen")
    scores = {name: np.array([sum(case["methods"][name][role]["mrstft"] for role in ("reference", "transfer")) / 2
                             for case in comparison["cases"]]) for name in configs}
    deltas = scores["identity"] - scores["control"]
    bootstrap = np.random.default_rng(0).choice(deltas, (2000, len(deltas)), replace=True).mean(1)
    per_family = {}
    for family in families:
        mask = np.array([pool.codec.family(targets[case["state_id"]]) == family for case in comparison["cases"]])
        per_family[family] = {"states": int(mask.sum()), "control": float(scores["control"][mask].mean()),
                              "identity": float(scores["identity"][mask].mean()), "identity_minus_control": float(deltas[mask].mean())}
    artifacts = ["features/features.pt", "control/checkpoint.pt", "identity/checkpoint.pt", "control/model.pt", "identity/model.pt",
                 "features/summary.json", "control/summary.json", "identity/summary.json", "comparison/summary.json"]
    summary = {"version": 1, "screen": plan["screen"], "plan_sha256": canonical_sha256(plan), "pool_id": pool.pool_id,
               "feature_bank": bank, "paired_states_by_family": coverage, "training": results,
               "comparison": comparison["comparison"], "identity_diagnostics": comparison["identity_diagnostics"],
               "paired_identity_minus_control": {"planned": len(panel), "scored": len(deltas), "mean_delta": float(deltas.mean()),
                                                  "ci95": np.quantile(bootstrap, [.025, .975]).tolist(),
                                                  "identity_wins": int((deltas < 0).sum()), "per_family": per_family},
               "features_and_minibatches_matched": True, "feature_retrieval_sha256": retrieval_digest(left["replay"]),
               "renders_requested": 248, "renders_accepted": bank["accepted"] + 120, "test_split_evaluated": False,
               "elapsed_seconds": bank["elapsed_seconds"] + sum(result["elapsed_seconds_this_segment"] for result in results.values())
                                  + comparison["elapsed_seconds"],
               "artifacts": {name: hashlib.sha256((args.output / name).read_bytes()).hexdigest() for name in artifacts},
               "audio_files_written": 0, "paid_compute": False, "promotion": "none_single_seed_feasibility_screen"}
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "status.json", {"complete": True})
    print(json.dumps({"comparison": summary["comparison"], "identity_diagnostics": summary["identity_diagnostics"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

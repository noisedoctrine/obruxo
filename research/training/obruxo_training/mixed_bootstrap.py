"""Joint replay of original and wider recordings with unchanged acoustic admission."""

from dataclasses import asdict
import json
from pathlib import Path

import torch

from obruxo_data.hashing import sha256_bytes

from .extended_features import features_for
from .mixed_features import digest, validate_cache
from .native_fit import require_qualification
from .profiled_sampler import training_sampler
from .stream import StreamConfig
from .trainer import TrainConfig
from .varied_bootstrap import run as run_comparison


def make_plan(pool, qualification_path, cache_root):
    cache_root = Path(cache_root)
    summary = json.loads((cache_root / "summary.json").read_text())
    feature_path = cache_root / "features.pt"
    if not summary["complete"] or digest(feature_path) != summary["cache_file_sha256"]:
        raise ValueError("mixed initialization requires completed unchanged feature preparation")
    saved = torch.load(feature_path, map_location="cpu", weights_only=True)
    if len(saved["replay"]) != 1616 or saved["distinct_parent_observations"] != 1212:
        raise ValueError("mixed initialization requires all original and wider training observations")
    config = TrainConfig(steps=12000, replay_capacity=1616, replay_mib=64, batch_size=32, learning_rate=.001,
                         seed=0, checkpoint_every=250, render_budget=1616, cpu_threads=2, prediction_scope="basic",
                         preroll_frames=32768, sampling_policy="paired_views", input_performances="mixed")
    sampler = training_sampler(pool, config)
    frontend = features_for(pool.codec, saved["basic_pitch"])
    validate_cache(saved, pool, sampler, frontend, config, StreamConfig(prefetch_clips=1))
    initial_root = Path(saved["origins"]["original"]["root"])
    wide_root = Path(saved["origins"]["wide"]["root"])
    plan = json.loads((wide_root / "plan.json").read_text())
    qualification = json.loads(qualification_path.read_text())
    require_qualification(qualification, preroll_frames=32768)
    if (plan["renderer_identity"] != qualification["renderer_identity"] or plan["pool_id"] != pool.pool_id
            or digest(qualification_path) != plan["qualification_sha256"]
            or digest(initial_root / "fit-000/model.pt") != plan["initial_model"]["file_sha256"]):
        raise ValueError("mixed initialization baseline or native qualification changed")
    held = {item["state_id"]: item for item in saved["replay"] if item["sequence_id"] % 4 == 1}
    wide_cases = [held[row["state_id"]] for row in plan["panels"]["original"]]
    names = sorted(set(plan["source"]) | {"mixed_bootstrap.py", "mixed_features.py", "mixed_input_performances.py"})
    plan.update(experiment="mixed_basic_initialization_v1", config=asdict(config), observation_contract=sampler.observation_contract,
                feature_cache={"file_sha256": digest(feature_path), "summary_sha256": digest(cache_root / "summary.json"),
                               "preparation_plan_id": summary["plan_id"], "weighted_records": 1616, "distinct_parent_observations": 1212},
                sources=[{key: item[key] for key in ("sequence_id", "state_id", "recipe_id", "performance_id", "origin")} for item in saved["replay"]],
                panels={"original": plan["panels"]["original"], "wide": [{key: item[key] for key in ("sequence_id", "state_id", "recipe_id", "performance_id")}
                                                                          for item in wide_cases]},
                limits={"wall_seconds": 3600, "process_tree_bytes": 4 * 1024**3, "archive_bytes": 512 * 1024**2,
                        "training_attempts": 0, "evaluation_attempts": 202, "optimizer_updates": 12000},
                planned_requests={"training": 0, "evaluation": 192, "total": 192},
                recovery="explicit_resume_original_deadline_zero_source_and_ten_evaluation_extra_attempts",
                source={name: sha256_bytes(Path(__file__).with_name(name).read_bytes().replace(b"\r\n", b"\n")) for name in names})
    plan["parents"]["cached_source_audits"] = {name: origin["audit_sha256"] for name, origin in saved["origins"].items()}
    return plan, initial_root, feature_path


def run(pool, qualification_path, cache_root, output, *, execute=False, resume=False):
    plan, initial_root, feature_path = make_plan(pool, qualification_path, cache_root)
    return run_comparison(pool, plan, initial_root, output, execute=execute, resume=resume, feature_cache=feature_path)

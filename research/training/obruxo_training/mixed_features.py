"""Reuse independently audited source features with explicit original/wide origins."""

from pathlib import Path
from time import perf_counter


from obruxo_data.hashing import canonical_sha256, sha256_bytes

from .audit import write_json
from .benchmark import MemoryMonitor
from .checkpoints import atomic_torch_save, load_training_checkpoint
from .extended_features import features_for
from .features import tensor_digest
from .mixed_input_performances import MixedSampler
from .profiled_sampler import training_sampler
from .stream import OnlineRenderStream, StreamConfig

FORMAT = "obruxo_mixed_features_v1"
LIMITS = {"wall_seconds": 300, "process_tree_bytes": 4 * 1024**3, "archive_bytes": 128 * 1024**2}


def digest(path):
    return sha256_bytes(Path(path).read_bytes())


def origin_descriptor(root, audit_path):
    root, audit_path = Path(root).resolve(), Path(audit_path).resolve()
    fit = sorted(root.glob("fit-*"))[-1]
    return {"root": str(root), "audit": str(audit_path), "audit_sha256": digest(audit_path),
            "checkpoint": (fit / "checkpoint.pt").relative_to(root).as_posix(), "checkpoint_sha256": digest(fit / "checkpoint.pt")}


def load_origins(origins, pool, sampler, frontend):
    from .trainer import TrainConfig, _run_contract, validate_replay
    from .varied_bootstrap import verified_parent
    if set(origins) != {"original", "wide"} or type(sampler) is not MixedSampler:
        raise ValueError("mixed features require both frozen source banks")
    parents, native = {}, None
    states = sum(assignment.split == "train" for assignment in pool.manifest.assignments)
    for name, descriptor in origins.items():
        root, audit_path = Path(descriptor["root"]), Path(descriptor["audit"])
        if digest(audit_path) != descriptor["audit_sha256"] or digest(root / descriptor["checkpoint"]) != descriptor["checkpoint_sha256"]:
            raise ValueError("mixed feature parent file changed")
        plan, summary, audit = verified_parent(root, audit_path)
        expected_experiment = "settled_basic_full_bootstrap_v1" if name == "original" else "varied_basic_initialization_v1"
        count = states if name == "original" else 2 * states
        if (plan["experiment"] != expected_experiment or plan["pool_id"] != pool.pool_id
                or not summary["all_outcomes_accepted"] or (name == "original" and not summary["preflight_passed"])
                or not audit["model_optimizer_batch_and_global_rng_exact"] or audit["native_renders_added"] != 0
                or audit["training_source_receipts_verified"] != count):
            raise ValueError("mixed features lack a complete independently audited source bank")
        source = load_training_checkpoint(root / descriptor["checkpoint"])
        config = TrainConfig(**source["run_contract"]["training"])
        source_sampler = training_sampler(pool, config)
        expected_sampler = sampler.original if name == "original" else sampler.wide
        stream = StreamConfig.from_dict(source["run_contract"]["stream"])
        expected = _run_contract(pool, source_sampler, frontend, config, stream, source["run_contract"].get("replay_seed"),
                                 initial_model=source["run_contract"].get("initial_model"))
        if (source["run_contract"] != expected or source_sampler.sampler_id != expected_sampler.sampler_id
                or tensor_digest(source["basic_pitch"]) != frontend.frozen_digest or source["renderer_identity"] != plan["renderer_identity"]
                or source["cursor"]["next_sequence_id"] != count or len(source["replay"]) != count
                or source["step"] != plan["config"]["steps"] or audit["replayed_updates"] != source["step"]):
            raise ValueError("mixed feature parent contracts, coverage or frontend changed")
        validate_replay(source["replay"], source_sampler, source["cursor"], config)
        if [item["sequence_id"] for item in source["replay"]] != list(range(count)):
            raise ValueError("mixed source must retain every accepted observation")
        if native is not None and native != source["renderer_identity"]:
            raise ValueError("mixed feature banks use different native renderers")
        native = source["renderer_identity"]
        parents[name] = source
    return parents, native


def feature_digest(replay):
    return canonical_sha256([{"sequence_id": item["sequence_id"], "sha256": tensor_digest(item["features"])} for item in replay])


def remap_replay(sampler, parents):
    replay, observations = [], set()
    for sequence in range(sampler.epoch_size):
        recipe = sampler.recipe_at(sequence)
        parent = parents[recipe.origin]["replay"][recipe.origin_sequence_id]
        replay.append({**parent, "sequence_id": sequence, "recipe_id": recipe.recipe_id, "performance_id": recipe.performance_id, "uses": 0,
                       "origin": {"bank": recipe.origin, "sequence_id": recipe.origin_sequence_id, "recipe_id": recipe.origin_recipe_id}})
        observations.add((recipe.origin, recipe.origin_sequence_id))
    return replay, len(observations)


def validate_rows(saved, sampler, frontend, config, stream_config, parents, native):
    from .trainer import replay_bytes, validate_replay
    if (saved.get("format") != FORMAT or saved["pool_id"] != sampler.pool.pool_id or saved["codec"] != sampler.pool.codec.to_dict()
            or saved["sampler_id"] != sampler.sampler_id or saved["observation_contract"] != sampler.observation_contract
            or saved["frontend"] != frontend.describe() or tensor_digest(saved["basic_pitch"]) != frontend.frozen_digest
            or saved["stream"] != stream_config.to_dict() or saved["renderer_identity"] != native
            or config.input_performances != "mixed" or len(saved["replay"]) != sampler.epoch_size
            or saved["cursor"]["next_sequence_id"] != sampler.epoch_size
            or len(saved["replay"]) > config.replay_capacity or replay_bytes(saved["replay"]) > config.replay_mib * 1024**2
            or saved["cursor"]["next_sequence_id"] > config.render_budget):
        raise ValueError("mixed feature cache has incompatible contracts or exceeds its budget")
    OnlineRenderStream(sampler, 1, config=stream_config, resume=saved["cursor"])
    validate_replay(saved["replay"], sampler, saved["cursor"], config)
    observations = set()
    for item in saved["replay"]:
        recipe = sampler.recipe_at(item["sequence_id"])
        parent = parents[recipe.origin]["replay"][recipe.origin_sequence_id]
        origin = {"bank": recipe.origin, "sequence_id": recipe.origin_sequence_id, "recipe_id": recipe.origin_recipe_id}
        if (item["origin"] != origin or parent["recipe_id"] != recipe.origin_recipe_id
                or parent["state_id"] != item["state_id"] or parent.get("performance_id", recipe.performance_id) != item["performance_id"]
                or parent["audio_sha256"] != item["audio_sha256"] or tensor_digest(parent["features"]) != tensor_digest(item["features"])):
            raise ValueError("mixed observation differs from its audited origin")
        observations.add((recipe.origin, recipe.origin_sequence_id))
    if saved["distinct_parent_observations"] != len(observations) or saved["features_sha256"] != feature_digest(saved["replay"]):
        raise ValueError("mixed feature content or distinct observation count changed")


def validate_cache(saved, pool, sampler, frontend, config, stream_config):
    parents, native = load_origins(saved["origins"], pool, sampler, frontend)
    validate_rows(saved, sampler, frontend, config, stream_config, parents, native)


def prepare_cache(pool, original_root, original_audit, wide_root, wide_audit, output):
    from .trainer import TrainConfig, replay_bytes
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("mixed feature preparation requires a new output directory")
    started, monitor = perf_counter(), MemoryMonitor()
    monitor.start()
    try:
        origins = {"original": origin_descriptor(original_root, original_audit), "wide": origin_descriptor(wide_root, wide_audit)}
        names = ("mixed_features.py", "mixed_input_performances.py", "trainer.py", "profiled_sampler.py", "features.py", "paired.py")
        plan = {"operation": FORMAT, "pool_id": pool.pool_id, "origins": origins, "limits": LIMITS,
                "schedule": ["original", "wide_held", "original", "wide_alternate"], "native_requests": 0, "optimizer_updates": 0,
                "source": {name: sha256_bytes(Path(__file__).with_name(name).read_bytes().replace(b"\r\n", b"\n")) for name in names}}
        output.mkdir(parents=True)
        write_json(output / "plan.json", plan)
        first = load_training_checkpoint(Path(origins["original"]["root"]) / origins["original"]["checkpoint"])
        frontend = features_for(pool.codec, first["basic_pitch"])
        sampler, stream = MixedSampler(pool), StreamConfig(prefetch_clips=1)
        parents, native = load_origins(origins, pool, sampler, frontend)
        replay, distinct_observations = remap_replay(sampler, parents)
        cursor = OnlineRenderStream(sampler, 1, config=stream).state_dict()
        cursor["next_sequence_id"] = sampler.epoch_size
        saved = {"format": FORMAT, "pool_id": pool.pool_id, "codec": pool.codec.to_dict(), "sampler_id": sampler.sampler_id,
                 "observation_contract": sampler.observation_contract, "frontend": frontend.describe(), "basic_pitch": first["basic_pitch"],
                 "stream": stream.to_dict(), "cursor": cursor, "renderer_identity": native, "origins": origins,
                 "replay": replay, "distinct_parent_observations": distinct_observations, "features_sha256": feature_digest(replay)}
        config = TrainConfig(replay_capacity=len(replay), replay_mib=64, batch_size=4, render_budget=len(replay), cpu_threads=2,
                             prediction_scope="basic", preroll_frames=32768, sampling_policy="paired_views", input_performances="mixed")
        validate_rows(saved, sampler, frontend, config, stream, parents, native)
        atomic_torch_save(output / "features.pt", saved)
        summary = {"format": FORMAT, "plan_id": canonical_sha256(plan), "complete": True, "pool_id": pool.pool_id, "source_banks": {name: {key: value for key, value in origin.items() if key not in {"root", "audit"}}
                                                                                   for name, origin in origins.items()},
                   "new_native_requests": 0, "optimizer_updates": 0, "weighted_records": len(replay), "distinct_parent_observations": distinct_observations,
                   "counted_replay_bytes": replay_bytes(replay), "cache_file_sha256": digest(output / "features.pt"),
                   "cache_file_bytes": (output / "features.pt").stat().st_size, "features_sha256": saved["features_sha256"],
                   "elapsed_seconds": perf_counter() - started, "peak_process_tree_rss_bytes": monitor.peak_bytes, "rss_sampling_errors": monitor.errors}
        if (summary["elapsed_seconds"] > LIMITS["wall_seconds"] or monitor.peak_bytes is None or monitor.peak_bytes > LIMITS["process_tree_bytes"]
                or monitor.errors or summary["cache_file_bytes"] > LIMITS["archive_bytes"]):
            raise RuntimeError("mixed feature preparation resource limit failed")
        write_json(output / "summary.json", summary)
        return summary
    finally:
        monitor.close()

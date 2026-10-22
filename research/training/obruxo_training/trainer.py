"""Bounded online/replay bootstrap training with checkpoints at optimizer boundaries."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import torch

from obruxo_data.hashing import canonical_sha256, sha256_bytes

from .audit import write_json
from .benchmark import MemoryMonitor
from .checkpoints import TrainingCheckpointWriter, atomic_torch_save, load_training_checkpoint
from .codec import PilotCodec, PilotTarget, load_codec
from .curve_labels import CURVE_LABEL_KEYS, SOURCE_CURVES
from .features import tensor_digest, validate_features
from .extended_features import features_for
from .model import PilotInverse, batch_features, conditioning_from_spec, cyclic_head_from_spec, loss_terms, model_spec, predicted_targets
from .paired import complete_pairs, identity_terms, paired_batch_indices
from .pilot import PilotPool, PilotSampler
from .profiled_sampler import profile_metadata, training_sampler
from .stream import OnlineRenderStream, StreamConfig


@dataclass(frozen=True)
class TrainConfig:
    mode: str = "overfit"
    steps: int = 400
    replay_capacity: int = 32
    replay_mib: int = 8
    batch_size: int = 16
    learning_rate: float = 0.003
    seed: int = 0
    learning_seed_override: int | None = None
    checkpoint_every: int = 50
    render_budget: int = 512
    cpu_threads: int = 4
    sampling_policy: str = "pairs"
    updates_per_render: int = 1
    fresh_examples: int | None = None
    identity_weight: float = 0.
    identity_temperature: float = .1
    conditioning: str = "concat"
    cyclic_head: str = "sigmoid"
    prediction_scope: str = "codec"
    preroll_frames: int = 0
    input_performances: str = "initial"

    def __post_init__(self):
        positive = (self.steps, self.replay_capacity, self.replay_mib, self.batch_size,
                    self.checkpoint_every, self.render_budget, self.cpu_threads, self.updates_per_render)
        if any(type(value) is not int or value <= 0 for value in positive) or type(self.seed) is not int:
            raise ValueError("training counts must be positive integers and seed an integer")
        if self.learning_seed_override is not None and type(self.learning_seed_override) is not int:
            raise ValueError("learning_seed_override must be an integer or None")
        maximum_records = 2048 if self.prediction_scope == "basic" else 512
        if self.mode not in {"overfit", "online"} or self.replay_capacity > maximum_records or self.replay_mib > 256:
            raise ValueError(f"pilot mode must be overfit/online with replay at most {maximum_records} examples and 256 MiB")
        if self.batch_size > self.replay_capacity or self.render_budget < self.replay_capacity:
            raise ValueError("batch must fit replay and render budget must cover initial replay")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning rate must be positive and finite")
        if self.sampling_policy not in {"pairs", "state_round_robin", "paired_views", "performance_round_robin", "attack_compatible", "source_balanced"}:
            raise ValueError("unknown pilot sampling policy")
        if self.conditioning not in {"concat", "cross_attention"}:
            raise ValueError("unsupported performance conditioning")
        if self.cyclic_head not in {"sigmoid", "periodic"}:
            raise ValueError("unsupported cyclic output head")
        if self.prediction_scope not in {"codec", "basic"}:
            raise ValueError("unsupported prediction scope")
        profile_metadata(self.preroll_frames)
        if self.preroll_frames and self.prediction_scope != "basic":
            raise ValueError("qualified bootstrap render profile requires explicit basic prediction scope")
        if self.input_performances not in {"initial", "varied", "mixed"}:
            raise ValueError("unsupported input performance policy")
        if self.input_performances == "varied" and (self.prediction_scope != "basic" or self.preroll_frames != 32768 or self.sampling_policy != "paired_views"):
            raise ValueError("varied inputs require basic scope, qualified pre-roll and paired_views sampling")
        if self.input_performances == "mixed" and (self.prediction_scope != "basic" or self.preroll_frames != 32768
                                                   or self.sampling_policy != "paired_views" or self.mode != "overfit"):
            raise ValueError("mixed inputs require basic scope, qualified pre-roll, paired_views and fixed cached replay")
        if (not math.isfinite(self.identity_weight) or self.identity_weight < 0
                or not math.isfinite(self.identity_temperature) or self.identity_temperature <= 0):
            raise ValueError("identity weight must be nonnegative and temperature positive and finite")
        if self.identity_weight and self.sampling_policy != "paired_views":
            raise ValueError("identity loss requires paired_views sampling")
        if self.sampling_policy == "paired_views" and (self.batch_size < 4 or self.batch_size % 2 or self.replay_capacity % 2):
            raise ValueError("paired views need an even batch of at least four and even replay capacity")
        if self.fresh_examples is not None and (type(self.fresh_examples) is not int
                                                or not self.batch_size <= self.fresh_examples <= self.render_budget):
            raise ValueError("fresh_examples must cover the warmup and fit the render-attempt budget")
        if self.mode != "online" and (self.updates_per_render != 1 or self.fresh_examples is not None):
            raise ValueError("bounded refresh settings apply to online mode only")
        if self.fresh_examples is not None and self.steps < 1 + (self.fresh_examples - self.batch_size) * self.updates_per_render:
            raise ValueError("optimizer step budget cannot reach the requested fresh-example target")


def capture_rng(device: str) -> dict:
    numpy_state = np.random.get_state()
    result = {"python": random.getstate(), "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
              "torch": torch.get_rng_state(), "device": device, "accelerator": None}
    kind = torch.device(device).type
    if kind in {"cuda", "xpu"}:
        result["accelerator"] = getattr(torch, kind).get_rng_state(device).cpu()
    return result


def restore_rng(value: dict, device: str) -> None:
    random.setstate(value["python"])
    name, keys, position, gaussian, cached = value["numpy"]
    np.random.set_state((name, np.array(keys, dtype=np.uint32), position, gaussian, cached))
    torch.set_rng_state(value["torch"])
    kind = torch.device(device).type
    if device == value["device"] and value["accelerator"] is not None:
        getattr(torch, kind).set_rng_state(value["accelerator"], device)


def replay_bytes(items: list[dict]) -> int:
    features = sum(tensor.numel() * tensor.element_size() for item in items for tensor in item["features"].values())
    curves = [item[name] for item in items for name in CURVE_LABEL_KEYS if name in item]
    if any(not isinstance(value, torch.Tensor) for value in curves):
        raise ValueError("curve replay labels must be tensors")
    return features + sum(value.numel() * value.element_size() for value in curves)


def validate_replay(items: list[dict], sampler: PilotSampler, cursor: dict, config: TrainConfig) -> None:
    if len(items) > config.replay_capacity or replay_bytes(items) > config.replay_mib * 1024 * 1024:
        raise ValueError("saved replay exceeds its budget")
    sequences = [item["sequence_id"] for item in items]
    if sequences != sorted(set(sequences)) or any(not 0 <= value < cursor["next_sequence_id"] for value in sequences):
        raise ValueError("saved replay and acknowledged cursor disagree")
    targets = {entry.state_id: entry.target.to_dict() for entry in sampler.pool.entries}
    for item in items:
        expected_profile = profile_metadata(config.preroll_frames)
        if any(item.get(key) != expected_profile.get(key) for key in ("render_profile", "render_phase_controls")):
            raise ValueError("saved replay render profile or phase policy changed")
        recipe = sampler.recipe_at(item["sequence_id"])
        if recipe.split != "train" or recipe.performance_role != "reference":
            raise ValueError("replay contains held-out data")
        if item["recipe_id"] != recipe.recipe_id or item["state_id"] != recipe.state_id or item["target"] != targets[recipe.state_id]:
            raise ValueError("saved replay does not match its training recipe")
        if config.sampling_policy in {"paired_views", "performance_round_robin", "attack_compatible", "source_balanced"} and item.get("performance_id") != recipe.performance_id:
            label = "paired" if config.sampling_policy == "paired_views" else "performance-balanced"
            raise ValueError(f"{label} replay performance differs from its training recipe")
        validate_features(item["features"])
        for flag, key, method, label, unexpected in SOURCE_CURVES:
            if not getattr(sampler.pool.codec, flag, False):
                if key in item:
                    raise ValueError(f"unexpected {unexpected} supervision for this codec")
                continue
            expected = torch.tensor(getattr(sampler.pool.codec, method)(recipe.render_request().preset_json), dtype=torch.float32)
            actual = item.get(key)
            if not isinstance(actual, torch.Tensor) or actual.dtype != torch.float32 or actual.requires_grad or not torch.equal(actual, expected):
                raise ValueError(f"saved {label} differ from their source recipe")


def curve_loss_inputs(items, codec):
    return {key: torch.stack([item[key] for item in items]) for flag, key, *_ in SOURCE_CURVES if getattr(codec, flag, False)}


@torch.no_grad()
def replay_metrics(model, items, codec, device, config=None) -> dict:
    model.eval()
    prediction = model(batch_features(items, device))
    targets = [PilotTarget.from_dict(item["target"]) for item in items]
    terms = loss_terms(prediction, targets, codec, **curve_loss_inputs(items, codec))
    if config is not None and config.sampling_policy == "paired_views":
        indices = [index for pairs in complete_pairs(items).values() for index in pairs[0]]
        if len(indices) >= 4:
            identity = identity_terms(prediction["representation"][indices], config.identity_temperature)
            terms["loss"] = terms["loss"] + config.identity_weight * identity["identity_loss"]
            terms.update(identity)
    return {name: float(value) for name, value in terms.items()}


def _require_prediction_scope(codec, scope):
    if scope == "basic" and (type(codec) is not PilotCodec or codec.to_dict() != PilotCodec().to_dict()):
        raise ValueError("basic prediction scope requires the original nine-control/two-choice pilot pool; broad presets cannot be silently projected")


def _run_contract(pool, sampler, frontend, config, stream_config, replay_seed=None, *, initial_model=None) -> dict:
    _require_prediction_scope(pool.codec, config.prediction_scope)
    training = asdict(config)
    # Preserve the existing v1 contract for the unchanged legacy schedule.
    for name, default in (("sampling_policy", "pairs"), ("updates_per_render", 1), ("fresh_examples", None), ("preroll_frames", 0), ("input_performances", "initial")):
        if training[name] == default:
            del training[name]
    if training["learning_seed_override"] is None:
        del training["learning_seed_override"]
    if training["identity_weight"] == 0:
        del training["identity_weight"]
    if training["identity_temperature"] == .1:
        del training["identity_temperature"]
    if training["conditioning"] == "concat":
        del training["conditioning"]
    if training["cyclic_head"] == "sigmoid":
        del training["cyclic_head"]
    if training["prediction_scope"] == "codec":
        del training["prediction_scope"]
    contract = {"trainer": "pilot_supervised_bootstrap_v1", "pool_id": pool.pool_id,
            "contract": pool.codec.contract.to_dict(), "codec": pool.codec.to_dict(), "sampler_id": sampler.sampler_id,
            "model": model_spec(pool.codec, config.conditioning, config.cyclic_head), "frontend": frontend.describe(), "training": training,
            "stream": stream_config.to_dict(), "optimizer": "AdamW_weight_decay_0", "precision": "float32",
            "scheduler": None, "scaler": None,
            "replay_policy": "bounded_refresh_then_replay_v1" if config.fresh_examples is not None or config.updates_per_render != 1
                             else "fixed_overfit_or_fifo_online_v1"}
    if replay_seed is not None:
        contract["replay_seed"] = replay_seed
    if config.identity_weight:
        contract["identity_objective"] = {"loss": "bidirectional_paired_info_nce_v1", "representation": "decoder_fusion_128",
                                          "normalization": "l2", "negatives": "distinct_training_states_in_batch"}
    contract.update(profile_metadata(config.preroll_frames))
    if hasattr(sampler, "observation_contract"):
        contract["observation_contract"] = sampler.observation_contract
    if initial_model is not None:
        contract["initial_model"] = initial_model
    return contract


def _rejected_prefix_history(path, source, sampler):
    """Require every accepted feature and its original acknowledged attempt ledger."""
    message = "feature seed must retain a complete accepted recipe prefix or a verified complete attempt history"
    history_path = path.with_name("renders.jsonl")
    if source["run_contract"].get("replay_seed") or not history_path.is_file():
        raise ValueError(message)
    raw = history_path.read_bytes()
    history = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    if len(history) != source["cursor"]["next_sequence_id"]:
        raise ValueError(message)
    accepted, counts = [], {"accepted": 0, "rejected": 0, "error": 0}
    for sequence, row in enumerate(history):
        recipe = sampler.recipe_at(sequence)
        status = row.get("status")
        if (type(row.get("sequence_id")) is not int or row["sequence_id"] != sequence
                or row.get("recipe_id") != recipe.recipe_id or status not in counts):
            raise ValueError("feature seed attempt history differs from its scheduled recipes")
        if status != "error":
            identity = {key: value for key, value in row.get("provenance", {}).items() if key != "request_id"}
            if identity != source["renderer_identity"]:
                raise ValueError("feature seed attempt history has different renderer provenance")
        counts[status] += 1
        if status == "accepted":
            accepted.append(row)
    totals = source["totals"]
    if (totals["render_attempts"] != len(history) or totals["accepted"] != counts["accepted"]
            or totals["rejected"] != counts["rejected"] or totals["errors"] != counts["error"]):
        raise ValueError("feature seed attempt history and checkpoint counters disagree")
    if [item["sequence_id"] for item in source["replay"]] != [row["sequence_id"] for row in accepted]:
        raise ValueError(message)
    for item, row in zip(source["replay"], accepted, strict=True):
        if item["audio_sha256"] != row.get("qa", {}).get("audio_float32_sha256"):
            raise ValueError("feature seed audio digest differs from its accepted attempt")
    return {"render_log_sha256": sha256_bytes(raw), "accepted": counts["accepted"],
            "rejected": counts["rejected"], "errors": counts["error"]}


def load_replay_seed(path: Path, pool, sampler, frontend, config, stream_config) -> tuple:
    raw = path.read_bytes()
    source = load_training_checkpoint(path)
    if source.get("format") == "obruxo_mixed_features_v1":
        from .mixed_features import validate_cache
        validate_cache(source, pool, sampler, frontend, config, stream_config)
        return source, {"policy": "verified_original_and_wide_features_fresh_learning_state_v1", "checkpoint_sha256": sha256_bytes(raw),
                        "source_run_id": canonical_sha256(source["origins"]), "examples": len(source["replay"]),
                        "distinct_parent_observations": source["distinct_parent_observations"],
                        "acknowledged_recipes": source["cursor"]["next_sequence_id"]}
    if source.get("format") == "obruxo_paired_features_v1":
        if config.preroll_frames:
            raise ValueError("legacy paired feature cache lacks the corrected render profile")
        from .paired_features import validate_cache
        if config.sampling_policy != "paired_views":
            raise ValueError("paired feature cache requires the paired_views sampler")
        accepted = validate_cache(source, pool, sampler, frontend, stream_config)
        replay, cursor = source["replay"], source["cursor"]
        if (len(replay) > config.replay_capacity or replay_bytes(replay) > config.replay_mib * 1024 * 1024
                or cursor["next_sequence_id"] > config.render_budget or not replay
                or (config.fresh_examples is not None and len(replay) > config.fresh_examples)):
            raise ValueError("paired feature cache exceeds the training budget")
        descriptor = {"policy": "qa_qualified_complete_pairs_fresh_learning_state_v1", "checkpoint_sha256": sha256_bytes(raw),
                      "source_run_id": canonical_sha256({"features": source["features_sha256"], "history": source["history"]}),
                      "examples": len(replay), "acknowledged_recipes": cursor["next_sequence_id"],
                      "raw_accepted_views": accepted, "discarded_unpaired_accepted_views": accepted - len(replay)}
        return source, descriptor
    if source.get("format") != "obruxo_pilot_training_v1" or type(source.get("step")) is not int or source["step"] < 0:
        raise ValueError("feature seeding needs a collected pilot checkpoint")
    contract = source["run_contract"]
    source_config = TrainConfig(**contract["training"])
    if source_config.preroll_frames != config.preroll_frames:
        raise ValueError("feature seed belongs to a different render profile")
    source_stream = StreamConfig.from_dict(contract["stream"])
    expected = _run_contract(pool, sampler, frontend, source_config, source_stream, contract.get("replay_seed"), initial_model=contract.get("initial_model"))
    if contract != expected or tensor_digest(source["basic_pitch"]) != frontend.frozen_digest:
        raise ValueError("feature seed has incompatible pool, sampler or frozen frontend contracts")
    replay, cursor = source["replay"], source["cursor"]
    validate_replay(replay, sampler, cursor, source_config)
    validate_replay(replay, sampler, cursor, config)
    OnlineRenderStream(sampler, 1, config=stream_config, resume=cursor)
    if not replay:
        raise ValueError("feature seed must retain a complete accepted recipe prefix")
    history = None
    if [item["sequence_id"] for item in replay] != list(range(cursor["next_sequence_id"])):
        history = _rejected_prefix_history(path, source, sampler)
    if config.fresh_examples is not None and len(replay) > config.fresh_examples:
        raise ValueError("feature seed exceeds the accepted-example target")
    if cursor["next_sequence_id"] > config.render_budget or not source["renderer_identity"]:
        raise ValueError("feature seed exceeds the recipe budget or has no renderer provenance")
    descriptor = {"policy": "features_only_fresh_model_optimizer_rng_v1", "checkpoint_sha256": sha256_bytes(raw),
                  "source_run_id": canonical_sha256(contract), "examples": len(replay),
                  "acknowledged_recipes": cursor["next_sequence_id"]}
    if history is not None:
        descriptor.update(policy="qa_qualified_complete_attempt_prefix_fresh_learning_state_v1", history=history)
    return source, descriptor


def initial_model_state(path, pool, frontend, config):
    raw = torch.load(path, map_location="cpu", weights_only=True)
    expected_profile = profile_metadata(config.preroll_frames)
    if (raw.get("format") != "obruxo_pilot_inference_v1" or raw.get("training_pool_id") != pool.pool_id
            or raw.get("codec") != pool.codec.to_dict() or raw.get("model_spec") != model_spec(pool.codec, config.conditioning, config.cyclic_head)
            or raw.get("frontend") != frontend.describe()
            or any(raw.get(key) != expected_profile.get(key) for key in ("render_profile", "render_phase_controls"))):
        raise ValueError("initial model has incompatible codec, frontend, pool or render profile")
    with torch.random.fork_rng(devices=[]):
        restore_model(raw)  # Validate actual tensors and frozen frontend without consuming this run's RNG.
    descriptor = {"initialization": "verified_inference_weights_fresh_optimizer_and_learning_rng_v1", "file_sha256": sha256_bytes(Path(path).read_bytes()),
                  "model_state_sha256": tensor_digest(raw["model"]), "frontend_state_sha256": tensor_digest(raw["basic_pitch"]),
                  "training_pool_id": pool.pool_id, **expected_profile}
    return raw["model"], descriptor


def train_pilot(pool: PilotPool, output: Path, *, config: TrainConfig | None = None, device: str = "cpu",
                stream_config: StreamConfig | None = None, renderer_factory=None, resume: Path | None = None,
                stop_after: int | None = None, progress=None, after_step=None, seed_replay: Path | None = None,
                initial_model: Path | None = None) -> dict:
    """Train with fresh renders inside the loop. after_step supports interruption testing."""
    config, stream_config = config or TrainConfig(), stream_config or StreamConfig(prefetch_clips=1)
    _require_prediction_scope(pool.codec, config.prediction_scope)
    if config.preroll_frames and stream_config.worker_jobs != 1:
        raise ValueError("qualified basic bootstrap uses one fresh native worker per request")
    if resume is not None and seed_replay is not None:
        raise ValueError("resume embeds its replay seed; do not supply seed_replay again")
    if resume is not None and initial_model is not None:
        raise ValueError("resume embeds its initial model; do not supply initial_model again")
    if config.input_performances == "mixed" and resume is None and seed_replay is None:
        raise ValueError("mixed inputs require their verified feature cache")
    if torch.device(device).type not in {"cpu", "cuda", "xpu"}:
        raise ValueError("pilot supports CPU, CUDA or XPU explicitly; no automatic fallback")
    learning_seed = config.seed if config.learning_seed_override is None else config.learning_seed_override
    torch.set_num_threads(config.cpu_threads)
    torch.use_deterministic_algorithms(True)
    random.seed(learning_seed)
    np.random.seed(learning_seed % 2**32)
    torch.manual_seed(learning_seed)
    saved = load_training_checkpoint(resume) if resume else None
    frontend = features_for(pool.codec, saved["basic_pitch"] if saved else None)
    sampler = training_sampler(pool, config)
    paired = config.sampling_policy == "paired_views"
    if paired and len({entry.state_id for entry in pool.entries if pool.manifest.by_state[entry.state_id].split == "train"}) < config.batch_size // 2:
        raise ValueError("paired training needs enough distinct training states for its batch")
    source, replay_seed = None, saved["run_contract"].get("replay_seed") if saved else None
    if seed_replay is not None:
        source, replay_seed = load_replay_seed(seed_replay, pool, sampler, frontend, config, stream_config)
    starting_weights, initialization = None, saved["run_contract"].get("initial_model") if saved else None
    if initial_model is not None:
        starting_weights, initialization = initial_model_state(initial_model, pool, frontend, config)
    contract = _run_contract(pool, sampler, frontend, config, stream_config, replay_seed, initial_model=initialization)
    if saved and (saved.get("format") != "obruxo_pilot_training_v1" or saved["run_contract"] != contract):
        raise ValueError("checkpoint training contract changed; start an explicit new run")
    model = PilotInverse(pool.codec, config.conditioning, config.cyclic_head).to(device)
    if starting_weights is not None:
        model.load_state_dict(starting_weights, strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0)
    generator = torch.Generator().manual_seed(learning_seed)
    replay, step, initial, render_ready, renderer_identity = [], 0, None, False, None
    totals = {"render_attempts": 0, "accepted": 0, "rejected": 0, "errors": 0, "feature_seconds": 0.0,
              "optimizer_seconds": 0.0, "sampled_examples": 0, "reused_feature_draws": 0, "maximum_gradient_norm": 0.0}
    cursor = OnlineRenderStream(sampler, 1, config=stream_config).state_dict()
    seeded_recipes = replay_seed["acknowledged_recipes"] if replay_seed else 0
    if source is not None:
        replay = [{**item, "uses": 0} for item in source["replay"]]
        cursor, renderer_identity = source["cursor"], source["renderer_identity"]
        totals["accepted"] = len(replay)
    if saved:
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        replay, step, initial, totals, cursor = (saved[key] for key in ("replay", "step", "initial_metrics", "totals", "cursor"))
        render_ready, renderer_identity = saved["render_ready"], saved["renderer_identity"]
        if not 0 <= step <= config.steps or totals["render_attempts"] + seeded_recipes != cursor["next_sequence_id"]:
            raise ValueError("checkpoint step/cursor/counters disagree")
        OnlineRenderStream(sampler, 1, config=stream_config, resume=cursor)
        validate_replay(replay, sampler, cursor, config)
        generator.set_state(saved["batch_rng"])
        restore_rng(saved["rng"], device)
    limit = config.steps if stop_after is None else stop_after
    collect_only = saved is None and step == limit == 0
    if type(limit) is not int or not (step < limit <= config.steps or collect_only):
        raise ValueError("stop_after must follow the saved step and not exceed total steps")
    output.mkdir(parents=True, exist_ok=False)
    checkpoint_writer = TrainingCheckpointWriter(output)
    started, checkpoint_seconds, feature_seconds_start = perf_counter(), 0.0, totals["feature_seconds"]
    accepted_start = totals["accepted"]
    monitor = MemoryMonitor()
    monitor.start()
    write_json(output / "status.json", {"complete": False, "stage": "training"})

    def checkpoint():
        nonlocal checkpoint_seconds
        tick = perf_counter()
        frontend.require_frozen()
        value = {"format": "obruxo_pilot_training_v1", "run_contract": contract, "step": step,
                 "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": None, "scaler": None,
                 "rng": capture_rng(device), "batch_rng": generator.get_state(), "cursor": cursor,
                 "replay": replay, "basic_pitch": frontend.basic_pitch.state_dict(), "initial_metrics": initial,
                 "render_ready": render_ready, "renderer_identity": renderer_identity,
                 "totals": totals, "runtime": {"torch": str(torch.__version__), "device": device},
                 "recovery": "exact_retained_features_same_runtime_new_native_audio_recipe_equivalence_only"}
        checkpoint_writer.save(value)
        checkpoint_seconds += perf_counter() - tick

    try:
        checkpoint()
        with ExitStack() as stack:
            stream = None

            def ingest():
                nonlocal cursor, stream, render_ready, renderer_identity
                if stream is None:
                    remaining = config.render_budget - cursor["next_sequence_id"]
                    if remaining <= 0:
                        raise RuntimeError("fresh-render budget exhausted")
                    stream = stack.enter_context(OnlineRenderStream(sampler, remaining, config=stream_config,
                                                 renderer_factory=renderer_factory, resume=cursor))
                for outcome in stream:
                    totals["render_attempts"] += 1
                    status = outcome.metadata["status"]
                    totals[{"accepted": "accepted", "rejected": "rejected", "error": "errors"}[status]] += 1
                    if outcome.audio is not None:
                        identity = {key: value for key, value in outcome.metadata["provenance"].items() if key != "request_id"}
                        if config.preroll_frames and identity.get("settings", {}).get("preroll_frames") != config.preroll_frames:
                            raise ValueError("accepted bootstrap audio lacks its declared render profile")
                        if renderer_identity is not None and identity != renderer_identity:
                            raise ValueError("renderer provenance changed during training or recovery")
                        renderer_identity = identity
                        tick = perf_counter()
                        features = frontend(outcome.audio)
                        totals["feature_seconds"] += perf_counter() - tick
                        recipe = outcome.recipe
                        target = next(entry.target for entry in pool.entries if entry.state_id == recipe.state_id)
                        item = {"features": features, "target": target.to_dict(), "sequence_id": recipe.sequence_id,
                                "state_id": recipe.state_id, "recipe_id": recipe.recipe_id,
                                "audio_sha256": outcome.metadata["qa"]["audio_float32_sha256"], "uses": 0}
                        item.update(profile_metadata(config.preroll_frames))
                        if paired or config.sampling_policy in {"performance_round_robin", "attack_compatible", "source_balanced"}:
                            item["performance_id"] = recipe.performance_id
                        for flag, key, method, *_ in SOURCE_CURVES:
                            if getattr(pool.codec, flag, False):
                                item[key] = torch.tensor(getattr(pool.codec, method)(recipe.render_request().preset_json), dtype=torch.float32)
                        candidate = [*replay, item][-config.replay_capacity:]
                        if replay_bytes(candidate) > config.replay_mib * 1024 * 1024:
                            raise ValueError("feature replay exceeds its byte budget")
                        replay[:] = candidate
                        render_ready = True
                    # Persist diagnostics before the cursor; a failed feature extraction is never acknowledged.
                    with (output / "renders.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({"sequence_id": outcome.recipe.sequence_id,
                                                 "recipe_id": outcome.recipe.recipe_id, **outcome.metadata}) + "\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                    stream.acknowledge(outcome.recipe.sequence_id)
                    cursor = stream.state_dict()
                    checkpoint()
                    if progress:
                        progress({"stage": "render", "sequence_id": outcome.recipe.sequence_id, "status": status,
                                  "replay_examples": len(replay), "step": step})
                    if outcome.audio is not None:
                        return
                raise RuntimeError("fresh-render budget exhausted before the next usable example")

            warmup = config.replay_capacity if config.mode == "overfit" else config.batch_size
            while len(replay) < warmup or (paired and len(complete_pairs(replay)) < config.batch_size // 2):
                ingest()
            if config.mode == "overfit" and stream is not None:
                stream.close()
            if initial is None:
                initial = replay_metrics(model, replay, pool.codec, device, config)
                checkpoint()
            while step < limit:
                refresh_due = config.mode == "online" and step > 0 and step % config.updates_per_render == 0
                refresh_allowed = config.fresh_examples is None or totals["accepted"] < config.fresh_examples
                if refresh_due and refresh_allowed and not render_ready:
                    ingest()
                while paired and len(complete_pairs(replay)) < config.batch_size // 2:
                    if config.fresh_examples is not None and totals["accepted"] >= config.fresh_examples:
                        raise RuntimeError("accepted-example budget cannot supply a complete paired batch")
                    ingest()
                if stream is not None and config.fresh_examples is not None and totals["accepted"] >= config.fresh_examples:
                    stream.close()
                    stream = None
                tick = perf_counter()
                indices = (paired_batch_indices(replay, config.batch_size, generator) if paired else
                           torch.randperm(len(replay), generator=generator)[:config.batch_size].tolist())
                items = [replay[index] for index in indices]
                model.train()
                optimizer.zero_grad(set_to_none=True)
                prediction = model(batch_features(items, device))
                terms = loss_terms(prediction, [PilotTarget.from_dict(item["target"]) for item in items], pool.codec, **curve_loss_inputs(items, pool.codec))
                if paired:
                    identity = identity_terms(prediction["representation"], config.identity_temperature)
                    terms["loss"] = terms["loss"] + config.identity_weight * identity["identity_loss"]
                    terms.update(identity)
                if not torch.isfinite(terms["loss"]):
                    raise RuntimeError("nonfinite training loss")
                terms["loss"].backward()
                if any(parameter.grad is None for parameter in model.parameters()):
                    raise RuntimeError("a trainable parameter is disconnected from the loss")
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
                optimizer.step()
                # Force the asynchronous accelerator boundary before committing a consumed optimizer step.
                values = {name: float(value.detach()) for name, value in terms.items()}
                if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
                    raise RuntimeError("nonfinite model weights after update")
                totals["maximum_gradient_norm"] = max(totals["maximum_gradient_norm"], float(norm))
                totals["optimizer_seconds"] += perf_counter() - tick
                totals["sampled_examples"] += len(items)
                for item in items:
                    totals["reused_feature_draws"] += item["uses"] > 0
                    item["uses"] += 1
                step += 1
                render_ready = False
                if step % config.checkpoint_every == 0 or step == limit:
                    checkpoint()
                    if progress:
                        progress({"stage": "optimizer", "step": step, **values})
                if after_step:
                    after_step(step)
        final = replay_metrics(model, replay, pool.codec, device, config)
        frontend.require_frozen()
        complete = step == config.steps
        if complete:
            export_model(output / "model.pt", model, frontend, pool.codec, pool.pool_id, preroll_frames=config.preroll_frames)
        summary = {"run_id": canonical_sha256(contract), "mode": config.mode, "complete": complete, "step": step,
                   "device": device, "torch": str(torch.__version__), "config": asdict(config), "totals": totals,
                   "replay_examples": len(replay), "replay_bytes": replay_bytes(replay), "initial_metrics": initial,
                   "replay_distinct_states": len({item["state_id"] for item in replay}),
                   "replay_distinct_audio_buffers": len({item["audio_sha256"] for item in replay}),
                   "final_metrics": final, "frozen_weights_and_buffers_unchanged": True,
                   "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
                   "model_state_sha256": tensor_digest(model.state_dict()), "frontend": frontend.describe(),
                   "elapsed_seconds_this_segment": perf_counter() - started, "checkpoint_seconds_this_segment": checkpoint_seconds,
                   "checkpoint_storage": checkpoint_writer.statistics,
                   "feature_seconds_this_segment": totals["feature_seconds"] - feature_seconds_start,
                   "fresh_accepted_this_segment": totals["accepted"] - accepted_start,
                   "seeded_feature_examples": replay_seed["examples"] if replay_seed else 0,
                   "replay_seed": replay_seed,
                   "fresh_example_target_reached": config.fresh_examples is None or totals["accepted"] >= config.fresh_examples,
                   "peak_process_tree_rss_bytes": monitor.peak_bytes, "rss_sampling_errors": monitor.errors,
                   "audio_files_written": 0, "cursor": cursor, "pool_id": pool.pool_id,
                   "renderer_identity": renderer_identity,
                   "tiny_overfit_gate": config.mode == "overfit" and final["continuous_mae"] < 0.05
                                        and final["categorical_accuracy"] == 1.0 and final["loss"] < initial["loss"] * 0.05,
                   "quality_claim": "supervised_bootstrap_only_real_render_validation_required"}
        if getattr(pool.codec, "lossy_curves", False):
            summary["curve_target_bytes"] = sum(item["curve_targets"].numel() * item["curve_targets"].element_size() for item in replay)
            summary["route_curve_target_bytes"] = sum(item["route_curve_targets"].numel() * item["route_curve_targets"].element_size() for item in replay if "route_curve_targets" in item)
            if getattr(pool.codec, "lossy_warps", False):
                summary["warp_curve_target_bytes"] = sum(item["warp_curve_targets"].numel() * item["warp_curve_targets"].element_size() for item in replay)
            summary["feature_bytes"] = summary["replay_bytes"] - summary["curve_target_bytes"] - summary["route_curve_target_bytes"] - summary.get("warp_curve_target_bytes", 0)
        if initialization is not None:
            summary["initial_model"] = initialization
        write_json(output / "summary.json", summary)
        write_json(output / "status.json", {"complete": complete, "stage": "complete" if complete else "preempted", "step": step})
        return summary
    except BaseException as error:
        write_json(output / "status.json", {"complete": False, "stage": "interrupted", "error_type": type(error).__name__})
        raise
    finally:
        monitor.close()


def export_model(path: Path, model, frontend, codec, pool_id: str, *, preroll_frames=0) -> None:
    atomic_torch_save(path, inference_state(model, frontend, codec, pool_id, preroll_frames=preroll_frames))


def inference_state(model, frontend, codec, pool_id: str, *, preroll_frames=0) -> dict:
    frontend.require_frozen()
    return {"format": "obruxo_pilot_inference_v1", "model_spec": model_spec(codec, model.conditioning, model.cyclic_head), "model": model.state_dict(),
            "basic_pitch": frontend.basic_pitch.state_dict(), "frontend": frontend.describe(), "contract": codec.contract.to_dict(),
            "codec": codec.to_dict(), "training_pool_id": pool_id, "input": "audio_only", **profile_metadata(preroll_frames),
            **({"qualified_for_release": False} if preroll_frames else {})}


def load_model(path: Path, device: str = "cpu") -> tuple:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    return restore_model(saved, device)


def restore_model(saved, device: str = "cpu") -> tuple:
    codec = load_codec(saved["codec"])
    conditioning = conditioning_from_spec(saved["model_spec"])
    cyclic_head = cyclic_head_from_spec(saved["model_spec"])
    if saved.get("format") != "obruxo_pilot_inference_v1" or saved["model_spec"] != model_spec(codec, conditioning, cyclic_head):
        raise ValueError("unsupported inference artifact")
    codec.contract.require_match(saved["contract"])
    if saved["codec"] != codec.to_dict():
        raise ValueError("inference codec changed")
    frontend = features_for(codec, saved["basic_pitch"])
    if frontend.describe() != saved["frontend"]:
        raise ValueError("inference frontend or frozen weights changed")
    model = PilotInverse(codec, conditioning, cyclic_head).to(device).eval()
    model.load_state_dict(saved["model"], strict=True)
    return model, frontend, codec


@torch.no_grad()
def predict_audio(model, frontend, codec, audio: np.ndarray, sample_rate: int = 44100) -> PilotTarget:
    features = frontend(audio, sample_rate)
    device = next(model.parameters()).device
    model.eval()
    return predicted_targets(model(batch_features([{"features": features}], device)), codec)[0]

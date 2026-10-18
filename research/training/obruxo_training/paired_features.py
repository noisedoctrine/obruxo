"""An audited, bounded paired feature cache, retaining complete QA-qualified pairs."""

import json
from pathlib import Path
from time import perf_counter

from obruxo_data.hashing import canonical_sha256, sha256_bytes

from .audit import write_json
from .features import tensor_digest, validate_features
from .extended_features import features_for
from .paired import complete_pairs
from .pilot import PilotSampler
from .stream import NativeRendererFactory, OnlineRenderStream, StreamConfig


FORMAT = "obruxo_paired_features_v1"


def validate_cache(saved, pool, sampler, frontend, stream_config):
    if (saved.get("format") != FORMAT or saved["pool_id"] != pool.pool_id or saved["sampler_id"] != sampler.sampler_id
            or saved["codec"] != pool.codec.to_dict() or saved["frontend"] != frontend.describe()
            or tensor_digest(saved["basic_pitch"]) != frontend.frozen_digest or saved["stream"] != stream_config.to_dict()):
        raise ValueError("paired feature cache has incompatible contracts")
    cursor, history = saved["cursor"], saved["history"]
    OnlineRenderStream(sampler, 1, config=stream_config, resume=cursor)
    if len(history) != cursor["next_sequence_id"] or len(history) % 2:
        raise ValueError("paired feature history must cover a complete scheduled attempt prefix")
    accepted, recipes = [], {}
    targets = {entry.state_id: entry.target.to_dict() for entry in pool.entries}
    for sequence, row in enumerate(history):
        recipe = sampler.recipe_at(sequence)
        recipes[sequence] = (recipe.state_id, recipe.performance_id, recipe.recipe_id)
        if row["sequence_id"] != sequence or row["recipe_id"] != recipe.recipe_id or row["status"] not in {"accepted", "rejected", "error"}:
            raise ValueError("paired feature history differs from its scheduled recipes")
        if row["status"] != "error":
            identity = {key: value for key, value in row["provenance"].items() if key != "request_id"}
            if identity != saved["renderer_identity"]:
                raise ValueError("paired feature renderer provenance changed")
        if row["status"] == "accepted":
            accepted.append({"sequence_id": sequence, "state_id": recipe.state_id, "performance_id": recipe.performance_id})
    qualified = {accepted[index]["sequence_id"] for pairs in complete_pairs(accepted).values() for pair in pairs for index in pair}
    if [item["sequence_id"] for item in saved["replay"]] != sorted(qualified):
        raise ValueError("paired cache must retain exactly the complete qualified pairs")
    for item in saved["replay"]:
        sequence = item["sequence_id"]
        state, performance, recipe_id = recipes[sequence]
        if (item["recipe_id"] != recipe_id or item["state_id"] != state or item["performance_id"] != performance
                or item["target"] != targets[state] or item["audio_sha256"] != history[sequence]["qa"]["audio_float32_sha256"]):
            raise ValueError("paired cached feature differs from its accepted recipe")
        validate_features(item["features"])
    if saved["features_sha256"] != canonical_sha256([{"sequence_id": item["sequence_id"], "sha256": tensor_digest(item["features"])}
                                                    for item in saved["replay"]]):
        raise ValueError("paired cached feature content changed")
    return len(accepted)


def prepare_paired_features(pool, output: Path, *, states=64, renderer_factory=None, stream_config=None,
                            recover_checkpoint: Path | None = None, recover_log: Path | None = None, progress=None):
    """Recover an audited render prefix or collect it directly; never extend the attempt budget."""
    from .checkpoints import atomic_torch_save, load_training_checkpoint
    if type(states) is not int or not 2 <= states <= 256 or (recover_checkpoint is None) != (recover_log is None):
        raise ValueError("paired cache needs 2..256 states and both recovery inputs together")
    started = perf_counter()
    sampler = PilotSampler(pool, policy="paired_views", seed=0)
    if 2 * states > sampler.epoch_size:
        raise ValueError("paired cache cannot repeat training states within its attempt prefix")
    stream_config = stream_config or StreamConfig(prefetch_clips=1, timeout_seconds=30)
    source = load_training_checkpoint(recover_checkpoint) if recover_checkpoint else None
    frontend = features_for(pool.codec, source["basic_pitch"] if source else None)
    replay, history, renderer_identity, origin = [], [], None, None
    if source is not None:
        contract = source["run_contract"]
        if (source.get("format") != "obruxo_pilot_training_v1" or contract["pool_id"] != pool.pool_id
                or contract["codec"] != pool.codec.to_dict() or contract["sampler_id"] != sampler.sampler_id
                or contract["frontend"] != frontend.describe() or contract["stream"] != stream_config.to_dict()
                or source["cursor"]["next_sequence_id"] != 2 * states):
            raise ValueError("recovery checkpoint differs from the requested paired prefix")
        history = [json.loads(line) for line in recover_log.read_text(encoding="utf-8").splitlines()]
        history = [row for row in history if row["sequence_id"] < 2 * states]
        replay, cursor, renderer_identity = source["replay"], source["cursor"], source["renderer_identity"]
        if [item["sequence_id"] for item in replay] != [row["sequence_id"] for row in history if row["status"] == "accepted"]:
            raise ValueError("recovered prefix must retain every accepted feature before pair qualification")
        origin = {"checkpoint_sha256": sha256_bytes(recover_checkpoint.read_bytes()), "render_log_sha256": sha256_bytes(recover_log.read_bytes()),
                  "source_step": source["step"], "policy": "recover_rendered_features_only_discard_learning_state"}
    else:
        targets = {entry.state_id: entry.target.to_dict() for entry in pool.entries}
        with OnlineRenderStream(sampler, 2 * states, config=stream_config, renderer_factory=renderer_factory or NativeRendererFactory()) as stream:
            for outcome in stream:
                recipe = outcome.recipe
                history.append({"sequence_id": recipe.sequence_id, "recipe_id": recipe.recipe_id, **outcome.metadata})
                if outcome.audio is not None:
                    identity = {key: value for key, value in outcome.metadata["provenance"].items() if key != "request_id"}
                    if renderer_identity is not None and identity != renderer_identity:
                        raise ValueError("paired feature renderer provenance changed")
                    renderer_identity = identity
                    replay.append({"sequence_id": recipe.sequence_id, "state_id": recipe.state_id, "performance_id": recipe.performance_id,
                                   "recipe_id": recipe.recipe_id, "audio_sha256": outcome.metadata["qa"]["audio_float32_sha256"],
                                   "target": targets[recipe.state_id], "features": frontend(outcome.audio), "uses": 0})
                stream.acknowledge(recipe.sequence_id)
                if progress:
                    progress({"stage": "render", "sequence_id": recipe.sequence_id, "status": outcome.metadata["status"],
                              "replay_examples": len(replay), "step": 0})
            cursor = stream.state_dict()
    pairs = complete_pairs(replay)
    retained = {index for groups in pairs.values() for pair in groups for index in pair}
    qualified = [item for index, item in enumerate(replay) if index in retained]
    saved = {"format": FORMAT, "pool_id": pool.pool_id, "codec": pool.codec.to_dict(), "sampler_id": sampler.sampler_id,
             "frontend": frontend.describe(), "basic_pitch": frontend.basic_pitch.state_dict(), "stream": stream_config.to_dict(),
             "cursor": cursor, "renderer_identity": renderer_identity, "history": history, "replay": qualified, "origin": origin,
             "features_sha256": canonical_sha256([{"sequence_id": item["sequence_id"], "sha256": tensor_digest(item["features"])} for item in qualified])}
    accepted = validate_cache(saved, pool, sampler, frontend, stream_config)
    if len(pairs) < 2:
        raise ValueError("paired cache has fewer than two qualified states")
    output.mkdir(parents=True, exist_ok=False)
    atomic_torch_save(output / "features.pt", saved)
    summary = {"format": FORMAT, "planned_states": states, "qualified_states": len(pairs), "attempted": 2 * states, "accepted": accepted,
               "rejected_or_errors": 2 * states - accepted, "retained_views": len(qualified), "discarded_unpaired_accepted_views": accepted - len(qualified),
               "new_render_attempts": 0 if source else 2 * states, "origin": origin, "elapsed_seconds": perf_counter() - started,
               "source_render_wall_seconds": sum(row.get("wall_seconds", 0.) for row in history),
               "renderer_identity": renderer_identity, "features_sha256": saved["features_sha256"], "audio_files_written": 0}
    write_json(output / "summary.json", summary)
    return summary

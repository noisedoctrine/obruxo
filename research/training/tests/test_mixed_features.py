from dataclasses import replace
import copy

import pytest
import torch

from obruxo_training import mixed_features as cache
from obruxo_training.checkpoints import atomic_torch_save, load_training_checkpoint
from obruxo_training.features import AudioFeatures, tensor_digest
from obruxo_training.mixed_input_performances import MixedSampler
from obruxo_training.model import PilotInverse
from obruxo_training.pilot import PilotPool
from obruxo_training.profiled_sampler import profile_metadata
from obruxo_training.stream import OnlineRenderStream, StreamConfig
from obruxo_training.trainer import TrainConfig, export_model, load_replay_seed, train_pilot


def synthetic_cache(pool, sampler, frontend, stream):
    targets = {entry.state_id: entry.target.to_dict() for entry in pool.entries}
    parents = {}
    generator = torch.Generator().manual_seed(77)
    for name, source in (("original", sampler.original), ("wide", sampler.wide)):
        items = []
        for index in range(source.epoch_size):
            recipe = source.recipe_at(index)
            items.append({"state_id": recipe.state_id, "sequence_id": index, "recipe_id": recipe.recipe_id,
                          "performance_id": recipe.performance_id, "target": targets[recipe.state_id], "audio_sha256": f"fixture-{name}-{index}",
                          "features": {"pitch": torch.rand(3521, generator=generator), "timbre": torch.rand(2, 96, 32, generator=generator)},
                          "uses": 0, **profile_metadata(32768)})
        parents[name] = {"replay": items}
    native = {"renderer_id": pool.codec.contract.renderer_id, "backend_version": "fixture", "engine_fingerprint": "fixture",
              "settings": {"preroll_frames": 32768}}
    replay, distinct = cache.remap_replay(sampler, parents)
    cursor = OnlineRenderStream(sampler, 1, config=stream).state_dict()
    cursor["next_sequence_id"] = sampler.epoch_size
    saved = {"format": cache.FORMAT, "pool_id": pool.pool_id, "codec": pool.codec.to_dict(), "sampler_id": sampler.sampler_id,
             "observation_contract": sampler.observation_contract, "frontend": frontend.describe(), "basic_pitch": frontend.basic_pitch.state_dict(),
             "stream": stream.to_dict(), "cursor": cursor, "renderer_identity": native, "origins": {"original": "fixture", "wide": "fixture"},
             "replay": replay, "distinct_parent_observations": distinct, "features_sha256": cache.feature_digest(replay)}
    return saved, parents, native


def test_mixed_feature_training_has_no_renders_and_exact_recovery(tmp_path, monkeypatch):
    pool, frontend = PilotPool.create(12, seed=0), AudioFeatures()
    sampler, stream = MixedSampler(pool), StreamConfig(prefetch_clips=1)
    saved, parents, native = synthetic_cache(pool, sampler, frontend, stream)
    monkeypatch.setattr(cache, "load_origins", lambda *_args: (parents, native))
    config = TrainConfig(steps=6, replay_capacity=sampler.epoch_size, replay_mib=64, batch_size=4, render_budget=sampler.epoch_size,
                         checkpoint_every=2, cpu_threads=2, prediction_scope="basic", preroll_frames=32768,
                         sampling_policy="paired_views", input_performances="mixed")
    rng = torch.get_rng_state().clone()
    cache.validate_cache(saved, pool, sampler, frontend, config, stream)
    assert torch.equal(rng, torch.get_rng_state())
    assert saved["replay"][0]["features"] is saved["replay"][2]["features"]
    for change in ("features", "origin", "frontend", "scope"):
        changed = copy.deepcopy(saved)
        if change == "features":
            changed["replay"][0]["features"]["pitch"][0] += 1
            changed["features_sha256"] = cache.feature_digest(changed["replay"])
        elif change == "origin":
            changed["replay"][0]["origin"]["bank"] = "wide"
        elif change == "frontend":
            changed["frontend"]["sample_rate"] = 22050
        with pytest.raises(ValueError, match="mixed"):
            cache.validate_cache(changed, pool, sampler, frontend, replace(config, input_performances="initial") if change == "scope" else config, stream)
    path, initial_path = tmp_path / "features.pt", tmp_path / "initial.pt"
    atomic_torch_save(path, saved)
    export_model(initial_path, PilotInverse(pool.codec), frontend, pool.codec, pool.pool_id, preroll_frames=32768)

    def validator_with_random_imports(*_args):
        torch.rand(7)
        return parents, native

    monkeypatch.setattr(cache, "load_origins", validator_with_random_imports)
    rng = torch.get_rng_state().clone()
    load_replay_seed(path, pool, sampler, frontend, config, stream)
    assert torch.equal(rng, torch.get_rng_state())

    def forbid_render():
        raise AssertionError("cached training must not render")

    def run(name, **kwargs):
        return train_pilot(pool, tmp_path / name, config=config, stream_config=stream, renderer_factory=forbid_render,
                           seed_replay=None if "resume" in kwargs else path, initial_model=None if "resume" in kwargs else initial_path, **kwargs)

    full = run("full")
    run("stopped", stop_after=2)
    recovered = run("recovered", resume=tmp_path / "stopped/checkpoint.pt")
    assert full["model_state_sha256"] == recovered["model_state_sha256"] and full["totals"]["render_attempts"] == 0
    left, right = [load_training_checkpoint(tmp_path / name / "checkpoint.pt") for name in ("full", "recovered")]
    torch.testing.assert_close(left["optimizer"], right["optimizer"], rtol=0, atol=0)
    assert torch.equal(left["batch_rng"], right["batch_rng"]) and torch.equal(left["rng"]["torch"], right["rng"]["torch"])
    assert full["replay_seed"]["distinct_parent_observations"] * 4 == len(left["replay"]) * 3
    assert tensor_digest(left["basic_pitch"]) == frontend.frozen_digest

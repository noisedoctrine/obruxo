from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from obruxo_data.hashing import canonical_sha256
from obruxo_training.codec import load_codec
from obruxo_training.contracts import TrainingContract
from obruxo_training.extended_features import ExtendedAudioFeatures, features_for
from obruxo_training.features import AudioFeatures
from obruxo_training.observation_bank import PROFILE, observation_bank
from obruxo_training.observation_codec import extend_observations
from obruxo_training.performance_eligibility import attack_seconds, longest_held_seconds
from obruxo_training.pilot import PilotPool, PilotSampler
from obruxo_training.stream import request_frames
from test_corpus_pool import original_pool, files  # noqa: F401
from test_corpus_inputs import corpus  # noqa: F401
from test_curve_inputs import layout_codec  # noqa: F401
from test_combined import codec, expression  # noqa: F401
from test_factory_assets import bank  # noqa: F401
from test_lfo_bank import population  # noqa: F401


def test_new_profile_retains_bank_sources_and_legacy_contract_identity(original_pool):  # noqa: F811
    before = original_pool.codec.to_dict()
    extended = extend_observations(original_pool)
    assert original_pool.codec.to_dict() == before
    assert extended.entries == original_pool.entries and extended.manifest == original_pool.manifest
    assert extended.pool_id != original_pool.pool_id
    assert "performance_profile" not in original_pool.codec.contract.to_dict()
    assert original_pool.codec.contract.to_dict()["version"] == 2
    assert extended.codec.contract.to_dict()["version"] == 3
    assert extended.codec.contract.performance_profile == PROFILE
    old = original_pool.codec.contract.performances()
    new = extended.codec.contract.performances()
    assert len(old) == 16 and len(new) == 25 and new[:16] == old
    assert max(longest_held_seconds(entry.performance()) for entry in new if entry.role == "reference") == 40.
    assert load_codec(extended.codec.to_dict()).codec_id == extended.codec.codec_id
    with pytest.raises(ValueError, match="unsupported performance profile"):
        TrainingContract.create(original_pool.codec.schema, original_pool.codec.catalog, performance_profile="typo")


def test_slowest_attack_is_observable_and_transfer_training_remains_forbidden(original_pool, tmp_path):  # noqa: F811
    extended = extend_observations(original_pool)
    first = extended.entries[0]
    raw = json.loads(extended.source_preset(first.target, 0))
    raw["settings"].update({name: extended.codec.schema.parameters[name]["max"] for name in ("env_1_delay", "env_1_attack")})
    assert extended.codec.schema.validate(raw).valid
    target = extended.codec.encode(json.dumps(raw).encode())
    span = attack_seconds(extended.codec, target)
    assert 32. < span < 40.
    # The sampler consults the new bank even before the source is opened.
    selected = replace(extended, entries=(replace(first, target=target),))
    sampler = PilotSampler(selected, policy="attack_compatible")
    choices = [sampler._performances[index] for index in sampler._eligible[first.state_id]]
    assert choices and all(longest_held_seconds(entry.performance()) >= span for entry in choices)
    with pytest.raises(ValueError, match="transfer performances"):
        PilotSampler(selected, role="transfer")
    extended.save(tmp_path / "pool.json")
    assert PilotPool.load(tmp_path / "pool.json").pool_id == extended.pool_id
    # Largest declared request fits the existing worker's 32 MiB cap and new 80s frontend.
    from obruxo_data.render.base import RenderRequest
    for entry in observation_bank():
        request = RenderRequest(preset_json=json.dumps(raw), performance=entry.performance(), renderer_id=extended.codec.contract.renderer_id,
                                tail_seconds=extended.codec.contract.tail_seconds)
        assert request_frames(request) * 8 < 32 * 1024**2 and request_frames(request) / 44100 <= 80


def test_long_frontend_uses_entire_recording_and_preserves_short_features():
    torch.set_num_threads(2)
    old = AudioFeatures()
    new = ExtendedAudioFeatures(old.basic_pitch.state_dict())
    time = np.arange(44100, dtype=np.float32) / 44100
    short = np.repeat((.1 * np.sin(2 * np.pi * 220 * time))[:, None], 2, axis=1).astype(np.float32)
    old_features, new_features = old(short), new(short)
    assert all(torch.equal(old_features[name], new_features[name]) for name in old_features)
    assert old.frontend_id != new.frontend_id and old.describe()["maximum_audio_seconds"] == 16
    long = np.zeros((18 * 44100, 2), dtype=np.float32)
    long[-44100:] = short
    result = new(long)
    blank = new(np.zeros_like(long))
    assert not torch.equal(result["timbre"], blank["timbre"])
    assert not torch.equal(result["pitch"][:-1], blank["pitch"][:-1])
    assert result["pitch"][-1] == 18 / 16
    with pytest.raises(ValueError, match="16 seconds"):
        old(long)
    with pytest.raises(ValueError, match="80 seconds"):
        new(np.zeros((80 * 44100 + 1, 2), dtype=np.float32))
    old.require_frozen()
    new.require_frozen()


def test_extended_training_and_export_recover_with_selected_frontend(original_pool, tmp_path):  # noqa: F811
    from obruxo_training.trainer import TrainConfig, load_model, train_pilot
    from obruxo_training.stream import StreamConfig
    from test_stream import FakeRendererFactory
    pool = extend_observations(original_pool)
    assert isinstance(features_for(pool.codec), ExtendedAudioFeatures)
    config = TrainConfig(steps=2, replay_capacity=2, batch_size=2, render_budget=3, checkpoint_every=1, cpu_threads=2,
                         sampling_policy="attack_compatible")

    def train(name, **kwargs):
        return train_pilot(pool, tmp_path / name, config=config, stream_config=StreamConfig(prefetch_clips=1),
                           renderer_factory=FakeRendererFactory(), **kwargs)

    full = train("full")
    train("partial", stop_after=1)
    recovered = train("recovered", resume=tmp_path / "partial/checkpoint.pt")
    assert full["model_state_sha256"] == recovered["model_state_sha256"]
    _, frontend, exported = load_model(tmp_path / "full/model.pt")
    assert isinstance(frontend, ExtendedAudioFeatures) and exported.codec_id == pool.codec.codec_id
    saved = torch.load(tmp_path / "full/model.pt", weights_only=True)
    assert saved["contract"]["performance_bank_id"] == canonical_sha256(saved["codec"]["performance_bank"])

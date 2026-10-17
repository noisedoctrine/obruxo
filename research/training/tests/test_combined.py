from dataclasses import replace
import json

import pytest
import torch

from obruxo_training.codec import load_codec
from obruxo_training.combined_codec import CombinedCodec
from obruxo_training.combined_policy import FIXED
from obruxo_training.pilot import PilotPool
from test_expression import codec as expression  # noqa: F401
from test_factory_assets import bank  # noqa: F401
from test_lfo_bank import population  # noqa: F401


@pytest.fixture(scope="module")
def codec(expression):  # noqa: F811
    return CombinedCodec(expression.base.bank, expression.lfo_bank.to_dict())


@pytest.fixture(scope="module")
def pool(codec):
    return PilotPool.create(32, codec=codec)


def test_policy_covers_every_scalar_and_freezes_only_declared_fields(codec):
    assert set(codec.policy) == codec.schema.scalar_names and len(codec.policy) == 775
    assert {name for name, rule in codec.policy.items() if rule["kind"] == "fixed"} == set(FIXED)
    assert codec.contract.tail_seconds == 4. and len(codec.continuous) == 7187 and len(codec.categorical) == 898
    assert load_codec(codec.to_dict()).codec_id == codec.codec_id
    for name, rule in codec.policy.items():
        if rule["kind"] != "choice":
            continue
        spec = codec.base.scalar._parameters[name]
        assert all(spec.minimum <= value <= spec.maximum for value in rule["values"])


def test_extended_tail_is_explicit_bounded_and_preserves_the_legacy_default(codec):
    from obruxo_training.codec import PilotCodec
    from obruxo_training.contracts import TrainingContract
    legacy = PilotCodec()
    assert TrainingContract.create(codec.schema, codec.catalog).tail_seconds == 2.
    assert TrainingContract.create(codec.schema, codec.catalog, tail_seconds=4.).contract_id == codec.contract.contract_id
    with pytest.raises(ValueError, match="full preset profile"):
        TrainingContract.create(legacy.schema, legacy.catalog, tail_seconds=4.)
    with pytest.raises(ValueError, match="reviewed"):
        TrainingContract.create(codec.schema, codec.catalog, tail_seconds=16.)


def test_inactive_native_audio_destinations_are_excluded_without_relabeling_supported_choices(codec, pool):
    values = dict(codec.categorical)
    for name in ("osc_1_destination", "osc_2_destination", "osc_3_destination", "sample_destination"):
        assert values[name] == tuple(range(5))
        assert codec.policy[name]["excluded_native_values"] == list(range(5, 15))
    raw = json.loads(codec.source_preset(pool.entries[0].target, 0))
    raw["settings"]["sample_destination"] = 5.
    with pytest.raises(ValueError):
        codec.encode(json.dumps(raw).encode())


def test_original_curves_maps_seeds_and_positional_references_survive(codec, pool, expression):  # noqa: F811
    parent = PilotPool.create(32, codec=expression)
    assert pool.manifest.parent.to_dict() == parent.manifest.parent.to_dict()
    for index in range(4):
        raw = codec.source_preset(pool.entries[index].target, index)
        assert codec.encode(raw) == pool.entries[index].target
        actual = json.loads(raw)["settings"]
        original = json.loads(expression.source_preset(parent.entries[index].target, index))["settings"]
        for name in ("lfos", "custom_warps", "random_values"):
            assert actual[name] == original[name]
        for current, previous in zip(actual["modulations"], original["modulations"], strict=True):
            if previous["source"]:
                assert current == previous
    target = pool.entries[0].target
    with pytest.raises(ValueError, match="original source policy"):
        codec.source_preset(replace(target, continuous=(1. - target.continuous[0], *target.continuous[1:])), 0)


def test_combined_pool_roundtrip_and_cli(codec, pool, tmp_path):
    from obruxo_training.cli import main
    path = tmp_path / "saved.json"
    pool.save(path)
    assert PilotPool.load(path).pool_id == pool.pool_id
    assets, curves, output = tmp_path / "assets.json", tmp_path / "curves.json", tmp_path / "pool"
    assets.write_text(json.dumps(codec.base.bank))
    curves.write_text(json.dumps(codec.lfo_bank.to_dict()))
    assert main(["combined-pool", "--asset-bank", str(assets), "--lfo-bank", str(curves), "--output", str(output), "--count", "12"]) == 0
    assert isinstance(PilotPool.load(output / "pool.json").codec, CombinedCodec)


def test_combined_training_recovers_without_midi_inputs(codec, tmp_path):
    from obruxo_training.model import batch_features
    from obruxo_training.stream import StreamConfig
    from obruxo_training.trainer import TrainConfig, train_pilot
    from test_stream import FakeRendererFactory
    pool = PilotPool.create(12, codec=codec)
    config = TrainConfig(steps=3, replay_capacity=4, batch_size=2, render_budget=4, cpu_threads=1, checkpoint_every=1)

    def train(name, **kwargs):
        return train_pilot(pool, tmp_path / name, config=config, stream_config=StreamConfig(prefetch_clips=1), renderer_factory=FakeRendererFactory(), **kwargs)

    complete = train("complete")
    train("partial", stop_after=1)
    restored = train("restored", resume=tmp_path / "partial/checkpoint.pt")
    assert complete["model_state_sha256"] == restored["model_state_sha256"]
    saved = torch.load(tmp_path / "complete/checkpoint.pt", weights_only=True)
    assert set(batch_features(saved["replay"])) == {"timbre", "pitch"}

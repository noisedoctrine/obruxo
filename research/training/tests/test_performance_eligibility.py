from dataclasses import replace

import pytest
import torch

from obruxo_data.midi import EventKind, MidiEvent, Performance
from obruxo_training.performance_eligibility import attack_seconds, eligible, longest_held_seconds
from obruxo_training.pilot import PilotPool, PilotSampler
from test_combined import codec, expression  # noqa: F401
from test_factory_assets import bank  # noqa: F401
from test_lfo_bank import population  # noqa: F401


def test_duration_counts_tempo_and_master_sustain_for_member_channel():
    performance = Performance(bpm=120)
    performance.add_note(pitch=60, velocity=90, start_tick=0, duration_ticks=120, channel=2)
    performance.events.extend((MidiEvent(60, 10, EventKind.CONTROL_CHANGE, 0, (64, 127)), MidiEvent(480, 11, EventKind.CONTROL_CHANGE, 0, (64, 0))))
    performance.end_tick = 960
    assert longest_held_seconds(performance) == .5
    performance.events[-1] = MidiEvent(960, 11, EventKind.CONTROL_CHANGE, 0, (64, 0))
    assert longest_held_seconds(performance) == 1.


def test_same_tick_pedal_down_cannot_recover_a_released_note_and_tempo_changes_count():
    performance = Performance(bpm=120)
    performance.add_note(pitch=60, velocity=90, start_tick=0, duration_ticks=960)
    performance.add_tempo(60, tick=480)
    performance.events.append(MidiEvent(960, 10, EventKind.CONTROL_CHANGE, 0, (64, 127)))
    performance.end_tick = 1920
    assert longest_held_seconds(performance) == 1.5


@pytest.fixture(scope="module")
def pool(codec):  # noqa: F811
    return PilotPool.create(32, codec=codec)


def test_sampler_preserves_sources_and_only_selects_observable_reference_views(pool):
    sampler = PilotSampler(pool, policy="attack_compatible")
    recipes = [sampler.recipe_at(index) for index in range(sampler.epoch_size * 2)]
    assert len({recipe.state_id for recipe in recipes[:sampler.epoch_size]}) == sampler.epoch_size
    for recipe in recipes:
        index = next(index for index, entry in enumerate(pool.entries) if entry.state_id == recipe.state_id)
        target = pool.entries[index].target
        assert recipe.performance_role == "reference" and recipe.split == "train"
        assert recipe.render_request().preset_json.encode() == pool.codec.source_preset(target, index)
        assert eligible(pool.codec, target, recipe.render_request().performance)
    assert sampler.sampler_id != PilotSampler(pool, policy="state_round_robin").sampler_id


def test_short_view_is_rejected_without_modifying_envelope_or_highpass(pool):
    from obruxo_training.codec import PilotTarget
    source_codec, target = pool.codec, pool.entries[0].target
    scalar = source_codec.base.scalar
    settings = scalar._values(PilotTarget(target.continuous[:len(scalar.continuous)], target.categorical[:len(scalar.categorical)]))
    settings.update(env_1_delay=.58, env_1_attack=.4, osc_1_spectral_morph_type=8., osc_1_spectral_morph_amount=.75)
    prefix = scalar._target(settings)
    target = replace(target, continuous=prefix.continuous + target.continuous[len(prefix.continuous):],
                     categorical=prefix.categorical + target.categorical[len(prefix.categorical):])
    before = target.to_dict()
    short, held = Performance(bpm=120), Performance(bpm=120)
    short.add_note(pitch=60, velocity=90, start_tick=0, duration_ticks=120)
    held.add_note(pitch=60, velocity=90, start_tick=0, duration_ticks=1440)
    assert not eligible(source_codec, target, short) and eligible(source_codec, target, held)
    assert attack_seconds(source_codec, target) > .125
    assert target.to_dict() == before
    unchanged = scalar._values(PilotTarget(target.continuous[:len(scalar.continuous)], target.categorical[:len(scalar.categorical)]))
    assert {name: unchanged[name] for name in ("env_1_delay", "env_1_attack", "osc_1_spectral_morph_type", "osc_1_spectral_morph_amount")} == pytest.approx({
        name: settings[name] for name in ("env_1_delay", "env_1_attack", "osc_1_spectral_morph_type", "osc_1_spectral_morph_amount")})


def test_compatible_schedule_recovers_exactly_and_checks_saved_performance_ids(codec, tmp_path):  # noqa: F811
    from obruxo_training.stream import StreamConfig
    from obruxo_training.trainer import TrainConfig, train_pilot, validate_replay
    from test_stream import FakeRendererFactory
    pool = PilotPool.create(12, codec=codec)
    config = TrainConfig(steps=3, replay_capacity=4, batch_size=2, render_budget=4, cpu_threads=1, checkpoint_every=1, sampling_policy="attack_compatible")

    def train(name, **kwargs):
        return train_pilot(pool, tmp_path / name, config=config, stream_config=StreamConfig(prefetch_clips=1), renderer_factory=FakeRendererFactory(), **kwargs)

    complete = train("complete")
    train("partial", stop_after=1)
    recovered = train("recovered", resume=tmp_path / "partial/checkpoint.pt")
    assert complete["model_state_sha256"] == recovered["model_state_sha256"]
    saved = torch.load(tmp_path / "complete/checkpoint.pt", weights_only=True)
    saved["replay"][0]["performance_id"] = "0" * 64
    with pytest.raises(ValueError, match="performance differs"):
        validate_replay(saved["replay"], PilotSampler(pool, policy="attack_compatible"), saved["cursor"], config)

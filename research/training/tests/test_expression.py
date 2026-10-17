from fractions import Fraction
import json

import mido
import pytest
import torch

from obruxo_data.midi import EventKind, MidiEvent, Performance, TempoMap
from obruxo_data.render.base import FULL_EXPRESSION_RENDERER_ID, FULL_SEED_RENDERER_ID, RenderRequest
from obruxo_data.render.capabilities import EXPRESSION_CAPABILITIES, RendererCapabilities
from obruxo_data.render.midi_transport import write_sample_midi
from obruxo_data.render.vita import VitalRenderer
from obruxo_data.vital.full_contract import FullPresetContract
from obruxo_training.codec import PilotTarget, load_codec
from obruxo_training.contracts import RenderRecipe
from obruxo_training.expression_bank import expression_bank
from obruxo_training.expression_check import legacy_chord_noteoffs
from obruxo_training.expression_codec import EXPRESSION_ROUTES, ExpressionCodec
from obruxo_training.lfo_bank import fit_lfo_bank
from obruxo_training.nested_codec import NestedStateCodec
from obruxo_training.performances import bank_for, initial_bank
from obruxo_training.pilot import PilotPool, PilotSampler
from obruxo_training.splits import SplitLedger
from test_factory_assets import bank  # noqa: F401
from test_lfo_bank import population  # noqa: F401


@pytest.fixture(scope="module")
def codec(bank, population):  # noqa: F811 - shared pytest fixtures
    manifest, train, _ = population
    return ExpressionCodec(bank, fit_lfo_bank(train, manifest, maximum_curves=48).to_dict())


@pytest.fixture(scope="module")
def pool(codec):
    return PilotPool.create(32, codec=codec)


def expressive_performance():
    value = Performance(bpm=120)
    value.add_note(pitch=69, velocity=96, start_tick=1, duration_ticks=960, channel=15, release_velocity=112)
    value.events.extend((MidiEvent(120, 3, EventKind.PITCH_BEND, 15, (4096,)), MidiEvent(240, 4, EventKind.CHANNEL_PRESSURE, 15, (90,)),
                         MidiEvent(480, 5, EventKind.CONTROL_CHANGE, 15, (74, 127))))
    value.end_tick = 1440
    return value


@pytest.mark.parametrize("sample_rate", [44100, 48000, 96000])
def test_smf_clock_preserves_messages_channels_release_velocity_and_exact_sample_grid(tmp_path, sample_rate):
    value = expressive_performance()
    value.add_tempo(90, tick=600)
    path = tmp_path / "events.mid"
    report = write_sample_midi(value, path, sample_rate)
    midi = mido.MidiFile(path)
    tempo = midi.tracks[0][0].tempo
    tick, records = 0, []
    for message in midi.tracks[0]:
        tick += message.time
        if not message.is_meta:
            records.append((message, Fraction(tick * tempo * sample_rate, midi.ticks_per_beat * 1_000_000)))
    assert [position for _, position in records] == [Fraction(row["sample"] * 4 + 1, 4) for row in report["events"]]
    assert [row["sample"] for row in report["events"]] == [TempoMap.from_performance(value).tick_to_sample(event.tick, sample_rate)
                                                        for event in value.canonical_events() if event.kind != EventKind.TEMPO]
    assert all(message.channel == 15 for message, _ in records)
    assert records[-1][0].type == "note_off" and records[-1][0].velocity == 112
    assert {message.type for message, _ in records} == {"note_on", "note_off", "pitchwheel", "aftertouch", "control_change"}
    assert write_sample_midi(value, path, sample_rate) == report


def test_half_even_boundaries_ties_and_zero_sample_notes(tmp_path):
    value = Performance(ticks_per_beat=88200, bpm=60)
    for index, tick in enumerate((1, 3, 127, 129, 255, 257)):
        value.add_note(pitch=60 + index, velocity=90, start_tick=tick, duration_ticks=8820)
    result = write_sample_midi(value, tmp_path / "boundary.mid", 44100)
    assert [row["sample"] for row in result["events"] if row["kind"] == "note_on"] == [0, 2, 64, 64, 128, 128]
    value = Performance(ticks_per_beat=88200, bpm=60)
    value.add_note(pitch=60, velocity=90, start_tick=0, duration_ticks=1)
    with pytest.raises(ValueError, match="zero samples"):
        write_sample_midi(value, tmp_path / "invalid.mid", 44100)


def test_legacy_chord_float_floor_does_not_move_the_new_transport_boundary(tmp_path):
    value = next(entry.performance() for entry in initial_bank() if entry.name == "chord_wide")
    expected = TempoMap.from_performance(value).tick_to_sample(1680, 44100)
    assert expected == 61740 and int((expected / 44100) * 44100) == 61739
    actual = write_sample_midi(value, tmp_path / "correct.mid", 44100)
    assert {row["sample"] for row in actual["events"] if row["kind"] == "note_off"} == {61740}
    probe = write_sample_midi(legacy_chord_noteoffs(value), tmp_path / "legacy-probe.mid", 44100)
    assert {row["sample"] for row in probe["events"] if row["kind"] == "note_off"} == {61739}


def test_new_capabilities_are_explicit_and_legacy_profile_refuses_release_velocity():
    value = expressive_performance()
    assert value.validate(EXPRESSION_CAPABILITIES).valid
    old = RendererCapabilities(max_channels=1)
    assert "note_off_velocity" not in old.to_dict() and RendererCapabilities.from_dict(old.to_dict()) == old
    assert RendererCapabilities.from_dict(EXPRESSION_CAPABILITIES.to_dict()) == EXPRESSION_CAPABILITIES
    assert "midi.capability.note_off_velocity" in {row.code for row in value.validate(old).diagnostics}
    value.events.append(MidiEvent(720, 6, EventKind.CONTROL_CHANGE, 15, (7, 127)))
    assert "midi.capability.control_change" in {row.code for row in value.validate(EXPRESSION_CAPABILITIES).diagnostics}
    value.add_tempo(90, tick=1000)
    assert "midi.capability.tempo_changes" in {row.code for row in value.validate(EXPRESSION_CAPABILITIES).diagnostics}


def test_profile_validation_requires_expression_identity_and_qualified_events():
    renderer = object.__new__(VitalRenderer)
    renderer.renderer_id = FULL_EXPRESSION_RENDERER_ID
    renderer.full_contract = FullPresetContract(renderer_id=FULL_EXPRESSION_RENDERER_ID)
    renderer._capabilities = EXPRESSION_CAPABILITIES
    request = RenderRequest(preset_json=json.dumps(renderer.full_contract.template()), performance=expressive_performance(), renderer_id=renderer.renderer_id)
    assert not renderer._validate_request(request)
    renderer.renderer_id = FULL_SEED_RENDERER_ID
    with pytest.raises(ValueError, match="does not match"):
        renderer._validate_request(request)


def test_expression_pool_preserves_roots_source_labels_and_existing_nested_codec_identity(codec, pool, tmp_path):
    original = NestedStateCodec(codec.base.bank, codec.lfo_bank.to_dict())
    old_pool = PilotPool.create(32, codec=original)
    assert pool.manifest.parent.to_dict() == old_pool.manifest.parent.to_dict()
    assert original.contract.renderer_id == FULL_SEED_RENDERER_ID and codec.contract.renderer_id == FULL_EXPRESSION_RENDERER_ID
    assert original.contract.performance_bank_id != codec.contract.performance_bank_id
    for index, entry in enumerate(pool.entries[:4]):
        raw = codec.source_preset(entry.target, index)
        assert codec.encode(raw) == entry.target
        settings = json.loads(raw)["settings"]
        assert settings["volume"] == 2500. and settings["mpe_enabled"] == 1.
        routes = {(route["source"], route["destination"]) for route in settings["modulations"]}
        assert set(EXPRESSION_ROUTES) <= routes
        assert all(shape["num_points"] <= 8 for shape in settings["lfos"])
        assert all(shape["num_points"] <= 6 for shape in settings["custom_warps"])
    path = tmp_path / "pool.json"
    pool.save(path)
    assert PilotPool.load(path).pool_id == pool.pool_id
    assert load_codec(codec.to_dict()).codec_id == codec.codec_id and load_codec(original.to_dict()).codec_id == original.codec_id


def test_changed_expression_connections_are_refused(codec, pool):
    target = pool.entries[0].target
    routes = codec.routing.routes(codec._routing_target(target))
    slot = next(index for index, route in enumerate(routes) if route["source"] == "mod_wheel")
    flags = list(target.categorical)
    offset = codec.routing._curve_cat + slot * 4 + 1
    flags[offset] = (flags[offset] + 1) % len(codec.routing.sources)
    with pytest.raises(ValueError, match="original source policy"):
        codec.source_preset(PilotTarget(target.continuous, tuple(flags)), 0)


def test_sampler_covers_new_reference_bank_and_transfer_is_rejected_for_training(codec, pool):
    sampler = PilotSampler(pool, policy="performance_round_robin")
    expected = {entry.performance_id for entry in expression_bank() if entry.role == "reference"}
    recipes = [sampler.recipe_at(index) for index in range(sampler.epoch_size)]
    assert {recipe.performance_id for recipe in recipes} == expected
    assert {recipe.performance_id for recipe in recipes[:8]} == expected
    assert len({(recipe.state_id, recipe.performance_id) for recipe in recipes}) == sampler.epoch_size
    assert len(bank_for(codec.contract.renderer_id)) == 16 and len(initial_bank()) == 8
    entry = next(entry for entry in expression_bank() if entry.role == "transfer" and entry.name.startswith("expression_"))
    recipe = recipes[0]
    with pytest.raises(ValueError, match="held-out state or transfer"):
        RenderRecipe.create(recipe.render_request().preset_json.encode(), entry.performance(), schema=codec.schema, catalog=codec.catalog,
                            contract=codec.contract, ledger=SplitLedger(pool.manifest), purpose="train", sequence_id=0)


def test_expression_training_recovers_exactly_with_audio_only_features(codec, tmp_path):
    from obruxo_training.model import batch_features
    from obruxo_training.stream import StreamConfig
    from obruxo_training.trainer import TrainConfig, train_pilot
    from test_stream import FakeRendererFactory
    pool = PilotPool.create(12, codec=codec)
    config = TrainConfig(steps=3, replay_capacity=4, batch_size=2, render_budget=4, cpu_threads=1, checkpoint_every=1, sampling_policy="performance_round_robin")

    def train(name, **kwargs):
        return train_pilot(pool, tmp_path / name, config=config, stream_config=StreamConfig(prefetch_clips=1), renderer_factory=FakeRendererFactory(), **kwargs)

    complete = train("complete")
    train("partial", stop_after=1)
    recovered = train("recovered", resume=tmp_path / "partial/checkpoint.pt")
    assert complete["model_state_sha256"] == recovered["model_state_sha256"]
    state = torch.load(tmp_path / "complete/checkpoint.pt", weights_only=True)
    assert set(batch_features(state["replay"])) == {"timbre", "pitch"}
    assert state["run_contract"]["contract"]["midi"] == EXPRESSION_CAPABILITIES.to_dict()
    assert all(set(item["features"]) == {"timbre", "pitch"} for item in state["replay"])


def test_expression_pool_cli_uses_separate_performance_contract(codec, tmp_path):
    from obruxo_training.cli import main
    assets, curves, output = tmp_path / "assets.json", tmp_path / "curves.json", tmp_path / "pool"
    assets.write_text(json.dumps(codec.base.bank))
    curves.write_text(json.dumps(codec.lfo_bank.to_dict()))
    assert main(["expression-pool", "--asset-bank", str(assets), "--lfo-bank", str(curves), "--output", str(output), "--count", "12"]) == 0
    assert isinstance(PilotPool.load(output / "pool.json").codec, ExpressionCodec)

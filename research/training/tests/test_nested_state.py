from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from obruxo_data.hashing import canonical_json
from obruxo_data.render.base import FULL_PRESET_RENDERER_ID, FULL_SEED_RENDERER_ID, RenderRequest
from obruxo_data.render.full_state import verify_full_nested_state
from obruxo_data.vital.full_contract import FullPresetContract
from obruxo_training.codec import PilotTarget, load_codec
from obruxo_training.codec_check import CodecCheckSampler
from obruxo_training.curve_labels import CURVE_LABEL_KEYS
from obruxo_training.features import PITCH_SIZE
from obruxo_training.full_scalar import FullScalarCodec
from obruxo_training.lfo_bank import fit_lfo_bank
from obruxo_training.model import PilotInverse, batch_features, loss_terms, predicted_targets
from obruxo_training.nested_check import SEED_EDGES
from obruxo_training.nested_codec import FAMILIES, NestedStateCodec, seed_bits
from obruxo_training.pilot import PilotPool, PilotSampler
from test_factory_assets import bank  # noqa: F401
from test_lfo_bank import population  # noqa: F401
from test_full_render import loaded_state


@pytest.fixture(scope="module")
def codec(bank, population):  # noqa: F811 - shared pytest fixtures
    manifest, train, _ = population
    return NestedStateCodec(bank, fit_lfo_bank(train, manifest, maximum_curves=48).to_dict())


@pytest.fixture(scope="module")
def pool(codec):
    return PilotPool.create(32, codec=codec)


def labels(codec, pool, targets):
    indices = {entry.target: index for index, entry in enumerate(pool.entries)}
    raw = [codec.source_preset(target, indices[target]) for target in targets]
    methods = (codec.source_curves, codec.source_route_maps, codec.source_warp_curves)
    return {key: torch.tensor(np.stack([method(source) for source in raw])) for key, method in zip(CURVE_LABEL_KEYS, methods, strict=True)}


def test_all_nested_outputs_reload_exactly_and_keep_frozen_roots(codec, pool, tmp_path):
    assert (len(codec.continuous), len(codec.categorical)) == (7187, 898)
    assert pool.manifest.parent.to_dict() == PilotPool.create(32, codec=FullScalarCodec()).manifest.parent.to_dict()
    assert {codec.family(entry.target) for entry in pool.entries if pool.manifest.by_state[entry.state_id].split == "train"} == set(FAMILIES)
    path = tmp_path / "pool.json"
    pool.save(path)
    restored = PilotPool.load(path)
    assert restored.pool_id == pool.pool_id and load_codec(codec.to_dict()).codec_id == codec.codec_id
    for index, entry in enumerate(pool.entries[:3]):
        assert codec.canonical_target(entry.target) == entry.target
        assert codec.decode(entry.target) == restored.codec.decode(entry.target)
        original = codec.source_preset(entry.target, index)
        assert codec.encode(original) == entry.target
        assert all(shape["num_points"] <= 6 for shape in json.loads(original)["settings"]["custom_warps"])
        assert all(shape["num_points"] == 97 for shape in json.loads(codec.decode(entry.target))["settings"]["custom_warps"])


@pytest.mark.parametrize("seed", SEED_EDGES)
def test_uint32_seed_heads_preserve_all_bits_without_float_rounding(codec, seed):
    target = codec.default_target()
    expected = (seed, seed ^ 0xFFFFFFFF, (seed + 1) & 0xFFFFFFFF)
    target = PilotTarget(target.continuous, target.categorical[:codec._routing_cat] + seed_bits(expected))
    assert codec.seeds(target) == expected
    decoded = json.loads(codec.decode(target))
    assert tuple(value["seed"] for value in decoded["settings"]["random_values"]) == expected
    assert codec.seeds(codec.encode(canonical_json(decoded).encode())) == expected


@pytest.mark.parametrize("invalid", [-1, 2**32, .5, True, float("nan")])
def test_illegal_seed_values_are_refused(invalid):
    with pytest.raises(ValueError, match="unsigned 32-bit"):
        seed_bits([invalid, 4, 4])


def test_modes_and_numbered_oscillators_control_masks_without_erasing_state(codec):
    document = json.loads(codec.decode(codec.default_target()))
    document["settings"].update(osc_1_on=1., osc_2_on=1., osc_3_on=0., osc_1_spectral_morph_type=14.,
                                 osc_2_spectral_morph_type=6., osc_3_spectral_morph_type=15.)
    target = codec.encode(canonical_json(document).encode())
    assert codec.active_warps(target) == (True, False, False)
    assert codec.active_seeds(target) == (False, True, False)
    assert codec.continuous_mask(target)[codec._warp_start:] == (True,) * 97 + (False,) * 194
    assert codec.categorical_mask(target)[codec._routing_cat:] == (False,) * 32 + (True,) * 32 + (False,) * 32
    continuous = list(target.continuous)
    continuous[codec._warp_start + 194] = .123
    categories = target.categorical[:codec._routing_cat] + seed_bits((0xFFFFFFFF, 4, 16777217))
    saved = codec.canonical_target(PilotTarget(tuple(continuous), categories))
    assert PilotTarget.from_dict(saved.to_dict()) == saved
    decoded = json.loads(codec.decode(saved))["settings"]
    assert decoded["random_values"][0]["seed"] == 0xFFFFFFFF
    assert decoded["custom_warps"][2]["points"][1] == np.float32(.123)


def test_changed_original_warp_and_seed_labels_cannot_relabel_source_audio(codec, pool):
    target = pool.entries[0].target
    values = list(target.continuous)
    values[codec._warp_start + 10] = .1234
    with pytest.raises(ValueError, match="original custom curves and seeds"):
        codec.source_preset(PilotTarget(tuple(values), target.categorical), 0)
    flags = list(target.categorical)
    flags[codec._routing_cat + 31] ^= 1
    with pytest.raises(ValueError, match="original custom curves and seeds"):
        codec.source_preset(PilotTarget(target.continuous, tuple(flags)), 0)


def test_audio_only_loss_trains_warp_and_seed_heads_only_when_active(codec, pool):
    torch.set_num_threads(1)
    targets = [next(entry.target for entry in pool.entries if codec.family(entry.target) == name) for name in ("warp_1_mode_13", "seed_2")]
    model = PilotInverse(codec)
    prediction = model({"timbre": torch.randn(2, 2, 96, 32), "pitch": torch.randn(2, PITCH_SIZE)})
    expected = labels(codec, pool, targets)
    with pytest.raises(ValueError, match="original custom warp curves"):
        loss_terms(prediction, targets, codec, curve_targets=expected["curve_targets"], route_curve_targets=expected["route_curve_targets"])
    terms = loss_terms(prediction, targets, codec, **expected)
    terms["loss"].backward()
    assert terms["warp_curve_mse"] > 0
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    assert model.continuous.weight.grad[codec._warp_start:codec._warp_start + 97].count_nonzero()
    assert not model.continuous.weight.grad[codec._warp_start + 97:].count_nonzero()
    assert model.categorical[codec._routing_cat + 32].weight.grad.count_nonzero()
    assert not model.categorical[codec._routing_cat].weight.grad.count_nonzero()
    assert not model.categorical[codec._routing_cat + 64].weight.grad.count_nonzero()
    for target in predicted_targets(prediction, codec):
        assert codec.schema.loads(codec.decode(target))[1].valid


def test_all_thirteen_saved_target_families_are_checked_without_lossy_reencoding(codec, pool, monkeypatch):
    monkeypatch.setattr(codec, "encode", lambda *args: pytest.fail("native saved-target check must not reencode raw curves"))
    check = CodecCheckSampler(pool)
    assert len(check.recipes) == 39 and {case["family"] for case in check.cases} == set(FAMILIES)
    assert all(case["serialized_target_and_decoder_exact"] for case in check.cases)


def test_nested_checkpoint_recovers_exactly_and_rejects_altered_original_warps(codec, tmp_path):
    from obruxo_training.evaluate import evaluate_pilot, retrieval_digest
    from obruxo_training.stream import StreamConfig
    from obruxo_training.trainer import TrainConfig, replay_bytes, train_pilot, validate_replay
    from test_stream import FakeRendererFactory
    pool = PilotPool.create(12, codec=codec)
    config = TrainConfig(steps=3, replay_capacity=3, batch_size=2, render_budget=3, cpu_threads=1, checkpoint_every=1, sampling_policy="state_round_robin")

    def train(name, **kwargs):
        return train_pilot(pool, tmp_path / name, config=config, stream_config=StreamConfig(prefetch_clips=1), renderer_factory=FakeRendererFactory(), **kwargs)

    complete = train("complete")
    train("partial", stop_after=1)
    resumed = train("resumed", resume=tmp_path / "partial/checkpoint.pt")
    assert complete["model_state_sha256"] == resumed["model_state_sha256"] and complete["cursor"] == resumed["cursor"]
    left = torch.load(tmp_path / "complete/checkpoint.pt", weights_only=True)
    right = torch.load(tmp_path / "resumed/checkpoint.pt", weights_only=True)
    assert torch.equal(left["batch_rng"], right["batch_rng"]) and torch.equal(left["rng"]["torch"], right["rng"]["torch"])
    for index, state in left["optimizer"]["state"].items():
        assert all(torch.equal(value, right["optimizer"]["state"][index][name]) for name, value in state.items())
    assert set(batch_features(left["replay"])) == {"timbre", "pitch"}
    label_bytes = sum(item[name].numel() * item[name].element_size() for item in left["replay"] for name in CURVE_LABEL_KEYS)
    assert replay_bytes(left["replay"]) == label_bytes + replay_bytes([{"features": item["features"]} for item in left["replay"]])
    assert complete["warp_curve_target_bytes"] == 3 * 3 * 97 * 4
    corrupt = deepcopy(left["replay"])
    corrupt[0]["warp_curve_targets"][0, 0] += .001
    assert retrieval_digest(corrupt) != retrieval_digest(left["replay"])
    with pytest.raises(ValueError, match="original custom warp curves differ"):
        validate_replay(corrupt, PilotSampler(pool, policy="state_round_robin"), left["cursor"], config)
    evaluated = evaluate_pilot(pool, tmp_path / "complete/checkpoint.pt", tmp_path / "evaluation", count=1, renderer_factory=FakeRendererFactory())
    assert evaluated["renders_attempted"] == evaluated["renders_accepted"] == 12


def test_nested_pool_cli_keeps_qualified_banks_and_decoder(codec, tmp_path):
    from obruxo_training.cli import main
    assets, curves, output = tmp_path / "assets.json", tmp_path / "curves.json", tmp_path / "pool"
    assets.write_text(json.dumps(codec.base.bank))
    curves.write_text(json.dumps(codec.lfo_bank.to_dict()))
    assert main(["nested-pool", "--asset-bank", str(assets), "--lfo-bank", str(curves), "--output", str(output), "--count", "12"]) == 0
    restored = PilotPool.load(output / "pool.json")
    assert isinstance(restored.codec, NestedStateCodec) and restored.codec.codec_id == codec.codec_id


def test_seed_profile_has_a_separate_contract_and_request_identity(codec):
    assert codec.contract.renderer_id == FULL_SEED_RENDERER_ID
    assert codec.routing.contract.renderer_id == FULL_PRESET_RENDERER_ID
    from obruxo_training.contracts import TrainingContract
    assert TrainingContract.create(codec.schema, codec.catalog).contract_id == codec.contract.contract_id
    performance = PilotSampler(PilotPool.create(1, codec=codec)).recipe_at(0).render_request().performance
    raw = codec.decode(codec.default_target()).decode()
    old = RenderRequest(preset_json=raw, performance=performance, renderer_id=FULL_PRESET_RENDERER_ID)
    new = RenderRequest(preset_json=raw, performance=performance, renderer_id=FULL_SEED_RENDERER_ID)
    assert old.request_id != new.request_id


def test_exact_seed_bit_alias_is_opt_in_and_does_not_change_other_state():
    contract = FullPresetContract()
    document = contract.template()
    document["settings"]["random_values"] = [{"seed": 2147483648}, {"seed": 4294967295}, {"seed": 4}]
    loaded = loaded_state(contract, document)
    loaded["settings"]["random_values"][:2] = [{"seed": -2147483648}, {"seed": -1}]
    before = deepcopy(loaded)
    with pytest.raises(RuntimeError, match="random_values"):
        verify_full_nested_state(document, loaded, contract.spec["fixed_native_ramps"])
    result = verify_full_nested_state(document, loaded, contract.spec["fixed_native_ramps"], signed_seed_bits=True)
    assert result["signed_seed_slots_converted"] == 2 and loaded == before
    for bad in (-2, -1., -(2**31) - 1, True):
        changed = deepcopy(loaded)
        changed["settings"]["random_values"][1]["seed"] = bad
        with pytest.raises(RuntimeError, match="random_values"):
            verify_full_nested_state(document, changed, contract.spec["fixed_native_ramps"], signed_seed_bits=True)
    loaded["settings"]["custom_warps"][0]["points"][1] = .5
    with pytest.raises(RuntimeError, match="custom_warps"):
        verify_full_nested_state(document, loaded, contract.spec["fixed_native_ramps"], signed_seed_bits=True)

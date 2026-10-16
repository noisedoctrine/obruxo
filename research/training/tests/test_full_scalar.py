from copy import deepcopy
import json
import random

import numpy as np
import pytest
import torch

from obruxo_data.hashing import canonical_json
from obruxo_data.render.base import FULL_PRESET_RENDERER_ID
from obruxo_training.codec import PilotCodec, PilotTarget, load_codec
from obruxo_training.codec_check import CodecCheckSampler
from obruxo_training.full_scalar import FullScalarCodec, QUANTIZE, effect_order, effect_rank
from obruxo_training.model import PilotInverse, _family_mean, loss_terms, model_spec, predicted_targets
from obruxo_training.pilot import PilotPool, PilotSampler
from obruxo_training.splits import FrozenVariantManifest, SplitLedger, SplitManifest
from obruxo_training.contracts import RenderRecipe


@pytest.fixture(scope="module")
def codec():
    return FullScalarCodec()


@pytest.fixture(scope="module")
def pool(codec):
    return PilotPool.create(32, seed=3, codec=codec)


def test_all_775_fields_are_represented_and_boundary_vectors_roundtrip(codec):
    assert len(codec._parameters) + 1 + len(QUANTIZE) == 775
    assert len(codec.continuous) == 424 and len(codec.categorical) == 406
    assert sum(len(options) for _, options in codec.categorical) == 3063
    for value in (0., .5, 1.):
        target = PilotTarget((value,) * len(codec.continuous), tuple(round(value * (len(options) - 1)) for _, options in codec.categorical))
        target = codec.canonical_target(target)
        raw = codec.decode(target)
        restored = codec.encode(raw)
        assert restored.categorical == target.categorical
        np.testing.assert_allclose(restored.continuous, target.continuous, rtol=0, atol=1e-12)
        left, right = json.loads(raw)["settings"], json.loads(codec.decode(restored))["settings"]
        assert all(np.float32(left[name]) == np.float32(right[name]) for name in codec.schema.scalar_names)
        assert len(codec.continuous_mask(target)) == 424 and len(codec.categorical_mask(target)) == 406


def test_random_output_choices_always_form_legal_orders_and_bitmasks(codec):
    rng = random.Random(12)
    for index in range(48):
        target = PilotTarget(tuple(rng.random() for _ in codec.continuous), tuple(rng.randrange(len(options)) for _, options in codec.categorical))
        settings = codec.settings(target)
        assert 0 <= settings["effect_chain_order"] <= 362879 and settings["effect_chain_order"].is_integer()
        assert all(0 <= settings[name] <= 8191 and settings[name].is_integer() for name in QUANTIZE)
        assert codec.encode(codec.decode(target)).categorical == target.categorical


def test_near_zero_gain_serializes_on_native_control_grid_without_relaxing_readback(codec):
    from obruxo_data.render.vita import verify_loaded_scalars
    target = codec.default_target()
    index = next(index for index, (name, *_) in enumerate(codec.continuous) if name == "reverb_low_shelf_gain")
    values = list(target.continuous)
    values[index] = float(torch.tensor(1 - 1e-4))
    target = codec.canonical_target(PilotTarget(tuple(values), target.categorical))
    assert codec.canonical_target(target) == target
    raw = json.loads(codec.decode(target))["settings"]["reverb_low_shelf_gain"]
    assert raw == -0.0005998611450195312
    assert verify_loaded_scalars({"settings": {"gain": raw}}, {"settings": {"gain": -0.0005998611450195312}}) == 1


def test_effect_supervision_ignores_only_disabled_order_and_keeps_raw_codec_lossless(codec):
    assert effect_order(0) == tuple(range(9))
    assert effect_order(1) == (0, 1, 2, 3, 4, 5, 6, 8, 7)
    assert effect_order(362879) == tuple(reversed(range(9)))
    targets = []
    for order in ((2, 4, 5, 0, 6, 1, 3, 8, 7), (1, 2, 0, 3, 4, 5, 6, 7, 8), (1, 2, 3, 0, 4, 5, 6, 7, 8)):
        settings = codec.settings(codec.default_target())
        settings.update(chorus_on=1., distortion_on=1., effect_chain_order=float(effect_rank(order)))
        assert effect_order(int(settings["effect_chain_order"])) == order
        target = codec._target(settings)
        assert json.loads(codec.decode(target))["settings"]["effect_chain_order"] == settings["effect_chain_order"]
        targets.append(codec.supervision_target(target))
    assert targets[0] == targets[1] and targets[1] != targets[2]


@pytest.mark.parametrize("name", ["effect_chain_order", *QUANTIZE])
def test_raw_fractional_orders_and_switch_masks_are_rejected(codec, name):
    document = json.loads(codec.decode(codec.default_target()))
    document["settings"][name] = .5
    with pytest.raises(ValueError, match="scalar state"):
        codec.encode(canonical_json(document).encode())


@pytest.mark.parametrize("family", ["lfos", "custom_warps", "random_values", "modulations", "sample", "wavetables"])
def test_fixed_nested_state_is_rejected_instead_of_projected_away(codec, family):
    document = json.loads(codec.decode(codec.default_target()))
    value = document["settings"][family]
    if family in {"lfos", "custom_warps"}:
        value[0]["powers"][0] += .1
    elif family == "random_values":
        value[0]["seed"] += 1
    elif family == "modulations":
        value[63] = {"source": "lfo_1", "destination": "modulation_1_amount"}
    elif family == "sample":
        value["sample_rate"] += 1
    else:
        value[0]["groups"][0]["components"][0]["keyframes"][0]["position"] += 1
    assert codec.schema.validate(document).valid
    with pytest.raises(ValueError, match="no silent projection"):
        codec.encode(canonical_json(document).encode())


def test_modern_pool_split_recipes_and_saved_contract_survive_reload(pool, tmp_path):
    path = tmp_path / "pool.json"
    pool.save(path)
    restored = PilotPool.load(path)
    assert restored.pool_id == pool.pool_id
    assert restored.codec.contract.to_dict()["version"] == 2
    assert restored.codec.contract.renderer_id == FULL_PRESET_RENDERER_ID
    assert load_codec(restored.codec.to_dict()).codec_id == pool.codec.codec_id
    assert PilotCodec().contract.to_dict()["version"] == 1
    sampler = PilotSampler(restored)
    assert sampler.recipe_at(0).render_request().renderer_id == FULL_PRESET_RENDERER_ID
    heldout = PilotSampler(restored, split="validation").recipe_at(0)
    request = heldout.render_request()
    with pytest.raises(ValueError, match="held-out"):
        RenderRecipe.create(request.preset_json.encode(), request.performance, schema=pool.codec.schema, catalog=pool.codec.catalog,
                            contract=pool.codec.contract, ledger=SplitLedger(pool.manifest), purpose="train", sequence_id=0)
    descriptor = deepcopy(pool.codec.to_dict())
    descriptor["continuous"][0]["mapping"] = "changed"
    with pytest.raises(ValueError, match="changed codec"):
        load_codec(descriptor)


def test_projected_pool_preserves_root_splits_and_rejects_reparenting(pool, tmp_path):
    assert isinstance(pool.manifest, FrozenVariantManifest)
    roots = pool.manifest.parent.by_state
    assert all(pool.manifest.by_state[state].split == roots[parent].split for state, parent in pool.manifest.variants)
    assert SplitManifest.from_dict(pool.manifest.to_dict()).manifest_id == pool.manifest.manifest_id
    assert pool.expand(40).expansion_from(pool)["preserved_states"] == 32
    with pytest.raises(ValueError, match="every parent"):
        FrozenVariantManifest(pool.manifest.parent, pool.manifest.variants[:-1])
    pairs = list(pool.manifest.variants)
    pairs[0] = (pairs[1][0], pairs[0][1])
    with pytest.raises(ValueError, match="unique"):
        FrozenVariantManifest(pool.manifest.parent, tuple(pairs))
    value = pool.to_dict()
    variants = value["splits"]["variants"]
    variants[0]["parent_state_id"], variants[1]["parent_state_id"] = variants[1]["parent_state_id"], variants[0]["parent_state_id"]
    from obruxo_data.hashing import canonical_sha256
    path = tmp_path / "reparented.json"
    path.write_text(json.dumps({"pool": value, "pool_id": canonical_sha256(value)}))
    with pytest.raises(ValueError, match="split manifest"):
        PilotPool.load(path)


def test_losses_balance_families_instead_of_counting_route_slots():
    errors = torch.tensor([[1.] * 64 + [0., 5.]])
    masks = torch.tensor([[1.] * 65 + [0.]])
    assert _family_mean(errors, masks, ("routes",) * 64 + ("oscillator", "inactive")) == .5


def test_phase_loss_wraps_and_inactive_values_are_preserved(codec):
    defaults = codec.default_target()
    phase = next(index for index, (name, *_) in enumerate(codec.continuous) if name == "osc_1_phase")
    values = list(defaults.continuous)
    predicted = torch.tensor([values])
    predicted[0, phase] = .01
    logits = [torch.nn.functional.one_hot(torch.tensor([index]), len(options)).float() * 20
              for (_, options), index in zip(codec.categorical, defaults.categorical, strict=True)]
    values[phase] = .99
    wrapped = loss_terms({"continuous": predicted, "categorical": logits}, [PilotTarget(tuple(values), defaults.categorical)], codec)
    values[phase], predicted[0, phase] = .49, .51
    adjacent = loss_terms({"continuous": predicted, "categorical": logits}, [PilotTarget(tuple(values), defaults.categorical)], codec)
    assert float(wrapped["continuous_mse"]) == pytest.approx(float(adjacent["continuous_mse"]), rel=1e-5)
    document = json.loads(codec.decode(defaults))
    document["settings"]["modulation_64_amount"] = .4
    target = codec.encode(canonical_json(document).encode())
    index = next(index for index, (name, *_) in enumerate(codec.continuous) if name == "modulation_64_amount")
    assert not codec.continuous_mask(target)[index]
    assert json.loads(codec.decode(target))["settings"]["modulation_64_amount"] == pytest.approx(.4)


def test_model_outputs_start_from_template_and_train_with_finite_masked_gradients(codec):
    torch.set_num_threads(1)
    torch.manual_seed(3)
    model = PilotInverse(codec)
    spec = model_spec(codec)
    assert spec["model"].startswith("full_scalar") and sum(parameter.numel() for parameter in model.parameters()) < 1_000_000
    from obruxo_training.features import PITCH_SIZE
    features = {"timbre": torch.randn(2, 2, 128, 128), "pitch": torch.randn(2, PITCH_SIZE)}
    defaults = codec.default_target()
    targets = [codec.sample_target(random.Random(index), index) for index in (0, 4)]
    initial = model(features)
    assert predicted_targets(initial, codec)[0].categorical == defaults.categorical
    terms = loss_terms(initial, targets, codec)
    terms["loss"].backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    inactive = next(index for index, (name, *_) in enumerate(codec.continuous) if name == "modulation_64_amount")
    assert model.continuous.weight.grad[inactive].count_nonzero() == 0
    assert model.continuous.weight.grad.count_nonzero() > 0
    for target in predicted_targets(initial, codec):
        assert codec.schema.loads(codec.decode(target))[1].valid


def test_codec_qualification_is_five_train_only_families_and_checks_every_scalar(pool):
    sampler = CodecCheckSampler(pool)
    assert len(sampler.cases) == 5 and len(sampler.recipes) == 15
    assert {case["family"] for case in sampler.cases} == set(pool.codec.to_dict()["families"])
    assert all(case["controls_equal_at_float32"] for case in sampler.cases)
    assert all(recipe.split == "train" for recipe in sampler.recipes)
    for recipe in sampler.recipes:
        settings = json.loads(recipe.render_request().preset_json)["settings"]
        assert all(settings[f"osc_{index}_random_phase"] == 0 for index in range(1, 4))

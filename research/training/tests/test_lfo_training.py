from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from obruxo_data.hashing import canonical_json
from obruxo_training.codec import PilotTarget, load_codec
from obruxo_training.codec_check import CodecCheckSampler
from obruxo_training.features import PITCH_SIZE
from obruxo_training.full_scalar import FullScalarCodec
from obruxo_training.lfo_bank import fit_lfo_bank
from obruxo_training.lfo_codec import LfoCurveCodec
from obruxo_training.lfo_loss import _shift, decoded_curves, reconstruction_loss
from obruxo_training.model import PilotInverse, loss_terms, model_spec, predicted_targets
from obruxo_training.pilot import PilotPool, PilotSampler
from obruxo_training.identity import PresetIdentity
from obruxo_training.reproducibility import FULL_PHASE_CONTROL, fixed_generator_phases
from test_factory_assets import bank  # noqa: F401
from test_lfo_bank import population  # noqa: F401


@pytest.fixture(scope="module")
def codec(bank, population):  # noqa: F811 - shared pytest fixtures
    manifest, train, _ = population
    return LfoCurveCodec(bank, fit_lfo_bank(train, manifest, maximum_curves=48).to_dict())


@pytest.fixture(scope="module")
def pool(codec):
    return PilotPool.create(32, codec=codec)


def original_curves(codec, pool, targets):
    indices = {entry.target: index for index, entry in enumerate(pool.entries)}
    return torch.tensor(np.stack([codec.source_curves(codec.source_preset(target, indices[target])) for target in targets]))


def test_eight_curve_outputs_reload_with_exact_saved_targets_and_unchanged_roots(codec, pool, tmp_path):
    assert (len(codec.continuous), len(codec.categorical)) == (688, 546)
    assert model_spec(codec)["model"].startswith("lfo_curve_")
    original = PilotPool.create(32, codec=FullScalarCodec())
    assert pool.manifest.parent.to_dict() == original.manifest.parent.to_dict()
    path = tmp_path / "pool.json"
    pool.save(path)
    restored = PilotPool.load(path)
    assert restored.pool_id == pool.pool_id and load_codec(codec.to_dict()).codec_id == codec.codec_id
    for entry in pool.entries[:2]:
        assert codec.decode(entry.target) == restored.codec.decode(entry.target)
        assert codec.canonical_target(entry.target) == entry.target
    assert {codec.family(entry.target) for entry in pool.entries if pool.manifest.by_state[entry.state_id].split == "train"} == set(codec.to_dict()["families"])
    with pytest.raises(ValueError, match="frozen split population"):
        PilotPool.create(1, seed=1, codec=codec)


def test_training_renders_original_curves_and_refuses_a_mismatched_code_label(codec, pool):
    sampler = PilotSampler(pool, policy="state_round_robin")
    recipe = sampler.recipe_at(0)
    index, entry = next((index, entry) for index, entry in enumerate(pool.entries) if entry.state_id == recipe.state_id)
    original = codec.source_preset(entry.target, index)
    decoded = codec.decode(entry.target)
    assert recipe.render_request().preset_json == original.decode()
    assert PresetIdentity.from_bytes(original).state == entry.state_id
    assert all(shape["num_points"] <= 8 for shape in json.loads(original)["settings"]["lfos"])
    assert all(shape["num_points"] == 97 for shape in json.loads(decoded)["settings"]["lfos"])
    assert codec.encode(original) == entry.target
    phases = list(entry.target.continuous)
    phases[codec._ncont] = (phases[codec._ncont] + .25) % 1
    with pytest.raises(ValueError, match="does not match its original"):
        codec.source_preset(PilotTarget(tuple(phases), entry.target.categorical), index)


def test_curve_projection_is_explicit_and_other_nested_changes_are_refused(codec, pool):
    target = pool.entries[0].target
    document = json.loads(codec.decode(target))
    assert all(shape["num_points"] == 97 for shape in document["settings"]["lfos"])
    projected = codec.encode(canonical_json(document).encode())
    assert codec._base_target(projected) == codec._base_target(target)
    assert codec.to_dict()["curve_projection"].endswith("not_a_lossless_roundtrip")
    for field in ("custom_warps", "random_values", "modulations"):
        changed = deepcopy(document)
        if field == "custom_warps":
            changed["settings"][field][0]["powers"][0] += .25
        elif field == "random_values":
            changed["settings"][field][0]["seed"] += 1
        else:
            changed["settings"][field][:2] = list(reversed(changed["settings"][field][:2]))
        with pytest.raises(ValueError):
            codec.encode(canonical_json(changed).encode())


def test_phase_and_gain_projection_is_idempotent_and_inactive_codes_survive(codec, pool):
    target = pool.entries[0].target
    extras = tuple([0., 1., 1e-8, .5, .99999][index % 5] for index in range(264))
    projected = codec.canonical_target(PilotTarget(target.continuous[:codec._ncont] + extras, target.categorical))
    assert codec.canonical_target(projected) == projected
    assert all(0 <= phase < 1 for code in codec.codes(projected) for phase in code.phases)
    assert all(-2 <= gain <= 2 for code in codec.codes(projected) for gain in code.gains)
    document = json.loads(codec.decode(projected))
    assert codec.schema.validate(document).valid


def test_torch_hard_decode_matches_the_existing_saved_code_decoder(codec, pool):
    torch.set_num_threads(1)
    targets = [entry.target for entry in pool.entries[:3]]
    categories = torch.tensor([target.categorical for target in targets])
    prediction = {"continuous": torch.tensor([target.continuous for target in targets]),
                  "categorical": [torch.nn.functional.one_hot(categories[:, index], len(options)).float() * 80
                                  for index, (_, options) in enumerate(codec.categorical)]}
    np.testing.assert_allclose(decoded_curves(prediction, codec).numpy(), np.stack([codec.curves(target) for target in targets]), atol=2e-5, rtol=0)


def test_periodic_interpolation_handles_float32_lattice_boundaries_and_tiny_phases():
    from lfo_era2.curve import circular_shift
    values = torch.linspace(0, 1, 97).repeat(101, 1)
    lattice = torch.arange(97, dtype=torch.float32) / 97
    phases = torch.cat((torch.nextafter(lattice, torch.ones_like(lattice)), torch.tensor([0., 1e-12, 1e-9, 1.]))).requires_grad_()
    result = _shift(values, phases)
    assert result.min() >= 0 and result.max() <= 1
    np.testing.assert_allclose(result.detach().numpy(), circular_shift(values.numpy(), phases.detach().numpy()), atol=2e-5, rtol=0)
    result[:, 0].sum().backward()
    assert torch.isfinite(phases.grad).all()


def test_reconstruction_loss_trains_curve_choices_phases_and_gains_only_when_audible(codec, pool):
    torch.set_num_threads(1)
    torch.manual_seed(4)
    targets = [pool.entries[0].target, pool.entries[1].target]
    prediction = {"continuous": torch.rand(2, len(codec.continuous), requires_grad=True),
                  "categorical": [torch.randn(2, len(options), requires_grad=True) for _, options in codec.categorical]}
    loss = reconstruction_loss(prediction, targets, codec, original_curves(codec, pool, targets))
    loss.backward()
    assert torch.isfinite(loss)
    active = codec.active_lfos(targets[0])
    for slot, enabled in enumerate(active):
        head = prediction["categorical"][codec._ncat + slot * 17]
        phases = prediction["continuous"].grad[0, codec._ncont + slot * 17:codec._ncont + (slot + 1) * 17]
        gains = prediction["continuous"].grad[0, codec._ncont + 136 + slot * 16:codec._ncont + 136 + (slot + 1) * 16]
        if enabled:
            assert head.grad[0].count_nonzero() and phases.count_nonzero() and gains.count_nonzero()
        else:
            assert not head.grad[0].count_nonzero() and not phases.count_nonzero() and not gains.count_nonzero()


def test_balanced_training_preserves_audio_only_inputs_and_masks_unused_curve_outputs(codec, pool):
    torch.set_num_threads(1)
    model = PilotInverse(codec)
    target = pool.entries[0].target
    features = {"timbre": torch.randn(1, 2, 96, 32), "pitch": torch.randn(1, PITCH_SIZE)}
    prediction = model(features)
    with pytest.raises(ValueError, match="original LFO curve targets"):
        loss_terms(prediction, [target], codec)
    terms = loss_terms(prediction, [target], codec, curve_targets=original_curves(codec, pool, [target]))
    terms["loss"].backward()
    assert terms["curve_reconstruction_mse"] >= 0
    for slot, enabled in enumerate(codec.active_lfos(target)):
        gradient = model.categorical[codec._ncat + slot * 17].weight.grad
        assert bool(gradient.count_nonzero()) == enabled
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    assert codec.schema.loads(codec.decode(predicted_targets(prediction, codec)[0]))[1].valid


def test_curve_native_qualification_restores_codes_without_reencoding_curves(codec, pool, monkeypatch):
    def refuse_reencoding(*args):
        pytest.fail("serialized-code correctness must not pass through a lossy encoder")
    monkeypatch.setattr(codec, "encode", refuse_reencoding)
    sampler = CodecCheckSampler(pool)
    assert len(sampler.cases) == 8 and len(sampler.recipes) == 24
    assert all(case["serialized_target_and_decoder_exact"] for case in sampler.cases)
    for recipe in sampler.recipes:
        settings = json.loads(recipe.render_request().preset_json)["settings"]
        assert all(settings[key] == value for key, value in FULL_PHASE_CONTROL.items())


def test_full_phase_control_fixes_sampler_without_changing_curves(codec, pool):
    document = json.loads(codec.decode(pool.entries[0].target))
    document["settings"]["sample_random_phase"] = 1.
    before = deepcopy(document["settings"]["lfos"])
    fixed = json.loads(fixed_generator_phases(canonical_json(document).encode()))
    assert fixed["settings"]["lfos"] == before
    assert all(fixed["settings"][key] == value for key, value in FULL_PHASE_CONTROL.items())


def test_curve_checkpoint_recovers_exactly_and_evaluation_repeats_literal_source(codec, tmp_path):
    from obruxo_training.evaluate import evaluate_pilot
    from obruxo_training.stream import StreamConfig
    from obruxo_training.trainer import TrainConfig, replay_bytes, train_pilot, validate_replay
    from obruxo_training.model import batch_features
    from test_stream import FakeRendererFactory
    pool = PilotPool.create(12, codec=codec)
    config = TrainConfig(steps=4, replay_capacity=4, batch_size=2, render_budget=4, cpu_threads=1,
                         checkpoint_every=1, sampling_policy="state_round_robin")
    stream, factory = StreamConfig(prefetch_clips=1), FakeRendererFactory()

    def train(name, **kwargs):
        return train_pilot(pool, tmp_path / name, config=config, stream_config=stream, renderer_factory=factory, **kwargs)

    complete = train("complete")
    collected = train("collected", stop_after=0)
    assert collected["step"] == 0 and collected["totals"]["accepted"] == 4
    seeded = train("seeded", seed_replay=tmp_path / "collected/checkpoint.pt")
    assert seeded["model_state_sha256"] == complete["model_state_sha256"]
    assert seeded["seeded_feature_examples"] == 4 and seeded["totals"]["render_attempts"] == 0
    train("partial", stop_after=2)
    resumed = train("resumed", resume=tmp_path / "partial/checkpoint.pt")
    assert complete["model_state_sha256"] == resumed["model_state_sha256"] and complete["cursor"] == resumed["cursor"]
    left = torch.load(tmp_path / "complete/checkpoint.pt", weights_only=True)
    right = torch.load(tmp_path / "resumed/checkpoint.pt", weights_only=True)
    assert torch.equal(left["batch_rng"], right["batch_rng"]) and torch.equal(left["rng"]["torch"], right["rng"]["torch"])
    for index, state in left["optimizer"]["state"].items():
        assert all(torch.equal(value, right["optimizer"]["state"][index][name]) for name, value in state.items())
    assert set(batch_features(left["replay"])) == {"timbre", "pitch"}
    label_bytes = sum(item["curve_targets"].numel() * item["curve_targets"].element_size() for item in left["replay"])
    assert replay_bytes(left["replay"]) == label_bytes + replay_bytes([{key: value for key, value in item.items() if key != "curve_targets"} for item in left["replay"]])
    corrupted = deepcopy(left["replay"])
    corrupted[0]["curve_targets"][0, 0] += .001
    with pytest.raises(ValueError, match="original LFO curve targets differ"):
        validate_replay(corrupted, PilotSampler(pool, policy="state_round_robin"), left["cursor"], config)
    result = evaluate_pilot(pool, tmp_path / "complete/checkpoint.pt", tmp_path / "evaluation", count=1, renderer_factory=factory)
    events = [json.loads(line) for line in (tmp_path / "evaluation/renders.jsonl").read_text().splitlines()]
    assert result["renders_attempted"] == result["renders_accepted"] == 12
    assert result["aggregates"]["codec_reconstruction"]["reference"]["scored"] == 1
    assert "mrstft" in result["cases"][0]["prediction_to_codec"]["model"]["reference"]
    assert not torch.equal(left["replay"][0]["curve_targets"], torch.tensor(codec.curves(PilotTarget.from_dict(left["replay"][0]["target"]))))
    for role in ("reference", "transfer"):
        requests = {event["method"]: event["request_id"] for event in events if event["role"] == role}
        assert requests["source"] == requests["source_repeat"]

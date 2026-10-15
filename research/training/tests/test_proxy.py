from dataclasses import replace
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from obruxo_training.codec import PilotCodec, PilotTarget  # noqa: E402
from obruxo_training.features import tensor_digest  # noqa: E402
from obruxo_training.performances import initial_bank  # noqa: E402
from obruxo_training.pilot import PilotPool  # noqa: E402
from obruxo_training.proxy import (FEATURE_SIZE, SoundFeatures, SoundProxy, feature_distance, load_proxy,  # noqa: E402
                                   patch_inputs, performance_inputs, refine_continuous)
from obruxo_training.proxy_screen import (ProxyConfig, balanced_entries, fit_proxy, perturbations, proxy_plan,  # noqa: E402
                                         run_proxy_screen, summarize, validate_records)
from test_stream import FakeRendererFactory  # noqa: E402


@pytest.fixture(scope="module")
def pool():
    return PilotPool.create(40, seed=0)


def test_fixed_features_preserve_gain_stereo_and_absolute_time():
    torch.set_num_threads(2)
    extractor = SoundFeatures()
    wave = (.1 * np.sin(np.arange(44100) * .037)).astype(np.float32)
    audio = np.stack((wave, .5 * wave), 1)
    features, quiet, swapped = [extractor(value) for value in (audio, audio * .5, audio[:, ::-1])]
    assert features.shape == (FEATURE_SIZE,) and torch.isfinite(features).all()
    assert not torch.equal(features, quiet)
    assert torch.equal(features[:384].reshape(2, 24, 8)[0], swapped[:384].reshape(2, 24, 8)[1])
    late = extractor(np.concatenate((np.zeros_like(audio), audio)))
    assert not torch.equal(features[-96:], late[-96:])
    with pytest.raises(ValueError, match="float32"):
        extractor(audio.astype(np.float64))


def test_patch_encoding_masks_inactive_values_and_keeps_active_gradients():
    codec = PilotCodec()
    continuous = torch.full((1, 9), .5, requires_grad=True)
    encoded = patch_inputs(continuous, torch.tensor([[0, 0]]), codec)
    encoded.sum().backward()
    assert torch.equal(continuous.grad[0, 2:5], torch.zeros(3))
    assert torch.equal(continuous.grad[0, [0, 1, 5, 6, 7, 8]], torch.ones(6))
    changed = continuous.detach().clone()
    changed[0, 2:5] = .9
    assert torch.equal(encoded, patch_inputs(changed, torch.tensor([[0, 0]]), codec))
    with pytest.raises(ValueError):
        patch_inputs(continuous, torch.tensor([[9, 0]]), codec)


def test_performance_conditioning_uses_actual_notes_not_ids():
    bank = initial_bank()
    played = performance_inputs(bank[0].performance())
    assert played.shape == (83,) and played[0] == 60 / 127
    low = performance_inputs(bank[4].performance())
    assert low[0] == 36 / 127 and not torch.equal(played, low)
    assert not torch.equal(played, performance_inputs(bank[0].performance(), tail_seconds=3))


def toy_records(pool):
    entries = balanced_entries(pool, "train", 8)
    played = initial_bank()[0]
    return [{"sequence": index, "state_id": entry.state_id, "performance_id": played.performance_id,
             "target": entry.target.to_dict(), "features": torch.full((FEATURE_SIZE,), .1 + .5 * entry.target.continuous[0])}
            for index, entry in enumerate(entries)]


def test_proxy_training_recovers_exact_weights_optimizer_rng_and_fits_sound(pool):
    records = toy_records(pool)
    config = ProxyConfig(training_states=8, validation_states=1, steps=80, checkpoint_every=20, batch_size=8, cpu_threads=2)
    _, complete = fit_proxy(records, pool.codec, config)
    recovered = {}
    def checkpoint(value):
        import copy
        recovered.clear()
        recovered.update(copy.deepcopy(value))
    def interrupt(step):
        if step == 21:
            raise InterruptedError("after committed step 20")
    with pytest.raises(InterruptedError):
        fit_proxy(records, pool.codec, config, checkpoint=checkpoint, after_step=interrupt)
    resumed, saved = fit_proxy(records, pool.codec, config, saved=recovered)
    assert saved["model_sha256"] == complete["model_sha256"]
    assert torch.equal(saved["batch_rng"], complete["batch_rng"])
    assert saved["trace"] == complete["trace"]
    assert saved["trace"][-1]["training_error"] < .1 * saved["trace"][0]["training_error"]
    for index, values in saved["optimizer"]["state"].items():
        for name, value in values.items():
            assert torch.equal(value, complete["optimizer"]["state"][index][name])
    target = PilotTarget.from_dict(records[0]["target"])
    before = tensor_digest(resumed.state_dict())
    result, update = refine_continuous(resumed, target, performance_inputs(initial_bank()[0].performance()), records[0]["features"])
    assert update["maximum_continuous_change"] <= .150001
    assert result.categorical == target.categorical and before == tensor_digest(resumed.state_dict())
    assert all(parameter.grad is None for parameter in resumed.parameters())
    resumed.train()
    with pytest.raises(ValueError, match="freeze"):
        refine_continuous(resumed, target, performance_inputs(initial_bank()[0].performance()), records[0]["features"])


def test_proxy_protocol_excludes_test_and_reserved_training_performances(pool):
    config = ProxyConfig(training_states=8, validation_states=2, steps=1)
    plan, recipes, validation = proxy_plan(pool, config)
    assert len(recipes) == 32
    assert {recipe.performance_role for recipe in recipes} == {"reference"}
    assert {recipe.split for recipe in recipes} == {"train"}
    assert all(pool.manifest.by_state[entry.state_id].split == "validation" for entry in validation)
    assert not plan["final_test_used"]
    with pytest.raises(ValueError, match="reserved"):
        balanced_entries(pool, "test", 1)
    target = balanced_entries(pool, "train", 1)[0].target
    variants, _ = perturbations(target, pool.codec, 0)
    assert set(variants) == {"small", "large", "category"}
    assert variants["small"].categorical == target.categorical
    assert variants["category"].categorical != target.categorical
    for value in variants.values():
        recovered = pool.codec.encode(pool.codec.decode(value))
        assert recovered.categorical == value.categorical
        assert recovered.continuous == pytest.approx(value.continuous)


def test_qualification_requires_actual_native_improvement_and_complete_panel(pool):
    plan, _, _ = proxy_plan(pool, ProxyConfig(training_states=8, validation_states=1, steps=1))
    errors = {"proxy": .1, **{name: 1. for name in plan["baselines"]}}
    candidates = {name: {"mrstft": value, "proxy_error": value} for name, value in (("small", .2), ("large", .8), ("category", .6), ("guided", .4))}
    rows = [{"role": role, "accepted_renders": 6, "forward_error": errors, "repeat_mrstft": .01, "candidates": candidates}
            for role in ("reference", "transfer")]
    assert summarize(rows, plan)["qualified_for_inverse_training"]
    rows[0]["candidates"] = {**candidates, "guided": {"mrstft": .9, "proxy_error": .01}}
    assert not summarize(rows, plan)["qualified_for_inverse_training"]
    assert not summarize(rows[:1], plan)["qualified_for_inverse_training"]


def test_native_collection_resume_rejects_tampering_and_never_uses_failed_audio(pool, tmp_path):
    config = ProxyConfig(training_states=8, validation_states=1, steps=2, checkpoint_every=1, cpu_threads=2)
    output = tmp_path / "screen"
    def interrupt(event):
        if event["stage"] == "collect" and event["attempted"] == 3:
            raise InterruptedError("after three durable native outcomes")
    with pytest.raises(InterruptedError):
        run_proxy_screen(pool, output, config=config, renderer_factory=FakeRendererFactory(), progress=interrupt)
    saved = torch.load(output / "checkpoint.pt", weights_only=True)
    assert saved["next_train"] == len(saved["records"]) == 3
    bad = saved["records"][0].copy()
    bad["performance_id"] = initial_bank()[4].performance_id
    _, recipes, _ = proxy_plan(pool, config)
    with pytest.raises(ValueError, match="recipe"):
        validate_records([bad], recipes, pool, 3)
    with pytest.raises(ValueError, match="changed"):
        run_proxy_screen(pool, output, config=replace(config, seed=1), resume=True)
    result = run_proxy_screen(pool, output, config=config, renderer_factory=FakeRendererFactory(), resume=True)
    assert result["render_attempts"] == 44 and result["renders_accepted"] == 44
    assert result["training_records"] == 32 and not result["qualified_for_inverse_training"]
    assert not list(output.glob("*.wav"))
    with pytest.raises(ValueError, match="qualification"):
        load_proxy(output / "proxy.pt")
    loaded = load_proxy(output / "proxy.pt", require_qualified=False)
    assert tensor_digest(loaded.state_dict()) == result["model_sha256"]
    saved = torch.load(output / "checkpoint.pt", weights_only=True)
    saved["records"][0]["features"][0] += 1
    torch.save(saved, output / "checkpoint.pt")
    with pytest.raises(ValueError, match="changed"):
        run_proxy_screen(pool, output, config=config, renderer_factory=FakeRendererFactory(), resume=True)
    assert json.loads((output / "summary.json").read_text())["final_test_used"] is False


def test_clipped_source_outcomes_remain_recorded_but_never_become_targets(pool, tmp_path):
    def stop(event):
        if event["stage"] == "collect" and event["attempted"] == 1:
            raise InterruptedError
    with pytest.raises(InterruptedError):
        run_proxy_screen(pool, tmp_path / "clipped", config=ProxyConfig(training_states=8, validation_states=1, steps=1),
                         renderer_factory=FakeRendererFactory("clipped"), progress=stop)
    saved = torch.load(tmp_path / "clipped/checkpoint.pt", weights_only=True)
    assert saved["next_train"] == 1 and saved["records"] == []
    history = json.loads((tmp_path / "clipped/renders.jsonl").read_text().strip())
    assert history["status"] == "rejected" and "audio.clipping" in history["diagnostic_codes"]


def test_proxy_rejects_bad_configuration_and_reserved_training_features(pool):
    with pytest.raises(ValueError):
        ProxyConfig(learning_rate=float("nan"))
    with pytest.raises(ValueError):
        ProxyConfig(training_states=1)
    records = toy_records(pool)
    records[0]["performance_id"] = initial_bank()[4].performance_id
    with pytest.raises(ValueError, match="reserved"):
        fit_proxy(records, pool.codec, ProxyConfig(training_states=8, steps=1))
    codec = PilotCodec()
    model = SoundProxy(codec).freeze()
    values = torch.full((1, 9), .5, requires_grad=True)
    features = model.predict_patch(values, torch.tensor([[0, 0]]), performance_inputs(initial_bank()[0].performance())[None])
    feature_distance(features, torch.ones_like(features), model.scale).mean().backward()
    assert torch.isfinite(values.grad).all() and values.grad[0, 0] != 0
    assert all(parameter.grad is None for parameter in model.parameters())

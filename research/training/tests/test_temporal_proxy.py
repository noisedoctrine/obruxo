import copy
from dataclasses import replace
import json
from time import time

import pytest

torch = pytest.importorskip("torch")

from obruxo_training.audit import write_json  # noqa: E402
from obruxo_training.codec import PilotCodec, PilotTarget  # noqa: E402
from obruxo_training.features import tensor_digest  # noqa: E402
from obruxo_training.performances import initial_bank  # noqa: E402
from obruxo_training.pilot import PilotPool  # noqa: E402
from obruxo_training.proxy import performance_inputs  # noqa: E402
from obruxo_training.spectral import SpectralTargets, spectral_loss  # noqa: E402
from obruxo_training.spectral_controls import BasicControlAdapter  # noqa: E402
from obruxo_training.temporal_proxy import (TemporalSpectralProxy, collate_frames, guide_patch,  # noqa: E402
                                          sample_frames, temporal_proxy_spec)
from obruxo_training.temporal_proxy_data import (NativeSession, RecordCache, TemporalProxyConfig, collect_frames,  # noqa: E402
                                               make_plan)
from obruxo_training.temporal_proxy_screen import (fit_temporal_proxy, load_temporal_proxy, qualification_summary)  # noqa: E402
from test_stream import FakeRendererFactory  # noqa: E402


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)


@pytest.fixture(scope="module")
def pool():
    return PilotPool.create(80, seed=0)


def records():
    time_axis = torch.arange(32768) / 44100
    mono = .1 * torch.sin(2 * torch.pi * 220 * time_axis)
    wave = torch.stack((mono, mono * .7))[None]
    target = SpectralTargets()(wave)
    values, times = sample_frames(target, torch.Generator().manual_seed(0))
    entry = PilotCodec().canonical_target(PilotTarget((.4,) * 9, (1, 1)))
    record = {"measurements": values, "times": times, "target": entry.to_dict(),
              "performance": performance_inputs(initial_bank()[0].performance())}
    return [record, copy.deepcopy(record)]


def test_frame_samples_preserve_exact_frequency_values_and_actual_times():
    source = torch.randn(1, 2, 44100) * .01
    full = SpectralTargets()(source)
    sampled, times = sample_frames(full, torch.Generator().manual_seed(3))
    from obruxo_training.spectral import FFT_SIZES, SAMPLE_RATE
    for i, size in enumerate(FFT_SIZES):
        indices = (times[i][0] * SAMPLE_RATE / (size // 4)).round().long()
        assert torch.equal(sampled["magnitudes"][i], full["magnitudes"][i][..., indices])
        assert sampled["magnitudes"][i].shape[-1] == 16
        assert (times[i][0, 1:] > times[i][0, :-1]).all()
    again, repeated_times = sample_frames(full, torch.Generator().manual_seed(3))
    assert all(torch.equal(a, b) for a, b in zip(times, repeated_times, strict=True))
    assert torch.equal(again["envelope"], sampled["envelope"])


def test_helper_is_small_temporal_and_preserves_control_gradients():
    torch.manual_seed(2)
    helper, adapter = TemporalSpectralProxy().freeze(), BasicControlAdapter()
    assert sum(p.numel() for p in helper.parameters()) < 2_000_000
    rows = records()
    target, times = collate_frames(rows)
    values = torch.full((2, 9), .5, requires_grad=True)
    logits = [torch.tensor([[0., 3., 2.]] * 2, requires_grad=True), torch.tensor([[0., 3.]] * 2, requires_grad=True)]
    played = torch.stack([row["performance"] for row in rows])
    before = tensor_digest(helper.state_dict())
    prediction = helper(adapter(values, logits), played, times)
    loss = spectral_loss(prediction, target)["loss"].mean()
    loss.backward()
    assert torch.isfinite(values.grad).all() and (values.grad.abs().sum(0) > 0).all()
    assert all(torch.isfinite(head.grad).all() and head.grad.abs().sum() > 0 for head in logits)
    assert before == tensor_digest(helper.state_dict()) and all(p.grad is None for p in helper.parameters())
    shifted = helper(adapter(values, logits, sample=False), played, tuple(t + .25 for t in times))
    original = helper(adapter(values, logits, sample=False), played, times)
    assert not torch.equal(shifted["magnitudes"][0], original["magnitudes"][0])


def test_temporal_fit_recovers_optimizer_and_minibatch_rng_exactly():
    rows, adapter = records(), BasicControlAdapter()
    config = TemporalProxyConfig(training_states=6, views=1, validation_states=2, steps=12, batch_size=2, checkpoint_every=3)
    contract = {"test": "same_records", "config": config.__dict__}
    _, full = fit_temporal_proxy(rows, adapter, config, contract)
    saved = {}
    def checkpoint(state):
        saved.clear()
        saved.update(copy.deepcopy(state))
    def interrupt(step):
        if step == 7:
            raise InterruptedError
    with pytest.raises(InterruptedError):
        fit_temporal_proxy(rows, adapter, config, contract, checkpoint=checkpoint, after_step=interrupt)
    _, recovered = fit_temporal_proxy(rows, adapter, config, contract, saved=saved)
    assert full["model_sha256"] == recovered["model_sha256"] and full["trace"] == recovered["trace"]
    assert torch.equal(full["batch_rng"], recovered["batch_rng"])
    for index, state in full["optimizer"]["state"].items():
        assert all(torch.equal(value, recovered["optimizer"]["state"][index][key]) for key, value in state.items())
    assert full["trace"][-1]["loss"] < full["trace"][0]["loss"]
    with pytest.raises(ValueError, match="contract changed"):
        fit_temporal_proxy(rows, adapter, config, {"changed": True}, saved=saved)


def test_plan_preserves_splits_and_uses_new_training_performances(pool):
    config = TemporalProxyConfig(training_states=6, views=8, validation_states=2, steps=1)
    plan, recipes, validation = make_plan(pool, config)
    reserved = {entry.performance_id for entry in initial_bank()}
    assert len(recipes) == 48 and len(validation) == 2
    assert all(pool.manifest.by_state[row["source_state_id"]].split == "train" for row in recipes)
    assert all(pool.manifest.by_state[entry.state_id].split == "validation" for entry in validation)
    assert not reserved.intersection(row["performance_id"] for row in recipes)
    assert not plan["test_split_used"]
    from obruxo_data.midi import Performance
    assert {len(Performance.from_dict(row["performance"]).note_spans()) for row in recipes} == {1, 3, 4}
    for offset in range(0, len(recipes), 8):
        for view in range(4):
            assert recipes[offset + view]["performance_id"] == recipes[offset + view + 4]["performance_id"]
            assert recipes[offset + view]["target"] != recipes[offset + view + 4]["target"]
    with pytest.raises(ValueError):
        replace(config, steps=12001)


def test_collection_recovery_and_changed_features_refused(pool, tmp_path):
    config = TemporalProxyConfig(training_states=6, views=1, validation_states=2, steps=1)
    plan, recipes, _ = make_plan(pool, config)
    write_json(tmp_path / "run.json", {"started_unix": time()})
    class Monitor:
        peak_bytes = 0
    session = NativeSession(tmp_path, plan, pool.codec, Monitor(), FakeRendererFactory())
    def stop(event):
        if event["completed"] == 2:
            raise InterruptedError
    with pytest.raises(InterruptedError):
        collect_frames(pool, tmp_path, recipes, session, stop)
    resumed = NativeSession(tmp_path, plan, pool.codec, Monitor(), FakeRendererFactory())
    rows = collect_frames(pool, tmp_path, recipes, resumed)
    assert len(rows) == 6 and all(row["accepted"] for row in rows)
    assert sum(row["event"] == "attempt" for row in resumed.events) == 6
    cache = RecordCache(tmp_path, rows)
    assert cache[0]["recipe_id"] == rows[0]["recipe_id"]
    path = tmp_path / "records" / rows[1]["file"]
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="record changed"):
        cache[1]


def test_native_guidance_is_required_even_if_helper_prediction_is_good():
    plan = {"config": {"validation_states": 1}, "baselines": ["training_mean", "nearest_training_record", "five_neighbor_interpolation"]}
    cases = []
    for role in ("reference", "transfer"):
        rows = {"source": {"forward_errors": {"helper": .1, **{name: 1. for name in plan["baselines"]}}},
                "repeat": {"native_mrstft": 0.}, "start": {"native_mrstft": 1., "native_spectral": 1., "helper_error": 1.,
                                                           "candidate": {"categorical": [1, 1]}}}
        for method in ("plus", "minus", "category", "pan", "detune", "continuous", "joint"):
            rows[method] = {"native_mrstft": .5, "native_spectral": .5, "helper_error": .5,
                            "guide": {"category_changed": method == "joint"}}
        cases.append({"role": role, "rows": rows})
    assert qualification_summary(cases, plan)["qualified_for_inverse_training"]
    cases[0]["rows"]["joint"]["native_mrstft"] = 2.
    assert not qualification_summary(cases, plan)["qualified_for_inverse_training"]
    json.dumps(qualification_summary(cases, plan), allow_nan=False)
    assert not qualification_summary(cases[:1], plan)["qualified_for_inverse_training"]


def test_unqualified_helper_is_refused_and_guided_update_is_bounded(tmp_path):
    helper, adapter = TemporalSpectralProxy().freeze(), BasicControlAdapter()
    saved = {"format": "basic_temporal_helper_v1", "codec": adapter.codec.to_dict(), "contract": adapter.contract(),
             "spec": temporal_proxy_spec(), "model": helper.state_dict(), "model_sha256": tensor_digest(helper.state_dict()),
             "qualified_for_inverse_training": False}
    path = tmp_path / "helper.pt"
    torch.save(saved, path)
    with pytest.raises(ValueError, match="qualification"):
        load_temporal_proxy(path)
    loaded, restored_adapter = load_temporal_proxy(path, require_qualified=False)
    row = records()[0]
    target = PilotTarget.from_dict(row["target"])
    full = SpectralTargets()(torch.randn(1, 2, 32768) * .01)
    result, trace = guide_patch(loaded, restored_adapter, target, row["performance"], full, joint=True, seed=0, steps=2)
    assert trace["maximum_continuous_change"] <= .150001 and trace["category_gradient_l1"] > 0
    adapter.codec.validate_target(result)
    assert tensor_digest(loaded.state_dict()) == saved["model_sha256"]

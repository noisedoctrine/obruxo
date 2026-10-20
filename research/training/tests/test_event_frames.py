import copy
from dataclasses import replace

import pytest
import torch

from obruxo_data.midi import Performance, TempoMap
from obruxo_training.codec import PilotCodec, PilotTarget
from obruxo_training.event_frames import event_frame_spec, note_event_seconds, sample_event_frames
from obruxo_training.paired_spectral import complete_pairs, paired_change_loss, require_aligned_pairs
from obruxo_training.pilot import PilotPool
from obruxo_training.proxy import performance_inputs
from obruxo_training.spectral import FFT_SIZES, SpectralTargets, spectral_loss
from obruxo_training.spectral_controls import BasicControlAdapter
from obruxo_training.features import tensor_digest
from obruxo_training.temporal_proxy import TemporalSpectralProxy, collate_frames, full_times, temporal_proxy_spec
from obruxo_training.temporal_proxy_data import RecordCache, TemporalProxyConfig, collect_frames, make_plan
from obruxo_training.temporal_proxy_screen import fit_temporal_proxy, load_temporal_proxy


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)


def measured_record(seconds=2):
    performance = Performance(bpm=120)
    performance.add_note(pitch=60, velocity=96, start_tick=120, duration_ticks=480)
    samples = int(seconds * 44100)
    wave = .01 * torch.sin(torch.arange(samples) / 10)
    full = SpectralTargets()(torch.stack((wave, wave))[None])
    values, times = sample_event_frames(full, performance, torch.Generator().manual_seed(2))
    target = PilotCodec().canonical_target(PilotTarget((.4,) * 9, (1, 1)))
    return {"measurements": values, "times": times, "target": target.to_dict(), "performance": performance_inputs(performance)}, full, performance


def test_event_anchors_preserve_exact_frames_and_every_valid_envelope_time():
    record, full, performance = measured_record()
    times = full_times(full)
    for index, size in enumerate(FFT_SIZES):
        sampled_times = record["times"][index][0]
        assert len(sampled_times) == 32 and (sampled_times[1:] > sampled_times[:-1]).all()
        for event in note_event_seconds(performance):
            for offset in event_frame_spec()["event_offsets_seconds"]:
                nearest = times[index][0, (times[index][0] - event - offset).abs().argmin()]
                assert (sampled_times == nearest).any()
        indices = (sampled_times * 44100 / (size // 4)).round().long()
        assert torch.equal(record["measurements"]["magnitudes"][index], full["magnitudes"][index][..., indices])
    assert torch.equal(record["measurements"]["envelope"], full["envelope"][..., full["envelope_mask"][0]])
    changed = copy.deepcopy(full)
    changed["envelope"] *= 1.2
    _, repeated_times = sample_event_frames(changed, performance, torch.Generator().manual_seed(2))
    assert all(torch.equal(a, b) for a, b in zip(record["times"], repeated_times, strict=True))


def test_event_policy_refuses_unsupported_performances_and_is_explicit():
    performance = Performance(bpm=120)
    for index in range(5):
        performance.add_note(pitch=60, velocity=96, start_tick=index * 480, duration_ticks=240)
    with pytest.raises(ValueError, match="eight unique"):
        note_event_seconds(performance)
    with pytest.raises(ValueError, match="paired-change"):
        TemporalProxyConfig(event_frames=True)
    assert temporal_proxy_spec()["training_envelope_frames"] == 16
    assert temporal_proxy_spec(True)["training_envelope_frames"] == "all_valid"
    _, full, _ = measured_record()
    with pytest.raises(ValueError, match="beyond the final"):
        long_note = Performance(bpm=120)
        long_note.add_note(pitch=60, velocity=96, start_tick=0, duration_ticks=4000)
        sample_event_frames(full, long_note, torch.Generator().manual_seed(0))


def test_variable_envelope_padding_does_not_change_losses_or_gradients():
    short, _, _ = measured_record()
    long, _, _ = measured_record(3)
    rows = [short, copy.deepcopy(short), long, copy.deepcopy(long)]
    for row in rows[1::2]:
        row["measurements"]["envelope"] *= 1.3
    target, _ = collate_frames(rows)
    prediction = {key: tuple(value.clone().requires_grad_() for value in target[key]) if key == "magnitudes"
                  else target[key].clone().requires_grad_() for key in ("magnitudes", "envelope")}
    mask = target["envelope_mask"]
    with torch.no_grad():
        prediction["envelope"].add_(100 * (~mask)[:, None])
    assert spectral_loss(prediction, target)["loss"].sum() == 0
    assert paired_change_loss(prediction, target).sum() == 0
    loss = spectral_loss(prediction, target)["loss"].sum() + paired_change_loss(prediction, target).sum()
    loss.backward()
    assert (prediction["envelope"].grad * (~mask)[:, None]).abs().sum() == 0


def test_event_collection_recovers_and_keeps_pairs_aligned(tmp_path):
    pool = PilotPool.create(80, seed=0)
    config = TemporalProxyConfig(training_states=6, steps=1, validation_states=2, paired_changes=True, event_frames=True)
    plan, recipes, _ = make_plan(pool, config)
    legacy, _, _ = make_plan(pool, replace(config, event_frames=False))
    assert plan != legacy and "event_frames.py" in plan["source"]
    assert plan["model"]["frame_sampling"] == event_frame_spec()
    class Session:
        calls = 0

        def render(self, target, performance, phase, label):
            self.calls += 1
            seconds = float(TempoMap.from_performance(performance).tick_to_seconds(performance.end_tick)) + 2
            wave = .01 * torch.sin(torch.arange(int(seconds * 44100)) / 10)
            return torch.stack((wave, wave), dim=1).numpy(), {"status": "accepted"}
    session = Session()
    session.plan = plan
    def interrupt(event):
        if event["completed"] == 3:
            raise InterruptedError
    with pytest.raises(InterruptedError):
        collect_frames(pool, tmp_path, recipes[:8], session, interrupt)
    rows = collect_frames(pool, tmp_path, recipes[:8], session)
    assert session.calls == 8
    cache = RecordCache(tmp_path, rows)
    assert len(complete_pairs(cache)) == 4
    for first, second in complete_pairs(cache):
        require_aligned_pairs([cache[first], cache[second]])
        assert cache[first]["measurements"]["envelope"].shape[-1] > 16


def test_event_batch_optimizer_resume_is_exact():
    class Cache(list):
        pass
    short, _, _ = measured_record()
    long, _, _ = measured_record(3)
    cache = Cache(copy.deepcopy(short if index % 4 < 2 else long) for index in range(48))
    cache.rows = [{"sequence": i} for i in range(48)]
    for index, row in enumerate(cache):
        row.update(pair_id=index // 8 * 4 + index % 4, source_state_id=str(index // 8))
    config = TemporalProxyConfig(training_states=6, steps=6, batch_size=4, checkpoint_every=2, paired_changes=True, event_frames=True)
    adapter, contract = BasicControlAdapter(), {"sampling": event_frame_spec()}
    _, reference = fit_temporal_proxy(cache, adapter, config, contract)
    saved = {}
    def checkpoint(value):
        saved.clear()
        saved.update(copy.deepcopy(value))
    def interrupt(step):
        if step == 3:
            raise InterruptedError
    with pytest.raises(InterruptedError):
        fit_temporal_proxy(cache, adapter, config, contract, checkpoint=checkpoint, after_step=interrupt)
    _, resumed = fit_temporal_proxy(cache, adapter, config, contract, saved=saved)
    assert reference["model_sha256"] == resumed["model_sha256"] and reference["trace"] == resumed["trace"]
    assert torch.equal(reference["batch_rng"], resumed["batch_rng"])
    with pytest.raises(ValueError, match="contract changed"):
        fit_temporal_proxy(cache, adapter, config, {"sampling": "old_uniform"}, saved=saved)


def test_event_artifact_remains_unqualified_and_rejects_changed_policy(tmp_path):
    helper, adapter = TemporalSpectralProxy().freeze(), BasicControlAdapter()
    saved = {"format": "basic_temporal_helper_v1", "codec": adapter.codec.to_dict(), "contract": adapter.contract(),
             "spec": temporal_proxy_spec(True), "model": helper.state_dict(), "model_sha256": tensor_digest(helper.state_dict()),
             "qualified_for_inverse_training": False}
    path = tmp_path / "event-helper.pt"
    torch.save(saved, path)
    with pytest.raises(ValueError, match="qualification"):
        load_temporal_proxy(path)
    restored, _ = load_temporal_proxy(path, require_qualified=False)
    assert tensor_digest(restored.state_dict()) == saved["model_sha256"]
    saved["spec"]["frame_sampling"]["event_offsets_seconds"] = [0., .5]
    torch.save(saved, path)
    with pytest.raises(ValueError, match="incompatible"):
        load_temporal_proxy(path, require_qualified=False)

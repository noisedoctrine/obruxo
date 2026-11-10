from types import SimpleNamespace

import numpy as np
import pytest
import torch

from obruxo_data.midi import Performance
from obruxo_training.batching import BatchConfig, predict_features
from obruxo_training.basic_direct_fit import SoundStore
from obruxo_training.codec import PilotCodec, PilotTarget
from obruxo_training.features import AudioFeatures, PITCH_SIZE, TIMBRE_SHAPE
from obruxo_training.evaluate import audio_distance, audio_distances
from obruxo_training.model import PilotInverse
from obruxo_training.native_inverse import native_reconstruction_loss
from obruxo_training.spectral import SpectralTargets
from obruxo_training.spectral_controls import BasicControlAdapter


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)


def test_batch_sizes_are_designer_selected_and_validated():
    assert BatchConfig(model=71, spectral=35, pitch=1000).to_dict() == {"model": 71, "spectral": 35, "pitch": 1000}
    for value in (0, -1, True, 1.5):
        for name in ("model", "spectral", "pitch"):
            with pytest.raises(ValueError, match="positive integers"):
                BatchConfig(**{name: value})


def test_only_the_pinned_codec_constant_relocation_is_compatible():
    from obruxo_training.batching import compatible_infrastructure_source
    old = "62e915fd0aae9d7e77644771deb4fc216ac2f614f930628f7aaa9fa63e23edb8"
    current = "6b2d4b37b7674d80bc98825d6b4efae2d13a01d4808c175826edafa1c230aaee"
    assert compatible_infrastructure_source("codec.py", old, current)
    assert not compatible_infrastructure_source("codec.py", old, "unreviewed-change")
    assert not compatible_infrastructure_source("codec.py", "unreviewed-source", current)
    assert not compatible_infrastructure_source("another.py", old, current)
    assert not compatible_infrastructure_source("trainer.py", "unknown-source", "unknown-edit")


def test_model_predictions_use_requested_batches_and_match_single_rows():
    torch.manual_seed(29)
    model = PilotInverse().eval()
    features = [{"timbre": torch.rand(TIMBRE_SHAPE), "pitch": torch.rand(PITCH_SIZE)} for _ in range(7)]
    seen = []
    hook = model.register_forward_pre_hook(lambda _module, args: seen.append(len(args[0]["timbre"])))
    with torch.no_grad():
        batched = predict_features(model, features, 3)
        assert seen == [3, 3, 1]
        serial = predict_features(model, features, 1)
    hook.remove()
    for left, right in zip(batched, serial, strict=True):
        torch.testing.assert_close(left["continuous"], right["continuous"], atol=1e-6, rtol=1e-6)
        for a, b in zip(left["categorical"], right["categorical"], strict=True):
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)


def test_model_batching_preserves_weighted_example_gradients():
    torch.manual_seed(71)
    model = PilotInverse().eval()
    features = [{"timbre": torch.rand(TIMBRE_SHAPE), "pitch": torch.rand(PITCH_SIZE)} for _ in range(7)]
    measured = []
    for size in (1, 3, 32):
        model.zero_grad(set_to_none=True)
        rows = predict_features(model, features, size)
        loss = sum((index + 1) * (row["continuous"].square().sum() + sum(head.square().sum() for head in row["categorical"]))
                   for index, row in enumerate(rows))
        loss.backward()
        measured.append((loss.detach(), [parameter.grad.clone() for parameter in model.parameters()]))
    for value, gradients in measured[1:]:
        torch.testing.assert_close(value, measured[0][0], atol=1e-5, rtol=1e-5)
        for left, right in zip(gradients, measured[0][1], strict=True):
            torch.testing.assert_close(left, right, atol=1e-5, rtol=1e-5)


def test_bulk_native_queries_preserve_draws_query_order_loss_and_gradients():
    adapter = BasicControlAdapter()
    initial = torch.full((4, 9), .4)
    records = []
    def evaluate(index, patch):
        return sum(value**2 for value in patch.continuous) + .1 * index + sum(patch.categorical)
    for bulk in (False, True):
        torch.manual_seed(17)
        raw = initial.clone().requires_grad_()
        logits = [torch.zeros(4, size, requires_grad=True) for size in (3, 2)]
        calls, batches = [], []
        def one(index, patch):
            calls.append((index, patch))
            return evaluate(index, patch)
        def many(queries):
            batches.append(len(queries))
            return [one(index, patch) for index, patch in queries]
        loss, estimates = native_reconstruction_loss({"continuous": raw, "categorical": logits}, adapter, one,
                                                     evaluate_many=many if bulk else None)
        loss.backward()
        records.append((loss.detach(), [estimate.to_dict() for estimate in estimates], calls, [raw.grad, *(head.grad for head in logits)]))
        if bulk:
            assert batches == [len(calls)] and len(calls) > 22
    assert torch.equal(records[0][0], records[1][0]) and records[0][1:3] == records[1][1:3]
    assert all(torch.equal(a, b) for a, b in zip(records[0][3], records[1][3], strict=True))


def test_batched_audit_transforms_preserve_scalar_score_reduction_order():
    rng = np.random.default_rng(31)
    reference = (rng.normal(size=(3, 176400, 2)) * .05).astype(np.float32)
    candidate = (rng.normal(size=reference.shape) * .04).astype(np.float32)
    expected = [audio_distance(left, right) for left, right in zip(reference, candidate, strict=True)]
    assert audio_distances(reference, candidate) == expected


def store_fixture(tmp_path, monkeypatch, batch_size, fail_at=None):
    import obruxo_training.basic_direct_fit as module
    performances = []
    references = []
    for index, samples in enumerate((8192, 12288)):
        performance = Performance()
        performance.add_note(pitch=60, velocity=96, start_tick=0, duration_ticks=480 * (index + 1))
        performances.append(performance)
        wave = np.sin(np.arange(samples) * .03).astype(np.float32) * .1
        references.append(np.stack((wave, wave * .6), axis=1))
    inputs = SimpleNamespace(codec=PilotCodec(), bytes=0, plan={"batching": BatchConfig(spectral=batch_size).to_dict(),
        "fit_audit_cells": [{"recipe": {"performances": [performance.to_dict()], "source_audio_sha256": [str(index)]}, "playback_view": 0}
                            for index, performance in enumerate(performances)], "limits": {"replay_bytes": 128 * 1024**2}})
    calls, seen = [], []
    class Native:
        failed = False
        def render(self, candidate, performance, _phase, label):
            calls.append(label)
            if len(calls) == fail_at and not self.failed:
                self.failed = True
                raise RuntimeError("simulated render interruption")
            index = int(performance.end_tick == 960)
            return references[index] * (.7 + candidate.continuous[0] * .1), {"label": label}
    class Measurements(SpectralTargets):
        def forward(self, audio, lengths=None):
            seen.append(len(audio))
            return super().forward(audio, lengths)
    store = SoundStore(tmp_path, inputs, Native())
    store.measurements = Measurements()
    store.check_limits = lambda: None
    store.source = lambda index: (references[index], {"source": index})
    monkeypatch.setattr(module, "require_request", lambda *_args: None)
    requests = [(1, 0, index % 2, PilotTarget((.2 + .05 * index,) * 9, (2, 1))) for index in range(7)]
    return store, requests, calls, seen


def test_spectral_batches_handle_mixed_lengths_caching_and_partial_failure(tmp_path, monkeypatch):
    store, requests, calls, seen = store_fixture(tmp_path, monkeypatch, 6, fail_at=5)
    with pytest.raises(RuntimeError, match="interruption"):
        store.probe_many(requests)
    assert len(store.probes) == 4  # Accepted audio before failure was scored and recorded.
    seen.clear()
    actual = store.probe_many(requests)
    assert len(calls) == 8 and max(seen) > 1 and len(store.probes) == 7
    serial, serial_requests, _, _ = store_fixture(tmp_path / "serial", monkeypatch, 1)
    (tmp_path / "serial").mkdir()
    expected = serial.probe_many(serial_requests)
    torch.testing.assert_close(torch.tensor(actual), torch.tensor(expected), atol=1e-6, rtol=1e-6)
    (tmp_path / "complete").mkdir()
    complete, complete_requests, _, _ = store_fixture(tmp_path / "complete", monkeypatch, 6)
    assert complete.probe_many(complete_requests) == actual
    assert store.probe_many(requests) == actual and len(calls) == 8
    with pytest.raises(ValueError, match="batching changed"):
        store_fixture(tmp_path, monkeypatch, 3)


def test_pitch_windows_are_batched_without_changing_feature_contract():
    frontend = AudioFeatures(pitch_batch_size=1)
    wave = (.1 * np.sin(np.arange(6 * 44100) * .03)).astype(np.float32)
    audio = np.stack((wave, wave * .6), axis=1)
    serial = frontend(audio)
    frontend.pitch_batch_size = 3
    sizes = []
    hook = frontend.basic_pitch.register_forward_pre_hook(lambda _module, args: sizes.append(len(args[0])))
    batched = frontend(audio)
    hook.remove()
    assert max(sizes) == 3
    assert torch.equal(serial["timbre"], batched["timbre"])
    # CPU CQT/convolution kernels use different float32 reduction layouts for window batches.
    torch.testing.assert_close(serial["pitch"], batched["pitch"], atol=1e-4, rtol=0.)

from dataclasses import dataclass, replace
import json
import multiprocessing as mp
from threading import Event, Timer

import numpy as np
import pytest

from obruxo_training.pilot import PilotPool, PilotSampler
from obruxo_training.stream import OnlineRenderStream, RenderWorker, StreamConfig, render_isolated, request_frames
from test_stream import FakeRenderer


@dataclass(frozen=True)
class CountingFactory:
    def __call__(self):
        return CountingRenderer()


class CountingRenderer:
    def __init__(self):
        self.calls = 0

    def render(self, request):
        self.calls += 1
        mode = json.loads(request.preset_json).get("preset_name", "valid")
        result = FakeRenderer(mode).render(request)
        result.qa["instance_calls"] = self.calls
        return result


@pytest.fixture(scope="module")
def pool():
    return PilotPool.create(12, seed=2)


def request_for(pool, mode="valid"):
    request = PilotSampler(pool).recipe_at(0).render_request()
    document = json.loads(request.preset_json)
    return replace(request, preset_json=json.dumps({**document, "preset_name": mode}))


def test_worker_reuses_renderer_and_recycles_at_cap(pool):
    before = {child.pid for child in mp.active_children()}
    request = request_for(pool)
    with RenderWorker(renderer_factory=CountingFactory(), max_jobs=2) as worker:
        rows = [worker.render(request, timeout=15) for _ in range(3)]
    assert rows[0][1]["worker_pid"] == rows[1][1]["worker_pid"] != rows[2][1]["worker_pid"]
    assert [row[1]["worker_job"] for row in rows] == [1, 2, 1]
    assert [row[1]["renderer_qa"]["instance_calls"] for row in rows] == [1, 2, 1]
    assert all(np.array_equal(rows[0][0], audio) and not audio.flags.writeable for audio, _ in rows)
    assert {child.pid for child in mp.active_children()} <= before
    with pytest.raises(RuntimeError, match="closed"):
        worker.render(request)


@pytest.mark.parametrize("mode,error", [("crash", "WorkerExited"), ("hang", "RenderTimeout"), ("dtype", "ValueError")])
def test_worker_failure_is_charged_and_next_call_uses_new_process(pool, mode, error):
    with RenderWorker(renderer_factory=CountingFactory(), max_jobs=8) as worker:
        _, first = worker.render(request_for(pool), timeout=15)
        audio, failed = worker.render(request_for(pool, mode), timeout=.2 if mode == "hang" else 15)
        assert audio is None and failed["status"] == "error" and failed["error_type"] == error
        audio, recovered = worker.render(request_for(pool), timeout=15)
        assert audio is not None and recovered["status"] == "accepted" and recovered["worker_job"] == 1
        assert first["worker_pid"] != recovered["worker_pid"]


def test_rejected_audio_does_not_escape_and_next_patch_keeps_worker(pool):
    with RenderWorker(renderer_factory=CountingFactory()) as worker:
        audio, rejected = worker.render(request_for(pool, "silence"), timeout=15)
        assert audio is None and rejected["status"] == "rejected" and "audio.silence" in rejected["diagnostic_codes"]
        audio, accepted = render_isolated(request_for(pool), worker=worker, timeout=15)
        assert audio is not None and accepted["worker_pid"] == rejected["worker_pid"]
        with pytest.raises(ValueError, match="factory differs"):
            render_isolated(request_for(pool), worker=worker, renderer_factory=lambda: None)


def test_cancellation_reaps_native_worker_before_reusing_shared_memory(pool):
    cancel = Event()
    with RenderWorker(renderer_factory=CountingFactory()) as worker:
        _, first = worker.render(request_for(pool), timeout=15)
        timer = Timer(.2, cancel.set)
        timer.start()
        try:
            audio, stopped = worker.render(request_for(pool, "hang"), timeout=15, cancel=cancel)
        finally:
            timer.cancel()
        assert audio is None and stopped["error_type"] == "Cancelled"
        _, recovered = worker.render(request_for(pool), timeout=15)
        assert recovered["worker_pid"] != first["worker_pid"]


def test_reuse_stream_preserves_backpressure_cursor_and_config_identity(pool):
    sampler = PilotSampler(pool)
    size = request_frames(sampler.recipe_at(0).render_request()) * 8
    config = StreamConfig(prefetch_clips=2, prefetch_bytes=size, worker_jobs=2)
    with OnlineRenderStream(sampler, 3, config=config, renderer_factory=CountingFactory()) as stream:
        rows = []
        for result in stream:
            rows.append(result)
            stream.acknowledge(result.recipe.sequence_id)
        cursor = stream.state_dict()
        assert stream.metrics()["peak_reserved_bytes"] == size and stream.metrics()["accepted_acknowledged"] == 3
    assert rows[0].metadata["worker_pid"] == rows[1].metadata["worker_pid"] != rows[2].metadata["worker_pid"]
    with OnlineRenderStream(sampler, 1, config=config, resume=cursor, renderer_factory=CountingFactory()) as resumed:
        row = next(resumed)
        assert row.recipe.sequence_id == 3 and row.metadata["worker_job"] == 1
    with pytest.raises(ValueError, match="different stream"):
        OnlineRenderStream(sampler, 1, config=replace(config, worker_jobs=3), resume=cursor)
    assert StreamConfig.from_dict(config.to_dict()) == config
    current = StreamConfig().to_dict()
    assert current["worker_jobs"] == 32 and current["plugin_lifetime"] == "job"
    assert StreamConfig.from_dict(current) == StreamConfig()
    with pytest.raises(ValueError, match="lifecycle changed"):
        StreamConfig.from_dict({**current, "stream": "isolated_online_render_v1"})
    with pytest.raises(ValueError, match="contract changed"):
        StreamConfig.from_dict({**current, "plugin_lifetime": "worker"})


def test_oversized_request_is_rejected_before_starting_process(pool):
    before = {child.pid for child in mp.active_children()}
    with RenderWorker(renderer_factory=CountingFactory(), max_bytes=1) as worker:
        with pytest.raises(ValueError, match="memory budget"):
            worker.render(request_for(pool))
    assert {child.pid for child in mp.active_children()} <= before
    for value in (0, 129, True, 1.5):
        with pytest.raises(ValueError, match="worker_jobs"):
            RenderWorker(max_jobs=value)


def test_reused_worker_training_recovers_exact_optimizer_and_rng(pool, tmp_path):
    torch = pytest.importorskip("torch")
    from obruxo_training.evaluate import load_evaluation_checkpoint
    from obruxo_training.trainer import TrainConfig, train_pilot

    config = TrainConfig(mode="online", steps=4, replay_capacity=2, batch_size=1, checkpoint_every=1, render_budget=4, cpu_threads=1)
    stream = StreamConfig(prefetch_clips=1, worker_jobs=2)

    def train(name, **kwargs):
        return train_pilot(pool, tmp_path / name, config=config, stream_config=stream, renderer_factory=CountingFactory(), **kwargs)

    complete = train("complete")
    train("interrupted", stop_after=2)
    resumed = train("resumed", resume=tmp_path / "interrupted/checkpoint.pt")
    assert complete["model_state_sha256"] == resumed["model_state_sha256"] and complete["cursor"] == resumed["cursor"]
    left = torch.load(tmp_path / "complete/checkpoint.pt", weights_only=True)
    right = torch.load(tmp_path / "resumed/checkpoint.pt", weights_only=True)
    assert torch.equal(left["batch_rng"], right["batch_rng"]) and torch.equal(left["rng"]["torch"], right["rng"]["torch"])
    for index, state in left["optimizer"]["state"].items():
        assert all(torch.equal(value, right["optimizer"]["state"][index][name]) for name, value in state.items())
    restored, _, _, _ = load_evaluation_checkpoint(tmp_path / "resumed/checkpoint.pt", pool)
    assert restored["run_contract"]["stream"] == stream.to_dict()
    with pytest.raises(ValueError, match="contract changed"):
        train_pilot(pool, tmp_path / "changed", config=config, stream_config=replace(stream, worker_jobs=3),
                    resume=tmp_path / "interrupted/checkpoint.pt")


def test_codec_correctness_uses_fixed_phase_variants_without_changing_training(pool):
    from obruxo_training.codec_check import CodecCheckSampler
    from obruxo_training.reproducibility import fixed_oscillator_phases

    original_pool = pool.to_dict()
    raw = pool.codec.decode(pool.entries[0].target)
    document = json.loads(raw)
    controlled = json.loads(fixed_oscillator_phases(raw))
    assert all(controlled["settings"][f"osc_{index}_random_phase"] == 0 for index in range(1, 4))
    for index in range(1, 4):
        controlled["settings"][f"osc_{index}_random_phase"] = document["settings"][f"osc_{index}_random_phase"]
    assert controlled == document
    sampler = CodecCheckSampler(pool)
    for recipe in sampler.recipes:
        settings = json.loads(recipe.render_request().preset_json)["settings"]
        assert recipe.split == "train" and all(settings[f"osc_{index}_random_phase"] == 0 for index in range(1, 4))
    assert pool.to_dict() == original_pool

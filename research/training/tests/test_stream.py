from dataclasses import dataclass
import multiprocessing as mp
import os
from time import perf_counter, sleep

import numpy as np
import pytest

from obruxo_data.render.base import RenderProvenance, RenderResult
from obruxo_training.pilot import PilotPool, PilotSampler
from obruxo_training.stream import OnlineRenderStream, StreamConfig, request_frames


@dataclass(frozen=True)
class FakeRendererFactory:
    mode: str = "valid"

    def __call__(self):
        return FakeRenderer(self.mode)


class FakeRenderer:
    def __init__(self, mode):
        self.mode = mode

    def render(self, request):
        if self.mode == "crash":
            os._exit(23)
        if self.mode == "hang":
            sleep(30)
        frames = request_frames(request)
        wave = (0.1 * np.sin(np.arange(frames) * 0.1)).astype(np.float32)
        audio = np.stack((wave, wave), axis=1)
        audio[-request.sample_rate:] = 0
        if self.mode == "silence":
            audio[:] = 0
        elif self.mode == "clipped":
            audio[0] = 2
        elif self.mode == "nan":
            audio[0] = np.nan
        elif self.mode == "tail":
            audio[-request.sample_rate:] = 0.1
        elif self.mode == "dtype":
            audio = audio.astype(np.float64)
        elif self.mode == "shape":
            audio = audio[:-1]
        provenance = RenderProvenance(request.request_id, request.renderer_id, "test-fixture", "fixture")
        # Deliberately omit QA: the stream must inspect the actual buffer itself.
        return RenderResult(audio, request.sample_rate, (), provenance)


@pytest.fixture(scope="module")
def pool():
    return PilotPool.create(12, seed=2)


def test_equal_numeric_timeouts_have_identical_stream_contracts(pool):
    sampler = PilotSampler(pool)
    left = OnlineRenderStream(sampler, 1, config=StreamConfig(timeout_seconds=30))
    right = OnlineRenderStream(sampler, 1, config=StreamConfig(timeout_seconds=30.0), resume=left.state_dict())
    assert left.stream_id == right.stream_id


def test_stream_enforces_byte_backpressure_and_consumption_acknowledgements(pool):
    sampler = PilotSampler(pool)
    size = request_frames(sampler.recipe_at(0).render_request()) * 8
    config = StreamConfig(prefetch_clips=2, prefetch_bytes=size)
    with OnlineRenderStream(sampler, 3, config=config, renderer_factory=FakeRendererFactory()) as stream:
        deadline = perf_counter() + 15
        while stream.metrics()["produced"] < 1 and perf_counter() < deadline:
            sleep(0.01)
        assert stream.metrics()["produced"] == 1
        for outcome in stream:
            assert outcome.metadata["status"] == "accepted"
            assert outcome.audio.nbytes == size and not outcome.audio.flags.writeable
            with pytest.raises(ValueError, match="acknowledged"):
                stream.acknowledge(outcome.recipe.sequence_id + 1)
            stream.acknowledge(outcome.recipe.sequence_id)
        metrics = stream.metrics()
        assert metrics["accepted_acknowledged"] == 3
        assert metrics["peak_reserved_bytes"] == size and metrics["peak_reserved_clips"] == 1
        assert stream.state_dict()["next_sequence_id"] == 3


@pytest.mark.parametrize("mode,code", [("silence", "audio.silence"), ("clipped", "audio.clipping"),
                                      ("nan", "audio.non_finite"), ("tail", "audio.tail_truncation")])
def test_actual_audio_qa_rejections_are_visible_not_consumed(pool, mode, code):
    with OnlineRenderStream(PilotSampler(pool), 1, renderer_factory=FakeRendererFactory(mode)) as stream:
        result = next(stream)
        assert result.metadata["status"] == "rejected" and result.audio is None
        assert code in result.metadata["diagnostic_codes"]
        stream.acknowledge(0)
        assert stream.metrics()["accepted_acknowledged"] == 0


@pytest.mark.parametrize("mode", ["shape", "dtype"])
def test_incompatible_render_buffers_fail_explicitly(pool, mode):
    with OnlineRenderStream(PilotSampler(pool), 1, renderer_factory=FakeRendererFactory(mode)) as stream:
        result = next(stream)
        assert result.audio is None and result.metadata["error_type"] == "ValueError"
        stream.acknowledge(0)


def test_crashed_workers_have_bounded_retries_and_do_not_drop_sequences(pool):
    config = StreamConfig(max_retries=1)
    with OnlineRenderStream(PilotSampler(pool), 2, config=config, renderer_factory=FakeRendererFactory("crash")) as stream:
        results = []
        for result in stream:
            results.append(result)
            stream.acknowledge(result.recipe.sequence_id)
        assert [result.recipe.sequence_id for result in results] == [0, 1]
        assert all(len(result.metadata["attempts"]) == 2 for result in results)
        assert all(result.metadata["error_type"] == "WorkerExited" for result in results)
        assert stream.metrics()["errors"] == 2


def test_hung_worker_times_out_and_is_reaped(pool):
    before = {process.pid for process in mp.active_children()}
    config = StreamConfig(timeout_seconds=0.2)
    with OnlineRenderStream(PilotSampler(pool), 1, config=config, renderer_factory=FakeRendererFactory("hang")) as stream:
        result = next(stream)
        assert result.metadata["error_type"] == "RenderTimeout"
        stream.acknowledge(0)
    assert {process.pid for process in mp.active_children()} <= before


def test_early_close_and_resume_regenerate_unacknowledged_recipe(pool):
    config = StreamConfig(prefetch_clips=1)
    with OnlineRenderStream(PilotSampler(pool), 5, config=config, renderer_factory=FakeRendererFactory()) as stream:
        first = next(stream)
        stream.acknowledge(first.recipe.sequence_id)
        pending = next(stream)
        cursor = stream.state_dict()
    assert cursor["next_sequence_id"] == 1
    with OnlineRenderStream(PilotSampler(pool), 1, config=config, renderer_factory=FakeRendererFactory(), resume=cursor) as resumed:
        replayed = next(resumed)
        assert replayed.recipe.recipe_id == pending.recipe.recipe_id
        resumed.acknowledge(1)
        assert resumed.state_dict()["next_sequence_id"] == 2
    with pytest.raises(ValueError, match="different stream"):
        OnlineRenderStream(PilotSampler(pool, seed=9), 1, config=config, resume=cursor)


def test_oversized_clip_fails_without_launching_worker(pool):
    with OnlineRenderStream(PilotSampler(pool), 1, config=StreamConfig(prefetch_bytes=8), renderer_factory=FakeRendererFactory()) as stream:
        with pytest.raises(RuntimeError, match="producer failed") as failure:
            next(stream)
        assert "byte budget" in str(failure.value.__cause__)


def test_contract_caps_cannot_be_overridden(pool):
    with pytest.raises(ValueError, match="exceed"):
        OnlineRenderStream(PilotSampler(pool), 1, config=StreamConfig(prefetch_clips=33))


def test_close_cancels_pending_worker_without_waiting_for_native_deadline(pool):
    before = {process.pid for process in mp.active_children()}
    started = perf_counter()
    with OnlineRenderStream(PilotSampler(pool), 1, renderer_factory=FakeRendererFactory("hang")) as stream:
        deadline = perf_counter() + 10
        while not stream.metrics()["peak_reserved_clips"] and perf_counter() < deadline:
            sleep(0.01)
        assert stream.metrics()["peak_reserved_clips"] == 1
    assert perf_counter() - started < 10
    assert {process.pid for process in mp.active_children()} <= before

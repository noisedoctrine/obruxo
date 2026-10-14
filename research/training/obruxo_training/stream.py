from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import json
import math
import multiprocessing as mp
from multiprocessing import shared_memory
from threading import Condition, Event, Thread
from time import perf_counter

import numpy as np

from obruxo_data.errors import Severity
from obruxo_data.hashing import canonical_sha256
from obruxo_data.midi import TempoMap
from obruxo_data.render.base import RenderRequest
from obruxo_data.render.qa import AudioQualityConfig, analyze_audio

from .contracts import RenderRecipe


REJECT_AUDIO_CODES = frozenset({"audio.silence", "audio.clipping", "audio.tail_truncation"})


@dataclass(frozen=True)
class StreamConfig:
    prefetch_clips: int = 4
    prefetch_bytes: int = 32 * 1024 * 1024
    timeout_seconds: float = 60.0
    max_retries: int = 0

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in (self.prefetch_clips, self.prefetch_bytes)):
            raise ValueError("prefetch limits must be positive integers")
        if type(self.timeout_seconds) not in {int, float} or not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("worker timeout must be positive and finite")
        if type(self.max_retries) is not int or not 0 <= self.max_retries <= 1:
            raise ValueError("max_retries must be 0 or 1")
        object.__setattr__(self, "timeout_seconds", float(self.timeout_seconds))

    def to_dict(self) -> dict:
        return {"stream": "isolated_online_render_v1", **asdict(self), "qa": asdict(AudioQualityConfig()),
                "reject_audio_codes": sorted(REJECT_AUDIO_CODES), "resume_audio": "regenerate_unacknowledged"}


@dataclass(frozen=True)
class NativeRendererFactory:
    plugin_path: str | None = None

    def __call__(self):
        from obruxo_data.render import VitalRenderer
        return VitalRenderer(plugin_path=self.plugin_path)


@dataclass(frozen=True)
class StreamOutcome:
    recipe: RenderRecipe
    audio: np.ndarray | None
    metadata: dict


def request_frames(request: RenderRequest) -> int:
    end_tick = request.performance.end_tick if request.end_tick is None else request.end_tick
    return TempoMap.from_performance(request.performance).render_frame_count(end_tick, request.tail_seconds, request.sample_rate)


def _render_job(connection, factory, request_json: str, memory_name: str, frames: int):
    """One native process per job; only small metadata crosses the pipe, audio uses bounded shared memory."""
    segment = None
    try:
        request = RenderRequest.from_dict(json.loads(request_json))
        renderer = factory()
        started = perf_counter()
        result = renderer.render(request)
        render_seconds = perf_counter() - started
        if result.sample_rate != request.sample_rate or result.audio.dtype != np.dtype("float32"):
            raise ValueError("renderer returned an incompatible sample rate or dtype")
        if result.audio.shape != (frames, request.channels):
            raise ValueError("renderer returned an incompatible audio shape")
        if result.provenance.request_id != request.request_id or result.provenance.renderer_id != request.renderer_id:
            raise ValueError("renderer provenance does not match the request")
        qa, audio_diagnostics = analyze_audio(result.audio, sample_rate=request.sample_rate, expected_frames=frames,
                                             expected_channels=request.channels)
        diagnostics = result.diagnostics + audio_diagnostics
        codes = sorted({item.code for item in diagnostics})
        rejected = any(item.severity == Severity.ERROR or item.code in REJECT_AUDIO_CODES for item in diagnostics)
        if not rejected:
            segment = shared_memory.SharedMemory(name=memory_name)
            buffer = np.ndarray((frames, request.channels), dtype=np.float32, buffer=segment.buf)
            buffer[:] = result.audio
            del buffer
        connection.send({"status": "rejected" if rejected else "accepted", "qa": qa, "renderer_qa": result.qa,
                         "diagnostic_codes": codes,
                         "render_seconds": render_seconds, "provenance": result.provenance.to_dict()})
    except Exception as error:
        connection.send({"status": "error", "error_type": type(error).__name__, "message": str(error)[:1000]})
    finally:
        if segment is not None:
            segment.close()
        connection.close()


def _render_once(request: RenderRequest, factory, timeout: float, cancel: Event) -> tuple[np.ndarray | None, dict]:
    frames = request_frames(request)
    memory = shared_memory.SharedMemory(create=True, size=frames * request.channels * 4)
    context = mp.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=_render_job, args=(send, factory, json.dumps(request.to_dict()), memory.name, frames), daemon=True)
    started = perf_counter()
    try:
        process.start()
        send.close()
        while not cancel.is_set():
            remaining = timeout - (perf_counter() - started)
            if remaining <= 0:
                return None, {"status": "error", "error_type": "RenderTimeout", "message": "native worker exceeded its deadline"}
            if receive.poll(min(0.1, remaining)):
                try:
                    metadata = receive.recv()
                except EOFError:
                    return None, {"status": "error", "error_type": "WorkerExited", "exit_code": process.exitcode}
                if metadata["status"] != "accepted":
                    return None, metadata
                audio = np.ndarray((frames, request.channels), dtype=np.float32, buffer=memory.buf).copy()
                audio.setflags(write=False)
                return audio, metadata
            if not process.is_alive():
                return None, {"status": "error", "error_type": "WorkerExited", "exit_code": process.exitcode}
        return None, {"status": "error", "error_type": "Cancelled"}
    finally:
        send.close()
        receive.close()
        if process.pid is not None:
            process.join(0.2)
            if process.is_alive():
                process.terminate()
                process.join(1)
            if process.is_alive():
                process.kill()
                process.join(1)
        process.close()
        memory.close()
        memory.unlink()


def render_isolated(request: RenderRequest, *, renderer_factory=None, timeout: float = 60) -> tuple[np.ndarray | None, dict]:
    """One QA-gated evaluator render without assigning a prediction to a source split."""
    config = StreamConfig(prefetch_clips=1, timeout_seconds=timeout)
    if request_frames(request) * request.channels * 4 > config.prefetch_bytes:
        raise ValueError("evaluation render exceeds the one-clip memory budget")
    started = perf_counter()
    audio, metadata = _render_once(request, renderer_factory or NativeRendererFactory(), timeout, Event())
    return audio, {**metadata, "wall_seconds": perf_counter() - started}


class OnlineRenderStream:
    """Single producer with byte/clip reservation before dispatch and explicit consumer acknowledgements."""

    def __init__(self, sampler, count: int, *, config: StreamConfig | None = None, renderer_factory=None, resume: dict | None = None):
        self.config = config or StreamConfig()
        contract = sampler.pool.codec.contract
        if self.config.prefetch_clips > contract.prefetch_clips or self.config.prefetch_bytes > contract.prefetch_bytes:
            raise ValueError("stream limits exceed the frozen training contract")
        if type(count) is not int or count <= 0:
            raise ValueError("stream count must be a positive integer")
        self.sampler, self.count = sampler, count
        self.factory = renderer_factory or NativeRendererFactory()
        self.stream_id = canonical_sha256({"sampler_id": sampler.sampler_id, "config": self.config.to_dict()})
        self._next_ack = 0
        if resume is not None:
            if resume.get("version") != 1 or resume.get("stream_id") != self.stream_id:
                raise ValueError("resume cursor belongs to a different stream")
            self._next_ack = resume["next_sequence_id"]
            if type(self._next_ack) is not int or self._next_ack < 0:
                raise ValueError("resume sequence must be a nonnegative integer")
        self._start = self._next_ack
        self._condition, self._stop = Condition(), Event()
        self._queue, self._delivered = deque(), deque()
        self._bytes, self._clips, self._done, self._error = 0, 0, False, None
        self._thread = None
        self._metrics = {"produced": 0, "accepted": 0, "rejected": 0, "errors": 0, "acknowledged": 0,
                         "accepted_acknowledged": 0, "peak_reserved_bytes": 0, "peak_reserved_clips": 0,
                         "producer_wait_seconds": 0.0, "consumer_wait_seconds": 0.0}

    def __enter__(self):
        if self._thread is not None or self._stop.is_set():
            raise RuntimeError("stream cannot be restarted; create a new stream from its cursor")
        self._thread = Thread(target=self._produce, name="obruxo-render-producer", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self.close()

    def __iter__(self):
        return self

    def __next__(self) -> StreamOutcome:
        if self._thread is None:
            raise RuntimeError("use the stream as a context manager")
        with self._condition:
            if len(self._delivered) >= self.config.prefetch_clips:
                raise RuntimeError("acknowledge delivered outcomes before requesting more")
            started = perf_counter()
            while not self._queue and not self._done and not self._stop.is_set():
                self._condition.wait()
            self._metrics["consumer_wait_seconds"] += perf_counter() - started
            if not self._queue:
                if self._error is not None:
                    raise RuntimeError(f"render producer failed: {self._error}") from self._error
                raise StopIteration
            outcome = self._queue.popleft()
            self._clips -= 1
            self._bytes -= 0 if outcome.audio is None else outcome.audio.nbytes
            self._delivered.append((outcome.recipe.sequence_id, outcome.metadata["status"]))
            self._condition.notify_all()
            return outcome

    def acknowledge(self, sequence_id: int) -> None:
        with self._condition:
            if not self._delivered or sequence_id != self._next_ack or sequence_id != self._delivered[0][0]:
                raise ValueError("outcomes must be acknowledged once, in sequence, after handling them")
            _, status = self._delivered.popleft()
            self._next_ack += 1
            self._metrics["acknowledged"] += 1
            self._metrics["accepted_acknowledged"] += status == "accepted"

    def state_dict(self) -> dict:
        with self._condition:
            return {"version": 1, "stream_id": self.stream_id, "next_sequence_id": self._next_ack,
                    "audio_replay": "regenerate_unacknowledged_not_bit_exact"}

    def metrics(self) -> dict:
        with self._condition:
            return dict(self._metrics)

    def close(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(5)
            if self._thread.is_alive():
                raise RuntimeError("render producer did not shut down")
        with self._condition:
            self._queue.clear()
            self._bytes, self._clips = 0, 0

    def _reserve(self, size: int) -> bool:
        if size > self.config.prefetch_bytes or size <= 0:
            raise ValueError("one render does not fit the prefetch byte budget")
        with self._condition:
            started = perf_counter()
            while not self._stop.is_set() and (self._clips >= self.config.prefetch_clips or self._bytes + size > self.config.prefetch_bytes):
                self._condition.wait()
            self._metrics["producer_wait_seconds"] += perf_counter() - started
            if self._stop.is_set():
                return False
            self._clips += 1
            self._bytes += size
            self._metrics["peak_reserved_clips"] = max(self._clips, self._metrics["peak_reserved_clips"])
            self._metrics["peak_reserved_bytes"] = max(self._bytes, self._metrics["peak_reserved_bytes"])
            return True

    def _produce(self) -> None:
        try:
            for sequence in range(self._start, self._start + self.count):
                if self._stop.is_set():
                    break
                recipe = self.sampler.recipe_at(sequence)
                request = recipe.render_request()
                size = request_frames(request) * request.channels * 4
                if not self._reserve(size):
                    break
                started, attempts, audio = perf_counter(), [], None
                for attempt in range(self.config.max_retries + 1):
                    audio, metadata = _render_once(request, self.factory, self.config.timeout_seconds, self._stop)
                    attempts.append({"attempt": attempt, **metadata})
                    if metadata["status"] != "error" or self._stop.is_set():
                        break
                metadata = {**metadata, "attempts": attempts, "wall_seconds": perf_counter() - started}
                outcome = StreamOutcome(recipe, audio, metadata)
                with self._condition:
                    if self._stop.is_set():
                        break
                    if audio is None:
                        self._bytes -= size
                    self._queue.append(outcome)
                    self._metrics["produced"] += 1
                    self._metrics[{"accepted": "accepted", "rejected": "rejected", "error": "errors"}[metadata["status"]]] += 1
                    self._condition.notify_all()
        except Exception as error:
            self._error = error
        finally:
            with self._condition:
                self._done = True
                self._condition.notify_all()

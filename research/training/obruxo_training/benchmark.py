from __future__ import annotations

import json
import os
from pathlib import Path
import platform
from threading import Event, Thread
from time import perf_counter

import numpy as np

from .audit import write_json
from .pilot import PilotPool, PilotSampler
from .stream import NativeRendererFactory, OnlineRenderStream, StreamConfig


def spectral_features(audio: np.ndarray) -> np.ndarray:
    """Small CPU consumer for pipeline profiling; not the eventual learned feature frontend."""
    mono = audio.mean(axis=1)
    windows = np.lib.stride_tricks.sliding_window_view(mono, 1024)[::512]
    spectrum = np.abs(np.fft.rfft(windows * np.hanning(1024), axis=1))
    return np.log1p(spectrum).mean(axis=0).astype(np.float32)


class MemoryMonitor:
    def __init__(self):
        self.peak_bytes, self.errors, self._stop, self._thread = None, 0, Event(), None

    def start(self):
        try:
            import psutil
        except ImportError:
            return
        process = psutil.Process()

        def sample():
            while not self._stop.is_set():
                try:
                    rss = process.memory_info().rss
                    for child in process.children(recursive=True):
                        try:
                            rss += child.memory_info().rss
                        except psutil.NoSuchProcess:
                            pass
                    self.peak_bytes = max(self.peak_bytes or 0, rss)
                except psutil.Error:
                    self.errors += 1
                self._stop.wait(0.05)
        self._thread = Thread(target=sample, name="obruxo-memory-monitor", daemon=True)
        self._thread.start()

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(1)


def benchmark_stream(pool: PilotPool, output: Path, *, count: int = 32, split: str = "train", role: str = "reference",
                     config: StreamConfig | None = None, renderer_factory=None, resume: dict | None = None, progress=None) -> dict:
    sampler = PilotSampler(pool, split=split, role=role)
    stream = OnlineRenderStream(sampler, count, config=config,
                                renderer_factory=renderer_factory or NativeRendererFactory(renderer_id=pool.codec.contract.renderer_id), resume=resume)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "status.json", {"complete": False, "stage": "render_and_feature_consumer"})
    write_json(output / "cursor.json", stream.state_dict())
    monitor = MemoryMonitor()
    monitor.start()
    started, feature_seconds, latencies, states = perf_counter(), 0.0, [], set()
    try:
        with (output / "events.jsonl").open("w", encoding="utf-8", newline="\n") as events, stream:
            for outcome in stream:
                recipe = outcome.recipe
                event = {"event": "render_outcome", "sequence_id": recipe.sequence_id, "recipe_id": recipe.recipe_id,
                         "request_id": recipe.render_request().request_id, "state_id": recipe.state_id,
                         "lineage_id": recipe.lineage_id, "split": recipe.split, "performance_id": recipe.performance_id,
                         "performance_role": recipe.performance_role, **outcome.metadata}
                events.write(json.dumps(event, allow_nan=False, sort_keys=True) + "\n")
                latencies.append(outcome.metadata["wall_seconds"])
                states.add(recipe.state_id)
                if outcome.audio is not None:
                    feature_started = perf_counter()
                    features = spectral_features(outcome.audio)
                    elapsed = perf_counter() - feature_started
                    feature_seconds += elapsed
                    events.write(json.dumps({"event": "consumed", "sequence_id": recipe.sequence_id,
                                             "recipe_id": recipe.recipe_id, "feature_seconds": elapsed,
                                             "feature_bins": len(features), "feature_mean": float(features.mean()),
                                             "audio_sha256": outcome.metadata["qa"]["audio_float32_sha256"]}) + "\n")
                else:
                    events.write(json.dumps({"event": "handled_failure", "sequence_id": recipe.sequence_id,
                                             "status": outcome.metadata["status"]}) + "\n")
                events.flush()
                os.fsync(events.fileno())
                stream.acknowledge(recipe.sequence_id)
                write_json(output / "cursor.json", stream.state_dict())
                if progress:
                    progress(stream.metrics()["acknowledged"], count, outcome.metadata["status"])
        elapsed = perf_counter() - started
    except BaseException as error:
        write_json(output / "status.json", {"complete": False, "stage": "interrupted", "error_type": type(error).__name__})
        raise
    finally:
        monitor.close()
    metrics = stream.metrics()
    result = {"version": 1, "benchmark": "native_renderer_and_cpu_spectral_consumer_v1", "pool_id": pool.pool_id,
              "codec_id": pool.codec.codec_id, "contract_id": pool.codec.contract.contract_id,
              "split_manifest_id": pool.manifest.manifest_id, "stream_id": stream.stream_id,
              "split": split, "performance_role": role, "config": stream.config.to_dict(),
              "requested": count, "distinct_states": len(states), **metrics,
              "elapsed_seconds": elapsed, "accepted_clips_per_second": metrics["accepted"] / elapsed,
              "consumer_wait_fraction": metrics["consumer_wait_seconds"] / elapsed,
              "feature_seconds": feature_seconds, "job_latency_median_seconds": float(np.median(latencies)),
              "job_latency_p95_seconds": float(np.percentile(latencies, 95)),
              "peak_process_tree_rss_bytes": monitor.peak_bytes, "rss_sampling_errors": monitor.errors,
              "rss_scope": "sampled_every_50ms_sum_parent_and_children_not_unique_physical_memory",
              "audio_files_written": 0, "optimizer_steps": 0, "platform": platform.platform(),
              "python": platform.python_version(), "numpy": np.__version__,
              "renderer_factory": type(stream.factory).__name__, "resume_cursor": stream.state_dict()}
    write_json(output / "summary.json", result)
    write_json(output / "status.json", {"complete": True, "stage": "complete", "all_accepted": metrics["accepted"] == count})
    return result

"""Small device/data-movement benchmark for Experiment 6 alignment kernels."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np

from .alignment5 import exact_align_cpu, exact_align_xpu


@dataclass(frozen=True)
class BenchCase:
    name: str
    targets: int
    codes: int
    resolution: int
    repeats: int
    calls_per_repeat: int = 1


def _make_arrays(seed: int, targets: int, codes: int, resolution: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, resolution, endpoint=False, dtype=np.float32)
    target = (
        0.5
        + 0.25 * np.sin(2 * np.pi * (rng.uniform(0.5, 4.0, size=(targets, 1)) * x + rng.random((targets, 1))))
        + 0.15 * rng.normal(size=(targets, resolution))
    )
    code = (
        0.5
        + 0.3 * np.sin(2 * np.pi * (rng.uniform(0.5, 6.0, size=(codes, 1)) * x + rng.random((codes, 1))))
        + 0.1 * rng.normal(size=(codes, resolution))
    )
    return np.clip(target, 0, 1).astype(np.float32), (code - code.mean(axis=1, keepdims=True)).astype(np.float32)


def _time_call(fn, repeats: int) -> tuple[float, list[float]]:
    times: list[float] = []
    fn()
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        times.append(time.perf_counter() - started)
    return statistics.median(times), times


def _sync_xpu() -> None:
    try:
        import torch

        if torch.xpu.is_available():
            torch.xpu.synchronize()
    except Exception:
        pass


def _run_cpu_worker(args: tuple[int, int, int, int]) -> float:
    seed, targets, codes, resolution = args
    target, code = _make_arrays(seed, targets, codes, resolution)
    started = time.perf_counter()
    exact_align_cpu(target, code)
    return time.perf_counter() - started


def _run_xpu_worker(args: tuple[int, int, int, int]) -> float:
    seed, targets, codes, resolution = args
    target, code = _make_arrays(seed, targets, codes, resolution)
    started = time.perf_counter()
    exact_align_xpu(target, code)
    _sync_xpu()
    return time.perf_counter() - started


def inspect_torch_devices() -> dict[str, object]:
    report: dict[str, object] = {}
    try:
        import torch

        report["torch_version"] = torch.__version__
        report["xpu_available"] = bool(torch.xpu.is_available())
        report["xpu_device_count"] = int(torch.xpu.device_count()) if hasattr(torch, "xpu") else 0
        if report["xpu_available"]:
            report["xpu_name"] = torch.xpu.get_device_name(0)
        report["cuda_available"] = bool(torch.cuda.is_available())
        report["mps_available"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        report["npu_backend_attrs"] = [name for name in ("npu", "privateuseone") if hasattr(torch, name)]
    except Exception as exc:
        report["torch_error"] = repr(exc)
    return report


def run_device_benchmark(output_dir: Path, *, repeats: int = 5, parallel: int = 2) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = [
        BenchCase("e6_batch2_stage_like", targets=2, codes=16, resolution=1024, repeats=repeats),
        BenchCase("e6_batch2_eval1920", targets=2, codes=16, resolution=1920, repeats=repeats),
        BenchCase("grouped_8_calls", targets=16, codes=16, resolution=1024, repeats=repeats, calls_per_repeat=8),
        BenchCase("fatter_batch", targets=32, codes=16, resolution=1024, repeats=max(3, repeats // 2)),
    ]
    rows: list[dict[str, object]] = []
    device_report = inspect_torch_devices()
    xpu_available = bool(device_report.get("xpu_available"))

    for case in cases:
        target, code = _make_arrays(1000 + len(rows), case.targets, case.codes, case.resolution)

        def cpu_serial() -> None:
            exact_align_cpu(target, code)

        median, times = _time_call(cpu_serial, case.repeats)
        rows.append({
            "case": case.name, "route": "cpu_serial", "parallel": 1,
            "median_seconds": median, "all_seconds": times,
            "target_count": case.targets, "code_count": case.codes, "resolution": case.resolution,
        })

        if xpu_available:
            def xpu_serial() -> None:
                exact_align_xpu(target, code)
                _sync_xpu()

            median, times = _time_call(xpu_serial, case.repeats)
            rows.append({
                "case": case.name, "route": "xpu_serial_copy_each_call", "parallel": 1,
                "median_seconds": median, "all_seconds": times,
                "target_count": case.targets, "code_count": case.codes, "resolution": case.resolution,
            })

        if case.name == "grouped_8_calls":
            small_target, small_code = _make_arrays(2000, 2, case.codes, case.resolution)

            def cpu_eight_small() -> None:
                for _ in range(case.calls_per_repeat):
                    exact_align_cpu(small_target, small_code)

            median, times = _time_call(cpu_eight_small, case.repeats)
            rows.append({
                "case": case.name, "route": "cpu_8_separate_small_calls", "parallel": 1,
                "median_seconds": median, "all_seconds": times,
                "target_count": 2, "code_count": case.codes, "resolution": case.resolution,
            })
            if xpu_available:
                def xpu_eight_small() -> None:
                    for _ in range(case.calls_per_repeat):
                        exact_align_xpu(small_target, small_code)
                    _sync_xpu()

                median, times = _time_call(xpu_eight_small, case.repeats)
                rows.append({
                    "case": case.name, "route": "xpu_8_separate_small_calls", "parallel": 1,
                    "median_seconds": median, "all_seconds": times,
                    "target_count": 2, "code_count": case.codes, "resolution": case.resolution,
                })

        worker_args = [(3000 + i, case.targets, case.codes, case.resolution) for i in range(max(1, parallel))]
        with ProcessPoolExecutor(max_workers=max(1, parallel)) as executor:
            started = time.perf_counter()
            list(executor.map(_run_cpu_worker, worker_args))
            elapsed = time.perf_counter() - started
        rows.append({
            "case": case.name, "route": "cpu_process_parallel_wall", "parallel": max(1, parallel),
            "median_seconds": elapsed, "all_seconds": [elapsed],
            "target_count": case.targets, "code_count": case.codes, "resolution": case.resolution,
        })

        if xpu_available:
            with ProcessPoolExecutor(max_workers=max(1, parallel)) as executor:
                started = time.perf_counter()
                list(executor.map(_run_xpu_worker, worker_args))
                elapsed = time.perf_counter() - started
            rows.append({
                "case": case.name, "route": "xpu_process_parallel_wall", "parallel": max(1, parallel),
                "median_seconds": elapsed, "all_seconds": [elapsed],
                "target_count": case.targets, "code_count": case.codes, "resolution": case.resolution,
            })

    report = {
        "environment": {
            "pid": os.getpid(),
            "parallel": parallel,
            "repeats": repeats,
            **device_report,
        },
        "results": rows,
    }
    path = output_dir / "experiment6_device_benchmark.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


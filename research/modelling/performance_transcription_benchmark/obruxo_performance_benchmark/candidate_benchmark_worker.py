"""One foreground, single-route worker for an executable #26 candidate."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any


def _basic_pitch_root() -> Path:
    return Path(__file__).resolve().parents[2] / "basic_pitch"


def _sync(torch: Any, device: Any) -> None:
    if device.type == "xpu":
        torch.xpu.synchronize(device)


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "median": float(median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "total": float(sum(values)),
    }


def _run(request: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import psutil
    import torch

    package_root = Path(__file__).resolve().parent.parent
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from obruxo_performance_benchmark.adapters import adapter_for
    from obruxo_performance_benchmark.artifacts import load_model_specs
    from obruxo_performance_benchmark.resources import (
        count_parameters,
        file_size,
        state_tensor_bytes_by_dtype,
        xpu_memory_bytes,
    )

    basic_root = _basic_pitch_root()
    if str(basic_root) not in sys.path:
        sys.path.insert(0, str(basic_root))
    from obruxo_basic_pitch.benchmark import load_config, load_manifest

    config_path = Path(str(request["config_path"])).resolve(strict=True)
    spec = load_model_specs(config_path)[str(request["model_id"])]
    route = str(request["route"])
    if route not in {"pytorch_cpu", "pytorch_xpu"}:
        return {
            "status": "runtime_failed",
            "failure_code": "unsupported_candidate_route",
            "route": route,
        }
    device_name = (
        "cpu" if route == "pytorch_cpu" else f"xpu:{int(request.get('xpu_index', 0))}"
    )
    if route == "pytorch_xpu" and (
        not hasattr(torch, "xpu") or not torch.xpu.is_available()
    ):
        return {
            "status": "unavailable",
            "failure_code": "torch_xpu_unavailable",
            "route": route,
        }
    device = torch.device(device_name)
    smoke_root = Path(str(request["smoke_manifest"])).resolve(strict=True)
    config = load_config(basic_root / "configs" / "backend_benchmark.yaml")
    cases = load_manifest(
        smoke_root, config, allow_derived_render=True, allow_missing_derived_audio=False
    )
    if any(not case.audio_path.is_file() for case in cases):
        return {
            "status": "unavailable",
            "failure_code": "derived_audio_unavailable",
            "route": route,
        }
    adapter = adapter_for(
        spec, Path(str(request["source_root"])), Path(str(request["checkpoint"]))
    )
    import_started = time.perf_counter()
    adapter.load(device=device.type)
    model_load_seconds = time.perf_counter() - import_started
    quantization_info = None
    if bool(request.get("quantized", False)):
        if route != "pytorch_cpu":
            return {
                "status": "unavailable",
                "failure_code": "quantized_cpu_only",
                "route": route,
            }
        from obruxo_performance_benchmark.quantization import (
            quantize_dynamic_linear_int8,
        )

        quantization = quantize_dynamic_linear_int8(getattr(adapter, "model", None))
        quantization_info = {
            "status": str(quantization.status),
            "original_linear_modules": int(quantization.original_linear_modules),
            "quantized_linear_modules": int(quantization.quantized_linear_modules),
            "engine": quantization.engine,
        }
        if quantization.status != "ok" or quantization.model is None:
            return {
                "status": "failed",
                "failure_code": "quantization_unsupported",
                "route": route,
                "quantization": quantization_info,
            }
        adapter.bind_model(quantization.model)
        if getattr(adapter, "active_model", None) is not quantization.model:
            return {
                "status": "failed",
                "failure_code": "quantized_runtime_failed",
                "route": route,
                "quantization": quantization_info,
            }
    model = getattr(adapter, "active_model", None)
    if model is None:
        model = getattr(adapter, "model", None)
    resources: dict[str, Any] = {
        "parameters": None,
        "trainable_parameters": None,
        "state_tensor_bytes_by_dtype": None,
        "checkpoint_bytes": file_size(
            Path(str(request["checkpoint"])).resolve(strict=True)
        ),
        "host_peak_rss_bytes": None,
        "xpu_memory": None,
    }
    if model is not None and hasattr(model, "parameters"):
        total, trainable = count_parameters(model)
        resources["parameters"] = total
        resources["trainable_parameters"] = trainable
        resources["state_tensor_bytes_by_dtype"] = state_tensor_bytes_by_dtype(model)
    sync = lambda: _sync(torch, device)
    first_case = cases[0]
    backward_measurement = getattr(adapter, "backward_measurement", None)
    if bool(request.get("quantized", False)):
        backward = {
            "status": "not_applicable",
            "reason": "quantized_variant_cpu_inference_only",
        }
    elif callable(backward_measurement):
        sync()
        backward_started = time.perf_counter()
        backward_measurement(first_case.audio_path)
        sync()
        backward = {
            "status": "measured",
            "seconds": float(time.perf_counter() - backward_started),
            "boundary": "native_forward_transcription_tensor_mean_backward",
            "reduction": "mean_square_of_official_transcription_tensor",
        }
    else:
        backward = {
            "status": "not_applicable",
            "reason": "adapter does not expose a natural differentiable boundary",
        }
    sync()
    started = time.perf_counter()
    adapter.transcribe(first_case.audio_path)
    sync()
    first_seconds = time.perf_counter() - started
    sync()
    warmup_started = time.perf_counter()
    for _ in range(config.warmup_iterations):
        adapter.transcribe(first_case.audio_path)
    sync()
    warmup_seconds = time.perf_counter() - warmup_started
    timed: list[float] = []
    for _ in range(config.timed_iterations):
        sync()
        started = time.perf_counter()
        adapter.transcribe(first_case.audio_path)
        sync()
        timed.append(time.perf_counter() - started)
    e2e_rows: list[dict[str, Any]] = []
    for case in cases:
        sync()
        started = time.perf_counter()
        output = adapter.transcribe(case.audio_path)
        sync()
        wall = time.perf_counter() - started
        from scipy.io import wavfile

        sample_rate, samples = wavfile.read(case.audio_path)
        audio_seconds = float(np.asarray(samples).shape[0] / sample_rate)
        e2e_rows.append(
            {
                "case_index": case.case_index,
                "audio_seconds": audio_seconds,
                "wall_seconds": float(wall),
                "status": "ok",
                "output_has_notes": output.notes is not None,
            }
        )
    resources["host_peak_rss_bytes"] = int(psutil.Process().memory_info().rss)
    resources["xpu_memory"] = xpu_memory_bytes() if device.type == "xpu" else None
    batch_scaling = {
        str(size): {
            "status": "not_applicable",
            "reason": "adapter exposes only full-clip stock transcription; synthetic serial calls are not native batching",
        }
        for size in (1, 2, 4, 8)
    }
    return {
        "status": "ok",
        "failure_code": None,
        "route": route,
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
            "psutil": psutil.__version__,
        },
        "startup": {
            "model_load_seconds": float(model_load_seconds),
            "first_call_seconds": float(first_seconds),
            "warmup_seconds": float(warmup_seconds),
        },
        "steady_state": {
            "median_seconds": _stats(timed)["median"],
            "seconds": _stats(timed),
            "calls_per_second": float(1.0 / _stats(timed)["median"]),
        },
        "batch_scaling": batch_scaling,
        "end_to_end": {
            "status": "measured",
            "cases": e2e_rows,
            "wall_seconds": float(sum(row["wall_seconds"] for row in e2e_rows)),
        },
        "resources": resources,
        "backward": backward,
        "native_batch_sizes": list(spec.native_batch_sizes),
        "quantization": quantization_info,
    }


def main() -> int:
    try:
        result = _run(json.load(sys.stdin))
    except MemoryError:
        result = {"status": "out_of_memory", "failure_code": "out_of_memory"}
    except Exception as exc:  # noqa: BLE001 - worker boundary serializes route failure
        failure_code = (
            "out_of_memory"
            if type(exc).__name__ == "OutOfMemoryError"
            else "candidate_runtime_error"
        )
        result = {
            "status": "out_of_memory"
            if failure_code == "out_of_memory"
            else "runtime_failed",
            "failure_code": failure_code,
            "error_type": type(exc).__name__,
        }
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

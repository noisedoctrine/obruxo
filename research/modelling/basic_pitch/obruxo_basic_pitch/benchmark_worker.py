"""One fresh-process Basic Pitch benchmark worker.

Only standard-library modules are imported before the request is parsed. The
worker never serializes source paths or per-source predictions.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


class _RouteError(RuntimeError):
    def __init__(self, status: str, code: str) -> None:
        super().__init__(code)
        self.status = status
        self.code = code


class _ParityError(RuntimeError):
    pass


class _TrainingError(RuntimeError):
    pass


def _measure_calls(
    invoke: Callable[[], Any],
    *,
    batch_size: int,
    warmup_iterations: int,
    timed_iterations: int,
    synchronize: Callable[[], None] | None = None,
) -> dict[str, float]:
    synchronize = synchronize or (lambda: None)
    synchronize()
    first_started = time.perf_counter()
    invoke()
    synchronize()
    first_seconds = time.perf_counter() - first_started

    synchronize()
    warmup_started = time.perf_counter()
    for _ in range(warmup_iterations):
        invoke()
    synchronize()
    warmup_seconds = time.perf_counter() - warmup_started

    samples = []
    for _ in range(timed_iterations):
        synchronize()
        started = time.perf_counter()
        invoke()
        synchronize()
        samples.append(time.perf_counter() - started)
    median_seconds = float(__import__("statistics").median(samples))
    total_seconds = float(sum(samples))
    windows_per_second = float(batch_size / median_seconds)
    return {
        "first_call_seconds": float(first_seconds),
        "warmup_seconds": float(warmup_seconds),
        "median_seconds": median_seconds,
        "min_seconds": float(min(samples)),
        "max_seconds": float(max(samples)),
        "total_seconds": total_seconds,
        "windows_per_second": windows_per_second,
        "audio_seconds_per_second": float(windows_per_second * 2.0),
    }


def _runtime_identity(
    np: Any, torch: Any, psutil: Any, ov: Any | None
) -> dict[str, str]:
    result = {
        "python": platform.python_version(),
        "numpy": str(np.__version__),
        "torch": str(torch.__version__),
        "psutil": str(psutil.__version__),
    }
    if ov is not None:
        result["openvino"] = str(ov.__version__)
    return result


def _load_state(
    torch: Any, model_type: Any, checkpoint_path: Path
) -> tuple[Any, float, float]:
    try:
        construct_started = time.perf_counter()
        model = model_type()
        construct_seconds = float(time.perf_counter() - construct_started)
        checkpoint_started = time.perf_counter()
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        checkpoint_seconds = float(time.perf_counter() - checkpoint_started)
    except Exception as exc:
        raise _RouteError("runtime_failed", "checkpoint_load_failed") from exc
    return model, construct_seconds, checkpoint_seconds


def _xpu_device(torch: Any, index: int) -> Any:
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise _RouteError("unavailable", "torch_xpu_unavailable")
    try:
        count = int(torch.xpu.device_count())
    except Exception as exc:
        raise _RouteError("unavailable", "torch_xpu_unavailable") from exc
    if index < 0 or index >= count:
        raise _RouteError("unavailable", "torch_xpu_unavailable")
    return torch.device(f"xpu:{index}")


def _reset_xpu_peak_memory(torch: Any, device: Any) -> bool:
    try:
        torch.xpu.reset_peak_memory_stats(device)
    except (AttributeError, RuntimeError, TypeError):
        return False
    return True


def _sync_xpu(torch: Any, device: Any) -> None:
    if device.type == "xpu":
        torch.xpu.synchronize(device)


def _torch_outputs(
    torch: Any, model: Any, device: Any, host_batch: Any, *, inference: bool
) -> dict[str, Any]:
    batch = torch.from_numpy(host_batch)
    if device.type == "xpu":
        batch = batch.to(device)
    context = torch.inference_mode() if inference else torch.enable_grad()
    with context:
        values = model(batch)
    if not inference:
        return values
    _sync_xpu(torch, device)
    return {
        name: value.detach().to("cpu", dtype=torch.float32).numpy()
        for name, value in values.items()
    }


def _openvino_outputs(compiled: Any, host_batch: Any, np: Any) -> dict[str, Any]:
    result = compiled(host_batch)
    values = [
        np.asarray(result[output], dtype=np.float32) for output in compiled.outputs
    ]
    if len(values) != 3:
        raise _RouteError("runtime_failed", "benchmark_runtime_error")
    return dict(zip(("note", "onset", "contour"), values, strict=True))


def _openvino_property(
    target: Any, name: str, *, device: str | None = None
) -> Any | None:
    getter = getattr(target, "get_property", None)
    if not callable(getter):
        return None
    try:
        return getter(device, name) if device is not None else getter(name)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None


def _openvino_property_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _openvino_property_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_openvino_property_value(item) for item in value]
    text = str(value)
    lowered = text.lower()
    if "float32" in lowered:
        return "float32"
    if "float16" in lowered:
        return "float16"
    upper = text.upper()
    for name in ("PERFORMANCE", "LATENCY", "THROUGHPUT"):
        if name in upper:
            return name
    return text


def _openvino_device_info(core: Any, compiled: Any, target: str) -> dict[str, Any]:
    return {
        "selected_device": target,
        "available_devices": sorted(
            str(device) for device in getattr(core, "available_devices", ())
        ),
        "full_device_name": _openvino_property_value(
            _openvino_property(core, "FULL_DEVICE_NAME", device=target)
        ),
        "device_architecture": _openvino_property_value(
            _openvino_property(core, "DEVICE_ARCHITECTURE", device=target)
        ),
        "execution_devices": _openvino_property_value(
            _openvino_property(compiled, "EXECUTION_DEVICES")
        ),
        "inference_precision_hint_requested": "float32",
        "inference_precision_hint_compiled": _openvino_property_value(
            _openvino_property(compiled, "INFERENCE_PRECISION_HINT")
        ),
        "execution_mode_hint_requested": "unconfigured (plugin default)",
        "execution_mode_hint_compiled": _openvino_property_value(
            _openvino_property(compiled, "EXECUTION_MODE_HINT")
        ),
        "performance_hint_compiled": _openvino_property_value(
            _openvino_property(compiled, "PERFORMANCE_HINT")
        ),
    }


def _openvino_memory_snapshot(core: Any, target: str) -> dict[str, Any]:
    statistics = _openvino_property(core, "GPU_MEMORY_STATISTICS", device=target)
    total_memory = _openvino_property(core, "GPU_DEVICE_TOTAL_MEM_SIZE", device=target)
    if not isinstance(statistics, Mapping):
        return {
            "openvino_gpu_memory_measurement_status": "unavailable",
            "openvino_gpu_memory_statistics_bytes": None,
            "openvino_gpu_memory_bytes": None,
            "openvino_gpu_total_memory_bytes": int(total_memory)
            if isinstance(total_memory, int)
            else None,
        }
    try:
        sanitized = {str(key): int(value) for key, value in statistics.items()}
    except (TypeError, ValueError):
        return {
            "openvino_gpu_memory_measurement_status": "unavailable",
            "openvino_gpu_memory_statistics_bytes": None,
            "openvino_gpu_memory_bytes": None,
            "openvino_gpu_total_memory_bytes": int(total_memory)
            if isinstance(total_memory, int)
            else None,
        }
    return {
        "openvino_gpu_memory_measurement_status": "ok",
        "openvino_gpu_memory_statistics_bytes": sanitized,
        "openvino_gpu_memory_bytes": sum(sanitized.values()),
        "openvino_gpu_total_memory_bytes": int(total_memory)
        if isinstance(total_memory, int)
        else None,
    }


def _event_signature(event: Any) -> tuple[float, float, int]:
    return event.start_time_s, event.end_time_s, event.pitch_midi


def _candidate_parity(
    np: Any, reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    from obruxo_basic_pitch.constants import FRAME_THRESHOLD, ONSET_THRESHOLD
    from obruxo_basic_pitch.parity import OutputParity, ParitySummary, assert_parity
    from obruxo_basic_pitch.postprocess import posteriorgrams_to_note_events

    errors: dict[str, Any] = {}
    output_parity: dict[str, OutputParity] = {}
    for name in ("contour", "note", "onset"):
        candidate_values = np.asarray(candidate[name], dtype=np.float64)
        difference = candidate_values - np.asarray(reference[name], dtype=np.float64)
        errors[f"{name}_non_finite_count"] = int(
            np.count_nonzero(~np.isfinite(candidate_values))
        )
        if not np.isfinite(difference).all():
            errors[f"{name}_max_abs_error"] = None
            output_parity[name] = OutputParity(float("inf"), float("inf"), float("inf"))
            continue
        absolute = np.abs(difference)
        maximum = float(np.max(absolute))
        errors[f"{name}_max_abs_error"] = maximum
        output_parity[name] = OutputParity(
            maximum,
            float(np.mean(absolute)),
            float(np.sqrt(np.mean(difference * difference))),
        )

    note_disagreements = int(
        np.count_nonzero(
            (candidate["note"] >= FRAME_THRESHOLD)
            != (reference["note"] >= FRAME_THRESHOLD)
        )
    )
    onset_disagreements = int(
        np.count_nonzero(
            (candidate["onset"] >= ONSET_THRESHOLD)
            != (reference["onset"] >= ONSET_THRESHOLD)
        )
    )
    reference_events = []
    candidate_events = []
    for index in range(reference["note"].shape[0]):
        reference_events.extend(
            posteriorgrams_to_note_events(
                {name: value[index] for name, value in reference.items()}
            )
        )
        candidate_events.extend(
            posteriorgrams_to_note_events(
                {name: value[index] for name, value in candidate.items()}
            )
        )
    event_count_disagreements = int(len(reference_events) != len(candidate_events))
    if event_count_disagreements:
        event_tuple_disagreements = 0
    else:
        event_tuple_disagreements = sum(
            left != right
            for left, right in zip(
                map(_event_signature, reference_events),
                map(_event_signature, candidate_events),
                strict=True,
            )
        )
    structure_disagreements = event_count_disagreements or event_tuple_disagreements
    pitch_bend_disagreements = (
        sum(
            (left.pitch_bend or ()) != (right.pitch_bend or ())
            for left, right in zip(reference_events, candidate_events, strict=True)
        )
        if len(reference_events) == len(candidate_events)
        else max(len(reference_events), len(candidate_events))
    )
    summary = ParitySummary(
        contour=output_parity["contour"],
        note=output_parity["note"],
        onset=output_parity["onset"],
        note_threshold_disagreements=note_disagreements,
        onset_threshold_disagreements=onset_disagreements,
        note_threshold_elements=int(reference["note"].size),
        onset_threshold_elements=int(reference["onset"].size),
        onnx_event_count=len(reference_events),
        torch_event_count=len(candidate_events),
        event_structure_disagreements=int(structure_disagreements),
        amplitude_max_abs_error=None,
        amplitude_mean_abs_error=None,
        pitch_bend_element_disagreements=int(pitch_bend_disagreements),
        synthetic_windows=int(reference["note"].shape[0]),
    )
    try:
        assert_parity(summary)
    except AssertionError:
        parity_passed = False
    else:
        parity_passed = True
    return {
        **errors,
        "parity_passed": parity_passed,
        "event_count_disagreements": event_count_disagreements,
        "event_tuple_disagreements": int(event_tuple_disagreements),
        "note_threshold_disagreements": note_disagreements,
        "onset_threshold_disagreements": onset_disagreements,
        "event_structure_disagreements": structure_disagreements,
        "pitch_bend_element_disagreements": int(pitch_bend_disagreements),
        "reference_event_count": len(reference_events),
        "candidate_event_count": len(candidate_events),
    }


def _openvino_target(core: Any, route: str, requested: str) -> str:
    available = set(core.available_devices)
    if route == "openvino_cpu":
        if "CPU" not in available:
            raise _RouteError("unavailable", "openvino_cpu_unavailable")
        return "CPU"
    if requested == "GPU":
        concrete = sorted(device for device in available if device.startswith("GPU."))
        if len(concrete) > 1:
            raise _RouteError("unavailable", "openvino_gpu_device_ambiguous")
        if "GPU" not in available:
            raise _RouteError("unavailable", "openvino_gpu_unavailable")
        return "GPU"
    if not requested.startswith("GPU") or requested not in available:
        raise _RouteError("unavailable", "openvino_gpu_unavailable")
    return requested


def _build_openvino(
    torch: Any, ov: Any, model: Any, route: str, requested: str
) -> tuple[Any, float, float, dict[str, Any], Any, str]:
    core = ov.Core()
    target = _openvino_target(core, route, requested)
    example = torch.zeros((1, 43_844, 1), dtype=torch.float32)
    conversion_started = time.perf_counter()
    try:
        converted = ov.convert_model(model, example_input=example)
        converted.reshape({converted.inputs[0]: ov.PartialShape([-1, 43_844, 1])})
    except Exception as exc:
        raise _RouteError("runtime_failed", "openvino_conversion_failed") from exc
    conversion_seconds = time.perf_counter() - conversion_started
    compile_started = time.perf_counter()
    try:
        # The fixed benchmark contract is float32; do not accept the GPU
        # plugin's default reduced-precision inference hint.
        compiled = core.compile_model(
            converted,
            target,
            {ov.properties.hint.inference_precision: ov.Type.f32},
        )
    except Exception as exc:
        raise _RouteError("runtime_failed", "openvino_compile_failed") from exc
    return (
        compiled,
        float(conversion_seconds),
        float(time.perf_counter() - compile_started),
        _openvino_device_info(core, compiled, target),
        core,
        target,
    )


def _memory_snapshot(
    psutil: Any,
    torch: Any,
    device: Any | None,
    *,
    openvino_core: Any | None = None,
    openvino_target: str | None = None,
) -> dict[str, Any]:
    process_info = psutil.Process().memory_info()
    host_peak = getattr(process_info, "peak_wset", None)
    result: dict[str, Any] = {
        "measurement_status": "ok" if host_peak is not None else "unavailable",
        "host_peak_rss_bytes": int(host_peak) if host_peak is not None else None,
        "pytorch_xpu_peak_allocated_bytes": None,
        "pytorch_xpu_peak_reserved_bytes": None,
        "openvino_gpu_memory_bytes": None,
        "openvino_gpu_memory_measurement_status": "not_applicable",
        "openvino_gpu_memory_statistics_bytes": None,
        "openvino_gpu_total_memory_bytes": None,
    }
    if device is not None and device.type == "xpu":
        try:
            result["pytorch_xpu_peak_allocated_bytes"] = int(
                torch.xpu.max_memory_allocated(device)
            )
            result["pytorch_xpu_peak_reserved_bytes"] = int(
                torch.xpu.max_memory_reserved(device)
            )
        except (AttributeError, RuntimeError, TypeError):
            result["measurement_status"] = "unavailable"
    if openvino_core is not None and openvino_target is not None:
        result.update(_openvino_memory_snapshot(openvino_core, openvino_target))
    return result


def _training_call(torch: Any, model: Any, device: Any, host_batch: Any) -> None:
    model.zero_grad(set_to_none=True)
    values = _torch_outputs(torch, model, device, host_batch, inference=False)
    loss = values["note"].mean() + values["onset"].mean() + values["contour"].mean()
    if not bool(torch.isfinite(loss).item()):
        raise _TrainingError("training loss was non-finite")
    loss.backward()
    _sync_xpu(torch, device)
    for parameter in model.parameters():
        if parameter.grad is None or not bool(
            torch.isfinite(parameter.grad).all().item()
        ):
            raise _TrainingError("training gradient was non-finite or missing")


def _prepared_windows(cases: Any, prepare_wav: Any, np: Any) -> tuple[list[Any], Any]:
    prepared = [prepare_wav(case.audio_path) for case in cases]
    windows = np.ascontiguousarray(
        np.concatenate([item.windows for item in prepared], axis=0), dtype=np.float32
    )
    return prepared, windows


def _end_to_end(
    cases: Any,
    prepare_wav: Any,
    unwrap_window_outputs: Any,
    predict: Callable[[Any], Mapping[str, Any]],
    np: Any,
    postprocess: Any,
    synchronize: Callable[[], None] | None = None,
) -> dict[str, Any]:
    synchronize = synchronize or (lambda: None)
    rows = []
    total_audio_seconds = 0.0
    total_wall_seconds = 0.0
    for case in cases:
        synchronize()
        started = time.perf_counter()
        prepared = prepare_wav(case.audio_path)
        chunks = []
        for start in range(0, prepared.windows.shape[0], 1):
            chunks.append(predict(prepared.windows[start : start + 1]))
        outputs = {
            name: np.concatenate([chunk[name] for chunk in chunks], axis=0)
            for name in ("note", "onset", "contour")
        }
        unwrapped = unwrap_window_outputs(
            outputs, original_sample_count=prepared.original_sample_count
        )
        events = [] if unwrapped["note"].shape[0] == 0 else postprocess(unwrapped)
        synchronize()
        wall_seconds = float(time.perf_counter() - started)
        total_audio_seconds += prepared.audio_seconds
        total_wall_seconds += wall_seconds
        rows.append(
            {
                "case_index": case.case_index,
                "status": "ok",
                "audio_seconds": float(prepared.audio_seconds),
                "wall_seconds": wall_seconds,
                "note_event_count": len(events),
            }
        )
    return {
        "audio_seconds": float(total_audio_seconds),
        "wall_seconds": float(total_wall_seconds),
        "audio_seconds_per_wall_second": float(total_audio_seconds / total_wall_seconds)
        if total_wall_seconds
        else 0.0,
        "cases": rows,
    }


def _run(request: Mapping[str, Any]) -> dict[str, Any]:
    runtime_import_started = time.perf_counter()
    try:
        import numpy as np
        import psutil
        import torch

        from obruxo_basic_pitch.inference import prepare_wav, unwrap_window_outputs
        from obruxo_basic_pitch.model import BasicPitchICASSP2022
        from obruxo_basic_pitch.parity import synthetic_windows
        from obruxo_basic_pitch.postprocess import posteriorgrams_to_note_events
    except (AttributeError, ImportError, OSError, RuntimeError, ValueError):
        return {"status": "runtime_failed", "failure_code": "benchmark_runtime_error"}
    runtime_import_seconds = float(time.perf_counter() - runtime_import_started)
    route = str(request.get("route", ""))
    mode = str(request.get("mode", ""))
    parity: dict[str, Any] | None = None
    try:
        parity_only = bool(request.get("parity_only", False))
        if not parity_only:
            from obruxo_basic_pitch.benchmark import BenchmarkConfig, load_manifest

            config_data = request["config"]
            config = BenchmarkConfig(
                version=1,
                precision="float32",
                process_repetitions=3,
                warmup_iterations=int(config_data["warmup_iterations"]),
                timed_iterations=int(config_data["timed_iterations"]),
                batch_sizes=tuple(int(value) for value in config_data["batch_sizes"]),
                end_to_end_batch_size=int(config_data["end_to_end_batch_size"]),
                smoke_min_cases=8,
                smoke_max_cases=12,
                coverage={},
            )
            cases = load_manifest(
                request["manifest_path"], config, allow_derived_render=True
            )
        checkpoint_path = Path(str(request["checkpoint_path"])).resolve(strict=True)
        model, model_construct_seconds, checkpoint_load_seconds = _load_state(
            torch, BasicPitchICASSP2022, checkpoint_path
        )
        model.eval()
        route_device = None
        xpu_peak_reset = None
        device_move_seconds = 0.0
        openvino_conversion_seconds = 0.0
        openvino_compile_seconds = 0.0
        openvino_device_info = None
        openvino_core = None
        openvino_target = None
        ov = None
        if route == "pytorch_cpu":
            route_device = torch.device("cpu")
        elif route == "pytorch_xpu":
            route_device = _xpu_device(torch, int(request.get("xpu_index", 0)))
            xpu_peak_reset = _reset_xpu_peak_memory(torch, route_device)
            _sync_xpu(torch, route_device)
            move_started = time.perf_counter()
            model.to(route_device)
            _sync_xpu(torch, route_device)
            device_move_seconds = float(time.perf_counter() - move_started)
        elif route in {"openvino_cpu", "openvino_gpu"}:
            try:
                import openvino as ov
            except Exception as exc:
                code = (
                    "openvino_cpu_unavailable"
                    if route == "openvino_cpu"
                    else "openvino_gpu_unavailable"
                )
                raise _RouteError("unavailable", code) from exc
            (
                compiled,
                openvino_conversion_seconds,
                openvino_compile_seconds,
                openvino_device_info,
                openvino_core,
                openvino_target,
            ) = _build_openvino(
                torch, ov, model, route, str(request.get("openvino_gpu_device", "GPU"))
            )
        else:
            raise _RouteError("runtime_failed", "benchmark_runtime_error")

        def predict(host_batch: Any) -> Mapping[str, Any]:
            if route in {"pytorch_cpu", "pytorch_xpu"}:
                return _torch_outputs(
                    torch, model, route_device, host_batch, inference=True
                )
            return _openvino_outputs(compiled, host_batch, np)

        public_windows = np.ascontiguousarray(synthetic_windows(), dtype=np.float32)
        canonical_model, _, _ = _load_state(
            torch, BasicPitchICASSP2022, checkpoint_path
        )
        canonical_model.eval()
        canonical = _torch_outputs(
            torch, canonical_model, torch.device("cpu"), public_windows, inference=True
        )
        candidate = predict(public_windows)
        parity = _candidate_parity(np, canonical, candidate)
        if parity_only:
            return {
                "route": route,
                "mode": mode,
                "repetition": int(request.get("repetition", 0)),
                "status": "ok" if parity["parity_passed"] else "parity_failed",
                "failure_code": None if parity["parity_passed"] else "parity_failed",
                "runtime": _runtime_identity(np, torch, psutil, ov),
                "parity_windows": int(public_windows.shape[0]),
                "parity": parity,
                "backend": openvino_device_info,
            }
        if not parity["parity_passed"]:
            raise _ParityError(
                "candidate parity changed threshold or stock note-event behavior"
            )

        memory_device = (
            route_device if route in {"pytorch_cpu", "pytorch_xpu"} else None
        )
        _prepared, windows = _prepared_windows(cases, prepare_wav, np)
        source_key = "model_only" if mode == "inference" else "training"
        if mode not in {"inference", "training"}:
            raise _RouteError("runtime_failed", "benchmark_runtime_error")
        batch_results: dict[str, Any] = {}
        synchronize = lambda: (
            _sync_xpu(torch, route_device) if route_device is not None else None
        )
        for batch_size in config.batch_sizes:
            host_batch = np.ascontiguousarray(windows[:batch_size], dtype=np.float32)
            if host_batch.shape[0] != batch_size:
                raise _RouteError("runtime_failed", "benchmark_runtime_error")
            if source_key == "model_only":
                invoke = lambda host_batch=host_batch: predict(host_batch)
            else:
                if route == "openvino_cpu" or route == "openvino_gpu":
                    raise _RouteError("runtime_failed", "benchmark_runtime_error")
                model.train()
                invoke = lambda host_batch=host_batch: _training_call(
                    torch, model, route_device, host_batch
                )
            measurement = _measure_calls(
                invoke,
                batch_size=batch_size,
                warmup_iterations=config.warmup_iterations,
                timed_iterations=config.timed_iterations,
                synchronize=synchronize,
            )
            if mode == "inference":
                measurement["first_inference_seconds"] = measurement.pop(
                    "first_call_seconds"
                )
                measurement["inference_warmup_seconds"] = measurement.pop(
                    "warmup_seconds"
                )
            else:
                measurement["first_training_step_seconds"] = measurement.pop(
                    "first_call_seconds"
                )
                measurement["training_warmup_seconds"] = measurement.pop(
                    "warmup_seconds"
                )
            batch_results[str(batch_size)] = measurement
        end_to_end = None
        if mode == "inference":
            end_to_end = _end_to_end(
                cases,
                prepare_wav,
                unwrap_window_outputs,
                predict,
                np,
                posteriorgrams_to_note_events,
                synchronize=synchronize,
            )
        memory = _memory_snapshot(
            psutil,
            torch,
            memory_device,
            openvino_core=openvino_core,
            openvino_target=openvino_target,
        )
        memory["xpu_peak_reset_before_move"] = xpu_peak_reset
        return {
            "route": route,
            "mode": mode,
            "status": "ok",
            "failure_code": None,
            "runtime": _runtime_identity(np, torch, psutil, ov),
            "backend": openvino_device_info,
            "startup": {
                "backend_import_seconds": runtime_import_seconds,
                "model_construct_seconds": model_construct_seconds,
                "checkpoint_load_seconds": checkpoint_load_seconds,
                "model_device_move_seconds": device_move_seconds
                if route == "pytorch_xpu"
                else None,
                "openvino_conversion_seconds": openvino_conversion_seconds
                if route.startswith("openvino_")
                else None,
                "openvino_compile_seconds": openvino_compile_seconds
                if route.startswith("openvino_")
                else None,
                "measurement_status": {
                    "model_device_move_seconds": "ok"
                    if route == "pytorch_xpu"
                    else "not_applicable",
                    "openvino_conversion_seconds": "ok"
                    if route.startswith("openvino_")
                    else "not_applicable",
                    "openvino_compile_seconds": "ok"
                    if route.startswith("openvino_")
                    else "not_applicable",
                },
            },
            "parity": parity,
            source_key: {"batch_sizes": batch_results},
            "end_to_end": end_to_end,
            "memory": memory,
        }
    except _RouteError as exc:
        return {
            "route": route,
            "mode": mode,
            "status": exc.status,
            "failure_code": exc.code,
        }
    except _ParityError:
        result = {
            "route": route,
            "mode": mode,
            "status": "parity_failed",
            "failure_code": "parity_failed",
        }
        if parity is not None:
            result["parity"] = parity
        return result
    except _TrainingError:
        return {
            "route": route,
            "mode": mode,
            "status": "runtime_failed",
            "failure_code": "non_finite_training_step",
        }
    except RuntimeError as exc:
        status = (
            "out_of_memory" if "out of memory" in str(exc).lower() else "runtime_failed"
        )
        code = (
            "out_of_memory" if status == "out_of_memory" else "benchmark_runtime_error"
        )
        return {"route": route, "mode": mode, "status": status, "failure_code": code}
    except (AttributeError, ImportError, KeyError, OSError, TypeError, ValueError):
        return {
            "route": route,
            "mode": mode,
            "status": "runtime_failed",
            "failure_code": "benchmark_runtime_error",
        }


def main() -> int:
    try:
        request = json.load(sys.stdin)
        result = _run(request)
    except (AttributeError, ImportError, KeyError, OSError, TypeError, ValueError):
        result = {"status": "runtime_failed", "failure_code": "benchmark_runtime_error"}
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

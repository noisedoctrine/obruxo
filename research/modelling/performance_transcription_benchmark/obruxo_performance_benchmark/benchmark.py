"""Fixed #24-style cost accounting for executable candidates."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .artifacts import ModelSpec


def _basic_pitch_root() -> Path:
    return Path(__file__).resolve().parents[2] / "basic_pitch"


def fixed_timing_contract() -> dict[str, Any]:
    """Expose the inherited #24 timing identity without changing its settings."""
    import sys

    root = _basic_pitch_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from obruxo_basic_pitch.benchmark import load_config

    config = load_config(root / "configs" / "backend_benchmark.yaml")
    return {
        "benchmark_spec_version": config.version,
        "precision": config.precision,
        "process_repetitions": config.process_repetitions,
        "warmup_iterations": config.warmup_iterations,
        "timed_iterations": config.timed_iterations,
        "batch_sizes": list(config.batch_sizes),
        "end_to_end_batch_size": config.end_to_end_batch_size,
        "routes": ["pytorch_cpu", "pytorch_xpu"],
        "fresh_process_per_repetition": True,
        "synchronization": "explicit_backend_synchronization",
        "timing_boundary": "open_wav_to_normalized_transcription_output",
    }


def aggregate_measurements(values: Sequence[float]) -> dict[str, float]:
    """Call #24's fixed median/min/max arithmetic directly."""
    import sys

    root = _basic_pitch_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from obruxo_basic_pitch.benchmark import aggregate_measurements as inherited

    return inherited(values)


def measure_callable(call: Callable[[], Any], *, repetitions: int = 10) -> dict[str, Any]:
    """Measure a supplied synthetic/native call with inherited descriptive stats."""
    import time

    if repetitions < 1:
        raise ValueError("at least one timed repetition is required")
    values = []
    for _ in range(repetitions):
        start = time.perf_counter()
        call()
        values.append(time.perf_counter() - start)
    return {"measurements": values, "summary": aggregate_measurements(values)}


def _approved_output(path: Path | str) -> Path:
    root = (Path(__file__).resolve().parents[1] / "outputs").resolve()
    candidate = Path(path).resolve(strict=False)
    if candidate == root or not candidate.is_relative_to(root):
        raise ValueError("benchmark output must be inside the ignored benchmark output area")
    return candidate


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _unavailable_routes(code: str, *, quantized: bool) -> list[dict[str, Any]]:
    if quantized:
        return [
            {"route": "pytorch_cpu", "status": "unavailable", "failure_code": code, "quantized": True},
            {"route": "pytorch_xpu", "status": "not_applicable", "failure_code": "quantized_cpu_only", "quantized": True},
        ]
    return [
        {"route": "pytorch_cpu", "status": "unavailable", "failure_code": code, "quantized": False},
        {"route": "pytorch_xpu", "status": "unavailable", "failure_code": code, "quantized": False},
    ]


def _unavailable_phases(code: str) -> dict[str, dict[str, Any]]:
    return {
        name: {"status": "unavailable", "seconds": None, "failure_code": code}
        for name in (
            "backend_import_seconds",
            "model_construct_seconds",
            "checkpoint_load_seconds",
            "model_device_move_seconds",
            "first_inference_seconds",
            "inference_warmup_seconds",
            "steady_state",
            "end_to_end",
        )
    }


def run_benchmark(
    spec: ModelSpec,
    adapter: object,
    smoke_manifest: Path,
    output_dir: Path,
    *,
    quantized: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Run the fixed route contract or persist an explicit unavailable result."""
    output = _approved_output(output_dir)
    if output.exists() and not force and any(output.iterdir()):
        raise FileExistsError("refusing to overwrite benchmark output without force=True")
    output.mkdir(parents=True, exist_ok=True)
    manifest = Path(smoke_manifest).resolve(strict=False)
    contract = fixed_timing_contract()
    _atomic_json(
        output / "model_lock.json",
        {
            "format_version": 1,
            "model_id": spec.model_id,
            "model_identity": spec.identity_digest(),
            "variant_id": "dynamic_int8_linear" if quantized else "full_precision",
            "adapter_identity": f"{type(adapter).__module__}.{type(adapter).__qualname__}",
            "stock_inference": dict(spec.stock_inference),
        },
    )
    code = "dependency_unavailable"
    case_count = 0
    if not manifest.is_file():
        code = "smoke_manifest_missing"
    elif not spec.is_available:
        code = "dependency_unavailable"
    else:
        import sys

        root = _basic_pitch_root()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from obruxo_basic_pitch.benchmark import load_config, load_manifest

        try:
            config = load_config(root / "configs" / "backend_benchmark.yaml")
            cases = load_manifest(manifest, config, allow_derived_render=True, allow_missing_derived_audio=True)
            case_count = len(cases)
            if any(not case.audio_path.is_file() for case in cases):
                code = "derived_audio_unavailable"
            else:
                load = getattr(adapter, "load", None)
                if callable(load):
                    load()
                code = "candidate_runtime_unimplemented"
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            code = type(exc).__name__.casefold()
    result = {
        "format_version": 1,
        "status": "unavailable" if not spec.is_available or not manifest.is_file() or code == "derived_audio_unavailable" else "failed",
        "failure_code": code,
        "model_id": spec.model_id,
        "variant_id": "dynamic_int8_linear" if quantized else "full_precision",
        "model_identity": spec.identity_digest(),
        "smoke_manifest_present": manifest.is_file(),
        "smoke_case_count": case_count,
        "timing_contract": contract,
        "routes": _unavailable_routes(code, quantized=quantized),
        "phases": _unavailable_phases(code),
        "resources": {"parameters": None, "trainable_parameters": None, "state_tensor_bytes_by_dtype": None, "checkpoint_bytes": None, "peak_rss_bytes": None, "xpu_memory_bytes": None},
        "native_batch_sizes": list(spec.native_batch_sizes),
        "backward": {"status": "not_applicable", "reason": "no executable native differentiable boundary"},
        "batch_scaling": {str(size): {"status": "not_applicable", "reason": "candidate unavailable"} for size in (1, 2, 4, 8)},
    }
    _atomic_json(output / "runtime.json", result)
    return result

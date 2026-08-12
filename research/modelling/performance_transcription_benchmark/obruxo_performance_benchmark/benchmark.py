"""Fixed #24-style cost accounting for executable candidates."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
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


def _run_candidate_worker(request: Mapping[str, Any]) -> dict[str, Any]:
    worker = Path(__file__).with_name("candidate_benchmark_worker.py")
    completed = subprocess.run(
        [sys.executable, str(worker)],
        input=json.dumps(dict(request), sort_keys=True),
        capture_output=True,
        text=True,
        check=False,
    )
    for line in reversed(completed.stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {"status": "runtime_failed", "failure_code": "candidate_worker_failed"}


def _aggregate_candidate_route(rows: Sequence[Mapping[str, Any]], *, route: str, native_batch_sizes: Sequence[int]) -> dict[str, Any]:
    successful = [row for row in rows if row.get("status") == "ok"]
    if len(successful) != len(rows):
        failure = next((row for row in rows if row.get("status") != "ok"), {})
        return {
            "route": route,
            "status": str(failure.get("status", "runtime_failed")),
            "failure_code": failure.get("failure_code", "candidate_worker_failed"),
            "repetitions": list(rows),
            "native_batch_sizes": list(native_batch_sizes),
        }
    e2e_rows: list[dict[str, Any]] = []
    for case_index in sorted({int(case["case_index"]) for row in successful for case in row["end_to_end"]["cases"]}):
        cases = [case for row in successful for case in row["end_to_end"]["cases"] if int(case["case_index"]) == case_index]
        audio_values = {float(case["audio_seconds"]) for case in cases}
        if len(audio_values) != 1:
            raise ValueError("candidate smoke audio duration changed across repetitions")
        wall_summary = aggregate_measurements([float(case["wall_seconds"]) for case in cases])
        e2e_rows.append(
            {
                "case_index": case_index,
                "audio_seconds": float(next(iter(audio_values))),
                "wall_seconds": wall_summary,
                "status": "ok",
            }
        )
    total_audio = float(sum(float(case["audio_seconds"]) for case in e2e_rows))
    total_wall = aggregate_measurements([float(row["end_to_end"]["wall_seconds"]) for row in successful])
    memories = [row.get("resources", {}) for row in successful]
    memory = dict(memories[0])
    for field in ("checkpoint_bytes", "parameters", "trainable_parameters", "state_tensor_bytes_by_dtype"):
        values = [item.get(field) for item in memories if item.get(field) is not None]
        if values and any(value != values[0] for value in values[1:]):
            raise ValueError(f"candidate resource invariant changed across repetitions: {field}")
        if values:
            memory[field] = values[0]
    for field in ("host_peak_rss_bytes",):
        values = [float(item[field]) for item in memories if item.get(field) is not None]
        if values:
            memory[field] = aggregate_measurements(values)
    xpu_values = [item.get("xpu_memory") for item in memories if item.get("xpu_memory") is not None]
    if xpu_values:
        if not all(isinstance(value, Mapping) for value in xpu_values):
            raise ValueError("candidate XPU memory measurements have inconsistent shapes")
        memory["xpu_memory"] = {
            key: aggregate_measurements([float(value[key]) for value in xpu_values if key in value])
            for key in sorted({key for value in xpu_values for key in value})
        }
    return {
        "route": route,
        "status": "ok",
        "failure_code": None,
        "repetitions": list(rows),
        "runtime": successful[0].get("runtime"),
        "startup": {
            "model_load_seconds": aggregate_measurements([float(row["startup"]["model_load_seconds"]) for row in successful]),
            "first_call_seconds": aggregate_measurements([float(row["startup"]["first_call_seconds"]) for row in successful]),
            "warmup_seconds": aggregate_measurements([float(row["startup"]["warmup_seconds"]) for row in successful]),
        },
        "steady_state": {
            "seconds": aggregate_measurements([float(row["steady_state"]["seconds"]["median"]) for row in successful]),
            "calls_per_second": aggregate_measurements([float(row["steady_state"]["calls_per_second"]) for row in successful]),
        },
        "batch_scaling": successful[0].get("batch_scaling", {}),
        "end_to_end": {
            "status": "measured",
            "audio_seconds": total_audio,
            "wall_seconds": total_wall,
            "audio_seconds_per_wall_second": total_audio / total_wall["median"] if total_wall["median"] else 0.0,
            "cases": e2e_rows,
        },
        "resources": memory,
        "backward": successful[0].get("backward", {}),
        "native_batch_sizes": list(native_batch_sizes),
        "quantization": successful[0].get("quantization"),
    }


def _source_stat_snapshot(cases: Sequence[Any]) -> dict[str, tuple[int, int]]:
    records: dict[str, tuple[int, int]] = {}
    for case in cases:
        for path in (case.audio_path, case.midi_path, case.preset_path):
            if path is None:
                continue
            candidate = Path(path).resolve(strict=True)
            stat = candidate.stat()
            records[str(candidate)] = (int(stat.st_size), int(stat.st_mtime_ns))
    return records


def _landed_basic_pitch_runtime(smoke_manifest: Path, spec: ModelSpec) -> tuple[dict[str, Any], int]:
    root = _basic_pitch_root()
    report_path = root / "reports" / "backend_benchmark.json"
    if not report_path.is_file():
        raise FileNotFoundError("landed Basic Pitch #24 benchmark report is unavailable")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("model_id") != "spotify-basic-pitch-icassp-2022-v0.4.0":
        raise ValueError("landed Basic Pitch #24 benchmark identity is invalid")
    from obruxo_basic_pitch.benchmark import load_config, load_manifest

    config = load_config(root / "configs" / "backend_benchmark.yaml")
    cases = load_manifest(smoke_manifest, config, allow_derived_render=True, allow_missing_derived_audio=True)
    smoke_set = report.get("smoke_set", {})
    if not isinstance(smoke_set, dict) or smoke_set.get("status") != "ok" or int(smoke_set.get("case_count", -1)) != len(cases):
        raise ValueError("landed Basic Pitch #24 smoke workload does not match the supplied manifest")
    if report.get("source_git_blob_sha1") != "c30e5f9438e798604b7177aa26be1fe64482f767":
        raise ValueError("landed Basic Pitch #24 benchmark source identity is invalid")
    return report, len(cases)


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
    if spec.model_id == "basic_pitch" and not quantized:
        try:
            report, case_count = _landed_basic_pitch_runtime(manifest, spec)
        except (FileNotFoundError, ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            result = {
                "format_version": 1,
                "status": "unavailable",
                "failure_code": "dependency_unavailable",
                "model_id": spec.model_id,
                "variant_id": "full_precision",
                "model_identity": spec.identity_digest(),
                "smoke_manifest_present": manifest.is_file(),
                "smoke_case_count": 0,
                "source": "landed_issue_24_report",
                "reason": type(exc).__name__,
                "timing_contract": contract,
                "routes": _unavailable_routes("dependency_unavailable", quantized=False),
                "phases": _unavailable_phases("dependency_unavailable"),
                "resources": {},
                "native_batch_sizes": list(spec.native_batch_sizes),
                "backward": {"status": "not_applicable", "reason": "landed route unavailable"},
                "batch_scaling": {},
            }
        else:
            routes = list(report.get("inference", [])) + list(report.get("training", []))
            result = {
                "format_version": 1,
                "status": "ok",
                "failure_code": None,
                "model_id": spec.model_id,
                "variant_id": "full_precision",
                "model_identity": spec.identity_digest(),
                "smoke_manifest_present": manifest.is_file(),
                "smoke_case_count": case_count,
                "source": "landed_issue_24_report",
                "timing_contract": report.get("config") or contract,
                "routes": routes,
                "inference": report.get("inference", []),
                "training": report.get("training", []),
                "smoke_set": report.get("smoke_set"),
                "conclusions": report.get("conclusions", {}),
                "resources": {},
                "native_batch_sizes": list(spec.native_batch_sizes),
                "backward": {"status": "inherited", "source": "landed_issue_24_report"},
                "batch_scaling": {},
            }
        _atomic_json(output / "runtime.json", result)
        return result
    code = "dependency_unavailable"
    case_count = 0
    routes: list[dict[str, Any]] = []
    route_failures: list[dict[str, Any]] = []
    quantization_info = None
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

        cases: Sequence[Any] = ()
        source_before: dict[str, tuple[int, int]] | None = None
        try:
            config = load_config(root / "configs" / "backend_benchmark.yaml")
            cases = load_manifest(manifest, config, allow_derived_render=True, allow_missing_derived_audio=False)
            case_count = len(cases)
            source_before = _source_stat_snapshot(cases)
            if any(not case.audio_path.is_file() for case in cases):
                code = "derived_audio_unavailable"
            else:
                source_root = getattr(adapter, "source_root", None) or root
                checkpoint = getattr(adapter, "checkpoint", None) or root / "artifacts" / "basic_pitch_icassp_2022.pt"
                for route in ("pytorch_cpu", "pytorch_xpu"):
                    if quantized and route != "pytorch_cpu":
                        routes.append({"route": route, "status": "not_applicable", "failure_code": "quantized_cpu_only", "quantized": True})
                        continue
                    repetitions = []
                    for repetition in range(config.process_repetitions):
                        repetitions.append(
                            _run_candidate_worker(
                                {
                                    "config_path": str(Path(__file__).resolve().parents[1] / "config" / "models.yaml"),
                                    "model_id": spec.model_id,
                                    "source_root": str(Path(source_root).resolve()),
                                    "checkpoint": str(Path(checkpoint).resolve()),
                                    "smoke_manifest": str(manifest),
                                    "route": route,
                                    "repetition": repetition,
                                    "quantized": quantized,
                                }
                            )
                        )
                    route_result = _aggregate_candidate_route(repetitions, route=route, native_batch_sizes=spec.native_batch_sizes)
                    routes.append(route_result)
                    if route_result.get("status") != "ok":
                        route_failures.append({"route": route, "status": route_result.get("status"), "failure_code": route_result.get("failure_code")})
                successful_routes = [route for route in routes if route.get("status") == "ok"]
                if successful_routes:
                    code = None
                    quantization_info = next((route.get("quantization") for route in successful_routes if route.get("quantization")), None)
                else:
                    code = route_failures[0].get("failure_code", "candidate_runtime_failed") if route_failures else "candidate_runtime_failed"
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            code = type(exc).__name__.casefold()
        finally:
            if source_before is not None:
                try:
                    source_after = _source_stat_snapshot(cases)
                except (OSError, RuntimeError, ValueError):
                    code = "source_audit_failed"
                    routes = []
                    route_failures = [{"route": "all", "status": "failed", "failure_code": code}]
                else:
                    if source_after != source_before:
                        code = "source_mutated"
                        routes = []
                        route_failures = [{"route": "all", "status": "failed", "failure_code": code}]
    status = "ok" if any(route.get("status") == "ok" for route in routes) else ("unavailable" if code in {"dependency_unavailable", "smoke_manifest_missing", "derived_audio_unavailable"} else "failed")
    result = {
        "format_version": 1,
        "status": status,
        "failure_code": code,
        "model_id": spec.model_id,
        "variant_id": "dynamic_int8_linear" if quantized else "full_precision",
        "model_identity": spec.identity_digest(),
        "smoke_manifest_present": manifest.is_file(),
        "smoke_case_count": case_count,
        "source": "fixed_issue_24_candidate_worker",
        "timing_contract": contract,
        "routes": routes or _unavailable_routes(code, quantized=quantized),
        "route_failures": route_failures,
        "phases": {},
        "resources": {},
        "native_batch_sizes": list(spec.native_batch_sizes),
        "backward": {"status": "not_applicable", "reason": "no executable natural differentiable boundary"},
        "batch_scaling": {},
        "quantization": quantization_info,
    }
    _atomic_json(output / "runtime.json", result)
    return result

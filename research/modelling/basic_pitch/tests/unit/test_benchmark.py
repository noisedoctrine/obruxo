from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from obruxo_basic_pitch.benchmark import (
    BenchmarkConfig,
    _aggregate_route,
    aggregate_measurements,
    benchmark_exit_code,
    crossover_audio_seconds,
    load_config,
    load_manifest,
    write_benchmark_reports,
)
from obruxo_basic_pitch.benchmark_worker import _openvino_target, _RouteError

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "backend_benchmark.yaml"


def test_fixed_benchmark_config_is_not_a_tuning_grid() -> None:
    config = load_config(CONFIG_PATH)
    assert config == BenchmarkConfig(
        version=1,
        precision="float32",
        process_repetitions=3,
        warmup_iterations=3,
        timed_iterations=10,
        batch_sizes=(1, 2, 4, 8),
        end_to_end_batch_size=1,
        smoke_min_cases=8,
        smoke_max_cases=12,
        coverage=config.coverage,
    )
    assert config.as_dict()["routes"] == {
        "inference": ["pytorch_cpu", "pytorch_xpu", "openvino_cpu", "openvino_gpu"],
        "training": ["pytorch_cpu", "pytorch_xpu"],
    }


def test_manifest_validation_and_sanitization() -> None:
    tmp_path = ROOT / "outputs" / f".test-manifest-{os.getpid()}"
    assert not tmp_path.exists()
    tmp_path.mkdir()
    cases = []
    try:
        for index in range(1, 9):
            audio = tmp_path / f"audio-{index}.wav"
            midi = tmp_path / f"midi-{index}.mid"
            audio.write_bytes(b"RIFF")
            midi.write_bytes(b"MThd")
            cases.append(
                {
                    "case_index": index,
                    "audio_path": str(audio),
                    "midi_path": str(midi),
                    "performance": "monophonic" if index < 5 else "polyphonic",
                    "role": "bass" if index == 1 else "other",
                    "envelope": "transient" if index < 5 else "sustained",
                    "duration_class": ("short", "medium", "long")[index % 3],
                    "note_density_class": "low" if index < 5 else "high",
                }
            )
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(
            json.dumps({"format_version": 1, "benchmark_spec_version": 1, "cases": cases}), encoding="utf-8"
        )
        manifest = load_manifest(manifest_path, load_config(CONFIG_PATH))
        assert len(manifest) == 8
        sanitized = manifest[0].sanitized()
        assert "audio_path" not in sanitized and "midi_path" not in sanitized
    finally:
        for path in tmp_path.glob("*"):
            path.unlink()
        tmp_path.rmdir()


def test_aggregate_measurements_and_crossover_formula() -> None:
    assert aggregate_measurements([1.0, 3.0, 2.0]) == {"median": 2.0, "min": 1.0, "max": 3.0, "total": 6.0}
    assert crossover_audio_seconds(1.0, 1.0, 3.0, 2.0) == 4.0
    assert crossover_audio_seconds(3.0, 1.0, 1.0, 2.0) is None


def test_openvino_gpu_never_falls_back_to_another_device() -> None:
    class Core:
        def __init__(self) -> None:
            self.available_devices = ["CPU", "GPU.0"]

    with pytest.raises(_RouteError) as error:
        _openvino_target(Core(), "openvino_gpu", "GPU")
    assert error.value.code == "openvino_gpu_unavailable"


def test_parity_failure_suppresses_timing_aggregation() -> None:
    config = load_config(CONFIG_PATH)
    result = _aggregate_route(
        [
            {"status": "parity_failed", "failure_code": "parity_failed"},
            {"status": "parity_failed", "failure_code": "parity_failed"},
            {"status": "parity_failed", "failure_code": "parity_failed"},
        ],
        "openvino_cpu",
        "inference",
        config,
    )
    assert result["status"] == "parity_failed"
    assert "batch_results" not in result
    assert benchmark_exit_code({"inference": [result], "training": []}) == 3


def test_unavailable_routes_are_not_failure_exit_code() -> None:
    assert benchmark_exit_code(
        {
            "inference": [{"status": "ok"}, {"status": "unavailable"}],
            "training": [{"status": "unavailable"}],
        }
    ) == 0


def test_report_write_is_atomic_and_requires_force_for_overwrite() -> None:
    report = {
        "format_version": 1,
        "benchmark_spec_version": 1,
        "model_id": "test-model",
        "runtime": {"python": "3.12"},
        "smoke_set": {"status": "unavailable"},
        "inference": [{"route": "pytorch_cpu", "status": "unavailable"}],
        "training": [{"route": "pytorch_cpu", "status": "unavailable"}],
    }
    suffix = str(os.getpid())
    json_path = ROOT / "reports" / f".test-backend-{suffix}.json"
    markdown_path = ROOT / "reports" / f".test-backend-{suffix}.md"
    assert not json_path.exists() and not markdown_path.exists()
    try:
        write_benchmark_reports(report, json_path, markdown_path)
        with pytest.raises(FileExistsError):
            write_benchmark_reports(report, json_path, markdown_path)
        write_benchmark_reports(report, json_path, markdown_path, force=True)
        assert "audio_path" not in json_path.read_text(encoding="utf-8")
    finally:
        for path in (json_path, markdown_path):
            if path.exists():
                path.unlink()

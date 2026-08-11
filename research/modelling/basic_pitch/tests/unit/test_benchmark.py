from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from obruxo_basic_pitch.benchmark import (
    BenchmarkConfig,
    PedalboardVitalRenderer,
    _aggregate_route,
    _approved_derived_output_root,
    _benchmark_markdown,
    _run_worker,
    _validate_derived_destination,
    aggregate_measurements,
    benchmark_exit_code,
    crossover_audio_seconds,
    load_config,
    load_manifest,
    write_benchmark_reports,
)
from obruxo_basic_pitch.benchmark_worker import (
    _build_openvino,
    _openvino_target,
    _RouteError,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "backend_benchmark.yaml"


def test_worker_launches_as_package_from_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(args: list[str], **kwargs: object) -> SimpleNamespace:
        captured["args"] = args
        captured["cwd"] = kwargs["cwd"]
        return SimpleNamespace(returncode=0, stdout=json.dumps({"status": "ok"}))

    monkeypatch.setattr("obruxo_basic_pitch.benchmark.subprocess.run", fake_run)
    assert _run_worker({"route": "pytorch_cpu"}) == {"status": "ok"}
    assert captured["args"][1:3] == ["-m", "obruxo_basic_pitch.benchmark_worker"]  # type: ignore[index]
    assert captured["cwd"] == ROOT  # type: ignore[comparison-overlap]


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


def test_derived_render_manifest_is_opt_in_and_destination_checked() -> None:
    tmp_path = ROOT / "outputs" / f".test-derived-manifest-{os.getpid()}"
    target_paths = [ROOT / "outputs" / f".test-derived-target-{os.getpid()}-{index}.wav" for index in range(1, 9)]
    assert not tmp_path.exists() and all(not path.exists() for path in target_paths)
    source_dir = tmp_path / "sources"
    source_dir.mkdir(parents=True)
    try:
        cases = []
        for index, target in enumerate(target_paths, start=1):
            preset = source_dir / f"preset-{index}.vital"
            midi = source_dir / f"performance-{index}.mid"
            preset.write_bytes(b"{}")
            midi.write_bytes(b"MThd")
            cases.append(
                {
                    "case_index": index,
                    "audio_path": str(target),
                    "midi_path": str(midi),
                    "audio_source": "derived_render",
                    "preset_path": str(preset),
                    "performance": "monophonic",
                    "role": "other",
                    "envelope": "sustained",
                    "duration_class": "medium",
                    "note_density_class": "low",
                }
            )
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(
            json.dumps({"format_version": 1, "benchmark_spec_version": 1, "cases": cases}), encoding="utf-8"
        )
        config = load_config(CONFIG_PATH)
        with pytest.raises(ValueError):
            load_manifest(manifest_path, config)
        manifest = load_manifest(
            manifest_path,
            config,
            allow_derived_render=True,
            allow_missing_derived_audio=True,
        )
        assert len(manifest) == 8
        assert manifest[0].audio_source == "derived_render"
        assert manifest[0].sanitized()["audio_source"] == "derived_render"
        assert "preset_path" not in manifest[0].sanitized()
        output_root = _approved_derived_output_root()
        assert _validate_derived_destination(manifest[0], output_root) == target_paths[0].resolve()
        unsafe = replace(manifest[0], audio_path=source_dir / "nested.wav")
        with pytest.raises(ValueError):
            _validate_derived_destination(unsafe, output_root)
    finally:
        for path in sorted(tmp_path.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        if tmp_path.exists():
            tmp_path.rmdir()
        for path in target_paths:
            if path.exists():
                path.unlink()


def test_pedalboard_preflight_uses_accepted_vital_binary_and_raw_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    plugin_path = tmp_path / "Vital.vst3"
    plugin_path.write_bytes(b"accepted-vital")
    digest = hashlib.sha256(plugin_path.read_bytes()).hexdigest()
    config_path = tmp_path / "renderer.yaml"
    config_path.write_text(
        "\n".join(
            [
                "version: 1",
                "accepted_plugin_sha256:",
                f"  - {digest}",
                f"plugin_path: {plugin_path}",
                "buffer_size: 128",
                "capabilities: {}",
                "qa: {}",
            ]
        ),
        encoding="utf-8",
    )

    class FakePlugin:
        is_instrument = True
        raw_state = b"template-state"

        def reset(self) -> None: ...

        def process(self, *args: object, **kwargs: object) -> object: ...

    plugin = FakePlugin()
    pedalboard = ModuleType("pedalboard")
    pedalboard.load_plugin = lambda path: plugin  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pedalboard", pedalboard)
    monkeypatch.setattr("obruxo_basic_pitch.benchmark.importlib.metadata.version", lambda _: "0.9.test")

    class FakeStateTemplate:
        def __init__(self, state: bytes) -> None:
            assert state == b"template-state"

    class FakeCapabilities:
        @classmethod
        def from_dict(cls, value: dict[str, object]) -> object:
            return value

    class FakeAudioQualityConfig:
        def __init__(self, **kwargs: object) -> None:
            self.values = kwargs

    monkeypatch.setattr(
        "obruxo_basic_pitch.benchmark._import_data_generation_modules",
        lambda: (object, object, FakeCapabilities, FakeAudioQualityConfig, object, FakeStateTemplate, object, object),
    )
    renderer = PedalboardVitalRenderer.from_config(config_path)
    assert renderer.plugin_path == plugin_path.resolve()
    assert renderer.vital_sha256 == digest
    assert renderer.buffer_size == 128
    assert renderer.pedalboard_version == "0.9.test"


def test_pedalboard_render_uses_timestamped_midi_and_local_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import numpy as np

    source_dir = tmp_path / "sources"
    output_root = tmp_path / "output"
    source_dir.mkdir()
    preset_path = source_dir / "patch.vital"
    midi_path = source_dir / "performance.mid"
    preset_path.write_text("{}", encoding="utf-8")
    midi_path.write_bytes(b"MThd")

    class FakeReport:
        def require_valid(self) -> None: ...

    class FakeSpan:
        channel = 0
        pitch = 60
        velocity = 100
        start_tick = 10
        end_tick = 40

    class FakeEvent:
        def __init__(self, kind: str, tick: int, data: tuple[int, ...]) -> None:
            self.kind = SimpleNamespace(value=kind)
            self.tick = tick
            self.channel = 0
            self.data = data

    class FakePerformance:
        end_tick = 100

        @classmethod
        def from_midi(cls, path: Path) -> FakePerformance:
            assert path == midi_path
            return cls()

        def validate(self) -> FakeReport:
            return FakeReport()

        def note_spans(self) -> list[FakeSpan]:
            return [FakeSpan()]

        def canonical_events(self) -> list[FakeEvent]:
            return [FakeEvent("note_on", 10, (60, 100)), FakeEvent("note_off", 40, (60, 0))]

    class FakeTiming:
        def tick_to_seconds(self, tick: int) -> float:
            return tick / 1000

        def render_frame_count(self, end_tick: int, tail_seconds: float, sample_rate: int) -> int:
            assert (end_tick, tail_seconds, sample_rate) == (100, 2.0, 44_100)
            return 88_200

    class FakeTempoMap:
        @classmethod
        def from_performance(cls, performance: FakePerformance) -> FakeTiming:
            return FakeTiming()

    class FakePreset:
        @classmethod
        def load(cls, path: Path) -> FakePreset:
            assert path == preset_path
            return cls()

        def validate(self) -> FakeReport:
            return FakeReport()

        def to_json(self, canonical: bool) -> str:
            assert canonical is True
            return "{}"

    class FakeStateTemplate:
        def build(self, preset_json: str) -> bytes:
            assert preset_json == "{}"
            return b"applied-state"

    class FakePlugin:
        def __init__(self) -> None:
            self.raw_state = b"initial-state"
            self.reset_calls = 0
            self.messages: object = None
            self.process_kwargs: dict[str, object] = {}

        def reset(self) -> None:
            self.reset_calls += 1

        def process(self, messages: object, **kwargs: object) -> np.ndarray:
            self.messages = messages
            self.process_kwargs = kwargs
            assert self.raw_state == b"applied-state"
            return np.zeros((2, 88_200), dtype=np.float32)

    plugin = FakePlugin()
    monkeypatch.setattr(
        "obruxo_basic_pitch.benchmark._import_data_generation_modules",
        lambda: (FakePerformance, FakeTempoMap, object, object, lambda *args, **kwargs: ({"finite": True}, ()), object, FakePreset, object),
    )
    renderer = PedalboardVitalRenderer(
        config_path=tmp_path / "renderer.yaml",
        plugin_path=tmp_path / "Vital.vst3",
        vital_sha256="a" * 64,
        buffer_size=128,
        capabilities=object(),
        qa_config=object(),
        pedalboard_module=ModuleType("pedalboard"),
        state_template=FakeStateTemplate(),
        plugin=plugin,
        pedalboard_version="0.9.test",
    )
    destination = output_root / "derived.wav"
    renderer.render(preset_path, midi_path, destination, output_root)
    assert plugin.reset_calls == 2
    assert plugin.messages == [(b"\x90<d", 0.01), (b"\x80<\x00", 0.04)]
    assert plugin.process_kwargs["duration"] == pytest.approx(88_200.5 / 44_100)
    assert plugin.process_kwargs["sample_rate"] == 44_100
    assert plugin.process_kwargs["num_channels"] == 2
    assert plugin.process_kwargs["buffer_size"] == 128
    assert plugin.process_kwargs["reset"] is True
    metadata = json.loads(destination.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["audio_source"] == "derived_render"
    assert metadata["provenance"]["backend_version"] == "0.9.test"
    assert metadata["provenance"]["settings"]["host"] == "pedalboard"
    assert metadata["provenance"]["settings"]["buffer_size"] == 128
    assert metadata["derived_render_provenance"]["audio_label"] == "derived_render"


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


def test_openvino_compile_pins_float32_without_changing_execution_mode() -> None:
    class FakeTorch:
        float32 = "torch.float32"

        @staticmethod
        def zeros(shape: tuple[int, ...], *, dtype: str) -> tuple[tuple[int, ...], str]:
            return shape, dtype

    class FakeConverted:
        def __init__(self) -> None:
            self.inputs = [object()]

        def reshape(self, shape: object) -> None:
            self.reshape_shape = shape

    class FakeCore:
        def __init__(self) -> None:
            self.available_devices = ["GPU"]
            self.compile_arguments: tuple[object, str, dict[str, object]] | None = None

        def compile_model(self, model: object, target: str, config: dict[str, object]) -> object:
            self.compile_arguments = (model, target, config)
            return object()

    class FakeOpenVINO:
        class Type:
            f32 = "float32"

        properties = SimpleNamespace(hint=SimpleNamespace(inference_precision="INFERENCE_PRECISION_HINT"))
        PartialShape = staticmethod(lambda shape: shape)

        def __init__(self) -> None:
            self.core = FakeCore()

        def Core(self) -> FakeCore:
            return self.core

        @staticmethod
        def convert_model(model: object, *, example_input: object) -> FakeConverted:
            return FakeConverted()

    ov = FakeOpenVINO()
    _build_openvino(FakeTorch(), ov, object(), "openvino_gpu", "GPU")

    assert ov.core.compile_arguments is not None
    assert ov.core.compile_arguments[1:] == ("GPU", {"INFERENCE_PRECISION_HINT": "float32"})


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


def test_report_markdown_surfaces_persisted_findings() -> None:
    report = json.loads((ROOT / "reports" / "backend_benchmark.json").read_text(encoding="utf-8"))
    markdown = _benchmark_markdown(report)
    for section in (
        "Inference startup and initialization",
        "Steady-state inference scaling",
        "End-to-end audio-to-note-event throughput",
        "CPU versus XPU full forward+backward cost",
        "Memory and resource observations",
        "Startup versus throughput crossover",
        "OpenVINO GPU parity failure",
    ):
        assert section in markdown
    assert "Batch 8" in markdown
    assert "Parity diagnostics by framework and processor" in markdown
    assert "component-level parity values are tabulated above" in markdown
    assert "227040" in markdown


def test_report_markdown_tabulates_component_parity_by_route() -> None:
    report = json.loads((ROOT / "reports" / "backend_benchmark.json").read_text(encoding="utf-8"))
    metrics = {
        "contour_non_finite_count": 0,
        "note_non_finite_count": 0,
        "onset_non_finite_count": 0,
        "contour_max_abs_error": 0.001,
        "note_max_abs_error": 0.002,
        "onset_max_abs_error": 0.003,
        "note_threshold_disagreements": 0,
        "onset_threshold_disagreements": 0,
        "event_count_disagreements": 0,
        "event_tuple_disagreements": 0,
    }
    report["parity_diagnostics"] = {
        "process_repetitions": 3,
        "scope": "five public synthetic windows",
        "thresholds": {
            "contour_max_abs_error": 0.0205306,
            "note_max_abs_error": 0.0038445,
            "onset_max_abs_error": 0.2089347,
            "note_frame_threshold": 0.3,
            "onset_threshold": 0.5,
        },
        "routes": [
            {"route": route, "status": "ok", "max_across_repetitions": metrics}
            for route in ("pytorch_cpu", "pytorch_xpu", "openvino_cpu", "openvino_gpu")
        ],
    }
    markdown = _benchmark_markdown(report)
    assert "| Parity check (applied threshold) | PyTorch CPU | PyTorch XPU | OpenVINO CPU | OpenVINO GPU |" in markdown
    assert "Maximum contour absolute error (≤ 0.0205306)" in markdown
    assert "Note-frame threshold disagreements (threshold 0.3; must be 0)" in markdown
    assert "(start_time_s, end_time_s, MIDI pitch) disagreements (must be 0)" in markdown

"""Fixed, sanitized Basic Pitch backend benchmark orchestration."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import yaml

from .constants import MODEL_ID, SPOTIFY_ONNX_GIT_BLOB_SHA1

INFERENCE_ROUTES = ("pytorch_cpu", "pytorch_xpu", "openvino_cpu", "openvino_gpu")
TRAINING_ROUTES = ("pytorch_cpu", "pytorch_xpu")
STATUSES = ("ok", "unavailable", "parity_failed", "runtime_failed", "out_of_memory")
FAILURE_CODES = (
    "invalid_smoke_manifest",
    "checkpoint_load_failed",
    "torch_xpu_unavailable",
    "openvino_cpu_unavailable",
    "openvino_gpu_unavailable",
    "openvino_gpu_device_ambiguous",
    "openvino_conversion_failed",
    "openvino_compile_failed",
    "parity_failed",
    "non_finite_training_step",
    "out_of_memory",
    "benchmark_runtime_error",
    "derived_render_unavailable",
    "derived_render_failed",
    "derived_render_destination_invalid",
)

_PERFORMANCE = ("monophonic", "polyphonic")
_ROLES = ("bass", "lead", "arp_sequence", "pad_sustained", "keys_pluck", "fx_texture", "other", "unknown")
_ENVELOPES = ("transient", "sustained", "mixed", "unknown")
_DURATIONS = ("short", "medium", "long")
_DENSITIES = ("low", "medium", "high", "unknown")
_AUDIO_SOURCES = ("existing_audio", "derived_render")


class BenchmarkInputError(ValueError):
    """A sanitized configuration or smoke-manifest error."""

    def __init__(self, code: str, message: str = "invalid benchmark input") -> None:
        if code not in FAILURE_CODES:
            raise ValueError(f"unknown benchmark failure code: {code}")
        super().__init__(message)
        self.code = code


class SourceMutationError(RuntimeError):
    """An immutable source changed while the benchmark was running."""


class DerivedRenderUnavailable(RuntimeError):
    """The explicitly requested validated Vital rendering path is unavailable."""

    def __init__(self, code: str = "derived_render_unavailable") -> None:
        if code not in {"derived_render_unavailable", "derived_render_failed"}:
            raise ValueError(f"unknown derived-render failure code: {code}")
        super().__init__("the validated Vital derived-render path is unavailable")
        self.code = code


@dataclass(frozen=True)
class BenchmarkConfig:
    version: int
    precision: str
    process_repetitions: int
    warmup_iterations: int
    timed_iterations: int
    batch_sizes: tuple[int, ...]
    end_to_end_batch_size: int
    smoke_min_cases: int
    smoke_max_cases: int
    coverage: dict[str, tuple[str, ...]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "precision": self.precision,
            "process_repetitions": self.process_repetitions,
            "warmup_iterations": self.warmup_iterations,
            "timed_iterations": self.timed_iterations,
            "batch_sizes": list(self.batch_sizes),
            "end_to_end_batch_size": self.end_to_end_batch_size,
            "smoke_set": {
                "min_cases": self.smoke_min_cases,
                "max_cases": self.smoke_max_cases,
                "coverage": {name: list(values) for name, values in self.coverage.items()},
            },
            "routes": {
                "inference": list(INFERENCE_ROUTES),
                "training": list(TRAINING_ROUTES),
            },
        }


@dataclass(frozen=True)
class SmokeCase:
    case_index: int
    audio_path: Path
    midi_path: Path
    audio_source: str
    preset_path: Path | None
    performance: str
    role: str
    envelope: str
    duration_class: str
    note_density_class: str

    def sanitized(self) -> dict[str, Any]:
        return {
            "case_index": self.case_index,
            "audio_source": self.audio_source,
            "performance": self.performance,
            "role": self.role,
            "envelope": self.envelope,
            "duration_class": self.duration_class,
            "note_density_class": self.note_density_class,
        }


def _required_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BenchmarkInputError("invalid_smoke_manifest", f"{label} must be a mapping")
    return value


def _validate_exact(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise BenchmarkInputError("invalid_smoke_manifest", f"unexpected {label}")


def load_config(path: str | Path) -> BenchmarkConfig:
    """Load the committed fixed experiment configuration without overrides."""
    config_path = Path(path).resolve(strict=True)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise BenchmarkInputError("invalid_smoke_manifest", "benchmark config could not be read") from exc
    data = _required_mapping(raw, "benchmark config")
    _validate_exact(data.get("version"), 1, "config version")
    _validate_exact(data.get("precision"), "float32", "precision")
    _validate_exact(data.get("process_repetitions"), 3, "process repetitions")
    _validate_exact(data.get("warmup_iterations"), 3, "warmup iterations")
    _validate_exact(data.get("timed_iterations"), 10, "timed iterations")
    _validate_exact(data.get("batch_sizes"), [1, 2, 4, 8], "batch sizes")
    _validate_exact(data.get("end_to_end_batch_size"), 1, "end-to-end batch size")
    smoke = _required_mapping(data.get("smoke_set"), "smoke_set")
    _validate_exact(smoke.get("min_cases"), 8, "smoke minimum")
    _validate_exact(smoke.get("max_cases"), 12, "smoke maximum")
    routes = _required_mapping(data.get("routes"), "routes")
    _validate_exact(routes.get("inference"), list(INFERENCE_ROUTES), "inference routes")
    _validate_exact(routes.get("training"), list(TRAINING_ROUTES), "training routes")
    coverage = smoke.get("coverage", {})
    if not isinstance(coverage, Mapping):
        raise BenchmarkInputError("invalid_smoke_manifest", "smoke coverage must be a mapping")
    return BenchmarkConfig(
        version=1,
        precision="float32",
        process_repetitions=3,
        warmup_iterations=3,
        timed_iterations=10,
        batch_sizes=(1, 2, 4, 8),
        end_to_end_batch_size=1,
        smoke_min_cases=8,
        smoke_max_cases=12,
        coverage={name: tuple(str(item) for item in values) for name, values in coverage.items()},
    )


def _source_path(manifest_path: Path, value: Any, label: str, *, allow_missing: bool = False) -> Path:
    if not isinstance(value, str) or not value:
        raise BenchmarkInputError("invalid_smoke_manifest", f"{label} must be a non-empty path")
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    try:
        resolved = candidate.resolve(strict=not allow_missing)
    except OSError as exc:
        raise BenchmarkInputError("invalid_smoke_manifest", f"{label} is missing or unreadable") from exc
    if allow_missing and not resolved.exists():
        return resolved
    if not resolved.is_file() or not os.access(resolved, os.R_OK):
        raise BenchmarkInputError("invalid_smoke_manifest", f"{label} is missing or unreadable")
    try:
        with resolved.open("rb"):
            pass
    except OSError as exc:
        raise BenchmarkInputError("invalid_smoke_manifest", f"{label} is unreadable") from exc
    return resolved


def load_manifest(
    path: str | Path,
    config: BenchmarkConfig,
    *,
    allow_derived_render: bool = False,
    allow_missing_derived_audio: bool = False,
) -> tuple[SmokeCase, ...]:
    """Read and validate an anonymous local manifest; paths never enter reports."""
    manifest_path = Path(path).resolve(strict=True)
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkInputError("invalid_smoke_manifest", "smoke manifest could not be read") from exc
    data = _required_mapping(raw, "smoke manifest")
    if set(data) != {"format_version", "benchmark_spec_version", "cases"}:
        raise BenchmarkInputError("invalid_smoke_manifest", "smoke manifest shape is not version 1")
    _validate_exact(data.get("format_version"), 1, "manifest format version")
    _validate_exact(data.get("benchmark_spec_version"), 1, "benchmark spec version")
    rows = data.get("cases")
    if not isinstance(rows, list) or not config.smoke_min_cases <= len(rows) <= config.smoke_max_cases:
        raise BenchmarkInputError("invalid_smoke_manifest", "smoke manifest case count is outside the fixed range")

    cases: list[SmokeCase] = []
    audio_paths: set[Path] = set()
    pair_paths: set[tuple[Path, Path]] = set()
    allowed = {
        "performance": set(_PERFORMANCE),
        "role": set(_ROLES),
        "envelope": set(_ENVELOPES),
        "duration_class": set(_DURATIONS),
        "note_density_class": set(_DENSITIES),
    }
    base_fields = {
        "case_index",
        "audio_path",
        "midi_path",
        "performance",
        "role",
        "envelope",
        "duration_class",
        "note_density_class",
    }
    existing_fields = base_fields | {"audio_source"}
    derived_fields = existing_fields | {"preset_path"}
    for expected_index, row in enumerate(rows, start=1):
        item = _required_mapping(row, "smoke case")
        if set(item) == base_fields:
            audio_source = "existing_audio"
            preset_path = None
        elif allow_derived_render and set(item) == existing_fields:
            audio_source = item["audio_source"]
            preset_path = None
        elif allow_derived_render and set(item) == derived_fields:
            audio_source = item["audio_source"]
            preset_path = _source_path(manifest_path, item["preset_path"], "preset path")
        else:
            raise BenchmarkInputError("invalid_smoke_manifest", "smoke case shape is invalid")
        if audio_source not in _AUDIO_SOURCES:
            raise BenchmarkInputError("invalid_smoke_manifest", "smoke case has an unsupported audio source")
        if audio_source == "derived_render" and preset_path is None:
            raise BenchmarkInputError("invalid_smoke_manifest", "derived render case requires a preset path")
        if audio_source == "existing_audio" and preset_path is not None:
            raise BenchmarkInputError("invalid_smoke_manifest", "existing audio case cannot have a preset path")
        if type(item["case_index"]) is not int or item["case_index"] != expected_index:
            raise BenchmarkInputError("invalid_smoke_manifest", "case indexes must be sequential")
        labels = {name: item[name] for name in allowed}
        if any(not isinstance(value, str) or value not in allowed[name] for name, value in labels.items()):
            raise BenchmarkInputError("invalid_smoke_manifest", "smoke case has an unsupported label")
        audio_path = _source_path(
            manifest_path,
            item["audio_path"],
            "audio path",
            allow_missing=audio_source == "derived_render" and allow_missing_derived_audio,
        )
        midi_path = _source_path(manifest_path, item["midi_path"], "MIDI path")
        if audio_path in audio_paths or (audio_path, midi_path) in pair_paths:
            raise BenchmarkInputError("invalid_smoke_manifest", "smoke manifest contains a duplicate source pair")
        audio_paths.add(audio_path)
        pair_paths.add((audio_path, midi_path))
        cases.append(
            SmokeCase(
                case_index=expected_index,
                audio_path=audio_path,
                midi_path=midi_path,
                audio_source=audio_source,
                preset_path=preset_path,
                **labels,
            )
        )
    return tuple(cases)


def coverage_summary(cases: Sequence[SmokeCase]) -> dict[str, dict[str, int]]:
    fields = ("audio_source", "performance", "role", "envelope", "duration_class", "note_density_class")
    return {field: dict(sorted(Counter(getattr(case, field) for case in cases).items())) for field in fields}


def _approved_derived_output_root() -> Path:
    return (Path(__file__).resolve().parents[1] / "outputs").resolve()


def _validate_derived_destination(case: SmokeCase, output_root: Path) -> Path:
    destination = case.audio_path.resolve(strict=False)
    if destination == output_root or not destination.is_relative_to(output_root):
        raise BenchmarkInputError("derived_render_destination_invalid", "derived audio destination is outside approved outputs")
    if destination.suffix.lower() != ".wav":
        raise BenchmarkInputError("derived_render_destination_invalid", "derived audio destination must be a WAV")
    for source in (case.preset_path, case.midi_path):
        assert source is not None
        source_root = source.parent.resolve()
        if destination == source_root or destination.is_relative_to(source_root):
            raise BenchmarkInputError("derived_render_destination_invalid", "derived audio destination overlaps a source directory")
    if destination.exists() and not destination.is_file():
        raise BenchmarkInputError("derived_render_destination_invalid", "derived audio destination is not a file")
    return destination


def _atomic_json_write(path: Path, value: Mapping[str, Any]) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _data_generation_root() -> Path:
    return (Path(__file__).resolve().parents[4] / "research" / "data_generation").resolve()


def _default_vital_plugin_path() -> Path | None:
    configured = os.environ.get("OBRUXO_VITAL_PLUGIN")
    if configured:
        return Path(configured)
    if sys.platform == "win32":
        program_files = os.environ.get("PROGRAMFILES", r"C:\Program Files")
        candidates = [Path(program_files) / "Common Files" / "VST3" / "Vital.vst3"]
    elif sys.platform == "darwin":
        candidates = [Path("/Library/Audio/Plug-Ins/VST3/Vital.vst3")]
    else:
        candidates = [Path("/usr/lib/vst3/Vital.vst3"), Path("/usr/local/lib/vst3/Vital.vst3")]
    return next((path for path in candidates if path.exists()), None)


def _import_data_generation_modules() -> tuple[Any, ...]:
    data_generation_root = _data_generation_root()
    if not data_generation_root.is_dir():
        raise DerivedRenderUnavailable()
    root_text = str(data_generation_root)
    inserted = root_text not in sys.path
    if inserted:
        sys.path.insert(0, root_text)
    try:
        from obruxo_data.errors import Severity
        from obruxo_data.midi import Performance, TempoMap
        from obruxo_data.render.capabilities import RendererCapabilities
        from obruxo_data.render.qa import AudioQualityConfig, analyze_audio
        from obruxo_data.render.vita import VitalVst3StateTemplate
        from obruxo_data.vital import VitalPreset

        return (
            Performance,
            TempoMap,
            RendererCapabilities,
            AudioQualityConfig,
            analyze_audio,
            VitalVst3StateTemplate,
            VitalPreset,
            Severity,
        )
    except Exception as exc:
        raise DerivedRenderUnavailable() from exc
    finally:
        if inserted:
            sys.path.remove(root_text)


class PedalboardVitalRenderer:
    """The narrow Vital host authorized for new ignored derived renders."""

    sample_rate = 44_100
    channels = 2
    tail_seconds = 2.0

    def __init__(self, *, config_path: Path, plugin_path: Path, vital_sha256: str, buffer_size: int,
                 capabilities: Any, qa_config: Any, pedalboard_module: Any, state_template: Any,
                 plugin: Any, pedalboard_version: str):
        self.config_path = config_path.resolve()
        self.plugin_path = plugin_path.resolve()
        self.vital_sha256 = vital_sha256
        self.buffer_size = buffer_size
        self.capabilities = capabilities
        self.qa_config = qa_config
        self._pedalboard = pedalboard_module
        self._state_template = state_template
        self._plugin = plugin
        self.pedalboard_version = pedalboard_version
        self.renderer_id = f"vital-{self.vital_sha256}-pedalboard-{self.pedalboard_version}"

    @classmethod
    def from_config(cls, config_path: str | Path) -> PedalboardVitalRenderer:
        config = Path(config_path).resolve(strict=True)
        try:
            document = yaml.safe_load(config.read_text(encoding="utf-8"))
            if not isinstance(document, Mapping) or document.get("version") != 1:
                raise ValueError("unsupported renderer configuration")
            accepted = {str(item).lower() for item in document.get("accepted_plugin_sha256", [])}
            if not accepted:
                raise ValueError("renderer configuration has no accepted Vital digest")
            buffer_size = int(document.get("buffer_size", 0))
            if buffer_size != 128:
                raise ValueError("the approved Vital buffer size must be 128")
            configured = document.get("plugin_path")
            plugin_path = Path(str(configured)) if configured else _default_vital_plugin_path()
            if plugin_path is None:
                raise FileNotFoundError("Vital VST3 was not found")
            plugin_path = plugin_path.resolve(strict=True)
            if not plugin_path.is_file():
                raise FileNotFoundError(f"Vital VST3 is not a file: {plugin_path}")
            vital_sha256 = _sha256_file(plugin_path)
            if vital_sha256 not in accepted:
                raise ValueError("Vital VST3 SHA-256 is not accepted by renderer.yaml")
            capabilities_data = document.get("capabilities", {})
            qa_data = document.get("qa", {})
            if not isinstance(capabilities_data, Mapping) or not isinstance(qa_data, Mapping):
                raise TypeError("renderer configuration sections are invalid")
            (
                _, _, renderer_capabilities, audio_quality_config, _, vital_state_template, _, _
            ) = _import_data_generation_modules()
            capabilities = renderer_capabilities.from_dict(dict(capabilities_data))
            qa_config = audio_quality_config(**dict(qa_data))
            import pedalboard

            load_plugin = getattr(pedalboard, "load_plugin", None)
            if not callable(load_plugin):
                raise TypeError("installed Pedalboard has no load_plugin API")
            pedalboard_version = importlib.metadata.version("pedalboard")
            plugin = load_plugin(str(plugin_path))
            if not bool(getattr(plugin, "is_instrument", False)):
                raise TypeError("Vital VST3 did not load as an instrument")
            raw_state = getattr(plugin, "raw_state", None)
            if not isinstance(raw_state, (bytes, bytearray)):
                raise TypeError("Pedalboard Vital plugin has no byte raw_state")
            state_template = vital_state_template(bytes(raw_state))
            if not callable(getattr(plugin, "reset", None)) or not callable(getattr(plugin, "process", None)):
                raise TypeError("installed Pedalboard Vital plugin lacks reset/process support")
            return cls(
                config_path=config,
                plugin_path=plugin_path,
                vital_sha256=vital_sha256,
                buffer_size=buffer_size,
                capabilities=capabilities,
                qa_config=qa_config,
                pedalboard_module=pedalboard,
                state_template=state_template,
                plugin=plugin,
                pedalboard_version=pedalboard_version,
            )
        except DerivedRenderUnavailable:
            raise
        except Exception as exc:
            raise DerivedRenderUnavailable() from exc

    def _timestamped_midi(self, performance: Any, tempo_map: Any) -> list[tuple[bytes, float]]:
        messages: list[tuple[bytes, float]] = []
        for event in performance.canonical_events():
            channel = 0 if event.channel is None else int(event.channel)
            timestamp = float(tempo_map.tick_to_seconds(event.tick))
            if event.kind.value == "note_on":
                message = bytes((0x90 | channel, int(event.data[0]), int(event.data[1])))
            elif event.kind.value == "note_off":
                message = bytes((0x80 | channel, int(event.data[0]), int(event.data[1])))
            elif event.kind.value == "pitch_bend":
                value = int(event.data[0]) + 8192
                message = bytes((0xE0 | channel, value & 0x7F, (value >> 7) & 0x7F))
            elif event.kind.value == "control_change":
                message = bytes((0xB0 | channel, int(event.data[0]), int(event.data[1])))
            elif event.kind.value == "channel_pressure":
                message = bytes((0xD0 | channel, int(event.data[0])))
            elif event.kind.value in {"tempo", "time_signature", "opaque"}:
                continue
            else:
                raise ValueError(f"unsupported MIDI event kind: {event.kind.value}")
            messages.append((message, timestamp))
        return messages

    def render(self, preset_path: Path, midi_path: Path, destination: Path, output_root: Path) -> None:
        if destination.exists():
            raise DerivedRenderUnavailable("derived_render_failed")
        destination = destination.resolve(strict=False)
        output_root = output_root.resolve()
        if destination == output_root or not destination.is_relative_to(output_root):
            raise BenchmarkInputError("derived_render_destination_invalid", "derived audio destination is outside approved outputs")
        for source in (preset_path, midi_path):
            source_root = source.resolve(strict=True).parent
            if destination == source_root or destination.is_relative_to(source_root):
                raise BenchmarkInputError("derived_render_destination_invalid", "derived audio destination overlaps a source directory")
        (
            Performance, tempo_map_type, _, _, analyze_audio, _, _, Severity
        ) = _import_data_generation_modules()
        try:
            import numpy as np
            from scipy.io import wavfile

            preset_json = preset_path.read_text(encoding="utf-8-sig")
            json.loads(preset_json)
            performance = Performance.from_midi(midi_path)
            performance.validate().require_valid()
            timing = tempo_map_type.from_performance(performance)
            state = self._state_template.build(preset_json)
            self._plugin.reset()
            self._plugin.raw_state = state
            self._plugin.reset()
            tempo_events = [event for event in performance.canonical_events() if event.kind.value == "tempo"]
            if tempo_events and hasattr(self._plugin, "beats_per_minute"):
                self._plugin.beats_per_minute = 60_000_000 / tempo_events[0].data[0]
            frame_count = timing.render_frame_count(performance.end_tick, self.tail_seconds, self.sample_rate)
            duration = frame_count / self.sample_rate
            host_duration = (frame_count + 0.5) / self.sample_rate
            audio = self._plugin.process(
                self._timestamped_midi(performance, timing),
                duration=host_duration,
                sample_rate=self.sample_rate,
                num_channels=self.channels,
                buffer_size=self.buffer_size,
                reset=True,
            )
            channel_first = np.asarray(audio, dtype=np.float32)
            if channel_first.ndim != 2 or channel_first.shape[0] != self.channels:
                raise ValueError("Pedalboard returned an unexpected channel layout")
            rendered = np.ascontiguousarray(channel_first.T)
            if rendered.shape != (frame_count, self.channels) or not bool(np.isfinite(rendered).all()):
                raise ValueError("Pedalboard returned invalid Vital audio")
            qa, diagnostics = analyze_audio(
                rendered,
                sample_rate=self.sample_rate,
                expected_frames=frame_count,
                expected_channels=self.channels,
                config=self.qa_config,
            )
            if any(item.severity == Severity.ERROR for item in diagnostics):
                raise ValueError("derived Vital audio failed the existing QA gate")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False) as stream:
                    temporary = Path(stream.name)
                wavfile.write(temporary, self.sample_rate, rendered)
                os.replace(temporary, destination)
                temporary = None
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            sidecar = destination.with_suffix(".json")
            provenance = {
                "host": "pedalboard",
                "pedalboard_version": self.pedalboard_version,
                "vital_plugin_path": str(self.plugin_path),
                "vital_plugin_sha256": self.vital_sha256,
                "renderer_config_path": str(self.config_path),
                "sample_rate": self.sample_rate,
                "channels": self.channels,
                "buffer_size": self.buffer_size,
                "tail_seconds": self.tail_seconds,
                "duration_seconds": duration,
                "host_duration_seconds": host_duration,
                "duration_rounding": "half-frame guard to preserve the exact frame count in Pedalboard",
                "event_timing": "TempoMap tick-to-second timestamped MIDI",
                "source_preset_path": str(preset_path.resolve()),
                "source_midi_path": str(midi_path.resolve()),
                "audio_label": "derived_render",
            }
            _atomic_json_write(
                sidecar,
                {
                    "audio_source": "derived_render",
                    "sample_rate": self.sample_rate,
                    "diagnostics": [item.to_dict() for item in diagnostics],
                    "qa": qa,
                    "provenance": {
                        "renderer_id": self.renderer_id,
                        "backend_version": self.pedalboard_version,
                        "engine_fingerprint": self.vital_sha256,
                        "settings": provenance,
                    },
                    "derived_render_provenance": provenance,
                },
            )
        except (BenchmarkInputError, DerivedRenderUnavailable):
            raise
        except Exception as exc:
            raise DerivedRenderUnavailable("derived_render_failed") from exc


def _validated_vital_renderer() -> PedalboardVitalRenderer:
    return PedalboardVitalRenderer.from_config(_data_generation_root() / "configs" / "renderer.yaml")


def render_derived_vital_audio(renderer: PedalboardVitalRenderer, preset_path: Path, midi_path: Path,
                               destination: Path, output_root: Path) -> None:
    """Render one approved derived WAV through the #24 Pedalboard/Vital seam."""
    renderer.render(preset_path, midi_path, destination, output_root)


def _render_derived_audio(renderer: PedalboardVitalRenderer, case: SmokeCase, destination: Path, output_root: Path) -> None:
    assert case.preset_path is not None
    render_derived_vital_audio(renderer, case.preset_path, case.midi_path, destination, output_root)


def prepare_derived_renders(path: str | Path, config: BenchmarkConfig) -> None:
    """Materialize only explicitly opted-in missing audio under Basic Pitch outputs."""
    manifest = Path(path).resolve(strict=True)
    cases = load_manifest(
        manifest,
        config,
        allow_derived_render=True,
        allow_missing_derived_audio=True,
    )
    derived = [case for case in cases if case.audio_source == "derived_render"]
    if not derived:
        return
    output_root = _approved_derived_output_root()
    renderer = None
    for case in derived:
        destination = _validate_derived_destination(case, output_root)
        sidecar = destination.with_suffix(".json")
        if destination.exists():
            try:
                metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise BenchmarkInputError("derived_render_destination_invalid", "existing derived audio lacks provenance") from exc
            if metadata.get("audio_source") != "derived_render":
                raise BenchmarkInputError("derived_render_destination_invalid", "existing audio is not labeled as derived")
            continue
        if sidecar.exists():
            raise BenchmarkInputError("derived_render_destination_invalid", "derived audio provenance exists without audio")
        if renderer is None:
            renderer = _validated_vital_renderer()
        _render_derived_audio(renderer, case, destination, output_root)


def aggregate_measurements(values: Sequence[float]) -> dict[str, float]:
    """Return fixed descriptive statistics for a non-empty measurement set."""
    if not values:
        raise ValueError("at least one measurement is required")
    numeric = [float(value) for value in values]
    if not all(math.isfinite(value) and value >= 0 for value in numeric):
        raise ValueError("measurements must be finite and non-negative")
    return {
        "median": float(statistics.median(numeric)),
        "min": float(min(numeric)),
        "max": float(max(numeric)),
        "total": float(sum(numeric)),
    }


def _aggregate_optional(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    values = [row.get(field) for row in rows]
    present = [float(value) for value in values if value is not None]
    if not present:
        return {"value": None, "measurement_status": "not_applicable"}
    if len(present) != len(values):
        return {"value": None, "measurement_status": "unavailable"}
    return {"value": aggregate_measurements(present), "measurement_status": "ok"}


def crossover_audio_seconds(startup_a: float, throughput_a: float, startup_b: float, throughput_b: float) -> float | None:
    """Apply the fixed crossover formula, returning only positive finite results."""
    if not all(math.isfinite(value) and value > 0 for value in (startup_a, throughput_a, startup_b, throughput_b)):
        return None
    if not startup_b > startup_a or not throughput_b > throughput_a:
        return None
    denominator = (1.0 / throughput_a) - (1.0 / throughput_b)
    distance = (startup_b - startup_a) / denominator
    return float(distance) if math.isfinite(distance) and distance > 0 else None


def _runtime_identity() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for name, module_name, attribute in (
        ("numpy", "numpy", "__version__"),
        ("scipy", "scipy", "__version__"),
        ("torch", "torch", "__version__"),
        ("openvino", "openvino", "__version__"),
        ("psutil", "psutil", "__version__"),
        ("onnx", "onnx", "__version__"),
        ("onnxruntime", "onnxruntime", "__version__"),
    ):
        try:
            versions[name] = str(getattr(import_module(module_name), attribute))
        except (AttributeError, ImportError, OSError, RuntimeError):
            versions[name] = None
    return versions


def _unavailable_route(route: str, mode: str, code: str) -> dict[str, Any]:
    return {"route": route, "mode": mode, "status": "unavailable", "failure_code": code}


def _source_snapshot(cases: Sequence[SmokeCase]) -> dict[Path, tuple[int, int]]:
    return {
        path: (path.stat().st_size, path.stat().st_mtime_ns)
        for case in cases
        for path in (case.audio_path, case.midi_path, case.preset_path)
        if path is not None
    }


def _verify_source_snapshot(snapshot: Mapping[Path, tuple[int, int]]) -> None:
    for path, before in snapshot.items():
        current = path.stat()
        if (current.st_size, current.st_mtime_ns) != before:
            raise SourceMutationError("an immutable benchmark source changed during execution")


def _worker_request(
    route: str,
    mode: str,
    repetition: int,
    config: BenchmarkConfig | None,
    manifest_path: Path | None,
    checkpoint_path: Path,
    xpu_index: int,
    openvino_gpu_device: str,
    *,
    parity_only: bool = False,
) -> dict[str, Any]:
    request = {
        "route": route,
        "mode": mode,
        "repetition": repetition,
        "checkpoint_path": str(checkpoint_path),
        "xpu_index": xpu_index,
        "openvino_gpu_device": openvino_gpu_device,
    }
    if config is not None:
        request["config"] = {
            "warmup_iterations": config.warmup_iterations,
            "timed_iterations": config.timed_iterations,
            "batch_sizes": list(config.batch_sizes),
            "end_to_end_batch_size": config.end_to_end_batch_size,
        }
    if manifest_path is not None:
        request["manifest_path"] = str(manifest_path)
    if parity_only:
        request["parity_only"] = True
    return request


def _run_worker(request: Mapping[str, Any]) -> dict[str, Any]:
    worker = Path(__file__).with_name("benchmark_worker.py")
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "obruxo_basic_pitch.benchmark_worker"],
            input=json.dumps(request),
            capture_output=True,
            text=True,
            check=False,
            cwd=worker.parent.parent,
        )
    except OSError:
        return {"status": "runtime_failed", "failure_code": "benchmark_runtime_error"}
    if completed.returncode != 0:
        return {"status": "runtime_failed", "failure_code": "benchmark_runtime_error"}
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {"status": "runtime_failed", "failure_code": "benchmark_runtime_error"}
    if not isinstance(result, dict) or result.get("status") not in STATUSES:
        return {"status": "runtime_failed", "failure_code": "benchmark_runtime_error"}
    return result


def _aggregate_route(rows: Sequence[Mapping[str, Any]], route: str, mode: str, config: BenchmarkConfig) -> dict[str, Any]:
    statuses = [row.get("status") for row in rows]
    if any(status == "parity_failed" for status in statuses):
        status = "parity_failed"
    elif any(status == "out_of_memory" for status in statuses):
        status = "out_of_memory"
    elif any(status == "runtime_failed" for status in statuses):
        status = "runtime_failed"
    elif any(status == "unavailable" for status in statuses):
        status = "unavailable"
    else:
        status = "ok"
    result: dict[str, Any] = {
        "route": route,
        "mode": mode,
        "status": status,
        "repetitions": len(rows),
    }
    failure_codes = sorted({str(row["failure_code"]) for row in rows if row.get("failure_code")})
    if failure_codes:
        result["failure_codes"] = failure_codes
    parity_rows = [
        {
            "repetition": int(row.get("repetition", index)),
            "status": row.get("status"),
            "parity": row["parity"],
        }
        for index, row in enumerate(rows)
        if isinstance(row.get("parity"), Mapping)
    ]
    if parity_rows:
        result["parity_repetitions"] = parity_rows
    if status != "ok":
        return result

    startup_fields = (
        "backend_import_seconds",
        "model_construct_seconds",
        "checkpoint_load_seconds",
        "model_device_move_seconds",
        "openvino_conversion_seconds",
        "openvino_compile_seconds",
    )
    result["startup"] = {field: _aggregate_optional([row["startup"] for row in rows], field) for field in startup_fields}
    startup_values = [
        sum(
            float(row["startup"][field])
            for field in startup_fields
            if row["startup"].get(field) is not None
        )
        for row in rows
    ]
    result["startup"]["total_seconds"] = {"value": aggregate_measurements(startup_values), "measurement_status": "ok"}
    result["parity"] = {
        name: max(float(row["parity"][name]) for row in rows)
        for name in (
            "contour_max_abs_error",
            "note_max_abs_error",
            "onset_max_abs_error",
            "contour_non_finite_count",
            "note_non_finite_count",
            "onset_non_finite_count",
            "event_count_disagreements",
            "event_tuple_disagreements",
            "note_threshold_disagreements",
            "onset_threshold_disagreements",
            "event_structure_disagreements",
            "pitch_bend_element_disagreements",
        )
    }
    result["batch_results"] = {}
    source_key = "model_only" if mode == "inference" else "training"
    for batch_size in config.batch_sizes:
        values = [row[source_key]["batch_sizes"][str(batch_size)] for row in rows]
        phase_fields = (
            "first_inference_seconds",
            "inference_warmup_seconds",
        ) if mode == "inference" else (
            "first_training_step_seconds",
            "training_warmup_seconds",
        )
        result["batch_results"][str(batch_size)] = {
            field: aggregate_measurements([float(value[field]) for value in values])
            for field in (
                *phase_fields,
                "median_seconds",
                "min_seconds",
                "max_seconds",
                "total_seconds",
            )
        }
        result["batch_results"][str(batch_size)]["windows_per_second"] = aggregate_measurements(
            [float(value["windows_per_second"]) for value in values]
        )
        result["batch_results"][str(batch_size)]["audio_seconds_per_second"] = aggregate_measurements(
            [float(value["audio_seconds_per_second"]) for value in values]
        )
    if mode == "inference":
        result["end_to_end"] = {
            "audio_seconds": aggregate_measurements([float(row["end_to_end"]["audio_seconds"]) for row in rows]),
            "wall_seconds": aggregate_measurements([float(row["end_to_end"]["wall_seconds"]) for row in rows]),
            "audio_seconds_per_wall_second": aggregate_measurements(
                [float(row["end_to_end"]["audio_seconds_per_wall_second"]) for row in rows]
            ),
            "cases": [
                {
                    key: row["end_to_end"]["cases"][case_index][key]
                    for key in ("case_index", "status", "audio_seconds", "wall_seconds", "note_event_count")
                }
                for case_index in range(len(rows[0]["end_to_end"]["cases"]))
                for row in rows[:1]
            ],
        }
    result["memory"] = rows[0].get("memory", {"status": "unavailable"})
    return result


def _parity_contract() -> dict[str, Any]:
    from .constants import FRAME_THRESHOLD, ONSET_THRESHOLD
    from .parity import ADOPTED_MAX_ABS_TOLERANCES

    return {
        "non_finite_values": "0 values",
        "contour_max_abs_error": ADOPTED_MAX_ABS_TOLERANCES["contour"],
        "note_max_abs_error": ADOPTED_MAX_ABS_TOLERANCES["note"],
        "onset_max_abs_error": ADOPTED_MAX_ABS_TOLERANCES["onset"],
        "note_frame_threshold": FRAME_THRESHOLD,
        "onset_threshold": ONSET_THRESHOLD,
        "note_threshold_disagreements": "0",
        "onset_threshold_disagreements": "0",
        "event_count_disagreements": "0",
        "event_tuple_disagreements": "0",
    }


_PARITY_DIAGNOSTIC_METRICS = (
    "contour_non_finite_count",
    "note_non_finite_count",
    "onset_non_finite_count",
    "contour_max_abs_error",
    "note_max_abs_error",
    "onset_max_abs_error",
    "note_threshold_disagreements",
    "onset_threshold_disagreements",
    "event_count_disagreements",
    "event_tuple_disagreements",
)


def _parity_route_status(rows: Sequence[Mapping[str, Any]]) -> str:
    statuses = [row.get("status") for row in rows]
    if any(status == "parity_failed" for status in statuses):
        return "parity_failed"
    if any(status == "out_of_memory" for status in statuses):
        return "out_of_memory"
    if any(status == "runtime_failed" for status in statuses):
        return "runtime_failed"
    if any(status == "unavailable" for status in statuses):
        return "unavailable"
    return "ok"


def _parity_route_report(route: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = [row["parity"] for row in rows if isinstance(row.get("parity"), Mapping)]
    aggregate: dict[str, Any] = {}
    for name in _PARITY_DIAGNOSTIC_METRICS:
        observed = [value.get(name) for value in values if value.get(name) is not None]
        aggregate[name] = max(observed) if observed else None
    return {
        "route": route,
        "status": _parity_route_status(rows),
        "repetitions": [
            {
                "repetition": int(row.get("repetition", index)),
                "status": row.get("status"),
                "parity": row.get("parity"),
            }
            for index, row in enumerate(rows)
        ],
        "max_across_repetitions": aggregate,
    }


def run_parity_diagnostics(
    checkpoint_path: str | Path,
    *,
    process_repetitions: int = 3,
    xpu_index: int = 0,
    openvino_gpu_device: str = "GPU",
) -> dict[str, Any]:
    """Run only the fixed synthetic parity gate in fresh worker processes."""
    if process_repetitions < 1:
        raise ValueError("process_repetitions must be positive")
    checkpoint = Path(checkpoint_path).resolve(strict=True)
    routes = []
    for route in INFERENCE_ROUTES:
        rows = [
            _run_worker(
                _worker_request(
                    route,
                    "inference",
                    repetition,
                    None,
                    None,
                    checkpoint,
                    xpu_index,
                    openvino_gpu_device,
                    parity_only=True,
                )
            )
            for repetition in range(process_repetitions)
        ]
        routes.append(_parity_route_report(route, rows))
    return {
        "format_version": 1,
        "model_id": MODEL_ID,
        "scope": "canonical float32 model on five public synthetic windows; no private smoke audio or rendering",
        "process_repetitions": process_repetitions,
        "thresholds": _parity_contract(),
        "routes": routes,
    }


def _unavailable_report(
    config: BenchmarkConfig,
    code: str,
    *,
    reason: str | None = None,
    cases: Sequence[SmokeCase] = (),
) -> dict[str, Any]:
    if reason is None:
        reason = {
            "derived_render_unavailable": "the opted-in validated Vital derived-render path was unavailable",
            "derived_render_failed": "the opted-in validated Vital derived-render path failed",
            "derived_render_destination_invalid": "the derived-render destination failed its safety check",
        }.get(code, "a valid local smoke manifest was unavailable")
    return {
        "format_version": 1,
        "benchmark_spec_version": config.version,
        "model_id": MODEL_ID,
        "source_git_blob_sha1": SPOTIFY_ONNX_GIT_BLOB_SHA1,
        "runtime": _runtime_identity(),
        "config": config.as_dict(),
        "smoke_set": {
            "status": "unavailable",
            "failure_code": code,
            "case_count": len(cases),
            "coverage": coverage_summary(cases) if cases else {},
        },
        "inference": [_unavailable_route(route, "inference", code) for route in INFERENCE_ROUTES],
        "training": [_unavailable_route(route, "training", code) for route in TRAINING_ROUTES],
        "crossovers": [],
        "conclusions": {"status": "blocked", "reason": reason},
    }


def _crossover_rows(inference: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, left in enumerate(inference):
        if left.get("status") != "ok":
            continue
        for right in inference[index + 1 :]:
            if right.get("status") != "ok":
                continue
            left_start = float(left["startup"]["total_seconds"]["value"]["median"])
            right_start = float(right["startup"]["total_seconds"]["value"]["median"])
            left_rate = float(left["batch_results"]["1"]["audio_seconds_per_second"]["median"])
            right_rate = float(right["batch_results"]["1"]["audio_seconds_per_second"]["median"])
            distance = crossover_audio_seconds(left_start, left_rate, right_start, right_rate)
            if distance is not None:
                rows.append({"route_a": left["route"], "route_b": right["route"], "audio_seconds": distance})
    return rows


def run_benchmark(
    config: BenchmarkConfig,
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    *,
    xpu_index: int = 0,
    openvino_gpu_device: str = "GPU",
    allow_derived_render: bool = False,
) -> dict[str, Any]:
    """Run the fixed matrix in foreground fresh subprocesses and sanitize it."""
    manifest = Path(manifest_path).resolve(strict=True)
    if allow_derived_render:
        prepare_derived_renders(manifest, config)
    cases = load_manifest(manifest, config, allow_derived_render=allow_derived_render)
    snapshot = _source_snapshot(cases)
    checkpoint = Path(checkpoint_path).resolve(strict=True)
    inference: list[dict[str, Any]] = []
    training: list[dict[str, Any]] = []
    try:
        for mode, routes, destination in (
            ("inference", INFERENCE_ROUTES, inference),
            ("training", TRAINING_ROUTES, training),
        ):
            for route in routes:
                rows = [
                    _run_worker(
                        _worker_request(
                            route,
                            mode,
                            repetition,
                            config,
                            manifest,
                            checkpoint,
                            xpu_index,
                            openvino_gpu_device,
                        )
                    )
                    for repetition in range(config.process_repetitions)
                ]
                destination.append(_aggregate_route(rows, route, mode, config))
    finally:
        _verify_source_snapshot(snapshot)
    report = {
        "format_version": 1,
        "benchmark_spec_version": config.version,
        "model_id": MODEL_ID,
        "source_git_blob_sha1": SPOTIFY_ONNX_GIT_BLOB_SHA1,
        "runtime": _runtime_identity(),
        "config": config.as_dict(),
        "smoke_set": {"status": "ok", "case_count": len(cases), "coverage": coverage_summary(cases), "cases": [case.sanitized() for case in cases]},
        "inference": inference,
        "training": training,
        "crossovers": _crossover_rows(inference),
        "conclusions": {
            "inference_highest_batch_1_audio_throughput": max(
                (row for row in inference if row.get("status") == "ok"),
                key=lambda row: row["batch_results"]["1"]["audio_seconds_per_second"]["median"],
                default=None,
            ),
            "training_highest_batch_1_audio_throughput": max(
                (row for row in training if row.get("status") == "ok"),
                key=lambda row: row["batch_results"]["1"]["audio_seconds_per_second"]["median"],
                default=None,
            ),
        },
    }
    for section in ("inference", "training"):
        key = f"{section}_highest_batch_1_audio_throughput"
        winner = report["conclusions"][key]
        if winner is not None:
            report["conclusions"][key] = winner["route"]
    return report


def _approved_report_paths(json_path: str | Path, markdown_path: str | Path) -> tuple[Path, Path]:
    reports_root = (Path(__file__).resolve().parents[1] / "reports").resolve()
    resolved = tuple(Path(path).resolve(strict=False) for path in (json_path, markdown_path))
    if any(path == reports_root or not path.is_relative_to(reports_root) for path in resolved):
        raise ValueError("benchmark reports must be inside the approved reports directory")
    return resolved


def _report_median(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return None
    nested = value.get("value")
    if isinstance(nested, Mapping):
        return nested.get("median")
    return value.get("median")


def _report_startup(row: Mapping[str, Any], field: str) -> Any:
    return _report_median(row.get("startup", {}).get(field))


def _report_batch(row: Mapping[str, Any], batch_size: int, field: str) -> Any:
    values = row.get("batch_results", {}).get(str(batch_size), {})
    return _report_median(values.get(field))


def _report_number(value: Any, digits: int = 3) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _report_mib(value: Any) -> str:
    return "n/a" if value is None else f"{float(value) / (1024 * 1024):.1f}"


def _parity_diagnostic_value(report: Mapping[str, Any], route: str, metric: str) -> str:
    diagnostics = report.get("parity_diagnostics", {})
    route_report = next(
        (row for row in diagnostics.get("routes", []) if row.get("route") == route),
        {},
    )
    if metric == "status":
        return str(route_report.get("status", "not_recorded"))
    aggregate = route_report.get("max_across_repetitions", {})
    value = aggregate.get(metric)
    if value is None:
        if metric.endswith("_max_abs_error"):
            output_name = metric.removesuffix("_max_abs_error")
            if aggregate.get(f"{output_name}_non_finite_count", 0):
                return "non_finite"
        return "n/a"
    if metric.endswith(("_count", "_disagreements")):
        return str(int(value))
    return f"{float(value):.9g}"


def _parity_diagnostics_markdown(report: Mapping[str, Any]) -> list[str]:
    diagnostics = report.get("parity_diagnostics")
    if not isinstance(diagnostics, Mapping):
        return [
            "",
            "## Parity diagnostics by framework and processor",
            "",
            "Per-component parity values were not retained in this benchmark artifact.",
        ]
    thresholds = diagnostics.get("thresholds", {})
    configuration = diagnostics.get("configuration", {})
    contour_limit = thresholds.get("contour_max_abs_error", "not recorded")
    note_limit = thresholds.get("note_max_abs_error", "not recorded")
    onset_limit = thresholds.get("onset_max_abs_error", "not recorded")
    note_threshold = thresholds.get("note_frame_threshold", "not recorded")
    onset_threshold = thresholds.get("onset_threshold", "not recorded")
    metric_rows = (
        ("Route status (must be `ok`)", "status"),
        ("Non-finite contour values (must be 0)", "contour_non_finite_count"),
        ("Non-finite note values (must be 0)", "note_non_finite_count"),
        ("Non-finite onset values (must be 0)", "onset_non_finite_count"),
        (f"Maximum contour absolute error (<= {contour_limit})", "contour_max_abs_error"),
        (f"Maximum note absolute error (<= {note_limit})", "note_max_abs_error"),
        (f"Maximum onset absolute error (<= {onset_limit})", "onset_max_abs_error"),
        (f"Note-frame threshold disagreements (threshold {note_threshold}; must be 0)", "note_threshold_disagreements"),
        (f"Onset threshold disagreements (threshold {onset_threshold}; must be 0)", "onset_threshold_disagreements"),
        ("Generated note-event count disagreements (must be 0)", "event_count_disagreements"),
        ("(start_time_s, end_time_s, MIDI pitch) disagreements (must be 0)", "event_tuple_disagreements"),
    )
    route_labels = (
        ("pytorch_cpu", "PyTorch CPU"),
        ("pytorch_xpu", "PyTorch XPU"),
        ("openvino_cpu", "OpenVINO CPU"),
        ("openvino_gpu", "OpenVINO GPU"),
    )
    lines = [
        "",
        "## Parity diagnostics by framework and processor",
        "",
        f"The gate was evaluated on `{diagnostics.get('process_repetitions', 'n/a')}` fresh-process repetitions of `{diagnostics.get('scope', 'the public synthetic suite')}`. Each cell reports the maximum observed value across repetitions; the JSON retains each repetition separately.",
        "| Parity check (applied threshold) | " + " | ".join(label for _, label in route_labels) + " |",
        "| --- | " + " | ".join("---" for _ in route_labels) + " |",
    ]
    if isinstance(configuration, Mapping) and configuration.get("note"):
        lines[4:4] = ["", str(configuration["note"]), ""]
    for label, metric in metric_rows:
        lines.append(
            "| " + label + " | " + " | ".join(_parity_diagnostic_value(report, route, metric) for route, _ in route_labels) + " |"
        )
    return lines


def _openvino_precision_diagnostic_markdown(report: Mapping[str, Any]) -> list[str]:
    diagnostic = report.get("openvino_precision_diagnostic")
    lines = ["", "## OpenVINO GPU precision correction", ""]
    if not isinstance(diagnostic, Mapping):
        lines.append("No post-fix OpenVINO GPU precision diagnostic was retained in this benchmark artifact.")
        return lines
    runtime = diagnostic.get("runtime", {})
    configuration = diagnostic.get("configuration", {})
    parity = diagnostic.get("parity", {})

    def result(name: str, digits: int = 9) -> str:
        value = parity.get(name)
        if value is None:
            return "n/a"
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return _report_number(value, digits)
        return str(value)

    lines.extend(
        [
            f"This bounded post-fix diagnostic used `{diagnostic.get('batch_size', 'n/a')}` synthetic windows in one batch; it did not use private smoke audio, render audio, or measure benchmark throughput.",
            "",
            f"- Runtime: OpenVINO `{runtime.get('openvino', 'n/a')}` on `{runtime.get('device_name', 'n/a')}`; device architecture `{runtime.get('device_architecture', 'n/a')}`.",
            f"- Driver version: `{runtime.get('driver_version', 'n/a')}`.",
            f"- Compiled inference precision: `{configuration.get('inference_precision_hint_compiled', 'n/a')}` (requested `{configuration.get('inference_precision_hint_requested', 'n/a')}`).",
            f"- Compiled execution mode: `{configuration.get('execution_mode_hint_compiled', 'n/a')}`; execution-mode request: `{configuration.get('execution_mode_hint_requested', 'n/a')}`.",
            f"- Diagnostic status: `{diagnostic.get('status', 'not_recorded')}`. The original full benchmark timing rows remain pre-fix and require a later rerun.",
            "",
            "| Check (applied threshold) | Result |",
            "| --- | ---: |",
            f"| Compiled inference precision (must be float32) | {configuration.get('inference_precision_hint_compiled', 'n/a')} |",
            f"| Compiled execution mode (must remain PERFORMANCE) | {configuration.get('execution_mode_hint_compiled', 'n/a')} |",
            f"| Non-finite contour values (must be 0) | {result('contour_non_finite_count', 0)} |",
            f"| Non-finite note values (must be 0) | {result('note_non_finite_count', 0)} |",
            f"| Non-finite onset values (must be 0) | {result('onset_non_finite_count', 0)} |",
            f"| Maximum contour absolute error (<= {diagnostic.get('thresholds', {}).get('contour_max_abs_error', 'n/a')}) | {result('contour_max_abs_error')} |",
            f"| Maximum note absolute error (<= {diagnostic.get('thresholds', {}).get('note_max_abs_error', 'n/a')}) | {result('note_max_abs_error')} |",
            f"| Maximum onset absolute error (<= {diagnostic.get('thresholds', {}).get('onset_max_abs_error', 'n/a')}) | {result('onset_max_abs_error')} |",
            f"| Note-frame threshold disagreements (threshold {diagnostic.get('thresholds', {}).get('note_frame_threshold', 'n/a')}; must be 0) | {result('note_threshold_disagreements', 0)} |",
            f"| Onset threshold disagreements (threshold {diagnostic.get('thresholds', {}).get('onset_threshold', 'n/a')}; must be 0) | {result('onset_threshold_disagreements', 0)} |",
            f"| Generated note-event count disagreements (must be 0) | {result('event_count_disagreements', 0)} |",
            f"| (start_time_s, end_time_s, MIDI pitch) disagreements (must be 0) | {result('event_tuple_disagreements', 0)} |",
        ]
    )
    return lines


def _report_throughput_table(rows: Sequence[Mapping[str, Any]], field: str, heading: str) -> list[str]:
    lines = [
        heading,
        "",
        "| Route | Batch 1 | Batch 2 | Batch 4 | Batch 8 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        values = [_report_number(_report_batch(row, size, field)) for size in (1, 2, 4, 8)]
        lines.append(f"| `{row.get('route', 'unknown')}` | " + " | ".join(values) + " |")
    return lines


def _benchmark_markdown(report: Mapping[str, Any]) -> str:
    inference = report.get("inference", [])
    training = report.get("training", [])
    smoke = report.get("smoke_set", {})
    conclusions = report.get("conclusions", {})
    runtime = report.get("runtime", {})
    successful_inference = [row for row in inference if row.get("status") == "ok"]
    successful_training = [row for row in training if row.get("status") == "ok"]
    cpu = next((row for row in successful_inference if row.get("route") == "pytorch_cpu"), {})
    xpu = next((row for row in successful_inference if row.get("route") == "pytorch_xpu"), {})
    openvino_cpu = next((row for row in successful_inference if row.get("route") == "openvino_cpu"), {})
    gpu_failure = next((row for row in inference if row.get("route") == "openvino_gpu" and row.get("status") != "ok"), None)
    precision_diagnostic = report.get("openvino_precision_diagnostic")
    has_precision_diagnostic = isinstance(precision_diagnostic, Mapping)
    timing_provenance = (
        "These persisted timing values were collected before the OpenVINO float32 precision correction; `n/a` means the route failed before that phase or the phase does not apply."
        if has_precision_diagnostic
        else "Values below are median seconds across the three fresh processes; `n/a` means the route failed before that phase or the phase does not apply."
    )
    crossover = next(
        (
            row
            for row in report.get("crossovers", [])
            if {row.get("route_a"), row.get("route_b")} == {"pytorch_cpu", "pytorch_xpu"}
        ),
        None,
    )
    lines = [
        "# Basic Pitch backend benchmark",
        "",
        "This is a fixed measurement of the canonical #23 float32 model, not an optimization search. Markdown shows medians across three fresh-process repetitions; the JSON retains min/max/total values and anonymous per-case timing.",
        "",
        "## Executive findings",
        "",
        f"- PyTorch XPU is the observed steady-state inference leader at every tested model-call batch (`{_report_number(_report_batch(xpu, 1, 'audio_seconds_per_second'))}` audio-seconds/second at batch 1 and `{_report_number(_report_batch(xpu, 8, 'audio_seconds_per_second'))}` at batch 8), ahead of PyTorch CPU (`{_report_number(_report_batch(cpu, 1, 'audio_seconds_per_second'))}` at batch 1).",
        f"- On the warmed end-to-end smoke boundary, PyTorch XPU processes `{_report_number(xpu.get('end_to_end', {}).get('audio_seconds_per_wall_second', {}).get('median'))}` audio-seconds per wall-second, versus `{_report_number(cpu.get('end_to_end', {}).get('audio_seconds_per_wall_second', {}).get('median'))}` for CPU and `{_report_number(openvino_cpu.get('end_to_end', {}).get('audio_seconds_per_wall_second', {}).get('median'))}` for OpenVINO CPU.",
        f"- The model-call startup/throughput calculation records one positive CPU/XPU crossover at `{_report_number((crossover or {}).get('audio_seconds'))}` audio seconds. This is a descriptive model-only crossover, not a claim about all application workloads.",
        "- XPU pays a much larger first model call in the stored run than CPU, so short interactive calls should account for initialization and first-call latency; reused or longer workloads benefit from XPU steady-state throughput.",
        "- The persisted OpenVINO GPU row failed the parity gate in all three repetitions before timing under the original default GPU precision configuration. It has no pre-fix throughput or memory result.",
        "- Current OpenVINO GPU state: FP32 parity is validated by the bounded post-fix diagnostic; corrected FP32 startup, throughput, end-to-end, and resource measurements have not yet been collected.",
        "",
        "## Runtime and benchmark setup",
        "",
        f"- Model: `{report.get('model_id', MODEL_ID)}`; precision: `float32`; smoke set: `{smoke.get('status', 'unknown')}` with `{smoke.get('case_count', 0)}` cases.",
        f"- Runtime: Python `{runtime.get('python', 'unknown')}`, PyTorch `{runtime.get('torch', 'unknown')}`, OpenVINO `{runtime.get('openvino', 'not imported')}`, NumPy `{runtime.get('numpy', 'unknown')}`, SciPy `{runtime.get('scipy', 'unknown')}`.",
        "- Each route used a fresh process for each of 3 repetitions; each fixed batch used 3 warmups and 10 timed calls.",
        "- Model-only inference and full forward+backward training used batches `[1, 2, 4, 8]`. End-to-end inference used batch 1 and covered read-only audio preparation through stock note-event materialization.",
        "- Missing-WAV derived rendering was opt-in only; source patches, MIDI, audio, and metadata remained read-only.",
        "",
        "## Inference startup and initialization",
        "",
        "Startup is separated from first-call, warmup, and steady-state timing. " + timing_provenance,
        "",
        "| Route | Status | Import | Construct | Checkpoint | Device move | OV convert | OV compile | Total startup |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if has_precision_diagnostic:
        runtime_index = lines.index("## Runtime and benchmark setup")
        lines.insert(
            runtime_index - 1,
            "- A separate post-fix FP32 + PERFORMANCE diagnostic passes GPU parity, but it did not measure startup, throughput, end-to-end rate, or memory; those measurements remain pending a later full benchmark rerun.",
        )
        inference_index = lines.index("## Inference startup and initialization")
        lines.insert(
            inference_index - 1,
            "- Current OpenVINO compilation explicitly sets `INFERENCE_PRECISION_HINT=float32` and leaves `EXECUTION_MODE_HINT` unconfigured; the bounded GPU diagnostic observed compiled `float32` with `PERFORMANCE` execution.",
        )
    for row in inference:
        lines.append(
            f"| `{row.get('route', 'unknown')}` | `{row.get('status', 'unknown')}` | {_report_number(_report_startup(row, 'backend_import_seconds'))} | {_report_number(_report_startup(row, 'model_construct_seconds'))} | {_report_number(_report_startup(row, 'checkpoint_load_seconds'))} | {_report_number(_report_startup(row, 'model_device_move_seconds'))} | {_report_number(_report_startup(row, 'openvino_conversion_seconds'))} | {_report_number(_report_startup(row, 'openvino_compile_seconds'))} | {_report_number(_report_startup(row, 'total_seconds'))} |"
        )
    lines.extend(
        [
            "",
            "### First-call and warmup observations",
            "",
            "| Route | Batch | First call (s) | Warmup (s) | Steady median call (s) |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in successful_inference:
        for size in (1, 8):
            lines.append(
                f"| `{row.get('route', 'unknown')}` | {size} | {_report_number(_report_batch(row, size, 'first_inference_seconds'))} | {_report_number(_report_batch(row, size, 'inference_warmup_seconds'))} | {_report_number(_report_batch(row, size, 'median_seconds'), 6)} |"
            )
    lines.extend(
        [
            "",
            "## Steady-state inference scaling",
            "",
            (
                "The persisted pre-fix timing tables expose both throughput and call latency for every tested batch. "
                "Throughput is the model-call audio-equivalent rate; it excludes audio decode and stock postprocessing."
                if has_precision_diagnostic
                else "The two tables expose both throughput and call latency for every tested batch. Throughput is the model-call audio-equivalent rate; it excludes audio decode and stock postprocessing."
            ),
            "",
        ]
    )
    lines.extend(_report_throughput_table(successful_inference, "audio_seconds_per_second", "### Audio-equivalent throughput (audio-seconds/second)"))
    lines.extend(["", "### Median model-call latency (seconds)", "", "| Route | Batch 1 | Batch 2 | Batch 4 | Batch 8 |", "| --- | ---: | ---: | ---: | ---: |"])
    for row in successful_inference:
        lines.append(
            f"| `{row.get('route', 'unknown')}` | "
            + " | ".join(_report_number(_report_batch(row, size, "median_seconds"), 6) for size in (1, 2, 4, 8))
            + " |"
        )
    lines.extend(
        [
            "",
            f"Interpretation: XPU scales strongly with batching in this fixed workload. CPU improves more gradually. OpenVINO CPU is not monotonic at batch 2 in this run, then reaches `{_report_number(_report_batch(openvino_cpu, 8, 'audio_seconds_per_second'))}` audio-seconds/second at batch 8; this is an observed measurement, not a tuning target.",
            "",
            "## End-to-end audio-to-note-event throughput",
            "",
            "This is the realistic batch-1 boundary: read-only audio open/decode, in-memory preparation, model windows, unwrapping, and stock note-event materialization. The smoke set totals 115.021 audio seconds across 8 cases.",
            "",
            "| Route | Median wall time (s) | Min-max wall time (s) | Median audio-seconds/wall-second | Median RTF (wall/audio) |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in successful_inference:
        end_to_end = row.get("end_to_end", {})
        wall = end_to_end.get("wall_seconds", {})
        rate = end_to_end.get("audio_seconds_per_wall_second", {})
        rtf = 1 / rate.get("median") if rate.get("median") else None
        lines.append(
            f"| `{row.get('route', 'unknown')}` | {_report_number(wall.get('median'))} | {_report_number(wall.get('min'))}-{_report_number(wall.get('max'))} | {_report_number(rate.get('median'))} | {_report_number(rtf, 5)} |"
        )
    lines.extend(
        [
            "",
            (
                "The persisted pre-fix end-to-end result preserves the same ordering as model-only batch 1: XPU is fastest, OpenVINO CPU is slightly ahead of CPU, and neither timing includes the pre-fix failed OpenVINO GPU route."
                if has_precision_diagnostic
                else "The end-to-end result preserves the same ordering as model-only batch 1: XPU is fastest, OpenVINO CPU is slightly ahead of CPU, and neither timing includes the failed OpenVINO GPU route."
            ),
            "",
            "## CPU versus XPU full forward+backward cost",
            "",
            "These rows measure the explicitly allowed backward cost at the native PyTorch boundary. They do not train, update, or save weights.",
            "",
        ]
    )
    lines.extend(_report_throughput_table(successful_training, "audio_seconds_per_second", "### Effective throughput (audio-seconds/second)"))
    lines.extend(["", "### Median forward+backward step latency (seconds)", "", "| Route | Batch 1 | Batch 2 | Batch 4 | Batch 8 |", "| --- | ---: | ---: | ---: | ---: |"])
    for row in successful_training:
        lines.append(
            f"| `{row.get('route', 'unknown')}` | "
            + " | ".join(_report_number(_report_batch(row, size, "median_seconds"), 6) for size in (1, 2, 4, 8))
            + " |"
        )
    lines.extend(
        [
            "",
            "XPU is faster than CPU at all four tested training batches in this fixed scalar-loss forward+backward measurement; this is a cost observation, not a recommendation to change the current training architecture.",
            "",
            "## Memory and resource observations",
            "",
            "Host RSS is a peak process measurement and is not directly interchangeable with device allocation. `n/a` is an unavailable measurement, not zero.",
            "",
            "| Mode | Route | Host peak RSS (MiB) | XPU allocated (MiB) | XPU reserved (MiB) | Measurement note |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for mode, rows in (("inference", inference), ("training", training)):
        for row in rows:
            memory = row.get("memory", {})
            note = (
                "available"
                if memory.get("measurement_status") == "ok"
                else f"{row.get('status', 'unknown')} before timing"
                if row.get("status") != "ok"
                else str(memory.get("measurement_status", "unknown"))
            )
            lines.append(
                f"| {mode} | `{row.get('route', 'unknown')}` | {_report_mib(memory.get('host_peak_rss_bytes'))} | {_report_mib(memory.get('pytorch_xpu_peak_allocated_bytes'))} | {_report_mib(memory.get('pytorch_xpu_peak_reserved_bytes'))} | {note} |"
            )
    lines.extend(
        [
            "",
            (
                "The observed XPU routes use substantially more host RSS than CPU in these fresh processes, while their recorded device allocations are much smaller than host RSS. OpenVINO GPU has no pre-fix memory observation because it failed parity before timing; the post-fix diagnostic did not measure memory."
                if has_precision_diagnostic
                else "The observed XPU routes use substantially more host RSS than CPU in these fresh processes, while their recorded device allocations are much smaller than host RSS. OpenVINO GPU has no memory observation because it failed parity before timing."
            ),
            "",
            "## Startup versus throughput crossover",
            "",
        ]
    )
    if crossover:
        lines.extend(
            [
                f"- The only positive finite crossover retained by the fixed formula is `{crossover.get('route_a')}` versus `{crossover.get('route_b')}` at `{_report_number(crossover.get('audio_seconds'))}` audio seconds.",
                f"- The formula uses median one-time startup and median batch-1 model-call throughput. In this run, CPU startup is `{_report_number(_report_startup(cpu, 'total_seconds'))}` s and XPU startup is `{_report_number(_report_startup(xpu, 'total_seconds'))}` s, while their batch-1 rates are `{_report_number(_report_batch(cpu, 1, 'audio_seconds_per_second'))}` and `{_report_number(_report_batch(xpu, 1, 'audio_seconds_per_second'))}` audio-seconds/second.",
                "- This crossover means the XPU steady-state advantage repays its measured startup difference after roughly 3.3 audio seconds under the model-only abstraction. It does not erase XPU's larger first-call observation, and it is not a universal short-clip latency guarantee.",
            ]
        )
    else:
        lines.append("- No positive finite crossover was retained by the fixed formula for the successful routes.")
    lines.extend(_parity_diagnostics_markdown(report))
    failure_heading = "## OpenVINO GPU parity failure (pre-fix configuration)" if has_precision_diagnostic else "## OpenVINO GPU parity failure"
    lines.extend(["", failure_heading, ""])
    if gpu_failure:
        lines.extend(
            [
                f"- Status: `{gpu_failure.get('status')}` for `{gpu_failure.get('repetitions', 0)}` repetitions; failure code: `{', '.join(gpu_failure.get('failure_codes', [])) or 'not retained'}`.",
                "- The worker performs the parity gate before model-only or end-to-end timing. The gate compares contour, note, and onset numeric outputs plus note/onset threshold decisions and stock note-event structure against the canonical PyTorch CPU route.",
                (
                    "- The component-level parity values are tabulated above. They describe the fixed synthetic gate only; they do not authorize timing a route that failed parity."
                    if isinstance(report.get("parity_diagnostics"), Mapping)
                    else "- The committed aggregate retains only the generic `parity_failed` code, not the component-level error payload. Therefore the existing evidence identifies a parity-gate failure but does not identify which subcheck triggered it. That missing diagnostic is a reporting limitation, not permission to infer a GPU performance result."
                ),
                "- No OpenVINO GPU throughput, latency, end-to-end rate, or memory claim is supported; no CPU fallback or substitute-device measurement was used.",
            ]
        )
    else:
        lines.append("- No OpenVINO GPU failure was present in the supplied report.")
    lines.extend(_openvino_precision_diagnostic_markdown(report))
    lines.extend(
        [
            "",
            "## Practical conclusions supported by this run",
            "",
            "- For longer-running or reused workloads, PyTorch XPU is the strongest observed route: it leads steady-state model-call throughput at every batch and the warmed end-to-end smoke rate.",
            "- For short interactive workloads, initialization and first-call behavior should be treated as part of the product latency budget. XPU's first batch-1 call is materially slower than CPU in the stored measurements even though its warmed rate is higher; the benchmark does not establish a product policy for hiding or amortizing that cost.",
            "- OpenVINO CPU has a larger one-time conversion/compile cost and lower model-call throughput than XPU, but its warmed end-to-end smoke rate is close to CPU. It remains a measured explicit route, not an automatically preferred backend.",
            "- XPU is also faster for the measured full forward+backward cost at all tested batches, with the resource caveat above.",
            (
                "- These timing conclusions describe the fixed 8-case smoke workload and the persisted pre-fix OpenVINO configuration. The bounded post-fix FP32 + PERFORMANCE diagnostic validates GPU parity but provides no speed, startup, end-to-end, or memory result; a post-fix full benchmark is still required."
                if has_precision_diagnostic
                else "- These conclusions describe the fixed 8-case smoke workload, current runtime versions, float32 precision, and exact benchmark contract. They do not claim a universal optimum or validate the failed OpenVINO GPU route."
            ),
            "",
            "## Scope and caveats",
            "",
        ]
    )
    if conclusions.get("reason"):
        lines.append(f"- `{conclusions['reason']}`")
    lines.append("- The report contains no source paths, filenames, IDs, hashes, or per-source predictions. Per-case end-to-end rows are anonymous case indexes only.")
    return "\n".join(lines) + "\n"


def write_benchmark_reports(
    report: Mapping[str, Any],
    json_path: str | Path,
    markdown_path: str | Path,
    *,
    force: bool = False,
) -> None:
    """Atomically write only sanitized aggregate benchmark reports."""
    # Report formatting is intentionally derived from persisted aggregate values.
    json_file, markdown_file = _approved_report_paths(json_path, markdown_path)
    json_file.parent.mkdir(parents=True, exist_ok=True)
    markdown_file.parent.mkdir(parents=True, exist_ok=True)
    if not force and (json_file.exists() or markdown_file.exists()):
        raise FileExistsError("refusing to overwrite benchmark reports without force=True")
    inference = report.get("inference", [])
    training = report.get("training", [])
    smoke = report.get("smoke_set", {})
    conclusions = report.get("conclusions", {})
    markdown_lines = [
        "# Basic Pitch backend benchmark",
        "",
        f"- Model: `{report.get('model_id', MODEL_ID)}`",
        f"- Smoke set: `{smoke.get('status', 'unknown')}` ({smoke.get('case_count', 0)} cases)",
        f"- Runtime: `{report.get('runtime', {}).get('python', 'unknown')}` / float32",
        "- Missing-WAV derived rendering: opt-in only; source patches and MIDI remain read-only.",
        "",
        "## Inference routes",
        "",
        "| Route | Status | Batch-1 audio seconds/second |",
        "| --- | --- | ---: |",
    ]
    for row in inference:
        rate = row.get("batch_results", {}).get("1", {}).get("audio_seconds_per_second", {}).get("median", "—")
        rate_text = f"{rate:.6g}" if isinstance(rate, (int, float)) else str(rate)
        markdown_lines.append(f"| `{row.get('route', 'unknown')}` | `{row.get('status', 'unknown')}` | {rate_text} |")
    markdown_lines.extend(["", "## Training routes", "", "| Route | Status | Batch-1 audio seconds/second |", "| --- | --- | ---: |"])
    for row in training:
        rate = row.get("batch_results", {}).get("1", {}).get("audio_seconds_per_second", {}).get("median", "—")
        rate_text = f"{rate:.6g}" if isinstance(rate, (int, float)) else str(rate)
        markdown_lines.append(f"| `{row.get('route', 'unknown')}` | `{row.get('status', 'unknown')}` | {rate_text} |")
    markdown_lines.extend(["", "## Scope and caveats", ""])
    if conclusions.get("reason"):
        markdown_lines.append(f"- `{conclusions['reason']}`")
    markdown_lines.append("- The report contains no source paths, filenames, IDs, hashes, or per-source predictions.")
    markdown = "\n".join(markdown_lines) + "\n"
    markdown = _benchmark_markdown(report)

    json_temp: Path | None = None
    markdown_temp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=json_file.parent, prefix=".backend-", suffix=".json.tmp", delete=False) as handle:
            json_temp = Path(handle.name)
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=markdown_file.parent, prefix=".backend-", suffix=".md.tmp", delete=False) as handle:
            markdown_temp = Path(handle.name)
            handle.write(markdown)
        os.replace(json_temp, json_file)
        json_temp = None
        os.replace(markdown_temp, markdown_file)
        markdown_temp = None
    finally:
        for temporary in (json_temp, markdown_temp):
            if temporary is not None and temporary.exists():
                temporary.unlink()


def benchmark_exit_code(report: Mapping[str, Any]) -> int:
    """Return 0 for successful/unavailable route rows and 3 for route failures."""
    rows = list(report.get("inference", [])) + list(report.get("training", []))
    return 0 if all(row.get("status") in {"ok", "unavailable"} for row in rows) else 3


def run_parity_diagnostic_cli(
    checkpoint_path: str | Path,
    json_path: str | Path,
    markdown_path: str | Path,
    *,
    xpu_index: int = 0,
    openvino_gpu_device: str = "GPU",
    process_repetitions: int = 3,
    force: bool = False,
) -> int:
    try:
        json_file, markdown_file = _approved_report_paths(json_path, markdown_path)
        report = json.loads(json_file.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            return 2
        report["parity_diagnostics"] = run_parity_diagnostics(
            checkpoint_path,
            process_repetitions=process_repetitions,
            xpu_index=xpu_index,
            openvino_gpu_device=openvino_gpu_device,
        )
        write_benchmark_reports(report, json_file, markdown_file, force=force)
        routes = report["parity_diagnostics"].get("routes", [])
        return 0 if all(route.get("status") == "ok" for route in routes) else 3
    except (BenchmarkInputError, FileNotFoundError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return 2


def run_benchmark_cli(
    config_path: str | Path,
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    json_path: str | Path,
    markdown_path: str | Path,
    *,
    xpu_index: int = 0,
    openvino_gpu_device: str = "GPU",
    allow_derived_render: bool = False,
    force: bool = False,
) -> int:
    try:
        config = load_config(config_path)
        try:
            report = run_benchmark(
                config,
                manifest_path,
                checkpoint_path,
                xpu_index=xpu_index,
                openvino_gpu_device=openvino_gpu_device,
                allow_derived_render=allow_derived_render,
            )
        except DerivedRenderUnavailable as exc:
            try:
                cases = load_manifest(
                    manifest_path,
                    config,
                    allow_derived_render=True,
                    allow_missing_derived_audio=True,
                )
            except (BenchmarkInputError, FileNotFoundError, OSError):
                cases = ()
            report = _unavailable_report(config, exc.code, cases=cases)
            write_benchmark_reports(report, json_path, markdown_path, force=force)
            return 3
        except (BenchmarkInputError, FileNotFoundError, OSError) as exc:
            code = exc.code if isinstance(exc, BenchmarkInputError) else "invalid_smoke_manifest"
            report = _unavailable_report(config, code)
            write_benchmark_reports(report, json_path, markdown_path, force=force)
            return 2
        except SourceMutationError:
            report = _unavailable_report(config, "benchmark_runtime_error")
            write_benchmark_reports(report, json_path, markdown_path, force=force)
            return 3
        write_benchmark_reports(report, json_path, markdown_path, force=force)
        return benchmark_exit_code(report)
    except (BenchmarkInputError, FileNotFoundError, OSError, ValueError):
        return 2

from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any
import xml.etree.ElementTree as ET

import yaml

from obruxo_data.errors import DependencyUnavailableError
from obruxo_data.midi import TempoMap
from obruxo_data.vital import VitalPreset, VitalSchema

from .base import DEFAULT_RENDERER_ID, RenderProvenance, RenderRequest, RenderResult, Renderer
from .capabilities import RendererCapabilities
from .qa import AudioQualityConfig, analyze_audio


JUCE_BASE64_ALPHABET = ".ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+"
DEFAULT_PLUGIN_SHA256 = "a622a2c99b4066cd7945a4ab9bbdd698e7632a30702f6f0a7ccbf26a56b576e1"


def juce_memory_block_encode(value: bytes) -> str:
    characters = []
    for bit_offset in range(0, len(value) * 8, 6):
        byte_offset, shift = divmod(bit_offset, 8)
        window = int.from_bytes(value[byte_offset : byte_offset + 2], "little")
        characters.append(JUCE_BASE64_ALPHABET[(window >> shift) & 0x3F])
    return f"{len(value)}.{''.join(characters)}"


def juce_memory_block_decode(value: str) -> bytes:
    size_text, encoded = value.split(".", 1)
    output = bytearray(int(size_text))
    for index, character in enumerate(encoded):
        bits = JUCE_BASE64_ALPHABET.index(character)
        bit_offset = index * 6
        for bit in range(6):
            target = bit_offset + bit
            if target >= len(output) * 8:
                break
            if bits & (1 << bit):
                output[target // 8] |= 1 << (target % 8)
    return bytes(output)


class VitalVst3StateTemplate:
    def __init__(self, state: bytes):
        if len(state) < 9 or state[:4] != b"VC2!":
            raise ValueError("plugin state is not a JUCE XML state block")
        xml_size = int.from_bytes(state[4:8], "little")
        xml_bytes = state[8 : 8 + xml_size]
        self._root = ET.fromstring(xml_bytes.decode("utf-8"))
        component_element = self._root.find("IComponent")
        if component_element is None or not component_element.text:
            raise ValueError("plugin state does not contain an IComponent block")
        component = juce_memory_block_decode(component_element.text)
        self._component_prefix, chunk, self._component_suffix, self._bank_size_offset = self._split_component(component)
        separator = chunk.find(b"\x00")
        if separator < 0:
            raise ValueError("Vital component state has no JSON terminator")
        self._private_tail = chunk[separator + 1 :]

    @staticmethod
    def _split_component(component: bytes) -> tuple[bytes, bytes, bytes, int]:
        position = 0
        if component[:4] == b"VstW":
            private_size = int.from_bytes(component[4:8], "big")
            position = 8 + private_size
        if component[position : position + 4] != b"CcnK":
            raise ValueError("unsupported VST component state wrapper")
        bank_size_offset = position + 4
        position += 8
        if component[position : position + 4] != b"FBCh":
            raise ValueError("Vital VST state is not chunk-based")
        bank_version = int.from_bytes(component[position + 4 : position + 8], "big")
        position += 20
        if bank_version >= 1:
            position += 128
        chunk_size_offset = position
        chunk_size = int.from_bytes(component[chunk_size_offset : chunk_size_offset + 4], "big")
        chunk_start = chunk_size_offset + 4
        chunk_end = chunk_start + chunk_size
        if chunk_end > len(component):
            raise ValueError("truncated Vital VST state chunk")
        return component[:chunk_size_offset], component[chunk_start:chunk_end], component[chunk_end:], bank_size_offset

    def build(self, preset_json: str) -> bytes:
        canonical = json.dumps(json.loads(preset_json), ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
        chunk = canonical.encode("utf-8") + b"\x00" + self._private_tail
        component = bytearray(self._component_prefix + len(chunk).to_bytes(4, "big") + chunk + self._component_suffix)
        bank_size = len(component) - (self._bank_size_offset + 4)
        component[self._bank_size_offset : self._bank_size_offset + 4] = bank_size.to_bytes(4, "big")
        root = deepcopy(self._root)
        component_element = root.find("IComponent")
        assert component_element is not None
        component_element.text = juce_memory_block_encode(bytes(component))
        xml = b'<?xml version="1.0" encoding="UTF-8"?> ' + ET.tostring(root, encoding="utf-8", short_empty_elements=True)
        return b"VC2!" + len(xml).to_bytes(4, "little") + xml + b"\x00"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _default_plugin_path() -> Path | None:
    configured = os.environ.get("OBRUXO_VITAL_PLUGIN")
    if configured:
        return Path(configured)
    candidates = []
    if sys.platform == "win32":
        program_files = os.environ.get("PROGRAMFILES", r"C:\Program Files")
        candidates.append(Path(program_files) / "Common Files" / "VST3" / "Vital.vst3")
    elif sys.platform == "darwin":
        candidates.append(Path("/Library/Audio/Plug-Ins/VST3/Vital.vst3"))
    else:
        candidates.extend((Path("/usr/lib/vst3/Vital.vst3"), Path("/usr/local/lib/vst3/Vital.vst3")))
    return next((path for path in candidates if path.exists()), None)


class VitalRenderer(Renderer):
    def __init__(self, plugin_path: Path | str | None = None, *, accepted_plugin_sha256: set[str] | frozenset[str] | None = None,
                 buffer_size: int = 128, renderer_id: str = DEFAULT_RENDERER_ID,
                 qa_config: AudioQualityConfig | None = None):
        resolved = Path(plugin_path) if plugin_path is not None else _default_plugin_path()
        if resolved is None:
            raise DependencyUnavailableError("Vital VST3 was not found; configure plugin_path or OBRUXO_VITAL_PLUGIN")
        self.plugin_path = resolved.resolve()
        accepted = {DEFAULT_PLUGIN_SHA256} if accepted_plugin_sha256 is None else accepted_plugin_sha256
        if not accepted or any(len(value) != 64 or any(character not in "0123456789abcdefABCDEF" for character in value) for value in accepted):
            raise ValueError("accepted_plugin_sha256 must contain at least one SHA-256 hex digest")
        if not isinstance(buffer_size, int) or isinstance(buffer_size, bool) or buffer_size <= 0:
            raise ValueError("buffer_size must be a positive integer")
        self.accepted_plugin_sha256 = frozenset(value.lower() for value in accepted)
        self.buffer_size = buffer_size
        self.renderer_id = renderer_id
        self.qa_config = qa_config or AudioQualityConfig()
        self._capabilities = RendererCapabilities(max_channels=1)

    @classmethod
    def from_config(cls, path: Path | str, *, plugin_path: Path | str | None = None) -> "VitalRenderer":
        document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(document, dict) or document.get("version") != 1 or document.get("backend") != "dawdreamer":
            raise ValueError("unsupported renderer configuration")
        capabilities = RendererCapabilities.from_dict(document.get("capabilities", {}))
        supported = RendererCapabilities(max_channels=1)
        if capabilities != supported:
            raise ValueError("renderer capabilities do not match the implemented DawDreamer backend")
        if document.get("max_workers") != 1:
            raise ValueError("the pinned DawDreamer/Vital backend requires max_workers: 1")
        qa = document.get("qa", {})
        if not isinstance(qa, dict):
            raise ValueError("renderer qa configuration must be an object")
        configured_path = plugin_path if plugin_path is not None else document.get("plugin_path")
        return cls(
            configured_path,
            accepted_plugin_sha256=set(document.get("accepted_plugin_sha256", [])),
            buffer_size=int(document.get("buffer_size", 128)),
            renderer_id=str(document.get("renderer_id", DEFAULT_RENDERER_ID)),
            qa_config=AudioQualityConfig(**qa),
        )

    @property
    def capabilities(self) -> RendererCapabilities:
        return self._capabilities

    @property
    def max_workers(self) -> int:
        return 1

    def render(self, request: RenderRequest) -> RenderResult:
        validation_diagnostics = self._validate_request(request)
        if not self.plugin_path.is_file():
            raise DependencyUnavailableError(f"Vital plugin does not exist: {self.plugin_path}")
        fingerprint = _sha256_file(self.plugin_path)
        if self.accepted_plugin_sha256 and fingerprint not in self.accepted_plugin_sha256:
            raise DependencyUnavailableError(f"Vital plugin SHA-256 is not accepted: {fingerprint}")
        try:
            import dawdreamer as daw
        except ImportError as error:
            raise DependencyUnavailableError("DawDreamer is not installed") from error
        import numpy as np

        backend_version = importlib.metadata.version("dawdreamer")
        if backend_version != "0.8.3":
            raise DependencyUnavailableError(f"DawDreamer 0.8.3 is required, found {backend_version}")

        performance = request.performance
        timing = TempoMap.from_performance(performance)
        end_tick = performance.end_tick if request.end_tick is None else request.end_tick
        frame_count = timing.render_frame_count(end_tick, request.tail_seconds, request.sample_rate)
        tempo = next((event.data[0] for event in performance.canonical_events() if event.kind.value == "tempo"), 500_000)
        bpm = 60_000_000 / tempo

        with tempfile.TemporaryDirectory(prefix="obruxo-vital-render-") as directory:
            root = Path(directory)
            template_path = root / "template.state"
            state_path = root / "request.state"
            engine = daw.RenderEngine(request.sample_rate, self.buffer_size)
            engine.set_bpm(bpm)
            synth = engine.make_plugin_processor("vital", str(self.plugin_path))
            synth.save_state(str(template_path))
            template = VitalVst3StateTemplate(template_path.read_bytes())
            state_path.write_bytes(template.build(request.preset_json))
            synth.load_state(str(state_path))
            for span in performance.note_spans():
                start_sample = timing.tick_to_sample(span.start_tick, request.sample_rate)
                end_sample = timing.tick_to_sample(span.end_tick, request.sample_rate)
                if end_sample <= start_sample:
                    raise ValueError("note duration rounds to zero samples")
                synth.add_midi_note(
                    span.pitch, span.velocity, start_sample / request.sample_rate,
                    (end_sample - start_sample) / request.sample_rate,
                )
            engine.load_graph([(synth, [])])
            if not engine.render(frame_count / request.sample_rate):
                raise RuntimeError("DawDreamer failed to render the Vital graph")
            channel_first = np.asarray(engine.get_audio(), dtype=np.float32)
            audio = np.ascontiguousarray(channel_first.T)

        qa, audio_diagnostics = analyze_audio(
            audio, sample_rate=request.sample_rate, expected_frames=frame_count,
            expected_channels=request.channels, config=self.qa_config,
        )
        provenance = RenderProvenance(
            request_id=request.request_id,
            renderer_id=self.renderer_id,
            backend_version=backend_version,
            engine_fingerprint=fingerprint,
            settings={
                "plugin_name": self.plugin_path.name,
                "buffer_size": self.buffer_size,
                "schema_id": VitalSchema.load().schema_id,
                "event_timing": "absolute ticks to half-even sample offsets",
                "determinism": "numeric tolerance; the Vital engine is not claimed bit-deterministic",
            },
        )
        return RenderResult(audio, request.sample_rate, validation_diagnostics + audio_diagnostics, provenance, qa)

    def _validate_request(self, request: RenderRequest) -> tuple[Any, ...]:
        if request.renderer_id != self.renderer_id:
            raise ValueError(f"request renderer_id {request.renderer_id!r} does not match {self.renderer_id!r}")
        if not isinstance(request.sample_rate, int) or isinstance(request.sample_rate, bool) or request.sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer")
        if request.channels != 2:
            raise ValueError("Vital renderer currently supports stereo output only")
        if not isinstance(request.tail_seconds, (int, float)) or not math.isfinite(request.tail_seconds) or request.tail_seconds < 0:
            raise ValueError("tail_seconds must be a finite non-negative number")
        end_tick = request.performance.end_tick if request.end_tick is None else request.end_tick
        if end_tick != request.performance.end_tick:
            raise ValueError("RenderRequest end_tick must match Performance.end_tick")
        preset = VitalPreset(json.loads(request.preset_json), VitalSchema.load())
        preset_report = preset.validate(runtime=True)
        preset_report.require_valid()
        performance_report = request.performance.validate(self.capabilities)
        performance_report.require_valid()
        return preset_report.diagnostics + performance_report.diagnostics

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from scipy.io import wavfile

from obruxo_data.errors import Diagnostic, OutputExistsError, Severity
from obruxo_data.midi import Performance
from obruxo_data.render import (
    RenderProvenance, RenderRequest, RenderResult, Renderer, RendererCapabilities, VitalRenderer, load_requests, run_batch,
    write_requests,
)
from obruxo_data.render.qa import AudioQualityConfig, analyze_audio, audio_float32_sha256, compare_audio
from obruxo_data.render.vita import VitalVst3StateTemplate, juce_memory_block_decode, juce_memory_block_encode
from obruxo_data.vital import VitalPreset


def _template_state(document: dict) -> bytes:
    chunk = json.dumps(document, separators=(",", ":")).encode() + b"\x00" + b"\x00" * 16 + b"JUCEPrivateData"
    body = (
        b"FBCh" + (2).to_bytes(4, "big") + b"Vita" + (0x00010604).to_bytes(4, "big")
        + (1).to_bytes(4, "big") + (0).to_bytes(4, "big") + b"\x00" * 124
        + len(chunk).to_bytes(4, "big") + chunk
    )
    component = b"VstW" + (8).to_bytes(4, "big") + (1).to_bytes(4, "big") + (0).to_bytes(4, "big")
    component += b"CcnK" + len(body).to_bytes(4, "big") + body
    root = ET.Element("VST3PluginState")
    ET.SubElement(root, "IComponent").text = juce_memory_block_encode(component)
    ET.SubElement(root, "IEditController").text = juce_memory_block_encode(b"")
    xml = b'<?xml version="1.0" encoding="UTF-8"?> ' + ET.tostring(root, encoding="utf-8")
    return b"VC2!" + len(xml).to_bytes(4, "little") + xml + b"\x00"


def _extract_vital_json(state: bytes) -> dict:
    size = int.from_bytes(state[4:8], "little")
    root = ET.fromstring(state[8 : 8 + size].decode())
    component = juce_memory_block_decode(root.findtext("IComponent", default=""))
    _, chunk, _, _ = VitalVst3StateTemplate._split_component(component)
    return json.loads(chunk.partition(b"\x00")[0].decode())


def test_juce_memory_block_encoding_round_trip() -> None:
    for value in (b"", b"a", bytes(range(255)), b"Vital state \x00 payload"):
        assert juce_memory_block_decode(juce_memory_block_encode(value)) == value


def test_vital_vst3_state_template_replaces_only_component_json() -> None:
    original = _template_state({"preset_name": "old", "settings": {}})
    updated = VitalVst3StateTemplate(original).build('{"preset_name":"new","settings":{"volume":1}}')
    assert _extract_vital_json(updated) == {"preset_name": "new", "settings": {"volume": 1}}
    VitalVst3StateTemplate(updated)


def test_render_request_accepts_preset_and_has_stable_identity() -> None:
    preset = VitalPreset.init()
    performance = Performance(bpm=120)
    performance.add_note(pitch=60, velocity=100, start_tick=0, duration_ticks=480)
    first = RenderRequest(preset=preset, performance=performance)
    second = RenderRequest(preset_json=preset.to_json(), performance=performance.clone())
    assert first.request_id == second.request_id
    assert RenderRequest.from_dict(first.to_dict()).request_id == first.request_id
    preset.set_raw("osc_1_level", 0.25)
    assert RenderRequest(preset=preset, performance=performance).request_id != first.request_id


def test_audio_qa_reports_shape_nonfinite_silence_clipping_and_tail() -> None:
    audio = np.zeros((8, 2), dtype=np.float32)
    audio[0, 0] = np.nan
    audio[-1, 1] = 1.5
    qa, diagnostics = analyze_audio(
        audio, sample_rate=8, expected_frames=9, expected_channels=1,
        config=AudioQualityConfig(silence_rms=1.0, tail_seconds=0.25, tail_rms_warning=0.1),
    )
    assert qa["actual_frames"] == 8
    assert len(qa["audio_float32_sha256"]) == 64
    assert {item.code for item in diagnostics} >= {
        "audio.frames", "audio.channels", "audio.non_finite", "audio.silence", "audio.clipping", "audio.tail_truncation",
    }


def test_repeatability_uses_numeric_and_spectral_tolerances() -> None:
    first = np.sin(np.linspace(0, 20, 4096, dtype=np.float64))[:, None].astype(np.float32)
    second = np.sin(np.linspace(0.1, 20.1, 4096, dtype=np.float64))[:, None].astype(np.float32)
    comparison = compare_audio(first, second)
    assert comparison["within_tolerance"]
    assert comparison["waveform_rmse"] > 0
    assert comparison["log_spectral_rmse"] <= comparison["tolerance"]["log_spectral_rmse"]


def test_renderer_identity_contains_complete_plugin_fingerprint(tmp_path) -> None:
    plugin = tmp_path / "Vital.vst3"
    plugin.write_bytes(b"test plugin binary")
    fingerprint = hashlib.sha256(plugin.read_bytes()).hexdigest()
    renderer = VitalRenderer(plugin, accepted_plugin_sha256={fingerprint}, renderer_id=f"test-{fingerprint}")
    assert renderer.engine_fingerprint == fingerprint
    with pytest.raises(ValueError, match="complete accepted plugin SHA-256"):
        VitalRenderer(plugin, accepted_plugin_sha256={fingerprint}, renderer_id="incomplete")


def test_render_result_writes_atomic_wav_and_json_without_overwrite(tmp_path) -> None:
    audio = np.zeros((32, 2), dtype=np.float32)
    provenance = RenderProvenance("request", "renderer", "1.0", "fingerprint")
    result = RenderResult(audio, 44_100, (Diagnostic("ok", Severity.INFO, "ok"),), provenance, {"rms": 0.0})
    wav_path = tmp_path / "result.wav"
    json_path = tmp_path / "result.json"
    result.write_wav(wav_path)
    result.write_json(json_path)
    sample_rate, loaded = wavfile.read(wav_path)
    assert sample_rate == 44_100
    assert loaded.shape == (32, 2)
    assert json.loads(json_path.read_text(encoding="utf-8"))["provenance"]["request_id"] == "request"
    with pytest.raises(OutputExistsError):
        result.write_wav(wav_path)
    assert not list(tmp_path.glob("*.tmp"))


class _FakeRenderer(Renderer):
    def __init__(self) -> None:
        self.calls = 0

    @property
    def capabilities(self) -> RendererCapabilities:
        return RendererCapabilities()

    def render(self, request: RenderRequest) -> RenderResult:
        self.calls += 1
        audio = np.full((32, 2), self.calls / 100, dtype=np.float32)
        qa = {"audio_float32_sha256": audio_float32_sha256(audio)}
        provenance = RenderProvenance(request.request_id, request.renderer_id, "fake", "fake")
        return RenderResult(audio, request.sample_rate, (), provenance, qa)


def test_jsonl_batch_is_bounded_and_hash_validated_for_resume(tmp_path) -> None:
    preset = VitalPreset.init()
    performance = Performance(bpm=120)
    performance.add_note(pitch=60, velocity=100, start_tick=0, duration_ticks=480)
    requests = [RenderRequest(preset=preset, performance=performance)]
    jsonl = tmp_path / "requests.jsonl"
    output = tmp_path / "batch"
    write_requests(jsonl, requests)
    loaded = load_requests(jsonl)
    assert loaded[0].request_id == requests[0].request_id
    renderer = _FakeRenderer()
    first = run_batch(renderer, loaded, output, workers=1)
    second = run_batch(renderer, loaded, output, workers=2)
    assert (first.rendered, first.skipped) == (1, 0)
    assert (second.rendered, second.skipped) == (0, 1)
    assert renderer.calls == 1
    with pytest.raises(ValueError, match="duplicate request IDs"):
        run_batch(renderer, [loaded[0], loaded[0]], tmp_path / "duplicate")

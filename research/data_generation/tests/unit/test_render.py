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
from obruxo_data.render.vita import (
    VitalVst3StateTemplate, _add_note_events, _render_after_preroll, _render_exact_frames, juce_memory_block_decode, juce_memory_block_encode, native_preset_document,
    verify_loaded_controls, verify_loaded_scalars,
)
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


@pytest.mark.parametrize("frames", [1, 123480, 154350])
def test_native_duration_truncation_preserves_exact_requested_sample_prefix(frames):
    class TruncatingEngine:
        def render(self, seconds):
            count = int(seconds * 44100)
            self.audio = np.stack((np.arange(count), -np.arange(count))).astype(np.float32)
            return True

        def get_audio(self):
            return self.audio

    engine = TruncatingEngine()
    assert int((123480 / 44100) * 44100) == 123479  # The actual 150-BPM failure.
    output = _render_exact_frames(engine, frames, 44100, 2)
    assert output.shape == (frames, 2) and output.dtype == np.float32 and output.flags.c_contiguous
    assert np.array_equal(output, np.stack((np.arange(frames), -np.arange(frames)), axis=1))


@pytest.mark.parametrize("shape", [(2, 99), (1, 101), (101,)])
def test_exact_frame_render_refuses_short_or_malformed_output(shape):
    class BrokenEngine:
        def render(self, seconds):
            return True

        def get_audio(self):
            return np.zeros(shape, dtype=np.float32)

    with pytest.raises(RuntimeError, match="expected at least"):
        _render_exact_frames(BrokenEngine(), 100, 44100, 2)


def test_vital_vst3_state_template_replaces_only_component_json() -> None:
    original = _template_state({"preset_name": "old", "settings": {}})
    updated = VitalVst3StateTemplate(original).build('{"preset_name":"new","settings":{"volume":1}}')
    assert _extract_vital_json(updated) == {"preset_name": "new", "settings": {"volume": 1}}
    VitalVst3StateTemplate(updated)


def test_vita_development_version_is_adapted_only_for_reviewed_plugin() -> None:
    template = VitalVst3StateTemplate(_template_state({"synth_version": "1.6.4", "settings": {}}))
    preset = VitalPreset.init()
    preset.set_raw("osc_1_transpose", -12)
    adapted = _extract_vital_json(template.build(preset.to_json()))
    assert adapted["synth_version"] == "1.6.4"
    assert all(table["version"] == "1.6.4" for table in adapted["settings"]["wavetables"])
    assert adapted["settings"]["osc_1_transpose"] == -12
    assert preset.to_dict()["synth_version"] == "99999.9.9"
    unknown = VitalVst3StateTemplate(_template_state({"synth_version": "1.7.0", "settings": {}}))
    with pytest.raises(ValueError, match="reviewed"):
        unknown.build(preset.to_json())


def test_native_preset_document_bridges_only_the_reviewed_version() -> None:
    preset = VitalPreset.init()
    document, diagnostics = native_preset_document(preset.to_json(), "1.6.4")
    assert document["synth_version"] == "1.6.4"
    assert all(table["version"] == "1.6.4" for table in document["settings"]["wavetables"])
    assert [item.code for item in diagnostics] == ["vital.native.version_bridge"]
    with pytest.raises(ValueError, match="reviewed Vital 1.6.4"):
        native_preset_document(preset.to_json(), "1.7.0")
    unknown = preset.to_dict()
    unknown["synth_version"] = "1.7.0"
    with pytest.raises(ValueError, match="pinned native"):
        native_preset_document(json.dumps(unknown), "1.6.4")


def test_native_readback_checks_lfo_and_modulation_routes() -> None:
    expected = {"settings": {"osc_1_level": 0.3, "lfos": [], "modulations": []}}
    assert verify_loaded_controls(expected, expected) == 1
    for key in ("lfos", "modulations"):
        loaded = {"settings": {**expected["settings"], key: [{}]}}
        with pytest.raises(RuntimeError, match=f"requested {key}"):
            verify_loaded_controls(expected, loaded)


def test_historical_training_profile_preserves_bridge_and_readback_tolerance() -> None:
    original = VitalPreset.init().to_json()
    legacy, _ = native_preset_document(original, "1.6.4", legacy_training=True)
    current, _ = native_preset_document(original, "1.6.4")
    assert legacy["synth_version"] == "1.0.8" and current["synth_version"] == "1.6.4"
    assert legacy["settings"] == json.loads(original)["settings"]
    requested = {"settings": {"osc_1_level": 1.0, "lfos": [], "modulations": []}}
    loaded = {"settings": {**requested["settings"], "osc_1_level": 1.000005}}
    assert verify_loaded_controls(requested, loaded, legacy_training=True) == 1
    with pytest.raises(RuntimeError, match="scalar"):
        verify_loaded_controls(requested, loaded)
    with pytest.raises(RuntimeError, match="modulations"):
        verify_loaded_controls(requested, {"settings": {**requested["settings"], "modulations": [{}]}}, legacy_training=True)


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


def test_preroll_is_hashed_and_legacy_request_identity_stays_unchanged():
    preset, performance = VitalPreset.init(), Performance()
    performance.add_note(pitch=60, velocity=90, start_tick=0, duration_ticks=480)
    legacy = RenderRequest(preset=preset, performance=performance)
    assert "preroll_frames" not in legacy.identity_payload()
    warm = RenderRequest(preset=preset, performance=performance, preroll_frames=8192)
    assert warm.request_id != legacy.request_id
    assert RenderRequest.from_dict(warm.to_dict()).request_id == warm.request_id
    for invalid in (-1, True, .5, 44101):
        with pytest.raises(ValueError, match="preroll_frames"):
            RenderRequest(preset=preset, performance=performance, preroll_frames=invalid)


def test_preroll_shifts_notes_without_changing_duration_and_keeps_first_attack_frame():
    from obruxo_data.midi import TempoMap
    preset, performance = VitalPreset.init(), Performance(bpm=110)
    performance.add_note(pitch=60, velocity=90, start_tick=0, duration_ticks=480)
    performance.add_note(pitch=67, velocity=80, start_tick=720, duration_ticks=240)
    request = RenderRequest(preset=preset, performance=performance, preroll_frames=8192)
    calls = []
    class Synth:
        def add_midi_note(self, *values):
            calls.append(values)
    timing = TempoMap.from_performance(performance)
    _add_note_events(Synth(), request, timing)
    for call, note in zip(calls, performance.note_spans(), strict=True):
        start, end = timing.tick_to_sample(note.start_tick, 44100), timing.tick_to_sample(note.end_tick, 44100)
        assert call == (note.pitch, note.velocity, (start + 8192 + .125) / 44100, (end - start + .125) / 44100)
        for quantize in (int, round):
            assert quantize(call[2] * 44100) == start + 8192
            assert quantize(call[3] * 44100) == end - start
            assert quantize((call[2] + call[3]) * 44100) == end + 8192
    class Engine:
        def render(self, seconds):
            self.audio = np.zeros((2, round(seconds * 44100)), dtype=np.float32)
            self.audio[:, 8192] = (.25, -.5)
            self.audio[:, 8192 + 99] = (.1, .2)
            return True
        def get_audio(self):
            return self.audio
    body = _render_after_preroll(Engine(), 100, 44100, 2, 8192)
    assert body.shape == (100, 2) and body.flags.c_contiguous
    np.testing.assert_array_equal(body[0], np.array([.25, -.5], dtype=np.float32))
    np.testing.assert_array_equal(body[-1], np.array([.1, .2], dtype=np.float32))


def test_preroll_moves_all_expressive_midi_by_the_same_exact_sample_offset(tmp_path):
    from obruxo_data.midi import EventKind, MidiEvent
    from obruxo_data.render.midi_transport import write_sample_midi
    performance = Performance(bpm=110)
    performance.add_note(pitch=60, velocity=90, start_tick=0, duration_ticks=480, release_velocity=37)
    performance.events.append(MidiEvent(240, 4, EventKind.PITCH_BEND, 0, (4096,)))
    performance.events.append(MidiEvent(300, 5, EventKind.CONTROL_CHANGE, 0, (74, 101)))
    cold = write_sample_midi(performance, tmp_path / "cold.mid", 44100)
    warm = write_sample_midi(performance, tmp_path / "warm.mid", 44100, preroll_frames=8192)
    assert cold["file_sha256"] != warm["file_sha256"]
    for first, second in zip(cold["events"], warm["events"], strict=True):
        assert second == {**first, "sample": first["sample"] + 8192}
    assert warm["note_off_velocity"] == "preserved" and warm["all_events"]


def test_plugin_scalar_readback_detects_rejection_and_accepts_float32_rounding():
    expected = {"settings": {"osc_1_level": 0.3, "osc_1_transpose": -12., "wavetables": []}}
    assert verify_loaded_scalars(expected, {"settings": {"osc_1_level": 0.3000000119, "osc_1_transpose": -12}}) == 2
    for actual in ({}, {"osc_1_level": 0.3, "osc_1_transpose": 0}, {"osc_1_level": True, "osc_1_transpose": -12}):
        with pytest.raises(RuntimeError, match="did not load"):
            verify_loaded_scalars(expected, {"settings": actual})


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

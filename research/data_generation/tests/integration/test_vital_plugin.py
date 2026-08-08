from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from obruxo_data.cli import DEFAULT_RENDERER, main
from obruxo_data.midi import Performance, TempoMap
from obruxo_data.render import RenderRequest, VitalRenderer, run_batch
from obruxo_data.render.qa import compare_audio
from obruxo_data.vital import VitalPreset


pytestmark = [pytest.mark.integration, pytest.mark.vita, pytest.mark.reference_plugin]


def _plugin_path() -> Path:
    configured = os.environ.get("OBRUXO_VITAL_PLUGIN")
    path = Path(configured) if configured else Path(r"C:\Program Files\Common Files\VST3\Vital.vst3")
    if not path.is_file():
        pytest.skip("user-supplied Vital VST3 is unavailable")
    return path


def test_two_note_sequence_renders_continuously_with_qa_and_repeat_tolerance(tmp_path) -> None:
    pytest.importorskip("vita")
    pytest.importorskip("dawdreamer")
    preset = VitalPreset.init()
    preset.set_raw("osc_1_random_phase", 0.0)
    performance = Performance(ticks_per_beat=480, bpm=120)
    performance.add_note(pitch=60, velocity=100, start_tick=0, duration_ticks=360)
    performance.add_note(pitch=64, velocity=90, start_tick=480, duration_ticks=360)
    performance.end_tick = 960
    request = RenderRequest(preset=preset, performance=performance, tail_seconds=0.25)
    renderer = VitalRenderer(_plugin_path())

    first = renderer.render(request)
    intervening = VitalPreset.init()
    intervening.set_raw("osc_1_level", 0.1)
    renderer.render(RenderRequest(preset=intervening, performance=performance, tail_seconds=0.25))
    second = renderer.render(request)
    expected = TempoMap.from_performance(performance).render_frame_count(960, 0.25, 44_100)
    assert first.audio.shape == (expected, 2)
    assert first.audio.dtype == np.float32
    assert np.isfinite(first.audio).all()
    assert first.qa["rms"] > first.qa["silence_threshold"]
    assert not [item for item in first.diagnostics if item.severity.value == "error"]
    assert "vital.runtime.canonicalization" in {item.code for item in first.diagnostics}
    assert compare_audio(first.audio, second.audio)["within_tolerance"]

    wav_path = tmp_path / "sequence.wav"
    result_path = tmp_path / "sequence.json"
    first.write_wav(wav_path)
    first.write_json(result_path)
    sample_rate, audio = wavfile.read(wav_path)
    assert sample_rate == 44_100
    assert audio.shape == first.audio.shape


def test_vital_batch_rejects_unsafe_in_process_concurrency(tmp_path) -> None:
    pytest.importorskip("vita")
    pytest.importorskip("dawdreamer")
    requests = []
    for pitch in (60, 64):
        performance = Performance(ticks_per_beat=480, bpm=120)
        performance.add_note(pitch=pitch, velocity=100, start_tick=0, duration_ticks=240)
        performance.end_tick = 480
        requests.append(RenderRequest(preset=VitalPreset.init(), performance=performance, tail_seconds=0.1))
    renderer = VitalRenderer(_plugin_path())
    with pytest.raises(ValueError, match="at most 1 worker"):
        run_batch(renderer, requests, tmp_path / "batch", workers=2)
    summary = run_batch(renderer, requests, tmp_path / "batch", workers=1)
    assert (summary.rendered, summary.skipped) == (2, 0)
    assert summary.request_ids == tuple(request.request_id for request in requests)
    assert all((tmp_path / "batch" / f"{request.request_id}.wav").is_file() for request in requests)


def test_offline_cli_smoke(tmp_path) -> None:
    pytest.importorskip("vita")
    pytest.importorskip("dawdreamer")
    preset = tmp_path / "simple.vital"
    midi = tmp_path / "simple.mid"
    wav = tmp_path / "simple.wav"
    result = tmp_path / "simple.json"
    assert main(["vital", "init", "--output", str(preset)]) == 0
    assert main([
        "midi", "create-note", "--pitch", "60", "--velocity", "100", "--beats", "1",
        "--end-beats", "2", "--output", str(midi),
    ]) == 0
    assert main([
        "render", str(preset), str(midi), "--tail", "0.1", "--renderer-config", str(DEFAULT_RENDERER),
        "--plugin-path", str(_plugin_path()), "--output", str(wav), "--result", str(result),
    ]) == 0
    assert wav.is_file() and result.is_file()

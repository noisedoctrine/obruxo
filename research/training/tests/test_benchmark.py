import json

import pytest

from obruxo_training.benchmark import benchmark_stream
from obruxo_training.codec_check import check_codec_audio
from obruxo_training.pilot import PilotPool
from test_stream import FakeRendererFactory


@pytest.fixture(scope="module")
def pool():
    return PilotPool.create(8, seed=0)


def test_benchmark_records_consumption_and_resume_without_audio_files(pool, tmp_path):
    output = tmp_path / "run"
    result = benchmark_stream(pool, output, count=2, renderer_factory=FakeRendererFactory())
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    assert [item["event"] for item in events] == ["render_outcome", "consumed"] * 2
    assert result["accepted"] == result["accepted_acknowledged"] == 2
    assert result["audio_files_written"] == result["optimizer_steps"] == 0
    assert not list(output.rglob("*.wav"))
    cursor = json.loads((output / "cursor.json").read_text())
    resumed = benchmark_stream(pool, tmp_path / "resumed", count=1, renderer_factory=FakeRendererFactory(), resume=cursor)
    assert resumed["resume_cursor"]["next_sequence_id"] == 3
    with pytest.raises(FileExistsError):
        benchmark_stream(pool, output, count=1, renderer_factory=FakeRendererFactory())


def test_benchmark_keeps_failed_examples_in_the_denominator(pool, tmp_path):
    result = benchmark_stream(pool, tmp_path / "run", count=1, renderer_factory=FakeRendererFactory("silence"))
    assert result["requested"] == result["produced"] == result["rejected"] == 1
    assert result["accepted_acknowledged"] == result["accepted"] == 0
    assert not json.loads((tmp_path / "run/status.json").read_text())["all_accepted"]


def test_consumer_failure_keeps_last_acknowledged_cursor(pool, tmp_path, monkeypatch):
    def fail(_):
        raise RuntimeError("consumer failed")
    monkeypatch.setattr("obruxo_training.benchmark.spectral_features", fail)
    with pytest.raises(RuntimeError, match="consumer failed"):
        benchmark_stream(pool, tmp_path / "run", count=3, renderer_factory=FakeRendererFactory())
    assert not json.loads((tmp_path / "run/status.json").read_text())["complete"]
    assert json.loads((tmp_path / "run/cursor.json").read_text())["next_sequence_id"] == 0


def test_native_codec_check_compares_repeats_and_reconstruction_separately(pool, tmp_path):
    result = check_codec_audio(pool, tmp_path / "check", renderer_factory=FakeRendererFactory())
    assert result["requested"] == 3 * result["covered_category_combinations"]
    assert result["all_controls_equal_at_float32"]
    assert result["source_repeats_within_reference_tolerance"] == result["covered_category_combinations"]
    assert result["decoded_within_reference_tolerance"] == result["covered_category_combinations"]
    assert result["accepted"] == result["requested"]

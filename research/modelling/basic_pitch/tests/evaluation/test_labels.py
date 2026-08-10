from __future__ import annotations

import os
from pathlib import Path

from obruxo_basic_pitch.evaluation.labels import add_source_metadata, performance_labels

ROOT = Path(__file__).resolve().parents[2]


def _write_performance(path: Path, *, tempo_change: bool = False, adjacent: bool = False) -> None:
    from obruxo_basic_pitch.evaluation.labels import _ensure_data_generation_importable

    _ensure_data_generation_importable()
    from obruxo_data.midi import Performance

    performance = Performance(ticks_per_beat=480, bpm=None)
    performance.add_tempo(120, tick=0)
    if tempo_change:
        performance.add_tempo(60, tick=480)
    performance.add_note(pitch=60, velocity=80, start_tick=0, duration_ticks=480)
    performance.add_note(
        pitch=62,
        velocity=90,
        start_tick=480 if adjacent else 0,
        duration_ticks=480,
    )
    performance.save_midi(path)


def test_tempo_map_and_half_open_polyphony() -> None:
    root = ROOT / "outputs" / f".test-evaluation-labels-{os.getpid()}"
    assert not root.exists()
    root.mkdir(parents=True)
    try:
        adjacent = root / "adjacent.mid"
        _write_performance(adjacent, adjacent=True)
        notes, labels = performance_labels(adjacent)
        assert [round(note.onset_s, 6) for note in notes] == [0.0, 0.5]
        assert [round(note.offset_s, 6) for note in notes] == [0.5, 1.0]
        assert labels["maximum_polyphony"] == 1
        assert labels["polyphony_class"] == "monophonic"
        assert labels["simultaneous_note_onsets"] is False

        changed_tempo = root / "tempo-change.mid"
        _write_performance(changed_tempo, tempo_change=True, adjacent=True)
        changed_notes, changed_labels = performance_labels(changed_tempo)
        assert round(changed_notes[1].onset_s, 6) == 0.5
        assert round(changed_notes[1].offset_s, 6) == 1.5
        assert changed_labels["duration_s"] == 1.5
    finally:
        for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        root.rmdir()


def test_objective_descriptors_and_unknown_source_metadata() -> None:
    root = ROOT / "outputs" / f".test-evaluation-labels-{os.getpid()}"
    assert not root.exists()
    root.mkdir(parents=True)
    try:
        path = root / "poly.mid"
        _write_performance(path)
        _, labels = performance_labels(path)
        assert labels["maximum_polyphony"] == 2
        assert labels["polyphony_class"] == "polyphonic"
        assert labels["note_count"] == 2
        assert labels["minimum_pitch_midi"] == 60
        assert labels["maximum_pitch_midi"] == 62
        assert labels["pitch_span_semitones"] == 2
        assert labels["median_note_duration_s"] == 0.5
        assert labels["notes_per_second"] == 4.0
        enriched = add_source_metadata(labels, None)
        assert enriched["instrument"] == "unknown"
        assert enriched["genre"] == "unknown"
        assert enriched["label_sources"]["instrument"] == "unknown"
    finally:
        for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        root.rmdir()

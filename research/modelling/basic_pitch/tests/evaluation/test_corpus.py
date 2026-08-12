from __future__ import annotations

import inspect
import json
import os
from pathlib import Path

import numpy as np
import pytest
from obruxo_basic_pitch.evaluation.corpus import (
    build_evaluation_manifest,
    load_evaluation_manifest,
)
from obruxo_basic_pitch.evaluation.labels import _ensure_data_generation_importable
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[2]


def _remove_tree(path: Path) -> None:
    if not path.exists():
        return
    for child in sorted(
        path.rglob("*"), key=lambda item: len(item.parts), reverse=True
    ):
        if child.is_file() or child.is_symlink():
            child.unlink()
        elif child.is_dir():
            child.rmdir()
    path.rmdir()


def _write_midi(path: Path, pitch: int = 60) -> None:
    _ensure_data_generation_importable()
    from obruxo_data.midi import Performance

    performance = Performance(ticks_per_beat=480, bpm=120)
    performance.add_note(pitch=pitch, velocity=96, start_tick=0, duration_ticks=480)
    performance.save_midi(path)


def _write_audio(path: Path) -> None:
    wavfile.write(path, 22_050, np.zeros(22_050, dtype=np.int16))


def _make_pair(
    root: Path, name: str, *, modern: bool = False, pitch: int = 60
) -> tuple[Path, Path]:
    directory = root / name
    directory.mkdir()
    midi = directory / "performance.mid"
    audio = directory / "render.wav"
    _write_midi(midi, pitch=pitch)
    _write_audio(audio)
    if modern:
        audio.with_suffix(".json").write_text(
            json.dumps({"request_id": "local-only", "qa": {"silence_warning": True}}),
            encoding="utf-8",
        )
    return midi, audio


def test_actual_direct_directory_pairing_is_deterministic_and_read_only() -> None:
    assert (
        inspect.signature(build_evaluation_manifest)
        .parameters["allow_derived_render"]
        .default
        is False
    )
    output_root = ROOT / "outputs" / f".test-evaluation-corpus-{os.getpid()}"
    assert not output_root.exists()
    corpus = output_root / "sources" / "presetshare_files" / "data"
    corpus.mkdir(parents=True)
    _, legacy_audio = _make_pair(corpus, "pair-z", pitch=60)
    modern_midi, _ = _make_pair(corpus, "pair-a", modern=True, pitch=62)
    duplicate_midi, duplicate_audio = _make_pair(corpus, "pair-b", pitch=60)
    duplicate_midi.write_bytes((corpus / "pair-z" / "performance.mid").read_bytes())
    duplicate_audio.write_bytes(legacy_audio.read_bytes())

    ambiguous = corpus / "ambiguous"
    ambiguous.mkdir()
    _write_midi(ambiguous / "one.mid")
    _write_midi(ambiguous / "two.mid", pitch=64)
    _write_audio(ambiguous / "render.wav")
    missing_audio = corpus / "missing-audio"
    missing_audio.mkdir()
    _write_midi(missing_audio / "performance.mid")
    missing_midi = corpus / "missing-midi"
    missing_midi.mkdir()
    _write_audio(missing_midi / "render.wav")

    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in corpus.rglob("*")
        if path.is_file()
    }
    output = output_root / "artifacts" / "manifest.jsonl"
    audit = output_root / "artifacts" / "pairing_audit.json"
    try:
        summary = build_evaluation_manifest(corpus, output=output, audit=audit)
        manifest_text = output.read_text(encoding="utf-8")
        assert summary["candidate_count"] == 6
        assert summary["eligible_count"] == 2
        assert summary["ambiguous_count"] == 1
        assert summary["excluded_by_reason"] == {
            "pair.ambiguous": 1,
            "pair.duplicate_identity": 1,
            "pair.missing_audio": 1,
            "pair.missing_midi": 1,
        }
        pairs = load_evaluation_manifest(output)
        assert len(pairs) == 2
        modern = next(pair for pair in pairs if pair.midi_path == modern_midi.resolve())
        assert modern.audio_source == "existing_audio"
        assert modern.provenance_status == "available"
        assert modern.qa_warning_codes == ("qa.silence_warning",)
        audit_value = json.loads(audit.read_text(encoding="utf-8"))
        assert audit_value["source_snapshot"]["source_stat_mismatches"] == 0
        assert audit_value["candidates"]
        assert "pair-" in manifest_text
        after = {
            path: (path.read_bytes(), path.stat().st_mtime_ns)
            for path in corpus.rglob("*")
            if path.is_file()
        }
        assert before == after
        with pytest.raises(FileExistsError):
            build_evaluation_manifest(corpus, output=output, audit=audit)
        build_evaluation_manifest(corpus, output=output, audit=audit, force=True)
        assert output.read_text(encoding="utf-8") == manifest_text
        assert not list(output_root.glob("*.tmp"))
    finally:
        _remove_tree(output_root)

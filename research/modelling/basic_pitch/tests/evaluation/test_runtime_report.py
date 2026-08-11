from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from obruxo_basic_pitch.evaluation.corpus import build_evaluation_manifest
from obruxo_basic_pitch.evaluation.labels import _ensure_data_generation_importable
from obruxo_basic_pitch.evaluation.report import write_sanitized_reports
from obruxo_basic_pitch.evaluation.runner import (
    BackendUnavailable,
    evaluate_corpus,
    validate_backend_id,
)
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[2]


def _remove_tree(path: Path) -> None:
    if not path.exists():
        return
    for child in sorted(path.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if child.is_file() or child.is_symlink():
            child.unlink()
        elif child.is_dir():
            child.rmdir()
    path.rmdir()


def _write_pair(root: Path) -> None:
    _ensure_data_generation_importable()
    from obruxo_data.midi import Performance

    directory = root / "pair"
    directory.mkdir()
    performance = Performance(ticks_per_beat=480, bpm=120)
    performance.add_note(pitch=60, velocity=80, start_tick=0, duration_ticks=480)
    performance.save_midi(directory / "performance.mid")
    wavfile.write(directory / "render.wav", 22_050, np.zeros(22_050, dtype=np.int16))


def _fake_result(_: object) -> dict[str, object]:
    counts = {"reference_count": 1, "prediction_count": 1, "tp": 1, "f1": 1.0}
    return {
        "status": "ok",
        "failure_code": None,
        "audio_seconds": 1.0,
        "predicted_note_count": 1,
        "metrics": {
            "notes": {
                "onset_pitch": counts,
                "onset_pitch_offset": counts,
                "out_of_range_reference_notes": 0,
                "timing_diagnostics": {},
                "velocity": {},
                "pitch_confusion": {},
            },
            "frames": counts,
        },
    }


def test_resume_identity_backend_guard_and_sanitized_report() -> None:
    output_root = ROOT / "outputs" / f".test-evaluation-runtime-{os.getpid()}"
    assert not output_root.exists()
    corpus = output_root / "sources" / "data"
    corpus.mkdir(parents=True)
    _write_pair(corpus)
    report_suffix = str(os.getpid())
    public_json = ROOT / "reports" / f".test-presetshare-{report_suffix}.json"
    public_markdown = ROOT / "reports" / f".test-presetshare-{report_suffix}.md"
    assert not public_json.exists() and not public_markdown.exists()
    output = output_root / "artifacts" / "manifest.jsonl"
    audit = output_root / "artifacts" / "pairing_audit.json"
    calls: list[str] = []

    def predictor(pair: object) -> dict[str, object]:
        calls.append("called")
        return _fake_result(pair)

    try:
        build_evaluation_manifest(corpus, output=output, audit=audit)
        first = evaluate_corpus(output, output_root, predictor=predictor)
        assert first["successful_pair_count"] == 1
        assert len(calls) == 1
        calls.clear()
        resumed = evaluate_corpus(output, output_root, predictor=predictor)
        assert resumed["successful_pair_count"] == 1
        assert calls == []

        changed_manifest = output_root / "manifest-copy.jsonl"
        changed_manifest.write_text(output.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        evaluate_corpus(changed_manifest, output_root, predictor=predictor)
        assert len(calls) == 1

        with pytest.raises(BackendUnavailable):
            validate_backend_id("openvino_gpu")
        report = write_sanitized_reports(
            audit,
            output_root / "run.json",
            output_root / "aggregates.json",
            public_json,
            public_markdown,
        )
        text = public_json.read_text(encoding="utf-8") + public_markdown.read_text(encoding="utf-8")
        assert report["aggregate"]["pair_count"] == 1
        assert report["runtime_provenance"]["selected_backend"] == "pytorch_cpu"
        assert report["runtime_provenance"]["selection_rationale"]
        assert "Runtime provenance and #24 route decision" in public_markdown.read_text(encoding="utf-8")
        assert str(corpus) not in text
        assert "performance.mid" not in text
        assert "render.wav" not in text
        assert "pair-" not in text
        assert "preset_id" not in text
        assert "pair_id" not in text
    finally:
        if public_json.exists():
            public_json.unlink()
        if public_markdown.exists():
            public_markdown.unlink()
        _remove_tree(output_root)

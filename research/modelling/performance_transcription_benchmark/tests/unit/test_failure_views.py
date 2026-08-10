from __future__ import annotations

from obruxo_basic_pitch.evaluation.labels import ReferenceNote
from obruxo_basic_pitch.evaluation.metrics import note_metrics
from obruxo_basic_pitch.postprocess import NoteEvent
from obruxo_performance_benchmark.evaluate import build_quality_views


def _labels() -> dict[str, str]:
    return {"polyphony_class": "monophonic", "duration_class": "short", "note_density_class": "low", "pitch_register_class": "mid", "instrument": "unknown", "genre": "unknown", "type": "unknown", "vital_style": "unknown"}


def test_failure_penalty_routes_failed_note_through_landed_metrics() -> None:
    reference = [ReferenceNote(0.0, 1.0, 60, 100)]
    good = NoteEvent(0.0, 1.0, 60, 1.0, None)
    rows = [
        {"pair_id": "synthetic-good", "preset_id": None, "labels": _labels(), "status": "ok", "metrics": note_metrics(reference, [good])},
        {"pair_id": "synthetic-failed", "preset_id": None, "labels": _labels(), "status": "runtime_failed", "failure_code": "transcription_runtime_error", "frame_count": 0},
    ]
    views = build_quality_views(rows, {"synthetic-good": reference, "synthetic-failed": reference}, frame_model=False, bootstrap_replicates=10)
    assert views["success_only"]["eligible_pairs"] == 2
    assert views["success_only"]["successful_pairs"] == 1
    assert views["success_only"]["coverage"] == 0.5
    assert views["failure_penalized"]["aggregate"]["micro"]["onset_pitch"]["fn"] == 1
    assert views["failure_penalized"]["aggregate"]["failure_analysis"]["timing"]["onset"]["support"] == 0

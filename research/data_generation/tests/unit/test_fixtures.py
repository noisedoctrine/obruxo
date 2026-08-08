from __future__ import annotations

import json
from pathlib import Path

from obruxo_data.midi import Performance
from obruxo_data.render import RendererCapabilities
from obruxo_data.vital import VitalPreset


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_authored_vital_fixtures_validate() -> None:
    names = (
        "canonical_init.vital", "oscillator_1.vital", "oscillator_1_filter_1.vital",
        "lfo_1_to_oscillator_1.vital", "reverb_enabled.vital",
    )
    for name in names:
        report = VitalPreset.load(FIXTURES / "vital" / name).validate()
        assert report.valid, (name, report.to_dict())


def test_performance_fixtures_preserve_boundaries_and_capabilities() -> None:
    held = Performance.from_midi(FIXTURES / "midi" / "held_note_with_tail.mid")
    boundary = Performance.from_midi(FIXTURES / "midi" / "sequential_boundary.mid")
    chord = Performance.from_midi(FIXTURES / "midi" / "overlapping_chord.mid")
    tempo_change = Performance.from_midi(FIXTURES / "midi" / "tempo_change_unsupported.mid")
    assert held.end_tick == 960 and held.note_spans()[0].end_tick == 480
    assert boundary.validate().valid and chord.validate().valid and tempo_change.validate().valid
    assert not tempo_change.validate(RendererCapabilities(tempo_changes=False)).valid


def test_reference_result_matrix_is_complete_and_within_tolerance() -> None:
    results = json.loads((FIXTURES / "reference_results.json").read_text(encoding="utf-8"))
    assert results["artifact_schema"] == "obruxo_vital_reference_results_v1"
    assert results["schema_id"] == VitalPreset.init().schema.schema_id
    assert results["repeatability_failures"] == []
    assert len(results["cases"]) == 15
    assert len({(case["preset"], case["performance"]) for case in results["cases"]}) == 15
    assert all(case["repeatability"]["within_tolerance"] for case in results["cases"])
    assert all(len(case["request_id"]) == 64 and len(case["audio_float32_sha256"]) == 64 for case in results["cases"])

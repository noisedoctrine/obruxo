import importlib.util
from pathlib import Path

path = Path(__file__).parents[1] / "reports/scripts/diagnose_input_notes.py"
spec = importlib.util.spec_from_file_location("input_notes", path)
notes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(notes)


def test_pitch_coverage_excludes_note_off_and_zero_velocity_events():
    events = [{"kind": kind, "data": [pitch, velocity]} for kind, pitch, velocity in
              (("note_on", 60, 90), ("note_on", 36, 0), ("note_off", 84, 90), ("note_on", 67, 80), ("note_on", 60, 100))]
    assert notes.pitches({"events": events}) == [60, 67]


def test_pattern_table_detects_filter_labels_that_can_be_guessed_without_audio():
    recipes = [{"target": {"categorical": [i % 3, i % 2]}} for i in range(8)]
    table = notes.pattern_table(recipes, [i % 4 for i in range(8)])
    assert table["pattern_parity_only_filter_accuracy"] == 1
    assert table["filter_off_then_on_by_held_staccato_melody_chord"] == [[2, 0, 2, 0], [0, 2, 0, 2]]
    balanced = [{"target": {"categorical": [0, value]}} for value in (0, 1) for _ in range(4)]
    assert notes.pattern_table(balanced, [i % 4 for i in range(8)])["pattern_parity_only_filter_accuracy"] == .5

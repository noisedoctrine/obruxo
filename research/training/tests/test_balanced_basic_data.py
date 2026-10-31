import copy

import pytest
from obruxo_data.midi import Performance, TempoMap
from obruxo_training.balanced_basic_data import (
    SPEC,
    grid_rows,
    performance_ids,
    performances,
    validate_rows,
)


def patches():
    return [{"training_index": index, "state_id": str(index), "target": {"continuous": [.2 + position * .1] * 9, "categorical": [position, 1]}}
            for position, index in enumerate(SPEC["training_indices"])]


def test_all_three_patches_receive_identical_note_and_duration_coverage():
    rows = grid_rows(patches())
    validate_rows(rows)
    identities = [{row["performance_id"] for row in rows if row["state_id"] == str(index)} for index in SPEC["training_indices"]]
    assert len(rows) == 18 and len(identities[0]) == 6 and identities[0] == identities[1] == identities[2]
    for row in rows:
        midi = Performance.from_dict(row["performance"])
        note, = midi.note_spans()
        timing = TempoMap.from_performance(midi)
        assert note.pitch == row["pitch"] and note.velocity == 96 and note.start_tick == 0
        assert float(timing.tick_to_seconds(note.end_tick)) == pytest.approx(.4 if row["duration_ticks"] == 384 else 1.2)
        assert timing.render_frame_count(midi.end_tick, .25, 44100) == 187425


def test_reserved_performances_are_excluded_before_rendering():
    first = performances()[0]
    nested = {"performances": [{"nested": {"performance": first["performance"], "performance_id": first["performance_id"]}}]}
    forbidden = performance_ids(nested)
    assert forbidden == {first["performance_id"]}
    with pytest.raises(ValueError, match="reserved"):
        grid_rows(patches(), forbidden)


@pytest.mark.parametrize("change", ["missing", "duplicated", "mislabelled", "changed_controls"])
def test_partial_or_incorrectly_labelled_source_banks_cannot_qualify(change):
    rows = copy.deepcopy(grid_rows(patches()))
    if change == "missing":
        rows.pop()
    elif change == "duplicated":
        rows[0] = rows[1]
    elif change == "mislabelled":
        rows[0]["pitch"] = 37
    else:
        rows[0]["target"] = copy.deepcopy(rows[0]["target"])
        rows[0]["target"]["continuous"][0] = .9
    with pytest.raises(ValueError):
        validate_rows(rows)

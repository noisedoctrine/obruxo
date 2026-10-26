import copy

import pytest

from obruxo_training.basic_input_diagnosis import describe, select_examples
from obruxo_training.codec import PilotCodec


def evidence():
    cells, rows = [], []
    target = {"continuous": [0.5] * 9, "categorical": [0, 0]}
    for family in range(4):
        for view, playback in ((0, 0), (0, 1), (0, 2), (1, 1), (1, 0), (2, 2), (2, 0)):
            index = len(cells)
            cells.append({"input_view": view, "playback_view": playback, "training_index": family,
                          "recipe": {"state_id": f"{family:064x}", "target": target}})
            for arm in ("parameter", "spectral"):
                prediction = {"continuous": [0.6 if arm == "spectral" else 0.5] * 9, "categorical": [0, 0]}
                rows.append({"index": index, "arm": arm, "candidate": prediction,
                             "scores": {"spectral": 1. + (family * .1 if arm == "spectral" else 0)}})
    return cells, rows


def test_control_diagnosis_counts_each_input_patch_once_and_uses_source_activity():
    cells, rows = evidence()
    result = describe(cells, rows, PilotCodec())
    assert len(result) == 6 and all(row["independent_patches"] == 4 for row in result.values())
    for key, row in result.items():
        assert row["correct_unison"] == row["correct_filter"] == 4
        errors = row["source_active_normalized_control_mae"]
        assert all(errors[name] is None for name in ("osc_1_unison_detune", "filter_1_cutoff", "filter_1_resonance"))
        assert errors["osc_1_level"] == pytest.approx(.1 if key.endswith("spectral") else 0.)


def test_different_playback_does_not_authorize_a_different_inferred_patch():
    cells, rows = evidence()
    rows[2] = copy.deepcopy(rows[2])
    rows[2]["candidate"]["continuous"][0] = .9
    with pytest.raises(ValueError, match="different patches across playback"):
        describe(cells, rows, PilotCodec())


def test_listening_selection_covers_every_group_and_keeps_first_playback():
    cells, rows = evidence()
    selected = select_examples(cells, rows)
    assert len(selected) == 12 and len({row["group"] for row in selected}) == 6
    assert len({(row["group"], row["family_id"]) for row in selected}) == 12
    for row in selected:
        assert row["family_id"] == f"{(1 if row['selection'] == 'median_gap' else 3):064x}"
        if row["group"] == "original:transfer":
            assert cells[row["index"]]["playback_view"] == 1


def test_listening_rejects_duplicate_finals_and_does_not_drop_them():
    cells, rows = evidence()
    with pytest.raises(ValueError, match="duplicate"):
        select_examples(cells, [*rows, rows[0]])

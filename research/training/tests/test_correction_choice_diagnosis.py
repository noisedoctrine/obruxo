import pytest

from obruxo_training.conditioned_bridge import read
from obruxo_training.correction_choice_diagnosis import choice_row, source_contract, summarize


def test_single_choice_costs_keep_anchor_on_ties_and_report_real_improvement():
    estimate = {"categorical": [[1., 1., 2.], [.9, 1.]], "anchor": {"categorical": [1, 1]}, "value": 1.}
    assert choice_row(estimate, 0)["best_single_edit"] == 1
    row = choice_row(estimate, 1)
    assert row["lower_cost_edit_available"] and row["relative_gain"] == pytest.approx(.1)


@pytest.mark.parametrize("costs", [[.5, .2], [.5, float("nan"), 1.], [.5, -1., 1.], [.5, .9, 1.]])
def test_changed_anchor_and_invalid_native_costs_are_rejected(costs):
    with pytest.raises(ValueError, match="measured anchor"):
        choice_row({"categorical": [costs], "anchor": {"categorical": [1]}, "value": .2}, 0)


def test_summary_keeps_training_views_and_repeated_measurements_separate():
    prediction = {"model": "spectral", "view": 1, "active_control_mae": .3, "control_change": .01,
                  "category_correct": [0, 1], "category_changed": [0, 0]}
    choices = [{"head": 0, "input_view": view, "lower_cost_edit_available": gain > 0, "relative_gain": gain}
               for view, gain in ((0, 0.), (1, .2), (1, 0.))]
    result = summarize([prediction], choices)
    assert result["prediction_groups"][0]["category_changes_from_initial"] == [0, 0]
    original, wider = result["measured_choice_groups"]
    assert original["lower_cost_edits"] == 0
    assert wider["repeated_observations"] == 2 and wider["lower_cost_edits"] == 1
    assert wider["mean_relative_gain_all_observations"] == pytest.approx(.1)
    assert wider["mean_relative_gain_when_edit_available"] == pytest.approx(.2)


def test_unverified_parent_rejected_before_loading_any_tensors(tmp_path):
    from obruxo_training.audit import write_json
    proof = {"verified": False}
    write_json(tmp_path / "verification.json", proof)
    assert not read(tmp_path / "verification.json")["verified"]
    with pytest.raises(ValueError, match="verified failed"):
        source_contract(tmp_path, tmp_path / "missing.pt")

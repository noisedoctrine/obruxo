import copy

import pytest

from obruxo_training.codec import PilotCodec
from obruxo_training.update_diagnosis import inspect, make_plan, movement
from test_basic_direct_fit import completed  # noqa: F401


def test_movement_separates_active_controls_conditional_choices_and_finite_change():
    before = {"continuous": [.4] * 9, "categorical": [0, 0]}
    after = {"continuous": [.41, .4, .7, .4, .4, .4, .4, .4, .4], "categorical": [2, 1]}
    estimate = {"anchor": before, "continuous": [2., 0., 0., 0., 0., 0., 0., 0., 0.], "categorical": [[1., .8, .6], [1., .9]]}
    result = movement(before, after, estimate, PilotCodec())
    assert result["maximum_control_change"] == pytest.approx(.3)
    assert result["maximum_previously_active_change"] == pytest.approx(.01)
    assert result["above_measurement_spacing"] and result["category_heads_changed"] == 2
    assert result["linear_continuous_change"] == pytest.approx(.02)
    assert result["conditional_category_change"] == pytest.approx(-.5)
    assert result["estimated_sound_change"] == pytest.approx(-.48)
    assert "final_step_measured_change" not in result


def test_stale_native_anchor_and_duplicate_trajectories_are_refused(tmp_path):
    before = {"continuous": [.4] * 9, "categorical": [0, 0]}
    estimate = {"anchor": copy.deepcopy(before)}
    estimate["anchor"]["continuous"][0] = .5
    with pytest.raises(ValueError, match="stale"):
        movement(before, before, estimate, PilotCodec())
    with pytest.raises(ValueError, match="distinct"):
        make_plan([tmp_path, tmp_path], tmp_path, tmp_path, tmp_path)


def test_full_saved_fit_stays_exact_and_only_final_step_has_measured_after_audio(completed):  # noqa: F811
    inputs, source = completed
    result = inspect(inputs, source)
    assert result["model_adam_rng_replay_exact"] and result["replayed_updates_per_arm"] == 4
    assert len(result["updates"]) == 4 and len(result["observations"]) == 16
    measured = result["final_step_measured_observations"]
    assert len(measured) == 4 and all(row["step"] == 4 for row in measured)
    assert all(row["final_step_measured_change"] == row["final_step_measured_after"] - row["final_step_measured_before"] for row in measured)
    assert all("final_step_measured_change" not in row for row in result["observations"][:12])
    assert result["scalar_probes_reused"] > 0
    assert all(row["bank_inputs"] == 4 for row in result["updates"])

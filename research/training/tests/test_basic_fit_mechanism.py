import copy

import pytest
import torch

from obruxo_training.basic_fit_mechanism import exposures, joint_probes, pooled_pitch


def test_pooled_note_layout_retains_low_high_and_silent_cases():
    for midi in (21, 60, 108):
        pitch = torch.zeros(3521)
        pitch[(midi - 21) * 8:(midi - 20) * 8] = .7
        offset = 1408 + ((midi - 21) * 3 + 1) * 8
        pitch[offset:offset + 8] = .6
        result = pooled_pitch(pitch)
        assert result["note_top_midi"] == midi
        assert result["contour_peak_midi"] == pytest.approx(midi + 1 / 3)
    assert pooled_pitch(torch.zeros(3521))["note_top_midi"] is None
    with pytest.raises(ValueError, match="layout"):
        pooled_pitch(torch.zeros(440))


def evidence():
    cells = [{"training_index": 0, "input_view": view, "playback_view": playback} for view in (1, 2) for playback in (0, view)]
    finals = [{"index": index, "arm": arm, "scores": {"spectral": .55 if arm == "parameter" else .8}}
              for index in range(4) for arm in ("parameter", "spectral")]
    probes = []
    for index, cell in enumerate(cells):
        probes.append({"query": {"index": index, "candidate": {"x": .5}}, "loss": .4 if cell["playback_view"] == 0 else .6})
        if cell["playback_view"] == 0:
            probes.append({"query": {"index": index, "candidate": {"x": .9}}, "loss": .01})
    return {"fit_audit_cells": cells}, probes, finals


def test_best_patch_requires_the_same_candidate_measured_under_both_performances():
    plan, probes, finals = evidence()
    result = joint_probes(plan, probes, finals)
    for view in ("1", "2"):
        row = result[view]["rows"][0]
        assert row["best_probed_joint_loss"] == .5
        assert row["complete_measured_candidates"] == row["incomplete_candidates_not_used"] == 1
        assert row["probe_beats_parameter"] and row["final_worse_than_probe"]


def test_conflicting_repeat_costs_are_rejected_and_final_improvements_are_retained():
    plan, probes, finals = evidence()
    changed = copy.deepcopy(probes[0])
    changed["loss"] += .01
    with pytest.raises(ValueError, match="inconsistent"):
        joint_probes(plan, [*probes, changed], finals)
    for row in finals:
        if row["arm"] == "spectral":
            row["scores"]["spectral"] = .3
    result = joint_probes(plan, probes, finals)
    assert result["1"]["best_probed_or_final_sound_mean"] == .3


def test_primary_observation_exposures_are_counted_per_patch_and_input():
    plan = {"updates": [{"examples": [{"state": 0, "input_view": 0}, {"state": 0, "input_view": view}]} for view in (1, 2, 1, 2)]}
    counts = {(row["training_index"], row["input_view"]): row["scheduled_primary_observations"] for row in exposures(plan)}
    assert counts == {(0, 0): 4, (0, 1): 2, (0, 2): 2}

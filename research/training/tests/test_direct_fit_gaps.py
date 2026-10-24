import copy

import pytest
import torch

from obruxo_training.codec import PilotCodec, PilotTarget
from obruxo_training.direct_fit_gaps import final_gaps, inspect_gradients, output_alignment
from test_basic_direct_fit import completed  # noqa: F401


def test_saved_two_arm_fit_replays_exactly_with_extra_auxiliary_gradient_inspection(completed):  # noqa: F811
    inputs, source = completed
    saved = torch.load(source / "checkpoint.pt", map_location="cpu", weights_only=True)
    result = inspect_gradients(inputs, source, saved)
    assert result["model_adam_rng_replay_exact"] and result["offline_replayed_updates_per_arm"] == 4
    assert len(result["neural_steps"]) == 4 and len(result["paired_output_observations"]) == 8
    assert result["scalar_native_probes_reused"] > 0
    assert all(row["auxiliary_norm"] > 0 and -1.000001 <= row["gradient_cosine"] <= 1.000001 for row in result["neural_steps"])


def test_same_choice_continuous_gap_and_category_help_are_counted_separately():
    plan = {"training": [{"state_id": state, "target": {"categorical": [0, 0]}} for state in ("a", "b")],
            "fit_audit_cells": [{"recipe": {"state_id": state}, "input_view": 1, "playback_view": view} for state in ("a", "b") for view in (0, 1)]}
    finals = {}
    for i in range(4):
        for arm in ("parameter", "spectral"):
            same = i < 2
            value = (1. if arm == "parameter" else 2.) if same else (.8 if arm == "parameter" else .2)
            finals[i, arm] = {"candidate": {"continuous": [0.] * 9, "categorical": [int(not same and arm == "spectral"), 0]}, "scores": {"spectral": value}}
    result = final_gaps(plan, finals)
    assert result["sound_worse_patches"] == result["sound_better_patches"] == 1
    assert result["same_choice_gap_sum"] == 1 and result["different_choice_gap_sum"] == pytest.approx(-.6)
    assert result["net_gap_sum_over_patches"] == pytest.approx(.4)
    altered = copy.deepcopy(finals)
    altered[1, "spectral"]["candidate"]["continuous"][0] = .1
    with pytest.raises(ValueError, match="changed across playback"):
        final_gaps(plan, altered)


def test_output_alignment_uses_only_source_active_control_labels():
    target = PilotTarget((.4,) * 9, (0, 0))
    native = [1., 1., -100., -100., -100., 1., 1., 1., 1.]
    result = output_alignment(torch.full((1, 9), .6), target, native, PilotCodec())
    assert result["gradient_dot"] == pytest.approx(.4)

from itertools import product

import pytest
import torch

from obruxo_training.controlled_probes import FACTORS, factor_readouts, patch_prediction_stability, patch_readout, validate_grid


def grid():
    return [{"state_id": str(state), "categorical": [state % 3, state % 2], **dict(zip(FACTORS, values, strict=True))}
            for state in range(12) for values in product(*FACTORS.values())]


def test_controlled_readouts_recover_known_independent_factors_and_patch_information():
    rows = grid()
    # Nonordinal one-hot factors permit a linear classifier to represent all classes.
    feature = torch.cat([torch.nn.functional.one_hot(torch.tensor([values.index(row[name]) for row in rows]), len(values))
                         for name, values in FACTORS.items()], 1).float()
    assert all(value["accuracy"] == 1. for value in factor_readouts(feature, rows).values())
    identity = torch.eye(12).repeat_interleave(12, dim=0)
    assert patch_readout(identity, rows)["accuracy"] == 1.
    assert patch_readout(torch.zeros_like(identity), rows)["accuracy"] == pytest.approx(1 / 12)


def test_prediction_stability_counts_only_one_factor_changes():
    rows = grid()
    continuous = torch.zeros(len(rows), 9)
    continuous[:, 0] = torch.tensor([(row["pitch"] - 48) / 24 for row in rows])
    active = torch.ones(len(rows), 9, dtype=torch.bool)
    value = patch_prediction_stability(continuous, torch.zeros(len(rows), 2, dtype=torch.int64), rows, active)
    assert value["pitch"]["pairs"] == 144
    assert value["pitch"]["normalized_control_mean_absolute_change"][0] == pytest.approx(2 / 3)
    assert value["velocity"]["pairs"] == value["duration_ticks"]["pairs"] == 72
    assert value["velocity"]["normalized_control_maximum_absolute_change"] == [0.] * 9
    with pytest.raises(ValueError, match="complete unique"):
        validate_grid(rows[:-1])

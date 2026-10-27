"""Independent-factor readouts and patch stability on a complete performance grid."""

from itertools import combinations, product

import torch
from torch.nn import functional as F

from .representation_probes import NULL_SEEDS, classify, grouped_folds, ridge_predict

FACTORS = {"pitch": (48, 60, 72), "velocity": (64, 112), "duration_ticks": (384, 1152)}


def validate_grid(rows):
    states = sorted({row["state_id"] for row in rows})
    expected = set(product(*FACTORS.values()))
    for state in states:
        cells = [tuple(row[name] for name in FACTORS) for row in rows if row["state_id"] == state]
        if len(cells) != len(expected) or set(cells) != expected:
            raise ValueError("controlled probes require the complete unique factorial for every patch")
    if not states:
        raise ValueError("empty controlled probe grid")
    return states


@torch.no_grad()
def factor_readouts(features, rows):
    validate_grid(rows)
    fold_ids = grouped_folds(rows, 2)
    labels = [torch.tensor([values.index(row[name]) for row in rows]) for name, values in FACTORS.items()]
    target = torch.cat([F.one_hot(value, len(choices)) for value, choices in zip(labels, FACTORS.values(), strict=True)], 1).double()
    predicted, nulls = torch.empty_like(target), [torch.empty_like(target) for _ in NULL_SEEDS]
    for fold in range(2):
        train, test = fold_ids != fold, fold_ids == fold
        if not train.any() or not test.any():
            raise ValueError("controlled factor probe needs both whole-patch folds")
        columns = [target[train]]
        for seed in NULL_SEEDS:
            shuffled = target.clone()
            generator = torch.Generator().manual_seed(seed)
            for state in sorted({row["state_id"] for index, row in enumerate(rows) if train[index]}):
                indices = [index for index, row in enumerate(rows) if row["state_id"] == state]
                order = torch.randperm(len(indices), generator=generator).tolist()
                shuffled[indices] = target[[indices[index] for index in order]]
            columns.append(shuffled[train])
        value = ridge_predict(features[train], torch.cat(columns, 1), features[test])
        predicted[test] = value[:, :7]
        for index, _seed in enumerate(NULL_SEEDS):
            nulls[index][test] = value[:, 7 * (index + 1):7 * (index + 2)]
    result, offset = {}, 0
    for actual, (name, values) in zip(labels, FACTORS.items(), strict=True):
        selected = slice(offset, offset + len(values))
        result[name] = classify(actual, predicted[:, selected], [value[:, selected] for value in nulls], len(values))
        offset += len(values)
    return result


@torch.no_grad()
def patch_readout(features, rows):
    states = validate_grid(rows)
    actual = torch.tensor([states.index(row["state_id"]) for row in rows])
    train = torch.tensor([row["pitch"] == 60 for row in rows])
    target = F.one_hot(actual, len(states)).double()
    columns = [target[train]]
    for seed in NULL_SEEDS:
        permutation = torch.randperm(len(states), generator=torch.Generator().manual_seed(seed))
        columns.append(F.one_hot(permutation[actual[train]], len(states)).double())
    value = ridge_predict(features[train], torch.cat(columns, 1), features[~train])
    width = len(states)
    result = classify(actual[~train], value[:, :width], [value[:, width * (i + 1):width * (i + 2)] for i in range(len(NULL_SEEDS))], width)
    result.update(training_observations=int(train.sum()), training_pitch=60, test_pitches=[48, 72], split="same_patches_unseen_pitch_performances")
    return result


def patch_prediction_stability(continuous, categorical, rows, active):
    """Compare one factor at a time; source controls are identical within each patch."""
    states = validate_grid(rows)
    lookup = {(row["state_id"], *(row[name] for name in FACTORS)): index for index, row in enumerate(rows)}
    result = {}
    for name, values in FACTORS.items():
        pairs = []
        other = [key for key in FACTORS if key != name]
        for state in states:
            for settings in product(*(FACTORS[key] for key in other)):
                for lower, upper in combinations(values, 2):
                    a = dict(zip(other, settings, strict=True))
                    b = {**a, name: upper}
                    a[name] = lower
                    pairs.append((lookup[(state, *(a[key] for key in FACTORS))], lookup[(state, *(b[key] for key in FACTORS))]))
        left, right = (torch.tensor(value) for value in zip(*pairs, strict=True))
        if not torch.equal(active[left], active[right]):
            raise ValueError("same-patch activity changed with its performance")
        difference = (continuous[left] - continuous[right]).abs().double()
        result[name] = {"pairs": len(pairs), "normalized_control_mean_absolute_change": [float(difference[active[left, index], index].mean()) for index in range(9)],
                        "normalized_control_maximum_absolute_change": [float(difference[active[left, index], index].max()) for index in range(9)],
                        "category_flip_fraction": (categorical[left] != categorical[right]).double().mean(0).tolist()}
    return result

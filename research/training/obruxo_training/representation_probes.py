"""Fixed linear readouts and grouped diagnostics for frozen audio representations."""

from collections import defaultdict
import hashlib
from statistics import mean

import torch
from torch.nn import functional as F

RIDGE = .1
NULL_SEEDS = (0, 1, 2, 3, 4)


def grouped_folds(rows, folds=3):
    if type(folds) is not int or folds < 2:
        raise ValueError("probe needs at least two patch folds")
    strata, assignment = defaultdict(set), {}
    for row in rows:
        strata[tuple(row["categorical"])].add(row["state_id"])
    for category, states in sorted(strata.items()):
        ordered = sorted(states, key=lambda state: hashlib.sha256(("representation_probe_v1:" + state).encode()).hexdigest())
        for index, state in enumerate(ordered):
            if state in assignment:
                raise ValueError("one patch has inconsistent category labels")
            assignment[state] = index % folds
    return torch.tensor([assignment[row["state_id"]] for row in rows])


@torch.no_grad()
def ridge_predict(train, target, test, *, ridge=RIDGE):
    if (train.ndim != 2 or test.ndim != 2 or target.ndim != 2 or len(train) != len(target)
            or train.shape[1] != test.shape[1] or not len(train) or not train.shape[1] or ridge <= 0):
        raise ValueError("ridge readout needs aligned nonempty train/test matrices and positive regularization")
    train, target, test = train.double(), target.double(), test.double()
    if not all(torch.isfinite(value).all() for value in (train, target, test)):
        raise ValueError("ridge readout requires finite features and targets")
    center, scale = train.mean(0), train.std(0, unbiased=False)
    scale = torch.where(scale > 1e-8, scale, 1.)
    x, query = (train - center) / scale, (test - center) / scale
    y_center = target.mean(0)
    y = target - y_center
    width = x.shape[1]
    if width <= len(x):
        gram = x.T @ x / width + ridge * torch.eye(width, dtype=x.dtype)
        weights = torch.linalg.solve(gram, x.T @ y / width)
    else:
        gram = x @ x.T / width + ridge * torch.eye(len(x), dtype=x.dtype)
        weights = x.T @ torch.linalg.solve(gram, y) / width
    return query @ weights + y_center


def permuted_targets(rows, train, view_labels, categorical, seed):
    """Shuffle performances within patches and patch labels between whole patches."""
    generator = torch.Generator().manual_seed(seed)
    indices = train.nonzero().flatten().tolist()
    by_state = defaultdict(list)
    for index in indices:
        by_state[rows[index]["state_id"]].append(index)
    states = sorted(by_state)
    style = view_labels.clone()
    labels = categorical.clone()
    other_states = [states[index] for index in torch.randperm(len(states), generator=generator).tolist()]
    for state, other in zip(states, other_states, strict=True):
        current = sorted(by_state[state], key=lambda index: rows[index]["view"])
        order = torch.randperm(len(current), generator=generator).tolist()
        style[current] = view_labels[[current[index] for index in order]]
        labels[current] = categorical[by_state[other][0]]
    return style, labels


def regression_metrics(actual, predicted, baseline):
    if not len(actual) or actual.shape != predicted.shape or actual.shape != baseline.shape:
        raise ValueError("probe score cannot omit its complete active denominator")
    if not all(torch.isfinite(value).all() for value in (actual, predicted, baseline)):
        raise ValueError("nonfinite probe score")
    error, prior = (predicted - actual).double(), (baseline - actual).double()
    mse, prior_mse = float(error.square().mean()), float(prior.square().mean())
    return {"observations": len(actual), "mae": float(error.abs().mean()), "rmse": mse ** .5,
            "mean_baseline_rmse": prior_mse ** .5, "mse_vs_mean_baseline": mse / prior_mse if prior_mse else None}


def classify(actual, scores, null_scores, classes):
    predicted = scores.argmax(1)
    confusion = torch.bincount(actual * classes + predicted, minlength=classes * classes).reshape(classes, classes)
    nulls = [float((value.argmax(1) == actual).double().mean()) for value in null_scores]
    support = confusion.sum(1)
    if not (support > 0).all():
        raise ValueError("probe must score every declared class")
    return {"observations": len(actual), "accuracy": float((predicted == actual).double().mean()), "uniform_class_chance": 1 / classes,
            "balanced_accuracy": float((confusion.diag() / support).double().mean()), "largest_class_fraction": float(support.max() / support.sum()),
            "shuffled_training_accuracies": nulls, "shuffled_training_mean_accuracy": mean(nulls),
            "confusion_actual_rows_predicted_columns": confusion.tolist()}


@torch.no_grad()
def cached_readouts(features, rows, active, *, folds=3):
    if len(features) != len(rows) or active.shape != (len(rows), 9):
        raise ValueError("probe rows and active parameter masks differ")
    views = torch.tensor([row["view"] for row in rows])
    categories = torch.tensor([row["categorical"] for row in rows])
    controls = torch.tensor([row["continuous"] for row in rows], dtype=torch.float64)
    performance = torch.tensor([row["performance"] for row in rows], dtype=torch.float64)
    fold_ids = grouped_folds(rows, folds)
    # Fixed columns: view class, true MIDI summaries, unison/filter classes.
    target = torch.cat((F.one_hot(views, 3), performance, F.one_hot(categories[:, 0], 3), F.one_hot(categories[:, 1], 2)), 1).double()
    predicted = torch.empty_like(target)
    baselines = torch.empty_like(performance)
    null_prediction = [torch.empty((len(rows), 8), dtype=torch.float64) for _ in NULL_SEEDS]
    control_prediction = torch.zeros_like(controls)
    control_baseline = torch.zeros_like(controls)
    group_indices = ((0, 1, 5, 6, 7, 8), (2,), (3, 4))
    for fold in range(folds):
        train, test = fold_ids != fold, fold_ids == fold
        if not train.any() or not test.any():
            raise ValueError("empty grouped probe fold")
        columns = [target[train]]
        for seed in NULL_SEEDS:
            style, category = permuted_targets(rows, train, views, categories, seed)
            columns.append(torch.cat((F.one_hot(style[train], 3), F.one_hot(category[train, 0], 3), F.one_hot(category[train, 1], 2)), 1).double())
        values = ridge_predict(features[train], torch.cat(columns, 1), features[test])
        predicted[test] = values[:, :target.shape[1]]
        baselines[test] = performance[train].mean(0)
        for index in range(len(NULL_SEEDS)):
            null_prediction[index][test] = values[:, target.shape[1] + 8 * index:target.shape[1] + 8 * (index + 1)]
        for indices in group_indices:
            mask = active[:, indices[0]]
            if not all(torch.equal(mask, active[:, index]) for index in indices):
                raise ValueError("probe grouped controls have different activity masks")
            selected, queried = train & mask, test & mask
            if not selected.any() or not queried.any():
                raise ValueError("active control missing from a probe fold")
            value = ridge_predict(features[selected], controls[selected][:, indices], features[queried])
            for column, index in enumerate(indices):
                control_prediction[queried, index] = value[:, column]
                control_baseline[queried, index] = controls[selected, index].mean()
    return {"performance_view": classify(views, predicted[:, :3], [value[:, :3] for value in null_prediction], 3),
            "performance_summaries": {name: regression_metrics(performance[:, index], predicted[:, 3 + index], baselines[:, index])
                                      for index, name in enumerate(("mean_midi", "mean_velocity", "mean_note_seconds"))},
            "source_categories": {"unison": classify(categories[:, 0], predicted[:, 6:9], [value[:, 3:6] for value in null_prediction], 3),
                                  "filter": classify(categories[:, 1], predicted[:, 9:11], [value[:, 6:8] for value in null_prediction], 2)},
            "source_continuous": [regression_metrics(controls[active[:, index], index], control_prediction[active[:, index], index],
                                                       control_baseline[active[:, index], index]) for index in range(9)],
            "fold_patch_counts": [len({row["state_id"] for index, row in enumerate(rows) if fold_ids[index] == fold}) for fold in range(folds)]}


@torch.no_grad()
def cross_performance_retrieval(features, rows):
    lookup = {(row["state_id"], row["view"]): index for index, row in enumerate(rows)}
    states = sorted({row["state_id"] for row in rows})
    if len(lookup) != len(rows) or any((state, view) not in lookup for state in states for view in (0, 1, 2)):
        raise ValueError("retrieval requires complete unique three-view patch groups")
    results = {}
    for source, target in ((0, 1), (1, 0), (0, 2), (2, 0)):
        gallery = features[[lookup[state, source] for state in states]].double()
        queries = features[[lookup[state, target] for state in states]].double()
        center, scale = gallery.mean(0), gallery.std(0, unbiased=False)
        scale = torch.where(scale > 1e-8, scale, 1.)
        gallery, queries = (gallery - center) / scale, (queries - center) / scale
        similarity = F.normalize(queries, dim=1) @ F.normalize(gallery, dim=1).T
        predicted = similarity.argmax(1)
        results[f"{source}_to_{target}"] = {"queries": len(states), "candidates": len(states), "chance": 1 / len(states),
             "accuracy": float((predicted == torch.arange(len(states))).double().mean()),
             "zero_gallery_vectors": int((gallery.norm(dim=1) < 1e-12).sum()), "zero_query_vectors": int((queries.norm(dim=1) < 1e-12).sum())}
    return results

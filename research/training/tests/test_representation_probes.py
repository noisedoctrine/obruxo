import pytest
import torch

from obruxo_training.representation_probes import cross_performance_retrieval, grouped_folds, permuted_targets, regression_metrics, ridge_predict


def test_ridge_matches_explicit_training_only_solution():
    generator = torch.Generator().manual_seed(19)
    for count, width in ((12, 4), (4, 12)):
        train = torch.randn(count, width, generator=generator).double()
        train[:, 0] = 3.
        test = torch.randn(7, width, generator=generator).double() + 20
        target = torch.randn(count, 2, generator=generator).double()
        center, scale = train.mean(0), train.std(0, unbiased=False)
        scale = torch.where(scale > 1e-8, scale, 1.)
        x = (train - center) / scale
        weights = torch.linalg.solve(x.T @ x + .1 * width * torch.eye(width, dtype=x.dtype), x.T @ (target - target.mean(0)))
        expected = (test - center) / scale @ weights + target.mean(0)
        actual = ridge_predict(train, target, test)
        assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)
        assert torch.allclose(actual, torch.cat([ridge_predict(train, target, row[None]) for row in test]))


def test_folds_and_nulls_preserve_patch_groups():
    rows = [{"state_id": str(state), "view": view, "categorical": [state % 3, state % 2]} for state in range(36) for view in range(3)]
    fold = grouped_folds(rows)
    for state in range(36):
        assert len(set(fold[state * 3:state * 3 + 3].tolist())) == 1
    views = torch.tensor([row["view"] for row in rows])
    labels = torch.tensor([row["categorical"] for row in rows])
    train = fold != 0
    shuffled_view, shuffled_category = permuted_targets(rows, train, views, labels, 2)
    assert torch.equal(shuffled_view[~train], views[~train])
    assert torch.equal(shuffled_category[~train], labels[~train])
    assert not torch.equal(shuffled_view[train], views[train])
    for state in range(36):
        indices = slice(3 * state, 3 * state + 3)
        assert sorted(shuffled_view[indices].tolist()) == [0, 1, 2]
        assert shuffled_category[indices].unique(dim=0).shape[0] == 1


def test_retrieval_detects_identity_and_collapse_and_rejects_duplicate_views():
    rows = [{"state_id": str(state), "view": view} for state in range(9) for view in range(3)]
    features = torch.eye(9).repeat_interleave(3, dim=0)
    assert all(value["accuracy"] == 1. for value in cross_performance_retrieval(features, rows).values())
    collapsed = cross_performance_retrieval(torch.ones_like(features), rows)
    assert all(value["accuracy"] == value["chance"] and value["zero_gallery_vectors"] == 9 for value in collapsed.values())
    with pytest.raises(ValueError, match="complete unique"):
        cross_performance_retrieval(features, [rows[0], *rows[:-1]])


def test_regression_complete_finite_denominator():
    actual = torch.tensor([0., 2.])
    assert regression_metrics(actual, actual, torch.ones(2))["mse_vs_mean_baseline"] == 0.
    with pytest.raises(ValueError, match="nonfinite"):
        regression_metrics(actual, torch.tensor([float("nan"), 2.]), actual)

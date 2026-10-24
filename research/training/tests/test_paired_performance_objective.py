import copy

import pytest

from obruxo_training.paired_performance_objective import INPUT_TRANSFER, evaluation_views, include_input_performance


def test_same_candidate_has_input_then_transfer_views_and_legacy_is_unchanged():
    example = {"input_view": 2, "view": 0}
    assert evaluation_views({}, example) == (0,)
    assert evaluation_views({"reconstruction_objective": INPUT_TRANSFER}, example) == (2, 0)
    with pytest.raises(ValueError, match="unknown"):
        evaluation_views({"reconstruction_objective": "unreviewed"}, example)


@pytest.mark.parametrize("example", [{"input_view": 0, "view": 0}, {"input_view": 3, "view": 0}, {"input_view": True, "view": 0}])
def test_input_and_transfer_cannot_silently_collapse_to_one_performance(example):
    with pytest.raises(ValueError, match="distinct valid"):
        evaluation_views({"reconstruction_objective": INPUT_TRANSFER}, example)


def test_doubled_render_budget_is_explicit_and_does_not_mutate_the_original():
    original = {"updates": [None] * 128, "limits": {"qualification_attempts": 1280, "optimizer_attempts": 132},
                "planned_maximum_requests": {"training_targets": 192, "gradient_probes": 5632, "fit_audit": 768, "validation_audit": 504, "total": 7096},
                "storage_bound": {"estimated_archive_upper_bound_bytes": 4088393207}}
    before = copy.deepcopy(original)
    plan = include_input_performance(original)
    assert original == before
    assert plan["planned_maximum_requests"]["total"] == 12728
    assert plan["limits"]["training_attempts"] == 11840
    assert plan["limits"]["training_attempts"] + plan["limits"]["qualification_attempts"] == 13120
    assert plan["limits"]["wall_seconds"] == 36000
    assert plan["limits"]["optimizer_attempts"] == 132
    assert plan["storage_bound"]["estimated_archive_upper_bound_bytes"] < plan["limits"]["archive_bytes"]
    with pytest.raises(ValueError, match="already declared"):
        include_input_performance(plan)
    original["storage_bound"]["estimated_archive_upper_bound_bytes"] = 5 * 1024**3
    with pytest.raises(ValueError, match="archive ceiling"):
        include_input_performance(original)

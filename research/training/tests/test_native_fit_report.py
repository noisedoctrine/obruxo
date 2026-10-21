import copy

import pytest

from obruxo_training.native_fit_report import learning_diagnostics, paired_comparison, recovered_timeouts, terminal_interruption


def cases():
    return [{"state_id": str(index), "view": view,
             "rows": {name: {"scores": {"mrstft": value}} for name, value in
                      (("initial", 3. + index), ("parameter", 2. + index), ("spectral", 1. + index + .1 * view))}}
            for index in range(4) for view in (0, 1)]


def test_timeout_audit_preserves_error_and_refuses_other_or_nonterminal_failures():
    accepted = {"status": "accepted"}
    timeout = {"status": "error", "error_type": "RenderTimeout"}
    rejected = {"status": "rejected", "qa": {"clipping_count": 1}}
    assert terminal_interruption([accepted, timeout]) == timeout
    assert terminal_interruption([accepted, rejected]) == rejected
    for rows in ([accepted], [timeout, accepted], [timeout, timeout], [{"status": "error", "error_type": "RenderCrash"}],
                 [{**timeout, "qa": {"finite": False}}]):
        with pytest.raises(ValueError):
            terminal_interruption(rows)


def test_timeout_recovery_requires_later_identical_accepted_request_and_preserves_the_failure():
    common = {"phase": "training", "label": "gradient:106:10", "request_id": "request", "source_sha256": "source",
              "controlled_sha256": "controlled", "renderer_identity": {"profile": 32768}}
    failed = {**common, "attempt": 1, "status": "error", "error_type": "RenderTimeout"}
    accepted = {**common, "attempt": 2, "status": "accepted", "qa": {"finite": True}}
    original = copy.deepcopy(failed)
    assert recovered_timeouts([failed, accepted]) == [{"failed_attempt": 1, "accepted_attempt": 2,
                                                      "phase": "training", "request_id": "request"}]
    assert failed == original
    assert recovered_timeouts([accepted]) == []
    for rows in ([failed], [accepted, failed], [failed, {**accepted, "request_id": "different"}],
                 [{**failed, "status": "rejected"}, accepted], [{**failed, "error_type": "RenderCrash"}, accepted]):
        with pytest.raises(ValueError):
            recovered_timeouts(rows)


def test_learning_diagnostics_distinguish_native_response_from_a_zero_network_gradient():
    trace = [{"gradients": {"continuous_l1": .2, "category_l1": [0.]},
              "estimates": [{"categorical": [[.1, .5]], "anchor": {"categorical": [0]}}]}]
    fit = [{"state_id": "state", "view": 0, "rows": {name: {"candidate": {"categorical": [choice]}}
            for name, choice in (("initial", 0), ("parameter", 1), ("spectral", 0))}}]
    report = learning_diagnostics(trace, fit, [{"name": "filter", "values": [0, 1]}])
    assert report["categorical"]["filter"]["steps_with_native_response_but_exact_zero_score_gradient"] == 1
    assert report["hard_export_category_usage"]["parameter"]["filter"]["changed_from_initial"] == 1
    assert report["hard_export_category_usage"]["spectral"]["filter"]["changed_from_initial"] == 0
    with pytest.raises(ValueError, match="duplicate"):
        learning_diagnostics(trace, fit + fit, [{"name": "filter", "values": [0, 1]}])


def test_paired_intervals_preserve_both_performances_and_exact_constant_difference():
    result = paired_comparison(cases())
    assert result["independent_source_states"] == 4
    assert result["groups"]["reference"]["parameter"]["mean_delta"] == -1.
    assert result["groups"]["reference"]["parameter"]["mean_delta_95_percent_interval"] == [-1., -1.]
    assert result["groups"]["transfer"]["initial"]["mean_delta"] == pytest.approx(-1.9)
    assert result == paired_comparison(list(reversed(cases())))


def test_comparison_does_not_drop_missing_duplicate_or_nonfinite_observations():
    with pytest.raises(ValueError, match="both views"):
        paired_comparison(cases()[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        paired_comparison(cases() + [cases()[0]])
    for invalid in (float("nan"), -1.):
        changed = copy.deepcopy(cases())
        changed[0]["rows"]["spectral"]["scores"]["mrstft"] = invalid
        with pytest.raises(ValueError, match="finite"):
            paired_comparison(changed)

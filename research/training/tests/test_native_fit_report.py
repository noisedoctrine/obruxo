import copy

import pytest

from obruxo_training.native_fit_report import paired_comparison


def cases():
    return [{"state_id": str(index), "view": view,
             "rows": {name: {"scores": {"mrstft": value}} for name, value in
                      (("initial", 3. + index), ("parameter", 2. + index), ("spectral", 1. + index + .1 * view))}}
            for index in range(4) for view in (0, 1)]


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

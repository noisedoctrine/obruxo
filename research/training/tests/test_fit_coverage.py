import copy
import json
from pathlib import Path

import pytest

from obruxo_data.hashing import canonical_sha256
from obruxo_training.fit_coverage import control_errors, diagnose, nearest_target


def bind(report, pool):
    pool["pool_id"] = canonical_sha256(pool["pool"])
    report["plan"]["pool_id"] = pool["pool_id"]
    report["plan_id"] = canonical_sha256(report["plan"])


def fixture():
    report = json.loads((Path(__file__).parents[1] / "reports/data/calibrated_native_inverse_fit_summary.json").read_text())
    entries, assignments = [], []
    for name, split in (("training", "train"), ("validation", "validation")):
        for row in report["plan"][name]:
            entries.append({"state_id": row["state_id"], "target": row["target"]})
            assignments.append({"state_id": row["state_id"], "split": split})
    pool = {"pool": {"codec": report["plan"]["codec"], "entries": entries, "splits": {"assignments": assignments}}}
    bind(report, pool)
    return report, pool


def test_reserved_membership_is_rejected_before_statistics():
    report, pool = fixture()
    pool["pool"]["splits"]["assignments"][-1]["split"] = "test"  # Synthetic reassignment, no real test targets.
    bind(report, pool)
    with pytest.raises(ValueError, match="split, target or complete case membership"):
        diagnose(report, pool)


@pytest.mark.parametrize("change", ["duplicate", "different_transfer_prediction"])
def test_complete_pairs_and_one_prediction_per_state_are_required(change):
    report, pool = fixture()
    if change == "duplicate":
        report["validation_cases"][-1] = copy.deepcopy(report["validation_cases"][0])
    else:
        report["validation_cases"][1]["methods"]["spectral"]["candidate"]["continuous"][0] += .1
    with pytest.raises(ValueError, match="membership|cross-performance"):
        diagnose(report, pool)


def test_inactive_controls_do_not_create_false_error_or_coverage_gaps():
    target = {"continuous": [.4] * 9, "categorical": [0, 0]}
    candidate = copy.deepcopy(target)
    for index in (2, 3, 4):
        candidate["continuous"][index] = .99
    query, option = {"state_id": "query", "target": target}, {"state_id": "option", "target": candidate}
    assert nearest_target(query, [option])["normalized_control_rmse"] == 0
    candidate["continuous"][0] += .2
    case = {"state_id": "query", "view": 0, "methods": {name: {"candidate": candidate} for name in ("initial", "parameter", "spectral")}}
    rows = control_errors([case], {"query": query}, [{"name": str(index)} for index in range(9)])
    assert rows[0]["methods"]["spectral"]["absolute_error"]["mean"] == pytest.approx(.2)
    assert rows[3]["source_active_states"] == 0
    assert rows[3]["methods"]["spectral"]["absolute_error"] == {"count": 0}

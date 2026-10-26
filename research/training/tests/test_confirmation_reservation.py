from collections import Counter
from dataclasses import replace
import itertools
import json

import pytest

from obruxo_data.hashing import canonical_sha256
from obruxo_data.midi import Performance, TempoMap
from obruxo_training.confirmation_reservation import BANK_SIZE, LIMITS, PER_STRATUM, ROLES, confirmation_bank, development_precision, select_families
from obruxo_training.pilot import PilotPool


@pytest.fixture(scope="module")
def populations():
    parent = PilotPool.create(512, seed=0)
    return parent, parent.expand(4096)


def test_reserved_families_are_new_validation_roots_and_preserve_every_old_assignment(populations):
    parent, expanded = populations
    cases, proof = select_families(parent, expanded)
    assert len(cases) == 192 and proof["preserved_states"] == 512
    assert not {row["state_id"] for row in cases} & {entry.state_id for entry in parent.entries}
    assert len({row["lineage_id"] for row in cases}) == len(cases)
    assert all(row["split"] == "validation" and expanded.manifest.by_state[row["state_id"]].lineage_id == row["lineage_id"] for row in cases)
    assert all(expanded.manifest.by_state[row.state_id] == row for row in parent.manifest.assignments)
    assert select_families(parent, expanded)[0] == cases


def test_performance_assignment_cannot_reveal_either_source_category(populations):
    cases, _proof = select_families(*populations)
    counts = Counter((tuple(row["category"]), row["bank_index"]) for row in cases)
    expected = {(category, bank): PER_STRATUM // BANK_SIZE for category in itertools.product(range(3), range(2)) for bank in range(BANK_SIZE)}
    assert dict(counts) == expected


def test_changed_parent_and_insufficient_new_validation_are_rejected(populations):
    parent, expanded = populations
    with pytest.raises(ValueError, match="same seed"):
        select_families(replace(parent, seed=1), expanded)
    with pytest.raises(ValueError, match="insufficient independent"):
        select_families(parent, parent.expand(520))


def test_bank_has_distinct_valid_complete_observations_within_storage_budget():
    bank = confirmation_bank()
    assert len(bank) == 24 and len({row["performance_id"] for row in bank}) == 24
    assert Counter(row["role"] for row in bank) == {role: BANK_SIZE for role in ROLES}
    for row in bank:
        performance = Performance.from_dict(row["performance"])
        assert canonical_sha256(performance.to_dict()) == row["performance_id"]
        assert row["render_frames"] == TempoMap.from_performance(performance).render_frame_count(performance.end_tick, 2., 44100)
        assert 2 * 44100 < row["render_frames"] <= 5 * 44100
    assert 192 * 19 == LIMITS["planned_native_requests"]
    assert LIMITS["native_requests"] - LIMITS["planned_native_requests"] == 192
    assert LIMITS["planned_native_requests"] * 5 * 44100 * 2 * 4 + 512 * 1024**2 < LIMITS["archive_bytes"]


def test_precision_uses_independent_families_and_rejects_changed_evidence(tmp_path):
    cells = [{"input_view": 0, "playback_view": playback, "recipe": {"state_id": family}}
             for family in ("one", "two") for playback in (0, 1, 2)]
    plan = {"fit_audit_cells": cells}
    (tmp_path / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    rows = [{"index": index, "arm": arm, "scores": {"mrstft": 1. if arm == "parameter" else (0.5 if index < 3 else 1.5)}}
            for index in range(6) for arm in ("parameter", "spectral")]
    (tmp_path / "finals.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    from obruxo_training.confirmation_reservation import digest
    verified = {"verified": True, "plan_id": canonical_sha256(plan),
                "evidence_sha256": {name: digest(tmp_path / name) for name in ("plan.json", "finals.jsonl")}}
    (tmp_path / "verification.json").write_text(json.dumps(verified), encoding="utf-8")
    result = development_precision(tmp_path)
    assert all(row["independent_families"] == 2 for row in result["roles"].values())
    assert result["roles"]["reference"]["relative_paired_sd"] == result["roles"]["transfer"]["relative_paired_sd"]
    (tmp_path / "finals.jsonl").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="changed after verification"):
        development_precision(tmp_path)

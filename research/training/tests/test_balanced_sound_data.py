from collections import Counter, defaultdict

import pytest

from obruxo_data.hashing import canonical_sha256
from obruxo_training.balanced_sound_data import counterbalanced_validation_records, training_records
from obruxo_training.performances import initial_bank
from obruxo_training.pilot import PilotPool
from obruxo_training.temporal_proxy_data import category_balanced_entries


@pytest.fixture(scope="module")
def pool():
    return PilotPool.create(512, seed=0)


def test_training_patterns_are_balanced_within_every_category_and_order_independent(pool):
    entries = category_balanced_entries(pool, "train", 64)
    records = training_records(pool, entries)
    reverse = {row["state_id"]: row for row in training_records(pool, entries[::-1])}
    groups = defaultdict(Counter)
    for record in records:
        assert record == reverse[record["state_id"]]
        assert record["input_patterns"][0] == "held"
        groups[tuple(record["target"]["categorical"])][record["input_patterns"][1]] += 1
    assert len(groups) == 6
    for counts in groups.values():
        values = [counts[pattern] for pattern in ("staccato", "melody", "chord")]
        assert max(values) - min(values) <= 1


def test_new_development_panel_covers_all_patterns_in_every_category_and_preserves_sources(pool):
    entries = category_balanced_entries(pool, "validation", 12)
    records = counterbalanced_validation_records(pool, entries)
    assert len(records) == 24 and Counter(row["state_id"] for row in records) == Counter({entry.state_id: 2 for entry in entries})
    reserved = {row.performance_id for row in initial_bank()}
    for view in (0, 1):
        groups = defaultdict(Counter)
        for record in records:
            assert canonical_sha256(record["performances"][view]) not in reserved
            groups[tuple(record["target"]["categorical"])][record["input_patterns"][view]] += 1
        assert len(groups) == 6 and all(counts == Counter(held=1, staccato=1, melody=1, chord=1) for counts in groups.values())
    with pytest.raises(ValueError, match="wrong-split"):
        training_records(pool, entries)
    with pytest.raises(ValueError, match="two states"):
        counterbalanced_validation_records(pool, entries[:6])

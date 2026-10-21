from collections import Counter
import random
from types import SimpleNamespace

import pytest

from obruxo_training.codec import PilotCodec, PilotTarget
from obruxo_training.pilot import PilotEntry
from obruxo_training.training_coverage import expand_training


def pool_fixture():
    codec, rng = PilotCodec(), random.Random(7)
    entries = [PilotEntry(f"train-{index:03d}", codec.canonical_target(PilotTarget(tuple(rng.random() for _ in range(9)), (index % 6 // 2, index % 2))))
               for index in range(120)]
    entries += [PilotEntry("reserved-validation", None), PilotEntry("reserved-test", None)]
    assignments = {entry.state_id: SimpleNamespace(split="train") for entry in entries[:120]}
    assignments.update({"reserved-validation": SimpleNamespace(split="validation"), "reserved-test": SimpleNamespace(split="test")})
    return SimpleNamespace(codec=codec, entries=entries, manifest=SimpleNamespace(by_state=assignments)), entries[:32]


def test_expansion_preserves_original_states_balances_categories_and_never_reads_reserved_targets():
    pool, original = pool_fixture()
    selected, contract = expand_training(pool, original)
    assert selected[:32] == original and len({entry.state_id for entry in selected}) == 64
    assert Counter(entry.target.categorical for entry in selected) == {tuple(row["indices"]): row["count"] for row in contract["category_quotas"]}
    assert all(pool.manifest.by_state[entry.state_id].split == "train" for entry in selected)
    pool.entries = list(reversed(pool.entries))  # Pool ordering and unread reserved targets cannot change ties or selection.
    again, metadata = expand_training(pool, original)
    assert again == selected and metadata == contract and len(contract["additions"]) == 32


def test_reserved_original_or_insufficient_training_population_cannot_enter_expansion():
    pool, original = pool_fixture()
    pool.manifest.by_state[original[0].state_id].split = "validation"
    with pytest.raises(ValueError, match="training states only"):
        expand_training(pool, original)
    pool, original = pool_fixture()
    pool.entries = pool.entries[:40]
    with pytest.raises(ValueError, match="not enough training states"):
        expand_training(pool, original)


def test_first_addition_is_the_farthest_candidate_in_the_most_underfilled_category():
    pool, _original = pool_fixture()
    entries = []
    for index, entry in enumerate(pool.entries):
        if entry.target is None:
            entries.append(entry)
            continue
        value = .5 if index < 32 else 1. if index == 32 else .51
        entries.append(PilotEntry(entry.state_id, pool.codec.canonical_target(PilotTarget((value,) * 9, entry.target.categorical))))
    pool.entries = entries
    selected, contract = expand_training(pool, entries[:32])
    assert selected[32].state_id == "train-032"
    assert contract["additions"][0]["category_indices"] == [1, 0]
    assert contract["additions"][0]["normalized_control_rmse"] == pytest.approx(.5)

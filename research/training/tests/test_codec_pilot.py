from dataclasses import replace
import json

import numpy as np
import pytest

from obruxo_training.codec import CONTINUOUS, PilotCodec, PilotTarget
from obruxo_training.cli import main
from obruxo_training.pilot import PilotPool, PilotSampler
from obruxo_training.splits import build_splits


@pytest.fixture(scope="module")
def pool():
    return PilotPool.create(count=48, seed=17)


def test_state_schedule_covers_every_training_state_and_rotates_all_performances():
    pool = PilotPool.create(12, seed=2)
    sampler = PilotSampler(pool, policy="state_round_robin")
    states = {entry.state_id for entry in pool.manifest.assignments if entry.split == "train"}
    assert sampler.epoch_size == len(states)
    seen = {state: set() for state in states}
    for epoch in range(4):
        recipes = [sampler.recipe_at(epoch * sampler.epoch_size + index) for index in range(sampler.epoch_size)]
        assert {recipe.state_id for recipe in recipes} == states
        for recipe in recipes:
            assert recipe.split == "train" and recipe.performance_role == "reference"
            seen[recipe.state_id].add(recipe.performance_id)
    assert all(len(performances) == 4 for performances in seen.values())
    fresh = PilotSampler(pool, policy="state_round_robin")
    assert fresh.recipe_at(3).recipe_id == sampler.recipe_at(3).recipe_id
    assert fresh.sampler_id != PilotSampler(pool).sampler_id


def test_pilot_round_trip_preserves_active_controls_and_masks(pool):
    for entry in pool.entries:
        codec = pool.codec
        raw = codec.decode(entry.target)
        encoded = codec.encode(raw)
        assert encoded.categorical == entry.target.categorical
        np.testing.assert_allclose(encoded.continuous, entry.target.continuous, rtol=0, atol=1e-14)
        left, right = json.loads(raw), json.loads(codec.decode(encoded))
        for name, *_ in CONTINUOUS:
            assert np.float32(left["settings"][name]) == np.float32(right["settings"][name])
        mask = codec.continuous_mask(encoded)
        assert mask[2] == (encoded.categorical[0] != 0)
        assert mask[3] == mask[4] == (encoded.categorical[1] != 0)


@pytest.mark.parametrize("change", ["effect", "asset", "route", "inactive", "range"])
def test_codec_rejects_unsupported_state_instead_of_projecting(change):
    codec = PilotCodec()
    document = json.loads(codec.decode(PilotTarget((0.5,) * len(CONTINUOUS), (0, 0))))
    settings = document["settings"]
    if change == "effect":
        settings["reverb_on"] = 1.0
    elif change == "asset":
        settings["wavetables"][0]["name"] = "unrepresented asset metadata"
    elif change == "route":
        settings["modulations"][0] = {"source": "lfo_1", "destination": "osc_1_level"}
    elif change == "inactive":
        settings["filter_1_cutoff"] = 90.0
    else:
        settings["osc_1_level"] = 0.9
    with pytest.raises(ValueError):
        codec.encode(json.dumps(document).encode())


@pytest.mark.parametrize("target", [PilotTarget((float("nan"),) * 9, (0, 0)), PilotTarget((1.1,) * 9, (0, 0)),
                                   PilotTarget((0.5,) * 8, (0, 0)), PilotTarget((0.5,) * 9, (3, 0))])
def test_codec_rejects_malformed_predictions(target):
    with pytest.raises(ValueError):
        PilotCodec().decode(target)


def test_pool_is_reproducible_compact_and_tamper_checked(pool, tmp_path):
    assert pool.pool_id == PilotPool.create(count=48, seed=17).pool_id
    path = tmp_path / "pool.json"
    pool.save(path)
    assert PilotPool.load(path).pool_id == pool.pool_id
    assert path.stat().st_size < 100_000
    with pytest.raises(FileExistsError):
        pool.save(path)
    value = json.loads(path.read_text())
    value["pool"]["entries"][0]["target"]["continuous"][0] = 0.1
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="artifact changed"):
        PilotPool.load(path)


def test_expansion_cli_preserves_parent_targets_and_all_frozen_splits(pool, tmp_path):
    parent_path = tmp_path / "parent.json"
    pool.save(parent_path)
    original = parent_path.read_bytes()
    output = tmp_path / "expanded"
    assert main(["expand-pilot", "--pool", str(parent_path), "--count", "80", "--output", str(output)]) == 0
    expanded = PilotPool.load(output / "pool.json")
    assert expanded.entries[:len(pool.entries)] == pool.entries
    assert all(expanded.manifest.by_state[state] == assignment for state, assignment in pool.manifest.by_state.items())
    assert parent_path.read_bytes() == original
    declaration = json.loads((output / "expansion.json").read_text())
    assert declaration == expanded.expansion_from(pool)
    assert declaration["preserved_states"] == 48 and declaration["added_states"] == 32
    assert declaration["parent_pool_id"] == pool.pool_id and declaration["expanded_pool_id"] == expanded.pool_id
    assert all(declaration["added_states_by_split"][split] > 0 for split in ("train", "validation", "test"))
    assert expanded.pool_id != pool.pool_id


def test_expansion_rejects_changed_prefix_seed_and_frozen_assignments(pool):
    expanded = pool.expand(64)
    changed = replace(expanded, entries=(expanded.entries[1], expanded.entries[0], *expanded.entries[2:]))
    with pytest.raises(ValueError, match="ordered parent population"):
        changed.expansion_from(pool)
    with pytest.raises(ValueError, match="same seed"):
        replace(expanded, seed=pool.seed + 1).expansion_from(pool)
    manifest = build_splits((entry.state_id for entry in expanded.entries), seed=pool.seed + 1)
    with pytest.raises(ValueError, match="frozen lineage or split"):
        replace(expanded, manifest=manifest).expansion_from(pool)
    with pytest.raises(ValueError, match="invalid independent-root population"):
        replace(expanded, entries=(*expanded.entries[:-1], expanded.entries[0])).expansion_from(pool)


@pytest.mark.parametrize("count", [True, 0, 48, 4097])
def test_expansion_rejects_non_growth_and_unbounded_counts(pool, count):
    with pytest.raises(ValueError, match="must exceed the parent"):
        pool.expand(count)


def test_sampler_covers_pairs_and_replays_sequence_without_evaluation_leakage(pool):
    sampler = PilotSampler(pool)
    recipes = [sampler.recipe_at(index) for index in range(sampler.epoch_size)]
    assert len({(recipe.state_id, recipe.performance_id) for recipe in recipes}) == sampler.epoch_size
    assert all(recipe.split == "train" and recipe.performance_role == "reference" for recipe in recipes)
    restored = PilotSampler(pool)
    assert restored.recipe_at(9).recipe_id == recipes[9].recipe_id
    assert sampler.recipe_at(sampler.epoch_size).sequence_id == sampler.epoch_size
    with pytest.raises(ValueError, match="reserved"):
        PilotSampler(pool, role="transfer")
    held_out = PilotSampler(pool, split="validation", role="transfer").recipe_at(0)
    assert held_out.purpose == "evaluate" and held_out.split == "validation" and held_out.performance_role == "transfer"


def test_display_metadata_is_not_a_codec_target(pool):
    codec = pool.codec
    raw = codec.decode(pool.entries[0].target)
    value = json.loads(raw)
    value["preset_name"] = "another label"
    np.testing.assert_allclose(codec.encode(json.dumps(value).encode()).continuous, codec.encode(raw).continuous)

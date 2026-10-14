from dataclasses import replace
import hashlib
import json

import pytest

from obruxo_training.contracts import RenderRecipe, TrainingContract
from obruxo_training.identity import PresetIdentity
from obruxo_training.performances import initial_bank
from obruxo_training.splits import SplitLedger, SplitManifest, build_splits


def digest(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def test_transitive_lineage_and_duplicates_are_order_independent():
    a, b, c, d = map(digest, range(4))
    first = build_splits([a, b, c, d, a], parents={b: [a], c: [b]}, seed=42)
    second = build_splits([d, c, b, a], parents={c: [b], b: [a]}, seed=42)
    assert first.manifest_id == second.manifest_id
    assert len({first.by_state[state].lineage_id for state in (a, b, c)}) == 1
    assert len({first.by_state[state].split for state in (a, b, c)}) == 1
    assert SplitManifest.from_dict(first.to_dict()) == first
    with pytest.raises(TypeError):
        first.by_state[a] = first.by_state[b]


@pytest.mark.parametrize("parents", [{digest(0): [digest(99)]}, {digest(0): [digest(1)], digest(1): [digest(0)]}])
def test_missing_ancestors_and_cycles_are_rejected(parents):
    with pytest.raises(ValueError):
        build_splits([digest(0), digest(1)], parents=parents)


@pytest.mark.parametrize("parents", [[], "", {digest(0): digest(1)}, {digest(0): ["not-a-state-id"]}])
def test_malformed_ancestry_cannot_silently_disable_grouping(parents):
    with pytest.raises(ValueError):
        build_splits([digest(0), digest(1)], parents=parents)


def test_online_variants_inherit_and_cannot_cross_frozen_groups():
    manifest = build_splits(map(digest, range(30)))
    ledger = SplitLedger(manifest)
    train = next(entry for entry in manifest.assignments if entry.split == "train")
    test = next(entry for entry in manifest.assignments if entry.split == "test")
    child = ledger.register_variant(digest("child"), train.state_id)
    assert (child.lineage_id, child.split) == (train.lineage_id, "train")
    with pytest.raises(ValueError, match="collides"):
        ledger.register_variant(test.state_id, child.state_id)
    restored = SplitLedger.from_dict(ledger.to_dict(), manifest)
    assert restored.lookup(child.state_id) == child
    assert restored.manifest.manifest_id == manifest.manifest_id
    with pytest.raises(ValueError, match="different frozen"):
        SplitLedger.from_dict(ledger.to_dict(), build_splits(map(digest, range(30)), seed=1))
    serialized = ledger.to_dict()
    serialized["variants"][0]["parent_state_id"] = test.state_id
    assert ledger.to_dict()["variants"][0]["parent_state_id"] == train.state_id


def test_split_file_tampering_is_rejected():
    manifest = build_splits([digest(0)])
    value = manifest.to_dict()
    value["assignments"][0]["split"] = "test" if value["assignments"][0]["split"] == "train" else "train"
    with pytest.raises(ValueError, match="split assignment"):
        SplitManifest.from_dict(value)


def recipe_factory(init_raw, schema, catalog, desired_split="train"):
    state = PresetIdentity.from_bytes(init_raw).state
    manifest = next(manifest for seed in range(100) if (manifest := build_splits([state], seed=seed)).by_state[state].split == desired_split)
    contract = TrainingContract.create(schema, catalog)

    def make(performance=None, **kwargs):
        return RenderRecipe.create(init_raw, performance or initial_bank()[0].performance(), schema=schema,
                                   catalog=catalog, contract=kwargs.pop("contract", contract), ledger=SplitLedger(manifest),
                                   purpose=kwargs.pop("purpose", "train"), sequence_id=kwargs.pop("sequence_id", 0), **kwargs)
    return make


def test_recipe_binds_source_performance_and_immutable_request(init_raw, schema, catalog):
    make = recipe_factory(init_raw, schema, catalog)
    performance = initial_bank()[0].performance()
    recipe = make(performance)
    before = recipe.render_request().request_id
    performance.events.clear()
    assert recipe.render_request().request_id == before
    assert recipe.recipe_id != make(sequence_id=1).recipe_id
    assert recipe.render_request().sample_rate == 44100
    assert recipe.render_request().channels == 2
    assert recipe.raw_sha256 == hashlib.sha256(init_raw).hexdigest()
    assert recipe.canonical_sha256 == PresetIdentity.from_bytes(init_raw).canonical


def test_reserved_performance_cannot_be_relabelled_for_training(init_raw, schema, catalog):
    make = recipe_factory(init_raw, schema, catalog)
    transfer = initial_bank()[4].performance()
    with pytest.raises(ValueError, match="transfer performance"):
        make(transfer)
    assert make(transfer, purpose="evaluate").performance_role == "transfer"
    changed = initial_bank()[0].performance()
    changed.add_note(pitch=72, velocity=80, start_tick=0, duration_ticks=240)
    with pytest.raises(ValueError, match="outside the frozen"):
        make(changed)


def test_held_out_patch_cannot_enter_training(init_raw, schema, catalog):
    make = recipe_factory(init_raw, schema, catalog, desired_split="test")
    with pytest.raises(ValueError, match="held-out state"):
        make()
    assert make(purpose="evaluate").split == "test"


def test_changed_contract_cannot_resume_or_create_recipe(init_raw, schema, catalog):
    contract = TrainingContract.create(schema, catalog)
    changed = replace(contract, prefetch_clips=16)
    with pytest.raises(ValueError, match="contract changed"):
        contract.require_match(changed.to_dict())
    with pytest.raises(ValueError, match="contract changed"):
        recipe_factory(init_raw, schema, catalog)(contract=changed)
    saved = json.loads(json.dumps(contract.to_dict()))
    contract.require_match(saved)


def test_schema_content_changes_invalidate_contract(tmp_path, schema, catalog):
    contract = TrainingContract.create(schema, catalog)
    atlas = tmp_path / schema.parameter_atlas_path.name
    atlas.write_bytes(schema.parameter_atlas_path.read_bytes() + b"\n")
    modified = TrainingContract.create(replace(schema, parameter_atlas_path=atlas), catalog)
    assert modified.schema_id == contract.schema_id and modified.contract_id != contract.contract_id
    with pytest.raises(ValueError, match="contract changed"):
        contract.require_match(modified.to_dict())

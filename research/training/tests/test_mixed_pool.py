from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest
import torch

from obruxo_data.hashing import canonical_json, canonical_sha256
from obruxo_training.mixed_manifest import MixedManifest
from obruxo_training.mixed_pool import MixedPool, evaluation_groups
from obruxo_training.performance_eligibility import eligible, longest_held_seconds
from obruxo_training.pilot import PilotPool, PilotSampler
from obruxo_training.splits import SplitManifest, build_splits
from test_corpus_pool import original_pool, files  # noqa: F401
from test_corpus_inputs import corpus  # noqa: F401
from test_curve_inputs import layout_codec  # noqa: F401
from test_combined import codec, expression  # noqa: F401
from test_factory_assets import bank  # noqa: F401
from test_lfo_bank import population  # noqa: F401


def digest(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


@pytest.fixture(scope="module")
def mixture(tmp_path_factory, original_pool, files, layout_codec):  # noqa: F811
    directory = tmp_path_factory.mktemp("mixture")
    corpus_path, authored_path = directory / "corpus.json", directory / "authored.json"
    original_pool.save(corpus_path)
    authored = PilotPool.create(16, codec=layout_codec)
    authored.save(authored_path)
    return MixedPool.create(corpus_path, authored_path, files[2])


def test_frozen_union_and_program_aliases_never_move_splits():
    left, right = [build_splits([digest(f"{group}-{index}") for index in range(24)]) for group in ("left", "right")]
    manifest = MixedManifest((("corpus", left), ("authored", right)))
    assert SplitManifest.from_dict(manifest.to_dict()) == manifest
    assert all(manifest.by_state[state] == entry for parent in (left, right) for state, entry in parent.by_state.items())
    a = next(entry for entry in left.assignments if entry.split == "train")
    b = next(entry for entry in right.assignments if entry.split == "train")
    program = digest("same program")
    groups = evaluation_groups(((program, (("corpus", a.state_id), ("authored", b.state_id))),), manifest)
    assert groups[a.lineage_id] == groups[b.lineage_id]
    held = next(entry for entry in right.assignments if entry.split == "validation")
    with pytest.raises(ValueError, match="crosses frozen splits"):
        evaluation_groups(((program, (("corpus", a.state_id), ("authored", held.state_id))),), manifest)


def test_mixture_preserves_parents_adds_validation_and_never_schedules_final_test(mixture, tmp_path):
    for parent in (mixture.corpus.manifest, mixture.authored.manifest):
        assert all(mixture.manifest.by_state[state] == entry for state, entry in parent.by_state.items())
    assert mixture.summary()["sources_by_split"]["validation"]["corpus"] == len(mixture.validation) > 0
    assert mixture.summary()["sources_by_split"]["validation"]["authored"] > 0
    assert all(mixture.manifest.by_state[entry.state_id].split != "test" for entry in mixture.entries)
    with pytest.raises(ValueError, match="no pilot states"):
        PilotSampler(mixture, split="test")
    for source in ("authored", "corpus", "corpus_validation"):
        index = next(index for index, entry in enumerate(mixture.entries) if entry.source == source)
        entry = mixture.entries[index]
        raw = mixture.source_preset(entry.target, index)
        recipe_source = json.loads(raw)
        assert mixture.codec.encode(raw) == entry.target
        assert canonical_sha256(recipe_source) == canonical_sha256(json.loads(mixture.source_preset(entry.target, index)))
    path = tmp_path / "mixture.json"
    mixture.save(path)
    loaded = PilotPool.load(path)
    assert loaded.pool_id == mixture.pool_id and loaded.entries == mixture.entries
    changed = json.loads(path.read_text())
    changed["pool"]["sources"]["corpus"] = "0" * 64
    changed["pool_id"] = canonical_sha256(changed["pool"])
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="mixed sources"):
        PilotPool.load(path)


def test_balanced_attempts_visit_every_state_and_use_adequate_references(mixture):
    sampler = PilotSampler(mixture, policy="source_balanced")
    by_state = {entry.state_id: entry for entry in mixture.entries}
    groups, seen = [], set()
    for sequence in range(sampler.epoch_size):
        recipe = sampler.recipe_at(sequence)
        entry = by_state[recipe.state_id]
        groups.append(mixture.sampling_group(entry))
        seen.add(recipe.state_id)
        assert recipe.split == "train" and recipe.performance_role == "reference"
        assert eligible(mixture.codec, entry.target, recipe.render_request().performance)
        assert recipe.render_request().preset_json == canonical_json(json.loads(mixture.source_preset(entry.target, mixture.entries.index(entry))))
    assert groups == ["authored", "corpus"] * (sampler.epoch_size // 2)
    assert seen == {entry.state_id for entry in mixture.entries if mixture.manifest.by_state[entry.state_id].split == "train"}
    assert longest_held_seconds(sampler.recipe_at(0).render_request().performance) <= 1.5
    with pytest.raises(ValueError, match="mixed training pool"):
        PilotSampler(mixture, split="validation", policy="source_balanced")
    assert PilotSampler(mixture, policy="source_balanced").sampler_id == sampler.sampler_id


def test_source_corruption_and_final_test_injection_fail(mixture):
    index = next(index for index, entry in enumerate(mixture.entries) if entry.source == "corpus_validation")
    entry = mixture.entries[index]
    source = mixture.validation[entry.source_index]
    path = mixture.corpus.corpus_root / source.relative_path
    original = path.read_bytes()
    try:
        path.write_bytes(original + b" ")
        with pytest.raises(ValueError, match="original bytes changed"):
            mixture.source_preset(entry.target, index)
    finally:
        path.write_bytes(original)
    test_index = next(index for index, entry in enumerate(mixture.authored.entries) if mixture.authored.manifest.by_state[entry.state_id].split == "test")
    test_entry = mixture.authored.entries[test_index]
    injected = replace(mixture.entries[0], state_id=test_entry.state_id, target=test_entry.target, source="authored", source_index=test_index)
    with pytest.raises(ValueError, match="final-test"):
        replace(mixture, entries=(injected,) + mixture.entries[1:])
    entries = tuple(replace(value, raw_sha256="0" * 64) if value.state_id == entry.state_id else value for value in mixture.entries)
    with pytest.raises(ValueError, match="original corpus byte hash"):
        replace(mixture, entries=entries)


def test_deferred_authored_verification_still_checks_the_source_before_use(mixture, tmp_path):
    saved = {"pool_id": mixture.authored.pool_id, "pool": mixture.authored.to_dict()}
    saved["pool"]["entries"][0]["state_id"] = digest("wrong original state")
    saved["pool_id"] = canonical_sha256(saved["pool"])
    path = tmp_path / "corrupted.json"
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError):
        PilotPool.load(path, verify_sources=False)
    loaded = PilotPool.load(Path(mixture.source_paths["authored"]), verify_sources=False)
    with pytest.raises(ValueError, match="pilot source differs"):
        loaded.source_preset(loaded.entries[1].target, 0)
    saved = {"pool_id": mixture.authored.pool_id, "pool": mixture.authored.to_dict()}
    position = next(index for index, (name, *_) in enumerate(mixture.authored.codec.continuous) if name == "osc_1_pan")
    saved["pool"]["entries"][0]["target"]["continuous"][position] = .25
    saved["pool_id"] = canonical_sha256(saved["pool"])
    path.write_text(json.dumps(saved))
    deferred = PilotPool.load(path, verify_sources=False)
    with pytest.raises(ValueError, match="differs|changed"):
        deferred.source_preset(deferred.entries[0].target, 0)


def test_mixed_training_recovers_exactly_and_exports_only_model_codec(mixture, tmp_path):
    from obruxo_training.stream import StreamConfig
    from obruxo_training.trainer import TrainConfig, load_model, train_pilot
    from test_stream import FakeRendererFactory
    config = TrainConfig(steps=2, replay_capacity=4, batch_size=4, render_budget=5, checkpoint_every=1, cpu_threads=2,
                         sampling_policy="source_balanced")

    def train(name, **kwargs):
        return train_pilot(mixture, tmp_path / name, config=config, stream_config=StreamConfig(prefetch_clips=1),
                           renderer_factory=FakeRendererFactory(), **kwargs)

    full = train("full")
    train("partial", stop_after=1)
    recovered = train("recovered", resume=tmp_path / "partial/checkpoint.pt")
    assert full["model_state_sha256"] == recovered["model_state_sha256"]
    saved = torch.load(tmp_path / "full/checkpoint.pt", weights_only=True)
    states = {item["state_id"] for item in saved["replay"]}
    assert {mixture.sampling_group(entry) for entry in mixture.entries if entry.state_id in states} == {"authored", "corpus"}
    _, _, exported = load_model(tmp_path / "full/model.pt")
    assert exported.codec_id == mixture.codec.codec_id
    assert "source_paths" not in json.dumps(exported.to_dict())
    from obruxo_training.evaluate import evaluate_pilot
    evaluate_pilot(mixture, tmp_path / "full/checkpoint.pt", tmp_path / "evaluation", count=2,
                   renderer_factory=FakeRendererFactory())
    plan = json.loads((tmp_path / "evaluation/plan.json").read_text())
    assert plan["primary"] == "equal_weight_mean_of_reference_and_transfer_per_lineage_mrstft"
    assert plan["case_lineages"] and all(group == mixture.evaluation_group(state) for state, group in plan["case_lineages"].items())

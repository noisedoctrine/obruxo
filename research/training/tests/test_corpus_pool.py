from copy import deepcopy
import json
import shutil

import pytest
import torch

from obruxo_data.hashing import canonical_json, canonical_sha256
from obruxo_training.audit import audit_corpus, write_json
from obruxo_training.corpus_pool import CorpusPool
from obruxo_training.pilot import PilotPool, PilotSampler
from test_corpus_inputs import corpus  # noqa: F401
from test_curve_inputs import layout_codec  # noqa: F401
from test_combined import codec, expression  # noqa: F401
from test_factory_assets import bank  # noqa: F401
from test_lfo_bank import population  # noqa: F401


@pytest.fixture(scope="module")
def files(tmp_path_factory, corpus):  # noqa: F811
    root = tmp_path_factory.mktemp("original-corpus")
    sources, snapshot, candidates = root / "sources", root / "snapshot", root / "candidates.json"
    sources.mkdir()
    for index in range(12):
        document = corpus.schema.template()
        document["macro1"] = "Original macro label"
        document["settings"].update(osc_1_pan=(index - 6) / 20, osc_1_random_phase=.37, sample_random_phase=1.)
        (sources / f"{index}.vital").write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    audit_corpus(sources, snapshot, schema=corpus.schema, catalog=corpus.catalog)
    records = json.loads((snapshot / "members.json").read_text())["records"]
    entries = [{"state_id": row["identity"]["state"], "lineage_id": row["lineage_id"], "relative_path": row["relative_path"],
                "raw_sha256": row["identity"]["raw"], "target": corpus.encode((sources / row["relative_path"]).read_bytes()).to_dict(),
                "label_changes": []} for row in records if row["static_eligible"] and row["split"] == "train"]
    manifest = json.loads((snapshot / "splits.json").read_text())
    write_json(candidates, {"codec": corpus.to_dict(), "split_manifest": manifest, "entries": entries})
    return root, sources, snapshot, candidates


@pytest.fixture(scope="module")
def original_pool(files):
    _, sources, snapshot, candidates = files
    return CorpusPool.create(snapshot, sources, candidates)


def test_original_bytes_phases_and_frozen_manifest_survive_reload_and_relocation(original_pool, files, tmp_path):
    _, sources, snapshot, _ = files
    path = tmp_path / "pool.json"
    original_pool.save(path)
    loaded = PilotPool.load(path)
    relocated = tmp_path / "relocated"
    shutil.copytree(sources, relocated)
    moved = PilotPool.load(path, corpus_root=relocated)
    assert loaded.pool_id == moved.pool_id == original_pool.pool_id
    assert loaded.manifest.to_dict() == json.loads((snapshot / "splits.json").read_text())
    assert len(loaded.manifest.assignments) > len(loaded.entries) >= 2
    sampler = PilotSampler(loaded, policy="attack_compatible")
    recipe = sampler.recipe_at(0)
    entry = next(entry for entry in loaded.entries if entry.state_id == recipe.state_id)
    raw = (sources / entry.relative_path).read_bytes()
    assert loaded.source_preset(entry.target, loaded.entries.index(entry)) == raw
    assert recipe.render_request().preset_json == canonical_json(json.loads(raw))
    document = json.loads(recipe.render_request().preset_json)
    assert document["macro1"] == "Original macro label" and document["settings"]["osc_1_random_phase"] == .37
    assert document["settings"]["sample_random_phase"] == 1.
    with pytest.raises(ValueError, match="no pilot states"):
        PilotSampler(loaded, split="test")
    assert moved.source_preset(moved.entries[0].target, 0) == loaded.source_preset(loaded.entries[0].target, 0)


def test_changed_sources_and_target_labels_fail_before_render(original_pool, files, tmp_path):
    _, sources, _, _ = files
    relocated = tmp_path / "sources"
    shutil.copytree(sources, relocated)
    saved = {"pool_id": original_pool.pool_id, "pool": original_pool.to_dict(), "corpus_root": str(relocated)}
    changed = CorpusPool.from_saved(saved)
    entry = changed.entries[0]
    path = relocated / entry.relative_path
    original = path.read_bytes()
    path.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="source bytes changed"):
        changed.source_preset(entry.target, 0)
    path.write_bytes(original)
    value = deepcopy(saved)
    position = next(index for index, (name, *_) in enumerate(original_pool.codec.continuous) if name == "osc_1_pan")
    value["pool"]["entries"][0]["target"]["continuous"][position] = .25
    value["pool_id"] = canonical_sha256(value["pool"])
    corrupted = CorpusPool.from_saved(value)
    with pytest.raises(ValueError, match="target does not match"):
        corrupted.source_preset(corrupted.entries[0].target, 0)


def test_membership_paths_and_policy_tampering_are_rejected(original_pool, files, tmp_path):
    _, _, snapshot, candidates = files
    labels = json.loads(candidates.read_text())
    labels["entries"].pop()
    invalid = tmp_path / "incomplete.json"
    write_json(invalid, labels)
    with pytest.raises(ValueError, match="every eligible training source"):
        CorpusPool.create(snapshot, original_pool.corpus_root, invalid)
    saved = {"pool_id": original_pool.pool_id, "pool": original_pool.to_dict(), "corpus_root": str(original_pool.corpus_root)}
    for field, value in (("relative_path", "../escape.vital"), ("relative_path", "C:/escape.vital")):
        changed = deepcopy(saved)
        changed["pool"]["entries"][0][field] = value
        changed["pool_id"] = canonical_sha256(changed["pool"])
        with pytest.raises(ValueError, match="relative path|escapes"):
            CorpusPool.from_saved(changed)
    changed = deepcopy(saved)
    changed["pool"]["source_policy"]["audio"] = "decode_labels"
    changed["pool_id"] = canonical_sha256(changed["pool"])
    with pytest.raises(ValueError, match="source policy changed"):
        CorpusPool.from_saved(changed)


def test_corpus_training_recovers_and_export_does_not_embed_source_paths(original_pool, tmp_path):
    from obruxo_training.stream import StreamConfig
    from obruxo_training.trainer import TrainConfig, load_model, train_pilot
    from test_stream import FakeRendererFactory
    config = TrainConfig(steps=2, replay_capacity=2, batch_size=2, render_budget=3, checkpoint_every=1, cpu_threads=2,
                         sampling_policy="attack_compatible")

    def train(name, **kwargs):
        return train_pilot(original_pool, tmp_path / name, config=config, stream_config=StreamConfig(prefetch_clips=1),
                           renderer_factory=FakeRendererFactory(), **kwargs)

    complete = train("complete")
    train("partial", stop_after=1)
    recovered = train("recovered", resume=tmp_path / "partial/checkpoint.pt")
    assert complete["model_state_sha256"] == recovered["model_state_sha256"]
    _, _, restored_codec = load_model(tmp_path / "complete/model.pt")
    assert restored_codec.codec_id == original_pool.codec.codec_id
    saved = torch.load(tmp_path / "complete/model.pt", weights_only=True)
    assert str(original_pool.corpus_root) not in str(saved)

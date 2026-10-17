from copy import deepcopy
import json

import pytest

from obruxo_data.errors import ValidationError
from obruxo_data.hashing import canonical_json
from obruxo_training.codec import load_codec
from obruxo_training.corpus_inputs import CorpusInputCodec, prepare_corpus_labels
from obruxo_training.pilot import seeded_random
from test_curve_inputs import layout_codec  # noqa: F401
from test_combined import codec, expression  # noqa: F401
from test_factory_assets import bank  # noqa: F401
from test_lfo_bank import population  # noqa: F401


@pytest.fixture(scope="module")
def corpus(layout_codec):  # noqa: F811
    return CorpusInputCodec(layout_codec.base.bank, layout_codec.lfo_bank.to_dict())


def source(corpus):
    target = corpus.parent.sample_target(seeded_random(corpus.generator, 0, 0), 0)
    return json.loads(corpus.parent.source_preset(target, 0))


def test_display_and_disabled_unknown_asset_changes_never_touch_source_or_controls(corpus):
    document = source(corpus)
    settings = document["settings"]
    settings.update(osc_2_on=0., sample_on=0.)
    settings["wavetables"][1]["full_normalize"] = not settings["wavetables"][1]["full_normalize"]
    settings["sample"]["sample_rate"] += 1
    document.update(preset_style="Custom style", macro1="Display label")
    original = deepcopy(document)
    prepared, changes = prepare_corpus_labels(document, corpus.base, corpus.schema)
    assert document == original
    assert {name: settings[name] for name in corpus.schema.scalar_names} == {
        name: prepared["settings"][name] for name in corpus.schema.scalar_names}
    paths = {row["path"] for row in changes}
    assert {"/preset_style", "/macro1", "/settings/wavetables/1", "/settings/sample"} <= paths
    for name in ("lfos", "custom_warps", "modulations", "random_values"):
        assert prepared["settings"][name] == settings[name]
    assert prepare_corpus_labels(prepared, corpus.base, corpus.schema) == (prepared, [])
    target = corpus.encode(canonical_json(document).encode())
    assert target == corpus.parent.encode(canonical_json(prepared).encode())
    assert load_codec(corpus.to_dict()).codec_id == corpus.codec_id


def test_known_disabled_assets_are_retained_and_active_unknown_assets_are_rejected(corpus):
    document = source(corpus)
    document["settings"].update(osc_1_on=0., sample_on=0.)
    prepared, changes = prepare_corpus_labels(document, corpus.base, corpus.schema)
    assert prepared == document and not changes
    document["settings"]["osc_1_on"] = 1.
    wave = document["settings"]["wavetables"][0]
    wave["full_normalize"] = not wave["full_normalize"]
    prepared, changes = prepare_corpus_labels(document, corpus.base, corpus.schema)
    assert prepared["settings"]["wavetables"][0] == document["settings"]["wavetables"][0] and not changes
    with pytest.raises(ValueError, match="unregistered factory asset"):
        corpus.encode(canonical_json(document).encode())


def test_invalid_controls_and_versions_are_not_repaired(corpus):
    document = source(corpus)
    document["settings"]["osc_1_on"] = 2.
    with pytest.raises(ValidationError):
        corpus.encode(canonical_json(document).encode())
    document["settings"]["osc_1_on"] = 1.
    document["synth_version"] = "1.6.4"
    with pytest.raises(ValidationError):
        corpus.encode(canonical_json(document).encode())


def test_original_authored_targets_and_other_codec_instances_are_unchanged(corpus, layout_codec):  # noqa: F811
    legacy_id = layout_codec.codec_id
    parent_id = corpus.parent.codec_id
    for index in range(8):
        target = corpus.parent.sample_target(seeded_random(corpus.generator, 0, index), index)
        raw = corpus.parent.source_preset(target, index)
        assert corpus.encode(raw) == target and corpus.source_preset(target, index) == raw
    assert corpus.parent.codec_id == parent_id and corpus.codec_id != parent_id and layout_codec.codec_id == legacy_id
    assert dict(layout_codec.categorical)["osc_2_destination"] == tuple(range(5))
    assert dict(corpus.categorical)["osc_2_destination"] == tuple(range(15))


def test_modulator_only_destination_is_preserved_and_only_empty_maps_are_omitted(corpus):
    document = source(corpus)
    settings = document["settings"]
    settings.update(osc_2_on=1., osc_2_destination=9., osc_2_level=0., osc_1_distortion_type=7., osc_1_distortion_amount=.3)
    mapping = deepcopy(settings["lfos"][0])
    settings["modulations"][62] = {"source": "lfo_1", "destination": "osc_1_pan", "line_mapping": deepcopy(mapping)}
    settings["modulations"][63] = {"source": "", "destination": "", "line_mapping": deepcopy(mapping)}
    original = deepcopy(document)
    prepared, changes = prepare_corpus_labels(document, corpus.base, corpus.schema)
    assert document == original and prepared["settings"]["modulations"][62] == settings["modulations"][62]
    assert prepared["settings"]["modulations"][63] == {"source": "", "destination": ""}
    assert changes == [{"path": "/settings/modulations/63/line_mapping", "reason": "ignored_empty_connection_mapping"}]
    target = corpus.encode(canonical_json(document).encode())
    decoded = json.loads(corpus.decode(target))["settings"]
    assert decoded["osc_2_on"] == 1. and decoded["osc_2_destination"] == 9. and decoded["osc_2_level"] == 0.


def test_full_snapshot_saves_its_declared_performance_bank_and_rejects_legacy_replacement(corpus, tmp_path):
    from obruxo_data.hashing import canonical_sha256
    from obruxo_training.audit import audit_corpus, load_snapshot, write_json
    from obruxo_training.performances import bank_document
    source_dir, output = tmp_path / "source", tmp_path / "snapshot"
    source_dir.mkdir()
    (source_dir / "One.vital").write_bytes(canonical_json(corpus.schema.template()).encode())
    result = audit_corpus(source_dir, output, schema=corpus.schema, catalog=corpus.catalog)
    assert result["population_complete"] and result["static_eligible_states"] == 1
    contract = json.loads((output / "contract.json").read_text())
    performances = json.loads((output / "performances.json").read_text())
    assert len(performances["entries"]) == 16 and canonical_sha256(performances) == contract["performance_bank_id"]
    load_snapshot(output, corpus.schema)
    write_json(output / "performances.json", bank_document())
    status = json.loads((output / "status.json").read_text())
    status["artifact_ids"]["performances.json"] = canonical_sha256(bank_document())
    write_json(output / "status.json", status)
    with pytest.raises(ValueError, match="performance bank changed"):
        load_snapshot(output, corpus.schema)


def test_full_domain_model_trains_and_exports_with_its_own_codec(corpus, tmp_path):
    from obruxo_training.pilot import PilotPool
    from obruxo_training.trainer import TrainConfig, load_model, train_pilot
    from test_stream import FakeRendererFactory
    pool = PilotPool.create(8, codec=corpus)
    config = TrainConfig(steps=2, replay_capacity=4, batch_size=2, render_budget=4, checkpoint_every=1, cpu_threads=2)
    result = train_pilot(pool, tmp_path / "training", config=config, renderer_factory=FakeRendererFactory())
    _, _, restored = load_model(tmp_path / "training/model.pt")
    assert result["complete"] and result["frozen_weights_and_buffers_unchanged"]
    assert restored.codec_id == corpus.codec_id and dict(restored.categorical)["osc_2_destination"] == tuple(range(15))

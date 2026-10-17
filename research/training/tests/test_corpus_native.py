from copy import deepcopy
import json

import numpy as np
import pytest

from obruxo_data.render.base import FULL_CORPUS_RENDERER_ID, FULL_CURVE_RENDERER_ID, RenderRequest
from obruxo_data.render.corpus_inputs import native_scalar_grid, prepare_native_corpus, verify_native_corpus_controls
from obruxo_data.render.vita import verify_loaded_scalars
from obruxo_data.vital.full_contract import FullPresetContract
from obruxo_training.codec import load_codec
from obruxo_training.corpus_codec import CorpusNativeCodec
from obruxo_training.performances import bank_document_for, capabilities_for
from test_corpus_inputs import corpus  # noqa: F401
from test_curve_inputs import layout_codec  # noqa: F401
from test_combined import codec, expression  # noqa: F401
from test_factory_assets import bank  # noqa: F401
from test_lfo_bank import population  # noqa: F401


def test_exact_native_roundtrip_accepts_known_case_without_weakening_old_profile():
    schema = FullPresetContract(renderer_id=FULL_CORPUS_RENDERER_ID)
    document = schema.template()
    document["settings"]["lfo_1_smooth_time"] = .41464757919311523
    loaded = deepcopy(document)
    loaded["settings"]["lfo_1_smooth_time"] = .41464805603027344
    with pytest.raises(RuntimeError, match="lfo_1_smooth_time"):
        verify_loaded_scalars(document, loaded)
    count, ledger = verify_native_corpus_controls(document, loaded, schema)
    assert count == 775 and ledger == [{"name": "lfo_1_smooth_time", "requested": .41464757919311523,
                                        "loaded": .41464805603027344, "exact_native_grid": .41464805603027344}]
    loaded["settings"]["lfo_1_smooth_time"] = float(np.nextafter(np.float32(.41464805603027344), np.float32(1.)))
    with pytest.raises(RuntimeError, match="lfo_1_smooth_time"):
        verify_native_corpus_controls(document, loaded, schema)


@pytest.mark.parametrize("name,value", [("osc_1_on", 0.1), ("effect_chain_order", 1.), ("osc_1_level", .9), ("volume", None)])
def test_missing_discrete_and_real_control_changes_fail(name, value):
    schema = FullPresetContract(renderer_id=FULL_CORPUS_RENDERER_ID)
    document, loaded = schema.template(), schema.template()
    loaded["settings"][name] = value
    with pytest.raises(RuntimeError, match=name):
        verify_native_corpus_controls(document, loaded, schema)


def test_only_empty_connection_map_is_omitted_and_original_is_preserved():
    document = FullPresetContract().template()
    mapping = deepcopy(document["settings"]["lfos"][0])
    document["settings"]["modulations"][0] = {"source": "lfo_1", "destination": "osc_1_pan", "line_mapping": deepcopy(mapping)}
    document["settings"]["modulations"][63]["line_mapping"] = deepcopy(mapping)
    before = deepcopy(document)
    prepared, changes = prepare_native_corpus(document)
    assert document == before and prepared["settings"]["modulations"][0] == before["settings"]["modulations"][0]
    assert prepared["settings"]["modulations"][63] == {"source": "", "destination": ""}
    assert changes == [{"path": "/settings/modulations/63/line_mapping", "reason": "native_ignores_empty_connection_mapping"}]
    assert {name: value for name, value in prepared["settings"].items() if name != "modulations"} == {
        name: value for name, value in document["settings"].items() if name != "modulations"}
    assert prepare_native_corpus(prepared) == (prepared, [])


def test_active_mapping_readback_still_requires_exact_state():
    schema = FullPresetContract(renderer_id=FULL_CORPUS_RENDERER_ID)
    document = schema.template()
    document["settings"]["modulations"][0] = {"source": "lfo_1", "destination": "osc_1_pan",
                                               "line_mapping": deepcopy(document["settings"]["lfos"][0])}
    loaded = deepcopy(document)
    loaded["settings"]["modulations"][0]["line_mapping"]["points"][1] = .25
    with pytest.raises(RuntimeError, match="modulations"):
        verify_native_corpus_controls(document, loaded, schema)


def test_renderer_configuration_preserves_expression_capabilities(tmp_path, monkeypatch):
    from obruxo_data.render.capabilities import EXPRESSION_CAPABILITIES
    from obruxo_data.render.vita import VitalRenderer
    received = {}

    def initialize(self, plugin_path, **kwargs):
        received.update(kwargs)

    monkeypatch.setattr(VitalRenderer, "__init__", initialize)
    path = tmp_path / "renderer.json"
    path.write_text(json.dumps({"version": 1, "backend": "dawdreamer", "renderer_id": FULL_CORPUS_RENDERER_ID,
                               "max_workers": 1, "capabilities": EXPRESSION_CAPABILITIES.to_dict()}))
    VitalRenderer.from_config(path)
    assert received["renderer_id"] == FULL_CORPUS_RENDERER_ID


def test_grid_matches_existing_encoder_across_control_ranges(corpus):  # noqa: F811
    schema, scalar = corpus.schema, corpus.base.scalar
    for fraction in (0., .0137, .5, .9873, 1.):
        settings = schema.template()["settings"]
        for name, low, high, _ in scalar.continuous:
            settings[name] = low + fraction * (high - low)
        expected = scalar._values(scalar._target(settings))
        for name, *_ in scalar.continuous:
            assert np.float32(native_scalar_grid(name, settings[name], schema)) == np.float32(expected[name]), name


def test_new_codec_and_schema_keep_older_identities_and_all_midi_capabilities(corpus):  # noqa: F811
    original = corpus.codec_id
    current = CorpusNativeCodec(corpus.base.bank, corpus.lfo_bank.to_dict())
    assert corpus.codec_id == original and current.codec_id != original
    assert load_codec(current.to_dict()).codec_id == current.codec_id
    source, target = [FullPresetContract(renderer_id=value).spec for value in (FULL_CURVE_RENDERER_ID, FULL_CORPUS_RENDERER_ID)]
    assert source == {name: value for name, value in target.items() if name != "corpus_native_policy"}
    assert len(bank_document_for(FULL_CORPUS_RENDERER_ID)["entries"]) == 16
    assert capabilities_for(FULL_CORPUS_RENDERER_ID) == capabilities_for(FULL_CURVE_RENDERER_ID)
    from obruxo_data.midi import Performance
    performance = Performance(bpm=120)
    performance.add_note(pitch=60, velocity=90, start_tick=0, duration_ticks=480)
    request = RenderRequest(preset_json=json.dumps(current.schema.template()), performance=performance, renderer_id=FULL_CORPUS_RENDERER_ID)
    assert request.renderer_id == FULL_CORPUS_RENDERER_ID and request.preset_json

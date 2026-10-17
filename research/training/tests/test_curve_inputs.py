from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from obruxo_data.hashing import canonical_json
from obruxo_data.render.base import FULL_CURVE_RENDERER_ID, FULL_EXPRESSION_RENDERER_ID
from obruxo_data.render.curve_inputs import float32, prepare_curve_inputs
from obruxo_data.vital.full_contract import FullPresetContract
from obruxo_training.codec import load_codec
from obruxo_training.curve_codec import CurveInputCodec
from obruxo_training.pilot import PilotPool, PilotSampler
from test_combined import codec, expression  # noqa: F401
from test_factory_assets import bank  # noqa: F401
from test_lfo_bank import population  # noqa: F401


@pytest.mark.parametrize("x,y,power,smooth", [(0., 0., -2., False), (.37, .2, .37, True), (1., 1., 2., False)])
def test_singletons_in_every_slot_become_constant_native_curves_without_changing_source(x, y, power, smooth):
    document = FullPresetContract().template()
    shape = {"name": "One point", "num_points": 1, "points": [x, y], "powers": [power], "smooth": smooth}
    settings = document["settings"]
    settings["lfos"] = [deepcopy(shape) for _ in range(8)]
    settings["custom_warps"] = [deepcopy(shape) for _ in range(3)]
    settings["modulations"] = [{"source": "lfo_1", "destination": "osc_1_pan", "line_mapping": deepcopy(shape)} for _ in range(64)]
    before = canonical_json(document)
    applied, changes = prepare_curve_inputs(document)
    assert canonical_json(document) == before and len(changes) == 75
    curves = applied["settings"]["lfos"] + applied["settings"]["custom_warps"] + [route["line_mapping"] for route in applied["settings"]["modulations"]]
    expected = {**shape, "num_points": 2, "points": [0., float32(y), 1., float32(y)], "powers": [0., 0.]}
    assert all(curve == expected for curve in curves)
    assert all(change["singleton_expanded"] for change in changes)
    assert prepare_curve_inputs(applied) == (applied, [])
    assert {key: value for key, value in settings.items() if key not in {"lfos", "custom_warps", "modulations"}} == {
        key: value for key, value in applied["settings"].items() if key not in {"lfos", "custom_warps", "modulations"}}


def test_multi_point_geometry_is_only_rounded_to_native_float32():
    document = FullPresetContract().template()
    shape = document["settings"]["lfos"][0]
    shape.update(num_points=3, points=[0., .1, .37, .8, 1., .2], powers=[.37, -2., 0.], smooth=True)
    applied, changes = prepare_curve_inputs(document)
    after = applied["settings"]["lfos"][0]
    assert after == {**shape, "points": [float32(value) for value in shape["points"]], "powers": [float32(value) for value in shape["powers"]]}
    assert changes == [{"path": "/settings/lfos/0", "singleton_expanded": False, "before_points": 3, "applied_points": 3}]


@pytest.mark.parametrize("changes", [{"num_points": 0}, {"num_points": 101}, {"points": [.5]}, {"powers": []},
    {"points": [-.1, .5]}, {"points": [True, .5]}, {"points": [float("nan"), .5]}, {"powers": [float("inf")]}, {"powers": [1e39]}])
def test_invalid_curves_fail_closed(changes):
    document = FullPresetContract().template()
    document["settings"]["lfos"][0] = {"name": "One", "num_points": 1, "points": [.37, .2], "powers": [0.], "smooth": False, **changes}
    with pytest.raises(ValueError):
        prepare_curve_inputs(document)


def test_new_profile_is_explicit_and_older_contract_spec_stays_unchanged():
    old = FullPresetContract(renderer_id=FULL_EXPRESSION_RENDERER_ID)
    new = FullPresetContract(renderer_id=FULL_CURVE_RENDERER_ID)
    assert "curve_input_policy" not in old.spec and new.spec["curve_input_policy"]["raw_source_preserved"]
    assert {key: value for key, value in new.spec.items() if key != "curve_input_policy"} == old.spec


def test_renderer_configuration_accepts_the_new_profile_with_expression_capabilities(tmp_path, monkeypatch):
    from obruxo_data.render.capabilities import EXPRESSION_CAPABILITIES
    from obruxo_data.render.vita import VitalRenderer
    received = {}

    def initialize(self, plugin_path, **kwargs):
        received.update(kwargs)

    monkeypatch.setattr(VitalRenderer, "__init__", initialize)
    path = tmp_path / "renderer.json"
    path.write_text(json.dumps({"version": 1, "backend": "dawdreamer", "renderer_id": FULL_CURVE_RENDERER_ID,
                               "max_workers": 1, "capabilities": EXPRESSION_CAPABILITIES.to_dict()}))
    VitalRenderer.from_config(path)
    assert received["renderer_id"] == FULL_CURVE_RENDERER_ID


@pytest.fixture(scope="module")
def layout_codec(codec):  # noqa: F811
    return CurveInputCodec(codec.base.bank, codec.lfo_bank.to_dict())


@pytest.fixture(scope="module")
def layout_pool(layout_codec):
    return PilotPool.create(32, codec=layout_codec)


def test_codec_and_pool_roundtrip_preserve_original_singleton_sources_and_old_bank_identity(layout_pool, tmp_path):
    source_codec = layout_pool.codec
    assert load_codec(source_codec.to_dict()).codec_id == source_codec.codec_id
    assert load_codec(source_codec.parent.to_dict()).codec_id == source_codec.parent.codec_id
    assert source_codec.lfo_bank.bank_id == source_codec.parent.lfo_bank.bank_id
    assert source_codec.contract.renderer_id == FULL_CURVE_RENDERER_ID
    assert source_codec.contract.contract_id != source_codec.parent.contract.contract_id
    seen = set()
    for index, entry in enumerate(layout_pool.entries):
        raw = source_codec.source_preset(entry.target, index)
        original = json.loads(raw)
        applied, changes = prepare_curve_inputs(original)
        assert changes and all(change["singleton_expanded"] for change in changes)
        seen.update(change["path"].split("/")[2] for change in changes)
        assert source_codec.encode(raw) == entry.target
        assert source_codec.parent.encode(canonical_json(applied).encode()) == entry.target
        active = {"lfos": source_codec.active_lfos(entry.target), "custom_warps": source_codec.active_warps(entry.target),
                  "modulations": source_codec.active_route_maps(entry.target)}
        assert all(active[change["path"].split("/")[2]][int(change["path"].split("/")[3])] for change in changes if change["singleton_expanded"])
        for family, method in (("lfos", "source_curves"), ("custom_warps", "source_warp_curves")):
            values = getattr(source_codec, method)(raw)
            for slot, shape in enumerate(original["settings"][family]):
                if shape["num_points"] == 1:
                    np.testing.assert_array_equal(values[slot], np.full(97, float32(shape["points"][1]), dtype=np.float32))
    assert seen == {"lfos", "custom_warps", "modulations"}
    layout_pool.save(tmp_path / "pool.json")
    assert PilotPool.load(tmp_path / "pool.json").pool_id == layout_pool.pool_id
    recipe = PilotSampler(layout_pool, policy="attack_compatible").recipe_at(0)
    assert recipe.render_request().renderer_id == FULL_CURVE_RENDERER_ID and recipe.performance_role == "reference"


def test_singleton_audio_labels_train_and_resume_with_identical_model(layout_codec, tmp_path):
    from obruxo_training.stream import StreamConfig
    from obruxo_training.trainer import TrainConfig, load_model, train_pilot
    from test_stream import FakeRendererFactory
    pool = PilotPool.create(12, codec=layout_codec)
    config = TrainConfig(steps=3, replay_capacity=4, batch_size=2, render_budget=4, checkpoint_every=1, cpu_threads=2)

    def run(name, **kwargs):
        return train_pilot(pool, tmp_path / name, config=config, stream_config=StreamConfig(prefetch_clips=1), renderer_factory=FakeRendererFactory(), **kwargs)

    full = run("full")
    run("partial", stop_after=1)
    restored = run("restored", resume=tmp_path / "partial/checkpoint.pt")
    assert full["model_state_sha256"] == restored["model_state_sha256"]
    _, _, exported_codec = load_model(tmp_path / "full/model.pt")
    assert exported_codec.codec_id == layout_codec.codec_id
    saved = torch.load(tmp_path / "full/checkpoint.pt", weights_only=True)
    assert all(set(item["features"]) == {"timbre", "pitch"} for item in saved["replay"])

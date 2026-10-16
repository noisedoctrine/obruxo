import json

import pytest

from obruxo_training.capability_audit import _record_shape, _shape_report, _shape_stats, audit_capabilities
from obruxo_training.pilot import PilotPool
from test_expression import codec, pool  # noqa: F401
from test_factory_assets import bank  # noqa: F401
from test_lfo_bank import population  # noqa: F401


def test_inventory_separates_masking_constant_labels_and_variation_without_a_quality_claim(pool):  # noqa: F811
    report = audit_capabilities(pool)
    assert report["scalar_count"] == 775 and sum(report["scalar_status_counts"].values()) == 775
    assert {row["name"] for row in report["scalars"]} == pool.codec.schema.scalar_names
    assert report["audited_train_states"] == report["states_by_split"]["train"]
    rows = {row["name"]: row for row in report["scalars"]}
    assert rows["sample_on"]["status"] == "constant" and rows["sample_level"]["status"] == "masked"
    assert rows["osc_1_spectral_morph_type"]["status"] == "varied"
    assert report["structured"]["wavetables_and_sample"][3]["supervised_states"] == 0
    assert report["limits"]["render_attempts"] == report["limits"]["optimizer_steps"] == 0
    assert not report["limits"]["model_promotion"] and not report["limits"]["final_test_audio"]
    assert len(report["performances"]["reference"]) == len(report["performances"]["transfer"]) == 8
    assert not any(key in json.dumps(report) for key in ('"wave_data"', '"samples_stereo"', '"preset_json"', '"asset_bank"'))


def test_audit_reads_original_shapes_only_for_frozen_training_states(pool, monkeypatch):  # noqa: F811
    original, calls = pool.codec.source_preset, []

    def checked_source(target, index):
        assert pool.manifest.by_state[pool.entries[index].state_id].split == "train"
        calls.append(index)
        return original(target, index)

    monkeypatch.setattr(pool.codec, "source_preset", checked_source)
    result = audit_capabilities(pool)
    assert len(calls) == result["audited_train_states"] and len(set(calls)) == len(calls)
    assert all(slot["supervised_point_counts"] for slot in result["structured"]["lfos"] if slot["supervised_states"])


def test_narrow_codec_is_refused_instead_of_claiming_full_inventory():
    with pytest.raises(ValueError, match="complete nested-state codec"):
        audit_capabilities(PilotPool.create(4))


def test_renaming_a_curve_does_not_inflate_shape_diversity():
    stats = _shape_stats()
    shape = {"name": "first", "num_points": 2, "points": [0., 1., 1., 0.], "powers": [0., 0.]}
    _record_shape(stats, shape, True)
    _record_shape(stats, {**shape, "name": "second"}, True)
    _record_shape(stats, {**shape, "points": [0., 1., 1., 1.]}, False)
    assert _shape_report(stats) == {"supervised_states": 2, "distinct_supervised_shapes": 1, "supervised_point_counts": [2]}

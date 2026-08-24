from __future__ import annotations

from copy import deepcopy
import json

from research.vital.build_vital_usage_census import (
    analyze_document,
    build_census,
    line_mapping_is_linear,
    semantic_nested_state,
    write_categorical_values_csv,
    write_continuous_bins_csv,
)
from research.data_generation.obruxo_data.vital.atlas import VitalSchema


def make_document(version: str = "1.0.8") -> tuple[VitalSchema, dict]:
    schema = VitalSchema.load()
    document = deepcopy(schema.load_init_document())
    document["synth_version"] = version
    return schema, document


def test_semantics_keep_slot_identity_and_route_states_separate() -> None:
    schema, document = make_document()
    settings = document["settings"]
    settings["osc_1_on"] = 0.0
    settings["osc_2_on"] = 1.0
    settings["osc_3_on"] = 0.0
    settings["osc_1_level"] = 0.25
    settings["modulations"][0] = {"source": "lfo_1", "destination": "osc_1_level"}
    settings["modulation_1_amount"] = 0.0
    settings["modulations"][1] = {"source": "lfo_2", "destination": "osc_2_level"}
    settings["modulation_2_amount"] = 0.5
    settings["modulation_2_bypass"] = 1.0
    settings["modulations"][2] = {"source": "lfo_3", "destination": "osc_2_level"}
    settings["modulation_3_amount"] = 0.5
    settings["lfos"][2]["points"] = [0.0, 0.0, 0.5, 1.0, 1.0, 0.0]

    observation = analyze_document(document, schema)

    assert observation.active_slots["oscillator"] == ("2",)
    assert observation.active_slots["lfo"] == ("3",)
    assert observation.active_custom_lfo_slots == ("3",)
    routes = {route.slot: route for route in observation.routes if route.connected}
    assert not routes[1].amount_nonzero and not routes[1].live
    assert routes[2].bypassed and not routes[2].live
    assert routes[3].live


def test_census_conditions_defaults_and_modulation_on_a_default_value(tmp_path) -> None:
    schema, document = make_document()
    settings = document["settings"]
    settings["osc_1_on"] = 0.0
    settings["osc_1_level"] = 0.25
    settings["modulations"][0] = {"source": "lfo_1", "destination": "osc_1_level"}
    settings["modulation_1_amount"] = 1.0
    settings["modulation_1_bypass"] = 0.0
    first = tmp_path / "first.vital"
    first.write_text(json.dumps(document), encoding="utf-8")

    _, default_document = make_document()
    default_document["settings"]["modulations"][0] = {"source": "lfo_1", "destination": "osc_1_level"}
    default_document["settings"]["modulation_1_amount"] = 1.0
    second = tmp_path / "second.vital"
    second.write_text(json.dumps(default_document), encoding="utf-8")

    census = build_census(tmp_path, schema=schema, progress_every=0)
    parameter = census["file_weighted"]["parameters"]["osc_1_level"]
    assert parameter["eligible"] == 2
    assert parameter["non_default"] == 1
    assert parameter["active_eligible"] == int(default_document["settings"]["osc_1_on"] != 0)
    assert parameter["modulated"] == 2


def test_version_introduced_parameters_use_restricted_denominators_and_unknown_defaults(tmp_path) -> None:
    schema, old_document = make_document("1.0.8")
    old_path = tmp_path / "old.vital"
    old_path.write_text(json.dumps(old_document), encoding="utf-8")
    _, spectral_document = make_document("1.5.5")
    spectral_document["settings"]["osc_1_spectral_morph_phase"] = 0.5
    spectral_path = tmp_path / "spectral.vital"
    spectral_path.write_text(json.dumps(spectral_document), encoding="utf-8")
    _, ramp_document = make_document("1.6.4")
    ramp_document["settings"]["modulation_1_ramp_up"] = 0.5
    ramp_path = tmp_path / "ramp.vital"
    ramp_path.write_text(json.dumps(ramp_document), encoding="utf-8")

    census = build_census(tmp_path, schema=schema, progress_every=0)
    spectral = census["file_weighted"]["parameters"]["osc_1_spectral_morph_phase"]
    ramp = census["file_weighted"]["parameters"]["modulation_1_ramp_up"]
    assert spectral["eligible"] == 1
    assert spectral["nonzero"] == 1
    assert spectral["non_default"] is None
    assert ramp["eligible"] == 1
    assert ramp["nonzero"] == 1
    assert ramp["non_default"] is None
    assert spectral["parameter_type"] == "continuous"
    assert spectral["distribution"]["domain"] == "raw"
    assert spectral["observed_value_summary"]["count"] == 1


def test_parameter_distributions_cover_enum_frequencies_and_continuous_modes(tmp_path) -> None:
    schema, first_document = make_document()
    first_document["settings"]["chorus_on"] = 0.0
    first_document["settings"]["osc_1_level"] = schema.parameters["osc_1_level"].default
    (tmp_path / "first.vital").write_text(json.dumps(first_document), encoding="utf-8")

    _, second_document = make_document()
    second_document["settings"]["chorus_on"] = 1.0
    second_document["settings"]["osc_1_level"] = 0.25
    (tmp_path / "second.vital").write_text(json.dumps(second_document), encoding="utf-8")

    census = build_census(tmp_path, schema=schema, progress_every=0)
    categorical = census["file_weighted"]["parameters"]["chorus_on"]
    frequencies = {row["value"]: row for row in categorical["value_frequencies"]}
    assert categorical["parameter_type"] == "categorical"
    assert categorical["options"] == ["Off", "On"]
    assert frequencies[0.0]["count"] == 1
    assert frequencies[0.0]["label"] == "Off"
    assert frequencies[1.0]["count"] == 1
    assert frequencies[1.0]["label"] == "On"
    assert sum(row["frequency"] for row in categorical["value_frequencies"]) == 1.0

    continuous = census["file_weighted"]["parameters"]["osc_1_level"]
    assert continuous["parameter_type"] == "continuous"
    assert continuous["distribution"]["domain"] == "normalized"
    assert continuous["observed_value_summary"]["count"] == 2
    assert continuous["default_count"] == 1
    assert sum(continuous["distribution"]["counts"]) == 2
    assert set(continuous["distribution"]["quantiles"]) == {"p01", "p05", "p25", "p50", "p75", "p95", "p99"}
    assert continuous["distribution"]["dominant_bins"]


def test_distribution_exports_are_row_oriented(tmp_path) -> None:
    schema, document = make_document()
    document["settings"]["chorus_on"] = 1.0
    document["settings"]["osc_1_level"] = 0.25
    (tmp_path / "one.vital").write_text(json.dumps(document), encoding="utf-8")
    census = build_census(tmp_path, schema=schema, progress_every=0)
    categorical_path = tmp_path / "categorical.csv"
    continuous_path = tmp_path / "continuous.csv"
    write_categorical_values_csv(categorical_path, census)
    write_continuous_bins_csv(continuous_path, census)

    categorical_text = categorical_path.read_text(encoding="utf-8")
    assert "chorus_on,Chorus Switch,Indexed,0.0,1.0,1.0,On" in categorical_text
    continuous_rows = continuous_path.read_text(encoding="utf-8").splitlines()
    assert sum(row.startswith("osc_1_level,") for row in continuous_rows) == 64


def test_duplicate_and_payload_policies(tmp_path) -> None:
    schema, document = make_document()
    document["settings"]["sample"]["samples"] = "private-looking-payload"
    payload_descriptor = semantic_nested_state(document["settings"]["sample"])
    assert "private-looking-payload" not in json.dumps(payload_descriptor)
    numeric_payload_descriptor = semantic_nested_state({"samples": [1.25, 2.5, 3.75]})
    assert "1.25" not in json.dumps(numeric_payload_descriptor)
    content = json.dumps(document)
    (tmp_path / "one.vital").write_text(content, encoding="utf-8")
    (tmp_path / "duplicate.vital").write_text(content, encoding="utf-8")
    census = build_census(tmp_path, schema=schema, progress_every=0)
    assert census["file_weighted"]["parsed_files"] == 2
    assert census["exact_deduplicated"]["parsed_files"] == 1
    assert census["exact_duplicate_files"] == 1
    assert census["exact_unique_content_groups"] == 1


def test_line_mapping_classification() -> None:
    linear = {"num_points": 2, "points": [0.0, 0.0, 1.0, 1.0], "powers": [1.0, 1.0]}
    curved = {"num_points": 2, "points": [0.0, 0.0, 1.0, 1.0], "powers": [2.0, 2.0]}
    assert line_mapping_is_linear(linear)
    assert not line_mapping_is_linear(curved)


def test_wavetable_display_metadata_and_numeric_types_are_not_custom_state() -> None:
    schema, document = make_document()
    stock_observation = analyze_document(document, schema)
    assert stock_observation.active_wavetable_states["named_stock_or_unresolved_content"] == 1
    document["settings"]["wavetables"][0]["name"] = "renamed display label"
    observation = analyze_document(document, schema)
    assert observation.active_wavetable_custom_slots == ()
    assert observation.active_wavetable_states["named_nonstock_or_unresolved_content"] == 1

from dataclasses import replace
import hashlib
import json

import pytest

from obruxo_training.assets import AssetCatalog, asset_digest
from obruxo_training.eligibility import inspect_preset
from obruxo_training.identity import PresetIdentity, parse_document


def test_identity_ignores_only_declared_display_fields_and_object_order():
    first = b'{"settings":{"routes":[1,2],"author":"nested"},"preset_name":"a"}'
    second = b'{ "preset_name":"b", "settings": {"author":"nested", "routes":[1,2]} }'
    left, right = PresetIdentity.from_bytes(first), PresetIdentity.from_bytes(second)
    # Golden values from the existing corpus identity audit; do not silently fork its rule.
    assert left.to_dict() == {
        "raw": "11c447c7eb6ad26bee2cdcce85f1bb9fa1b2afbb38b69a3e3f427e9120c6fced",
        "canonical": "4fb751e7e3bcc0cc036bdff932143f7c387e3b9a8c30d740f90821bf91b93782",
        "state": "89e01f17ac0c89b67506a63ad340139180154c4ef9e5ace97254d64e4209be30",
    }
    assert left.raw != right.raw and left.canonical != right.canonical and left.state == right.state
    assert left.state != PresetIdentity.from_bytes(first.replace(b"[1,2]", b"[2,1]")).state
    assert left.state != PresetIdentity.from_bytes(first.replace(b"nested", b"changed")).state
    assert left.canonical == PresetIdentity.from_bytes(b"\xef\xbb\xbf" + first + b"\n").canonical


def test_identity_preserves_decimal_precision_and_numeric_types():
    values = [b'{"x":1}', b'{"x":1.0}', b'{"x":1.00}', b'{"x":1.00000000000000000001}']
    assert len({PresetIdentity.from_bytes(value).state for value in values}) == len(values)


@pytest.mark.parametrize("raw", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}', b'[]', b'\xff'])
def test_invalid_identity_never_silently_canonicalizes(raw):
    with pytest.raises(ValueError):
        PresetIdentity.from_bytes(raw)


def test_named_asset_edit_does_not_pass_but_rename_does(init_raw, schema, catalog):
    original = parse_document(init_raw)["settings"]["wavetables"][0]
    renamed = dict(original, name="not the original name", author="someone else")
    assert asset_digest("wavetable", original) == asset_digest("wavetable", renamed)
    assert catalog.identify("wavetable", renamed) is not None
    document = json.loads(init_raw)
    document["settings"]["wavetables"][0]["full_normalize"] = 1 - document["settings"]["wavetables"][0]["full_normalize"]
    result = inspect_preset(json.dumps(document).encode(), schema, catalog)
    assert "unresolved_active_asset" in result["reasons"]
    assert result["assets"][0]["status"] == "unresolved"


def test_unresolved_inactive_assets_are_visible_but_not_called_custom(init_raw, schema, catalog):
    document = json.loads(init_raw)
    document["settings"]["wavetables"][1]["full_normalize"] = 123
    result = inspect_preset(json.dumps(document).encode(), schema, catalog)
    assert result["static_eligible"]
    assert result["assets"][1]["status"] == "unresolved"
    assert not result["assets"][1]["active"]


def test_non_object_wavetable_cannot_bypass_asset_gate(init_raw, schema, catalog):
    document = json.loads(init_raw)
    document["settings"]["wavetables"][0] = None
    result = inspect_preset(json.dumps(document).encode(), schema, catalog)
    assert not result["static_eligible"] and "asset_structure_invalid" in result["reasons"]


def test_modern_state_is_preserved_and_not_claimed_supported(init_raw, schema, catalog):
    document = json.loads(init_raw)
    document["synth_version"] = "1.6.4"
    document["settings"]["modulation_1_ramp_up"] = 0.1
    raw = json.dumps(document).encode()
    result = inspect_preset(raw, schema, catalog)
    assert not result["static_eligible"]
    assert result["unsupported_fields"] == ["modulation_1_ramp_up"]
    assert "unreviewed_synth_version" in result["reasons"]
    assert json.loads(raw) == document


def test_static_eligibility_does_not_claim_audio_qa(init_raw, schema, catalog):
    result = inspect_preset(init_raw, schema, catalog)
    assert result["static_eligible"] and result["audio_qa"] == "not_run"
    assert result["assets"][0]["provenance"] == "reviewed_init"


def test_reference_manifest_verifies_content_and_source_provenance(tmp_path, init_raw):
    source = tmp_path / "reference.vital"
    source.write_bytes(init_raw)
    entry = {"asset_id": "declared-reference", "kind": "wavetable", "slot": 1, "path": source.name,
             "source_sha256": hashlib.sha256(init_raw).hexdigest(), "provenance": "factory_export"}
    manifest = tmp_path / "references.json"
    manifest.write_text(json.dumps({"version": 1, "references": [entry]}))
    catalog = AssetCatalog.from_reference_manifest(manifest)
    assert catalog.references[0].provenance == "factory_export"
    source.write_bytes(init_raw + b"\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        AssetCatalog.from_reference_manifest(manifest)


def test_catalog_rejects_ambiguous_reference_mapping(catalog):
    reference = catalog.references[0]
    with pytest.raises(ValueError, match="ambiguous"):
        AssetCatalog((reference, replace(reference, asset_id="another-name")))

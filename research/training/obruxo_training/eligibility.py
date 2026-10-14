from __future__ import annotations

from collections import Counter
import json

from obruxo_data.vital import VitalSchema
from obruxo_data.vital.validation import validate_document

from .assets import AssetCatalog
from .identity import PresetIdentity, parse_document


REVIEWED_VERSIONS = frozenset({f"1.0.{index}" for index in range(9)} | {"99999.9.9"})


def eligibility_policy() -> dict:
    return {"rule": "static_eligibility_v1", "allowed_synth_versions": sorted(REVIEWED_VERSIONS),
            "unknown_or_missing_controls": "reject", "unresolved_active_assets": "reject",
            "unresolved_inactive_assets": "record", "native_load_and_audio_qa": "required_after_static_gate"}


def inspect_preset(raw: bytes, schema: VitalSchema, catalog: AssetCatalog) -> dict:
    """Static eligibility only: no native plugin, audio QA, or destructive migration."""
    identities = PresetIdentity.from_bytes(raw)
    document = parse_document(raw)
    runtime_document = json.loads(raw.decode("utf-8-sig"))
    version = document.get("synth_version")
    reasons = []
    if not isinstance(version, str) or version not in REVIEWED_VERSIONS:
        reasons.append("unreviewed_synth_version")
    report = validate_document(runtime_document, schema)
    if not report.valid:
        reasons.append("schema_invalid")
    settings = document.get("settings", {})
    assets = []
    if isinstance(settings, dict):
        wavetables = settings.get("wavetables", [])
        if isinstance(wavetables, list):
            if any(not isinstance(asset, dict) for asset in wavetables):
                reasons.append("asset_structure_invalid")
            for slot, asset in enumerate(wavetables[:3], 1):
                if isinstance(asset, dict):
                    assets.append(_inspect_asset(catalog, "wavetable", slot, asset, settings.get(f"osc_{slot}_on") == 1))
        if isinstance(settings.get("sample"), dict):
            assets.append(_inspect_asset(catalog, "sample", 1, settings["sample"], settings.get("sample_on") == 1))
    if any(asset["active"] and asset["status"] == "unresolved" for asset in assets):
        reasons.append("unresolved_active_asset")
    return {
        "identity": identities.to_dict(), "synth_version": version if isinstance(version, str) else "invalid_or_missing",
        "static_eligible": not reasons, "reasons": reasons, "audio_qa": "not_run", "assets": assets,
        "schema_diagnostic_counts": dict(sorted(Counter(item.code for item in report.diagnostics).items())),
        "unsupported_fields": sorted({item.parameter for item in report.diagnostics if item.code == "vital.parameter.unknown"}),
        "missing_fields": sorted({item.parameter for item in report.diagnostics if item.code == "vital.parameter.missing"}),
    }


def _inspect_asset(catalog: AssetCatalog, kind: str, slot: int, asset: dict, active: bool) -> dict:
    reference = catalog.identify(kind, asset)
    return {"kind": kind, "slot": slot, "active": active, "status": "matched_reference" if reference else "unresolved",
            "asset_id": reference.asset_id if reference else None, "provenance": reference.provenance if reference else None}

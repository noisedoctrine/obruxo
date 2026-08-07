from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


VITA_REVISION = "342bc90aca7ab2b6e7a487f8e54a0158a5ccab76"
VITAL_REVISION = "636ca0ef517a4db087a6a08a6a8a5e704e21f836"
MIGRATION_ONLY_PARAMETERS = frozenset({
    "compressor_low_band_unused",
    "filter_1_osc1_input", "filter_1_osc2_input", "filter_1_osc3_input", "filter_1_sample_input",
    "filter_2_osc1_input", "filter_2_osc2_input", "filter_2_osc3_input", "filter_2_sample_input",
    "filter_fx_filter_input", "filter_fx_osc1_input", "filter_fx_osc2_input", "filter_fx_osc3_input", "filter_fx_sample_input",
    "sub_direct_out", "sub_level", "sub_on", "sub_pan", "sub_transpose", "sub_transpose_quantize", "sub_tune", "sub_waveform",
})


def _atomic_write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _runtime_inventory(synth: Any) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for name in sorted(synth.get_controls()):
        details = synth.get_control_details(name)
        inventory[name] = {
            "min": float(details.min),
            "max": float(details.max),
            "default": float(details.default_value),
            "post_offset": float(details.post_offset),
            "display_multiply": float(details.display_multiply),
            "scale": details.scale.name,
            "display_units": details.display_units,
            "display_name": details.display_name,
            "is_discrete": bool(details.is_discrete),
            "options": list(details.options),
        }
    return inventory


def _parse_source_number(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    text = value.rstrip("fF")
    try:
        parsed = float(text)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _reconcile(inventory: dict[str, dict[str, Any]], source_atlas_path: Path) -> dict[str, Any]:
    source = json.loads(source_atlas_path.read_text(encoding="utf-8-sig"))
    source_parameters = {item["name"]: item for item in source["parameters"]}
    overrides = {item["name"]: item["default"] for item in source.get("default_overrides", [])}
    runtime_names = set(inventory)
    source_names = set(source_parameters)
    source_only = sorted(source_names - runtime_names)
    runtime_only = sorted(runtime_names - source_names)
    default_mismatches = []
    unparsed_defaults = []
    for name in sorted(runtime_names & source_names):
        raw_source = overrides.get(name, source_parameters[name].get("default"))
        expected = _parse_source_number(raw_source)
        if expected is None:
            unparsed_defaults.append({"name": name, "source_default": raw_source})
        elif not math.isclose(expected, inventory[name]["default"], rel_tol=1e-6, abs_tol=1e-6):
            default_mismatches.append({"name": name, "source": expected, "runtime": inventory[name]["default"]})
    unexplained_source_only = sorted(set(source_only) - MIGRATION_ONLY_PARAMETERS)
    unexplained = bool(runtime_only or unexplained_source_only or default_mismatches)
    return {
        "source": source.get("source", {}),
        "source_registered_count": len(source_names),
        "runtime_control_count": len(runtime_names),
        "source_only": source_only,
        "classified_migration_only": sorted(set(source_only) & MIGRATION_ONLY_PARAMETERS),
        "unexplained_source_only": unexplained_source_only,
        "runtime_only": runtime_only,
        "default_mismatches": default_mismatches,
        "unparsed_defaults": unparsed_defaults,
        "unexplained_drift": unexplained,
    }


def probe_schema(output: Path, *, source_atlas_path: Path, force: bool = False) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()) and not force:
        raise FileExistsError(f"refusing to overwrite reviewed schema bundle: {output}")
    try:
        import vita
    except ImportError as error:
        raise RuntimeError("schema probing requires the pinned Vita runtime") from error

    synth = vita.Synth()
    synth.load_init_preset()
    init_text = synth.to_json()
    init_document = json.loads(init_text)
    inventory = _runtime_inventory(synth)
    vocabulary = {
        "artifact_schema": "obruxo_vital_modulation_vocab_v1",
        "sources": sorted(vita.get_modulation_sources()),
        "destinations": sorted(vita.get_modulation_destinations()),
    }
    reconciliation = _reconcile(inventory, source_atlas_path)
    if reconciliation["unexplained_drift"]:
        raise RuntimeError(f"unexplained Vita/source schema drift: {json.dumps(reconciliation, sort_keys=True)}")

    init_sha256 = hashlib.sha256(init_text.encode("utf-8")).hexdigest()
    schema_id = f"vital-1.0.8-vita-0.1.0-{init_sha256[:12]}"
    manifest = {
        "schema_id": schema_id,
        "vita_revision": VITA_REVISION,
        "vital_revision": VITAL_REVISION,
        "init_preset": "init.vital",
        "parameter_atlas": "parameter_inventory.json",
        "modulation_vocab": "modulation_vocab.json",
        "reconciliation_report": "reconciliation.json",
        "init_preset_sha256": init_sha256,
        "engine_identity": {
            "vita_version": "0.1.0",
            "vita_revision": VITA_REVISION,
            "vital_source_revision": VITAL_REVISION,
            "runtime_synth_version": init_document.get("synth_version"),
            "sampler_policy": "fixed deterministic load_init_preset payload committed in init.vital",
        },
    }
    atlas = {
        "artifact_schema": "obruxo_vital_runtime_inventory_v1",
        "schema_id": schema_id,
        "parameter_count": len(inventory),
        "parameters": inventory,
    }
    output.mkdir(parents=True, exist_ok=True)
    _atomic_write(output / "init.vital", init_text)
    _atomic_write(output / "parameter_inventory.json", json.dumps(atlas, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    _atomic_write(output / "modulation_vocab.json", json.dumps(vocabulary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    _atomic_write(output / "reconciliation.json", json.dumps(reconciliation, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    _atomic_write(output / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return {"schema_id": schema_id, "output": str(output), "reconciliation": reconciliation}

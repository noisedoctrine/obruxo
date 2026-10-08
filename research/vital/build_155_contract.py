"""Build a reviewable structural contract from 1.5.5 presets; never copy their payloads."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from full_contract import json_digest

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "research/data_generation/obruxo_data/vital/schema/vital-1.0.8-vita-0.1.0"
DEST = Path(__file__).with_name("contract_1_5_5")
NESTED = ("custom_warps", "lfos", "modulations", "random_values", "sample", "wavetables")


def kind(value):
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    return {dict: "object", list: "array", str: "string", type(None): "null"}[type(value)]


def observe(node, value):
    tag = kind(value)
    # Wavetable components have different keyframe contracts for each component type.
    if tag == "object" and "type" in value:
        observe(node.setdefault("variants", {}).setdefault(value["type"], {}), {k: v for k, v in value.items() if k != "type"})
        return
    node.setdefault("types", set()).add(tag)
    if tag == "object":
        keys = set(value)
        node["required"] = node.get("required", keys) & keys
        for key, child in value.items():
            observe(node.setdefault("properties", {}).setdefault(key, {}), child)
    elif tag == "array":
        for child in value:
            observe(node.setdefault("items", {}), child)


def serializable(node):
    if isinstance(node, set):
        return sorted(node)
    if isinstance(node, dict):
        return {k: serializable(v) for k, v in sorted(node.items())}
    return node


def build(corpus, native_init, renderer_ranges):
    shape, counts, scalars, versions = {}, Counter(), None, set()
    identities = []
    for index, path in enumerate(sorted(corpus.rglob("*.vital"))):
        if index % 1000 == 0:
            print(f"Examined {index} files", flush=True)
        raw = path.read_bytes()
        try:
            document = json.loads(raw)
        except (ValueError, UnicodeError):
            counts["malformed"] += 1
            continue
        counts[str(document.get("synth_version"))] += 1
        if document.get("synth_version") != "1.5.5":
            continue
        identities.append(hashlib.sha256(raw).hexdigest())
        settings = document["settings"]
        names = {k for k, v in settings.items() if kind(v) == "number"}
        if scalars is not None and names != scalars:
            raise ValueError("1.5.5 scalar key inventory is inconsistent")
        scalars = names
        observe(shape, {**document, "settings": {k: settings[k] for k in NESTED}})
        versions.update(w["version"] for w in settings["wavetables"])
    if counts["1.5.5"] == 0:
        raise ValueError("no 1.5.5 documents found")
    inventory_raw = (BASE / "parameter_inventory.json").read_bytes()
    inventory = json.loads(inventory_raw)["parameters"]
    phase_names = {f"osc_{n}_spectral_morph_phase" for n in range(1, 4)}
    if scalars != inventory.keys() | phase_names:
        raise ValueError("scalar inventory differs from reviewed 775-control boundary")
    native = json.loads(native_init.read_bytes())
    if native["synth_version"] != "1.6.4":
        raise ValueError("expected the reviewed native 1.6.4 initial state")
    ramps = {f"modulation_{n}_ramp_{direction}": native["settings"][f"modulation_{n}_ramp_{direction}"]
             for n in range(1, 65) for direction in ("up", "down")}
    if set(ramps.values()) != {-10.0}:
        raise ValueError("native ramp defaults changed")
    range_bytes = renderer_ranges.read_bytes()
    measured = json.loads(range_bytes)
    if measured["plugin_version"] != "1.6.4" or set(measured["parameters"]) != scalars:
        raise ValueError("renderer ranges must cover all 775 pinned scalar names")
    component_schema = shape["properties"]["settings"]["properties"]["wavetables"]["items"]["properties"]["groups"]["items"]["properties"]["components"]["items"]
    # The pinned source factory also registers this WaveSource subclass, absent
    # from this corpus cohort. It inherits the same serialized fields.
    component_schema["variants"]["Shepard Tone Source"] = deepcopy(component_schema["variants"]["Wave Source"])
    spec = {"id": "vital-1.5.5-contract-v1", "version": "1.5.5", "scalar_names": sorted(scalars),
            "baseline_inventory_sha256": json_digest(inventory_raw),
            "modulation_vocab_sha256": json_digest((BASE / "modulation_vocab.json").read_bytes()),
            "renderer_ranges_sha256": json_digest(range_bytes),
            "source_supplements": {"revision": "636ca0ef517a4db087a6a08a6a8a5e704e21f836", "component": "Shepard Tone Source inherits Wave Source serialization"},
            "shape": serializable(shape), "wavetable_versions_observed": sorted(versions),
            "fixed_native_ramps": ramps, "max_file_bytes": 64 * 1024 * 1024,
            "phase_authoring_range": [0.0, 1.0],
            "evidence": {"versions": dict(sorted(counts.items())), "target_files": counts["1.5.5"],
                         "corpus_content_digest": hashlib.sha256("\n".join(sorted(identities)).encode()).hexdigest(),
                         "native_init_sha256": hashlib.sha256(native_init.read_bytes()).hexdigest(),
                         "range_scope": "775 raw endpoints measured in Vital 1.6.4 for the 1.5.5 key inventory; not historical 1.5.5 binary metadata"}}
    # Factory assets are from the pinned baseline, never taken from a corpus preset.
    template = json.loads((BASE / "init.vital").read_bytes())
    template["synth_version"] = "1.5.5"
    for wavetable in template["settings"]["wavetables"]:
        wavetable["version"] = "1.5.5"
    for key in (*sorted(phase_names), "custom_warps", "random_values"):
        template["settings"][key] = native["settings"][key]
    DEST.mkdir(exist_ok=True)
    template_bytes = (json.dumps(template, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    spec["template_sha256"] = hashlib.sha256(template_bytes).hexdigest()
    (DEST / "init.vital").write_bytes(template_bytes)
    (DEST / "renderer_ranges.json").write_bytes(range_bytes)
    (DEST / "contract.json").write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(spec["evidence"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--native-init", required=True, type=Path)
    parser.add_argument("--renderer-ranges", required=True, type=Path)
    args = parser.parse_args()
    build(args.corpus, args.native_init, args.renderer_ranges)

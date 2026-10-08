"""Recover raw authoring bounds from the reviewed 1.6.4 plugin for 1.5.5 keys.

These are renderer bounds, not a claim to recover the historical 1.5.5 binary's
metadata. Matching uses unique display names and verifies raw saved state.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import tempfile

from full_contract import FullPresetContract
from obruxo_data.render import VitalRenderer
from obruxo_data.render.vita import VitalVst3StateTemplate


def probe():
    import dawdreamer as daw

    contract, renderer = FullPresetContract(), VitalRenderer()
    engine = daw.RenderEngine(22050, 128)
    plugin = engine.make_plugin_processor("vital", str(renderer.plugin_path))
    descriptions = plugin.get_parameters_description()
    by_name = defaultdict(list)
    for item in descriptions:
        by_name[item["name"]].append(item)
    matched = {}
    for name in sorted(contract.scalar_names):
        display = contract.baseline_parameters[name]["display_name"] if name in contract.baseline_parameters else f"Oscillator {name.split('_')[1]} Frequency Morph Phase"
        candidates = by_name.get(display, [])
        if len(candidates) == 1:
            matched[name] = candidates[0]
    with tempfile.TemporaryDirectory(prefix="obruxo-ranges-") as directory:
        root = Path(directory)
        plugin.save_state(str(root / "initial.state"))
        initial = VitalVst3StateTemplate((root / "initial.state").read_bytes()).preset_document
        states = []
        for endpoint in (0.0, 1.0):
            for description in matched.values():
                plugin.set_parameter(description["index"], endpoint)
            plugin.save_state(str(root / "endpoint.state"))
            states.append(VitalVst3StateTemplate((root / "endpoint.state").read_bytes()).preset_document["settings"])
    ranges = {}
    for name, metadata in matched.items():
        lo, hi = states[0][name], states[1][name]
        if lo > hi:
            raise ValueError(f"inverted raw parameter range for {name}")
        ranges[name] = {"min": lo, "max": hi, "default": initial["settings"][name],
                        "is_discrete": metadata["isDiscrete"], "display_name": metadata["name"], "parameter_index": metadata["index"]}
    return {"renderer_id": renderer.renderer_id, "plugin_version": "1.6.4", "matched_count": len(ranges),
            "unmatched": sorted(contract.scalar_names - ranges.keys()), "parameters": ranges,
            "scope": "Raw saved-state endpoints for uniquely named exposed parameters; historical 1.5.5 bounds are not asserted."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = probe()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"matched": report["matched_count"], "unmatched": report["unmatched"]}))

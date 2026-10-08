"""Measure old-preset loading/default filling in the reviewed local Vital plugin."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

from preset_contract import PresetContract
from obruxo_data.render import VitalRenderer
from obruxo_data.render.vita import VitalVst3StateTemplate, verify_loaded_scalars
from obruxo_data.vital.validation import _difference_pointers


def audit() -> dict:
    import dawdreamer as daw
    import vita

    renderer = VitalRenderer()  # Enforces the reviewed plugin fingerprint before loading native code.
    results = []
    for version in ("1.0.8", "1.5.5", "1.6.4"):
        preset = PresetContract().template
        preset.set_raw("osc_1_transpose", -12)
        preset.set_raw("osc_1_level", 0.3)
        document = preset.to_dict()
        document["synth_version"] = version
        for table in document["settings"]["wavetables"]:
            table["version"] = version
        synth = vita.Synth()
        if not synth.load_json(json.dumps(document)):
            raise RuntimeError(f"Vita rejected probe version {version}")
        drift = sorted(_difference_pointers(document, json.loads(synth.to_json())))

        with tempfile.TemporaryDirectory(prefix="obruxo-compatibility-") as directory:
            root = Path(directory)
            engine = daw.RenderEngine(22_050, 128)
            plugin = engine.make_plugin_processor("vital", str(renderer.plugin_path))
            plugin.save_state(str(root / "initial.state"))
            template = VitalVst3StateTemplate((root / "initial.state").read_bytes())
            (root / "requested.state").write_bytes(template.build(json.dumps(document)))
            plugin.load_state(str(root / "requested.state"))
            plugin.save_state(str(root / "loaded.state"))
            loaded = VitalVst3StateTemplate((root / "loaded.state").read_bytes()).preset_document
            checked = verify_loaded_scalars(document, loaded)
            added = sorted(loaded["settings"].keys() - document["settings"].keys())
            filled_defaults = all(loaded["settings"][key] == template.preset_document["settings"][key] for key in added)
            plugin.add_midi_note(60, 100, 0., 0.5)
            engine.load_graph([(plugin, [])])
            if not engine.render(0.6):
                raise RuntimeError("native render failed")
            audio = engine.get_audio().mean(axis=0)
            peak_hz = float(np.fft.rfftfreq(len(audio), 1 / 22_050)[np.abs(np.fft.rfft(audio)).argmax()])
            if not np.isfinite(audio).all() or abs(peak_hz - 130.8128) > 3 or not filled_defaults:
                raise RuntimeError(f"native compatibility check failed for {version}")
        results.append({"preset_version": version, "loaded_version": loaded["synth_version"],
                        "provided_scalar_controls_verified": checked, "added_settings_keys": added,
                        "added_fields_equal_native_init_defaults": filled_defaults,
                        "vita_roundtrip_changed_paths": drift, "peak_hz": peak_hz})
    return {"renderer_id": renderer.renderer_id, "cases": results,
            "scope": "Pinned authored 772-scalar oscillator-only template; not arbitrary patches or every Vital version."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    report = audit()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, allow_nan=False))

"""Native readback/render checks for authored full 1.5.5 factory-based presets."""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import tempfile

import numpy as np

from full_contract import FullPresetContract
from obruxo_data.render import VitalRenderer
from obruxo_data.render.vita import VitalVst3StateTemplate, verify_loaded_scalars
from obruxo_data.vital.validation import _difference_pointers


def verify():
    import dawdreamer as daw

    contract = FullPresetContract()
    renderer = VitalRenderer()  # Checks the installed plugin fingerprint.
    results = []
    for phase, shepard in ((0.0, False), (0.5, False), (1.0, False), (0.5, True)):
        document = contract.decode_scalars({"osc_1_level": 0.4, "osc_1_transpose": -12,
                                           "osc_1_spectral_morph_phase": phase, "osc_2_spectral_morph_phase": phase,
                                           "osc_3_spectral_morph_phase": phase})
        # Exercise all six nested families, including a real modulation connection.
        document["settings"]["modulations"][0] = {"source": "lfo_1", "destination": "osc_1_level"}
        document["settings"]["modulation_1_amount"] = 0.1
        document["settings"]["random_values"][0]["seed"] = 17
        document["settings"]["custom_warps"][0]["powers"][0] = 0.25
        if shepard:
            document["settings"]["wavetables"][0]["groups"][0]["components"][0]["type"] = "Shepard Tone Source"
        contract.validate(document).require_valid()
        with tempfile.TemporaryDirectory(prefix="obruxo-155-native-") as directory:
            root = Path(directory)
            engine = daw.RenderEngine(22050, 128)
            plugin = engine.make_plugin_processor("vital", str(renderer.plugin_path))
            plugin.save_state(str(root / "initial.state"))
            template = VitalVst3StateTemplate((root / "initial.state").read_bytes())
            (root / "requested.state").write_bytes(template.build(json.dumps(document, allow_nan=False)))
            plugin.load_state(str(root / "requested.state"))
            plugin.save_state(str(root / "loaded.state"))
            loaded = VitalVst3StateTemplate((root / "loaded.state").read_bytes()).preset_document
            scalar_count = verify_loaded_scalars(document, loaded)
            ramps = {k: loaded["settings"][k] for k in contract.spec["fixed_native_ramps"]}
            if ramps != contract.spec["fixed_native_ramps"]:
                raise RuntimeError("native ramp defaults differ from the contract")
            nested_changes = {}
            for key in ("lfos", "custom_warps", "random_values", "sample", "wavetables", "modulations"):
                changes = sorted(_difference_pointers(document["settings"][key], loaded["settings"][key]))
                if changes:
                    nested_changes[key] = changes
            allowed = {"sample": ["/samples"], "wavetables": ["/0/version", "/1/version", "/2/version"]}
            if any(key not in allowed or set(paths) - set(allowed[key]) for key, paths in nested_changes.items()):
                raise RuntimeError(f"unclassified nested readback drift: {nested_changes}")
            if any(w["version"] != "1.6.4" for w in loaded["settings"]["wavetables"]):
                raise RuntimeError("unexpected wavetable version migration")
            before = np.frombuffer(base64.b64decode(document["settings"]["sample"]["samples"]), dtype="<i2").astype(np.int32)
            after = np.frombuffer(base64.b64decode(loaded["settings"]["sample"]["samples"]), dtype="<i2").astype(np.int32)
            sample_delta = int(np.max(np.abs(before - after))) if before.shape == after.shape else 65536
            # Sample::stateToJson serializes the padded buffer from its beginning.
            # Recognize exactly the measured four-sample shift, not arbitrary drift.
            shift = 4
            shifted_delta = int(np.max(np.abs(before[:-shift] - after[shift:]))) if before.shape == after.shape else 65536
            if shifted_delta > 1 or np.any(after[:shift]):
                raise RuntimeError("sample drift differs from the reviewed four-zero prefix and PCM16 rounding")
            plugin.add_midi_note(60, 100, 0.0, 0.5)
            engine.load_graph([(plugin, [])])
            if not engine.render(0.6):
                raise RuntimeError("native render failed")
            audio = engine.get_audio()
            rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
            if not np.isfinite(audio).all() or rms <= 1e-6:
                raise RuntimeError("native audio is nonfinite or unexpectedly silent")
            results.append({"phase": phase, "shepard_source": shepard, "provided_scalars_verified": scalar_count, "ramp_defaults_verified": len(ramps),
                            "sample_max_pcm16_delta": sample_delta,
                            "sample_readback_shift_samples": shift, "sample_shift_aligned_max_pcm16_delta": shifted_delta,
                            "nested_changes": nested_changes, "finite_nonsilent_audio": True, "rms": rms})
    return {"renderer_id": renderer.renderer_id, "plugin_version": "1.6.4", "preset_version": "1.5.5", "cases": results,
            "scope": "Four full factory-based authoring cases; not a native validation of the entire corpus or the 1.5.5 binary."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = verify()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2))

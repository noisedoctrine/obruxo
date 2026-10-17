"""Compare native signed/unsigned encodings directly, outside the unsigned preset API."""
import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import tempfile

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


def main():
    import dawdreamer as daw
    import numpy as np
    from obruxo_data.hashing import canonical_sha256
    from obruxo_data.render.base import FULL_SEED_RENDERER_ID
    from obruxo_data.render.vita import VitalRenderer, VitalVst3StateTemplate
    from obruxo_training.asset_codec import FactoryAssetCodec
    from obruxo_training.audit import write_json
    from obruxo_training.reproducibility import FULL_PHASE_CONTROL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if importlib.metadata.version("dawdreamer") != "0.8.3":
        raise ValueError("DawDreamer 0.8.3 is required")
    renderer = VitalRenderer(renderer_id=FULL_SEED_RENDERER_ID)
    codec = FactoryAssetCodec(json.loads(args.asset_bank.read_text()))
    base = codec.schema.template()
    base["settings"]["wavetables"][0] = next(value for value in codec._waves if value.get("name") == "Basic Shapes")
    base["settings"].update(FULL_PHASE_CONTROL)
    base["settings"].update(volume=3844., osc_1_level=.15, osc_1_wave_frame=128., osc_1_spectral_morph_type=6., osc_1_spectral_morph_amount=.6)
    args.output.mkdir(parents=True, exist_ok=False)
    plan = {"check": "native_int32_uint32_seed_alias_v1", "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "renderer_id": FULL_SEED_RENDERER_ID, "plugin_sha256": renderer.engine_fingerprint, "factory_bank_id": codec.bank["bank_id"],
            "attempt_budget": 16, "uint32_cases": [2147483648, 2270940437, 4294967294, 4294967295],
            "performance": {"note": 60, "velocity": 100, "start_seconds": 0., "duration_seconds": .5, "render_seconds": .75},
            "phase_control": FULL_PHASE_CONTROL, "claim": "exact_readback_and_audio_alias_in_fixed_native_context", "final_test_audio": False}
    write_json(args.output / "plan.json", plan)
    rows = []
    for value in plan["uint32_cases"]:
        group = []
        for seed in (value, value, value - 2**32, value - 2**32):
            document = deepcopy(base)
            document["settings"]["random_values"][0]["seed"] = seed
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "state"
                engine = daw.RenderEngine(44100, 128)
                synth = engine.make_plugin_processor("vital", str(renderer.plugin_path))
                synth.save_state(str(path))
                template = VitalVst3StateTemplate(path.read_bytes())
                path.write_bytes(template.build(json.dumps(document)))
                synth.load_state(str(path))
                synth.save_state(str(path))
                loaded = VitalVst3StateTemplate(path.read_bytes()).preset_document
                synth.add_midi_note(60, 100, 0., .5)
                engine.load_graph([(synth, [])])
                engine.render(.75)
                audio = engine.get_audio().astype(np.float32)
                group.append({"input": seed, "loaded": loaded["settings"]["random_values"][0]["seed"],
                              "audio_sha256": hashlib.sha256(audio.tobytes()).hexdigest(), "peak": float(np.abs(audio).max())})
        passed = len({(row["audio_sha256"], row["loaded"]) for row in group}) == 1 and all(row["loaded"] == value - 2**32 for row in group)
        rows.append({"uint32": value, "int32": value - 2**32, "exact_repeated_audio_and_readback": passed, "outcomes": group})
        print(json.dumps(rows[-1]), flush=True)
    summary = {"plan_sha256": canonical_sha256(plan), "plan": plan, "cases": rows, "attempted": 16,
               "passed": all(row["exact_repeated_audio_and_readback"] for row in rows), "final_test_audio": False, "audio_files_written": 0}
    write_json(args.output / "summary.json", summary)
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

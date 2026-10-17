"""Inspect a frozen training source's loaded scalars; no tolerance changes or audio rendering."""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import tempfile

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


def main():
    import dawdreamer as daw
    from obruxo_data.hashing import canonical_json
    from obruxo_data.render.base import FULL_CURVE_RENDERER_ID
    from obruxo_data.render.curve_inputs import prepare_curve_inputs
    from obruxo_data.render.vita import VitalRenderer, VitalVst3StateTemplate
    from obruxo_data.vital.full_contract import FullPresetContract
    from obruxo_training.audit import load_snapshot, write_json
    from obruxo_training.full_scalar import FullScalarCodec
    from obruxo_training.reproducibility import FULL_PHASE_CONTROL
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot", "corpus", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--state", required=True)
    args = parser.parse_args()
    schema = FullPresetContract(renderer_id=FULL_CURVE_RENDERER_ID)
    _, _, manifest = load_snapshot(args.snapshot, schema)
    if args.state not in manifest.by_state or manifest.by_state[args.state].split != "train":
        raise ValueError("readback diagnosis is restricted to frozen training states")
    members = json.loads((args.snapshot / "members.json").read_text())["records"]
    row = next(row for row in members if row["identity"]["state"] == args.state)
    root = args.corpus.resolve(strict=True)
    source = root / row["relative_path"]
    path = source.resolve(strict=True)
    if not path.is_relative_to(root) or source.is_symlink() or args.output.resolve().is_relative_to(root):
        raise ValueError("readback paths escape the declared source/output boundaries")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != row["identity"]["raw"]:
        raise ValueError("frozen training source changed")
    document, report = schema.loads(raw)
    report.require_valid()
    document["settings"].update(FULL_PHASE_CONTROL)
    document, _ = prepare_curve_inputs(document)
    if importlib.metadata.version("dawdreamer") != "0.8.3":
        raise ValueError("readback requires the pinned DawDreamer 0.8.3")
    renderer = VitalRenderer(renderer_id=FULL_CURVE_RENDERER_ID)
    engine = daw.RenderEngine(44100, 128)
    engine.set_bpm(120.)
    plugin = engine.make_plugin_processor("vital", str(renderer.plugin_path))
    with tempfile.TemporaryDirectory(prefix="obruxo-readback-") as folder:
        folder = Path(folder)
        plugin.save_state(str(folder / "initial.state"))
        template = VitalVst3StateTemplate((folder / "initial.state").read_bytes())
        (folder / "requested.state").write_bytes(template.build(canonical_json(document)))
        plugin.load_state(str(folder / "requested.state"))
        plugin.save_state(str(folder / "loaded.state"))
        loaded = VitalVst3StateTemplate((folder / "loaded.state").read_bytes()).preset_document
    scalar = FullScalarCodec()
    projected = scalar._values(scalar._target(document["settings"]))
    differences = [{"name": name, "requested": document["settings"][name], "loaded": loaded["settings"].get(name),
                    "codec_grid": projected[name], "loaded_matches_codec_grid": loaded["settings"].get(name) == projected[name]}
                   for name in sorted(schema.scalar_names) if type(loaded["settings"].get(name)) not in (int, float)
                   or not math.isclose(document["settings"][name], loaded["settings"][name], rel_tol=1e-6, abs_tol=1e-7)]
    files = [Path(__file__), *(RESEARCH / "training/obruxo_training").glob("*.py"),
             *(RESEARCH / "data_generation/obruxo_data/render").glob("*.py"), RESEARCH / "data_generation/obruxo_data/vital/full_contract.py"]
    result = {"diagnosis": "native_scalar_readback_comparison_without_acceptance_change_v1", "state_id": args.state,
              "source_sha256_lf": {path.relative_to(RESEARCH).as_posix(): hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for path in files},
              "renderer_id": FULL_CURVE_RENDERER_ID, "phase_control": FULL_PHASE_CONTROL, "fresh_process_state_loads": 1,
              "differences": differences, "render_attempts": 0, "verifier_tolerance_changed": False, "final_test_audio": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    write_json(args.output, result)
    print(json.dumps({"state_id": args.state, "differences": differences}), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())

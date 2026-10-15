"""Compare original pinned factory exports with imported assets through native Vital.

The reference path loads the original table directly into Vital, whose legacy importer
understands it. It intentionally does not use Vita's export-roundtrip validator. The
candidate path uses the complete production renderer validation. This diagnostic is
never a training renderer and accepts only the repository's hash-pinned factory bank.
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import tempfile
from time import perf_counter

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


@dataclass(frozen=True)
class OriginalFactoryRenderer:
    plugin_path: str

    def __call__(self):
        return self

    def render(self, request):
        import dawdreamer as daw
        import numpy as np
        from obruxo_data.midi import TempoMap
        from obruxo_data.render.base import RenderProvenance, RenderResult
        from obruxo_data.render.vita import VitalRenderer, VitalVst3StateTemplate, verify_loaded_controls
        from obruxo_training.stream import request_frames
        renderer = VitalRenderer(self.plugin_path, renderer_id=request.renderer_id)  # Enforce the pinned native binary.
        request.performance.validate(renderer.capabilities).require_valid()
        timing = TempoMap.from_performance(request.performance)
        tempo = next((event.data[0] for event in request.performance.canonical_events() if event.kind.value == "tempo"), 500_000)
        with tempfile.TemporaryDirectory(prefix="obruxo-alpha-import-") as directory:
            path = Path(directory) / "state"
            engine = daw.RenderEngine(request.sample_rate, 128)
            engine.set_bpm(60_000_000 / tempo)
            plugin = engine.make_plugin_processor("vital", self.plugin_path)
            plugin.save_state(str(path))
            template = VitalVst3StateTemplate(path.read_bytes())
            path.write_bytes(template.build(request.preset_json))
            plugin.load_state(str(path))
            plugin.save_state(str(path))
            verify_loaded_controls(json.loads(request.preset_json), VitalVst3StateTemplate(path.read_bytes()).preset_document,
                                   legacy_training=True)
            for note in request.performance.note_spans():
                start, end = (timing.tick_to_sample(tick, request.sample_rate) for tick in (note.start_tick, note.end_tick))
                plugin.add_midi_note(note.pitch, note.velocity, start / request.sample_rate, (end-start) / request.sample_rate)
            engine.load_graph([(plugin, [])])
            if not engine.render(request_frames(request) / request.sample_rate):
                raise RuntimeError("native reference render failed")
            audio = np.ascontiguousarray(engine.get_audio().T, dtype=np.float32)
        provenance = RenderProvenance(request.request_id, request.renderer_id, "0.8.3", renderer.engine_fingerprint,
                                      {"validation": "original_pinned_factory_direct_native_reference"})
        return RenderResult(audio, request.sample_rate, (), provenance)


def main():
    from obruxo_data.hashing import canonical_json, canonical_sha256
    from obruxo_data.render.base import RenderRequest
    from obruxo_data.render.qa import compare_audio
    from obruxo_training.alpha import AlphaCodec, read_wavetable_bank
    from obruxo_training.audit import write_json
    from obruxo_training.codec import PilotTarget
    from obruxo_training.performances import initial_bank
    from obruxo_training.stream import NativeRendererFactory, render_isolated
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wavetable-root", type=Path, required=True)
    parser.add_argument("--plugin-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bank = read_wavetable_bank(RESEARCH / "training/manifests/alpha_factory_v1.json", args.wavetable_root)
    codec, cases = AlphaCodec(bank), []
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "status.json", {"complete": False})
    started = perf_counter()
    for index, reference in enumerate(bank["references"], 1):
        for frame in (0., 127.5, 255.):
            target = codec.default_target()
            values, categories = list(target.continuous), list(target.categorical)
            values[9], categories[4] = frame / 255., index
            imported = codec.decode(PilotTarget(tuple(values), tuple(categories))).decode()
            original = json.loads(imported)
            original["settings"]["wavetables"][0] = json.loads(reference["source_json"])
            outcomes = []
            for document, factory in ((canonical_json(original), OriginalFactoryRenderer(args.plugin_path)),
                                      (imported, NativeRendererFactory(args.plugin_path))):
                request = RenderRequest(preset_json=document, performance=initial_bank()[0].performance(), sample_rate=44100, channels=2,
                                        tail_seconds=2., renderer_id=codec.contract.renderer_id)
                outcomes.append(render_isolated(request, renderer_factory=factory, timeout=30))
            (source, source_meta), (candidate, candidate_meta) = outcomes
            comparison = compare_audio(source, candidate) if source is not None and candidate is not None else {}
            cases.append({"asset_id": reference["asset_id"], "frame": frame, "source_sha256": reference["source_sha256"],
                          "imported_sha256": canonical_sha256(json.loads(imported)["settings"]["wavetables"][0]),
                          "source": source_meta, "candidate": candidate_meta, "comparison": comparison,
                          "passed": comparison.get("within_tolerance", False)})
            print(json.dumps({"case": len(cases), "asset": reference["asset_id"], "frame": frame, "passed": cases[-1]["passed"]}), flush=True)
    result = {"check": "alpha_factory_import_native_v1", "cases": cases, "planned_cases": 9, "requested_renders": 18,
              "accepted_renders": sum(case[role]["status"] == "accepted" for case in cases for role in ("source", "candidate")),
              "passed": len(cases) == 9 and all(case["passed"] for case in cases),
              "elapsed_seconds": perf_counter() - started, "audio_files_written": 0}
    write_json(args.output / "summary.json", result)
    write_json(args.output / "status.json", {"complete": True, "passed": result["passed"]})
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

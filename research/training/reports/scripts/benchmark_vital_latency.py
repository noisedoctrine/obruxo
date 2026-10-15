"""Measure fresh-process Vital rendering and individual phases on five training families."""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter


@dataclass(frozen=True)
class TimedFactory:
    plugin_path: str | None

    def __call__(self):
        return TimedRenderer(self.plugin_path)


class TimedRenderer:
    def __init__(self, plugin_path):
        from obruxo_training.stream import NativeRendererFactory
        self.renderer = NativeRendererFactory(plugin_path)()

    def render(self, request):
        import dawdreamer as daw
        from obruxo_data.render import vita

        phases = {}

        def timed(name, function):
            def call(*args, **kwargs):
                started = perf_counter()
                try:
                    return function(*args, **kwargs)
                finally:
                    phases[name] = phases.get(name, 0.) + perf_counter() - started
            return call

        class Engine:
            def __init__(self, *args, **kwargs):
                self.engine = timed("engine_creation", original_engine)(*args, **kwargs)

            def __getattr__(self, name):
                function = getattr(self.engine, name)
                if name in {"make_plugin_processor", "render", "get_audio", "load_graph"}:
                    return timed(name, function)
                return function

        original_engine = daw.RenderEngine
        originals = {name: getattr(vita, name) for name in ("juce_memory_block_encode", "juce_memory_block_decode", "verify_loaded_controls")}
        daw.RenderEngine = Engine
        for name, function in originals.items():
            setattr(vita, name, timed(name, function))
        try:
            result = self.renderer.render(request)
        finally:
            daw.RenderEngine = original_engine
            for name, function in originals.items():
                setattr(vita, name, function)
        result.qa["phase_seconds"] = phases
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plugin-path")
    args = parser.parse_args()
    research = args.checkout.resolve() / "research"
    sys.path[:0] = [str(research / "training"), str(research / "data_generation")]
    from obruxo_data.render.base import RenderRequest
    from obruxo_training.pilot import PilotPool
    from obruxo_training.performances import initial_bank
    from obruxo_training.stream import render_isolated, request_frames

    if not args.requests.exists():
        pool = PilotPool.load(args.pool)
        families = {}
        for entry in pool.entries:
            if pool.manifest.by_state[entry.state_id].split == "train":
                families.setdefault(pool.codec.family(entry.target), entry)
        rows = []
        for family, entry in families.items():
            request = RenderRequest(preset_json=pool.codec.decode(entry.target).decode(), performance=initial_bank()[0].performance(),
                                    sample_rate=pool.codec.contract.sample_rate, channels=2, tail_seconds=pool.codec.contract.tail_seconds,
                                    renderer_id=pool.codec.contract.renderer_id)
            rows.append({"family": family, "state_id": entry.state_id, "split": "train", "request": request.to_dict()})
        args.requests.write_text(json.dumps({"pool_sha256": hashlib.sha256(args.pool.read_bytes()).hexdigest(), "rows": rows}, indent=2) + "\n")
    panel = json.loads(args.requests.read_text())
    if panel["pool_sha256"] != hashlib.sha256(args.pool.read_bytes()).hexdigest() or any(row["split"] != "train" for row in panel["rows"]):
        raise ValueError("timing panel differs from its training-only pool")
    if args.output.exists():
        raise FileExistsError(args.output)
    records = []
    started = perf_counter()
    for row in panel["rows"]:
        request = RenderRequest.from_dict(row["request"])
        audio, metadata = render_isolated(request, renderer_factory=TimedFactory(args.plugin_path), timeout=30)
        if audio is None or metadata["status"] != "accepted":
            raise ValueError(metadata)
        record = {"family": row["family"], "state_id": row["state_id"], "request_id": request.request_id,
                  "audio_seconds": request_frames(request) / request.sample_rate, **metadata}
        records.append(record)
        print(json.dumps({"family": row["family"], "wall_seconds": metadata["wall_seconds"],
                          "phase_seconds": metadata["renderer_qa"]["phase_seconds"]}), flush=True)
    output = {"format": "obruxo_vital_latency_v1", "training_only": True, "final_test_used": False,
              "checkout": str(args.checkout.resolve()), "panel_sha256": hashlib.sha256(args.requests.read_bytes()).hexdigest(),
              "renderer_source_sha256": hashlib.sha256((research / "data_generation/obruxo_data/render/vita.py").read_bytes()).hexdigest(),
              "elapsed_seconds": perf_counter() - started, "records": records}
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

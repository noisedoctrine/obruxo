"""A causal phase control for worker differences; never alters the training pool."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


def main():
    from obruxo_data.render.base import RenderRequest
    from obruxo_data.render.qa import compare_audio
    from obruxo_training.audit import write_json
    from obruxo_training.stream import NativeRendererFactory, RenderWorker, render_isolated

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plugin-path")
    args = parser.parse_args()
    panel = json.loads(args.panel.read_text())
    if any(row["split"] != "train" for row in panel["rows"]):
        raise ValueError("phase probe requires training-only source patches")
    args.output.mkdir(parents=True, exist_ok=False)
    requests = []
    for row in panel["rows"]:
        original = RenderRequest.from_dict(row["request"])
        document = json.loads(original.preset_json)
        for index in range(1, 4):
            document["settings"][f"osc_{index}_random_phase"] = 0.
        requests.append(replace(original, preset_json=json.dumps(document)))
    factory = NativeRendererFactory(args.plugin_path)
    events, baseline, checks = [], [], []

    def record(mode, index, request, metadata):
        events.append({"mode": mode, "case": index, "request_id": request.request_id, **metadata})
        with (args.output / "renders.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(events[-1]) + "\n")
        print(json.dumps({"mode": mode, "case": index, "status": metadata["status"]}), flush=True)
        if metadata["status"] != "accepted":
            raise ValueError(metadata)

    for index, request in enumerate(requests):
        audio, metadata = render_isolated(request, renderer_factory=factory, timeout=30.)
        record("fresh", index, request, metadata)
        baseline.append((audio, metadata))
    with RenderWorker(renderer_factory=factory, max_jobs=32) as worker:
        for index in range(10):
            selected = index % len(requests)
            request = requests[selected]
            audio, metadata = worker.render(request, timeout=30.)
            record("reuse", index, request, metadata)
            reference, expected = baseline[selected]
            checks.append({"family": panel["rows"][selected]["family"], "case": index, **compare_audio(reference, audio),
                           "audio_float32_identical": expected["qa"]["audio_float32_sha256"] == metadata["qa"]["audio_float32_sha256"],
                           "native_state_identical": expected["renderer_qa"]["native_state"] == metadata["renderer_qa"]["native_state"]})
    summary = {"format": "obruxo_worker_phase_control_v1", "panel_sha256": hashlib.sha256(args.panel.read_bytes()).hexdigest(),
               "intervention": "osc_1/2/3_random_phase_zero_only", "training_pool_changed": False, "final_test_used": False,
               "native_attempts": len(events), "accepted": len(events), "checks": checks,
               "all_within_tolerance": all(row["within_tolerance"] for row in checks),
               "all_audio_identical": all(row["audio_float32_identical"] for row in checks),
               "all_native_states_identical": all(row["native_state_identical"] for row in checks)}
    write_json(args.output / "summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "checks"}, indent=2), flush=True)


if __name__ == "__main__":
    main()

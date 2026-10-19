"""Rerender the A1 duration failures plus exact-repeat/control checks, at most 19 requests."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


def main():
    from obruxo_data.midi import Performance
    from obruxo_data.render.base import RenderRequest
    from obruxo_training.codec import PilotCodec, PilotTarget
    from obruxo_training.reproducibility import fixed_generator_phases
    from obruxo_training.stream import RenderWorker, NativeRendererFactory, request_frames
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    records = json.loads((args.source / "records.json").read_text())
    plan = json.loads((args.source / "plan.json").read_text())
    before = json.loads(args.before.read_text())
    failures = [row["sequence"] for row in records if row["receipt"].get("message") == "renderer returned an incompatible audio shape"]
    if len(failures) != 16 or before[0]["expected_frames"] != before[0]["actual_frames"] + 1:
        raise ValueError("unexpected duration regression inputs")
    codec, rows, start = PilotCodec(), [], perf_counter()
    if codec.to_dict() != plan["codec"]:
        raise ValueError("source codec changed")
    for index in [*failures, failures[0], 0, 0]:
        if perf_counter() - start > 180:
            raise RuntimeError("duration regression exceeds three-minute limit")
        recipe = plan["training_recipes"][index]
        request = RenderRequest(preset_json=fixed_generator_phases(codec.decode(PilotTarget.from_dict(recipe["target"]))).decode(),
                                performance=Performance.from_dict(recipe["performance"]), renderer_id=codec.contract.renderer_id,
                                tail_seconds=codec.contract.tail_seconds)
        with RenderWorker(renderer_factory=NativeRendererFactory(renderer_id=codec.contract.renderer_id), max_jobs=1) as worker:
            audio, receipt = worker.render(request, timeout=30.)
        if audio is None or receipt["status"] != "accepted" or len(audio) != request_frames(request):
            raise RuntimeError(f"duration regression rejected request {index}: {receipt}")
        rows.append({"recipe": index, "request_id": request.request_id, "frames": len(audio),
                     "audio_sha256": receipt["qa"]["audio_float32_sha256"], "status": receipt["status"],
                     "frame_policy": receipt["provenance"]["settings"]["frame_policy"]})
    if rows[0]["audio_sha256"] != rows[16]["audio_sha256"] or rows[-1]["audio_sha256"] != rows[-2]["audio_sha256"]:
        raise ValueError("fixed-phase repeat changed")
    if rows[-1]["audio_sha256"] != before[1]["qa"]["audio_float32_sha256"]:
        raise ValueError("previously correct control audio changed")
    result = {"complete": True, "source_plan_sha256": hashlib.sha256((args.source / "plan.json").read_bytes()).hexdigest(),
              "before": [{key: row[key] for key in ("index", "expected_frames", "actual_frames", "request_id")} for row in before],
              "after": rows, "exact_repeat_pairs": 2, "previously_correct_control_unchanged": True,
              "native_requests_this_check": 19, "total_native_requests_including_reproduction": 21,
              "elapsed_seconds": perf_counter() - start, "optimizer_updates": 0, "historical_run_unchanged": True}
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"passed": True, "native_requests": 19, "exact_repeats": 2, "seconds": result["elapsed_seconds"]}))


if __name__ == "__main__":
    main()

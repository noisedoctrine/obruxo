"""Freeze and qualify the fixed pre-roll renderer on the startup failure and control-response fixtures."""

import argparse
import json
from pathlib import Path
import sys
from time import time

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


def main():
    import numpy as np
    import torch
    from obruxo_data.hashing import canonical_sha256, sha256_bytes
    from obruxo_data.midi import Performance, TempoMap
    from obruxo_training.audit import write_json
    from obruxo_training.benchmark import MemoryMonitor
    from obruxo_training.codec import PilotCodec, PilotTarget
    from obruxo_training.native_fit import digest
    from obruxo_training.probe_capture import CaptureFactory
    from obruxo_training.temporal_proxy_data import NativeSession
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "prefix-audit", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    prefix = json.loads(args.prefix_audit.read_text())
    parent = json.loads((args.parent / "plan.json").read_text())
    for name, expected in prefix["evidence_sha256"].items():
        if digest(args.parent / name) != expected:
            raise ValueError("stopped parent changed")
    codec, failed = PilotCodec(), prefix["rejected_probe"]
    bad = PilotTarget.from_dict(failed["candidate"])
    values = list(bad.continuous)
    values[0] -= .1
    lower = codec.canonical_target(PilotTarget(tuple(values), bad.categorical))
    extreme = list(bad.continuous)
    extreme[0], extreme[1], extreme[5], extreme[7] = 1., .5, 0., 1.
    extreme = codec.canonical_target(PilotTarget(tuple(extreme), (2, 0)))
    recipe = parent["training"][failed["state_index"]]
    cases = [("cold_bad", bad, 0), ("source", PilotTarget.from_dict(recipe["target"]), 8192),
             ("source_repeat", PilotTarget.from_dict(recipe["target"]), 8192), ("bad", bad, 8192), ("bad_repeat", bad, 8192),
             ("lower", lower, 8192), ("long_preroll", bad, 16384), ("extreme", extreme, 8192)]
    paths = [Path(__file__), RESEARCH / "training/obruxo_training/temporal_proxy_data.py", RESEARCH / "training/obruxo_training/probe_capture.py"]
    paths += [RESEARCH / "data_generation/obruxo_data/render" / name for name in ("base.py", "vita.py", "midi_transport.py")]
    plan = {"format": "native_preroll_qualification_v1", "parent_plan_id": prefix["plan_id"], "codec": codec.to_dict(),
            "prefix_audit_sha256": digest(args.prefix_audit), "performance": recipe["performances"][failed["view"]],
            "cases": [{"name": name, "target": target.to_dict(), "preroll_frames": frames} for name, target, frames in cases],
            "source": {path.name: sha256_bytes(path.read_bytes().replace(b"\r\n", b"\n")) for path in paths},
            "limits": {"qualification_attempts": 8, "wall_seconds": 600, "process_tree_bytes": 8 * 1024**3, "archive_bytes": 64 * 1024**2},
            "gates": {"legacy_hash_exact": True, "warm_all_accepted": True, "repeats_sample_exact": True,
                      "level_relative_wave_error_max": 1e-5, "8192_vs_16384_relative_wave_error_max": 1e-5},
            "no_model_quality_claim": True, "test_split_used": False}
    if not args.execute:
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output / "plan.json", plan)
        print(json.dumps({"plan_id": canonical_sha256(plan), "native_requests": 8}))
        return
    if json.loads((args.output / "plan.json").read_text()) != plan or (args.output / "run.json").exists():
        raise ValueError("preroll check changed or was already attempted")
    started = time()
    write_json(args.output / "run.json", {"plan_id": canonical_sha256(plan), "started_unix": started})
    (args.output / "audio").mkdir()
    monitor = MemoryMonitor()
    monitor.start()
    sessions, rows, audio = {}, [], {}
    performance = Performance.from_dict(plan["performance"])
    frames = TempoMap.from_performance(performance).render_frame_count(performance.end_tick, codec.contract.tail_seconds, 44100)
    try:
        for case in plan["cases"]:
            preroll = case["preroll_frames"]
            if preroll not in sessions:
                directory = args.output / f"profile-{preroll}"
                directory.mkdir()
                write_json(directory / "run.json", {"plan_id": canonical_sha256(plan), "started_unix": started})
                execution = {**plan, "render_profile": {"preroll_frames": preroll}}
                sessions[preroll] = NativeSession(directory, execution, codec, monitor,
                                                CaptureFactory(str(args.output / "audio"), codec.contract.renderer_id))
            value, receipt = sessions[preroll].render(PilotTarget.from_dict(case["target"]), performance, "qualification", case["name"])
            captured = np.load(args.output / "audio" / f"{receipt['request_id']}.npy", allow_pickle=False)
            if captured.shape != (frames, 2):
                raise ValueError("pre-roll changed the requested observation length")
            if case["name"] == "cold_bad":
                if receipt["qa"]["audio_float32_sha256"] != failed["receipt"]["qa"]["audio_float32_sha256"]:
                    raise ValueError("zero-preroll legacy audio changed")
            elif value is None or receipt["qa"]["clipping_count"]:
                raise RuntimeError("pre-rolled output failed strict audio QA")
            audio[case["name"]] = captured
            rows.append({**case, "receipt": receipt})
            write_json(args.output / "cases.json", rows)
        repeats = all(np.array_equal(audio[name], audio[f"{name}_repeat"]) for name in ("source", "bad"))
        ratio = ((.15 + .2 * lower.continuous[0]) / (.15 + .2 * bad.continuous[0])) ** 2
        high, low, longer = (audio[name].astype(np.float64) for name in ("bad", "lower", "long_preroll"))
        level_error = float(np.linalg.norm(low - ratio * high) / np.linalg.norm(low))
        settling_error = float(np.linalg.norm(longer - high) / np.linalg.norm(high))
        quality = repeats and level_error <= 1e-5 and settling_error <= 1e-5
    finally:
        monitor.close()
    elapsed = time() - started
    archive = sum(path.stat().st_size for path in args.output.rglob("*") if path.is_file())
    resources = (monitor.errors == 0 and monitor.peak_bytes is not None and monitor.peak_bytes <= plan["limits"]["process_tree_bytes"]
                 and elapsed <= plan["limits"]["wall_seconds"] and archive + 1024**2 <= plan["limits"]["archive_bytes"])
    result = {"plan_id": canonical_sha256(plan), "plan": plan, "cases": rows, "preflight_passed": quality and resources,
              "repeats_sample_exact": repeats, "level_relative_wave_error": level_error,
              "8192_vs_16384_relative_wave_error": settling_error, "native_attempts": len(rows),
              "elapsed_seconds": elapsed, "peak_process_tree_rss_bytes": monitor.peak_bytes,
              "rss_sampling_errors": monitor.errors, "retained_archive_bytes_before_summary": archive,
              "within_resource_limits": resources, "no_model_quality_claim": True, "test_split_used": False}
    write_json(args.output / "summary.json", result)
    print(json.dumps({"preflight_passed": result["preflight_passed"], "level_error": level_error, "settling_error": settling_error}))


if __name__ == "__main__":
    main()

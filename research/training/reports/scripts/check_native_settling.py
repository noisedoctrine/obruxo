"""Test initial control settling with exact MIDI/sample shifts and post-render readback."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from time import time

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


def shifted_performance(performance, sample_rate, tail):
    from obruxo_data.midi import EventKind, TempoMap
    if any(event.kind not in {EventKind.TEMPO, EventKind.NOTE_ON, EventKind.NOTE_OFF} for event in performance.events):
        raise ValueError("settling diagnostic requires the basic note/tempo profile")
    timing = TempoMap.from_performance(performance)
    original_frames = timing.render_frame_count(performance.end_tick, tail, sample_rate)
    note_ticks = {event.tick for event in performance.events if event.kind in {EventKind.NOTE_ON, EventKind.NOTE_OFF}}
    for beats in range(1, 9):
        offset = beats * performance.ticks_per_beat
        frames = timing.tick_to_sample(offset, sample_rate)
        if frames < 4096:
            continue
        if any(timing.tick_to_sample(tick + offset, sample_rate) - timing.tick_to_sample(tick, sample_rate) != frames for tick in note_ticks):
            continue
        if timing.render_frame_count(performance.end_tick + offset, tail, sample_rate) - original_frames != frames:
            continue
        shifted = performance.clone()
        shifted.events = [replace(event, tick=event.tick + offset) if event.kind in {EventKind.NOTE_ON, EventKind.NOTE_OFF} else event
                          for event in performance.events]
        shifted.end_tick += offset
        return shifted, frames
    raise ValueError("could not preserve every MIDI sample offset exactly")


def main():
    import numpy as np
    import torch
    from obruxo_data.hashing import canonical_sha256, sha256_bytes
    from obruxo_data.midi import Performance, TempoMap
    from obruxo_data.render.qa import analyze_audio, audio_float32_sha256
    from obruxo_training.audit import write_json
    from obruxo_training.benchmark import MemoryMonitor
    from obruxo_training.codec import PilotCodec, PilotTarget
    from obruxo_training.native_fit import digest
    from obruxo_training.settling_capture import SettlingCaptureFactory
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
    codec = PilotCodec()
    failed = prefix["rejected_probe"]
    bad = PilotTarget.from_dict(failed["candidate"])
    values = list(bad.continuous)
    values[0] -= .1
    lower = codec.canonical_target(PilotTarget(tuple(values), bad.categorical))
    recipe = parent["training"][failed["state_index"]]
    original = Performance.from_dict(recipe["performances"][failed["view"]])
    shifted, offset = shifted_performance(original, 44100, codec.contract.tail_seconds)
    frames = TempoMap.from_performance(original).render_frame_count(original.end_tick, codec.contract.tail_seconds, 44100)
    paths = [Path(__file__), RESEARCH / "training/obruxo_training/settling_capture.py"]
    plan = {"format": "native_initial_settling_diagnostic_v1", "parent_plan_id": prefix["plan_id"], "codec": codec.to_dict(),
            "prefix_audit_sha256": digest(args.prefix_audit), "original_performance": original.to_dict(), "shifted_performance": shifted.to_dict(),
            "offset_frames": offset, "original_frames": frames, "source_target": recipe["target"], "bad": bad.to_dict(), "lower": lower.to_dict(),
            "cold_bad_audio_sha256": failed["receipt"]["qa"]["audio_float32_sha256"],
            "source": {path.name: sha256_bytes(path.read_bytes().replace(b"\r\n", b"\n")) for path in paths},
            "limits": {"qualification_attempts": 6, "wall_seconds": 600, "process_tree_bytes": 8 * 1024**3, "archive_bytes": 64 * 1024**2},
            "questions": ["post_render_control_state", "clipped_peak_after_silence", "level_response_after_silence", "repeat_exact"],
            "no_training_or_model_quality_claim": True, "test_split_used": False}
    if not args.execute:
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output / "plan.json", plan)
        print(json.dumps({"plan_id": canonical_sha256(plan), "requests": 6, "offset_frames": offset}))
        return
    if json.loads((args.output / "plan.json").read_text()) != plan or (args.output / "run.json").exists():
        raise ValueError("diagnostic changed or was already attempted")
    (args.output / "audio").mkdir()
    write_json(args.output / "run.json", {"plan_id": canonical_sha256(plan), "started_unix": time()})
    monitor = MemoryMonitor()
    monitor.start()
    session = NativeSession(args.output, plan, codec, monitor, SettlingCaptureFactory(str(args.output / "audio"), codec.contract.renderer_id))
    rows, bodies = [], {}
    try:
        cases = (("cold_source", PilotTarget.from_dict(recipe["target"]), original), ("cold_bad", bad, original),
                 ("shifted_source", PilotTarget.from_dict(recipe["target"]), shifted), ("shifted_bad", bad, shifted),
                 ("shifted_lower", lower, shifted), ("shifted_bad_repeat", bad, shifted))
        for name, target, performance in cases:
            _, receipt = session.render(target, performance, "qualification", f"settling:{name}")
            raw = np.load(args.output / "audio" / f"{receipt['request_id']}.npy", allow_pickle=False)
            state = json.loads((args.output / "audio" / f"{receipt['request_id']}.state.json").read_text())
            if audio_float32_sha256(raw) != receipt["qa"]["audio_float32_sha256"]:
                raise ValueError("observer changed native audio")
            trim = offset if name.startswith("shifted") else 0
            body = raw[trim:trim + frames]
            if len(body) != frames:
                raise ValueError("diagnostic crop lost requested frames")
            qa, _ = analyze_audio(body, sample_rate=44100, expected_frames=frames, expected_channels=2)
            rows.append({"name": name, "receipt": receipt, "body_qa": qa, "state": state,
                         "prefix_peak": float(np.abs(raw[:trim]).max(initial=0.)) if trim else None})
            bodies[name] = body.copy()
            write_json(args.output / "cases.json", rows)
        if rows[1]["receipt"]["qa"]["audio_float32_sha256"] != plan["cold_bad_audio_sha256"]:
            raise ValueError("instrumentation changed the cold rejected probe")
    finally:
        monitor.close()
    elapsed = time() - session.started
    raw_hi, raw_lo = .15 + .2 * bad.continuous[0], .15 + .2 * lower.continuous[0]
    expected_ratio = (raw_lo / raw_hi) ** 2
    high, low = bodies["shifted_bad"].astype(np.float64), bodies["shifted_lower"].astype(np.float64)
    ratio = float(np.sqrt(np.mean(low**2) / np.mean(high**2)))
    states_match = all(all(abs(value - row["state"]["after_render"][key]) < 1e-5 for key, value in row["state"]["requested"].items()
                          if value is not None and row["state"]["after_render"].get(key) is not None) for row in rows)
    summary = {"plan_id": canonical_sha256(plan), "plan": plan, "cases": rows, "native_attempts": 6,
               "shifted_bad_repeat_exact": bool(np.array_equal(bodies["shifted_bad"], bodies["shifted_bad_repeat"])),
               "all_requested_controls_match_after_render": states_match, "level_rms_ratio": ratio,
               "quadratic_level_ratio": expected_ratio, "relative_scaled_wave_error": float(np.linalg.norm(low - expected_ratio * high) / np.linalg.norm(low)),
               "cold_bad_peak": rows[1]["body_qa"]["peak"], "shifted_bad_peak": rows[3]["body_qa"]["peak"],
               "elapsed_seconds": elapsed, "peak_process_tree_rss_bytes": monitor.peak_bytes, "rss_sampling_errors": monitor.errors,
               "no_training_or_model_quality_claim": True, "test_split_used": False}
    write_json(args.output / "summary.json", summary)
    print(json.dumps({key: summary[key] for key in ("cold_bad_peak", "shifted_bad_peak", "level_rms_ratio", "quadratic_level_ratio", "all_requested_controls_match_after_render")}))


if __name__ == "__main__":
    main()

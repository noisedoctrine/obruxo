"""Compare the rejected category with its actual accepted base choice, retaining waveform evidence."""

import argparse
import json
from pathlib import Path
import shutil
import sys
from time import time

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


def main():
    import numpy as np
    import torch
    from obruxo_data.hashing import canonical_sha256, sha256_bytes
    from obruxo_data.midi import Performance
    from obruxo_data.render.qa import audio_float32_sha256
    from obruxo_training.audit import write_json
    from obruxo_training.benchmark import MemoryMonitor
    from obruxo_training.codec import PilotCodec, PilotTarget
    from obruxo_training.native_fit import digest
    from obruxo_training.probe_capture import CaptureFactory
    from obruxo_training.probe_validity import clipping_penalty, probe_loss, probe_validity_spec
    from obruxo_training.spectral import SpectralTargets
    from obruxo_training.temporal_proxy_data import NativeSession
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "prefix-audit", "prior", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    prefix = json.loads(args.prefix_audit.read_text())
    parent = json.loads((args.parent / "plan.json").read_text())
    for name, expected in prefix["evidence_sha256"].items():
        if digest(args.parent / name) != expected:
            raise ValueError("stopped parent changed")
    prior_events = [json.loads(line) for line in (args.prior / "renders.jsonl").read_text().splitlines()]
    prior_outcomes = [row for row in prior_events if row["event"] == "outcome"]
    if len(prior_outcomes) != 5 or prior_outcomes[-1]["label"] != "preflight:safe" or prior_outcomes[-1]["status"] != "rejected":
        raise ValueError("expected the retained five-request failed level comparator")
    codec = PilotCodec()
    failed = prefix["rejected_probe"]
    bad = PilotTarget.from_dict(failed["candidate"])
    safe = codec.canonical_target(PilotTarget.from_dict(prefix["pending_anchor"]))
    recipe = parent["training"][failed["state_index"]]
    measurements = [json.loads(line) for line in (args.parent / "measurements.jsonl").read_text().splitlines()]
    base = next(row for row in measurements if row["step"] == failed["optimizer_step"] and row["query"] == 0)
    if base["candidate"] != safe.to_dict():
        raise ValueError("safe comparator must be the actual accepted base selection")
    paths = [Path(__file__), RESEARCH / "training/obruxo_training/probe_validity.py", RESEARCH / "training/obruxo_training/probe_capture.py"]
    plan = {"format": "clipping_category_probe_preflight_v2", "parent_plan_id": prefix["plan_id"],
            "prefix_audit_sha256": digest(args.prefix_audit), "prior_events_sha256": digest(args.prior / "renders.jsonl"),
            "codec": parent["codec"], "policy": probe_validity_spec(), "bad": bad.to_dict(), "safe": safe.to_dict(),
            "recipe": recipe, "view": failed["view"], "expected_bad_audio_sha256": failed["receipt"]["qa"]["audio_float32_sha256"],
            "expected_safe_audio_sha256": base["audio_sha256"],
            "source": {path.name: sha256_bytes(path.read_bytes().replace(b"\r\n", b"\n")) for path in paths},
            "limits": {"qualification_attempts": 12, "wall_seconds": 600, "process_tree_bytes": 8 * 1024**3, "archive_bytes": 32 * 1024**2},
            "gates": {"source_and_safe_exact_repeats": True, "bad_historical_exact": True, "safe_cost_below_bad_penalty": True},
            "prior_failed_attempts": 5, "additional_planned_attempts": 6, "test_split_used": False}
    if not args.execute:
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output / "plan.json", plan)
        print(json.dumps({"plan_id": canonical_sha256(plan), "additional_requests": 6, "combined_requests": 11}))
        return
    if json.loads((args.output / "plan.json").read_text()) != plan or (args.output / "run.json").exists():
        raise ValueError("new comparison changed or was already attempted")
    shutil.copyfile(args.prior / "renders.jsonl", args.output / "renders.jsonl")
    (args.output / "audio").mkdir()
    write_json(args.output / "run.json", {"plan_id": canonical_sha256(plan), "started_unix": time()})
    monitor = MemoryMonitor()
    monitor.start()
    session = NativeSession(args.output, plan, codec, monitor, CaptureFactory(str(args.output / "audio"), codec.contract.renderer_id))
    performance = Performance.from_dict(recipe["performances"][failed["view"]])
    rows, audio = [], {}
    try:
        for name, candidate in (("source", PilotTarget.from_dict(recipe["target"])), ("source_repeat", PilotTarget.from_dict(recipe["target"])),
                                ("bad", bad), ("bad_repeat", bad), ("safe", safe), ("safe_repeat", safe)):
            value, receipt = session.render(candidate, performance, "qualification", f"preflight:v2_{name}")
            captured = np.load(args.output / "audio" / f"{receipt['request_id']}.npy", allow_pickle=False)
            if audio_float32_sha256(captured) != receipt["qa"]["audio_float32_sha256"]:
                raise ValueError("captured diagnostic audio changed")
            peak_frame, peak_channel = np.unravel_index(np.abs(captured).argmax(), captured.shape)
            row = {"name": name, "candidate": candidate.to_dict(), "receipt": receipt,
                   "peak_frame": int(peak_frame), "peak_channel": int(peak_channel), "peak_seconds": float(peak_frame / 44100)}
            if name.startswith("bad"):
                row["penalty"] = clipping_penalty(receipt, session.renderer_identity)
                if receipt["qa"]["audio_float32_sha256"] != plan["expected_bad_audio_sha256"]:
                    raise ValueError("historical rejected patch changed")
            else:
                if value is None:
                    raise RuntimeError("source or actual base selection rejected")
                audio[name] = value.copy()
                if name.startswith("safe") and receipt["qa"]["audio_float32_sha256"] != plan["expected_safe_audio_sha256"]:
                    raise ValueError("historical accepted base selection changed")
            rows.append(row)
            write_json(args.output / "cases.json", rows)
        if not np.array_equal(audio["source"], audio["source_repeat"]) or not np.array_equal(audio["safe"], audio["safe_repeat"]):
            raise ValueError("sample-exact repeats failed")
        record = torch.load(args.parent / "records" / f"{failed['state_index']:03d}.pt", map_location="cpu", weights_only=True)
        if rows[0]["receipt"]["qa"]["audio_float32_sha256"] != record["audio_sha256"][failed["view"]]:
            raise ValueError("source changed since original collection")
        safe_loss, _ = probe_loss(audio["safe"], rows[4]["receipt"], SpectralTargets()(torch.from_numpy(audio["source"]).T[None]), session.renderer_identity)
        if safe_loss >= rows[2]["penalty"]:
            raise ValueError("clipping cost does not discourage the invalid category")
    finally:
        monitor.close()
    elapsed = time() - session.started
    archive = sum(path.stat().st_size for path in args.output.rglob("*") if path.is_file())
    resources = (monitor.errors == 0 and monitor.peak_bytes is not None and monitor.peak_bytes <= plan["limits"]["process_tree_bytes"]
                 and elapsed <= plan["limits"]["wall_seconds"] and archive + 1024**2 <= plan["limits"]["archive_bytes"])
    summary = {"plan_id": canonical_sha256(plan), "plan": plan, "cases": rows, "preflight_passed": resources,
               "safe_spectral_loss": safe_loss, "bad_penalty": rows[2]["penalty"], "native_attempts": len(prior_outcomes) + len(rows),
               "prior_failed_preflight_attempts": len(prior_outcomes), "expected_clipping_rejections": 5,
               "elapsed_seconds": elapsed, "peak_process_tree_rss_bytes": monitor.peak_bytes, "within_resource_limits": resources,
               "renderer_identity": session.renderer_identity, "all_source_and_safe_repeats_exact": True,
               "all_bad_rejections_historical_exact": True, "test_split_used": False, "inverse_updates": 0}
    write_json(args.output / "summary.json", summary)
    print(json.dumps({"preflight_passed": resources, "safe_loss": safe_loss, "bad_penalty": rows[2]["penalty"], "peak_seconds": rows[2]["peak_seconds"]}))


if __name__ == "__main__":
    main()

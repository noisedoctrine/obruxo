"""Freeze and test the clipping-only feasibility cost on the exact stopped-run probe."""

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
    from obruxo_data.midi import Performance
    from obruxo_training.audit import write_json
    from obruxo_training.benchmark import MemoryMonitor
    from obruxo_training.codec import PilotCodec, PilotTarget
    from obruxo_training.native_fit import digest
    from obruxo_training.probe_validity import clipping_penalty, probe_loss, probe_validity_spec
    from obruxo_training.spectral import SpectralTargets
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
    failed = prefix["rejected_probe"]
    bad = PilotTarget.from_dict(failed["candidate"])
    values = list(bad.continuous)
    values[0] = max(0., values[0] - .1)
    safe = PilotCodec().canonical_target(PilotTarget(tuple(values), bad.categorical))
    recipe = parent["training"][failed["state_index"]]
    paths = [Path(__file__), RESEARCH / "training/obruxo_training/probe_validity.py"]
    plan = {"format": "clipping_probe_preflight_v1", "parent_plan_id": prefix["plan_id"], "prefix_audit_sha256": digest(args.prefix_audit),
            "codec": parent["codec"], "policy": probe_validity_spec(), "bad": bad.to_dict(), "safe": safe.to_dict(),
            "recipe": recipe, "view": failed["view"], "expected_bad_audio_sha256": failed["receipt"]["qa"]["audio_float32_sha256"],
            "source": {path.name: sha256_bytes(path.read_bytes().replace(b"\r\n", b"\n")) for path in paths},
            "limits": {"qualification_attempts": 12, "wall_seconds": 600, "process_tree_bytes": 8 * 1024**3, "archive_bytes": 32 * 1024**2},
            "gates": {"bad_exact_repeats": 2, "safe_exact_repeats": 2, "source_exact_repeats": 2,
                      "safe_loss_below_penalty": True, "positive_level_secant": True}, "test_split_used": False}
    if not args.execute:
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output / "plan.json", plan)
        print(json.dumps({"plan_id": canonical_sha256(plan), "planned_requests": 6, "cap": 12}))
        return
    if json.loads((args.output / "plan.json").read_text()) != plan or (args.output / "run.json").exists():
        raise ValueError("preflight changed or was already attempted")
    write_json(args.output / "run.json", {"plan_id": canonical_sha256(plan), "started_unix": time()})
    monitor = MemoryMonitor()
    monitor.start()
    session = NativeSession(args.output, plan, PilotCodec(), monitor)
    performance = Performance.from_dict(recipe["performances"][failed["view"]])
    rows, audio = [], {}
    try:
        for name, candidate in (("source", PilotTarget.from_dict(recipe["target"])), ("source_repeat", PilotTarget.from_dict(recipe["target"])),
                                ("bad", bad), ("bad_repeat", bad), ("safe", safe), ("safe_repeat", safe)):
            value, receipt = session.render(candidate, performance, "qualification", f"preflight:{name}")
            if name.startswith("bad"):
                penalty = clipping_penalty(receipt, session.renderer_identity)
                if receipt["qa"]["audio_float32_sha256"] != plan["expected_bad_audio_sha256"]:
                    raise ValueError("bad probe did not reproduce exact historical audio")
                rows.append({"name": name, "candidate": candidate.to_dict(), "penalty": penalty, "receipt": receipt})
            else:
                if value is None:
                    raise RuntimeError("source or safe headroom probe rejected")
                audio[name] = value.copy()
                rows.append({"name": name, "candidate": candidate.to_dict(), "receipt": receipt})
            write_json(args.output / "cases.json", rows)
        if not np.array_equal(audio["source"], audio["source_repeat"]) or not np.array_equal(audio["safe"], audio["safe_repeat"]):
            raise ValueError("source/safe repeat mismatch")
        record = torch.load(args.parent / "records" / f"{failed['state_index']:03d}.pt", map_location="cpu", weights_only=True)
        if rows[0]["receipt"]["qa"]["audio_float32_sha256"] != record["audio_sha256"][failed["view"]]:
            raise ValueError("parent source audio changed")
        target = SpectralTargets()(torch.from_numpy(audio["source"]).T[None])
        safe_loss, _ = probe_loss(audio["safe"], rows[4]["receipt"], target, session.renderer_identity)
        penalty = rows[2]["penalty"]
        secant = (penalty - safe_loss) / (bad.continuous[0] - safe.continuous[0])
        if safe_loss >= penalty or secant <= 0:
            raise ValueError("invalid-probe loss does not direct level away from rejection")
        passed = True
    finally:
        monitor.close()
    elapsed = time() - session.started
    archive = sum(path.stat().st_size for path in args.output.rglob("*") if path.is_file())
    resources = (monitor.errors == 0 and monitor.peak_bytes is not None and monitor.peak_bytes <= plan["limits"]["process_tree_bytes"]
                 and elapsed <= plan["limits"]["wall_seconds"] and archive + 1024**2 <= plan["limits"]["archive_bytes"])
    summary = {"plan_id": canonical_sha256(plan), "plan": plan, "cases": rows, "preflight_passed": passed and resources,
               "safe_spectral_loss": safe_loss, "bad_penalty": penalty, "level_secant": secant,
               "native_attempts": 6, "elapsed_seconds": elapsed, "peak_process_tree_rss_bytes": monitor.peak_bytes,
               "within_resource_limits": resources, "renderer_identity": session.renderer_identity,
               "all_source_and_safe_repeats_exact": True, "all_bad_rejections_historical_exact": True,
               "test_split_used": False, "inverse_updates": 0}
    write_json(args.output / "summary.json", summary)
    print(json.dumps({"preflight_passed": summary["preflight_passed"], "safe_spectral_loss": safe_loss, "bad_penalty": penalty}))


if __name__ == "__main__":
    main()

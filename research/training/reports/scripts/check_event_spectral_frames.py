"""Freeze and verify repaired training samples on the existing 12 diagnostic pairs."""

import argparse
import json
from pathlib import Path
import sys
from time import time

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]

from check_spectral_time_coverage import coverage, digest  # noqa: E402


def make_plan(source, prior_path):
    from obruxo_data.hashing import canonical_sha256, sha256_bytes
    from obruxo_data.midi import Performance
    from obruxo_training.event_frames import event_frame_spec, note_event_seconds
    parent = json.loads((source / "plan.json").read_text())
    prior = json.loads(prior_path.read_text())
    if (not prior["complete"] or prior["plan"]["source_plan_id"] != canonical_sha256(parent)
            or prior["plan"]["records_sha256"] != digest(source / "records.json")):
        raise ValueError("time-coverage evidence does not match the parent training run")
    for recipe in parent["training_recipes"]:
        note_event_seconds(Performance.from_dict(recipe["performance"]))
    paths = [Path(__file__), Path(__file__).with_name("check_spectral_time_coverage.py")]
    paths += [RESEARCH / "training/obruxo_training" / name for name in
              ("event_frames.py", "temporal_proxy.py", "temporal_proxy_data.py", "spectral.py", "paired_spectral.py")]
    paths.append(RESEARCH / "data_generation/obruxo_data/render/vita.py")
    return {"format": "event_spectral_frame_preflight_v1", "parent_plan_id": canonical_sha256(parent),
            "prior_evidence_sha256": digest(prior_path), "parent_records_sha256": digest(source / "records.json"),
            "source": {path.name: sha256_bytes(path.read_bytes().replace(b"\r\n", b"\n")) for path in paths},
            "sampling": event_frame_spec(), "pairs": prior["plan"]["pairs"], "codec": parent["codec"], "seed": parent["config"]["seed"],
            "limits": {"qualification_attempts": 36, "wall_seconds": 900, "process_tree_bytes": 8 * 1024**3, "archive_bytes": 64 * 1024**2},
            "gates": {"native_all_accepted_and_historical_exact": True, "repeat_pairs": 12, "full_envelopes_exact": True,
                      "all_spectral_responses_retained": "sampled_mean_change>1e-7_when_full_mean_change>1e-7",
                      "projected_1024_record_archive_bytes": 2 * 1024**3},
            "optimizer_updates": 0, "test_split_used": False, "qualification_changed": False}


def check_pair(pair, parent_rows, session, output, seed):
    import numpy as np
    import torch
    from obruxo_data.hashing import canonical_sha256
    from obruxo_data.midi import Performance
    from obruxo_training.codec import PilotTarget
    from obruxo_training.event_frames import sample_event_frames
    from obruxo_training.paired_spectral import require_aligned_pairs
    from obruxo_training.proxy import performance_inputs
    from obruxo_training.spectral import SpectralTargets, target_bytes
    from obruxo_training.temporal_proxy import collate_frames, full_times
    from obruxo_training.trainer import atomic_torch_save
    recipes, hashes, audio = pair["recipes"], [], []
    performance = Performance.from_dict(recipes[0]["performance"])
    for index in (0, 1, 0):
        rendered, receipt = session.render(PilotTarget.from_dict(recipes[index]["target"]), performance, "qualification",
                                           f"{pair['pair_id']}:{len(hashes)}")
        if rendered is None:
            raise RuntimeError("native preflight request rejected")
        expected = parent_rows[canonical_sha256(recipes[index])]["receipt"]["qa"]["audio_float32_sha256"]
        actual = receipt["qa"]["audio_float32_sha256"]
        if actual != expected:
            raise ValueError("historical training audio changed")
        hashes.append(actual)
        if len(audio) < 2:
            audio.append(rendered.copy())
    measured = SpectralTargets()(torch.from_numpy(np.stack(audio)).permute(0, 2, 1))
    records, files = [], []
    for index, recipe in enumerate(recipes):
        full = {key: tuple(value[index:index + 1] for value in values) if isinstance(values, tuple) else values[index:index + 1]
                for key, values in measured.items()}
        sampled, times = sample_event_frames(full, performance, torch.Generator().manual_seed(seed + pair["pair_id"]))
        if not torch.equal(sampled["envelope"], full["envelope"][..., full["envelope_mask"][0]]):
            raise ValueError("full envelope was not preserved")
        record = {"measurements": sampled, "times": times, "performance": performance_inputs(performance), "target": recipe["target"],
                  "pair_id": pair["pair_id"], "source_state_id": recipe["source_state_id"], "recipe_id": canonical_sha256(recipe)}
        records.append(record)
        path = output / f"{pair['pair_id']:04d}-{index}.pt"
        atomic_torch_save(path, record)
        files.append({"file": path.name, "sha256": digest(path), "serialized_bytes": path.stat().st_size,
                      "replay_bytes": target_bytes(sampled) + sum(value.numel() * value.element_size() for value in times)
                                      + record["performance"].numel() * 4})
    require_aligned_pairs(records)
    sampled, times = collate_frames(records)
    measurements = coverage(measured, sampled, full_times(measured), times)
    retained = all(row["sampled_mean_change"] > 1e-7 for row in measurements[:3] if row["full_mean_change"] > 1e-7)
    return {"group": pair["group"], "pair_id": pair["pair_id"], "coverage": measurements, "files": files,
            "spectral_responses_retained": retained, "full_envelopes_exact": True, "pair_times_exact": True,
            "audio_sha256": hashes, "repeat_exact": hashes[0] == hashes[2]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--prior-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    import torch
    from obruxo_data.hashing import canonical_sha256
    from obruxo_training.audit import write_json
    from obruxo_training.benchmark import MemoryMonitor
    from obruxo_training.codec import load_codec
    from obruxo_training.temporal_proxy_data import NativeSession
    torch.set_num_threads(2)
    plan = make_plan(args.source, args.prior_evidence)
    if not args.execute:
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output / "plan.json", plan)
        print(json.dumps({"plan_id": canonical_sha256(plan), "pairs": len(plan["pairs"]), "native_requests": 36}))
        return
    if json.loads((args.output / "plan.json").read_text()) != plan or (args.output / "run.json").exists():
        raise ValueError("plan changed or this diagnostic was already attempted")
    write_json(args.output / "run.json", {"started_unix": time(), "plan_id": canonical_sha256(plan)})
    parent_rows = {row["recipe_id"]: row for row in json.loads((args.source / "records.json").read_text())}
    monitor = MemoryMonitor()
    monitor.start()
    session = NativeSession(args.output, plan, load_codec(plan["codec"]), monitor)
    rows, failure = [], None
    try:
        for pair in plan["pairs"]:
            rows.append(check_pair(pair, parent_rows, session, args.output, plan["seed"]))
            write_json(args.output / "cases.json", rows)
            session.check_limits()
            print(json.dumps({"completed_pairs": len(rows), "retained": rows[-1]["spectral_responses_retained"]}), flush=True)
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        monitor.close()
        elapsed = time() - session.started
        serialized = max((record["serialized_bytes"] for row in rows for record in row["files"]), default=0)
        replay = max((record["replay_bytes"] for row in rows for record in row["files"]), default=0)
        parent_overhead = sum(path.stat().st_size for path in args.source.rglob("*") if path.is_file() and path.parent.name != "records")
        projected = 1024 * serialized + parent_overhead + 10 * 1024**2
        archive_bytes = sum(path.stat().st_size for path in args.output.rglob("*") if path.is_file())
        resource_ok = (monitor.errors == 0 and monitor.peak_bytes is not None and monitor.peak_bytes <= plan["limits"]["process_tree_bytes"]
                       and elapsed <= plan["limits"]["wall_seconds"] and archive_bytes + 1024**2 <= plan["limits"]["archive_bytes"])
        complete = len(rows) == 12 and failure is None
        passed = complete and resource_ok and projected <= plan["gates"]["projected_1024_record_archive_bytes"] and all(
            row["spectral_responses_retained"] and row["repeat_exact"] and row["full_envelopes_exact"] and row["pair_times_exact"] for row in rows)
        summary = {"complete": complete, "failure": failure, "plan_id": canonical_sha256(plan), "plan": plan, "cases": rows,
                   "preflight_passed": passed, "native_requests": sum(row["event"] == "attempt" for row in session.events),
                   "elapsed_seconds": elapsed, "peak_process_tree_rss_bytes": monitor.peak_bytes, "rss_sampling_errors": monitor.errors,
                   "within_resource_limits": resource_ok, "retained_archive_bytes_before_summary": archive_bytes,
                   "maximum_record_bytes": serialized, "maximum_replay_record_bytes": replay,
                   "conservative_replay_capacity": min(512, 512 * 1024**2 // replay) if replay else 0,
                   "projected_1024_record_archive_bytes": projected, "projection_uses_observed_maximum_not_guaranteed_future_size": True,
                   "optimizer_updates": 0, "qualified_for_inverse_training": False, "renderer_identity": session.renderer_identity,
                   "test_split_used": False, "audio_archived": False}
        write_json(args.output / "summary.json", summary)
    if not passed:
        raise RuntimeError("event-frame preflight did not pass; inspect saved evidence")


if __name__ == "__main__":
    main()

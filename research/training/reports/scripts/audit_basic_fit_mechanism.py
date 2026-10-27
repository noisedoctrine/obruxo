"""Audit saved note evidence, observation counts and paired native probes without training."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

REPOSITORY = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(REPOSITORY / "research/training"), str(REPOSITORY / "research/data_generation")]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    from obruxo_training.benchmark import MemoryMonitor
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fit", "features", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    implementation = subprocess.check_output(["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"], text=True).strip()
    monitor, started = MemoryMonitor(), perf_counter()
    monitor.start()
    try:
        import torch
        from obruxo_data.hashing import canonical_sha256
        from obruxo_training.audit import write_json
        from obruxo_training.basic_fit_mechanism import exposures, joint_probes, pooled_pitch
        from obruxo_training.conditioned_short_audit import cached_features, feature_bank
        torch.set_num_threads(2)
        plan = json.loads((args.fit / "plan.json").read_bytes())
        verified = json.loads((args.fit / "verification.json").read_bytes())
        if not verified["verified"] or verified["plan_id"] != canonical_sha256(plan) or digest(args.features) != plan["input_cache_sha256"]:
            raise ValueError("mechanism audit requires the verified fit and its exact feature cache")
        evidence = {name: digest(args.fit / name) for name in ("plan.json", "probes.jsonl", "finals.jsonl")}
        if any(value != verified["evidence_sha256"].get(name) for name, value in evidence.items()):
            raise ValueError("saved evidence changed after independent verification")
        args.output.mkdir(parents=True)
        write_json(args.output / "plan.json", {"implementation_commit": implementation, "fit_plan_id": verified["plan_id"], "evidence_sha256": evidence,
                                               "feature_cache_sha256": digest(args.features), "new_native_requests": 0, "optimizer_applications": 0,
                                               "limits": {"wall_seconds": 300, "process_tree_bytes": 4 * 1024**3, "archive_bytes": 8 * 1024**2}})
        bank, seen, notes = feature_bank(args.features), set(), []
        for cell in plan["fit_audit_cells"]:
            recipe = cell["recipe"]
            if cell["training_index"] is None or cell["input_view"] != 1 or recipe["state_id"] in seen:
                continue
            seen.add(recipe["state_id"])
            pitches = {event["data"][0] for event in recipe["performances"][1]["events"] if event["kind"] == "note_on" and event["data"][1] > 0}
            if len(pitches) != 1:
                raise ValueError("held-note audit requires a single reference pitch")
            notes.append({"training_index": cell["training_index"], "source_midi": next(iter(pitches)),
                          **pooled_pitch(cached_features(bank, recipe, 1)["pitch"])})
        if len(notes) != 12:
            raise ValueError("all twelve independent held-note inputs are required")
        def rows(name):
            return [json.loads(line) for line in (args.fit / name).read_text(encoding="utf-8").splitlines()]
        comparisons = joint_probes(plan, rows("probes.jsonl"), rows("finals.jsonl"))
        if any(row["patches"] != 12 for row in comparisons.values()):
            raise ValueError("both wider panels must retain all twelve patches")
        result = {"fit_plan_id": verified["plan_id"], "pooled_pitch_cases": notes, "scheduled_exposures": exposures(plan), "paired_probe_comparison": comparisons,
                  "pooled_note_matches_source": sum(row["note_top_midi"] == row["source_midi"] for row in notes),
                  "new_native_requests": 0, "optimizer_applications": 0, "qualified_for_release": False,
                  "claim": "saved_training_diagnostics_only; pooled_note_summary_not_full_transcription; best_probe_selection_is_oracle_not_a_deployable_model"}
    finally:
        monitor.close()
    elapsed = perf_counter() - started
    within = bool(elapsed < 300 and monitor.peak_bytes and monitor.peak_bytes < 4 * 1024**3 and not monitor.errors)
    result.update(elapsed_seconds=elapsed, peak_process_tree_rss_bytes=monitor.peak_bytes, rss_sampling_errors=monitor.errors, within_resource_limits=within)
    write_json(args.output / "summary.json", result)
    if sum(path.stat().st_size for path in args.output.iterdir() if path.is_file()) > 8 * 1024**2:
        raise ValueError("mechanism audit exceeds its evidence cap")
    print({key: result[key] for key in ("pooled_note_matches_source", "elapsed_seconds", "peak_process_tree_rss_bytes", "within_resource_limits")})
    return 0 if within else 1


if __name__ == "__main__":
    raise SystemExit(main())

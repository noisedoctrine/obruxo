"""Replay the frozen CIR/CLAP comparison with the local Vital JEPA V7 adapter."""

import argparse
from concurrent.futures import ProcessPoolExecutor, wait
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from time import perf_counter

from frozen_retrieval.rendering import accepted_receipts, render_dataset


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
SOURCE = OUTPUTS / "issue152-validation-train-midi-v3-50"
RUN = OUTPUTS / "issue167-v7"
DATASET = RUN / "dataset"
STATUS = RUN / "status.json"
LOG = RUN / "run.log"
RENDERER = ROOT.parents[1] / "data_generation/configs/renderer_full_corpus.yaml"
CHECKPOINTS = ROOT / "local_checkpoints"
MODELS = {"vital_jepa_v7": "clip", "cir_ast": "official_clip", "clap_music": "official_clip"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def status(phase, completed, total, failures=0, *, state="running"):
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_json(STATUS, {"phase": phase, "completed": completed, "total": total, "failures": failures,
                        "updated_at": timestamp, "state": state, "pid": os.getpid()})
    print(f"{timestamp} {phase} {completed:,}/{total:,}, {failures:,} failures ({state})", flush=True)


def _render_shard(plugin, ids, journal, stage_file):
    def progress(done, failed):
        write_json(stage_file, {"completed": done, "failures": failed})

    return render_dataset(DATASET, RENDERER, plugin_path=plugin, batch_size=8, request_ids=set(ids),
                          progress=progress, receipt_path=journal)


def recover_render_journals(shard_dir):
    journal = DATASET / "render_receipts.private.jsonl"
    rows = {row["request_id"]: row for row in read_rows(journal)}
    initial = sum(row["accepted"] for row in rows.values())
    for source in sorted(shard_dir.glob("receipts-*.jsonl")):
        with source.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.endswith("\n"):
                    break  # A stopped worker may leave an incomplete final append.
                row = json.loads(line)
                key = row["request_id"]
                if row["accepted"] or key not in rows:
                    rows[key] = row
    scratch = journal.with_suffix(".tmp")
    with scratch.open("w", encoding="utf-8", newline="\n") as out:
        for row in rows.values():
            out.write(json.dumps(row, separators=(",", ":")) + "\n")
    scratch.replace(journal)
    return sum(row["accepted"] for row in rows.values()) - initial


def _parallel_render(plugin, selected, plan, processes):
    journal = DATASET / "render_receipts.private.jsonl"
    previous = accepted_receipts(DATASET)
    remaining = [row["request_id"] for row in plan if row["request_id"] in selected and row["request_id"] not in previous]
    if not remaining:
        return {"render_processes": processes, "worker_results": [], "prior_accepted": len(previous)}
    shard_dir = RUN / f"render-workers-{os.getpid()}"
    shard_dir.mkdir()
    shards = [(remaining[index::processes], shard_dir / f"receipts-{index}.jsonl", shard_dir / f"progress-{index}.json")
              for index in range(min(processes, len(remaining)))]
    with ProcessPoolExecutor(max_workers=len(shards)) as pool:
        jobs = [pool.submit(_render_shard, plugin, *shard) for shard in shards]
        pending = set(jobs)
        last_progress = None
        while pending:
            _, pending = wait(pending, timeout=5)
            snapshots = [read_json(progress) for _, _, progress in shards if progress.is_file()]
            progress = (len(previous) + sum(row["completed"] for row in snapshots), sum(row["failures"] for row in snapshots))
            if progress != last_progress:
                status("Render", progress[0], len(selected), progress[1])
                last_progress = progress
        results = [job.result() for job in jobs]
    temporary = journal.with_suffix(".tmp")
    with temporary.open("wb") as dest:
        for source in (journal, *(receipt_path for _, receipt_path, _ in shards)):
            if source.is_file():
                with source.open("rb") as stream:
                    shutil.copyfileobj(stream, dest)
    temporary.replace(journal)
    return {"render_processes": processes, "worker_results": results, "prior_accepted": len(previous)}


def render(plugin, processes):
    DATASET.mkdir(parents=True, exist_ok=True)
    summary = read_json(SOURCE / "dataset_summary.private.json")
    for name in (*summary["manifest_sha256"], "dataset_summary.private.json",
                 "episodes_qa_qualified.private.jsonl", "candidate_sets.render_qualified.private.json"):
        target = DATASET / name
        if not target.exists():
            shutil.copyfile(SOURCE / name, target)
    sample = read_json(OUTPUTS / "model-efficiency-benchmark/sample.private.json")
    pairs = {(row["target_state_id"], row["query_midi_id"]) for row in read_rows(DATASET / "episodes_qa_qualified.private.jsonl")
             if row["condition"] == "cross_midi_common_gallery"}
    plan = read_rows(DATASET / "render_plan.private.jsonl")
    cohort_ids = {row["request_id"] for row in plan if (row["state_id"], row["midi_id"]) in pairs}
    sample_ids = {row["request_id"] for row in sample["records"]}
    selected = cohort_ids | sample_ids
    original = accepted_receipts(SOURCE)
    if len(cohort_ids) != 6450 or len(sample_ids) != 128 or not selected <= original.keys():
        raise ValueError("Frozen cohort or resource sample differs from the original accepted requests")
    status("Render", 0, len(selected))
    if processes == 1:
        result = render_dataset(DATASET, RENDERER, plugin_path=plugin, request_ids=selected,
                                progress=lambda done, failed: status("Render", done, len(selected), failed))
    else:
        result = _parallel_render(plugin, selected, plan, processes)
    receipts = accepted_receipts(DATASET)
    missing = selected - receipts.keys()
    matches = sum(row["audio_sha256"] == original[key]["audio_sha256"] for key, row in receipts.items() if key in selected)
    write_json(RUN / "render-comparison.json", {"cohort_requests": len(cohort_ids), "resource_requests": len(sample_ids),
               "selected_requests": len(selected), "accepted_requests": len(selected) - len(missing),
               "original_sha256_matches": matches, "qa": result})
    status("Render", len(selected), len(selected), len(missing), state="incomplete" if missing else "complete")
    if missing:
        raise ValueError(f"{len(missing)} original assignments lack a new QA-accepted render")


def _speed_shard(plugin, requests, output):
    from obruxo_data.render.batch import run_batch
    from obruxo_data.render.vita import VitalRenderer
    from frozen_retrieval.rendering import _accepted

    renderer = VitalRenderer.from_config(RENDERER, plugin_path=plugin)
    for request in requests:
        if request.renderer_id != renderer.renderer_id:
            raise ValueError("Timing probe renderer differs from the frozen manifest")
    run_batch(renderer, requests, output, workers=1, resume=False)
    return os.getpid(), {request.request_id: _accepted(read_json(output / f"{request.request_id}.json"))[0] for request in requests}


def render_speed(plugin, process_counts, clips_per_band):
    from frozen_retrieval.rendering import _plan_batches

    sample = read_json(OUTPUTS / "model-efficiency-benchmark/sample.private.json")["records"]
    bands = dict.fromkeys(row["duration_band"] for row in sample)
    chosen = [row for band in bands for row in [r for r in sample if r["duration_band"] == band][:clips_per_band]]
    ids = {row["request_id"] for row in chosen}
    source_root = Path(read_json(SOURCE / "dataset_summary.private.json")["source_root"])
    by_id = {request.request_id: request for _, batch in _plan_batches(SOURCE / "render_plan.private.jsonl",
             SOURCE / "performances.private.jsonl", source_root, 32, request_ids=ids) for request in batch}
    if len(by_id) != 4 * clips_per_band or max(process_counts) > len(by_id):
        raise ValueError("Timing probe requires balanced duration bands and at least one request per process")
    requests = [by_id[row["request_id"]] for row in chosen]
    results = {}
    with TemporaryDirectory(prefix="timing-", dir=RUN) as folder:
        for workers in process_counts:
            started = perf_counter()
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_speed_shard, plugin, requests[index::workers], Path(folder) / f"{workers}-{index}")
                           for index in range(workers)]
                outcomes = [job.result() for job in futures]
            elapsed = perf_counter() - started
            accepted = sum(sum(rows.values()) for _, rows in outcomes)
            actual_processes = len({pid for pid, _ in outcomes})
            results[str(workers)] = {"clips": len(requests), "accepted": accepted, "wall_seconds": elapsed,
                                     "accepted_per_second": accepted / elapsed, "actual_processes": actual_processes}
            print(f"{actual_processes}/{workers} processes: {accepted}/{len(requests)} accepted in {elapsed:.2f}s", flush=True)
    name = "render-speed-scaling.json" if len(process_counts) > 2 else "render-speed.json"
    write_json(RUN / name, results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("render", "render-speed"))
    parser.add_argument("--plugin", type=Path, default=Path("C:/Program Files/Common Files/VST3/Vital.vst3"))
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--timing-processes", nargs="+", type=int, default=(1, 4))
    parser.add_argument("--clips-per-band", type=int, default=2)
    args = parser.parse_args()
    if args.processes < 1:
        parser.error("--processes must be positive")
    if args.stage == "render-speed":
        if min(args.timing_processes) < 1 or not 1 <= args.clips_per_band <= 32:
            parser.error("Timing process counts must be positive and clips per band must be in 1–32")
        render_speed(args.plugin, args.timing_processes, args.clips_per_band)
        return
    try:
        render(args.plugin, args.processes)
    except Exception:
        current = read_json(STATUS) if STATUS.exists() else {"phase": "Render", "completed": 0, "total": 0, "failures": 0}
        status(current["phase"], current["completed"], current["total"], current["failures"], state="failed")
        raise


if __name__ == "__main__":
    main()

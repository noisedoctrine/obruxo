"""Replay the original native worker prefix for a missing Issue 187 training clip."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import run_issue187_surrogates as experiment
from obruxo_training.surrogate_pilot import native_session


def recover(source, output, label):
    plan = experiment.read_json(source / "plan.json")["plan"]
    stage = label.split("-", 1)[0]
    recipes = {row["pair_id"]: row for row in plan["stages"][stage]["pairs"]}
    outcomes = [row for row in experiment.read_rows(source / "renders.jsonl") if row["event"] == "outcome"]
    index = next(i for i in reversed(range(len(outcomes))) if outcomes[i]["label"] == label)
    expected = outcomes[index]
    prefix = []
    for entry in reversed(outcomes[:index + 1]):
        if entry.get("worker_pid") != expected["worker_pid"]:
            continue
        prefix.append(entry)
        if entry["worker_job"] == 1:
            break
    prefix.reverse()
    if [r["worker_job"] for r in prefix] != list(range(1, expected["worker_job"] + 1)):
        raise ValueError("original worker prefix is incomplete")
    comparisons = replay(output, recipes, stage, prefix, {label}, output / "renders.jsonl")
    experiment.baseline.write_json(output / f"{label.replace(':', '-')}-replay.json", comparisons)
    return comparisons[-1]["matches"]


def replay(output, recipes, stage, prefix, wanted, ledger):
    fields = dict(experiment.ROLES)
    jobs = []
    base = experiment.baseline
    for entry in prefix:
        pair_id, role = entry["label"].split(":")
        row = recipes[pair_id]
        jobs.append((entry["label"], base.PilotTarget.from_dict(row[fields[role]]),
                     experiment.Performance.from_dict(row["performance"])))
    codec = base.SurrogateCodec(base.PilotCodec(), expanded=stage == "expanded")
    output.mkdir(parents=True, exist_ok=True)
    comparisons = []
    with native_session(ledger.parent, codec, 1) as renderer:
        for job, prior in zip(jobs, prefix, strict=True):
            (wave, receipt), = renderer([job])
            qa = receipt.get("renderer_qa", {})
            comparison = {
                "label": job[0], "original_worker_job": prior["worker_job"],
                "original_sha256": prior["qa"]["audio_float32_sha256"],
                "replayed_sha256": qa.get("audio_float32_sha256"),
                "original_basic_session": prior["renderer_qa"].get("basic_session"),
                "replayed_basic_session": qa.get("basic_session"),
                "provenance_matches": prior["provenance"] == receipt.get("provenance"),
            }
            comparison["matches"] = (
                wave is not None and comparison["provenance_matches"]
                and comparison["original_sha256"] == comparison["replayed_sha256"]
            )
            comparisons.append(comparison)
            if comparison["matches"] and job[0] in wanted:
                experiment.require_audio(wave, prior, job[0])
                path = output / f"{job[0].replace(':', '-')}.npy"
                temporary = path.with_suffix(".tmp")
                with temporary.open("wb") as stream:
                    np.save(stream, wave, allow_pickle=False)
                temporary.replace(path)
    return comparisons


def recover_missing(source, output, existing_latents):
    """Replay needed worker prefixes once, leaving the verified latent cache untouched."""
    plan = experiment.read_json(source / "plan.json")["plan"]
    frozen = experiment.read_json(experiment.PROXY_SOURCE / "plan.json")["plan"]["training"]["pair_receipts"]
    accepted = {row["pair_id"] for row in frozen if row["accepted"]}
    recipes_by_stage = {stage: {r["pair_id"]: r for r in plan["stages"][stage]["pairs"]}
                        for stage in ("basic", "expanded")}
    wanted = set()
    for stage, recipes in recipes_by_stage.items():
        for pair_id in accepted.intersection(recipes):
            if (existing_latents / f"train-{stage}-{pair_id}.pt").is_file():
                continue
            wanted.update(f"{pair_id}:{role}" for role, _ in experiment.ROLES
                          if not (output / f"{pair_id}-{role}.npy").is_file())
    sessions, active, latest = [], {}, {}
    for row in experiment.read_rows(source / "renders.jsonl"):
        if row["event"] != "outcome" or "worker_job" not in row:
            continue
        pid, job = row["worker_pid"], row["worker_job"]
        if job == 1:
            active[pid] = len(sessions)
            sessions.append([])
        if pid in active:
            session = active[pid]
            sessions[session].append(row)
            if row["label"] in wanted:
                latest[row["label"]] = (session, len(sessions[session]))
    grouped = {}
    for label, (session, end) in latest.items():
        grouped.setdefault(session, {})[label] = end

    def execute(item):
        session, targets = item
        prefix = sessions[session][:max(targets.values())]
        stage = prefix[-1]["label"].split("-", 1)[0]
        if [r["worker_job"] for r in prefix] != list(range(1, len(prefix) + 1)):
            raise ValueError(f"incomplete worker prefix for {list(targets)}")
        worker_output = output / f"worker-{session}"
        worker_output.mkdir(exist_ok=True)
        results = replay(output, recipes_by_stage[stage], stage, prefix, targets, worker_output / "renders.jsonl")
        experiment.baseline.write_json(worker_output / "comparisons.json", results)
        return [r for r in results if r["label"] in targets]

    output.mkdir(parents=True, exist_ok=True)
    results, errors = [], []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [(item, pool.submit(execute, item)) for item in grouped.items()]
        for item, future in futures:
            try:
                results.extend(future.result())
            except (KeyError, OSError, RuntimeError, ValueError) as exc:
                errors.append({"session": item[0], "reason": str(exc)})
    missing = sorted(wanted - {r["label"] for r in results if r["matches"]})
    experiment.baseline.write_json(output / "recovery.json", {
        "planned": len(wanted), "recovered": len(wanted) - len(missing), "missing": missing,
        "errors": errors, "results": results,
    })
    return not missing


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=experiment.baseline.TRAINING_SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--label")
    mode.add_argument("--existing-latents", type=Path)
    args = parser.parse_args()
    result = (recover(args.source, args.output, args.label) if args.label
              else recover_missing(args.source, args.output, args.existing_latents))
    raise SystemExit(0 if result else 1)

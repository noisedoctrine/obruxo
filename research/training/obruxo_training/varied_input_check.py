"""Costed native QA/repeat check of varied training observations; no learning."""

from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from time import time

from obruxo_data.hashing import canonical_sha256, sha256_bytes
from obruxo_data.render.base import RenderRequest

from .audit import write_json
from .benchmark import MemoryMonitor
from .input_performances import VariedSampler
from .profiled_sampler import training_sampler
from .stream import NativeRendererFactory, render_isolated


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def append(path, row):
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def make_plan(pool, qualification_path, bootstrap_root, bootstrap_audit):
    from .native_fit import require_qualification
    from .trainer import TrainConfig
    qualification = json.loads(qualification_path.read_text())
    require_qualification(qualification, preroll_frames=32768)
    prior = json.loads((bootstrap_root / "summary.json").read_text())
    old_plan = json.loads((bootstrap_root / "plan.json").read_text())
    audit = json.loads(bootstrap_audit.read_text())
    required = {"summary.json", "plan.json", "fit-000/renders.jsonl"}
    if (not prior["complete"] or not prior["preflight_passed"] or old_plan["pool_id"] != pool.pool_id
            or prior["plan_id"] != canonical_sha256(old_plan) or audit["plan_id"] != prior["plan_id"]
            or not audit["model_optimizer_batch_and_global_rng_exact"] or audit["replayed_updates"] != 16000
            or audit["native_renders_added"] != 0 or not required <= audit["evidence_sha256"].keys()):
        raise ValueError("varied input check requires the audited full initialization")
    for name in required:
        if sha256_bytes((bootstrap_root / name).read_bytes()) != audit["evidence_sha256"][name]:
            raise ValueError("audited bootstrap evidence changed")
    sampler = VariedSampler(pool)
    entries = {entry.state_id: entry for entry in pool.entries}
    groups = defaultdict(list)
    for sequence in range(1, sampler.epoch_size, 2):
        recipe = sampler.recipe_at(sequence)
        groups[(entries[recipe.state_id].target.categorical, recipe.input_pattern)].append(recipe)
    if len(groups) != 18:
        raise ValueError("native preflight needs all six categories and three alternate patterns")
    # The loudest source in each cell stresses chords/unison without changing legal controls.
    selected = [min(group, key=lambda recipe: (-entries[recipe.state_id].target.continuous[0], recipe.state_id))
                for _, group in sorted(groups.items())]
    requests = []
    for selected_recipe in selected:
        for sequence in (selected_recipe.sequence_id - 1, selected_recipe.sequence_id):
            recipe = sampler.recipe_at(sequence)
            request = recipe.render_request()
            for repeat in range(2):
                requests.append({"kind": "varied", "state_id": recipe.state_id, "sequence_id": sequence,
                                 "pattern": recipe.input_pattern, "category": list(entries[recipe.state_id].target.categorical),
                                 "repeat": repeat, "recipe_id": recipe.recipe_id, "request_id": request.request_id,
                                 "request": request.to_dict()})
    old_sampler = training_sampler(pool, TrainConfig(**old_plan["config"]))
    receipts = {row["sequence_id"]: row for row in rows(bootstrap_root / "fit-000/renders.jsonl")}
    for sequence in range(4):
        recipe = old_sampler.recipe_at(sequence)
        receipt, request = receipts[sequence], recipe.render_request()
        if receipt["status"] != "accepted" or receipt["recipe_id"] != recipe.recipe_id or receipt["provenance"]["request_id"] != request.request_id:
            raise ValueError("bootstrap parity request changed")
        requests.append({"kind": "historical_parity", "sequence_id": sequence, "request_id": request.request_id,
                         "request": request.to_dict(), "expected_audio_sha256": receipt["qa"]["audio_float32_sha256"]})
    census = Counter()
    pitches = set()
    for sequence in range(sampler.epoch_size):
        recipe = sampler.recipe_at(sequence)
        census[recipe.input_pattern] += 1
        pitches.update(event.data[0] for event in recipe.render_request().performance.canonical_events() if event.kind.value == "note_on")
    names = ("varied_input_check.py", "input_performances.py", "profiled_sampler.py", "stream.py")
    return {"experiment": "varied_basic_input_native_check_v1", "pool_id": pool.pool_id, "manifest_id": pool.manifest.manifest_id,
            "observation_contract": sampler.observation_contract, "renderer_identity": qualification["renderer_identity"],
            "qualification_sha256": sha256_bytes(qualification_path.read_bytes()),
            "bootstrap_plan_id": prior["plan_id"], "bootstrap_audit_sha256": sha256_bytes(bootstrap_audit.read_bytes()),
            "source": {name: sha256_bytes(Path(__file__).with_name(name).read_bytes().replace(b"\r\n", b"\n")) for name in names},
            "requests": requests, "planned_native_requests": 76, "optimizer_updates": 0,
            "full_sampler_census": {"records": sampler.epoch_size, "patterns": dict(census), "distinct_pitches": sorted(pitches)},
            "selection": "highest_source_level_per_category_and_alternate_pattern_both_views_twice_plus_four_historical_requests",
            "limits": {"wall_seconds": 600, "process_tree_bytes": 4 * 1024**3, "archive_bytes": 64 * 1024**2, "native_attempts": 76},
            "gates": {"all_outcomes_accepted": True, "all_repeat_hashes_exact": True, "all_historical_hashes_exact": True},
            "test_split_used": False, "qualified_for_release": False}


def run_check(plan, output, *, execute=False, resume=False, renderer_factory=None):
    output = Path(output).resolve()
    if output.exists():
        if json.loads((output / "plan.json").read_text()) != plan:
            raise ValueError("frozen varied-input plan changed")
    else:
        output.mkdir(parents=True)
        write_json(output / "plan.json", plan)
    if not execute:
        return {"plan_id": canonical_sha256(plan), "executed": False}
    if (output / "run.json").exists() != resume:
        raise ValueError("existing execution needs explicit resume")
    if not resume:
        write_json(output / "run.json", {"started_unix": time(), "plan_id": canonical_sha256(plan)})
    started = json.loads((output / "run.json").read_text())["started_unix"]
    attempts, outcomes = rows(output / "attempts.jsonl"), rows(output / "renders.jsonl")
    if len(attempts) != len(outcomes) or len(attempts) > len(plan["requests"]):
        raise ValueError("unresolved charged attempt; do not silently repeat a possibly executed native request")
    for index, (attempt, outcome) in enumerate(zip(attempts, outcomes, strict=True)):
        if (attempt["index"] != index or outcome["index"] != index or outcome["status"] != "accepted"
                or attempt["request_id"] != plan["requests"][index]["request_id"] or outcome["request_id"] != attempt["request_id"]
                or {key: outcome["provenance"][key] for key in plan["renderer_identity"]} != plan["renderer_identity"]):
            raise ValueError("cannot resume changed or failed request history")
    monitor = MemoryMonitor()
    monitor.start()
    factory = renderer_factory or NativeRendererFactory(renderer_id=plan["renderer_identity"]["renderer_id"])

    def check():
        if time() - started >= plan["limits"]["wall_seconds"]:
            raise RuntimeError("varied-input original deadline exhausted")
        if (monitor.peak_bytes or 0) > plan["limits"]["process_tree_bytes"]:
            raise RuntimeError("varied-input memory budget exhausted")
        if sum(path.stat().st_size for path in output.rglob("*") if path.is_file()) > plan["limits"]["archive_bytes"]:
            raise RuntimeError("varied-input archive budget exhausted")

    try:
        for index in range(len(outcomes), len(plan["requests"])):
            check()
            if index >= plan["limits"]["native_attempts"]:
                raise RuntimeError("varied-input native attempt budget exhausted")
            spec = plan["requests"][index]
            request = RenderRequest.from_dict(spec["request"])
            if request.request_id != spec["request_id"]:
                raise ValueError("frozen request identity changed")
            # Charge durably in the parent before starting any worker, including failed starts.
            append(output / "attempts.jsonl", {"index": index, "request_id": request.request_id, "started_unix": time()})
            audio, metadata = render_isolated(request, renderer_factory=factory, timeout=min(60., max(.1, plan["limits"]["wall_seconds"] - (time() - started))))
            row = {"index": index, "request_id": request.request_id, "peak_process_tree_rss_bytes": monitor.peak_bytes,
                   "rss_sampling_errors": monitor.errors, **metadata}
            append(output / "renders.jsonl", row)
            outcomes.append(row)
            if audio is None or row["status"] != "accepted":
                raise RuntimeError("varied input failed native audio QA")
            identity = {key: row["provenance"][key] for key in plan["renderer_identity"]}
            if identity != plan["renderer_identity"]:
                raise ValueError("varied input used a different native renderer")
            print(json.dumps({"completed": index + 1, "planned": len(plan["requests"]), "kind": spec["kind"], "elapsed_seconds": time() - started}), flush=True)
            del audio
        check()
        hashes = {row["index"]: row["qa"]["audio_float32_sha256"] for row in outcomes}
        repeats = defaultdict(list)
        for index, spec in enumerate(plan["requests"]):
            if spec["kind"] == "varied":
                repeats[spec["request_id"]].append(hashes[index])
        exact = bool(repeats) and all(len(values) == 2 and values[0] == values[1] for values in repeats.values())
        parity = all(hashes[index] == spec["expected_audio_sha256"] for index, spec in enumerate(plan["requests"]) if spec["kind"] == "historical_parity")
        summary = {"plan_id": canonical_sha256(plan), "complete": True, "accepted": len(outcomes),
                   "all_repeat_hashes_exact": exact, "all_historical_hashes_exact": parity, "repeat_pairs": len(repeats),
                   "elapsed_seconds": time() - started,
                   "peak_process_tree_rss_bytes": max([monitor.peak_bytes or 0, *[row["peak_process_tree_rss_bytes"] or 0 for row in outcomes]]),
                   "rss_sampling_errors": monitor.errors + max([0, *[row["rss_sampling_errors"] for row in outcomes]]),
                   "archive_bytes": sum(path.stat().st_size for path in output.rglob("*") if path.is_file()),
                   "optimizer_updates": 0, "test_split_used": False, "qualified_for_release": False}
        summary["passed"] = (exact and parity and monitor.peak_bytes is not None and summary["rss_sampling_errors"] == 0
                             and summary["peak_process_tree_rss_bytes"] <= plan["limits"]["process_tree_bytes"])
        write_json(output / "summary.json", summary)
        write_json(output / "status.json", {"complete": True, "passed": summary["passed"]})
        return summary
    except BaseException as error:
        write_json(output / "status.json", {"complete": False, "error_type": type(error).__name__, "elapsed_seconds": time() - started})
        raise
    finally:
        monitor.close()

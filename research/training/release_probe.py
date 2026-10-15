"""Frozen development experiment: replace only release using training pitch neighbors.

Heavy imports stay inside functions so native spawn workers do not load Torch twice.
This runner never trains, exports a model, or evaluates the test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_path(root, artifact):
    path = root / artifact["path"]
    if file_sha(path) != artifact["sha256"]:
        raise ValueError(f"frozen artifact changed: {artifact['path']}")
    return path


def require_training_replay(replay, manifest):
    states = [item["state_id"] for item in replay]
    if not states or len(set(states)) != len(states):
        raise ValueError("probe requires exactly one retained view per training state")
    if any(state not in manifest.by_state or manifest.by_state[state].split != "train" for state in states):
        raise ValueError("retrieval candidates must come from the training split only")


def nearest_pitch(features, replay):
    import torch

    # Float64 and first-index tie breaking match the predeclared train-only probe.
    bank = torch.stack([item["features"]["pitch"] for item in replay]).double()
    distances = (bank - features["pitch"].double()).square().mean(dim=1)
    if not torch.isfinite(distances).all():
        raise ValueError("pitch distances must be finite")
    index = int(distances.argmin())
    return index, float(distances[index])


def replace_release(codec, target, release):
    from obruxo_training.codec import PilotTarget

    changed = codec.canonical_target(PilotTarget((*target.continuous[:-1], release), target.categorical))
    if changed.continuous[:-1] != target.continuous[:-1] or changed.categorical != target.categorical:
        raise ValueError("release intervention changed another target")
    before, after = (json.loads(codec.decode(value)) for value in (target, changed))
    before["settings"]["env_1_release"] = after["settings"]["env_1_release"]
    if before != after:
        raise ValueError("release intervention changed another preset field")
    return changed


def run(args):
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "research/data_generation"))
    import numpy as np
    import torch

    from obruxo_data.hashing import canonical_sha256
    from obruxo_data.render.base import RenderRequest
    from obruxo_training.audit import write_json
    from obruxo_training.evaluate import _restore_evaluation, audio_distance, curve_statistics, load_evaluation_checkpoint, retrieval_digest
    from obruxo_training.model import batch_features, predicted_targets
    from obruxo_training.performances import initial_bank
    from obruxo_training.pilot import PilotPool, PilotSampler
    from obruxo_training.stream import NativeRendererFactory, render_isolated

    started = perf_counter()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan["experiment"] != "pitch_neighbor_release_intervention_v1":
        raise ValueError("unsupported release experiment")
    for path, digest in plan["runtime_source_sha256"].items():
        checked_path(root, {"path": path, "sha256": digest})
    prior_plan = json.loads(checked_path(root, plan["prior_plan"]).read_text())
    prior_progress = json.loads(checked_path(args.artifact_root, plan["prior_progress"]).read_text())
    prior_renders = [json.loads(line) for line in checked_path(args.artifact_root, plan["prior_renders"]).read_text().splitlines()]
    prior_sources = {(item["case"], item["role"]): item["qa"]["audio_float32_sha256"]
                     for item in prior_renders if item["method"] == "source" and item["status"] == "accepted"}
    pool = PilotPool.load(checked_path(args.artifact_root, prior_plan["pools"]["expanded"]))
    parent = PilotPool.load(checked_path(args.artifact_root, prior_plan["pools"]["one"]))
    pool.expansion_from(parent)
    torch.set_num_threads(4)
    models, replay, identity, frontend, memory_digest = {}, None, None, None, None
    for name in plan["candidate_order"]:
        spec = prior_plan["candidates"][name]
        saved, model, candidate_frontend, _ = load_evaluation_checkpoint(checked_path(args.artifact_root, spec), pool)
        if retrieval_digest(saved["replay"]) != spec["replay_digest"]:
            raise ValueError("candidate replay differs from frozen training evidence")
        require_training_replay(saved["replay"], pool.manifest)
        if frontend is not None and (saved["renderer_identity"] != identity or candidate_frontend.frozen_digest != frontend.frozen_digest
                                     or spec["replay_digest"] != memory_digest):
            raise ValueError("candidate inference conditions differ")
        models[name] = model
        replay, identity, frontend, memory_digest = saved["replay"], saved["renderer_identity"], candidate_frontend, spec["replay_digest"]
    if len(replay) != plan["training_states"] or frontend.frozen_digest != plan["basic_pitch_sha256"]:
        raise ValueError("training population or frozen frontend changed")
    sampler = PilotSampler(pool, split="validation")
    recipes, seen = [], {entry.state_id for entry in parent.entries}
    for sequence in range(sampler.epoch_size):
        recipe = sampler.recipe_at(sequence)
        if recipe.state_id not in seen:
            recipes.append(recipe)
            seen.add(recipe.state_id)
    transfers = [item for item in initial_bank() if item.role == "transfer"]
    cases = [{"case": index, "state_id": recipe.state_id, "recipe_id": recipe.recipe_id,
              "reference_performance_id": recipe.performance_id,
              "transfer_performance_id": transfers[index % len(transfers)].performance_id} for index, recipe in enumerate(recipes)]
    if cases != plan["cases"] or cases != prior_plan["cases"] or [item.recipe_id for item in recipes] != plan["recipe_ids"]:
        raise ValueError("validation population or performance pairs changed")
    checked_path(args.artifact_root, prior_plan["retained_candidate"])
    output, codec = args.output, pool.codec
    rows, renders, completed, prior_elapsed = [], [], [], 0.0
    if args.resume:
        state, rows, renders = _restore_evaluation(output, plan, identity)
        completed, prior_elapsed = state["completed_cases"], state["elapsed_seconds"]
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "plan.json", plan)
        (output / "renders.jsonl").touch()
    reused_cases = len(rows)

    def save_progress(row=None):
        if row is not None:
            index = row["case"]
            path = output / f"case-{index:03d}.json"
            write_json(path, row)
            predictions = {path.name: file_sha(path) for path in output.glob(f"prediction-*-{index:03d}.vital")}
            completed.append({"case": index, "case_sha256": file_sha(path), "predictions": predictions})
        write_json(output / "progress.json", {"plan_sha256": canonical_sha256(plan), "completed_cases": completed,
                   "committed_render_count": len(renders), "renders_sha256": canonical_sha256(renders),
                   "elapsed_seconds": prior_elapsed + perf_counter() - started})

    def render(raw, performance, index, method, role):
        if len(renders) >= plan["maximum_render_attempts"]:
            raise RuntimeError("frozen total render budget exhausted")
        request = RenderRequest(preset_json=raw.decode(), performance=performance, sample_rate=codec.contract.sample_rate,
                                channels=2, tail_seconds=codec.contract.tail_seconds, renderer_id=codec.contract.renderer_id)
        audio, metadata = render_isolated(request, renderer_factory=NativeRendererFactory(), timeout=30)
        record = {"case": index, "method": method, "role": role, "request_id": request.request_id, **metadata}
        renders.append(record)
        with (output / "renders.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if audio is not None:
            if {key: value for key, value in metadata["provenance"].items() if key != "request_id"} != identity:
                raise ValueError("native renderer differs from training")
            if method == "source" and metadata["qa"]["audio_float32_sha256"] != prior_sources[index, role]:
                raise ValueError("source audio changed from the frozen validation evidence")
        return audio

    if not args.resume:
        save_progress()
    write_json(output / "status.json", {"complete": False, "stage": "release_intervention"})
    try:
        for index, recipe in enumerate(recipes):
            if index < reused_cases:
                continue
            request = recipe.render_request()
            source_raw = request.preset_json.encode()
            row = {**cases[index], "methods": {}, "parameters": {}}
            rows.append(row)
            reference = render(source_raw, request.performance, index, "source", "reference")
            if reference is None:
                row["failure"] = "source_reference_failed"
                save_progress(row)
                continue
            features = frontend(reference)
            neighbor_index, distance = nearest_pitch(features, replay)
            neighbor = replay[neighbor_index]
            row["neighbor"] = {"index": neighbor_index, "state_id": neighbor["state_id"], "distance": distance,
                               "release": neighbor["target"]["continuous"][-1]}
            targets = {}
            for name, model in models.items():
                with torch.no_grad():
                    target = predicted_targets(model(batch_features([{"features": features}])), codec)[0]
                filename = f"prediction-{name}-{index:03d}.vital"
                if hashlib.sha256(codec.decode(target)).hexdigest() != prior_progress["completed_cases"][index]["predictions"][filename]:
                    raise ValueError("reinferred baseline differs from prior prediction")
                targets[name] = target
                targets[f"{name}_pitch_release"] = replace_release(codec, target, row["neighbor"]["release"])
            truth = codec.encode(source_raw)
            for name, target in targets.items():
                codec.preset(target).validate().require_valid()
                (output / f"prediction-{name}-{index:03d}.vital").write_bytes(codec.decode(target))
                row["parameters"][name] = {"release": target.continuous[-1], "release_absolute_error": abs(target.continuous[-1] - truth.continuous[-1]),
                                            "unison_correct": target.categorical[0] == truth.categorical[0]}
            targets["source_repeat"] = truth
            transfer = transfers[index % len(transfers)].performance()
            transfer_source = render(source_raw, transfer, index, "source", "transfer")
            for name, target in targets.items():
                scores = {}
                for role, performance, source in (("reference", request.performance, reference), ("transfer", transfer, transfer_source)):
                    if source is None:
                        scores[role] = {"failure": "source_render_failed"}
                        continue
                    candidate = render(codec.decode(target), performance, index, name, role)
                    scores[role] = {"failure": "candidate_render_failed"} if candidate is None else audio_distance(source, candidate)
                row["methods"][name] = scores
            save_progress(row)
            print(json.dumps({"completed_cases": len(rows), "planned_cases": len(cases), "renders": len(renders),
                              "accepted": sum(item["status"] == "accepted" for item in renders)}), flush=True)
        comparisons, parameters, aggregates = {}, {}, {}
        for name in models:
            changed = f"{name}_pitch_release"
            comparisons[name] = curve_statistics(rows, [name, changed], baseline_name=name)["checkpoints"]
        for name in (*[item for baseline in models for item in (baseline, f"{baseline}_pitch_release")], "source_repeat"):
            parameter_rows = [row["parameters"][name] for row in rows if name in row["parameters"]]
            if parameter_rows:
                parameters[name] = {"scored": len(parameter_rows), "planned": len(cases),
                                    "release_mae": float(np.mean([item["release_absolute_error"] for item in parameter_rows])),
                                    "unison_accuracy": float(np.mean([item["unison_correct"] for item in parameter_rows]))}
            aggregates[name] = {}
            for role in ("reference", "transfer"):
                scores = [row["methods"].get(name, {}).get(role, {}) for row in rows]
                valid = [item for item in scores if "mrstft" in item]
                aggregates[name][role] = {"planned": len(cases), "scored": len(valid), "failed_or_unscorable": len(cases) - len(valid),
                                         **{f"mean_{metric}": float(np.mean([item[metric] for item in valid])) if valid else None
                                            for metric in ("mrstft", "rms_relative")}}
        frontend.require_frozen()
        checked_path(args.artifact_root, prior_plan["retained_candidate"])
        summary = {"experiment": plan["experiment"], "plan_sha256": file_sha(args.plan), "comparisons": comparisons,
                   "parameters": parameters, "aggregates": aggregates, "cases": rows, "renderer_identity": identity,
                   "retrieval_sha256": memory_digest, "retrieval_train_states": len(replay), "renders_attempted": len(renders),
                   "renders_accepted": sum(item["status"] == "accepted" for item in renders), "completed_cases_reused": reused_cases,
                   "elapsed_seconds": prior_elapsed + perf_counter() - started, "elapsed_includes_uncommitted_interrupted_work": not args.resume,
                   "render_wall_seconds": sum(item.get("wall_seconds", 0) for item in renders), "audio_files_written": 0,
                   "optimizer_updates": 0, "test_split_evaluated": False, "promotion": "none", "retained_candidate_unchanged": True}
        write_json(output / "summary.json", summary)
        write_json(output / "status.json", {"complete": True, "all_renders_accepted": summary["renders_accepted"] == len(renders)})
    except BaseException as error:
        write_json(output / "status.json", {"complete": False, "stage": "interrupted", "error_type": type(error).__name__})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True, help="repository root holding the ignored training outputs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

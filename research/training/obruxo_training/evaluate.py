"""Real-render diagnostics. Known MIDI stays in the evaluator, never in model inputs."""

from __future__ import annotations

import json
from pathlib import Path
import re
from time import perf_counter

import numpy as np
import torch

from obruxo_data.hashing import canonical_sha256, sha256_bytes
from obruxo_data.render.base import RenderRequest

from .audit import write_json
from .codec import PilotTarget
from .features import AudioFeatures, tensor_digest
from .model import PilotInverse, batch_features, predicted_targets
from .performances import initial_bank
from .pilot import PilotPool, PilotSampler
from .stream import OnlineRenderStream, StreamConfig, render_isolated
from .trainer import TrainConfig, _run_contract, export_model, validate_replay


@torch.no_grad()
def audio_distance(reference: np.ndarray, candidate: np.ndarray) -> dict:
    if reference.shape != candidate.shape or not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("render comparison needs aligned, finite audio")
    left, right = torch.from_numpy(reference.copy()).T, torch.from_numpy(candidate.copy()).T
    scores = []
    for size in (512, 2048, 8192):
        window = torch.hann_window(size)
        spectra = [torch.stft(value, size, hop_length=size // 4, window=window, return_complex=True).abs()
                   for value in (left, right)]
        convergence = (spectra[0] - spectra[1]).norm() / spectra[0].norm().clamp_min(1e-8)
        log_mae = (spectra[0].clamp_min(1e-5).log() - spectra[1].clamp_min(1e-5).log()).abs().mean()
        scores.append({"fft": size, "spectral_convergence": float(convergence), "log_magnitude_mae": float(log_mae)})
    left_rms, right_rms = left.square().mean().sqrt(), right.square().mean().sqrt()
    return {"mrstft": float(np.mean([item["spectral_convergence"] + item["log_magnitude_mae"] for item in scores])),
            "rms_relative": float((left_rms - right_rms).abs() / left_rms.clamp_min(1e-8)), "resolutions": scores}


def nearest_training_target(features: dict, replay: list[dict]) -> PilotTarget:
    # Equal branch weights, fixed scaling, no statistics or candidates from validation/test.
    distances = [sum(float((features[name] - item["features"][name]).square().mean()) for name in ("timbre", "pitch"))
                 for item in replay]
    return PilotTarget.from_dict(replay[int(np.argmin(distances))]["target"])


def load_evaluation_checkpoint(path: Path, pool):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("format") != "obruxo_pilot_training_v1" or saved["step"] < 1:
        raise ValueError("evaluation needs a trained pilot checkpoint")
    contract = saved["run_contract"]
    config = TrainConfig(**contract["training"])
    stream_config = StreamConfig.from_dict(contract["stream"])
    frontend = AudioFeatures(saved["basic_pitch"])
    sampler = PilotSampler(pool, seed=config.seed, policy=config.sampling_policy)
    if contract != _run_contract(pool, sampler, frontend, config, stream_config, contract.get("replay_seed")):
        raise ValueError("evaluation checkpoint belongs to incompatible contracts or a different pool")
    OnlineRenderStream(sampler, 1, config=stream_config, resume=saved["cursor"])
    validate_replay(saved["replay"], sampler, saved["cursor"], config)
    if not saved["replay"]:
        raise ValueError("training-only retrieval memory is empty")
    model = PilotInverse(pool.codec).eval()
    model.load_state_dict(saved["model"], strict=True)
    return saved, model, frontend, sampler


def evaluate_pilot(pool, checkpoint: Path, output: Path, *, count: int = 4, split: str = "validation",
                   renderer_factory=None, timeout: float = 60, progress=None) -> dict:
    saved, model, frontend, sampler = load_evaluation_checkpoint(checkpoint, pool)
    return _evaluate_models(pool, {"model": (saved, model)}, frontend, sampler, output, count=count, split=split,
                            renderer_factory=renderer_factory, timeout=timeout, progress=progress)


def retrieval_digest(replay: list[dict]) -> str:
    # Usage counters change during replay; the candidate audio, features and labels must not.
    return canonical_sha256([{**{key: value for key, value in item.items() if key not in {"features", "uses"}},
                              "features_sha256": tensor_digest(item["features"])} for item in replay])


def evaluate_learning_curve(pool, checkpoints: list[Path], output: Path, *, count: int = 21,
                            renderer_factory=None, timeout: float = 60, progress=None, resume: bool = False) -> dict:
    if not 2 <= len(checkpoints) <= 8:
        raise ValueError("a bounded learning curve requires two through eight checkpoints")
    models, artifacts, first, memory_id, previous_step = {}, {}, None, None, 0
    for checkpoint in checkpoints:
        saved, model, frontend, sampler = load_evaluation_checkpoint(checkpoint, pool)
        if saved["step"] <= previous_step:
            raise ValueError("curve checkpoints must have strictly increasing optimizer steps")
        digest = retrieval_digest(saved["replay"])
        if first is not None and (saved["run_contract"] != first["run_contract"]
                                  or saved["renderer_identity"] != first["renderer_identity"] or digest != memory_id):
            raise ValueError("curve checkpoints must share contracts, renderer and identical training retrieval memory")
        name = f"step_{saved['step']}"
        models[name] = (saved, model)
        artifacts[name] = {"checkpoint": str(checkpoint), "file_sha256": sha256_bytes(checkpoint.read_bytes())}
        first, memory_id, previous_step = saved, digest, saved["step"]
    return _evaluate_models(pool, models, frontend, sampler, output, count=count, split="validation", curve=True,
                            artifacts=artifacts, renderer_factory=renderer_factory, timeout=timeout, progress=progress, resume=resume)


def evaluate_comparison(pool, checkpoints: dict[str, Path], retrieval_checkpoint: Path, output: Path, *, count: int = 21,
                        renderer_factory=None, timeout: float = 60, progress=None, resume: bool = False,
                        checkpoint_pools: dict[str, PilotPool] | None = None, validation_pool: PilotPool | None = None,
                        identity_diagnostics: bool = False, validation_states: list[str] | None = None) -> dict:
    reserved = {"source", "fixed_template", "train_retrieval", "source_repeat"}
    if not 2 <= len(checkpoints) <= 8 or any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", name)
                                           or name in reserved for name in checkpoints):
        raise ValueError("comparison needs two through eight unique, safe, non-reserved model names")
    checkpoint_pools = checkpoint_pools or {}
    if set(checkpoint_pools) - checkpoints.keys():
        raise ValueError("comparison pool names must identify candidate checkpoints")
    extension = None
    if validation_pool is not None:
        proof = validation_pool.expansion_from(pool)
        if count != proof["added_states_by_split"]["validation"]:
            raise ValueError("fresh validation must cover every added validation state exactly once")
        extension = {"policy": "all_added_validation_states_v1", "parent_pool_id": pool.pool_id, "expansion": proof,
                     "excluded_state_ids": [entry.state_id for entry in pool.entries]}
    reference, _, frontend, sampler = load_evaluation_checkpoint(retrieval_checkpoint, pool)
    common_keys = ("contract", "codec", "model", "frontend", "precision")
    models, artifacts = {}, {}
    for name, path in checkpoints.items():
        training_pool = checkpoint_pools.get(name, pool)
        expansion = training_pool.expansion_from(pool) if training_pool.pool_id != pool.pool_id else None
        saved, model, _, _ = load_evaluation_checkpoint(path, training_pool)
        if (any(saved["run_contract"][key] != reference["run_contract"][key] for key in common_keys)
                or saved["renderer_identity"] != reference["renderer_identity"]):
            raise ValueError("comparison checkpoints must share inference contracts and renderer with the retrieval source")
        models[name] = (saved, model)
        artifacts[name] = {"checkpoint": str(path), "file_sha256": sha256_bytes(path.read_bytes()),
                           "training_pool_id": training_pool.pool_id, "pool_expansion": expansion}
    retrieval = {"replay": reference["replay"], "source": {"checkpoint": str(retrieval_checkpoint),
                  "file_sha256": sha256_bytes(retrieval_checkpoint.read_bytes()), "step": reference["step"],
                  "training_pool_id": pool.pool_id}}
    return _evaluate_models(validation_pool or pool, models, frontend, sampler, output, count=count, split="validation", curve=True,
                            comparison=True, artifacts=artifacts, retrieval=retrieval,
                            renderer_factory=renderer_factory, timeout=timeout, progress=progress, resume=resume,
                            validation_extension=extension, identity_diagnostics=identity_diagnostics, validation_states=validation_states)


def _restore_evaluation(output: Path, plan: dict, renderer_identity: dict) -> tuple:
    if json.loads((output / "plan.json").read_text()) != plan:
        raise ValueError("evaluation plan changed; cannot resume")
    state = json.loads((output / "progress.json").read_text())
    if state["plan_sha256"] != canonical_sha256(plan):
        raise ValueError("evaluation progress belongs to a different plan")
    renders = [json.loads(line) for line in (output / "renders.jsonl").read_text().splitlines()]
    if (state["committed_render_count"] > len(renders)
            or canonical_sha256(renders[:state["committed_render_count"]]) != state["renders_sha256"]):
        raise ValueError("committed evaluation render log changed")
    for render in renders:
        identity = {key: value for key, value in render.get("provenance", {}).items() if key != "request_id"}
        if render["status"] == "accepted" and identity != renderer_identity:
            raise ValueError("saved evaluation renderer differs from the training renderer")
    rows = []
    for index, completed in enumerate(state["completed_cases"]):
        path = output / f"case-{index:03d}.json"
        if completed["case"] != index or sha256_bytes(path.read_bytes()) != completed["case_sha256"]:
            raise ValueError("completed evaluation case changed")
        for name, digest in completed["predictions"].items():
            if sha256_bytes((output / name).read_bytes()) != digest:
                raise ValueError("completed evaluation prediction changed")
        row = json.loads(path.read_text())
        if row["case"] != index:
            raise ValueError("completed evaluation case order changed")
        rows.append(row)
    if len(rows) > len(plan["recipe_ids"]):
        raise ValueError("saved evaluation has more cases than its plan")
    return state, rows, renders


def curve_statistics(rows: list[dict], model_names: list[str], baseline_name: str = "train_retrieval") -> dict:
    """Paired state bootstrap; selection is descriptive validation, not a final-test claim."""
    if not rows:
        raise ValueError("curve statistics require planned cases")

    def paired_scores(name):
        scores = []
        for row in rows:
            roles = row["methods"].get(name, {})
            values = [roles.get(role, {}).get("mrstft") for role in ("reference", "transfer")]
            scores.append(float(np.mean(values)) if all(value is not None and np.isfinite(value) for value in values) else None)
        return scores

    baseline = paired_scores(baseline_name)
    summaries = {}
    for name in model_names:
        scores = paired_scores(name)
        paired = [(value, baseline[index]) for index, value in enumerate(scores)
                  if value is not None and baseline[index] is not None]
        valid = [value for value in scores if value is not None]
        comparison = {"planned": len(rows), "paired": len(paired), "mean_delta": None, "ci95": None, "wins": 0, "ties": 0}
        if paired:
            deltas = np.array([value - reference for value, reference in paired])
            indices = np.random.default_rng(0).integers(len(deltas), size=(2000, len(deltas)))
            means = deltas[indices].mean(axis=1)
            comparison.update(mean_delta=float(deltas.mean()), ci95=np.quantile(means, [.025, .975]).tolist(),
                              wins=int((deltas < 0).sum()), ties=int((deltas == 0).sum()))
        summaries[name] = {"planned": len(rows), "paired_roles_scored": len(valid),
                           "selection_eligible": len(valid) == len(rows),
                           "mean_reference_transfer_mrstft": float(np.mean(valid)) if valid else None,
                           f"against_{baseline_name}": comparison}
    eligible = [name for name in model_names if summaries[name]["selection_eligible"]]
    selected = min(eligible, key=lambda name: summaries[name]["mean_reference_transfer_mrstft"]) if eligible else None
    return {"primary": "equal_weight_mean_of_reference_and_transfer_per_state_mrstft", "checkpoints": summaries,
            "selected": selected, "selection_policy": "lowest_complete_validation_mean_earliest_step_breaks_ties",
            "bootstrap": {"unit": "state", "seed": 0, "resamples": 2000, "interval": "percentile_95",
                          "limit": "descriptive_validation_only_not_adjusted_for_checkpoint_selection"}}


def _evaluate_models(pool, models, frontend, training_sampler, output, *, count, split, curve=False, artifacts=None,
                     renderer_factory=None, timeout=60, progress=None, resume=False, comparison=False, retrieval=None,
                     validation_extension=None, identity_diagnostics=False, validation_states=None):
    if split not in {"train", "validation"} or type(count) is not int or count <= 0:
        raise ValueError("pilot diagnostics use a positive count from train or validation; final test stays reserved")
    if validation_states is not None and (split != "validation" or not isinstance(validation_states, list)
                                          or len(validation_states) != count or any(not isinstance(state, str) for state in validation_states)
                                          or len(set(validation_states)) != count):
        raise ValueError("explicit validation states must be a unique list matching the requested count")
    torch.set_num_threads(4)
    saved, _ = next(iter(models.values()))
    codec = pool.codec
    replay = saved["replay"] if retrieval is None else retrieval["replay"]
    sampler = training_sampler if split == "train" else PilotSampler(pool, split="validation")
    if split == "train":
        recipes = [sampler.recipe_at(item["sequence_id"]) for item in replay[:count]]
    else:
        recipes, seen = [], set(validation_extension["excluded_state_ids"] if validation_extension else ())
        wanted = set(validation_states) if validation_states is not None else None
        for sequence in range(sampler.epoch_size):
            recipe = sampler.recipe_at(sequence)
            if wanted is not None and recipe.state_id not in wanted:
                continue
            if recipe.state_id not in seen:
                recipes.append(recipe)
                seen.add(recipe.state_id)
            if len(recipes) == count:
                break
    if len(recipes) != count:
        raise ValueError("requested evaluation count exceeds available cases")
    if validation_states is not None:
        by_state = {recipe.state_id: recipe for recipe in recipes}
        recipes = [by_state[state] for state in validation_states]
    model_info = {name: {"step": state["step"], "model_state_sha256": tensor_digest(model.state_dict()),
                         **(artifacts or {}).get(name, {})} for name, (state, model) in models.items()}
    plan = {"models": model_info, "pool_id": pool.pool_id, "recipe_ids": [recipe.recipe_id for recipe in recipes],
            "retrieval_sha256": retrieval_digest(replay),
            "primary": "equal_weight_mean_of_reference_and_transfer_per_state_mrstft",
            "maximum_render_attempts": count * (8 + 2 * len(models)), "test_split_evaluated": False}
    if retrieval is not None:
        plan["retrieval_source"] = retrieval["source"]
    if validation_extension is not None:
        plan["validation_extension"] = validation_extension
    if identity_diagnostics:
        plan["identity_diagnostics"] = "heldout_state_reference_to_reserved_transfer_retrieval_v1"
    if validation_states is not None:
        plan["validation_states"] = validation_states
    transfers = [entry for entry in initial_bank() if entry.role == "transfer"]
    if comparison:
        plan.update(evaluation="pilot_real_render_comparison_v2", candidate_order=list(models),
                    cases=[{"state_id": recipe.state_id, "reference_performance_id": recipe.performance_id,
                            "transfer_performance_id": transfers[index % len(transfers)].performance_id}
                           for index, recipe in enumerate(recipes)])
    rows, renders, completed, prior_elapsed = [], [], [], 0.0
    if resume:
        state, rows, renders = _restore_evaluation(output, plan, saved["renderer_identity"])
        completed, prior_elapsed = state["completed_cases"], state["elapsed_seconds"]
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "plan.json", plan)
    write_json(output / "status.json", {"complete": False, "stage": "real_render_evaluation"})
    fixed = codec.default_target()
    reused_cases, previous_renders, started = len(rows), len(renders), perf_counter()

    def save_progress(row=None):
        if row is not None:
            index = row["case"]
            path = output / f"case-{index:03d}.json"
            write_json(path, row)
            predictions = {path.name: sha256_bytes(path.read_bytes())
                           for path in output.glob(f"prediction-*-{index:03d}.vital")}
            completed.append({"case": index, "case_sha256": sha256_bytes(path.read_bytes()), "predictions": predictions})
        write_json(output / "progress.json", {"plan_sha256": canonical_sha256(plan), "completed_cases": completed,
                                              "committed_render_count": len(renders), "renders_sha256": canonical_sha256(renders),
                                              "elapsed_seconds": prior_elapsed + perf_counter() - started})

    if not resume:
        (output / "renders.jsonl").touch()
        save_progress()

    def render(raw, performance, case_index, method, role):
        request = RenderRequest(preset_json=raw.decode(), performance=performance, sample_rate=codec.contract.sample_rate,
                                channels=2, tail_seconds=codec.contract.tail_seconds, renderer_id=codec.contract.renderer_id)
        audio, metadata = render_isolated(request, renderer_factory=renderer_factory, timeout=timeout)
        identity = {key: value for key, value in metadata.get("provenance", {}).items() if key != "request_id"}
        if audio is not None and identity != saved["renderer_identity"]:
            raise ValueError("evaluation renderer differs from the training renderer")
        row = {"case": case_index, "method": method, "role": role, "request_id": request.request_id, **metadata}
        renders.append(row)
        with (output / "renders.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        if progress:
            progress({"stage": "evaluation", "case": case_index, "method": method, "role": role, "status": metadata["status"]})
        return audio

    try:
        for index, recipe in enumerate(recipes):
            if index < reused_cases:
                continue
            request = recipe.render_request()
            source_raw = request.preset_json.encode()
            reference = render(source_raw, request.performance, index, "source", "reference")
            row = {"case": index, "state_id": recipe.state_id, "reference_performance_id": recipe.performance_id,
                   "transfer_performance_id": transfers[index % len(transfers)].performance_id, "methods": {}}
            rows.append(row)
            if reference is None:
                row["failure"] = "reference_render_failed"
                save_progress(row)
                continue
            features = frontend(reference)
            targets = {}
            if identity_diagnostics:
                row["representations"] = {name: {} for name in models}
            for name, (_, model) in models.items():
                with torch.no_grad():
                    prediction = model(batch_features([{"features": features}]))
                    predicted = predicted_targets(prediction, codec)[0]
                if identity_diagnostics:
                    row["representations"][name]["reference"] = prediction["representation"][0].cpu().tolist()
                codec.preset(predicted).validate().require_valid()
                filename = f"prediction-{name}-{index:03d}.vital" if curve else f"prediction-{index:03d}.vital"
                (output / filename).write_bytes(codec.decode(predicted))
                targets[name] = predicted
            targets.update(fixed_template=fixed, train_retrieval=nearest_training_target(features, replay),
                           source_repeat=codec.encode(source_raw))
            transfer = transfers[index % len(transfers)].performance()
            transfer_source = render(source_raw, transfer, index, "source", "transfer")
            if identity_diagnostics and transfer_source is not None:
                transfer_features = batch_features([{"features": frontend(transfer_source)}])
                with torch.no_grad():
                    for name, (_, model) in models.items():
                        row["representations"][name]["transfer"] = model(transfer_features)["representation"][0].cpu().tolist()
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
        aggregates = {}
        for method in (*models, "fixed_template", "train_retrieval", "source_repeat"):
            aggregates[method] = {}
            for role in ("reference", "transfer"):
                scores = [row["methods"].get(method, {}).get(role, {}) for row in rows]
                scored = [score for score in scores if "mrstft" in score]
                aggregates[method][role] = {"planned": count, "scored": len(scored), "failed_or_unscorable": count - len(scored),
                                           "mean_mrstft": float(np.mean([score["mrstft"] for score in scored])) if scored else None,
                                           "mean_rms_relative": float(np.mean([score["rms_relative"] for score in scored])) if scored else None}
        frontend.require_frozen()
        summary = {"evaluation": "pilot_real_render_curve_v1" if curve else "pilot_real_render_v1",
                   "split": split, "cases": rows, "aggregates": aggregates, "models": model_info,
                   "pool_id": pool.pool_id,
                   "retrieval_examples": len(replay), "retrieval_distinct_train_states": len({item["state_id"] for item in replay}),
                   "retrieval_policy": "equal_branch_mean_squared_feature_distance_train_replay_only",
                   "renderer_identity": saved["renderer_identity"], "renders_attempted": len(renders),
                   "renders_accepted": sum(item["status"] == "accepted" for item in renders),
                   "audio_files_written": 0, "elapsed_seconds": prior_elapsed + perf_counter() - started,
                   "elapsed_includes_uncommitted_interrupted_work": not resume,
                   "render_wall_seconds": sum(item.get("wall_seconds", 0) for item in renders),
                   "renders_attempted_this_segment": len(renders) - previous_renders, "completed_cases_reused": reused_cases,
                   "planned_uninterrupted_render_attempts": plan["maximum_render_attempts"],
                   "recovery": "reuse_committed_cases_rerender_entire_unfinished_case_with_all_checkpoints",
                   "metric": "mean_over_512_2048_8192_of_spectral_convergence_plus_log_magnitude_mae_stereo",
                   "metric_limit": "diagnostic_spectral_distance_not_a_calibrated_perceptual_score",
                   "selection": "first_unique_validation_states_or_first_retained_training_views_no_best_of_k",
                   "transfer": "same_patch_predicted_from_reference_audio_rendered_on_reserved_performance",
                   "test_split_evaluated": False}
        if identity_diagnostics:
            from .paired import representation_statistics
            summary["identity_diagnostics"] = representation_statistics(rows, list(models))
        if curve:
            summary["retrieval_sha256"] = retrieval_digest(replay)
            statistics = curve_statistics(rows, list(models))
            if comparison:
                summary["evaluation"] = "pilot_real_render_comparison_v2"
                summary["retrieval_source"] = retrieval["source"]
                summary["candidate_order"] = list(models)
                if validation_extension is not None:
                    summary["validation_extension"] = validation_extension
                    summary["selection"] = "all_new_validation_states_from_verified_prefix_expansion"
                statistics["selection_policy"] = "lowest_complete_validation_mean_input_order_breaks_ties"
            summary["comparison" if comparison else "learning_curve"] = statistics
            selected = statistics["selected"]
            if selected is not None:
                selected_state, selected_model = models[selected]
                export_model(output / "selected-model.pt", selected_model, frontend, codec, selected_state["run_contract"]["pool_id"])
        else:
            summary.update(model_state_sha256=model_info["model"]["model_state_sha256"], step=saved["step"])
        write_json(output / "summary.json", summary)
        write_json(output / "status.json", {"complete": True, "all_renders_accepted": summary["renders_accepted"] == len(renders)})
        return summary
    except BaseException as error:
        write_json(output / "status.json", {"complete": False, "stage": "interrupted", "error_type": type(error).__name__})
        raise

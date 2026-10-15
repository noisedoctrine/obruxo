"""Bounded, resumable native qualification of the forward sound helper."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import torch

from obruxo_data.hashing import canonical_sha256
from obruxo_data.render.base import RenderRequest

from .audit import write_json
from .benchmark import MemoryMonitor
from .codec import PilotTarget
from .contracts import RenderRecipe
from .evaluate import audio_distance
from .features import tensor_digest
from .performances import initial_bank
from .proxy import (SoundFeatures, SoundProxy, feature_distance, feature_spec, performance_inputs, refine_continuous,
                    target_inputs, validate_sound_features)
from .splits import SplitLedger
from .stream import NativeRendererFactory, render_isolated
from .trainer import atomic_torch_save


@dataclass(frozen=True)
class ProxyConfig:
    training_states: int = 64
    validation_states: int = 10
    steps: int = 4000
    batch_size: int = 32
    learning_rate: float = .001
    seed: int = 0
    checkpoint_every: int = 250
    replay_mib: int = 8
    cpu_threads: int = 4
    timeout: float = 30.

    def __post_init__(self):
        integers = (self.training_states, self.validation_states, self.steps, self.batch_size, self.checkpoint_every,
                    self.replay_mib, self.cpu_threads)
        if any(type(value) is not int or value < 1 for value in integers) or not 8 <= self.training_states <= 256:
            raise ValueError("proxy needs positive integer budgets and 8..256 training states")
        if type(self.seed) is not int or not math.isfinite(self.learning_rate) or not 0 < self.learning_rate <= .1:
            raise ValueError("invalid proxy learning configuration")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("proxy timeout must be positive and finite")


def balanced_entries(pool, split, count):
    if split not in {"train", "validation"}:
        raise ValueError("proxy qualification keeps the final test reserved")
    groups = {}
    for entry in pool.entries:
        if pool.manifest.by_state[entry.state_id].split == split:
            family = pool.codec.family(entry.target) if hasattr(pool.codec, "family") else "pilot"
            groups.setdefault(family, []).append(entry)
    selected = []
    for index in range(max(map(len, groups.values()), default=0)):
        selected.extend(entries[index] for entries in groups.values() if index < len(entries))
    if len(selected) < count:
        raise ValueError(f"proxy panel needs {count} distinct {split} states")
    return selected[:count]


def source_identity():
    research = Path(__file__).resolve().parents[2]
    paths = [*Path(__file__).parent.glob("*.py"), *(research / "data_generation/obruxo_data").rglob("*.py")]
    return {path.relative_to(research).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


def proxy_plan(pool, config):
    training = balanced_entries(pool, "train", config.training_states)
    validation = balanced_entries(pool, "validation", config.validation_states)
    ledger, recipes = SplitLedger(pool.manifest), []
    bank = initial_bank()
    for entry in training:
        for performance in bank[:4]:
            recipes.append(RenderRecipe.create(pool.codec.decode(entry.target), performance.performance(), schema=pool.codec.schema,
                                              catalog=pool.codec.catalog, contract=pool.codec.contract, ledger=ledger,
                                              purpose="train", sequence_id=len(recipes)))
    plan = {"format": "obruxo_sound_proxy_screen_v1", "pool_id": pool.pool_id, "codec_id": pool.codec.codec_id,
            "config": asdict(config), "features": feature_spec(), "source": source_identity(),
            "model": {"name": "hard_category_continuous_midi_mlp_v1", "hidden_widths": [256, 256], "activation": "silu_softplus",
                      "input": "canonical_masked_controls_one_hot_categories_16_sorted_notes_and_timing",
                      "normalization": "accepted_training_coordinate_std_floor_0.05", "group_weights": "equal",
                      "category_gradients": False},
            "runtime": {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
                        "device": "cpu", "precision": "float32"},
            "training_recipes": [{"recipe_id": recipe.recipe_id, "state_id": recipe.state_id,
                                  "performance_id": recipe.performance_id} for recipe in recipes],
            "validation_states": [entry.state_id for entry in validation],
            "panel_policy": "first_states_round_robin_authored_families_before_outcomes",
            "validation_performances": [{"reference": bank[index % 4].performance_id, "transfer": bank[4 + index % 4].performance_id}
                                        for index in range(len(validation))],
            "perturbations": {"continuous": [.05, .30], "category": "next_osc1_asset_or_unison",
                              "gradient_probe": {"steps": 16, "learning_rate": .03, "radius": .15}},
            "baselines": ["training_mean", "nearest_training_patch_and_performance", "five_neighbor_interpolation"],
            "render_budget": {"training": len(recipes), "validation": 12 * len(validation), "recovery_reserve": 12,
                              "maximum_attempts": len(recipes) + 12 * len(validation) + 12},
            "qualification": {"all_validation_renders_accepted": True, "reference_error_ratio_vs_each_baseline_max": .9,
                              "transfer_error_ratio_vs_each_baseline_max": 1., "native_ranking_concordance_min": .6,
                              "native_ranking_scored_fraction_min": .5,
                              "native_gradient_improvement_fraction_min": .6, "mean_native_gradient_delta_max": 0.},
            "limits": "single_seed_small_authored_vocabulary_no_release_promotion", "final_test_used": False,
            "audio_files_written": 0}
    return plan, recipes, validation


def fit_proxy(records, codec, config, *, saved=None, checkpoint=None, after_step=None):
    if not records:
        raise ValueError("proxy cannot train without accepted native features")
    torch.set_num_threads(config.cpu_threads)
    torch.manual_seed(config.seed)
    features = torch.stack([item["features"] for item in records])
    targets = [PilotTarget.from_dict(item["target"]) for item in records]
    patch = target_inputs(targets, codec)
    performances = {entry.performance_id: performance_inputs(entry.performance()) for entry in initial_bank()[:4]}
    try:
        played = torch.stack([performances[item["performance_id"]] for item in records])
    except KeyError as error:
        raise ValueError("proxy training contains a reserved or unknown performance") from error
    model = SoundProxy(codec, features.mean(0), features.std(0, unbiased=False).clamp_min(.05))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    rng = torch.Generator().manual_seed(config.seed)
    step, trace = 0, []
    if saved:
        if saved["data_sha256"] != data_digest(records) or not 0 <= saved["step"] <= config.steps:
            raise ValueError("proxy training data or step changed")
        if not torch.equal(saved["model"]["mean"], model.mean) or not torch.equal(saved["model"]["scale"], model.scale):
            raise ValueError("proxy normalization must be fitted on training features only")
        if tensor_digest(saved["model"]) != saved["model_sha256"]:
            raise ValueError("proxy model checkpoint changed")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        rng.set_state(saved["batch_rng"])
        step, trace = saved["step"], saved["trace"]

    def snapshot():
        return {"step": step, "model": model.state_dict(), "model_sha256": tensor_digest(model.state_dict()),
                "optimizer": optimizer.state_dict(), "batch_rng": rng.get_state(), "trace": trace,
                "data_sha256": data_digest(records)}

    if not trace:
        with torch.no_grad():
            trace.append({"step": 0, "training_error": float(feature_distance(model(patch, played), features, model.scale).mean())})
        if checkpoint:
            checkpoint(snapshot())
    while step < config.steps:
        indices = torch.randint(len(records), (min(config.batch_size, len(records)),), generator=rng)
        optimizer.zero_grad(set_to_none=True)
        loss = feature_distance(model(patch[indices], played[indices]), features[indices], model.scale).mean()
        if not torch.isfinite(loss):
            raise ValueError("nonfinite proxy loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        step += 1
        if step % config.checkpoint_every == 0 or step == config.steps:
            with torch.no_grad():
                trace.append({"step": step, "training_error": float(feature_distance(model(patch, played), features, model.scale).mean())})
            if checkpoint:
                checkpoint(snapshot())
        if after_step:
            after_step(step)
    return model.freeze(), snapshot()


def data_digest(records):
    return canonical_sha256([{**{key: value for key, value in item.items() if key != "features"},
                              "features_sha256": tensor_digest({"features": item["features"]})} for item in records])


def validate_records(records, recipes, pool, next_train):
    if type(next_train) is not int or not 0 <= next_train <= len(recipes):
        raise ValueError("invalid proxy render cursor")
    entries = {entry.state_id: entry for entry in pool.entries}
    sequences = [item["sequence"] for item in records]
    if sequences != sorted(set(sequences)) or any(not 0 <= sequence < next_train for sequence in sequences):
        raise ValueError("proxy data lies outside its acknowledged render prefix")
    for item in records:
        recipe = recipes[item["sequence"]]
        if (item["state_id"] != recipe.state_id or item["performance_id"] != recipe.performance_id
                or item["recipe_id"] != recipe.recipe_id or item["target"] != entries[recipe.state_id].target.to_dict()):
            raise ValueError("proxy training record differs from its train-only recipe")
        validate_sound_features(item["features"])


def baseline_features(records, codec, target, performance):
    patches = target_inputs([PilotTarget.from_dict(item["target"]) for item in records], codec)
    bank = {entry.performance_id: performance_inputs(entry.performance()) for entry in initial_bank()[:4]}
    played = torch.stack([bank[item["performance_id"]] for item in records])
    distance = (patches - target_inputs([target], codec)).square().mean(1) + (played - performance).square().mean(1)
    indices = torch.argsort(distance, stable=True)[:min(5, len(records))]
    weights = 1 / distance[indices].clamp_min(1e-6)
    features = torch.stack([item["features"] for item in records])
    return {"training_mean": features.mean(0), "nearest_training_patch_and_performance": features[indices[0]],
            "five_neighbor_interpolation": (features[indices] * weights[:, None]).sum(0) / weights.sum()}


def perturbations(target, codec, index):
    active = [position for position, enabled in enumerate(codec.continuous_mask(target)) if enabled]
    position = active[index % len(active)]
    direction = 1 if target.continuous[position] <= .5 else -1
    result = {}
    for name, change in (("small", .05), ("large", .30)):
        values = list(target.continuous)
        values[position] = min(1., max(0., values[position] + direction * change))
        result[name] = codec.canonical_target(PilotTarget(tuple(values), target.categorical))
    categories = list(target.categorical)
    slot = next((slot for slot, (name, _) in enumerate(codec.categorical) if name == "osc_1_asset"), 0)
    categories[slot] = (categories[slot] + 1) % len(codec.categorical[slot][1])
    result["category"] = codec.canonical_target(PilotTarget(target.continuous, tuple(categories)))
    return result, {"continuous_control": codec.continuous[position][0], "category_control": codec.categorical[slot][0]}


def summarize(rows, plan):
    complete = len(rows) == 2 * len(plan["validation_states"])
    qualified = complete and all(row.get("accepted_renders") == 6 for row in rows)
    errors = {}
    for role in ("reference", "transfer"):
        panel = [row for row in rows if row["role"] == role]
        scored = [row for row in panel if "forward_error" in row]
        methods = ["proxy", *plan["baselines"]]
        means = {name: float(np.mean([row["forward_error"][name] for row in scored])) if scored else None for name in methods}
        maximum = plan["qualification"][f"{role}_error_ratio_vs_each_baseline_max"]
        passed = len(scored) == len(panel) == len(plan["validation_states"]) and all(
            means["proxy"] is not None and means[name] > 0 and means["proxy"] / means[name] <= maximum for name in plan["baselines"])
        qualified &= passed
        errors[role] = {"planned": len(plan["validation_states"]), "scored": len(scored), "mean_errors": means, "passed": passed}
    concordant, comparisons, ties, deltas, improvements = 0, 0, 0, [], 0
    for row in rows:
        candidates = row.get("candidates", {})
        names = [name for name in ("small", "large", "category") if name in candidates]
        noise = row.get("repeat_mrstft", 0.)
        for index, left in enumerate(names):
            for right in names[index + 1:]:
                actual = candidates[left]["mrstft"] - candidates[right]["mrstft"]
                predicted = candidates[left]["proxy_error"] - candidates[right]["proxy_error"]
                if abs(actual) <= noise or abs(predicted) <= 1e-12:
                    ties += 1
                    continue
                comparisons += 1
                concordant += int(actual * predicted > 0)
        if "large" in candidates and "guided" in candidates:
            delta = candidates["guided"]["mrstft"] - candidates["large"]["mrstft"]
            deltas.append(delta)
            improvements += delta < -noise
    ratio = concordant / comparisons if comparisons else None
    fraction = improvements / len(deltas) if deltas else None
    mean_delta = float(np.mean(deltas)) if deltas else None
    gates = plan["qualification"]
    qualified &= (ratio is not None and ratio >= gates["native_ranking_concordance_min"]
                  and comparisons >= gates["native_ranking_scored_fraction_min"] * 6 * len(plan["validation_states"])
                  and len(deltas) == 2 * len(plan["validation_states"])
                  and fraction >= gates["native_gradient_improvement_fraction_min"] and mean_delta < gates["mean_native_gradient_delta_max"])
    return {"complete_panel": complete, "qualified_for_inverse_training": bool(qualified), "forward_errors": errors,
            "candidate_ranking": {"planned_comparisons": 6 * len(plan["validation_states"]), "scored": comparisons,
                                  "noise_or_proxy_ties": ties, "concordant": concordant, "concordance": ratio},
            "gradient_probe": {"planned": 2 * len(plan["validation_states"]), "scored": len(deltas),
                               "improvements_beyond_repeat_noise": improvements, "improvement_fraction": fraction,
                               "mean_native_mrstft_delta": mean_delta},
            "independent_validation_states": len(plan["validation_states"]),
            "statistics_limit": "descriptive_paired_performances_not_independent_samples", "final_test_used": False}


def run_proxy_screen(pool, output, *, config=None, renderer_factory=None, resume=False, progress=None):
    config = config or ProxyConfig()
    torch.set_num_threads(config.cpu_threads)
    plan, recipes, validation = proxy_plan(pool, config)
    plan_id = canonical_sha256(plan)
    output = Path(output)
    if resume:
        if json.loads((output / "plan.json").read_text()) != plan:
            raise ValueError("proxy plan, runtime, source or pool changed; cannot resume")
        state = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
        logs = [json.loads(line) for line in (output / "renders.jsonl").read_text().splitlines()]
        if (state["plan_id"] != plan_id or canonical_sha256(logs[:state["log_count"]]) != state["log_sha256"]
                or data_digest(state["records"]) != state["data_sha256"]
                or canonical_sha256(state["rows"]) != state["rows_sha256"]):
            raise ValueError("proxy checkpoint, features or committed render log changed")
        validate_records(state["records"], recipes, pool, state["next_train"])
        acknowledged = {row["sequence"]: row for row in logs[:state["log_count"]] if row["stage"] == "training"}
        retained = {item["sequence"]: item for item in state["records"]}
        for sequence in range(state["next_train"]):
            outcome = acknowledged.get(sequence, {})
            item = retained.get(sequence)
            if (outcome.get("recipe_id") != recipes[sequence].recipe_id or (outcome.get("status") == "accepted") != (item is not None)
                    or (item is not None and item["audio_sha256"] != outcome["qa"]["audio_float32_sha256"])):
                raise ValueError("proxy data does not match its acknowledged native QA outcomes")
        if len(logs) > plan["render_budget"]["maximum_attempts"] or any(row["attempt"] != index for index, row in enumerate(logs)):
            raise ValueError("invalid proxy native attempt history")
        for row in logs:
            if row["status"] != "error":
                identity = {key: value for key, value in row["provenance"].items() if key != "request_id"}
                if state["renderer_identity"] is not None and identity != state["renderer_identity"]:
                    raise ValueError("saved proxy renderer provenance changed")
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "plan.json", plan)
        (output / "renders.jsonl").touch()
        state = {"plan_id": plan_id, "next_train": 0, "records": [], "training": None, "rows": [], "renderer_identity": None}
        logs = []
    factory = renderer_factory or NativeRendererFactory()
    frontend = SoundFeatures()
    monitor = MemoryMonitor()
    started = perf_counter()
    entries = {entry.state_id: entry for entry in pool.entries}

    def save():
        state.update(log_count=len(logs), log_sha256=canonical_sha256(logs), data_sha256=data_digest(state["records"]),
                     rows_sha256=canonical_sha256(state["rows"]))
        atomic_torch_save(output / "checkpoint.pt", state)

    def render(request, context):
        if len(logs) >= plan["render_budget"]["maximum_attempts"]:
            raise ValueError("frozen proxy native-render attempt budget exhausted")
        audio, metadata = render_isolated(request, renderer_factory=factory, timeout=config.timeout)
        row = {"attempt": len(logs), "request_id": request.request_id, **context, **metadata}
        logs.append(row)
        with (output / "renders.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
        if metadata["status"] != "error":
            identity = {key: value for key, value in metadata["provenance"].items() if key != "request_id"}
            if state["renderer_identity"] is not None and identity != state["renderer_identity"]:
                raise ValueError("native renderer provenance changed")
            state["renderer_identity"] = identity
        return audio, metadata

    def notify(event):
        if progress:
            progress(event)

    save()
    monitor.start()
    try:
        for index in range(state["next_train"], len(recipes)):
            recipe = recipes[index]
            audio, metadata = render(recipe.render_request(), {"stage": "training", "sequence": index, "recipe_id": recipe.recipe_id})
            if audio is not None:
                state["records"].append({"sequence": index, "recipe_id": recipe.recipe_id, "state_id": recipe.state_id,
                                         "performance_id": recipe.performance_id, "target": entries[recipe.state_id].target.to_dict(),
                                         "audio_sha256": metadata["qa"]["audio_float32_sha256"], "features": frontend(audio)})
            state["next_train"] = index + 1
            retained = sum(item["features"].numel() * 4 for item in state["records"])
            if retained > config.replay_mib * 1024 * 1024:
                raise ValueError("proxy feature replay byte budget exceeded")
            save()
            notify({"stage": "collect", "attempted": index + 1, "planned": len(recipes), "accepted": len(state["records"]), "status": metadata["status"]})
        qualified_states = {item["state_id"] for item in state["records"]}
        def family(entry):
            return pool.codec.family(entry.target) if hasattr(pool.codec, "family") else "pilot"
        planned_families = {family(entries[recipe.state_id]) for recipe in recipes}
        coverage = Counter(family(entries[state_id]) for state_id in qualified_states)
        if len(state["records"]) < .8 * len(recipes) or set(coverage) != planned_families:
            raise ValueError("native QA left insufficient proxy training coverage; no budget extension")

        def checkpoint(training):
            state["training"] = training
            save()
            notify({"stage": "train", **training["trace"][-1]})

        model, state["training"] = fit_proxy(state["records"], pool.codec, config, saved=state["training"], checkpoint=checkpoint)
        save()
        bank, ledger = initial_bank(), SplitLedger(pool.manifest)
        for case in range(len(state["rows"]), 2 * len(validation)):
            index, role_index = divmod(case, 2)
            entry = validation[index]
            performance = bank[index % 4 + 4 * role_index]
            recipe = RenderRecipe.create(pool.codec.decode(entry.target), performance.performance(), schema=pool.codec.schema,
                                         catalog=pool.codec.catalog, contract=pool.codec.contract, ledger=ledger, purpose="evaluate", sequence_id=case)
            row = {"case": case, "state_id": entry.state_id, "family": family(entry), "role": performance.role,
                   "performance_id": performance.performance_id, "accepted_renders": 0, "candidates": {}}

            def candidate_render(target, name):
                request = RenderRequest.from_dict({**recipe.render_request().to_dict(), "preset_json": pool.codec.decode(target).decode()})
                audio, metadata = render(request, {"stage": "validation", "case": case, "method": name, "recipe_id": recipe.recipe_id})
                row["accepted_renders"] += metadata["status"] == "accepted"
                return audio

            source = candidate_render(entry.target, "source")
            if source is not None:
                desired = frontend(source)
                played = performance_inputs(performance.performance())
                with torch.no_grad():
                    predicted = model(target_inputs([entry.target], pool.codec), played[None])[0]
                    estimates = {"proxy": predicted, **baseline_features(state["records"], pool.codec, entry.target, played)}
                    row["forward_error"] = {name: float(feature_distance(value, desired, model.scale)) for name, value in estimates.items()}
                candidates, row["changed_controls"] = perturbations(entry.target, pool.codec, index)
                candidates["guided"], row["proxy_update"] = refine_continuous(model, candidates["large"], played, desired)
                for name, target in candidates.items():
                    audio = candidate_render(target, name)
                    if audio is not None:
                        with torch.no_grad():
                            predicted = model(target_inputs([target], pool.codec), played[None])[0]
                            score = float(feature_distance(predicted, desired, model.scale))
                        row["candidates"][name] = {**audio_distance(source, audio), "proxy_error": score,
                                                   "measurement_error": float(feature_distance(frontend(audio), desired, model.scale)),
                                                   "target": target.to_dict()}
                repeat = candidate_render(entry.target, "repeat")
                if repeat is not None:
                    row["repeat_mrstft"] = audio_distance(source, repeat)["mrstft"]
            state["rows"].append(row)
            save()
            write_json(output / f"case-{case:03d}.json", row)
            notify({"stage": "validate", "completed": case + 1, "planned": 2 * len(validation), "accepted": row["accepted_renders"]})
        summary = summarize(state["rows"], plan)
        summary.update(format=plan["format"], plan_id=plan_id, pool_id=pool.pool_id, render_attempts=len(logs),
                       renders_accepted=sum(row["status"] == "accepted" for row in logs),
                       training_records=len(state["records"]), training_states=len(qualified_states), training_families=dict(coverage),
                       retained_feature_bytes=sum(item["features"].numel() * 4 for item in state["records"]),
                       trainable_parameters=sum(parameter.numel() for parameter in model.parameters()),
                       training_trace=state["training"]["trace"], model_sha256=tensor_digest(model.state_dict()),
                       renderer_identity=state["renderer_identity"], elapsed_seconds_this_segment=perf_counter() - started,
                       peak_process_tree_rss_bytes=monitor.peak_bytes, audio_files_written=0,
                       inverse_model_trained=False, proxy_connected_to_inverse=False)
        write_json(output / "summary.json", summary)
        atomic_torch_save(output / "proxy.pt", {"format": "obruxo_sound_proxy_v1", "codec": pool.codec.to_dict(),
                                               "features": feature_spec(), "model": model.state_dict(), "model_sha256": tensor_digest(model.state_dict()),
                                               "plan_id": plan_id,
                                               "qualified_for_inverse_training": summary["qualified_for_inverse_training"]})
        return summary
    finally:
        monitor.close()

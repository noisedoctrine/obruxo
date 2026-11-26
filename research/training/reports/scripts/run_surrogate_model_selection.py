"""Compare held-out Vital audio proxies and direct inverse synthesis quality."""

# ruff: noqa: E402
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

TRAINING = Path(__file__).resolve().parents[2]
RESEARCH = TRAINING.parent
sys.path.insert(0, str(RESEARCH / "data_generation"))
sys.path.insert(0, str(TRAINING))
sys.path.insert(0, str(TRAINING / "reports" / "scripts"))

from obruxo_data.hashing import canonical_sha256, sha256_bytes
from obruxo_data.midi import Performance

from obruxo_training.audit import write_json
from obruxo_training.codec import PilotCodec, PilotTarget
from obruxo_training.evaluate import audio_distances
from obruxo_training.pilot import PilotPool
from obruxo_training.proxy import performance_inputs
from obruxo_training.surrogate_codec import SurrogateCodec
from obruxo_training.surrogate_pilot import Progress, collect, control_tensor, fit, native_session, source_fingerprints
from obruxo_training.surrogate_pilot_data import _expand_roots, _ordered_roots, perturb
from obruxo_training.surrogate_pilot_eval import candidate_categories, _target_on_device
from obruxo_training.temporal_proxy_data import training_performance
from obruxo_training.temporal_proxy import sample_frames
from obruxo_training.spectral import SpectralTargets, prepare_spectral_target, spectral_loss
from run_surrogate_pilot import write_control_inventory

BACKBONES = ("residual_mlp", "transformer", "conformer")
ROOTS = 96
PERFORMANCES = 24
VALIDATION_ROOTS = 64
VALIDATION_PERFORMANCES = 16
VIEWS_PER_ROOT = 2
OPTIMIZER_RADIUS = .15
OPTIMIZER_LR = .03


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _file_sha256(path):
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cached_training_rows(pilot_run):
    source = _read_json(pilot_run / "plan.json")
    plan = source["plan"]
    accepted, receipts, counts = [], [], {}
    pair_dir = pilot_run / "pairs"
    for stage in ("basic", "expanded"):
        counts[stage] = {"planned": 0, "accepted": 0, "rejected": 0}
        for row in plan["stages"][stage]["pairs"]:
            path = pair_dir / (row["pair_id"] + ".pt")
            if not path.is_file():
                raise FileNotFoundError(f"missing cached render pair: {path}")
            cached = torch.load(path, map_location="cpu", weights_only=True)
            if cached["id"] != canonical_sha256(row):
                raise ValueError(f"cached render recipe changed: {row['pair_id']}")
            counts[stage]["planned"] += 1
            counts[stage]["accepted" if cached["accepted"] else "rejected"] += 1
            receipts.append({"pair_id": row["pair_id"], "recipe_sha256": cached["id"],
                             "cache_sha256": _file_sha256(path), "accepted": cached["accepted"]})
            if cached["accepted"]:
                accepted.append(row)
    expected = {"basic": 1024, "expanded": 1022}
    if {stage: counts[stage]["accepted"] for stage in counts} != expected:
        raise ValueError(f"source run's accepted pair counts changed: {counts}")
    return plan, accepted, receipts, counts


def _fresh_midi_bank(seed, occupied, count=PERFORMANCES):
    if count % 4:
        raise ValueError("MIDI bank count must be divisible by the four performance styles")
    performances, used = [], set(occupied)
    for style in range(4):
        variants, candidate = [], 0
        while len(variants) < count // 4:
            value = training_performance(seed, candidate, style).to_dict()
            midi_id = canonical_sha256(value)
            if midi_id not in used:
                used.add(midi_id)
                variants.append({"midi_id": midi_id, "performance": value, "style": style,
                                 "variant": len(variants)})
            candidate += 1
            if candidate > 10000:
                raise RuntimeError("could not author 24 distinct confirmation performances")
        performances.extend(variants)
    return performances


def _confirmation_cases(pool_path, pilot_plan, reservation_path, codec, midi_seed):
    pool = PilotPool.load(pool_path)
    pool_dict = pool.to_dict()
    reservation = _read_json(reservation_path)
    old_states = {row["state_id"] for stage in pilot_plan["stages"].values()
                  for field in ("pairs", "cases") for row in stage[field]}
    reserved_states = {row["lineage_id"] for row in reservation["cases"]}
    old_midi = {row["midi_id"] for role in pilot_plan["midi"].values()
                for style in role for row in style}
    reserved_midi = {canonical_sha256(row["performance"]) for row in reservation["performances"]}
    roots = _ordered_roots(pool_dict, "validation", set())[:ROOTS]
    if len(roots) != ROOTS:
        raise ValueError(f"fresh validation pool has only {len(roots)} eligible roots")
    fresh_states = {row["state_id"] for row in roots}
    if len(fresh_states) != ROOTS or fresh_states & old_states or fresh_states & reserved_states:
        raise ValueError("confirmation patch roots overlap each other, the pilot or the reserved set")
    midi = _fresh_midi_bank(midi_seed, old_midi | reserved_midi)
    midi_ids = {row["midi_id"] for row in midi}
    if len(midi_ids) != PERFORMANCES or midi_ids & old_midi or midi_ids & reserved_midi:
        raise ValueError("confirmation MIDI overlaps a pilot or reserved performance")
    expanded = _expand_roots(roots, codec, ROOTS // 4, f"issue173-confirmation-{midi_seed}")
    cases = []
    for root_index, row in enumerate(expanded):
        for view in range(VIEWS_PER_ROOT):
            performance = midi[(root_index * VIEWS_PER_ROOT + view) % PERFORMANCES]
            cases.append({**row, **performance, "midi_role": "fresh",
                          "case_id": f"fresh-{root_index:03d}-{view}"})
    usage = Counter(row["midi_id"] for row in cases)
    if len(cases) != ROOTS * VIEWS_PER_ROOT or set(usage.values()) != {ROOTS * VIEWS_PER_ROOT // PERFORMANCES}:
        raise ValueError(f"confirmation performance assignments are not balanced: {usage}")
    return pool, cases, midi


def _validation_pairs(pool, pilot_plan, reservation_path, codec, final_midi, midi_seed):
    pool_dict = pool.to_dict()
    roots = _ordered_roots(pool_dict, "validation", set())
    selected = roots[ROOTS:ROOTS + VALIDATION_ROOTS]
    if len(selected) != VALIDATION_ROOTS:
        raise ValueError(f"fresh validation pool needs at least {ROOTS + VALIDATION_ROOTS} roots")
    if {row["state_id"] for row in selected} & {row["state_id"] for row in _ordered_roots(pool_dict, "validation", set())[:ROOTS]}:
        raise ValueError("proxy validation roots overlap the final inverse-synthesis panel")
    reservation = _read_json(reservation_path)
    old_midi = {row["midi_id"] for role in pilot_plan["midi"].values() for style in role for row in style}
    reserved_midi = {canonical_sha256(row["performance"]) for row in reservation["performances"]}
    final_midi_ids = {row["midi_id"] for row in final_midi}
    midi = _fresh_midi_bank(midi_seed, old_midi | reserved_midi | final_midi_ids, VALIDATION_PERFORMANCES)
    expanded = _expand_roots(selected, codec, VALIDATION_ROOTS // 4, f"issue173-proxy-validation-{midi_seed}")
    rows = []
    for root_index, root in enumerate(expanded):
        target = PilotTarget.from_dict(root["target"])
        for view in range(VIEWS_PER_ROOT):
            perf = midi[(root_index * VIEWS_PER_ROOT + view) % len(midi)]
            changed, control = perturb(codec, target, root_index * VIEWS_PER_ROOT + view, root["state_id"])
            rows.append({**root, **perf, "changed": changed.to_dict(), "control": control,
                         "pair_id": f"validation-{root_index:03d}-{view}"})
    usage = Counter(row["midi_id"] for row in rows)
    if len(rows) != VALIDATION_ROOTS * VIEWS_PER_ROOT or set(usage.values()) != {len(rows) // VALIDATION_PERFORMANCES}:
        raise ValueError(f"validation performance assignments are not balanced: {usage}")
    return rows, midi


def _plan(args, codec):
    pilot_source = _read_json(args.pilot_run / "plan.json")
    pilot_plan, training_rows, pair_receipts, pair_counts = _cached_training_rows(args.pilot_run)
    pool, cases, midi = _confirmation_cases(args.pool, pilot_plan, args.reservation, codec, args.midi_seed)
    validation_rows, validation_midi = _validation_pairs(
        pool, pilot_plan, args.reservation, codec, midi, args.validation_midi_seed)
    files = {**source_fingerprints()}
    for name in ("codec.py", "pilot.py", "proxy.py", "temporal_proxy.py", "temporal_proxy_data.py"):
        files[name] = sha256_bytes((TRAINING / "obruxo_training" / name).read_bytes())
    files["run_surrogate_pilot.py"] = sha256_bytes((TRAINING / "reports" / "scripts" /
                                                     "run_surrogate_pilot.py").read_bytes())
    files["stage_a_controls.jsonl"] = sha256_bytes((RESEARCH / "modelling" / "control_scope" /
                                                     "stage_a_controls.jsonl").read_bytes())
    files[Path(__file__).name] = sha256_bytes(Path(__file__).read_bytes())
    plan = {
        "format": "issue173_proxy_inverse_selection_v1",
        "codec": codec.spec(),
        "sources": files,
        "training": {
            "pilot_plan_id": pilot_source["id"],
            "pilot_pool_id": pilot_plan["pool_id"],
            "accepted_pairs": pair_counts,
            "pair_receipts": pair_receipts,
            "training_roots": len({row["state_id"] for row in training_rows}),
            "seed": args.seed,
            "steps": args.steps,
            "backbones": list(BACKBONES),
            "warm_start": False,
            "purpose": "select the best forward audio proxy and direct inverse-synthesis model",
            "device": str(args.device),
            "threads": args.threads,
        },
        "evaluation": {
            "pool_id": pool.pool_id,
            "pool_seed": pool.seed,
            "split": "fresh pool validation",
            "unused_test_roots": pool.summary()["states_by_split"]["test"],
            "patch_roots": ROOTS,
            "distinct_midi_performances": PERFORMANCES,
            "midi_bank": [{"midi_id": row["midi_id"], "style": row["style"], "variant": row["variant"],
                           "assigned_cases": sum(case["midi_id"] == row["midi_id"] for case in cases)}
                          for row in midi],
            "cases": cases,
            "known_midi": True,
            "categorical_policy": "score a deterministic 24-choice legal category shortlist from audio; refine the best 6 with continuous gradients",
            "starting_continuous": "expanded codec defaults for each shortlisted legal category",
            "optimizer": {"method": "projected Adam over shortlisted category candidates", "learning_rate": OPTIMIZER_LR,
                          "steps": args.gradient_steps, "normalized_radius": OPTIMIZER_RADIUS,
                          "select": "lowest surrogate loss across all category candidates and optimizer steps",
                          "legal_category_shortlist": [list(choices) for choices in candidate_categories(codec)],
                          "refined_category_count": 6},
            "native_comparison": "render direct inverse-synthesis output and codec-default baseline against the target under identical known MIDI",
            "validation_proxy": {"patch_roots": VALIDATION_ROOTS, "distinct_midi_performances": VALIDATION_PERFORMANCES,
                                 "pair_count": len(validation_rows), "clips_per_pair": 2,
                                 "midi_bank": [{"midi_id": row["midi_id"], "style": row["style"],
                                                "assigned_pairs": sum(pair["midi_id"] == row["midi_id"] for pair in validation_rows)}
                                               for row in validation_midi],
                                 "pair_recipes": validation_rows,
                                 "held_out_from_training_and_final_panel": True,
                                 "loss": "spectral/envelope error plus 0.25 times paired control-change error",
                                 "curve_interval_updates": args.validation_every},
        },
        "reservation_id": _read_json(args.reservation).get("reservation_id"),
        "midi_seed": args.midi_seed,
        "validation_midi_seed": args.validation_midi_seed,
        "gradient_steps": args.gradient_steps,
        "validation_every": args.validation_every,
        "render_workers": args.workers,
    }
    plan_id = canonical_sha256(plan)
    path = args.output / "plan.json"
    frozen = {"id": plan_id, "plan": plan}
    if path.exists():
        saved = _read_json(path)
        if saved.get("id") != plan_id or canonical_sha256(saved.get("plan")) != plan_id:
            raise ValueError("saved confirmation plan differs from the current inputs")
    else:
        write_json(path, frozen)
    return plan_id, plan, training_rows, cases, validation_rows


def _measure(audio):
    return SpectralTargets()(torch.from_numpy(audio.copy()).T[None])


def _expand_batch(value, size):
    if isinstance(value, dict):
        return {key: _expand_batch(item, size) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_expand_batch(item, size) for item in value)
    return value.expand(size, *value.shape[1:])


def _choose_patch(model, codec, target, times, played, *, steps):
    device = played.device
    categories = candidate_categories(codec)
    starts = [codec.canonical(PilotTarget(codec.defaults, choices)) for choices in categories]
    encoded = control_tensor(codec, starts, device)

    def scores(controls):
        measured = _expand_batch(target, len(controls))
        return spectral_loss(model(controls, played.expand(len(controls), -1), _expand_batch(times, len(controls)),
                                   measured["lengths"]), measured)["loss"]

    with torch.no_grad():
        prefilter_start = perf_counter()
        baseline = torch.cat([scores(group) for group in encoded.split(6)])
        prefilter_seconds = perf_counter() - prefilter_start
    selected = baseline.argsort(stable=True)[:6].tolist()
    chosen, template = [starts[index] for index in selected], encoded[selected]
    width = len(codec.continuous)
    initial, category_inputs = template[:, :width].detach(), template[:, width:].detach()
    active = torch.tensor([codec.masks(patch.categorical)[0] for patch in chosen], dtype=torch.bool, device=device)
    lower, upper = (initial - OPTIMIZER_RADIUS).clamp(0, 1), (initial + OPTIMIZER_RADIUS).clamp(0, 1)

    def project(values):
        bounded = torch.maximum(torch.minimum(values, upper), lower)
        return torch.where(active, bounded, initial)

    value = initial.clone().requires_grad_()
    optimizer = torch.optim.Adam([value], lr=OPTIMIZER_LR)
    best_losses, best_values = baseline[selected].detach(), initial.clone()
    forward_seconds = backward_seconds = update_seconds = 0.
    for _ in range(steps):
        started = perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = scores(torch.cat((value, category_inputs), dim=-1))
        forward_seconds += perf_counter() - started
        with torch.no_grad():
            improved = loss < best_losses
            best_losses[improved] = loss[improved]
            best_values[improved] = value[improved]
        started = perf_counter()
        loss.sum().backward()
        backward_seconds += perf_counter() - started
        started = perf_counter()
        optimizer.step()
        with torch.no_grad():
            value.copy_(project(value))
        update_seconds += perf_counter() - started
    with torch.no_grad():
        started = perf_counter()
        final = scores(torch.cat((value, category_inputs), dim=-1))
        forward_seconds += perf_counter() - started
        improved = final < best_losses
        best_losses[improved] = final[improved]
        best_values[improved] = value[improved]
    winner = int(best_losses.argmin())
    patch = codec.canonical(PilotTarget(tuple(best_values[winner].cpu().tolist()), chosen[winner].categorical))
    return patch, {"candidate_categories": len(categories), "refined_categories": len(chosen),
                   "start_surrogate_loss": float(baseline[selected].min()),
                   "best_surrogate_loss": float(best_losses[winner]), "steps": steps,
                   "active_continuous_controls": int(active[winner].sum()),
                   "category_prefilter_seconds": prefilter_seconds, "forward_seconds": forward_seconds,
                   "backward_seconds": backward_seconds, "update_seconds": update_seconds}


def _evaluate_case(case, models, codec, render, *, device, steps):
    target = PilotTarget.from_dict(case["target"])
    performance = Performance.from_dict(case["performance"])
    start = codec.canonical(PilotTarget(codec.defaults, tuple(0 for _ in codec.categorical)))
    case_id = case["case_id"]
    source_jobs = [(case_id + ":source", target, performance),
                   (case_id + ":source-repeat", target, performance),
                   (case_id + ":start", start, performance)]
    first = render(source_jobs)
    if any(audio is None for audio, _ in first):
        return {"case_id": case_id, "root": case["state_id"], "midi_id": case["midi_id"],
                "valid": False, "source_qa": [receipt.get("status") for _, receipt in first]}
    source, repeat, initial_audio = [audio for audio, _ in first]
    noise = audio_distances(np.stack([source]), np.stack([repeat]))[0]["mrstft"]
    start_error = audio_distances(np.stack([source]), np.stack([initial_audio]))[0]["mrstft"]
    device_target = _target_on_device(_measure(source), device)
    seed = int(canonical_sha256(case_id)[:12], 16)
    # Frame masks are kept on CPU by SpectralTargets; sample their indices with a
    # CPU generator even when the measured spectra and model live on the XPU.
    sampled, times = sample_frames(device_target, torch.Generator(device="cpu").manual_seed(seed))
    sampled = _target_on_device(sampled, device)
    sampled = prepare_spectral_target(sampled)
    played = performance_inputs(performance).to(device)[None]
    optimized, jobs, costs = {}, [], {}
    for backbone, model in models.items():
        patch, cost = _choose_patch(model, codec, sampled, times, played, steps=steps)
        optimized[backbone], costs[backbone] = patch, cost
        jobs.append((case_id + ":" + backbone, patch, performance))
    outcomes = render(jobs)
    results = {}
    for (backbone, patch), (audio, receipt) in zip(zip(models, optimized.values(), strict=True), outcomes, strict=True):
        results[backbone] = {"patch": patch.to_dict(), "optimizer": costs[backbone],
                             "native_mrstft": (audio_distances(np.stack([source]), np.stack([audio]))[0]["mrstft"]
                                               if audio is not None else None),
                             "native_qa": receipt.get("status")}
    return {"case_id": case_id, "root": case["state_id"], "midi_id": case["midi_id"],
            "midi_style": case["style"], "wet": bool(target.categorical[-1]), "valid": True,
            "target": target.to_dict(), "start": start.to_dict(), "start_native_mrstft": start_error,
            "repeat_noise": noise, "models": results}


def _bootstrap_interval(rows, values, *, samples=10000, seed=173):
    roots = sorted({row["root"] for row in rows})
    midis = sorted({row["midi_id"] for row in rows})
    root_index = {value: index for index, value in enumerate(roots)}
    midi_index = {value: index for index, value in enumerate(midis)}
    rng = np.random.default_rng(seed)
    root_counts = np.stack([np.bincount(rng.integers(len(roots), size=len(roots), dtype=np.int32),
                                        minlength=len(roots)) for _ in range(samples)])
    midi_styles = {row["midi_id"]: row.get("midi_style", "all") for row in rows}
    by_style = defaultdict(list)
    for midi in midis:
        by_style[midi_styles[midi]].append(midi_index[midi])
    midi_counts = np.zeros((samples, len(midis)), dtype=np.int16)
    sample_rows = np.arange(samples)
    for group in by_style.values():
        draws = rng.integers(len(group), size=(samples, len(group)), dtype=np.int32)
        for offset in range(len(group)):
            midi_counts[sample_rows, np.asarray(group)[draws[:, offset]]] += 1
    weights = root_counts[:, [root_index[row["root"]] for row in rows]] * \
        midi_counts[:, [midi_index[row["midi_id"]] for row in rows]]
    observed = np.asarray(values, dtype=np.float64)
    weighted = weights @ observed
    totals = weights.sum(axis=1)
    means = weighted[totals > 0] / totals[totals > 0]
    return np.quantile(means, [.025, .975]).tolist()


def _summarize(rows):
    summaries, valid_by_model = {}, {}
    common_rows = [row for row in rows if row.get("valid") and all(
        row.get("models", {}).get(name, {}).get("native_mrstft") is not None for name in BACKBONES)]
    for backbone in BACKBONES:
        complete = common_rows
        valid_by_model[backbone] = {row["case_id"]: row for row in complete}
        if not complete:
            summaries[backbone] = {"planned_cases": len(rows), "scored_cases": 0}
            continue
        scores = [row["models"][backbone]["native_mrstft"] for row in complete]
        roots = [{"root": row["root"], "midi_id": row["midi_id"], "midi_style": row["midi_style"]}
                 for row in complete]
        gains = [row["start_native_mrstft"] - score for row, score in zip(complete, scores, strict=True)]
        summaries[backbone] = {
            "planned_cases": len(rows), "scored_cases": len(complete), "patch_roots": len({row["root"] for row in complete}),
            "midi_performances": len({row["midi_id"] for row in complete}),
            "mean_start_native_mrstft": float(np.mean([row["start_native_mrstft"] for row in complete])),
            "mean_inverse_native_mrstft": float(np.mean(scores)),
            "inverse_native_mrstft_interval": _bootstrap_interval(roots, scores),
            "mean_inverse_change_from_default": float(np.mean(gains)),
            "mean_forward_seconds": float(np.mean([row["models"][backbone]["optimizer"]["forward_seconds"] for row in complete])),
            "mean_backward_seconds": float(np.mean([row["models"][backbone]["optimizer"]["backward_seconds"] for row in complete])),
            "mean_update_seconds": float(np.mean([row["models"][backbone]["optimizer"]["update_seconds"] for row in complete])),
            "mean_category_prefilter_seconds": float(np.mean([
                row["models"][backbone]["optimizer"]["category_prefilter_seconds"] for row in complete])),
        }
    common_ids = set.intersection(*(set(valid_by_model[name]) for name in BACKBONES))
    differences = {}
    for first_index, first_name in enumerate(BACKBONES):
        for second_name in BACKBONES[first_index + 1:]:
            paired_rows, deltas = [], []
            for case_id in sorted(common_ids):
                first, second = valid_by_model[first_name][case_id], valid_by_model[second_name][case_id]
                paired_rows.append({"root": first["root"], "midi_id": first["midi_id"], "midi_style": first["midi_style"]})
                deltas.append(first["models"][first_name]["native_mrstft"] -
                              second["models"][second_name]["native_mrstft"])
            differences[f"{first_name}_minus_{second_name}"] = {
                "paired_cases": len(deltas), "mean_mrstft_difference": float(np.mean(deltas)) if deltas else None,
                "interval": _bootstrap_interval(paired_rows, deltas, seed=174 + first_index)
                if deltas else None,
            }
    available = [name for name in BACKBONES if summaries[name].get("scored_cases")]
    best = min(available, key=lambda name: summaries[name]["mean_inverse_native_mrstft"]) if available else None
    clear = best is not None and len(available) == len(BACKBONES)
    if clear:
        for other in available:
            if other == best:
                continue
            key = f"{best}_minus_{other}" if BACKBONES.index(best) < BACKBONES.index(other) else f"{other}_minus_{best}"
            interval = differences[key]["interval"]
            clear &= interval is not None and (interval[1] < 0 if key.startswith(best + "_minus_") else interval[0] > 0)
    return {"backbones": summaries, "paired_inverse_differences": differences,
            "best_inverse_by_mean": best, "clear_inverse_winner": best if clear else None,
            "planned_cases": len(rows), "common_scored_cases": len(common_ids)}


def _publish_artifacts(output, result, plan):
    import csv
    import gzip
    import io

    destination = TRAINING / "reports" / "data"
    destination.mkdir(parents=True, exist_ok=True)

    def write_gzip(path, value):
        path.write_bytes(gzip.compress(value.encode("utf-8"), mtime=0))

    write_gzip(destination / "issue173_surrogate_model_selection_plan.json.gz",
               (output / "plan.json").read_text(encoding="utf-8"))
    write_gzip(destination / "issue173_surrogate_model_selection_controls.jsonl.gz",
               (output / "control-scope.jsonl").read_text(encoding="utf-8"))
    write_json(destination / "issue173_surrogate_model_selection_summary.json", result)
    fields = ("case_id", "patch_root", "midi_id", "midi_style", "wet", "backbone", "valid",
              "start_native_mrstft", "repeat_noise", "inverse_native_mrstft", "inverse_change_from_default",
              "surrogate_start_loss", "surrogate_best_loss", "candidate_categories", "refined_categories",
              "active_continuous_controls", "category_prefilter_seconds", "forward_seconds", "backward_seconds",
              "update_seconds", "native_qa")
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=fields)
    writer.writeheader()
    for path in sorted((output / "cases").glob("*.json")):
        case = _read_json(path)
        for backbone in BACKBONES:
            model = case.get("models", {}).get(backbone, {})
            metric = model.get("native_mrstft")
            gain = case["start_native_mrstft"] - metric if metric is not None and case.get("valid") else None
            writer.writerow({
                "case_id": case["case_id"], "patch_root": case.get("root"), "midi_id": case.get("midi_id"),
                "midi_style": case.get("midi_style"), "wet": case.get("wet"), "backbone": backbone,
                "valid": metric is not None and case.get("valid", False),
                "start_native_mrstft": case.get("start_native_mrstft"), "repeat_noise": case.get("repeat_noise"),
                "inverse_native_mrstft": metric, "inverse_change_from_default": gain,
                "surrogate_start_loss": model.get("optimizer", {}).get("start_surrogate_loss"),
                "surrogate_best_loss": model.get("optimizer", {}).get("best_surrogate_loss"),
                "candidate_categories": model.get("optimizer", {}).get("candidate_categories"),
                "refined_categories": model.get("optimizer", {}).get("refined_categories"),
                "active_continuous_controls": model.get("optimizer", {}).get("active_continuous_controls"),
                "category_prefilter_seconds": model.get("optimizer", {}).get("category_prefilter_seconds"),
                "forward_seconds": model.get("optimizer", {}).get("forward_seconds"),
                "backward_seconds": model.get("optimizer", {}).get("backward_seconds"),
                "update_seconds": model.get("optimizer", {}).get("update_seconds"),
                "native_qa": model.get("native_qa"),
            })
    write_gzip(destination / "issue173_surrogate_model_selection_cases.csv.gz", csv_buffer.getvalue())
    if (output / "validation-loss.csv").exists():
        write_gzip(destination / "issue173_surrogate_validation_loss.csv.gz",
                   (output / "validation-loss.csv").read_text(encoding="utf-8"))
    report_dir = TRAINING / "reports"
    figure_dir = report_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copyfile(output / "validation-loss-curves.png", figure_dir / "issue173_surrogate_validation_loss_curves.png")
    (report_dir / "VITAL_SURROGATE_MODEL_SELECTION.md").write_text(
        (output / "REPORT.md").read_text(encoding="utf-8"), encoding="utf-8")


def _inventory(output, codec, cases):
    continuous_names = {name for name, *_ in codec.continuous}
    category_names = {name for name, *_ in codec.categorical}
    category_aliases = {"filter_1_response": "filter_1_blend", "osc_1_waveform": "osc_1_wavetable_asset",
                        "osc_2_waveform": "osc_2_wavetable_asset"}
    category_keys = {category_aliases.get(name, name) for name in category_names}
    category_candidates = candidate_categories(codec)
    active = defaultdict(bool)
    for choices in category_candidates:
        cont_mask, _ = codec.masks(choices)
        for name, enabled in zip((name for name, *_ in codec.continuous), cont_mask, strict=True):
            active[name] |= enabled
    explored = {category_aliases.get(name, name): sorted({choices[index] for choices in category_candidates})
                for index, (name, _) in enumerate(codec.categorical)}
    inventory = write_control_inventory(output, {f"expanded_{name}": codec for name in BACKBONES})
    for rows in inventory.values():
        for row in rows:
            row["experiment_id"] = "issue173_surrogate_model_selection"
            name = row["control_key"]
            if name in continuous_names:
                row["changeable"] = active[name]
                row["output_type"] = "normalized_continuous_input"
                row["update_method"] = "projected Adam through the frozen surrogate"
                row["gradient_path"] = "known-MIDI stereo spectral/envelope loss to continuous value"
                row["training_signal"] = "real Vital stereo spectra and paired sound-change targets"
                if not active[name]:
                    row["fixed_value"] = row["default_value"]
                    row["fixed_reason"] = "inactive across the categorical search shortlist"
            elif name in category_keys:
                choices = explored.get(name, [])
                row.update(changeable=len(choices) > 1, fixed_value=choices[0] if len(choices) == 1 else None,
                           fixed_reason="category has one value in the deterministic legal-choice shortlist" if len(choices) == 1 else None,
                           candidate_values=choices, output_type="enumerated categorical conditioning input",
                           update_method="score 24 legal categories, refine the best 6 with continuous gradients",
                           gradient_path="none for categorical choices; discrete selection by surrogate loss",
                           training_signal="real Vital spectral targets condition the shared audio proxy; no category logits")
    with (output / "control-scope.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        import json
        for rows in inventory.values():
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    return inventory


def _scope_table(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["scope_family"]].append(row)
    lines = ["| Control family | Changeable / total | Surrogate input slots |", "|---|---:|---:|"]
    for name, group in groups.items():
        enabled = sum(bool(row["changeable"]) for row in group)
        slots = sum(row["model_input_width"] for row in group)
        lines.append(f"| {name} {'✅' if enabled else '❌'} | {enabled} / {len(group)} | {slots} |")
    lines.append(f"| **Total** | **{sum(bool(row['changeable']) for row in rows)} / {len(rows)}** | "
                 f"**{sum(row['model_input_width'] for row in rows)}** |")
    return lines


def _write_report(output, result, inventory, plan):
    fits, panels = result["fit"], result["backbones"]
    validation = {name: fits[name]["trace"][-1]["validation_loss"] for name in BACKBONES}
    proxy_best = min(validation, key=validation.get)
    inverse_best = result["best_inverse_by_mean"]
    table = ["| Model | Held-out proxy loss ↓ | Direct inverse MR-STFT ↓ | 95% crossed-cluster CI | Default start MR-STFT ↓ | Reduction vs default ↑ | Cases |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name in BACKBONES:
        row = panels[name]
        if not row.get("scored_cases"):
            table.append(f"| {name} | {validation[name]:.4f} | — | — | — | — | 0/{row['planned_cases']} |")
            continue
        interval = row["inverse_native_mrstft_interval"]
        table.append(f"| {name} | {validation[name]:.4f} | {row['mean_inverse_native_mrstft']:.4f} | "
                     f"[{interval[0]:.4f}, {interval[1]:.4f}] | {row['mean_start_native_mrstft']:.4f} | "
                     f"{row['mean_inverse_change_from_default']:+.4f} | {row['scored_cases']}/{row['planned_cases']} |")
    inverse_decision = (f"**{result['clear_inverse_winner']}** had a paired advantage over both alternatives whose 95% intervals excluded zero."
                        if result["clear_inverse_winner"] else
                        f"**{inverse_best}** had the lowest mean inverse error, but paired uncertainty does not establish a clear winner.")
    comparison_lines = [
        "| Paired contrast (first model minus second; lower is better) | Mean MR-STFT difference | 95% crossed-cluster interval |",
        "|---|---:|---:|",
    ]
    for label, difference in result["paired_inverse_differences"].items():
        interval = difference["interval"]
        rendered_interval = f"[{interval[0]:+.4f}, {interval[1]:+.4f}]" if interval else "not estimable"
        comparison_lines.append(f"| {label.replace('_minus_', ' vs ')} | {difference['mean_mrstft_difference']:+.4f} | {rendered_interval} |")
    sample = plan["evaluation"]
    val_sample = result["validation_sample"]
    opening = (f"The **{proxy_best}** had the lowest final held-out proxy loss; {inverse_decision} "
               "This compares residual MLP, Transformer, and Conformer for forward audio prediction and direct patch recovery.")
    lines = [
        "# Vital surrogate model selection", "", opening, "",
        "Lower held-out proxy loss means more accurate predicted stereo spectra/envelopes. Lower real-Vital MR-STFT means the synthesized audio is closer to the target.", "",
        "## Primary results", "", *table, "",
        f"The direct inverse panel contains **{sample['patch_roots']} fresh patch roots × 2 MIDI performances** = {result['planned_cases']} planned cases, "
        f"using {sample['distinct_midi_performances']} distinct performances "
        f"({result['planned_cases'] // sample['distinct_midi_performances']} cases per performance). "
        f"The independent proxy validation set contains **{val_sample['patch_roots']} different roots × 2 performances** = {val_sample['pair_count']} accepted paired targets "
        f"({val_sample['distinct_midi_performances']} distinct performances; {val_sample['clip_count']} accepted rendered clips). Both sets use exact known MIDI; "
        f"all final-panel performances are new to training. The {sample['unused_test_roots']} pool test roots remain untouched.", "",
        "The inverse starts from the same default patch for every model. It scores the same deterministic shortlist of 24 legal categorical combinations from target audio, "
        "keeps the best 6, then performs the same bounded continuous Adam search for each. Target category labels are not given to the optimizer.", "",
        "### Validation loss curves", "",
        "Each point shows a 100-update moving average of training loss and full-set held-out validation loss. The validation objective is spectral/envelope error plus "
        "0.25 × paired control-change error, measured on the same 128 held-out patch/MIDI pairs at each checkpoint.", "",
        "![Training and held-out validation loss curves](figures/issue173_surrogate_validation_loss_curves.png)", "",
        f"Lowest final proxy loss: **{proxy_best}**. Lowest mean native inverse MR-STFT: **{inverse_best}**. These answer different questions; if they differ, a single winner is not established.", "",
        "### Matched inverse-synthesis comparisons", "", *comparison_lines, "",
        "Intervals use 10,000 crossed-cluster bootstrap draws: patch roots are resampled independently, and MIDI performances are resampled within style. A negative contrast favors the first model. "
        "They describe uncertainty across the sampled patches and performances, not new renderer versions or every hand-authored Vital preset.", "",
        "## What this establishes", "",
        "This is a direct model-selection test under one end-to-end inverse-synthesis policy, conditional on accurate MIDI. It measures which proxy leads to lower real-Vital audio error after discrete category selection and continuous refinement. "
        "The held-out curves separately measure forward audio prediction during training. Neither result evaluates MIDI transcription errors or performance on an independently authored preset population.", "",
        "The expanded codec covers 21 Vital controls: 13 normalized continuous controls and 8 categorical controls. The model receives 33 conditioning slots and predicts audio features; it has no control-logit head. "
        "The inverse system selects categorical values by evaluating legal combinations and uses gradients only for continuous controls.", "",
        "## Expanded Vital control scope", "",
    ]
    for backbone in BACKBONES:
        lines += [f"### {backbone} control scope", "", *_scope_table(inventory[f"expanded_{backbone}"]), ""]
    lines += [
        "[Exact per-control inventory](data/issue173_surrogate_model_selection_controls.jsonl.gz)", "",
        "## Execution and reproducibility", "",
        f"All three models were initialized from scratch with seed {plan['training']['seed']} and trained for {plan['training']['steps']:,} updates on the same 512 expanded-codec roots. "
        f"Training reused the accepted cached Vital pairs: {plan['training']['accepted_pairs']['basic']['accepted']:,} basic and "
        f"{plan['training']['accepted_pairs']['expanded']['accepted']:,} expanded. Validation required {2 * val_sample['pair_count']} fresh native renders; the inverse panel required up to "
        f"up to {3 * result['planned_cases']} optimized-patch renders plus shared target, repeat, and default-start renders.", "",
        "[Loss-curve data](data/issue173_surrogate_validation_loss.csv.gz) · "
        "[Per-case inverse measurements](data/issue173_surrogate_model_selection_cases.csv.gz) · "
        "[Aggregate metrics](data/issue173_surrogate_model_selection_summary.json) · "
        "[Frozen plan and data provenance](data/issue173_surrogate_model_selection_plan.json.gz)", "",
        "Next, confirm the selected model on fresh patches and performances with transcribed MIDI, since transcription is part of the intended inverse-synthesis path.", "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def _write_loss_curves(output, fits):
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fields = ("backbone", "step", "train_loss", "validation_loss", "train_spectral_envelope_error",
              "validation_spectral_envelope_error", "train_control_change_error", "validation_control_change_error",
              "validation_clips", "validation_pairs")
    rows = []
    for backbone in BACKBONES:
        rows.extend({"backbone": backbone, **record} for record in fits[backbone]["trace"])
    with (output / "validation-loss.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    figure, axes = plt.subplots(1, len(BACKBONES), figsize=(15, 4.5), sharey=True)
    for axis, backbone in zip(axes, BACKBONES, strict=True):
        trace = fits[backbone]["trace"]
        axis.plot([row["step"] for row in trace if row["train_loss"] is not None],
                  [row["train_loss"] for row in trace if row["train_loss"] is not None],
                  label="Training (100-update mean)", linestyle="--")
        axis.plot([row["step"] for row in trace], [row["validation_loss"] for row in trace], label="Validation")
        axis.set_title(backbone.replace("_", " ").title())
        axis.set_xlabel("Optimizer updates")
        axis.grid(True, alpha=.25)
    axes[0].set_ylabel("Spectral/envelope + paired-change loss (lower is better)")
    axes[-1].legend(loc="best")
    figure.suptitle("Vital surrogate training and held-out validation loss")
    figure.tight_layout()
    figure.savefig(output / "validation-loss-curves.png", dpi=180)
    plt.close(figure)


def run(args):
    started = perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    codec = SurrogateCodec(PilotCodec(), expanded=True)
    plan_id, plan, rows, cases, validation_rows = _plan(args, codec)
    inventory = _inventory(args.output, codec, cases)
    if args.plan_only:
        print(json.dumps({"plan_id": plan_id, "training_pairs": plan["training"]["accepted_pairs"],
                          "training_roots": plan["training"]["training_roots"],
                          "evaluation_roots": len({case["state_id"] for case in cases}),
                          "evaluation_cases": len(cases),
                          "distinct_midi": len({case["midi_id"] for case in cases}),
                          "validation_roots": len({row["state_id"] for row in validation_rows}),
                          "validation_pairs": len(validation_rows),
                          "validation_clips": 2 * len(validation_rows),
                          "backbones": BACKBONES,
                          "reserved_test_roots_untouched": True}, indent=2))
        return
    progress = Progress(args.output)
    progress.stage_count = len(BACKBONES) + 3
    progress.set("Rendering independent proxy validation pairs", len(validation_rows), 1)
    with native_session(args.output, codec, args.workers) as render:
        validation_rows, validation_collection = collect(
            args.output, validation_rows, render, progress, pair_dir=args.output / "validation-pairs")
    models, fit_summaries = {}, {}
    for index, backbone in enumerate(BACKBONES):
        progress.set(f"Expanded surrogate training · {backbone}", args.steps, index + 2)
        models[backbone], fit_summaries[backbone] = fit(
            args.output, "expanded-independent-seed", backbone, codec, rows,
            steps=args.steps, plan_id=plan_id, device=args.device, progress=progress, seed=args.seed,
            warm_start=False, pairs_dir=args.pilot_run / "pairs", validation_rows=validation_rows,
            validation_pairs_dir=args.output / "validation-pairs", validation_every=args.validation_every)
    _write_loss_curves(args.output, fit_summaries)
    progress.set("Direct inverse-synthesis panel", len(cases), len(BACKBONES) + 2)
    case_dir = args.output / "cases"
    case_dir.mkdir(exist_ok=True)
    case_rows = []
    with native_session(args.output, codec, args.workers) as render:
        for index, case in enumerate(cases, 1):
            path = case_dir / (case["case_id"] + ".json")
            case_hash = canonical_sha256(case)
            if path.exists():
                saved = _read_json(path)
                if saved.get("case_hash") != case_hash:
                    raise ValueError(f"saved case differs from frozen plan: {case['case_id']}")
                row = saved
            else:
                row = _evaluate_case(case, models, codec, render, device=args.device, steps=args.gradient_steps)
                row["case_hash"] = case_hash
                write_json(path, row)
            if not row.get("valid"):
                progress.failures += 1
            elif any(value["native_mrstft"] is None for value in row["models"].values()):
                progress.failures += sum(value["native_mrstft"] is None for value in row["models"].values())
            case_rows.append(row)
            progress.update(index)
    summary = _summarize(case_rows)
    result = {**summary, "plan_id": plan_id, "fit": fit_summaries,
              "validation_collection": validation_collection,
              "validation_sample": {"patch_roots": len({row["state_id"] for row in validation_rows}),
                                    "pair_count": len(validation_rows),
                                    "clip_count": 2 * len(validation_rows),
                                    "distinct_midi_performances": len({row["midi_id"] for row in validation_rows})},
              "elapsed_seconds": perf_counter() - started,
              "planned_cases": len(cases)}
    write_json(args.output / "summary.json", result)
    _write_report(args.output, result, inventory, plan)
    _publish_artifacts(args.output, result, plan)
    progress.set("Complete", 1, len(BACKBONES) + 3)
    progress.update(1, state="complete", message="Summary and report saved")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-run", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--reservation", type=Path, default=TRAINING / "reports" / "data" / "basic_confirmation_reservation.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--gradient-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=174)
    parser.add_argument("--midi-seed", type=int, default=175)
    parser.add_argument("--validation-midi-seed", type=int, default=176)
    parser.add_argument("--validation-every", type=int, default=100)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

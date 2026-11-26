"""Compare held-out surrogate predictions and real-Vital inversion."""

from collections import defaultdict
import json
from time import perf_counter

import numpy as np
import torch

from obruxo_data.hashing import canonical_sha256, sha256_bytes
from obruxo_data.midi import Performance

from .audit import write_json
from .codec import PilotTarget
from .evaluate import audio_distances
from .proxy import performance_inputs
from .spectral import SpectralTargets, prepare_spectral_target, spectral_loss
from .surrogate_pilot import control_tensor
from .temporal_proxy import full_times, sample_frames


def _expanded_batch(value, size):
    if isinstance(value, dict):
        return {key: _expanded_batch(item, size) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_expanded_batch(item, size) for item in value)
    return value.expand(size, *value.shape[1:])


def candidate_categories(codec):
    legal = codec.legal_choices()
    if len(legal) <= 24:
        return legal
    rng = np.random.default_rng(172)
    order = rng.permutation(len(legal))
    choices = np.array(legal, dtype=np.int64)
    picked = [0]
    distance = (choices != choices[0]).sum(1)
    for _ in range(23):
        selection = int(order[distance[order].argmax()])
        picked.append(selection)
        distance = np.minimum(distance, (choices != choices[selection]).sum(1))
    return tuple(legal[index] for index in picked)


def _choose_patches(model, model_codec, search_codec, target, times, played, *, seed, steps):
    device = played.device
    categorical = candidate_categories(search_codec)
    starts = [search_codec.canonical(PilotTarget(search_codec.defaults, choices)) for choices in categorical]
    modeled = [model_codec.from_basic(patch) if model_codec.expanded and not search_codec.expanded else patch for patch in starts]
    encoded = control_tensor(model_codec, modeled, device)

    def score(tensor):
        count = len(tensor)
        measured = _expanded_batch(target, count)
        return spectral_loss(model(tensor, played.expand(count, -1),
                                   _expanded_batch(times, count), measured["lengths"]), measured)["loss"]

    with torch.no_grad():
        baseline = torch.cat([score(group) for group in encoded.split(6)])
    indices = baseline.argsort(stable=True)[:6].tolist()
    chosen, template = [starts[i] for i in indices], encoded[indices]
    width = len(search_codec.continuous)
    initial = template[:, :width].detach()
    mask = torch.tensor([search_codec.masks(start.categorical)[0] for start in chosen], device=device)
    radius = .15

    def project(value):
        bounded = torch.maximum(torch.minimum(value, initial + radius), initial - radius).clamp(0, 1)
        return torch.where(mask, bounded, initial)

    def scores(values):
        return score(torch.cat((values, template[:, width:]), dim=-1))

    best = baseline[indices].detach()
    best_gradient, best_random = initial.clone(), initial.clone()
    gradient_losses, random_losses = best.clone(), best.clone()
    controls = initial.clone().requires_grad_()
    optimizer = torch.optim.Adam([controls], lr=.03)
    random = torch.Generator().manual_seed(seed)
    forward_seconds = backward_seconds = 0.

    def record_gradient(value):
        nonlocal gradient_losses
        improve = value < gradient_losses
        best_gradient[improve] = controls[improve]
        gradient_losses = torch.minimum(value, gradient_losses)

    for _ in range(steps):
        start = perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = scores(controls)
        middle = perf_counter()
        loss.sum().backward()
        with torch.no_grad():
            record_gradient(loss.detach())
        optimizer.step()
        forward_seconds += middle - start
        backward_seconds += perf_counter() - middle
        with torch.no_grad():
            controls.copy_(project(controls))
            alternative = project(initial + (torch.rand(initial.shape, generator=random).to(device) * 2 - 1) * radius)
            value = scores(alternative)
            improve = value < random_losses
            best_random[improve] = alternative[improve]
            random_losses = torch.minimum(value, random_losses)
    with torch.no_grad():
        record_gradient(scores(controls))

    def winner(best_values, losses):
        idx = int(losses.argmin())
        patch = search_codec.canonical(PilotTarget(tuple(best_values[idx].detach().cpu().tolist()), chosen[idx].categorical))
        return patch, float(losses[idx])

    return {"gradient": winner(best_gradient, gradient_losses), "forward_only": winner(best_random, random_losses),
            "cost": {"forward_seconds": forward_seconds, "backward_seconds": backward_seconds,
                     "start_surrogate_loss": float(baseline[0]),
                     "gradient_steps": steps, "category_pool": len(categorical), "category_starts": len(chosen)}}


def _measure(audio):
    return SpectralTargets()(torch.from_numpy(audio.copy()).T[None])


def _target_on_device(target, device):
    return {key: tuple(item.to(device) for item in value) if isinstance(value, tuple) else value.to(device)
            for key, value in target.items()}


def evaluate_panel(output, label, cases, models, model_codec, search_codec, render, *, device, steps, progress):
    directory = output / "cases" / label
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in cases:
        case_id = case["case_id"]
        path = directory / (case_id + ".json")
        receipt_id = canonical_sha256(case)
        if path.exists():
            row = json.loads(path.read_text(encoding="utf-8"))
            if row["case_hash"] != receipt_id:
                raise ValueError("saved evaluation case differs from the frozen case")
        else:
            basic_audio = output / "cases" / "basic-source" / (case_id + ".npy") if label != "expanded" else None
            prior = None
            if label == "preservation":
                prior = json.loads((output / "cases/basic" / path.name).read_text(encoding="utf-8"))
                if prior["case_hash"] != receipt_id:
                    raise ValueError("preservation source differs from the basic panel")
            row = _evaluate_case(case, models, model_codec, search_codec, render, device, steps, basic_audio, prior)
            row["case_hash"] = receipt_id
            write_json(path, row)
        if not row.get("valid"):
            progress.failures += 1
        rows.append(row)
        progress.update(len(rows))
    return summarize(rows, tuple(models))


def _evaluate_case(case, models, model_codec, search_codec, render, device, steps, basic_audio, prior):
    case_id = case["case_id"]
    target = PilotTarget.from_dict(case["target"])
    performance = Performance.from_dict(case["performance"])
    start = search_codec.from_basic(search_codec.basic.default_target())
    row = {"case_id": case_id, "root": case["state_id"], "midi_id": case["midi_id"], "midi_role": case["midi_role"],
           "wet": bool(target.categorical[-1]) if search_codec.expanded else False, "valid": False, "models": {}}
    if prior is not None:
        if "source_audio_sha256" not in prior:
            row["source_qa"] = prior["source_qa"]
            return row
        source = np.load(basic_audio, allow_pickle=False)
        if sha256_bytes(source.tobytes()) != prior["source_audio_sha256"]:
            raise ValueError("basic panel audio changed")
        row.update(start=prior["start"], repeat_noise=prior["repeat_noise"])
    else:
        outputs = render([(case_id + ":" + name, patch, performance)
                          for name, patch in (("source", target), ("repeat", target), ("start", start))])
        audio = [value for value, _ in outputs]
        if any(value is None for value in audio):
            row["source_qa"] = [meta["status"] for _, meta in outputs]
            return row
        source, repeat, starting = audio
        distances = audio_distances(np.stack([source] * 2), np.stack([repeat, starting]))
        row.update(start=distances[1]["mrstft"], repeat_noise=distances[0]["mrstft"])
        if basic_audio is not None:
            basic_audio.parent.mkdir(exist_ok=True)
            with basic_audio.with_suffix(".partial").open("wb") as stream:
                np.save(stream, source, allow_pickle=False)
            basic_audio.with_suffix(".partial").replace(basic_audio)
            row["source_audio_sha256"] = sha256_bytes(source.tobytes())
    full = _target_on_device(_measure(source), device)
    exact_times = full_times(full)
    seed = int(case["midi_id"][:12], 16)
    sampled, times = sample_frames(full, torch.Generator(device=device).manual_seed(seed))
    sampled = prepare_spectral_target(sampled)
    played = performance_inputs(performance).to(device)[None]
    model_target = model_codec.from_basic(target) if model_codec.expanded and not search_codec.expanded else target
    encoded = control_tensor(model_codec, [model_target], device)
    jobs, measured = [], {}
    for backbone, model in models.items():
        with torch.no_grad():
            predicted = model(encoded, played, exact_times, full["lengths"])
            absolute = float(spectral_loss(predicted, full)["loss"].item())
        candidates = _choose_patches(model, model_codec, search_codec, sampled, times, played, seed=seed, steps=steps)
        model_row = {"forward_error": absolute, **candidates.pop("cost")}
        for method, (patch, loss) in candidates.items():
            model_row[method] = {"target": patch.to_dict(), "surrogate_loss": loss, "native_mrstft": None}
            jobs.append((case_id + ":" + backbone + ":" + method, patch, performance))
            measured[jobs[-1][0]] = model_row[method]
        row["models"][backbone] = model_row
    outcomes = render(jobs)
    accepted = [(job, value) for job, (value, _) in zip(jobs, outcomes, strict=True) if value is not None]
    if accepted:
        scores = audio_distances(np.stack([source] * len(accepted)), np.stack([value for _, value in accepted]))
        for (job, _), score in zip(accepted, scores, strict=True):
            measured[job[0]]["native_mrstft"] = score["mrstft"]
    row.update(valid=len(accepted) == len(jobs))
    return row


def summarize(rows, backbones):
    grouped = {"seen": [row for row in rows if row["midi_role"] == "seen"],
               "new": [row for row in rows if row["midi_role"] == "new"],
               "wet": [row for row in rows if row["wet"]], "dry": [row for row in rows if not row["wet"]]}
    results = {}
    for backbone in backbones:
        measurements = {}
        for role, planned in grouped.items():
            complete = [row for row in planned if row["valid"]]
            if not complete:
                measurements[role] = {"planned": len(planned), "scored": 0}
                continue
            native = [row["models"][backbone]["gradient"]["native_mrstft"] for row in complete]
            forward = [row["models"][backbone]["forward_only"]["native_mrstft"] for row in complete]
            gains = [row["start"] - value for row, value in zip(complete, native, strict=True)]
            roots = defaultdict(list)
            for row, gain in zip(complete, gains, strict=True):
                roots[row["root"]].append(gain)
            root_means = np.array([np.mean(values) for values in roots.values()])
            resampled = np.random.default_rng(172).choice(root_means, size=(1000, len(roots))).mean(1)
            measurements[role] = {"planned": len(planned), "scored": len(complete), "roots": len(roots),
                                  "midi": len({row["midi_id"] for row in complete}),
                                  "improved": sum(gain > max(1e-5, row["repeat_noise"])
                                                  for row, gain in zip(complete, gains, strict=True)),
                                  "misleading": sum(gain < -max(1e-5, row["repeat_noise"]) and
                                                    row["models"][backbone]["gradient"]["surrogate_loss"] <
                                                    row["models"][backbone]["start_surrogate_loss"]
                                                    for row, gain in zip(complete, gains, strict=True)),
                                  "mean_native_improvement": float(np.mean(gains)),
                                  "improvement_interval": np.quantile(resampled, [.025, .975]).tolist(),
                                  "mean_gain_over_forward_only": float(np.mean(np.array(forward) - native)),
                                  "mean_forward_error": float(np.mean([row["models"][backbone]["forward_error"] for row in complete])),
                                  "mean_forward_seconds": float(np.mean([row["models"][backbone]["forward_seconds"] for row in complete])),
                                  "mean_backward_seconds": float(np.mean([row["models"][backbone]["backward_seconds"] for row in complete]))}
        results[backbone] = measurements
    passing = [backbone for backbone, groups in results.items()
               if all(groups[role]["scored"] and groups[role]["improved"] >= .6 * groups[role]["planned"]
                      and groups[role]["mean_native_improvement"] > 0 for role in ("seen", "new"))
               and groups["new"]["mean_gain_over_forward_only"] > 0]
    winner = max(passing, key=lambda name: results[name]["new"]["mean_native_improvement"], default=None)
    return {"backbones": results, "candidate": winner, "passing": passing, "scorable_cases": sum(row["valid"] for row in rows),
            "planned_cases": len(rows)}

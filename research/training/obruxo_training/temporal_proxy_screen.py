"""Frozen local helper experiment: bounded feature collection, learning and native qualification."""

from __future__ import annotations

import json
from pathlib import Path
from time import time

import numpy as np
import torch

from obruxo_data.hashing import canonical_sha256, sha256_bytes
from obruxo_data.midi import Performance

from .audit import write_json
from .benchmark import MemoryMonitor
from .codec import PilotTarget, load_codec
from .evaluate import audio_distance
from .features import tensor_digest
from .proxy import performance_inputs
from .spectral import SpectralTargets, require_spectral_contract, spectral_loss
from .spectral_controls import BasicControlAdapter
from .temporal_proxy import (TemporalSpectralProxy, collate_frames, full_times, guide_patch, hard_controls,
                             perturbed_targets, temporal_proxy_spec)
from .temporal_proxy_data import NativeSession, RecordCache, TemporalProxyConfig, collect_frames, make_plan
from .trainer import atomic_torch_save


def fit_temporal_proxy(cache, adapter, config, contract, *, saved=None, checkpoint=None, after_step=None, check_limits=None):
    torch.manual_seed(config.seed)
    model = TemporalSpectralProxy()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator().manual_seed(config.seed + 41)
    step, trace = 0, []
    if saved is not None:
        require_spectral_contract(saved["contract"], contract)
        if tensor_digest(saved["model"]) != saved["model_sha256"]:
            raise ValueError("helper model checkpoint changed")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        generator.set_state(saved["batch_rng"])
        torch.set_rng_state(saved["torch_rng"])
        step, trace = saved["step"], saved["trace"]
        if not 0 <= step <= config.steps:
            raise ValueError("invalid helper checkpoint step")
    def state():
        return {"format": "basic_temporal_helper_training_v1", "contract": contract, "step": step,
                "model": model.state_dict(), "model_sha256": tensor_digest(model.state_dict()),
                "optimizer": optimizer.state_dict(), "batch_rng": generator.get_state(), "torch_rng": torch.get_rng_state(), "trace": trace}
    if len(cache) < 1:
        raise ValueError("helper training needs accepted records")
    while step < config.steps:
        if check_limits:
            check_limits()
        indices = torch.randint(len(cache), (config.batch_size,), generator=generator).tolist()
        records = [cache[index] for index in indices]
        target, times = collate_frames(records)
        controls = hard_controls(adapter, [PilotTarget.from_dict(record["target"]) for record in records])
        played = torch.stack([record["performance"] for record in records])
        optimizer.zero_grad(set_to_none=True)
        terms = spectral_loss(model(controls, played, times), target)
        loss = terms["loss"].mean()
        if not torch.isfinite(loss):
            raise ValueError("nonfinite helper training objective")
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("missing or nonfinite helper parameter gradient")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        step += 1
        if step == 1 or step % config.checkpoint_every == 0 or step == config.steps:
            trace.append({"step": step, **{name: float(value.detach().mean()) for name, value in terms.items()}})
            if checkpoint:
                checkpoint(state())
        if after_step:
            after_step(step)
    return model.freeze(), state()


def interpolate_frames(record, query_times):
    output = []
    for values, old, new in zip((*record["measurements"]["magnitudes"], record["measurements"]["envelope"]),
                                record["times"], query_times, strict=True):
        right = torch.searchsorted(old[0], new[0]).clamp(1, old.shape[-1] - 1)
        left = right - 1
        weight = ((new[0] - old[0, left]) / (old[0, right] - old[0, left]).clamp_min(1e-7)).clamp(0, 1)
        result = values[..., left] * (1 - weight) + values[..., right] * weight
        result = result * (new < record["measurements"]["lengths"][:, None] / 44100).reshape(1, *([1] * (values.ndim - 2)), -1)
        output.append(result)
    return {"magnitudes": tuple(output[:3]), "envelope": output[3]}


class ForwardBaselines:
    def __init__(self, cache, adapter):
        self.cache, self.adapter = cache, adapter
        self.mean, controls, performances = None, [], []
        for i in range(len(cache)):
            record = cache[i]
            values = [value.mean(-1, keepdim=True) for value in (*record["measurements"]["magnitudes"], record["measurements"]["envelope"])]
            self.mean = values if self.mean is None else [old + value for old, value in zip(self.mean, values, strict=True)]
            controls.append(hard_controls(adapter, [PilotTarget.from_dict(record["target"])]))
            performances.append(record["performance"])
        self.mean = [value / len(cache) for value in self.mean]
        self.controls, self.performances = torch.cat(controls), torch.stack(performances)

    def predict(self, target, played, times):
        controls = hard_controls(self.adapter, [target])
        distances = (self.controls - controls).square().mean(-1) + (self.performances - played).square().mean(-1)
        indices = torch.argsort(distances, stable=True)[:5]
        nearby = [interpolate_frames(self.cache[int(index)], times) for index in indices]
        weights = 1 / distances[indices].clamp_min(1e-6)
        weights = weights / weights.sum()
        mean = [value.expand(*value.shape[:-1], time.shape[-1]) for value, time in zip(self.mean, times, strict=True)]
        mixture = {"magnitudes": tuple(sum(weight * row["magnitudes"][r] for weight, row in zip(weights, nearby, strict=True)) for r in range(3)),
                   "envelope": sum(weight * row["envelope"] for weight, row in zip(weights, nearby, strict=True))}
        return {"training_mean": {"magnitudes": tuple(mean[:3]), "envelope": mean[3]},
                "nearest_training_record": nearby[0], "five_neighbor_interpolation": mixture}


def qualify_case(helper, adapter, baselines, entry, index, role, performance, output, session, plan_id):
    path = output / f"case-{index:03d}-{role}.pt"
    identity = {"plan_id": plan_id, "helper_sha256": tensor_digest(helper.state_dict()), "state_id": entry.state_id,
                "performance_id": canonical_sha256(performance.to_dict()), "case": index, "role": role}
    saved = torch.load(path, map_location="cpu", weights_only=True) if path.exists() else {"identity": identity, "rows": {}}
    require_spectral_contract(saved["identity"], identity)
    measure, played = SpectralTargets(), performance_inputs(performance)
    candidates = perturbed_targets(adapter.codec, entry.target, index)
    methods = ("source", "repeat", "start", "plus", "minus", "category", "pan", "detune", "continuous", "joint")
    for method in methods:
        session.check_limits()
        if method in saved["rows"]:
            continue
        guide = None
        if method in {"continuous", "joint"}:
            candidate, guide = guide_patch(helper, adapter, candidates["start"], played, saved["target"], joint=method == "joint",
                                          seed=1000 + index * 2 + (role == "transfer"))
        else:
            candidate = entry.target if method in {"source", "repeat"} else candidates[method]
        audio, receipt = session.render(candidate, performance, "qualification", f"{index}:{role}:{method}")
        if audio is None:
            raise RuntimeError("qualification render rejected; helper remains unqualified")
        row = {"method": method, "receipt": receipt, "candidate": candidate.to_dict()}
        current = measure(torch.from_numpy(audio.copy()).T[None])
        if method == "source":
            saved.update(audio=torch.from_numpy(audio.copy()), target=current)
            with torch.no_grad():
                times = full_times(current)
                estimates = {"helper": helper(hard_controls(adapter, [candidate]), played[None], times),
                             **baselines.predict(candidate, played, times)}
                row["forward_errors"] = {name: float(spectral_loss(value, current)["loss"].item()) for name, value in estimates.items()}
        else:
            row["native_mrstft"] = audio_distance(saved["audio"].numpy(), audio)["mrstft"]
            row["native_spectral"] = float(spectral_loss(current, saved["target"])["loss"].item())
            with torch.no_grad():
                estimate = helper(hard_controls(adapter, [candidate]), played[None], full_times(saved["target"]))
                row["helper_error"] = float(spectral_loss(estimate, saved["target"])["loss"].item())
            if method == "repeat":
                row["repeat_exact"] = bool(np.array_equal(saved["audio"].numpy(), audio))
                if not row["repeat_exact"]:
                    raise RuntimeError("fixed-phase qualification repeat failed")
        if guide is not None:
            row["guide"] = guide
        saved["rows"][method] = row
        atomic_torch_save(path, saved)
        if sum(file.stat().st_size for file in output.rglob("*.pt")) > session.plan["limits"]["archive_bytes"]:
            raise RuntimeError("helper feature/evaluation archive budget exceeded")
    return {**identity, "rows": saved["rows"]}


def qualification_summary(cases, plan):
    results, passed = {}, True
    for role in ("reference", "transfer"):
        selected = [case for case in cases if case["role"] == role]
        if len(selected) != plan["config"]["validation_states"]:
            return {"qualified_for_inverse_training": False, "failure": "incomplete qualification"}
        names = ("helper", *plan["baselines"])
        forward = {name: float(np.mean([case["rows"]["source"]["forward_errors"][name] for case in selected])) for name in names}
        ratio = .9 if role == "reference" else 1.
        forward_pass = forward["helper"] <= ratio * min(forward[name] for name in plan["baselines"])
        rankings = {}
        for group, methods in (("continuous", ("plus", "minus")), ("categorical", ("category",)), ("pan", ("pan",)), ("detune", ("detune",))):
            agreements, ties, planned, inactive = 0, 0, 0, 0
            for case in selected:
                rows = case["rows"]
                if group == "detune" and rows["start"]["candidate"]["categorical"][0] == 0:
                    inactive += 1
                    continue
                tolerance = max(1e-7, rows["repeat"]["native_mrstft"])
                for method in methods:
                    planned += 1
                    native = rows[method]["native_mrstft"] - rows["start"]["native_mrstft"]
                    proxy = rows[method]["helper_error"] - rows["start"]["helper_error"]
                    if abs(native) <= tolerance:
                        ties += 1
                    else:
                        agreements += int(native * proxy > 0)
            agreement = agreements / planned if planned else 0.
            rankings[group] = {"agree": agreements, "ties": ties, "planned": planned, "inactive": inactive,
                               "agreement": agreement, "pass": agreement >= .7}
        guidance = {}
        for method in ("continuous", "joint"):
            delta = [case["rows"][method]["native_mrstft"] - case["rows"]["start"]["native_mrstft"] for case in selected]
            spectral_delta = [case["rows"][method]["native_spectral"] - case["rows"]["start"]["native_spectral"] for case in selected]
            improved = sum(value < -max(1e-7, case["rows"]["repeat"]["native_mrstft"]) for value, case in zip(delta, selected, strict=True))
            random = np.random.default_rng(0)
            interval = np.quantile(np.mean(random.choice(delta, (2000, len(delta))), axis=1), [.025, .975]).tolist()
            guidance[method] = {"improved": improved, "planned": len(selected), "mean_native_delta": float(np.mean(delta)),
                                "mean_spectral_delta": float(np.mean(spectral_delta)), "paired_family_interval": interval,
                                "category_changes": sum(case["rows"][method]["guide"]["category_changed"] for case in selected),
                                "pass": bool(improved / len(selected) >= .6 and np.mean(delta) < 0 and np.mean(spectral_delta) < 0)}
        role_pass = forward_pass and all(row["pass"] for row in rankings.values()) and all(row["pass"] for row in guidance.values())
        passed = passed and role_pass
        results[role] = {"forward_errors": forward, "forward_pass": forward_pass, "rankings": rankings, "guidance": guidance, "pass": bool(role_pass)}
    return {"qualified_for_inverse_training": bool(passed), "roles": results, "claim": "bounded_helper_feasibility_not_inverse_quality"}


def run_temporal_proxy_screen(pool, output, *, config=None, resume=False, renderer_factory=None, progress=None):
    config = config or TemporalProxyConfig()
    torch.set_num_threads(2)
    plan, recipes, validation = make_plan(pool, config)
    plan_id = canonical_sha256(plan)
    if resume:
        require_spectral_contract(json.loads((output / "plan.json").read_text()), plan)
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "plan.json", plan)
        write_json(output / "run.json", {"started_unix": time(), "plan_id": plan_id})
    monitor = MemoryMonitor()
    monitor.start()
    session = NativeSession(output, plan, pool.codec, monitor, renderer_factory)
    failure, result = None, None
    try:
        rows = collect_frames(pool, output, recipes, session, progress)
        cache, adapter = RecordCache(output, rows), BasicControlAdapter(pool.codec)
        contract = {"plan_id": plan_id, "records_sha256": sha256_bytes((output / "records.json").read_bytes()), "adapter": adapter.contract()}
        checkpoint_path = output / "checkpoint.pt"
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True) if checkpoint_path.exists() else None
        def checkpoint(value):
            atomic_torch_save(checkpoint_path, value)
            if progress:
                progress({"stage": "train", "step": value["step"], **value["trace"][-1]})
        helper, training = fit_temporal_proxy(cache, adapter, config, contract, saved=saved, checkpoint=checkpoint, check_limits=session.check_limits)
        baselines = ForwardBaselines(cache, adapter)
        cases = []
        for i, entry in enumerate(validation):
            for role in ("reference", "transfer"):
                performance = Performance.from_dict(plan["validation_performances"][i][role]["performance"])
                case = qualify_case(helper, adapter, baselines, entry, i, role, performance, output, session, plan_id)
                cases.append(case)
                write_json(output / "cases.json", cases)
                if progress:
                    progress({"stage": "qualification", "completed": len(cases), "planned": 2 * len(validation)})
        result = {**qualification_summary(cases, plan), "plan_id": plan_id, "training_records": len(cache),
                  "trainable_parameters": sum(p.numel() for p in helper.parameters()), "training_trace": training["trace"]}
        export = {"format": "basic_temporal_helper_v1", "codec": pool.codec.to_dict(), "contract": adapter.contract(),
                  "spec": temporal_proxy_spec(), "model": helper.state_dict(), "model_sha256": tensor_digest(helper.state_dict()),
                  "qualified_for_inverse_training": result["qualified_for_inverse_training"], "qualification": result, "plan_id": plan_id}
        atomic_torch_save(output / "helper.pt", export)
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        monitor.close()
        elapsed = time() - session.started
        resources_ok = monitor.errors == 0 and monitor.peak_bytes is not None and monitor.peak_bytes <= plan["limits"]["process_tree_bytes"] and elapsed <= plan["limits"]["wall_seconds"]
        if result is not None and not resources_ok:
            result["qualified_for_inverse_training"] = False
            export["qualified_for_inverse_training"] = False
            atomic_torch_save(output / "helper.pt", export)
        summary = {"plan_id": plan_id, "complete": result is not None and failure is None, "failure": failure, "result": result,
                   "elapsed_seconds": elapsed, "peak_process_tree_rss_bytes": monitor.peak_bytes,
                   "rss_sampling_errors": monitor.errors, "within_resource_limits": resources_ok,
                   "render_attempts": {phase: sum(e["event"] == "attempt" and e["phase"] == phase for e in session.events)
                                       for phase in ("training", "qualification")}, "renderer_identity": session.renderer_identity,
                   "test_split_used": False, "training_audio_archived": False}
        write_json(output / "summary.json", summary)
    return summary


def load_temporal_proxy(path, *, require_qualified=True):
    saved = torch.load(Path(path), map_location="cpu", weights_only=True)
    adapter = BasicControlAdapter(load_codec(saved["codec"]))
    if (saved.get("format") != "basic_temporal_helper_v1" or saved["spec"] != temporal_proxy_spec()
            or saved["contract"] != adapter.contract() or saved["model_sha256"] != tensor_digest(saved["model"])):
        raise ValueError("incompatible or changed temporal helper")
    if require_qualified and not saved["qualified_for_inverse_training"]:
        raise ValueError("temporal helper has not passed native qualification")
    helper = TemporalSpectralProxy()
    helper.load_state_dict(saved["model"])
    return helper.freeze(), adapter

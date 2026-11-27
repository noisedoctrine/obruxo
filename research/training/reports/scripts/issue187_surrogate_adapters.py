"""Train gated frozen surrogates and replay the Issue 173 inversion panel."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import json
import math
import random
from time import perf_counter

import numpy as np
from scipy.io import wavfile
import torch
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

import run_issue187_surrogates as experiment
import run_surrogate_model_selection as inversion
from obruxo_training.issue187_stage2 import (
    Issue187Adapter, MidiEmbeddingCache, load_midi_encoder, performance_to_spans, trainable_parameter_count,
)
from obruxo_training.reconstruction_metrics import waveform_scores
from obruxo_training.spectral import FFT_SIZES, ENVELOPE_HOP, ENVELOPE_WINDOW, SAMPLE_RATE
from obruxo_training.surrogate_pilot import native_session


BASE = experiment.baseline
DEVICE, SEED = experiment.DEVICE, experiment.SEED
CAP_SECONDS, MAX_UPDATES, SAMPLER_STEPS = 5 * 3600, 8_000, 8
MILESTONES = (0, 500, 1000, 2000, 4000, 8000)
DIRECT = BASE.MAIN_ROOT / "research/training/outputs/issue185-direct-flow-learning-curves-20261127"


class MissingTrainingAudio(RuntimeError):
    pass


def cache_latents(output, codec, training, validation, waves, progress, *, recovered, prior=()):
    destination = output / f"latents-{codec.name}"
    destination.mkdir(exist_ok=True)
    groups = (("train", training, recovered, BASE.TRAINING_SOURCE),
              ("validation", validation, waves, experiment.PROXY_SOURCE))
    receipts = {source: experiment.receipts(source) for _, _, _, source in groups}
    jobs = [(f"{label}-{row['source_stage']}" if label == "train" else label, row, audio, source)
            for label, rows, audio, source in groups for row in rows]

    def existing_latent(filename):
        return next((root / filename for root in prior if (root / filename).is_file()), None)

    absent = []
    for label, row, audio, _ in jobs:
        pair = row["pair_id"]
        path = destination / f"{label}-{pair}.pt"
        if path.is_file() or existing_latent(path.name) is not None:
            continue
        absent.extend(f"{pair}:{role}" for role, _ in experiment.ROLES
                      if not (audio / f"{pair}-{role}.npy").is_file())
    if absent:
        BASE.write_json(output / "missing-training-audio.json", absent)
        raise MissingTrainingAudio(f"{len(absent)} required source clips are absent from the verified audio cache")
    progress.set("Stage 2 · Cache decoder input latents", len(jobs), 2)
    started = perf_counter()
    reused = 0
    for count, (label, row, wave_dir, source) in enumerate(jobs, 1):
        pair = row["pair_id"]
        path = destination / f"{label}-{pair}.pt"
        if not path.is_file():
            existing = existing_latent(path.name)
            if existing is not None:
                latents = torch.load(existing, map_location="cpu", weights_only=True)
                for latent, (role, _) in zip(latents, experiment.ROLES, strict=True):
                    frames = receipts[source][f"{pair}:{role}"]["qa"]["actual_frames"]
                    if latent.shape != (codec.model.latent_dim, math.ceil(frames / codec.hop)):
                        raise ValueError(f"{pair}:{role}: legacy cache shape differs from codec and render receipt")
                    if latent.dtype != torch.float32 or not torch.isfinite(latent).all():
                        raise ValueError(f"{pair}:{role}: invalid legacy cached latent")
                reused += 1
            else:
                audio = experiment.pair_audio([row], wave_dir, None, receipts[source])
                with torch.no_grad():
                    latents = [codec.encode(experiment.samples(wave), int(row["midi_id"][:8], 16) + j)[0].cpu()
                               for j, wave in enumerate(audio)]
            BASE.atomic_torch_save(path, latents)
        progress.update(count)
    BASE.write_json(output / "latent-cache.json", {"codec_revision": codec.checkpoint.name, "pairs": len(jobs),
                                                  "prior_pairs_reused_this_run": reused})
    return destination, perf_counter() - started


def generator_codec_matches(name):
    standalone = experiment.CHECKPOINTS[name] / "model.safetensors"
    generator = experiment.GENERATORS[name] / "model.safetensors"
    with safe_open(standalone, framework="pt", device="cpu") as left, safe_open(
        generator, framework="pt", device="cpu",
    ) as right:
        prefix = "pretransform.model."
        keys = set(left.keys())
        return (keys == {key.removeprefix(prefix) for key in right.keys() if key.startswith(prefix)}
                and all(torch.equal(left.get_tensor(key), right.get_tensor(prefix + key)) for key in sorted(keys)))


def generator_model(name, codec):
    location = experiment.CHECKPOINTS[name] if name == "open-small" else experiment.GENERATORS[name]
    config = experiment.read_json(location / "model_config.json")
    conditioning = config["model"]["conditioning"]
    conditioning["configs"] = [row for row in conditioning["configs"] if row["id"] != "prompt"]
    if config["model"]["diffusion"]["diffusion_objective"] not in ("rectified_flow", "rf_denoiser"):
        raise ValueError("the generator must use a rectified flow velocity objective")
    if name == "open-small":
        from stable_audio_tools.models.factory import create_model_from_config

        model = create_model_from_config(config)
    else:
        from stable_audio_3.factory import create_diffusion_cond_from_config

        model = create_diffusion_cond_from_config(config)
    from stable_audio_3.loading_utils import remap_state_dict_keys

    target = model.state_dict()
    with safe_open(location / "model.safetensors", framework="pt", device="cpu") as saved:
        tensors = {key: saved.get_tensor(key) for key in saved.keys()
                   if not key.startswith(("pretransform.", "conditioner.conditioners.prompt."))}
        tensors = remap_state_dict_keys(tensors, target)
    missing, unexpected = model.load_state_dict(tensors, strict=False)
    if unexpected or any(not key.startswith("pretransform.") for key in missing):
        raise ValueError(f"{name}: generator checkpoint differs from the connector")
    del target, tensors
    model.model.to(device=DEVICE, dtype=torch.float16)
    model.conditioner.to(device=DEVICE, dtype=torch.float16)
    model.pretransform.model = codec.model
    return model.eval().requires_grad_(False), conditioning["cond_dim"], config


def truncated_logistic(uniform):
    left = .075
    lower_cdf = torch.special.ndtr(torch.logit(uniform.new_tensor(left)))
    sampled = torch.special.ndtri(lower_cdf + (1 - lower_cdf) * uniform).sigmoid()
    return (sampled - left) / (1 - left)


class Adapter(nn.Module):
    def __init__(self, mode, backbone, codec, control_width, latent_dim, midi_encoder,
                 midi_cache, generator=None, context_width=None, generator_config=None):
        super().__init__()
        self.mode, self.codec, self.generator = mode, codec, generator
        self.name = f"{mode}-{backbone}"
        self.latent_dim = latent_dim
        self.network = Issue187Adapter(backbone, control_width, latent_dim if generator is None else context_width,
                                       generator=generator is not None)
        self.midi_encoder, self.midi_cache = midi_encoder, midi_cache
        self.generator_config = generator_config
        self.performance = None
        self.sizes = {}

    def midi(self, performance, length):
        # Vital patches do not carry GM program labels. Use GM Synth Lead for melodic notes.
        notes = performance_to_spans(performance, default_program=80)
        tokens = self.midi_cache.get_or_encode(notes, length / SAMPLE_RATE, self.midi_encoder, device=DEVICE)
        return tokens.clone()[None]

    def representation(self, controls, played, _lengths, frames):
        mask = torch.ones(played.shape[:2], dtype=torch.bool, device=played.device)
        hz = SAMPLE_RATE / self.codec.hop if self.generator is None else 25
        frames = frames if self.generator is None else played.shape[1]
        return self.network(controls, played, mask, target_frames=frames, target_hz=hz)

    def velocity(self, latent, time, tokens, duration, *, prompt_mask=None, recompute=False):
        generator = self.generator
        dtype = next(generator.model.parameters()).dtype
        patch = generator.min_input_length // self.codec.hop
        frames = latent.shape[-1]
        padded = F.pad(latent, (0, -frames % patch)).to(dtype)
        mask = torch.arange(padded.shape[-1], device=latent.device)[None] < frames
        number = generator.conditioner.conditioners["seconds_total"]([duration], latent.device)
        # The generator is frozen. Keep the prompt embeddings attached to the trainable adapter.
        prompt_mask = mask[:, :tokens.shape[1]] if prompt_mask is None else prompt_mask
        cond = {"prompt": [tokens.to(dtype), prompt_mask], "seconds_total": number,
                "inpaint_mask": [padded.new_zeros((1, 1, padded.shape[-1]))],
                "inpaint_masked_input": [torch.zeros_like(padded)]}

        def model_forward(x, prompt):
            return generator(x, time.to(dtype), {**cond, "prompt": [prompt, cond["prompt"][1]]},
                             padding_mask=mask)[..., :frames]

        if recompute and torch.is_grad_enabled():
            return checkpoint(model_forward, padded, cond["prompt"][0], use_reentrant=False).float()
        return model_forward(padded, cond["prompt"][0]).float()

    def train_loss(self, controls, midi, lengths, expected, rng):
        frames = expected.shape[-1]
        tokens = self.representation(controls, midi, lengths, frames)
        if self.generator is None:
            return F.mse_loss(tokens, expected)
        duration = int(lengths[0]) / SAMPLE_RATE
        tokens, mask = tokens
        time = self.training_time(frames, rng)
        noise = torch.randn(expected.shape, generator=rng).to(DEVICE)
        scaled = expected / self.generator.pretransform.scale
        noisy = torch.lerp(scaled, noise, time)
        predicted = self.velocity(noisy, time, tokens, duration, prompt_mask=mask)
        return F.mse_loss(predicted, noise - scaled)

    def training_time(self, frames, rng):
        sampler = self.generator_config.get("training", {}).get("timestep_sampler", "uniform")
        uniform = torch.rand(1, generator=rng)
        if sampler == "trunc_logit_normal":
            # Inverse CDF of the upstream truncated logistic normal, reversed as in training.
            uniform = 1 - truncated_logistic(uniform)
        elif sampler != "uniform":
            raise ValueError(f"unsupported native training timestep distribution: {sampler}")
        time = uniform.to(DEVICE)
        shift = self.generator.dist_shift
        return shift.shift(time, frames) if shift is not None else time

    def waveform(self, controls, midi, lengths, frames, seed):
        representation = self.representation(controls, midi, lengths, frames)
        if self.generator is None:
            latents = representation
        else:
            from stable_audio_3.inference.sampling import build_schedule

            tokens, mask = representation
            duration = int(lengths[0]) / SAMPLE_RATE
            shape = (1, self.latent_dim, frames)
            noise = BASE.cpu_generator(seed)
            latents = torch.randn(shape, generator=noise).to(DEVICE)
            schedule = build_schedule(SAMPLER_STEPS, dist_shift=self.generator.sampling_dist_shift,
                                      effective_seq_len=frames, device=DEVICE)
            for time, following in zip(schedule[:-1], schedule[1:], strict=True):
                velocity = self.velocity(latents, time[None], tokens, duration, prompt_mask=mask, recompute=True)
                if self.generator.diffusion_objective == "rf_denoiser":
                    # Upstream pingpong update. Every re-noise draw is keyed to the validation case.
                    latents = (1 - following) * (latents - time * velocity)
                    latents = latents + following * torch.randn(shape, generator=noise).to(DEVICE)
                else:
                    latents = latents + (following - time) * velocity
            latents = latents * self.generator.pretransform.scale
        return self.codec.decode(latents).float()[..., :int(lengths[0])]

    def forward(self, controls, _played, times, lengths):
        # Interface used by the Issue 173 category prefilter and projected Adam.
        waves = []
        for i, length in enumerate(lengths.tolist()):
            if length not in self.sizes:
                with torch.no_grad():
                    self.sizes[length] = math.ceil(length / self.codec.hop)
            midi = self.midi(self.performance, length)
            waves.append(self.waveform(controls[i:i + 1], midi, lengths[i:i + 1],
                                       self.sizes[length], SEED))
        audio = torch.cat(waves)
        magnitudes = tuple(experiment.frame_spectra(audio, time, size=size)
                           for time, size in zip(times[:3], FFT_SIZES, strict=True))
        channels = torch.cat((audio, (audio[:, :1] - audio[:, 1:2]) / 2), dim=1)
        energy = F.avg_pool1d(F.pad(channels.square(), (ENVELOPE_WINDOW // 2,) * 2),
                              ENVELOPE_WINDOW, stride=ENVELOPE_HOP)
        indices = (times[3] * (SAMPLE_RATE / ENVELOPE_HOP)).round().long()
        envelope = ((energy + 1e-14).sqrt() - 1e-7).clamp_min(0.)
        return {"magnitudes": magnitudes, "envelope": envelope.gather(2, indices[:, None].expand(-1, 3, -1))}


@torch.no_grad()
def validate(adapter, rows, controls, latents, waves, deadline, output, step, progress):
    data = experiment.PROXY_SOURCE / "validation-pairs"
    loader, cases = BASE._pair_loader(data), []
    reconstructed = output / "validation-waveforms" / f"{adapter.name}-{step}"
    reconstructed.mkdir(parents=True, exist_ok=True)
    manifest = []
    progress.set(f"Stage 2 · {adapter.name} validation at {step:,}", 2 * len(rows), 2)

    def score_and_save(record, original, wave, path):
        record.update(waveform_scores(original, wave), cir_cosine=None, visqol_audio=None)
        wavfile.write(path, SAMPLE_RATE, wave)
        return record

    def predictions():
        for index, row in enumerate(rows):
            if perf_counter() >= deadline:
                raise TimeoutError("the arm reached its five-hour cap during validation")
            batch, metas = BASE.collate_pairs([row], data, controls, DEVICE, loader)
            expected = torch.load(latents / f"validation-{row['pair_id']}.pt", weights_only=True)
            performance = experiment.Performance.from_dict(row["performance"])
            for j, ((role, _), meta) in enumerate(zip(experiment.ROLES, metas, strict=True)):
                clip = experiment.split_batch(batch, j)
                case_id = f"{row['pair_id']}-{role}"
                original_path = (waves / f"{case_id}.npy").resolve()
                original = np.load(original_path, allow_pickle=False)
                midi = adapter.midi(performance, len(original))
                started = perf_counter()
                predicted = adapter.waveform(clip["controls"], midi, clip["lengths"],
                                             expected[j].shape[-1], SEED + 2 * index + j)
                torch.xpu.synchronize()
                seconds = perf_counter() - started
                record = experiment.measured_case(predicted, original, clip, meta)
                wave = predicted[0].T.cpu().numpy()
                waveform_path = (reconstructed / f"{case_id}.wav").resolve()
                record["forward_seconds"] = seconds
                if step in (500, 8000) and index in (0, 32, 64, 96):
                    listening = output / "listening"
                    listening.mkdir(exist_ok=True)
                    name = f"{row['pair_id']}-{role}-{adapter.name}-{step}.wav"
                    wavfile.write(listening / name, SAMPLE_RATE, wave)
                yield record, original, wave, waveform_path, {
                    "case": case_id, "original_path": str(original_path), "wave_path": str(waveform_path),
                }

    with ThreadPoolExecutor(max_workers=1) as scorer:
        pending = deque()
        for record, original, wave, path, entry in predictions():
            manifest.append(entry)
            pending.append(scorer.submit(score_and_save, record, original, wave, path))
            if len(pending) >= 3:
                cases.append(pending.popleft().result())
                progress.update(len(cases))
        while pending:
            cases.append(pending.popleft().result())
            progress.update(len(cases))
    summary = BASE.aggregate_cases(cases)
    summary.update({key: float(np.mean([row[key] for row in cases])) for key in
                    ("mrstft", "side_to_mid_error_db", "rms_relative", "forward_seconds")})
    for key in ("mel_log1p", "stft_log1p", "ccpc", "si_sdr_db", "cir_cosine", "visqol_audio"):
        valid = [row[key] for row in cases if row[key] is not None]
        summary[key] = float(np.mean(valid)) if valid else None
        summary[f"{key}_valid"] = len(valid)
    BASE.write_json(output / f"validation-{adapter.name}-{step}.json", {"metrics": summary, "cases": cases})
    BASE.write_json(output / f"waveforms-{adapter.name}-{step}.json", manifest)
    return summary


def arm(output, adapter, training, validation, controls, latents, waves, direct, progress, started, *,
        smoke=False, target_step=500):
    deadline = started + CAP_SECONDS
    rng = BASE.cpu_generator(SEED)
    shuffle = random.Random(SEED)
    optimizer = torch.optim.AdamW(adapter.network.parameters(), lr=.001, weight_decay=.01)
    pair_cache = BASE._pair_loader(BASE.TRAINING_SOURCE / "pairs")

    @lru_cache(maxsize=128)
    def target(row_id, source):
        return torch.load(latents / f"train-{source}-{row_id}.pt", weights_only=True)

    report = {"mode": adapter.mode, "backbone": adapter.network.backbone_name, "updates": 0,
              "trainable_parameters": trainable_parameter_count(adapter.network),
              "checkpoints": {}, "stopped": None, "logs": []}
    arm_path = output / f"arm-{adapter.name}.json"
    resume_path = output / f"adapter-{adapter.name}-latest.pt"
    cap = 1 if smoke else MAX_UPDATES
    target_step = 1 if smoke else target_step
    milestones = (1,) if smoke else MILESTONES
    durations, block_loss = [], []
    cursor, order = 0, []
    torch.xpu.reset_peak_memory_stats()
    first_update = 1
    train_seconds = 0.
    prior_seconds = 0.
    pretrained = [adapter.codec.model, adapter.midi_encoder, adapter.generator]
    frozen_parameters = [p for model in pretrained if model is not None for p in model.parameters()] if smoke else []
    frozen_versions = [p._version for p in frozen_parameters]
    if resume_path.is_file():
        saved = torch.load(resume_path, map_location="cpu", weights_only=True)
        if saved["cap"] != cap or saved["codec_revision"] != adapter.codec.checkpoint.name:
            raise ValueError("adapter checkpoint differs from this experiment configuration")
        adapter.network.load_state_dict(saved["adapter"])
        optimizer.load_state_dict(saved["optimizer"])
        rng.set_state(saved["rng"])
        shuffle.setstate(saved["shuffle"])
        cursor, order, first_update = saved["cursor"], saved["order"], saved["update"] + 1
        report, durations, train_seconds = saved["report"], saved["durations"], saved["train_seconds"]
        prior_seconds = saved["elapsed_seconds"]
        deadline -= prior_seconds
        if report.get("stopped"):
            return report
        report["stopped"] = None

    def save(step):
        if step in milestones:
            validation_start = perf_counter()
            readings = validate(adapter, validation, controls, latents, waves, deadline, output, step, progress)
            durations.append(perf_counter() - validation_start)
            report["checkpoints"][str(step)] = {**readings, "peak_xpu_bytes": torch.xpu.max_memory_allocated(),
                                               "matched_direct_spectral_loss": direct.get(str(step), {}).get("spectral_loss")}
        checkpoint_data = {
            "adapter": adapter.network.state_dict(),
            "optimizer": optimizer.state_dict(), "update": step, "rng": rng.get_state(),
            "shuffle": shuffle.getstate(), "order": order, "cursor": cursor,
            "codec_revision": adapter.codec.checkpoint.name, "cap": cap,
            "report": report, "durations": durations, "train_seconds": train_seconds,
            "elapsed_seconds": prior_seconds + perf_counter() - started,
        }
        if step in milestones:
            BASE.atomic_torch_save(output / f"adapter-{adapter.name}-{step}.pt", checkpoint_data)
        BASE.atomic_torch_save(resume_path, checkpoint_data)
        BASE.write_json(arm_path, report)

    try:
        if first_update == 1 and not smoke:
            save(0)
        progress.set(f"Stage 2 · {adapter.name} updates", target_step, 2)
        tick = perf_counter()
        for step in range(first_update, target_step + 1):
            if perf_counter() >= deadline:
                raise TimeoutError("the arm reached its five-hour cap")
            if cursor + 4 > len(order):
                order = list(range(len(training)))
                shuffle.shuffle(order)
                cursor = 0
            selected = [training[i] for i in order[cursor:cursor + 4]]
            cursor += 4
            records = [record for row in selected for record in pair_cache(row["pair_id"])]
            patches = [BASE.PilotTarget.from_dict(record["target"]) for record in records]
            patches = [controls.from_basic(patch) if len(patch.continuous) == 9 else patch for patch in patches]
            patches = BASE.control_tensor(controls, patches, DEVICE)
            if smoke:
                patches.requires_grad_(True)
            performances = [experiment.Performance.from_dict(row["performance"]) for row in selected]
            lengths = torch.cat([record["measurements"]["lengths"] for record in records]).to(DEVICE)
            expected = [z for row in selected for z in target(row["pair_id"], row["source_stage"])]
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for i, latent in enumerate(expected):
                if perf_counter() >= deadline:
                    raise TimeoutError("the arm reached its five-hour cap during an update")
                clip = slice(i, i + 1)
                midi = adapter.midi(performances[i // 2], int(lengths[i]))
                loss = adapter.train_loss(patches[clip], midi, lengths[clip], latent[None].to(DEVICE), rng)
                (loss / len(expected)).backward()
                losses.append(loss.detach())
            gradient = torch.nn.utils.clip_grad_norm_(adapter.network.parameters(), 5.)
            if not torch.isfinite(gradient) or (step == 1 and gradient <= 0):
                raise RuntimeError("adapter gradients are missing or nonfinite")
            if smoke:
                if patches.grad is None or not torch.isfinite(patches.grad).all() or not patches.grad[:, :13].any():
                    raise RuntimeError("the optimizer update produced no finite continuous patch gradients")
                if any(p.requires_grad or p.grad is not None or p._version != version
                       for p, version in zip(frozen_parameters, frozen_versions, strict=True)):
                    raise RuntimeError("pretrained parameters must remain frozen")
                report["smoke"] = {"loss": float(torch.stack(losses).mean()), "gradient_norm": float(gradient),
                                   "continuous_patch_gradient_norm": float(patches.grad[:, :13].norm()),
                                   "optimizer_update_seconds": perf_counter() - tick}
            optimizer.step()
            block_loss.append(torch.stack(losses).mean().item())
            report["updates"] = step
            progress.update(step)
            if smoke:
                save(step)
                restored = torch.load(resume_path, map_location="cpu", weights_only=True)
                previous = adapter.network
                replacement = Issue187Adapter(previous.backbone_name, previous.patch.in_features,
                                              previous.connector.output.out_features,
                                              generator=adapter.generator is not None).to(DEVICE)
                replacement.load_state_dict(restored["adapter"])
                if any(not torch.equal(left, right) for left, right in
                       zip(previous.parameters(), replacement.parameters(), strict=True)):
                    raise RuntimeError("checkpoint reload changed trained adapter parameters")
                optimizer.load_state_dict(restored["optimizer"])
                report["smoke"]["checkpoint_reloaded"] = True
            if step % 100 == 0:
                block_seconds = perf_counter() - tick
                train_seconds += block_seconds
                remaining_checks = sum(m > step for m in MILESTONES)
                projected = prior_seconds + perf_counter() - started + train_seconds / step * (cap - step)
                projected += np.mean(durations) * remaining_checks
                entry = {"update": step, "loss": float(np.mean(block_loss)),
                         "updates_per_second": 100 / block_seconds, "projected_total_seconds": float(projected),
                         "peak_xpu_bytes": torch.xpu.max_memory_allocated()}
                report["logs"].append(entry)
                block_loss.clear()
                progress.update(step, log=True, message=f"{entry['updates_per_second']:.2f} updates/s, "
                                                        f"projected {projected / 3600:.2f} h")
                if step == 100 and projected > CAP_SECONDS:
                    report["stopped"] = "update-100 projection exceeds five hours"
                    save(step)
                    break
                save(step)
                if step in MILESTONES and step < target_step:
                    progress.set(f"Stage 2 · {adapter.name} updates", target_step, 2)
                    progress.update(step)
                tick = perf_counter()
    except (RuntimeError, TimeoutError, ValueError) as exc:
        report["stopped"] = str(exc)
        progress.failures += 1
        progress.update(progress.done, log=True, message=str(exc))
    report["elapsed_seconds"] = prior_seconds + perf_counter() - started
    BASE.write_json(arm_path, report)
    return report


def perturbation_probe(adapter, controls, case, render, capture):
    case_id = case["case_id"]
    source, start_audio = (capture[f"{case_id}:{suffix}"] for suffix in ("source", "start"))
    start = controls.canonical(BASE.PilotTarget(controls.defaults, (0,) * len(controls.categorical)))
    active, _ = controls.masks(start.categorical)
    index = next(i for i, enabled in enumerate(active) if enabled)
    values = list(start.continuous)
    values[index] += .05 if values[index] <= .95 else -.05
    changed = controls.canonical(BASE.PilotTarget(tuple(values), start.categorical))
    if changed == start:
        return {"reason": "codec canonicalization removed the perturbation"}
    performance = experiment.Performance.from_dict(case["performance"])
    (changed_audio, receipt), = render([(f"{case_id}:controlled-perturbation", changed, performance)])
    if changed_audio is None:
        return {"reason": receipt.get("status")}
    inputs = BASE.control_tensor(controls, [start, changed], DEVICE)
    midi = adapter.midi(performance, len(source))
    lengths = torch.tensor([len(source)], device=DEVICE)
    frames = adapter.sizes[len(source)]
    with torch.no_grad():
        predictions = [adapter.waveform(value[None], midi, lengths, frames, SEED)[0].T.cpu().numpy()
                       for value in inputs]
    baseline_native = experiment.audio_distance(source, start_audio)["mrstft"]
    native_delta = experiment.audio_distance(source, changed_audio)["mrstft"] - baseline_native
    predicted_delta = (experiment.audio_distance(source, predictions[1])["mrstft"]
                       - experiment.audio_distance(source, predictions[0])["mrstft"])
    patch = inputs[:1].detach().requires_grad_()
    prediction = adapter.waveform(patch, midi, lengths, frames, SEED)
    gradient, = torch.autograd.grad((prediction - experiment.samples(source)).square().mean(), patch)
    norm = float(gradient.norm())
    return {"control": controls.continuous[index][0], "native_mrstft_delta": native_delta,
            "predicted_mrstft_delta": predicted_delta,
            "direction_agrees": bool(np.sign(native_delta) == np.sign(predicted_delta)),
            "patch_gradient_norm": norm if math.isfinite(norm) else None}


def inverse_panel(output, adapter, controls, plan, progress):
    cases = plan["evaluation"]["cases"]
    if len(cases) != 192:
        raise ValueError("Stage 3 requires exactly the 192 Issue 173 cases")
    progress.set(f"Stage 3 · Native Vital inversion with {adapter.mode}", len(cases), 3)
    path = output / "stage3-cases.jsonl"
    previous = {row["case_id"] for row in experiment.read_rows(path)} if path.exists() else set()
    original_receipts = experiment.receipts(experiment.PROXY_SOURCE)
    torch.xpu.reset_peak_memory_stats()
    with native_session(output, controls, 4) as render, path.open("a", encoding="utf-8") as stream:
        for index, case in enumerate(cases, 1):
            case_id = case["case_id"]
            if case_id not in previous:
                original = experiment.read_json(experiment.PROXY_SOURCE / "cases" / f"{case_id}.json")
                if original["case_hash"] != BASE.canonical_sha256(case):
                    raise ValueError("Issue 173 inversion case identity changed")
                capture = {}

                def record(jobs):
                    outcomes = render(jobs)
                    capture.update({label: audio for (label, _, _), (audio, _) in zip(jobs, outcomes, strict=True)})
                    return outcomes

                started = perf_counter()
                adapter.performance = experiment.Performance.from_dict(case["performance"])
                result = inversion._evaluate_case(case, {adapter.name: adapter}, controls, record,
                                                   device=DEVICE, steps=plan["evaluation"]["optimizer"]["steps"])
                result["direct_native_mrstft"] = original.get("models", {}).get("residual_mlp", {}).get("native_mrstft")
                result["wall_seconds"] = perf_counter() - started
                measured = result.get("models", {}).get(adapter.name, {}).get("native_mrstft")
                if measured is None:
                    progress.failures += 1
                else:
                    source, candidate = (capture[f"{case_id}:{suffix}"] for suffix in ("source", adapter.name))
                    for role in ("source", "start"):
                        label = f"{case_id}:{role}"
                        if label in original_receipts:
                            experiment.require_audio(capture[label], original_receipts[label], label)
                    gain = float((source * candidate).sum() / max(1e-12, float((candidate ** 2).sum())))
                    result["gain_matched_mrstft"] = experiment.audio_distance(source, candidate * gain)["mrstft"]
                    result["native_waveform"] = waveform_scores(source, candidate)
                    result["start_waveform"] = waveform_scores(source, capture[f"{case_id}:start"])
                    result["improved"] = result["native_waveform"]["mel_log1p"] < result["start_waveform"]["mel_log1p"]
                    exported = BASE.PilotTarget.from_dict(result["models"][adapter.name]["patch"])
                    result["legal_patch"] = controls.canonical(exported) == exported
                    if (index - 1) % 16 == 0:
                        result["perturbation"] = perturbation_probe(adapter, controls, case, render, capture)
                    if index in (1, 33, 65, 97, 129, 161):
                        listening = output / "listening"
                        listening.mkdir(exist_ok=True)
                        for role, wave in (("source", source), ("start", capture[f"{case_id}:start"]),
                                           ("optimized", candidate)):
                            wavfile.write(listening / f"inversion-{index:03d}-{role}.wav", SAMPLE_RATE, wave)
                stream.write(json.dumps(result, allow_nan=False) + "\n")
                stream.flush()
            progress.update(index)
    results = list(experiment.read_rows(path))
    observed = [row for row in results if row.get("models", {}).get(adapter.name, {}).get("native_mrstft") is not None]
    matched = [row for row in observed if row["direct_native_mrstft"] is not None]
    differences = [row["models"][adapter.name]["native_mrstft"] - row["direct_native_mrstft"] for row in matched]
    stats = {"planned": len(cases), "valid": len(observed), "matched": len(matched), "model": adapter.name,
             "peak_xpu_bytes": torch.xpu.max_memory_allocated(), "improved": sum(row["improved"] for row in observed)}
    stats["legal_patches"] = sum(row.get("legal_patch", False) for row in observed)
    stats["perturbations"] = [row["perturbation"] for row in observed if "perturbation" in row]
    if matched:
        stats["matched_mrstft_difference"] = float(np.mean(differences))
        stats["patch_root_and_midi_interval"] = inversion._bootstrap_interval(matched, differences)
    BASE.write_json(output / "stage3.json", stats)


def run_later_stages(output, selection, plan, training, validation, controls, waves, progress, *,
                     recovered, prior_run=None, midi_checkpoint, smoke=False,
                     qualification=None, target_step=500, inversion_decision=None):
    path = output / "decisions.json"
    decision = experiment.read_json(path) if path.is_file() else {}
    winner = selection["codec"]
    decision["stage1_rescore_selection"] = selection
    if winner is None:
        decision.update(stage2="skipped", stage3="skipped", reason=selection["reason"])
        BASE.write_json(path, decision)
        return "complete"
    checkpoint = (experiment.GENERATORS["same-l"] if winner == "same-l-embedded"
                  else experiment.CHECKPOINTS[winner])
    if checkpoint.name != selection["codec_revision"]:
        raise ValueError("the Stage 1 decision does not match the requested codec checkpoint")
    generator_name = selection.get("generator")
    expected_generator = {"same-l-embedded": "same-l", "open-small": "open-small", "same-s": "same-s"}.get(winner)
    if generator_name and generator_name != expected_generator:
        raise ValueError("the selected generator/codec pair does not share a latent space")
    if generator_name == "same-s" and not generator_codec_matches("same-s"):
        raise ValueError("the Small Music embedded codec differs from the rescored standalone codec")
    if not smoke:
        if not selection["stage1_complete"]:
            raise ValueError("formal training requires the published complete Stage 1 comparison")
        qualified = experiment.read_json(qualification)
        checkpoint_fields = ("codec", "codec_revision", "generator")
        if (qualified["stage2"] != "qualified" or any(
            qualified["stage1_rescore_selection"].get(key) != selection.get(key) for key in checkpoint_fields
        )):
            raise ValueError("qualification does not match the selected study")
        qualified = set(qualified["qualified_configurations"])
    direct = experiment.read_json(DIRECT / "summary.json")["checkpoints"]["residual_mlp-direct"]
    direct = {step: row["metrics"] for step, row in direct.items()}
    codec = experiment.Codec(winner, checkpoint)
    prior = []
    expected = experiment.read_json(output / "input-receipts.json")
    if prior_run is not None:
        receipt = experiment.read_json(prior_run / "decisions.json")
        if receipt.get("stage2_codec_revision") == checkpoint.name:
            prior_input_receipts = experiment.read_json(prior_run / "input-receipts.json")
            if prior_input_receipts != expected:
                raise ValueError("legacy latent cache input receipts differ from the verified current inputs")
            prior.append(prior_run / f"latents-{codec.name}")
    if not smoke:
        qual_root = qualification.parent
        if experiment.read_json(qual_root / "input-receipts.json") != expected:
            raise ValueError("qualification input receipts differ from the verified current inputs")
        prior.append(qual_root / f"latents-{codec.name}")
    if smoke:
        training = [row for row in training if (
            any((root / f"train-{row['source_stage']}-{row['pair_id']}.pt").is_file() for root in prior)
            or all((recovered / f"{row['pair_id']}-{role}.npy").is_file() for role, _ in experiment.ROLES)
        )][:4]
        if len(training) != 4:
            raise MissingTrainingAudio("qualification requires four verified training pairs")
    decision.update(stage2="running", stage3="waiting", stage2_codec_revision=checkpoint.name,
                    generator_sampler_steps=SAMPLER_STEPS if generator_name else None,
                    seed=SEED, effective_batch_pairs=4)
    BASE.write_json(path, decision)
    try:
        latents, cache_seconds = cache_latents(output, codec, training, validation, waves, progress,
                                              recovered=recovered, prior=prior)
    except MissingTrainingAudio as exc:
        decision.update(stage2="blocked_missing_verified_audio", stage3="not_started", reason=str(exc))
        BASE.write_json(path, decision)
        progress.update(progress.done, state="blocked", message=str(exc))
        return "blocked"
    decision["shared_latent_cache_seconds"] = cache_seconds
    provenance = experiment.read_json(BASE.TRAINING / "reports/issue187_pretrained_vital_surrogates/"
                                      "STAGE2_CHECKPOINT_PROVENANCE.json")
    if BASE.sha256_file(midi_checkpoint) != provenance["extracted_encoder_sha256"]:
        raise ValueError("SpanSynth-Edit checkpoint differs from the pinned extracted encoder")
    midi = load_midi_encoder(midi_checkpoint, device=DEVICE).eval()
    midi_cache = MidiEmbeddingCache(output / "midi-embeddings")
    width = BASE.control_tensor(controls, [BASE.PilotTarget.from_dict(validation[0]["target"])], DEVICE).shape[-1]
    available = ("latent", "generator") if generator_name else ("latent",)
    if inversion_decision is not None:
        chosen = experiment.read_json(inversion_decision)
        mode, backbone = chosen["adapter"].split("-", 1)
        report = experiment.read_json(output / f"arm-{chosen['adapter']}.json")
        if chosen["adapter"] not in qualified or report["updates"] != MAX_UPDATES or mode not in available:
            raise ValueError("Stage 3 requires a selected, qualified, completed Stage 2 arm")
        generator, context_width, config = generator_model(generator_name, codec) if mode == "generator" else (None,) * 3
        adapter = Adapter(mode, backbone, codec, width, codec.model.latent_dim, midi, midi_cache,
                          generator, context_width, config).to(DEVICE)
        saved = torch.load(output / f"adapter-{adapter.name}-{MAX_UPDATES}.pt", map_location="cpu", weights_only=True)
        if saved["codec_revision"] != checkpoint.name:
            raise ValueError("Stage 3 adapter codec differs from the selected checkpoint")
        adapter.network.load_state_dict(saved["adapter"])
        adapter.eval().requires_grad_(False)
        decision.update(stage2="complete", stage2_selected=chosen, stage3="running")
        BASE.write_json(path, decision)
        inverse_panel(output, adapter, controls, plan, progress)
        decision["stage3"] = "complete"
        BASE.write_json(path, decision)
        return "complete"
    reports_path = output / "stage2.json"
    reports = experiment.read_json(reports_path) if reports_path.is_file() else {}
    if not smoke and target_step == MAX_UPDATES:
        qualified &= {name for name, report in reports.items()
                      if report["updates"] >= 500 and not report.get("stopped")}
        if not qualified:
            decision.update(stage2="incomplete", stage3="not_started", reason="no viable 500-update pilot")
            BASE.write_json(path, decision)
            return "incomplete"
    for mode in available:
        backbones = [b for b in ("residual_mlp", "transformer") if smoke or f"{mode}-{b}" in qualified]
        if not backbones:
            continue
        try:
            generator, context_width, config = generator_model(generator_name, codec) if mode == "generator" else (None,) * 3
            if generator is not None and (generator.io_channels != codec.model.latent_dim
                                           or generator.pretransform.downsampling_ratio != codec.hop):
                raise ValueError("generator latent shape or time grid differs from the selected codec")
            for backbone in backbones:
                started = perf_counter()
                torch.manual_seed(SEED)
                adapter = Adapter(mode, backbone, codec, width, codec.model.latent_dim, midi, midi_cache,
                                  generator, context_width, config).to(DEVICE)
                progress.set(f"Stage 2 · Load {adapter.name} arm", 1, 2)
                reports[adapter.name] = arm(output, adapter, training, validation, controls, latents, waves,
                                            direct, progress, started, smoke=smoke, target_step=target_step)
                BASE.write_json(reports_path, reports)
                del adapter
            del generator
        except (RuntimeError, ValueError, OSError) as exc:
            reports[mode] = {"stopped": str(exc), "checkpoints": {}, "updates": 0}
            progress.failures += 1
            progress.update(progress.done, log=True, message=str(exc))
        torch.xpu.empty_cache()
    BASE.write_json(reports_path, reports)
    if smoke:
        qualified = [name for name, report in reports.items()
                     if report.get("smoke", {}).get("checkpoint_reloaded") and "1" in report["checkpoints"]]
        decision.update(stage2="qualified" if qualified else "blocked_qualification",
                        qualified_configurations=qualified, stage3="not_started")
        BASE.write_json(path, decision)
        return "qualified" if qualified else "blocked"

    finished = {name: report["checkpoints"]["8000"] for name, report in reports.items()
                if "8000" in report["checkpoints"]}
    pilot_ready = any(report["updates"] >= 500 and not report["stopped"] for report in reports.values())
    state = "analysis_required" if finished else "pilot_complete" if pilot_ready else "incomplete"
    decision.update(stage2=state,
                    stage3="waiting_for_stage2_decision" if finished else "not_started",
                    completed_configurations=sorted(finished))
    BASE.write_json(path, decision)
    return state

"""Train gated frozen surrogates and replay the Issue 173 inversion panel."""

from functools import lru_cache
import itertools
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
from obruxo_training.flow_matching import _frame_context
from obruxo_training.spectral import FFT_SIZES, ENVELOPE_HOP, ENVELOPE_WINDOW, SAMPLE_RATE
from obruxo_training.surrogate_backbones import VitalSurrogate, WIDTH
from obruxo_training.surrogate_pilot import native_session


BASE = experiment.baseline
DEVICE, SEED = experiment.DEVICE, experiment.SEED
CAP_SECONDS, MAX_UPDATES, EULER_STEPS = 5 * 3600, 8_000, 8
MILESTONES = (0, 500, 1000, 2000, 4000, 8000)
DIRECT = BASE.MAIN_ROOT / "research/training/outputs/issue185-direct-flow-learning-curves-20261127"


def cache_latents(output, codec, controls, training, validation, waves, progress):
    destination = output / f"latents-{codec.name}"
    destination.mkdir(exist_ok=True)
    progress.set("Stage 2 · Cache decoder input latents", len(training) + len(validation), 2)
    started = perf_counter()
    count = 0
    for source, label, rows, render_codec, wave_dir in (
        (BASE.TRAINING_SOURCE, "train-basic", [row for row in training if row["source_stage"] == "basic"],
         BASE.SurrogateCodec(BASE.PilotCodec(), expanded=False), None),
        (BASE.TRAINING_SOURCE, "train-expanded", [row for row in training if row["source_stage"] == "expanded"],
         controls, None),
        (experiment.PROXY_SOURCE, "validation", validation, controls, waves),
    ):
        ledger = experiment.receipts(source)
        with native_session(output, render_codec, 4) as render:
            for offset in range(0, len(rows), 4):
                group = rows[offset:offset + 4]
                pending = [row for row in group if not (destination / f"{label}-{row['pair_id']}.pt").exists()]
                if pending:
                    audio = experiment.pair_audio(pending, wave_dir, render, ledger)
                    for i, row in enumerate(pending):
                        with torch.no_grad():
                            latents = [codec.encode(experiment.samples(wave), int(row["midi_id"][:8], 16) + j)[0].cpu()
                                       for j, wave in enumerate(audio[2 * i:2 * i + 2])]
                        BASE.atomic_torch_save(destination / f"{label}-{row['pair_id']}.pt", latents)
                count += len(group)
                progress.update(count)
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


def generator_model(name):
    from stable_audio_3.loading_utils import load_diffusion_cond

    location = experiment.GENERATORS[name]
    config = experiment.read_json(location / "model_config.json")
    conditioning = config["model"]["conditioning"]
    conditioning["configs"] = [row for row in conditioning["configs"] if row["id"] != "prompt"]
    if config["model"]["diffusion"]["diffusion_objective"] not in ("rectified_flow", "rf_denoiser"):
        raise ValueError("the generator must use a rectified flow velocity objective")
    model = load_diffusion_cond(config, str(location / "model.safetensors"), device=DEVICE, model_half=True)
    return model, conditioning["cond_dim"]


class Adapter(nn.Module):
    def __init__(self, mode, codec, control_width, latent_dim, generator=None, context_width=None):
        super().__init__()
        self.mode, self.codec, self.generator = mode, codec, generator
        self.latent_dim = latent_dim
        self.backbone = VitalSurrogate("residual_mlp", control_width)
        del self.backbone.heads
        self.head = nn.Linear(WIDTH, latent_dim if generator is None else context_width)
        self.sizes = {}

    def representation(self, controls, played, lengths, frames):
        times = torch.arange(frames, device=controls.device).float()[None] * (self.codec.hop / SAMPLE_RATE)
        return self.head(_frame_context(self.backbone, controls, played, times.expand(len(controls), -1), lengths))

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
            return F.mse_loss(tokens.transpose(1, 2), expected)
        duration = int(lengths[0]) / SAMPLE_RATE
        time = torch.rand(1, generator=rng).to(DEVICE)
        noise = torch.randn(expected.shape, generator=rng).to(DEVICE)
        scaled = expected / self.generator.pretransform.scale
        noisy = torch.lerp(scaled, noise, time)
        predicted = self.velocity(noisy, time, tokens, duration)
        return F.mse_loss(predicted, noise - scaled)

    def waveform(self, controls, midi, lengths, frames, seed):
        representation = self.representation(controls, midi, lengths, frames)
        if self.generator is None:
            latents = representation.transpose(1, 2)
        else:
            duration = int(lengths[0]) / SAMPLE_RATE
            shape = (1, self.latent_dim, frames)
            latents = BASE._deterministic_noise(shape, seed, DEVICE)
            for step in range(EULER_STEPS):
                time = latents.new_full((1,), 1 - step / EULER_STEPS)
                latents = latents - self.velocity(latents, time, representation, duration, recompute=True) / EULER_STEPS
            latents = latents * self.generator.pretransform.scale
        return self.codec.decode(latents).float()[..., :int(lengths[0])]

    def forward(self, controls, played, times, lengths):
        # Interface used by the Issue 173 category prefilter and projected Adam.
        waves = []
        for i, length in enumerate(lengths.tolist()):
            if length not in self.sizes:
                with torch.no_grad():
                    zero = torch.zeros(1, 2, length, device=DEVICE)
                    self.sizes[length] = self.codec.encode(zero).shape[-1]
            waves.append(self.waveform(controls[i:i + 1], played[i:i + 1], lengths[i:i + 1],
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
    progress.set(f"Stage 2 · {adapter.mode} validation at {step:,}", 2 * len(rows), 2)
    for index, row in enumerate(rows):
        if perf_counter() >= deadline:
            raise TimeoutError("the arm reached its five-hour cap during validation")
        batch, metas = BASE.collate_pairs([row], data, controls, DEVICE, loader)
        expected = torch.load(latents / f"validation-{row['pair_id']}.pt", weights_only=True)
        for j, ((role, _), meta) in enumerate(zip(experiment.ROLES, metas, strict=True)):
            clip = experiment.split_batch(batch, j)
            original = np.load(waves / f"{row['pair_id']}-{role}.npy", allow_pickle=False)
            started = perf_counter()
            predicted = adapter.waveform(clip["controls"], clip["performance"], clip["lengths"],
                                         expected[j].shape[-1], SEED + 2 * index + j)
            torch.xpu.synchronize()
            seconds = perf_counter() - started
            record = experiment.measured_case(predicted, original, clip, meta)
            record["forward_seconds"] = seconds
            cases.append(record)
            if step in (500, 8000) and index in (0, 32, 64, 96):
                listening = output / "listening"
                listening.mkdir(exist_ok=True)
                name = f"{row['pair_id']}-{role}-{adapter.mode}-{step}.wav"
                wavfile.write(listening / name, SAMPLE_RATE, predicted[0].T.cpu().numpy())
        progress.update(len(cases))
    summary = BASE.aggregate_cases(cases)
    summary.update({key: float(np.mean([row[key] for row in cases])) for key in
                    ("mrstft", "side_to_mid_error_db", "rms_relative", "forward_seconds")})
    BASE.write_json(output / f"validation-{adapter.mode}-{step}.json", {"metrics": summary, "cases": cases})
    return summary


def arm(output, adapter, training, validation, controls, latents, waves, direct, progress, started):
    deadline = started + CAP_SECONDS
    rng = BASE.cpu_generator(SEED)
    shuffle = random.Random(SEED)
    optimizer = torch.optim.AdamW(itertools.chain(adapter.backbone.parameters(), adapter.head.parameters()),
                                  lr=.001, weight_decay=.01)
    pair_cache = BASE._pair_loader(BASE.TRAINING_SOURCE / "pairs")

    @lru_cache(maxsize=128)
    def target(row_id, source):
        return torch.load(latents / f"train-{source}-{row_id}.pt", weights_only=True)

    report = {"mode": adapter.mode, "updates": 0, "checkpoints": {}, "stopped": None, "logs": []}
    arm_path = output / f"arm-{adapter.mode}.json"
    durations, block_loss = [], []
    cursor, order = 0, []
    torch.xpu.reset_peak_memory_stats()

    def save(step):
        if step in MILESTONES:
            validation_start = perf_counter()
            readings = validate(adapter, validation, controls, latents, waves, deadline, output, step, progress)
            durations.append(perf_counter() - validation_start)
            report["checkpoints"][str(step)] = {**readings, "peak_xpu_bytes": torch.xpu.max_memory_allocated(),
                                               "matched_direct_spectral_loss": direct.get(str(step), {}).get("spectral_loss")}
        BASE.atomic_torch_save(output / f"adapter-{adapter.mode}-{step}.pt", {
            "backbone": adapter.backbone.state_dict(), "head": adapter.head.state_dict(),
            "optimizer": optimizer.state_dict(), "update": step, "rng": rng.get_state(),
            "shuffle": shuffle.getstate(), "order": order, "cursor": cursor,
        })
        BASE.write_json(arm_path, report)

    try:
        save(0)
        progress.set(f"Stage 2 · {adapter.mode} updates", MAX_UPDATES, 2)
        tick = perf_counter()
        train_seconds = 0.
        for step in range(1, MAX_UPDATES + 1):
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
            rows = BASE.control_tensor(controls, patches, DEVICE)
            played = torch.stack([record["performance"] for record in records]).to(DEVICE)
            lengths = torch.cat([record["measurements"]["lengths"] for record in records]).to(DEVICE)
            expected = [z for row in selected for z in target(row["pair_id"], row["source_stage"])]
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for i, latent in enumerate(expected):
                if perf_counter() >= deadline:
                    raise TimeoutError("the arm reached its five-hour cap during an update")
                clip = slice(i, i + 1)
                loss = adapter.train_loss(rows[clip], played[clip], lengths[clip], latent[None].to(DEVICE), rng)
                (loss / len(expected)).backward()
                losses.append(loss.detach())
            gradient = torch.nn.utils.clip_grad_norm_(itertools.chain(adapter.backbone.parameters(),
                                                                      adapter.head.parameters()), 5.)
            if not torch.isfinite(gradient) or (step == 1 and gradient <= 0):
                raise RuntimeError("adapter gradients are missing or nonfinite")
            optimizer.step()
            block_loss.append(torch.stack(losses).mean().item())
            report["updates"] = step
            progress.update(step)
            if step % 100 == 0:
                block_seconds = perf_counter() - tick
                train_seconds += block_seconds
                remaining_checks = sum(m > step for m in MILESTONES)
                projected = perf_counter() - started + train_seconds / step * (MAX_UPDATES - step)
                projected += np.mean(durations) * remaining_checks
                entry = {"update": step, "loss": float(np.mean(block_loss)),
                         "updates_per_second": 100 / block_seconds, "projected_total_seconds": float(projected),
                         "peak_xpu_bytes": torch.xpu.max_memory_allocated()}
                report["logs"].append(entry)
                block_loss.clear()
                progress.update(step, log=True, message=f"{entry['updates_per_second']:.2f} updates/s, "
                                                        f"projected {projected / 3600:.2f} h")
                save(step)
                if step == 100 and projected > CAP_SECONDS:
                    report["stopped"] = "update-100 projection exceeds five hours"
                    break
                if step in MILESTONES and step < MAX_UPDATES:
                    progress.set(f"Stage 2 · {adapter.mode} updates", MAX_UPDATES, 2)
                    progress.update(step)
                tick = perf_counter()
    except (RuntimeError, TimeoutError, ValueError) as exc:
        report["stopped"] = str(exc)
        progress.failures += 1
        progress.update(progress.done, log=True, message=str(exc))
    report["elapsed_seconds"] = perf_counter() - started
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
    midi = inversion.performance_inputs(performance)[None].to(DEVICE)
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
                result = inversion._evaluate_case(case, {adapter.mode: adapter}, controls, record,
                                                   device=DEVICE, steps=plan["evaluation"]["optimizer"]["steps"])
                result["direct_native_mrstft"] = original.get("models", {}).get("residual_mlp", {}).get("native_mrstft")
                result["wall_seconds"] = perf_counter() - started
                measured = result.get("models", {}).get(adapter.mode, {}).get("native_mrstft")
                if measured is None:
                    progress.failures += 1
                else:
                    source, candidate = (capture[f"{case_id}:{suffix}"] for suffix in ("source", adapter.mode))
                    for role in ("source", "start"):
                        label = f"{case_id}:{role}"
                        if label in original_receipts:
                            experiment.require_audio(capture[label], original_receipts[label], label)
                    gain = float((source * candidate).sum() / max(1e-12, float((candidate ** 2).sum())))
                    result["gain_matched_mrstft"] = experiment.audio_distance(source, candidate * gain)["mrstft"]
                    result["improved"] = measured < result["start_native_mrstft"]
                    exported = BASE.PilotTarget.from_dict(result["models"][adapter.mode]["patch"])
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
    observed = [row for row in results if row.get("models", {}).get(adapter.mode, {}).get("native_mrstft") is not None]
    matched = [row for row in observed if row["direct_native_mrstft"] is not None]
    differences = [row["models"][adapter.mode]["native_mrstft"] - row["direct_native_mrstft"] for row in matched]
    stats = {"planned": len(cases), "valid": len(observed), "matched": len(matched), "model": adapter.mode,
             "peak_xpu_bytes": torch.xpu.max_memory_allocated(), "improved": sum(row["improved"] for row in observed)}
    stats["legal_patches"] = sum(row.get("legal_patch", False) for row in observed)
    stats["perturbations"] = [row["perturbation"] for row in observed if "perturbation" in row]
    if matched:
        stats["matched_mrstft_difference"] = float(np.mean(differences))
        stats["patch_root_and_midi_interval"] = inversion._bootstrap_interval(matched, differences)
    BASE.write_json(output / "stage3.json", stats)


def run_later_stages(output, stage1, _source, plan, training, validation, controls, waves, progress):
    direct = experiment.read_json(DIRECT / "summary.json")["checkpoints"]["residual_mlp-direct"]
    direct = {step: row["metrics"] for step, row in direct.items()}
    threshold = direct["8000"]["spectral_loss"]
    eligible = [name for name, row in stage1.items()
                if row["reconstruction"]["spectral_loss"] < threshold
                and math.isfinite(row["reconstruction"]["mrstft"])
                and 0 < row["reconstruction"]["decoder_gradient_norm"] < math.inf]
    winner = min(eligible, key=lambda name: (name not in experiment.GENERATORS,
                                            stage1[name]["reconstruction"]["spectral_loss"]), default=None)
    decision = {"stage1_selected": winner, "stage1_gate": {"spectral_loss_below": threshold, "requires_gradient": True},
                "stage2_selected": None, "stage3": "waiting", "training_update_cap": MAX_UPDATES,
                "seconds_per_arm": CAP_SECONDS, "generator_euler_steps": EULER_STEPS,
                "seed": SEED, "effective_batch_pairs": 4,
                "stage2_gate": "finish 8000 updates and beat the matched direct MLP by at least 2%"}
    path = output / "decisions.json"
    BASE.write_json(path, decision)
    if winner is None:
        decision.update(stage3="skipped", reason="no codec passed the Stage 1 reconstruction and gradient gate")
        BASE.write_json(path, decision)
        progress.update(progress.done, log=True, message=decision["reason"])
        return

    decision.update(stage2="blocked_pending_adapter_integration", stage3="not_started",
                    reason="Stage 2 preparation introduced replacement adapters, but this legacy runner still constructs the superseded Adapter. Integrate the prepared backbones before resuming; preserve existing latent cache files.")
    BASE.write_json(path, decision)
    progress.update(progress.done, log=True, message=decision["reason"])
    return

    compatible_generator = winner in experiment.GENERATORS
    if compatible_generator and not generator_codec_matches(winner):
        embedded = experiment.Codec(f"{winner}-embedded", experiment.GENERATORS[winner])
        recon = experiment.reconstruct(output, embedded.name, embedded, validation, controls, waves, progress)
        decision["embedded_codec"] = recon
        compatible_generator = recon["spectral_loss"] < threshold and 0 < recon["decoder_gradient_norm"] < math.inf
        if compatible_generator:
            codec = embedded
        else:
            del embedded
            torch.xpu.empty_cache()
            codec = experiment.Codec(winner)
    else:
        codec = experiment.Codec(winner)
    decision["stage2_codec_revision"] = codec.checkpoint.name
    BASE.write_json(path, decision)
    latents, cache_seconds = cache_latents(output, codec, controls, training, validation, waves, progress)
    decision["shared_latent_cache_seconds"] = cache_seconds
    first = torch.load(latents / f"validation-{validation[0]['pair_id']}.pt", weights_only=True)[0]
    width = BASE.control_tensor(controls, [BASE.PilotTarget.from_dict(validation[0]["target"])], DEVICE).shape[-1]
    available = ("latent", "generator") if compatible_generator else ("latent",)
    if "generator" not in available:
        decision["generator"] = "no qualified compatible generator for the selected codec"
    reports = {}
    for mode in available:
        started = perf_counter()
        progress.set(f"Stage 2 · Load {mode} arm", 1, 2)
        try:
            torch.manual_seed(SEED)
            generator, token_width = generator_model(winner) if mode == "generator" else (None, None)
            adapter = Adapter(mode, codec, width, first.shape[0], generator, token_width).to(DEVICE)
            reports[mode] = arm(output, adapter, training, validation, controls, latents, waves, direct, progress, started)
            del adapter, generator
        except (RuntimeError, ValueError, OSError) as exc:
            reports[mode] = {"stopped": str(exc), "checkpoints": {}}
            progress.failures += 1
            progress.update(progress.done, log=True, message=str(exc))
        torch.xpu.empty_cache()
    BASE.write_json(output / "stage2.json", reports)
    viable = {mode: report["checkpoints"]["8000"]["spectral_loss"] for mode, report in reports.items()
              if "8000" in report["checkpoints"] and report["checkpoints"]["8000"]["spectral_loss"] < .98 * threshold}
    mode = min(viable, key=viable.get, default=None)
    decision["stage2_selected"] = mode
    if mode is None:
        decision.update(stage3="skipped", reason="neither adapter met the matched Stage 2 gate")
        BASE.write_json(path, decision)
        progress.update(progress.done, log=True, message=decision["reason"])
        return
    decision["stage3"] = "running"
    BASE.write_json(path, decision)
    generator, token_width = generator_model(winner) if mode == "generator" else (None, None)
    adapter = Adapter(mode, codec, width, first.shape[0], generator, token_width).to(DEVICE)
    saved = torch.load(output / f"adapter-{mode}-8000.pt", map_location="cpu", weights_only=True)
    adapter.backbone.load_state_dict(saved["backbone"])
    adapter.head.load_state_dict(saved["head"])
    adapter.eval().requires_grad_(False)
    inverse_panel(output, adapter, controls, plan, progress)
    decision["stage3"] = "complete"
    BASE.write_json(path, decision)

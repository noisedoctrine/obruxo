"""Small, time-queryable spectral helper; qualification remains a separate native gate."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .codec import PilotTarget
from .proxy import PERFORMANCE_SIZE
from .spectral import ENVELOPE_HOP, ENVELOPE_WINDOW, FFT_SIZES, SAMPLE_RATE, spectral_loss
from .spectral_controls import temperature_at


def temporal_proxy_spec():
    return {"model": "basic_note_conditioned_spectral_frames_v1", "note_width": 32, "hidden_width": 128,
            "conditioning": "canonical_controls_actual_notes_elapsed_time_and_resolution",
            "output": "nonnegative_linear_magnitudes_from_softplus_log1p_head", "log_floor": .0001,
            "log_ceiling": 16., "training_frames_per_resolution": 16, "training_envelope_frames": 16,
            "frame_sampling": "uniform_without_replacement_valid_centers_keep_absolute_times_no_frequency_pooling",
            "qualification": "full_temporal_targets", "maximum_trainable_parameters": 2_000_000}


def hard_controls(adapter, targets):
    values = torch.tensor([target.continuous for target in targets], dtype=torch.float32)
    categories = torch.tensor([target.categorical for target in targets])
    scores = [F.one_hot(categories[:, i], width).float() for i, width in enumerate((3, 2))]
    return adapter(values, scores, sample=False)


def full_times(target):
    times = [torch.arange(value.shape[-1], device=value.device).float()[None].expand(len(value), -1) * (size // 4) / SAMPLE_RATE
             for value, size in zip(target["magnitudes"], FFT_SIZES, strict=True)]
    times.append(torch.arange(target["envelope"].shape[-1], device=target["envelope"].device).float()[None]
                 .expand(len(target["envelope"]), -1) * ENVELOPE_HOP / SAMPLE_RATE)
    return tuple(times)


def sample_frames(target, generator, count=16):
    """Retain exact full-frequency measurements at sampled times, not averaged time bins."""
    if len(target["lengths"]) != 1 or type(count) is not int or count < 1:
        raise ValueError("sample one measured recording at a time")
    times = full_times(target)
    values, sampled_times = [], []
    for index, (value, mask) in enumerate(zip((*target["magnitudes"], target["envelope"]),
                                            (*target["masks"], target["envelope_mask"]), strict=True)):
        valid = torch.nonzero(mask[0], as_tuple=False).flatten()
        if len(valid) < count:
            raise ValueError("recording has too few valid frames")
        selected = valid[torch.randperm(len(valid), generator=generator)[:count]].sort().values
        values.append(value[..., selected].clone())
        sampled_times.append(times[index][:, selected].clone())
    measured = {"magnitudes": tuple(values[:3]), "envelope": values[3], "lengths": target["lengths"].clone(),
                "masks": tuple(torch.ones(1, count, dtype=torch.bool) for _ in FFT_SIZES),
                "envelope_mask": torch.ones(1, count, dtype=torch.bool)}
    return measured, tuple(sampled_times)


def collate_frames(records):
    targets = [record["measurements"] for record in records]
    return ({"magnitudes": tuple(torch.cat([t["magnitudes"][r] for t in targets]) for r in range(3)),
             "masks": tuple(torch.cat([t["masks"][r] for t in targets]) for r in range(3)),
             **{key: torch.cat([t[key] for t in targets]) for key in ("envelope", "envelope_mask", "lengths")}},
            tuple(torch.cat([record["times"][r] for record in records]) for r in range(4)))


class TemporalSpectralProxy(nn.Module):
    def __init__(self):
        super().__init__()
        self.note = nn.Sequential(nn.Linear(7, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU())
        self.trunk = nn.Sequential(nn.Linear(54, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU())
        self.heads = nn.ModuleList(nn.Linear(128, width) for width in (*[2 * (n // 2 + 1) for n in FFT_SIZES], 3))
        for head in self.heads:
            nn.init.normal_(head.weight, std=.001)
            nn.init.constant_(head.bias, -2.)
        if sum(p.numel() for p in self.parameters()) > 2_000_000:
            raise ValueError("helper exceeds its declared parameter budget")

    def _hidden(self, controls, performance, times, resolution):
        if controls.shape != (len(performance), 14) or performance.shape[1] != PERFORMANCE_SIZE:
            raise ValueError("helper requires basic controls and actual note/timing inputs")
        notes = performance[:, :80].reshape(-1, 16, 5)
        time = times[:, :, None] / 16
        pitch, velocity, start, duration, valid = [notes[:, None, :, i] for i in range(5)]
        elapsed = time - start
        released = elapsed - duration
        active = ((elapsed >= 0) & (released < 0)).float()
        window = (*FFT_SIZES, ENVELOPE_WINDOW)[resolution]
        seen = (elapsed >= -window / (2 * SAMPLE_RATE * 16)).float() * valid
        features = torch.stack((pitch.expand_as(elapsed), velocity.expand_as(elapsed), duration.expand_as(elapsed),
                                elapsed, released, active, seen), dim=-1)
        note_features = (self.note(features) * seen[..., None]).sum(2)
        count = times.shape[1]
        code = F.one_hot(torch.full((len(controls), count), resolution, device=controls.device), 4).float()
        context = torch.cat((controls[:, None].expand(-1, count, -1), note_features,
                             performance[:, None, 80:].expand(-1, count, -1), time, code), dim=-1)
        return self.trunk(context)

    def forward(self, controls, performance, times):
        outputs = []
        for r, (time, head) in enumerate(zip(times, self.heads, strict=True)):
            log_magnitude = F.softplus(head(self._hidden(controls, performance, time, r))).clamp_max(16.)
            values = .0001 * torch.expm1(log_magnitude)
            if r < 3:
                values = values.reshape(len(values), values.shape[1], 2, FFT_SIZES[r] // 2 + 1).permute(0, 2, 3, 1)
            else:
                values = values.transpose(1, 2)
            outputs.append(values)
        return {"magnitudes": tuple(outputs[:3]), "envelope": outputs[3]}

    def freeze(self):
        self.eval().requires_grad_(False)
        self.zero_grad(set_to_none=True)
        return self


def guide_patch(helper, adapter, starting, performance, target, *, joint, seed, steps=16):
    """Bounded qualification only: source controls are not inputs to the optimization."""
    if helper.training or any(p.requires_grad for p in helper.parameters()):
        raise ValueError("freeze the helper before using its input gradients")
    if type(joint) is not bool or type(steps) is not int or steps < 1:
        raise ValueError("invalid guided update")
    torch.manual_seed(seed)
    start = torch.tensor([starting.continuous], dtype=torch.float32)
    continuous = nn.Parameter(start.clone())
    choices = [F.one_hot(torch.tensor([category]), size).float() * .2 for category, size in zip(starting.categorical, (3, 2), strict=True)]
    scores = [nn.Parameter(value) if joint else value for value in choices]
    optimizer = torch.optim.Adam([continuous, *(scores if joint else [])], lr=.03)
    times = full_times(target)
    category_gradient, initial = 0., None
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        encoded = adapter(continuous, scores, sample=joint, temperature=temperature_at(step, steps))
        loss = spectral_loss(helper(encoded, performance[None], times), target)["loss"].mean()
        if not torch.isfinite(loss):
            raise ValueError("nonfinite guided objective")
        if initial is None:
            initial = float(loss.detach())
        loss.backward()
        gradients = [continuous.grad, *(score.grad for score in scores if joint)]
        if not all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients):
            raise ValueError("missing or nonfinite guided gradients")
        if joint:
            category_gradient += sum(float(score.grad.abs().sum()) for score in scores)
        optimizer.step()
        with torch.no_grad():
            continuous.copy_(torch.minimum(torch.maximum(continuous, start - .15), start + .15).clamp(0, 1))
    result = adapter.export(continuous, scores)[0]
    with torch.no_grad():
        final = float(spectral_loss(helper(hard_controls(adapter, [result]), performance[None], times), target)["loss"].item())
    return result, {"initial_sampled_error": initial, "final_hard_error": final,
                    "maximum_continuous_change": float((continuous.detach() - start).abs().max()),
                    "category_gradient_l1": category_gradient, "category_changed": result.categorical != starting.categorical}


def perturbed_targets(codec, target, index):
    values = [min(.95, max(.05, value + (.2 if (index + i) % 2 else -.2))) for i, value in enumerate(target.continuous)]
    start = codec.canonical_target(PilotTarget(tuple(values), target.categorical))
    candidates = {"start": start}
    for name, delta in (("plus", .05), ("minus", -.05)):
        changed = [min(1., max(0., value + delta)) for value in start.continuous]
        candidates[name] = codec.canonical_target(PilotTarget(tuple(changed), start.categorical))
    category = index % 2
    choices = list(start.categorical)
    choices[category] = (choices[category] + 1) % (3 if category == 0 else 2)
    candidates["category"] = codec.canonical_target(PilotTarget(start.continuous, tuple(choices)))
    for name, position in (("pan", 1), ("detune", 2)):
        changed = list(start.continuous)
        changed[position] += .1 if changed[position] <= .5 else -.1
        candidates[name] = codec.canonical_target(PilotTarget(tuple(changed), start.categorical))
    return candidates

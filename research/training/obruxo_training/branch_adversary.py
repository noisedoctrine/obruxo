"""Training-only note removal from the timbre branch; no codec or inference heads."""

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from .paired import identity_terms

SPEC = {"format": "basic_timbre_note_adversary_v1", "branch_width": 768, "head_hidden": 32, "training_outputs": 2,
        "targets": ["mean_midi_scaled_36_to_84", "mean_note_seconds_divided_by_3"], "velocity_target": False,
        "normalization": "per_example_layer_norm_no_learned_statistics", "head_loss": "mean_squared_error",
        "encoder_gradient": "negative_coefficient_times_head_loss_gradient", "additional_inference_outputs": 0}


class ReverseGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, coefficient):
        if not math.isfinite(coefficient) or coefficient < 0:
            raise ValueError("reversal coefficient must be finite and nonnegative")
        ctx.coefficient = coefficient
        return value.clone()

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.coefficient * gradient, None


class NoteHead(nn.Module):
    def __init__(self, *, seed=0):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.layers = nn.Sequential(nn.Linear(768, 32), nn.GELU(), nn.Linear(32, 2))

    def forward(self, value, *, coefficient):
        if value.ndim != 2 or value.shape[1] != 768 or not torch.isfinite(value).all():
            raise ValueError("note adversary requires finite 768-dimensional timbre vectors")
        normalized = F.layer_norm(value, (768,))
        return self.layers(ReverseGradient.apply(normalized, coefficient))


def note_targets(rows):
    values = torch.tensor([[(row["performance"][0] - 36.) / 48., row["performance"][2] / 3.] for row in rows], dtype=torch.float32)
    if not torch.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("mean pitch/duration targets exceed the declared cached-training domain")
    return values


def initialize(encoder, *, seed=0, encoder_lr=1e-4, head_lr=1e-3):
    copied, head = copy.deepcopy(encoder).requires_grad_(True).train(), NoteHead(seed=seed)
    return {"encoder": copied, "head": head, "encoder_optimizer": torch.optim.Adam(copied.parameters(), lr=encoder_lr),
            "head_optimizer": torch.optim.Adam(head.parameters(), lr=head_lr)}


def warm_head(arm, values, targets):
    arm["head_optimizer"].zero_grad(set_to_none=True)
    loss = F.mse_loss(arm["head"](values.detach(), coefficient=0.), targets)
    if not torch.isfinite(loss):
        raise ValueError("nonfinite detached note-head warmup")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(arm["head"].parameters(), 5., error_if_nonfinite=True)
    arm["head_optimizer"].step()
    return float(loss.detach())


def step(arm, mel, targets, *, coefficient):
    for name in ("encoder_optimizer", "head_optimizer"):
        arm[name].zero_grad(set_to_none=True)
    representation = arm["encoder"](mel)
    identity = identity_terms(representation, temperature=.1)["identity_loss"]
    note = F.mse_loss(arm["head"](representation, coefficient=coefficient), targets)
    loss = identity + note
    if not torch.isfinite(loss):
        raise ValueError("nonfinite paired/adversarial encoder objective")
    loss.backward()
    encoder_norm = torch.nn.utils.clip_grad_norm_(arm["encoder"].parameters(), 5., error_if_nonfinite=True)
    head_norm = torch.nn.utils.clip_grad_norm_(arm["head"].parameters(), 5., error_if_nonfinite=True)
    for name in ("encoder_optimizer", "head_optimizer"):
        arm[name].step()
    return {"identity_loss": float(identity.detach()), "note_head_mse": float(note.detach()), "coefficient": coefficient,
            "encoder_gradient_norm": float(encoder_norm), "head_gradient_norm": float(head_norm)}


def snapshot(arm):
    return copy.deepcopy({name: value.state_dict() for name, value in arm.items()})


def restore(arm, saved):
    if set(arm) != set(saved):
        raise ValueError("branch checkpoint components changed")
    for name, value in arm.items():
        value.load_state_dict(saved[name])


def pair_indices(by_state, generator, *, patches=32):
    states = sorted(by_state)
    if len(states) < patches or patches < 2 or any(len(indices) != 3 for indices in by_state.values()):
        raise ValueError("branch sampling needs complete three-view patches and enough distinct identities")
    selected = torch.randperm(len(states), generator=generator)[:patches].tolist()
    return [by_state[states[state]][view] for state in selected for view in torch.randperm(3, generator=generator)[:2].tolist()]

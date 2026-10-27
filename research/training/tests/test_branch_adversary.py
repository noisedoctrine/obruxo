import copy

import pytest
import torch
from torch import nn

from obruxo_training.branch_adversary import NoteHead, ReverseGradient, initialize, note_targets, pair_indices, restore, snapshot, step, warm_head
from obruxo_training.paired_native_audit import equal_tree


def test_reversal_changes_only_input_gradient_and_preserves_head_learning_direction():
    value = torch.randn(4, 768, requires_grad=True)
    head = NoteHead(seed=2)
    target = torch.rand(4, 2)
    gradients = []
    for coefficient in (0., 1.):
        head.zero_grad(set_to_none=True)
        value.grad = None
        torch.nn.functional.mse_loss(head(value, coefficient=coefficient), target).backward()
        gradients.append((value.grad.clone(), [p.grad.clone() for p in head.parameters()]))
    direct = value.detach().clone().requires_grad_(True)
    torch.nn.functional.mse_loss(head.layers(torch.nn.functional.layer_norm(direct, (768,))), target).backward()
    assert torch.equal(gradients[0][0], torch.zeros_like(value))
    assert torch.equal(gradients[1][0], -direct.grad)
    assert all(torch.equal(a, b) for a, b in zip(gradients[0][1], gradients[1][1], strict=True))
    with pytest.raises(ValueError, match="coefficient"):
        ReverseGradient.apply(value, -1.)


def test_note_targets_use_known_pitch_and_duration_and_exclude_velocity():
    rows = [{"performance": [60., 64., 1.5]}, {"performance": [60., 112., 1.5]}]
    assert torch.equal(note_targets(rows), torch.full((2, 2), .5))
    with pytest.raises(ValueError, match="domain"):
        note_targets([{"performance": [120., 64., 1.]}])


def test_sampling_keeps_distinct_complete_patch_pairs_and_generator_recovery():
    bank = {str(i): list(range(3 * i, 3 * i + 3)) for i in range(40)}
    generator = torch.Generator().manual_seed(8)
    before = generator.get_state()
    indices = pair_indices(bank, generator)
    other = torch.Generator()
    other.set_state(before)
    assert indices == pair_indices(bank, other)
    assert len(set(index // 3 for index in indices)) == 32
    assert all(indices[i] // 3 == indices[i + 1] // 3 and indices[i] != indices[i + 1] for i in range(0, 64, 2))


def test_detached_head_warmup_and_encoder_adam_checkpoint_resume_are_exact():
    torch.set_num_threads(2)
    torch.manual_seed(5)
    encoder = nn.Sequential(nn.Flatten(), nn.Linear(12, 768), nn.GELU())
    arm = initialize(encoder)
    mel, targets = torch.rand(8, 3, 4), torch.rand(8, 2)
    original = copy.deepcopy(arm["encoder"].state_dict())
    with torch.no_grad():
        values = arm["encoder"](mel)
    warm_head(arm, values, targets)
    assert equal_tree(original, arm["encoder"].state_dict())
    step(arm, mel, targets, coefficient=.5)
    saved = snapshot(arm)
    expected = step(arm, mel, targets, coefficient=1.)
    cold = initialize(encoder)
    restore(cold, saved)
    actual = step(cold, mel, targets, coefficient=1.)
    assert actual == expected and equal_tree(snapshot(arm), snapshot(cold))
    assert not equal_tree(original, arm["encoder"].state_dict())

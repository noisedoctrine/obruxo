"""Check that Issue 185 recovery resumes model, optimizer, and RNG state exactly."""

import copy
import importlib.util
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "reports/scripts/run_issue185_learning_curves.py"
SPEC = importlib.util.spec_from_file_location("issue185_runner", SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def train_updates(model, optimizer, shuffle, noise_rng, order, cursor, rows, mode, count):
    values = torch.arange(len(rows) * 2, dtype=torch.float32).reshape(len(rows), 2) / 10
    targets = values.sum(dim=-1, keepdim=True)
    for _ in range(count):
        if cursor + 4 > len(order):
            order = torch.randperm(len(rows), generator=shuffle).tolist()
            cursor = 0
        indices = order[cursor:cursor + 4]
        cursor += 4
        batch = values[indices]
        noise = torch.randn(batch.shape, generator=noise_rng)
        if mode == "flow":
            batch = batch + noise * 0.1
        optimizer.zero_grad(set_to_none=True)
        loss = (model(batch) - targets[indices]).square().mean()
        loss.backward()
        optimizer.step()
    return order, cursor


@pytest.mark.parametrize("arm", runner.ARMS)
def test_recovery_matches_uninterrupted_model_optimizer_and_rng(arm, tmp_path):
    torch.manual_seed(183)
    initial = torch.nn.Linear(2, 1)
    initial_optimizer = torch.optim.AdamW(initial.parameters(), lr=.001)
    shuffle = torch.Generator(device="cpu").manual_seed(183)
    noise_rng = runner.BASE.cpu_generator(1183)
    rows = list(range(13))
    order = torch.randperm(len(rows), generator=shuffle).tolist()
    order, cursor = train_updates(initial, initial_optimizer, shuffle, noise_rng, order, 0, rows,
                                  arm.rsplit("-", 1)[1], 500)
    initial_state = copy.deepcopy(initial.state_dict())
    initial_optimizer_state = copy.deepcopy(initial_optimizer.state_dict())
    initial_shuffle_state = shuffle.get_state()
    initial_noise_state = noise_rng.get_state()
    initial_order, initial_cursor = order[:], cursor

    uninterrupted = torch.nn.Linear(2, 1)
    uninterrupted.load_state_dict(initial_state)
    uninterrupted_optimizer = torch.optim.AdamW(uninterrupted.parameters(), lr=.001)
    uninterrupted_optimizer.load_state_dict(copy.deepcopy(initial_optimizer_state))
    uninterrupted_shuffle = torch.Generator(device="cpu")
    uninterrupted_shuffle.set_state(initial_shuffle_state)
    uninterrupted_noise = runner.BASE.cpu_generator(0)
    uninterrupted_noise.set_state(initial_noise_state)
    uninterrupted_order, uninterrupted_cursor = train_updates(
        uninterrupted, uninterrupted_optimizer, uninterrupted_shuffle, uninterrupted_noise,
        initial_order[:], initial_cursor, rows, arm.rsplit("-", 1)[1], 500)

    interrupted = torch.nn.Linear(2, 1)
    interrupted.load_state_dict(initial_state)
    interrupted_optimizer = torch.optim.AdamW(interrupted.parameters(), lr=.001)
    interrupted_optimizer.load_state_dict(copy.deepcopy(initial_optimizer_state))
    interrupted_shuffle = torch.Generator(device="cpu")
    interrupted_shuffle.set_state(initial_shuffle_state)
    interrupted_noise = runner.BASE.cpu_generator(0)
    interrupted_noise.set_state(initial_noise_state)
    recovery_order, recovery_cursor = train_updates(
        interrupted, interrupted_optimizer, interrupted_shuffle, interrupted_noise,
        initial_order[:], initial_cursor, rows, arm.rsplit("-", 1)[1], 100)
    recovery = tmp_path / "recovery.pt"
    runner.save_state(
        recovery, plan_id="fixture-plan", arm=arm, step=600, model=interrupted,
        optimizer=interrupted_optimizer, order=recovery_order, cursor=recovery_cursor,
        shuffle=interrupted_shuffle, noise_rng=interrupted_noise,
        objective_total=0.0, objective_count=0, source_state_sha256="fixture-source",
    )

    restored = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=.001)
    restored_shuffle = torch.Generator(device="cpu")
    restored_noise = runner.BASE.cpu_generator(0)
    saved = torch.load(recovery, map_location="cpu", weights_only=True)
    step, resumed_order, resumed_cursor, _, _ = runner.restore_state(
        saved, plan_id="fixture-plan", arm=arm, source_state_sha256="fixture-source",
        model=restored, optimizer=restored_optimizer, shuffle=restored_shuffle,
        noise_rng=restored_noise, device=torch.device("cpu"),
    )
    assert step == 600
    resumed_order, resumed_cursor = train_updates(
        restored, restored_optimizer, restored_shuffle, restored_noise,
        resumed_order, resumed_cursor, rows, arm.rsplit("-", 1)[1], 400)

    for name, value in uninterrupted.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
    for name, value in uninterrupted_optimizer.state_dict()["state"].items():
        for key, tensor in value.items():
            other = restored_optimizer.state_dict()["state"][name][key]
            if isinstance(tensor, torch.Tensor):
                torch.testing.assert_close(other, tensor, rtol=0, atol=0)
            else:
                assert other == tensor
    assert resumed_order == uninterrupted_order
    assert resumed_cursor == uninterrupted_cursor
    assert torch.equal(restored_shuffle.get_state(), uninterrupted_shuffle.get_state())
    assert torch.equal(restored_noise.get_state(), uninterrupted_noise.get_state())

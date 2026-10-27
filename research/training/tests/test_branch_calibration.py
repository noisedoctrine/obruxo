import pytest
import torch
from torch import nn

from obruxo_training.branch_adversary import initialize, snapshot
from obruxo_training.branch_calibration import coefficient, measure
from obruxo_training.paired_native_audit import equal_tree


def test_calibration_equalizes_median_gradient_ratio_instead_of_loss_values():
    rows = [{"identity_norm": float(value), "unscaled_reversed_note_norm": .25 * value} for value in range(1, 9)]
    assert coefficient(rows) == 4.
    rows[-1]["unscaled_reversed_note_norm"] = 1000.
    assert coefficient(rows) == 4.


@pytest.mark.parametrize("note", [0., float("nan"), .000001])
def test_bad_or_extreme_calibration_is_refused_without_silent_clamping(note):
    rows = [{"identity_norm": 1., "unscaled_reversed_note_norm": note}] * 8
    with pytest.raises(ValueError):
        coefficient(rows)


def test_measurement_preserves_encoder_head_adam_rng_and_existing_gradients():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    arm = initialize(nn.Sequential(nn.Flatten(), nn.Linear(12, 768), nn.GELU()))
    mel, targets = torch.rand(8, 3, 4), torch.rand(8, 2)
    for parameter in arm["encoder"].parameters():
        parameter.grad = torch.ones_like(parameter)
    before, rng = snapshot(arm), torch.get_rng_state().clone()
    row = measure(arm, mel, targets)
    assert row["identity_norm"] > 0 and row["unscaled_reversed_note_norm"] > 0
    assert -1 <= row["gradient_cosine"] <= 1
    assert equal_tree(before, snapshot(arm)) and torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(parameter.grad, torch.ones_like(parameter)) for parameter in arm["encoder"].parameters())


def test_standalone_audit_restores_python_numpy_and_torch_before_replay(tmp_path, monkeypatch):
    import importlib.util
    import random
    from pathlib import Path
    import sys
    from types import ModuleType

    import numpy as np
    from obruxo_training.audit import write_json
    from obruxo_training.conditioned_bridge import read
    from obruxo_training.trainer import capture_rng, restore_rng

    original = capture_rng("cpu")
    try:
        random.seed(11)
        np.random.seed(11)
        torch.manual_seed(11)
        expected = capture_rng("cpu")
        screen, audit = tmp_path / "screen", tmp_path / "audit"
        screen.mkdir()
        torch.save({"rng": expected}, screen / "checkpoint.pt")
        random.seed(13)
        np.random.seed(13)
        torch.manual_seed(13)
        observed = []

        def replay():
            observed.append(capture_rng("cpu"))
            audit.mkdir()
            for name in ("plan.json", "verification.json"):
                write_json(audit / name, {})

        target = ModuleType("run_calibrated_branch_screen")
        target.main = replay
        monkeypatch.setitem(sys.modules, target.__name__, target)
        monkeypatch.setattr(sys, "argv", ["audit", "--output", str(screen), "--audit-output", str(audit), "--mode", "verify"])
        path = Path(__file__).parents[1] / "reports/scripts/audit_calibrated_branch_screen.py"
        spec = importlib.util.spec_from_file_location("calibrated_audit_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.main()
        assert equal_tree(observed[0], expected)
        assert read(audit / "verification.json")["rng_restoration"]["full_global_rng_restored"] == ["python", "numpy", "torch"]
    finally:
        restore_rng(original, "cpu")

import importlib.util
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location("time_coverage", Path(__file__).parents[1] / "reports/scripts/check_spectral_time_coverage.py")
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def test_coverage_finds_unsampled_response_and_excludes_invalid_tail():
    values = torch.zeros(2, 2, 1, 5)
    values[1, :, :, 2] = 1.
    values[1, :, :, 4] = 100.
    valid = torch.tensor([[True, True, True, True, False]]).expand(2, -1)
    full = {"magnitudes": (values,) * 3, "envelope": values[:, :, 0], "masks": (valid,) * 3, "envelope_mask": valid}
    sampled = {"magnitudes": (values[..., [0, 3]],) * 3, "envelope": values[:, :, 0, [0, 3]]}
    times = torch.arange(5).float()[None].expand(2, -1)
    rows = diagnostic.coverage(full, sampled, (times,) * 4, (times[:, [0, 3]],) * 4)
    for row in rows:
        assert row["sampled_mean_change"] == 0
        assert row["full_mean_change"] == pytest.approx(torch.log1p(torch.tensor(10000.)).item() / 4)
        assert row["peak_response_time_seconds"] == 2
        assert row["nearest_sample_to_peak_seconds"] == 1


def test_change_error_distinguishes_real_response_from_no_change():
    values = torch.ones(2, 2, 3, 4)
    values[1] *= 2
    valid = torch.ones(2, 4, dtype=torch.bool)
    target = {"magnitudes": (values,) * 3, "envelope": torch.ones(2, 3, 4), "masks": (valid,) * 3,
              "envelope_mask": valid, "lengths": torch.tensor([44100, 44100])}
    result = diagnostic.change_errors(target, target)
    assert result["helper"] == 0
    assert result["no_change"] == pytest.approx(1.)

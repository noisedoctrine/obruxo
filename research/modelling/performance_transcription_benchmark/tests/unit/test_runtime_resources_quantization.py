from __future__ import annotations

from pathlib import Path

import pytest
import torch
from obruxo_performance_benchmark.benchmark import (
    aggregate_measurements,
    fixed_timing_contract,
)
from obruxo_performance_benchmark.quantization import (
    quantize_dynamic_linear_int8,
    serialized_model_size,
    validate_quantized_route,
)
from obruxo_performance_benchmark.resources import (
    count_parameters,
    state_tensor_bytes_by_dtype,
)


def test_inherited_timing_arithmetic_and_toy_resources() -> None:
    summary = aggregate_measurements([1.0, 2.0, 3.0])
    assert summary["median"] == 2.0
    assert fixed_timing_contract()["batch_sizes"] == [1, 2, 4, 8]
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Conv1d(1, 1, 1))
    assert count_parameters(model) == (11, 11)
    assert state_tensor_bytes_by_dtype(model)["float32"] == 44


def test_dynamic_quantization_targets_only_linear_and_preserves_source(
    tmp_path: Path,
) -> None:
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Conv1d(1, 1, 1))
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    result = quantize_dynamic_linear_int8(model)
    assert result.status == "ok"
    assert result.original_linear_modules == 1
    assert result.quantized_linear_modules >= 1
    assert all(
        torch.equal(before[key], value) for key, value in model.state_dict().items()
    )
    assert serialized_model_size(result.model, tmp_path) > 0


def test_no_linear_and_forbidden_quantized_routes() -> None:
    result = quantize_dynamic_linear_int8(torch.nn.Conv1d(1, 1, 1))
    assert result.status == "not_applicable_no_linear"
    with pytest.raises(ValueError):
        validate_quantized_route(device="xpu")
    with pytest.raises(ValueError):
        validate_quantized_route(batch_size=2)
    with pytest.raises(ValueError):
        validate_quantized_route(backward=True)

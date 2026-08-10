"""The single permitted CPU dynamic qint8 Linear experiment."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QuantizationResult:
    status: str
    original_linear_modules: int
    quantized_linear_modules: int
    engine: str | None
    model: object | None


def _linear_modules(model: Any) -> list[Any]:
    import torch

    return [module for module in model.modules() if isinstance(module, torch.nn.Linear)]


def _is_cpu(model: Any) -> bool:
    devices = {tensor.device.type for tensor in (*model.parameters(), *model.buffers())}
    return not devices or devices == {"cpu"}


def quantize_dynamic_linear_int8(model: Any) -> QuantizationResult:
    import torch

    original = _linear_modules(model)
    engine = str(torch.backends.quantized.engine)
    if not _is_cpu(model):
        return QuantizationResult("quantization_unsupported", len(original), 0, engine, None)
    if not original:
        return QuantizationResult("not_applicable_no_linear", 0, 0, engine, model)
    before = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    try:
        quantized = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8, inplace=False)
    except (AttributeError, NotImplementedError, OSError, RuntimeError, TypeError, ValueError):
        return QuantizationResult("quantization_unsupported", len(original), 0, engine, None)
    for name, tensor in model.state_dict().items():
        if name not in before or not torch.equal(before[name], tensor.detach().cpu()):
            return QuantizationResult("quantization_unsupported", len(original), 0, engine, None)
    quantized_modules = [
        module
        for module in quantized.modules()
        if module.__class__.__module__.startswith("torch.ao.nn.quantized") and module.__class__.__name__.endswith("Linear")
    ]
    if not quantized_modules:
        return QuantizationResult("quantization_unsupported", len(original), 0, engine, None)
    return QuantizationResult("ok", len(original), len(quantized_modules), engine, quantized)


def validate_quantized_route(*, device: str = "cpu", batch_size: int = 1, backward: bool = False) -> None:
    if device != "cpu":
        raise ValueError("quantized evaluation is CPU-only")
    if batch_size != 1:
        raise ValueError("quantized evaluation does not run a batch sweep")
    if backward:
        raise ValueError("quantized evaluation does not run backward")


def serialized_model_size(model: Any, output_dir: Path) -> int:
    """Measure a temporary derived serialization, then remove only that file."""
    import torch

    root = Path(output_dir).resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=root, prefix=".quantized-", suffix=".pt", delete=False) as handle:
            temporary = Path(handle.name)
        torch.save(model, temporary)
        return temporary.stat().st_size
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

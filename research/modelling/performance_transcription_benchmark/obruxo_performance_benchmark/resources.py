"""Concrete parameter, tensor-byte, file-size, and memory helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def count_parameters(model: Any) -> tuple[int, int]:
    total = sum(int(parameter.numel()) for parameter in model.parameters())
    trainable = sum(
        int(parameter.numel())
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def state_tensor_bytes_by_dtype(model: Any) -> dict[str, int]:
    result: dict[str, int] = {}
    for tensor in model.state_dict().values():
        if not hasattr(tensor, "numel") or not hasattr(tensor, "element_size"):
            continue
        name = str(tensor.dtype).removeprefix("torch.")
        result[name] = result.get(name, 0) + int(tensor.numel()) * int(
            tensor.element_size()
        )
    return dict(sorted(result.items()))


def file_size(path: Path) -> int:
    return int(Path(path).stat().st_size)


def process_peak_rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except (ImportError, OSError, RuntimeError):
        return None


def xpu_memory_bytes() -> dict[str, int] | None:
    try:
        import torch

        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            return None
        return {
            "allocated": int(torch.xpu.memory_allocated()),
            "reserved": int(torch.xpu.memory_reserved()),
        }
    except (AttributeError, ImportError, RuntimeError):
        return None

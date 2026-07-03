"""Optional acceleration helpers.

Era 1 showed XPU is valuable for large batched oracle scoring, but not for tiny
smoke jobs. This module keeps that policy explicit: NumPy is always available,
and XPU is used only when requested or when `auto` sees a large enough workload.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Literal

import numpy as np

from .curve import as_curve_matrix

BackendPreference = Literal["auto", "numpy", "xpu"]


@dataclass(frozen=True)
class NearestResult:
    indices: np.ndarray
    losses: np.ndarray
    backend_used: str
    chunk_size: int


def xpu_available() -> bool:
    try:
        import torch

        return bool(hasattr(torch, "xpu") and torch.xpu.is_available())
    except Exception:
        return False


def choose_backend(
    preference: BackendPreference,
    *,
    pair_count: int,
    xpu_min_pairs: int = 131_072,
) -> str:
    if preference not in {"auto", "numpy", "xpu"}:
        raise ValueError("backend must be one of: auto, numpy, xpu")
    if preference == "numpy":
        return "numpy"
    if preference == "xpu":
        if not xpu_available():
            raise RuntimeError("XPU backend requested, but torch.xpu is unavailable")
        return "xpu"
    if pair_count >= xpu_min_pairs and xpu_available():
        return "xpu"
    return "numpy"


def nearest_indices(
    values: np.ndarray,
    codes: np.ndarray,
    *,
    backend: BackendPreference | None = None,
    chunk_size: int = 256,
    xpu_min_pairs: int = 131_072,
) -> NearestResult:
    """Find nearest code for each value row using mean squared error."""
    values_matrix = as_curve_matrix(values)
    codes_matrix = as_curve_matrix(codes)
    if values_matrix.shape[1] != codes_matrix.shape[1]:
        raise ValueError("values and codes must share the same resolution")
    if len(codes_matrix) == 0:
        raise ValueError("codes cannot be empty")

    preference = backend or os.environ.get("LFO_ERA2_BACKEND", "auto").strip().lower()
    selected = choose_backend(
        preference,  # type: ignore[arg-type]
        pair_count=int(len(values_matrix) * len(codes_matrix)),
        xpu_min_pairs=xpu_min_pairs,
    )
    if selected == "xpu":
        try:
            return _nearest_indices_torch_xpu(values_matrix, codes_matrix, chunk_size=chunk_size)
        except RuntimeError:
            if preference == "xpu":
                raise
            return _nearest_indices_numpy(values_matrix, codes_matrix, chunk_size=chunk_size)
    return _nearest_indices_numpy(values_matrix, codes_matrix, chunk_size=chunk_size)


def _nearest_indices_numpy(values: np.ndarray, codes: np.ndarray, *, chunk_size: int) -> NearestResult:
    chunk_size = max(1, int(chunk_size))
    indices = np.empty(len(values), dtype=np.int32)
    losses = np.empty(len(values), dtype=np.float32)
    for start in range(0, len(values), chunk_size):
        stop = min(start + chunk_size, len(values))
        difference = values[start:stop, None, :] - codes[None, :, :]
        mse = np.mean(difference * difference, axis=2)
        choice = np.argmin(mse, axis=1)
        indices[start:stop] = choice.astype(np.int32)
        losses[start:stop] = mse[np.arange(stop - start), choice].astype(np.float32)
    return NearestResult(indices=indices, losses=losses, backend_used="numpy", chunk_size=chunk_size)


def _nearest_indices_torch_xpu(values: np.ndarray, codes: np.ndarray, *, chunk_size: int) -> NearestResult:
    import torch

    device = "xpu:0"
    chunk_size = max(1, int(chunk_size))
    codes_t = torch.as_tensor(codes, dtype=torch.float32, device=device)
    indices = np.empty(len(values), dtype=np.int32)
    losses = np.empty(len(values), dtype=np.float32)
    start = 0
    current_chunk = chunk_size
    while start < len(values):
        stop = min(start + current_chunk, len(values))
        try:
            batch = torch.as_tensor(values[start:stop], dtype=torch.float32, device=device)
            difference = batch[:, None, :] - codes_t[None, :, :]
            mse = torch.mean(difference * difference, dim=2)
            best_loss, best_index = torch.min(mse, dim=1)
            indices[start:stop] = best_index.to("cpu").numpy().astype(np.int32)
            losses[start:stop] = best_loss.to("cpu").numpy().astype(np.float32)
            start = stop
        except RuntimeError as exc:
            message = str(exc).lower()
            if ("out of memory" not in message and "oom" not in message) or current_chunk == 1:
                raise
            current_chunk = max(1, current_chunk // 2)
            if hasattr(torch, "xpu") and hasattr(torch.xpu, "empty_cache"):
                torch.xpu.empty_cache()
    if hasattr(torch, "xpu"):
        torch.xpu.synchronize()
    return NearestResult(indices=indices, losses=losses, backend_used="xpu", chunk_size=current_chunk)


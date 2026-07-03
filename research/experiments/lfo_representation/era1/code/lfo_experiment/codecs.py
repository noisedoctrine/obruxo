"""Oracle codecs mapping dense curves to compact model-output candidates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .curve import resample_curve


@dataclass(frozen=True)
class Reconstruction:
    values: np.ndarray
    code_index: int | None = None


class CurveCodec:
    name: str
    dense_dimensions: int

    def reconstruct(self, values: np.ndarray) -> Reconstruction:
        raise NotImplementedError


def _expand_grid(grid: np.ndarray, resolution: int) -> np.ndarray:
    return resample_curve(np.asarray(grid, dtype=np.float64), resolution)


class DirectGridCodec(CurveCodec):
    def __init__(self, size: int):
        self.size = size
        self.name = f"grid_{size}"
        self.dense_dimensions = size

    def reconstruct(self, values: np.ndarray) -> Reconstruction:
        encoded = resample_curve(values, self.size)
        return Reconstruction(_expand_grid(encoded, len(values)))


class StockCodebookCodec(CurveCodec):
    def __init__(self, codebook: np.ndarray):
        if codebook.ndim != 2 or len(codebook) < 1:
            raise ValueError("codebook must be [entries, resolution]")
        self.codebook = codebook
        self.name = f"stock_{len(codebook)}"
        self.dense_dimensions = len(codebook)  # categorical logits

    def nearest(self, values: np.ndarray) -> int:
        mse = np.mean((self.codebook - values[None, :]) ** 2, axis=1)
        return int(np.argmin(mse))

    def reconstruct(self, values: np.ndarray) -> Reconstruction:
        index = self.nearest(values)
        return Reconstruction(self.codebook[index].copy(), index)


class StockResidualCodec(StockCodebookCodec):
    def __init__(self, codebook: np.ndarray, residual_size: int):
        super().__init__(codebook)
        self.residual_size = residual_size
        self.name = f"stock_{len(codebook)}_residual_{residual_size}"
        self.dense_dimensions = len(codebook) + 1 + residual_size

    def reconstruct(self, values: np.ndarray) -> Reconstruction:
        index = self.nearest(values)
        base = self.codebook[index]
        residual = resample_curve(values - base, self.residual_size)
        expanded = _expand_grid(residual, len(values))
        return Reconstruction(np.clip(base + expanded, 0.0, 1.0), index)

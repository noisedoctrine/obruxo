"""Metrics for Era 2 smoke runs and experiment rows."""

from __future__ import annotations

from typing import Any

import numpy as np

from .curve import as_curve_matrix


def rmse_per_curve(reference: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
    reference_matrix = as_curve_matrix(reference)
    reconstructed_matrix = as_curve_matrix(reconstructed)
    if reference_matrix.shape != reconstructed_matrix.shape:
        raise ValueError("reference and reconstructed curves must have the same shape")
    difference = reconstructed_matrix - reference_matrix
    return np.sqrt(np.mean(difference * difference, axis=1)).astype(np.float32)


def max_abs_error_per_curve(reference: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
    reference_matrix = as_curve_matrix(reference)
    reconstructed_matrix = as_curve_matrix(reconstructed)
    if reference_matrix.shape != reconstructed_matrix.shape:
        raise ValueError("reference and reconstructed curves must have the same shape")
    return np.max(np.abs(reconstructed_matrix - reference_matrix), axis=1).astype(np.float32)


def reconstruction_summary(reference: np.ndarray, reconstructed: np.ndarray) -> dict[str, float]:
    rmse = rmse_per_curve(reference, reconstructed)
    max_abs = max_abs_error_per_curve(reference, reconstructed)
    return {
        "row_count": float(len(rmse)),
        "median_rmse": float(np.median(rmse)),
        "p90_rmse": float(np.quantile(rmse, 0.90)),
        "p95_rmse": float(np.quantile(rmse, 0.95)),
        "p99_rmse": float(np.quantile(rmse, 0.99)),
        "max_rmse": float(np.max(rmse)),
        "strict_perfect_lfo_rate": float(np.mean((rmse <= 1e-6) & (max_abs <= 1e-5))),
        "node_max_error_median": float(np.median(max_abs)),
        "node_max_error_p95": float(np.quantile(max_abs, 0.95)),
    }


def flat_atom_usage(
    encoding_arrays: dict[str, np.ndarray],
    *,
    residual_layer_count: int,
    widths_by_residual_layer: list[int] | None = None,
) -> dict[str, Any]:
    usage: dict[str, Any] = {}
    for residual_layer in range(1, residual_layer_count + 1):
        key = f"residual_layer_{residual_layer}_index"
        values = np.asarray(encoding_arrays[key], dtype=np.int32)
        minlength = 0
        if widths_by_residual_layer is not None:
            minlength = int(widths_by_residual_layer[residual_layer - 1])
        counts = np.bincount(values, minlength=minlength) if len(values) else np.zeros(minlength, dtype=np.int64)
        total = float(np.sum(counts))
        probabilities = counts.astype(np.float64) / total if total else counts.astype(np.float64)
        nonzero = probabilities[probabilities > 0.0]
        entropy = -float(np.sum(nonzero * np.log2(nonzero))) if len(nonzero) else 0.0
        usage[f"residual_layer_{residual_layer}_atom_usage_entropy"] = entropy
        usage[f"residual_layer_{residual_layer}_dead_atom_rate"] = float(np.mean(counts == 0)) if len(counts) else 0.0
        usage[f"residual_layer_{residual_layer}_dominant_atom_share"] = float(np.max(probabilities)) if len(probabilities) else 0.0
    return usage

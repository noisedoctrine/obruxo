"""Oracle curve-reconstruction benchmark and Pareto report."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .catalog import load_codebook
from .codecs import DirectGridCodec, StockCodebookCodec, StockResidualCodec
from .curve import sample_shape
from .model import LfoShape


def _bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _metrics(reference: np.ndarray, reconstructed: np.ndarray) -> dict[str, float]:
    difference = reconstructed - reference
    reference_delta = np.diff(np.concatenate([reference, reference[:1]]))
    reconstructed_delta = np.diff(
        np.concatenate([reconstructed, reconstructed[:1]])
    )
    return {
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "max_abs_error": float(np.max(np.abs(difference))),
        "derivative_rmse": float(
            np.sqrt(np.mean((reconstructed_delta - reference_delta) ** 2))
        ),
    }


def run_oracle_benchmark(
    catalog_path: Path,
    codebook_path: Path,
    output_dir: Path,
    *,
    resolution: int = 1024,
    max_shapes: int | None = None,
    active_only: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    catalog = pd.read_csv(catalog_path, keep_default_na=False)
    if active_only:
        catalog = catalog[catalog["materially_active"].astype(str).str.lower() == "true"]
    if max_shapes is not None and len(catalog) > max_shapes:
        catalog = catalog.sample(max_shapes, random_state=0)

    codebook_entries = load_codebook(codebook_path)
    codebook = np.stack(
        [sample_shape(shape, resolution) for _, shape in codebook_entries]
    )
    codecs = [StockCodebookCodec(codebook)]
    codecs.extend(DirectGridCodec(size) for size in (8, 16, 32, 64))
    codecs.extend(StockResidualCodec(codebook, size) for size in (8, 16, 32))

    result_rows: list[dict[str, object]] = []
    for ordinal, row in enumerate(catalog.itertuples(index=False), 1):
        shape = LfoShape.from_serialized(
            row.points,
            row.powers,
            name=row.shape_name,
            smooth=_bool(row.smooth),
        )
        reference = sample_shape(shape, resolution)
        for codec in codecs:
            reconstructed = codec.reconstruct(reference)
            result_rows.append(
                {
                    "preset_id": row.preset_id,
                    "lfo_index": row.lfo_index,
                    "shape_signature": row.shape_signature,
                    "shape_name": row.shape_name,
                    "num_points": row.num_points,
                    "codec": codec.name,
                    "dense_dimensions": codec.dense_dimensions,
                    "code_index": reconstructed.code_index,
                    **_metrics(reference, reconstructed.values),
                }
            )
        if ordinal % 1000 == 0:
            print(f"Benchmarked {ordinal:,}/{len(catalog):,} shapes", flush=True)

    results = pd.DataFrame(result_rows)
    summary = (
        results.groupby(["codec", "dense_dimensions"], as_index=False)
        .agg(
            shapes=("shape_signature", "size"),
            rmse_median=("rmse", "median"),
            rmse_mean=("rmse", "mean"),
            rmse_p95=("rmse", lambda values: values.quantile(0.95)),
            max_error_p95=("max_abs_error", lambda values: values.quantile(0.95)),
            derivative_rmse_median=("derivative_rmse", "median"),
        )
        .sort_values("dense_dimensions")
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "oracle_results.csv", index=False)
    summary.to_csv(output_dir / "oracle_summary.csv", index=False)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(summary["dense_dimensions"], summary["rmse_median"])
    for row in summary.itertuples(index=False):
        axis.annotate(
            row.codec,
            (row.dense_dimensions, row.rmse_median),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xlabel("Dense output dimensions")
    axis.set_ylabel("Median sampled-curve RMSE")
    axis.set_title("LFO oracle reconstruction Pareto")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "oracle_pareto.png", dpi=160)
    plt.close(figure)

    print(summary.to_string(index=False))
    return results, summary

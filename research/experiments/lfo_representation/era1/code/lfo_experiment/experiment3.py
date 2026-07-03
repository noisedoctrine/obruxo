"""Experiment 3: frequency-first residual peeling with compact local controls."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .residual3 import (
    FlexibleEncoding,
    envelope_basis,
    flexible_beam_encode,
    flexible_decode,
    train_frequency_first_chain,
    train_frequency_topology_chain,
)
from .stacked import (
    TOPOLOGY_NAMES,
    CurveDataset,
    StackedChain,
    load_curve_dataset,
    load_stock_curves,
    metric_arrays,
    validation_conditions,
)


SCALING_MODES = ("none", "scalar", "linear", "step2")


def _name(chain: StackedChain, depth: int, mode: str, shifts: int) -> str:
    shift = f"_shift{shifts}" if shifts > 1 else ""
    return f"{chain.strategy}_b{chain.base_width}_k{chain.residual_width}_l{depth}_{mode}{shift}"


def evaluate(
    dataset: CurveDataset,
    chain: StackedChain,
    *,
    depth: int,
    mode: str,
    shifts: int = 1,
    beam_width: int = 32,
    fallback: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    conditions = validation_conditions(dataset, chain)
    target = dataset.validation_curves
    encoding = flexible_beam_encode(
        target,
        chain,
        conditions,
        depth=depth,
        mode=mode,
        shifts=shifts,
        beam_width=beam_width,
    )
    reconstructed = flexible_decode(chain, encoding, conditions, mode=mode, shifts=shifts)
    metrics = metric_arrays(target, reconstructed)
    validation = dataset.frame.iloc[dataset.validation_indices].reset_index(drop=True)
    parameter_count = len(envelope_basis(mode, target.shape[1]))
    condition_dimensions = len(TOPOLOGY_NAMES) if chain.condition_kind == "topology" else 0
    shift_dimensions = depth * shifts if shifts > 1 else 0
    dense_dimensions = (
        chain.base_width + depth * chain.residual_width + condition_dimensions
        + depth * parameter_count + shift_dimensions
    )
    effective_bits = math.log2(chain.base_width) + depth * math.log2(chain.residual_width)
    if chain.condition_kind == "topology":
        effective_bits += math.log2(3)
    if shifts > 1:
        effective_bits += depth * math.log2(shifts)
    name = _name(chain, depth, mode, shifts)
    frame = pd.DataFrame(
        {
            "preset_id": validation["preset_id"],
            "author_id": validation["author_id"],
            "shape_signature": validation["shape_signature"],
            "topology": [TOPOLOGY_NAMES[value] for value in dataset.topology[dataset.validation_indices]],
            "configuration": name,
            "strategy": chain.strategy,
            "depth": depth,
            "scaling": mode,
            "shifts": shifts,
            "scalars_per_layer": parameter_count,
            "dense_dimensions": dense_dimensions,
            "effective_bits": effective_bits,
            "stored_floats": chain.bases.size + chain.residuals[:depth].size,
            "base_index": encoding.base_indices,
            **{f"residual_{i + 1}": values for i, values in enumerate(encoding.residual_indices)},
            **{f"shift_{i + 1}": values for i, values in enumerate(encoding.shifts)},
            **metrics,
        }
    )
    for layer, coefficients in enumerate(encoding.coefficients):
        for parameter in range(coefficients.shape[1]):
            frame[f"scale_{layer + 1}_{parameter + 1}"] = coefficients[:, parameter]

    if fallback is not None:
        if len(fallback) != len(frame):
            raise ValueError("fallback result length does not match evaluation")
        use_fallback = fallback.rmse.to_numpy() < frame.rmse.to_numpy()
        metric_columns = ["rmse", "max_abs_error", "derivative_rmse", "base_index"]
        path_columns = [
            column for column in fallback.columns
            if column.startswith(("residual_", "shift_", "scale_")) and column in frame
        ]
        for column in [*metric_columns, *path_columns]:
            frame.loc[use_fallback, column] = fallback.loc[use_fallback, column].to_numpy()
        fallback_depth = int(fallback.depth.iloc[0])
        for layer in range(fallback_depth + 1, depth + 1):
            frame.loc[use_fallback, f"residual_{layer}"] = 0
            frame.loc[use_fallback, f"shift_{layer}"] = 0
            for parameter in range(parameter_count):
                frame.loc[use_fallback, f"scale_{layer}_{parameter + 1}"] = 0.0
        # Rebuild the arrays used by the utilization report from the accepted paths.
        for layer in range(depth):
            encoding.residual_indices[layer] = frame[f"residual_{layer + 1}"].to_numpy(dtype=np.int16)
            encoding.shifts[layer] = frame[f"shift_{layer + 1}"].to_numpy(dtype=np.int8)

    usage: list[dict[str, object]] = []
    for layer, indices in enumerate(encoding.residual_indices):
        for condition, label in enumerate(chain.condition_labels):
            members = conditions == condition
            counts = np.bincount(indices[members], minlength=chain.residual_width)
            total = max(1, int(np.sum(members)))
            for code, uses in enumerate(counts):
                usage.append(
                    {
                        "configuration": name,
                        "layer": layer + 1,
                        "condition": label,
                        "code": code,
                        "uses": int(uses),
                        "share": float(uses / total),
                        "is_noop": code == 0,
                    }
                )
    return frame, pd.DataFrame(usage)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "configuration", "strategy", "depth", "scaling", "shifts",
        "scalars_per_layer", "dense_dimensions", "effective_bits", "stored_floats",
    ]
    return (
        results.groupby(keys, as_index=False)
        .agg(
            shapes=("shape_signature", "size"),
            rmse_median=("rmse", "median"),
            rmse_mean=("rmse", "mean"),
            rmse_p90=("rmse", lambda x: x.quantile(0.90)),
            rmse_p95=("rmse", lambda x: x.quantile(0.95)),
            rmse_p99=("rmse", lambda x: x.quantile(0.99)),
            max_error_p95=("max_abs_error", lambda x: x.quantile(0.95)),
        )
        .sort_values(["dense_dimensions", "rmse_p95", "rmse_median"])
        .reset_index(drop=True)
    )


def _table(frame: pd.DataFrame, columns: list[str]) -> str:
    selected = frame[columns].copy()
    for column in selected.select_dtypes(include=["float"]).columns:
        selected[column] = selected[column].map(lambda value: f"{value:.6f}")
    return "\n".join(
        [
            "| " + " | ".join(columns) + " |",
            "|" + "|".join("---" for _ in columns) + "|",
            *["| " + " | ".join(map(str, row)) + " |" for row in selected.itertuples(index=False, name=None)],
        ]
    )


def write_findings(
    summary: pd.DataFrame,
    usage: pd.DataFrame,
    output: Path,
    experiment2_summary: Path | None,
) -> None:
    best_common = summary.loc[summary.rmse_median.idxmin()]
    best_tail = summary.loc[summary.rmse_p95.idxmin()]
    noops = (
        usage[usage.is_noop]
        .groupby(["configuration", "layer"], as_index=False)
        .agg(noop_share=("share", "mean"))
    )
    full_depth = summary[summary.depth == 4]
    comparison = "Experiment 2 summary unavailable."
    if experiment2_summary and experiment2_summary.exists():
        old = pd.read_csv(experiment2_summary)
        old = old.loc[old.groupby("strategy").rmse_p95.idxmin()]
        comparison = _table(old, ["configuration", "dense_dimensions", "rmse_median", "rmse_p95"])
    content = f"""# Experiment 3 Findings: Frequency-First Residual Peeling

## What changed from Experiment 2?

Experiment 2 capped repeated shape signatures during fitting. Experiment 3 removes that cap. Every training occurrence contributes to the selection objective, so early bases and residual codes preferentially absorb configurations that improve the largest amount of total corpus error. Each following layer is fit only to the residual left by the frozen prefix, and code 0 remains an exact no-op.

The base budget is fixed at 32 choices: the 15 provisional stock geometries plus 17 observed training medoids. Every tested stack uses 16 residual choices per layer and up to four layers.

## Did common cases peel off into the no-op branch?

The table reports held-out no-op usage by layer. Increasing no-op share would support the intended peeling behavior; flat or falling use means later dictionaries continue revisiting the same population.

{_table(noops, ['configuration', 'layer', 'noop_share'])}

## How much did compact scaling cost?

- `none`: no scalar outputs.
- `scalar`: one clipped gain per residual layer.
- `linear`: offset plus slope, two numbers per layer.
- `step2`: independent first-half and second-half gains, two numbers per layer.
- `shift4`: a higher-cost convolutional diagnostic with four circular placements per layer. It adds four placement logits per layer and is not treated as a free improvement.

{_table(full_depth, ['configuration', 'dense_dimensions', 'rmse_median', 'rmse_p95', 'rmse_p99'])}

## Did topology conditioning remain useful?

Topology-conditioned dictionaries are retained as a primary branch, not discarded. Their extra cost is three topology logits and additional stored dictionaries. Base conditioning is not retrained as the main approach here; Experiment 2's result remains a complementary high-storage upper bound.

## Best observed results

Best median: `{best_common.configuration}` at {best_common.rmse_median:.6f}.  
Best 95th percentile: `{best_tail.configuration}` at {best_tail.rmse_p95:.6f}.

## Experiment 2 reference points

{comparison}

## What this still does not answer

This remains an oracle geometry experiment. The next decision should consider sparse Vital refitting, rendered modulation error, and whether the audio encoder can predict the selected categories and small number of gains.
"""
    output.write_text(content, encoding="utf-8")


def run_experiment3(
    catalog_path: Path,
    codebook_path: Path,
    output_dir: Path,
    *,
    resolution: int = 1024,
    beam_width: int = 32,
    quick: bool = False,
    experiment2_summary: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_curve_dataset(catalog_path, resolution=resolution)
    stock_names, stock = load_stock_curves(codebook_path, resolution)
    shared, shared_utility = train_frequency_first_chain(
        dataset, stock, residual_width=4 if quick else 16, max_depth=2 if quick else 4
    )
    topology, topology_utility = train_frequency_topology_chain(dataset, shared)
    shared.save(output_dir / "codebooks" / f"frequency_first_shared_b32_k{shared.residual_width}")
    topology.save(output_dir / "codebooks" / f"frequency_first_topology_b32_k{shared.residual_width}")
    np.savez_compressed(
        output_dir / "selection_utilities.npz",
        shared=shared_utility,
        topology=topology_utility,
    )
    split = {
        "train_instances": len(dataset.train_indices),
        "validation_instances": len(dataset.validation_indices),
        "train_authors": int(dataset.frame.iloc[dataset.train_indices].author_id.nunique()),
        "validation_authors": int(dataset.frame.iloc[dataset.validation_indices].author_id.nunique()),
        "stock_names": stock_names,
        "frequency_cap": None,
        "selection_objective": "maximum summed training MSE reduction",
    }
    (output_dir / "manifest.json").write_text(json.dumps(split, indent=2), encoding="utf-8")

    depths = (1, 2) if quick else (1, 2, 3, 4)
    modes = ("none", "scalar") if quick else SCALING_MODES
    results: list[pd.DataFrame] = []
    usages: list[pd.DataFrame] = []
    for chain in (shared, topology):
        full_scalar: pd.DataFrame | None = None
        for mode in modes:
            previous: pd.DataFrame | None = None
            for depth in depths:
                print(f"Evaluating {chain.strategy}, {mode}, depth {depth}", flush=True)
                frame, use = evaluate(
                    dataset, chain, depth=depth, mode=mode,
                    beam_width=min(8, beam_width) if quick else beam_width,
                    fallback=previous,
                )
                results.append(frame)
                usages.append(use)
                previous = frame
                if mode == "scalar" and depth == max(depths):
                    full_scalar = frame
        if not quick:
            print(f"Evaluating {chain.strategy}, scalar + four circular shifts", flush=True)
            frame, use = evaluate(
                dataset, chain, depth=4, mode="scalar", shifts=4,
                beam_width=beam_width, fallback=full_scalar,
            )
            results.append(frame)
            usages.append(use)

    result = pd.concat(results, ignore_index=True)
    usage = pd.concat(usages, ignore_index=True)
    summary = summarize(result)
    topology_summary = (
        result.groupby(["configuration", "topology"], as_index=False)
        .agg(rmse_median=("rmse", "median"), rmse_p95=("rmse", lambda x: x.quantile(.95)))
    )
    result.to_csv(output_dir / "results.csv", index=False)
    usage.to_csv(output_dir / "code_usage.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    topology_summary.to_csv(output_dir / "topology_summary.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for strategy, group in summary.groupby("strategy"):
        axes[0].scatter(group.dense_dimensions, group.rmse_median, label=strategy)
        axes[1].scatter(group.dense_dimensions, group.rmse_p95, label=strategy)
    for axis, title in zip(axes, ("Median RMSE", "95th-percentile RMSE")):
        axis.set(title=title, xlabel="Dense output dimensions", ylabel="Curve RMSE")
        axis.grid(alpha=.25)
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "pareto.png", dpi=160)
    plt.close(figure)
    write_findings(summary, usage, output_dir / "EXPERIMENT_3_FINDINGS.md", experiment2_summary)
    print(summary.to_string(index=False))
    return result, summary

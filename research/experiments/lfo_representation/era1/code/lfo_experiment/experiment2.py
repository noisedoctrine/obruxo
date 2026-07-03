"""Orchestration and reporting for stacked residual codebook Experiment 2."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .stacked import (
    SEED,
    TOPOLOGY_NAMES,
    CurveDataset,
    StackedChain,
    beam_encode,
    decode_encoding,
    load_curve_dataset,
    load_stock_curves,
    metric_arrays,
    train_conditional_chain,
    train_shared_chain,
    validation_conditions,
)


BASE_WIDTHS = (15, 16, 24, 32)
RESIDUAL_WIDTHS = (4, 8, 16)
DEPTHS = (1, 2, 3, 4)


def _configuration_name(chain: StackedChain, depth: int, gains: bool) -> str:
    suffix = "_gain" if gains else ""
    return f"{chain.strategy}_b{chain.base_width}_k{chain.residual_width}_l{depth}{suffix}"


def evaluate_chain(
    dataset: CurveDataset,
    chain: StackedChain,
    *,
    depth: int,
    use_gains: bool,
    beam_width: int = 32,
    fallback_results: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = dataset.validation_curves
    conditions = validation_conditions(dataset, chain)
    encoding = beam_encode(
        targets,
        chain,
        conditions,
        depth=depth,
        beam_width=beam_width,
        use_gains=use_gains,
    )
    reconstructed = decode_encoding(chain, encoding, conditions)
    if beam_width > 1:
        greedy = beam_encode(
            targets,
            chain,
            conditions,
            depth=depth,
            beam_width=1,
            use_gains=use_gains,
        )
        greedy_reconstructed = decode_encoding(chain, greedy, conditions)
        beam_rmse = np.sqrt(np.mean((reconstructed - targets) ** 2, axis=1))
        greedy_rmse = np.sqrt(np.mean((greedy_reconstructed - targets) ** 2, axis=1))
        use_greedy = greedy_rmse < beam_rmse
        if np.any(use_greedy):
            reconstructed[use_greedy] = greedy_reconstructed[use_greedy]
            encoding.base_indices[use_greedy] = greedy.base_indices[use_greedy]
            for layer in range(depth):
                encoding.residual_indices[layer][use_greedy] = greedy.residual_indices[layer][use_greedy]
                encoding.gains[layer][use_greedy] = greedy.gains[layer][use_greedy]
    metrics = metric_arrays(targets, reconstructed)
    validation = dataset.frame.iloc[dataset.validation_indices].reset_index(drop=True)
    topology = dataset.topology[dataset.validation_indices]
    dense_dimensions = (
        chain.base_width
        + depth * chain.residual_width
        + (len(TOPOLOGY_NAMES) if chain.condition_kind == "topology" else 0)
        + (depth if use_gains else 0)
    )
    effective_bits = math.log2(chain.base_width) + depth * math.log2(chain.residual_width)
    if chain.condition_kind == "topology":
        effective_bits += math.log2(len(TOPOLOGY_NAMES))
    stored_floats = chain.bases.size + chain.residuals[:depth].size
    name = _configuration_name(chain, depth, use_gains)

    results = pd.DataFrame(
        {
            "preset_id": validation["preset_id"],
            "author_id": validation["author_id"],
            "shape_signature": validation["shape_signature"],
            "shape_name": validation["shape_name"],
            "topology": [TOPOLOGY_NAMES[item] for item in topology],
            "configuration": name,
            "strategy": chain.strategy,
            "base_width": chain.base_width,
            "residual_width": chain.residual_width,
            "depth": depth,
            "use_gains": use_gains,
            "dense_dimensions": dense_dimensions,
            "effective_bits": effective_bits,
            "stored_floats": stored_floats,
            "base_index": encoding.base_indices,
            **{f"residual_{i + 1}": values for i, values in enumerate(encoding.residual_indices)},
            **{f"gain_{i + 1}": values for i, values in enumerate(encoding.gains)},
            **metrics,
        }
    )
    if fallback_results is not None:
        if len(fallback_results) != len(results):
            raise ValueError("fallback result length does not match current evaluation")
        use_fallback = fallback_results["rmse"].to_numpy() < results["rmse"].to_numpy()
        copied_columns = [
            "base_index",
            "rmse",
            "max_abs_error",
            "derivative_rmse",
            *[f"residual_{i + 1}" for i in range(depth - 1)],
            *[f"gain_{i + 1}" for i in range(depth - 1)],
        ]
        for column in copied_columns:
            if column in fallback_results:
                results.loc[use_fallback, column] = fallback_results.loc[
                    use_fallback, column
                ].to_numpy()
        results.loc[use_fallback, f"residual_{depth}"] = 0
        results.loc[use_fallback, f"gain_{depth}"] = 1.0
        encoding.base_indices[use_fallback] = results.loc[
            use_fallback, "base_index"
        ].to_numpy(dtype=np.int32)
        for layer in range(depth):
            encoding.residual_indices[layer][use_fallback] = results.loc[
                use_fallback, f"residual_{layer + 1}"
            ].to_numpy(dtype=np.int16)
            encoding.gains[layer][use_fallback] = results.loc[
                use_fallback, f"gain_{layer + 1}"
            ].to_numpy(dtype=np.float32)

    usage_rows: list[dict[str, object]] = []
    for code, count in enumerate(np.bincount(encoding.base_indices, minlength=chain.base_width)):
        usage_rows.append(
            {"configuration": name, "layer": "base", "condition": "all", "code": code, "uses": int(count)}
        )
    for layer, indices in enumerate(encoding.residual_indices):
        for condition in range(chain.conditions):
            members = conditions == condition
            counts = np.bincount(indices[members], minlength=chain.residual_width)
            for code, count in enumerate(counts):
                usage_rows.append(
                    {
                        "configuration": name,
                        "layer": f"residual_{layer + 1}",
                        "condition": chain.condition_labels[condition],
                        "code": code,
                        "uses": int(count),
                    }
                )
    return results, pd.DataFrame(usage_rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "configuration",
        "strategy",
        "base_width",
        "residual_width",
        "depth",
        "use_gains",
        "dense_dimensions",
        "effective_bits",
        "stored_floats",
    ]
    return (
        results.groupby(keys, as_index=False)
        .agg(
            shapes=("shape_signature", "size"),
            rmse_median=("rmse", "median"),
            rmse_mean=("rmse", "mean"),
            rmse_p95=("rmse", lambda values: values.quantile(0.95)),
            max_error_p95=("max_abs_error", lambda values: values.quantile(0.95)),
            derivative_rmse_median=("derivative_rmse", "median"),
        )
        .sort_values(["dense_dimensions", "rmse_p95", "rmse_median"])
        .reset_index(drop=True)
    )


def topology_summary(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby(["configuration", "topology"], as_index=False)
        .agg(
            shapes=("shape_signature", "size"),
            rmse_median=("rmse", "median"),
            rmse_p95=("rmse", lambda values: values.quantile(0.95)),
            max_error_p95=("max_abs_error", lambda values: values.quantile(0.95)),
        )
    )


def pareto_finalists(summary: pd.DataFrame, limit: int = 3) -> pd.DataFrame:
    shared = summary[(summary["strategy"] == "shared") & (~summary["use_gains"].astype(bool))].copy()
    nondominated: list[int] = []
    for index, row in shared.iterrows():
        dominated = (
            (shared["dense_dimensions"] <= row["dense_dimensions"])
            & (shared["rmse_median"] <= row["rmse_median"])
            & (shared["rmse_p95"] <= row["rmse_p95"])
            & (
                (shared["dense_dimensions"] < row["dense_dimensions"])
                | (shared["rmse_median"] < row["rmse_median"])
                | (shared["rmse_p95"] < row["rmse_p95"])
            )
        ).any()
        if not dominated:
            nondominated.append(index)
    frontier = shared.loc[nondominated]
    selected: list[int] = []
    for column in ("dense_dimensions", "rmse_median", "rmse_p95"):
        if len(frontier):
            candidate = int(frontier[column].idxmin())
            if candidate not in selected:
                selected.append(candidate)
    if len(selected) < limit:
        for index in frontier.sort_values(["dense_dimensions", "rmse_p95"]).index:
            if int(index) not in selected:
                selected.append(int(index))
            if len(selected) == limit:
                break
    return shared.loc[selected[:limit]].reset_index(drop=True)


def _plots(summary: pd.DataFrame, output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for strategy, group in summary.groupby("strategy"):
        axes[0].scatter(group["dense_dimensions"], group["rmse_median"], label=strategy, alpha=0.8)
        axes[1].scatter(group["dense_dimensions"], group["rmse_p95"], label=strategy, alpha=0.8)
    axes[0].set_title("Median reconstruction error")
    axes[1].set_title("95th-percentile reconstruction error")
    for axis in axes:
        axis.set_xlabel("Dense output dimensions")
        axis.set_ylabel("Curve RMSE")
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "stacked_pareto.png", dpi=160)
    plt.close(figure)


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    selected = frame[columns].copy()
    for column in selected.select_dtypes(include=["float"]).columns:
        selected[column] = selected[column].map(lambda value: f"{value:.6f}")
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in selected.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


def write_findings(
    dataset: CurveDataset,
    summary: pd.DataFrame,
    topology: pd.DataFrame,
    finalists: pd.DataFrame,
    output_path: Path,
    baseline_path: Path | None,
) -> None:
    best_median = summary.loc[summary["rmse_median"].idxmin()]
    best_tail = summary.loc[summary["rmse_p95"].idxmin()]
    baseline_text = "Experiment 1 baseline summary was unavailable."
    if baseline_path and baseline_path.exists():
        baseline = pd.read_csv(baseline_path).sort_values("dense_dimensions")
        baseline_text = _markdown_table(
            baseline,
            ["codec", "dense_dimensions", "rmse_median", "rmse_p95"],
        )
    content = f"""# Experiment 2 Findings: Stacked Residual Codebooks

## What did this experiment ask?

Can a small sequence of categorical code selections reconstruct LFO curves more efficiently than the direct-grid and continuous-residual representations from Experiment 1?

## How were the codebooks constructed?

The 15 provisional stock curves were fixed. Additional bases and every non-zero residual codeword were snapped to observed training shapes or observed prefix residuals. Authors were deterministically split before fitting: {len(dataset.train_indices):,} training instances and {len(dataset.validation_indices):,} held-out instances. Codebooks never used held-out authors.

Each residual layer learned the error remaining after the preceding layers. Held-out encoding used beam search width 32. Conditional and scalar-gain variants were run only for three shared Pareto finalists.

## Which shared configurations survived?

{_markdown_table(finalists, ['configuration', 'dense_dimensions', 'effective_bits', 'rmse_median', 'rmse_p95'])}

## What achieved the best common-case error?

`{best_median['configuration']}` achieved median RMSE {best_median['rmse_median']:.6f} using {int(best_median['dense_dimensions'])} dense outputs.

## What achieved the best tail error?

`{best_tail['configuration']}` achieved 95th-percentile RMSE {best_tail['rmse_p95']:.6f} using {int(best_tail['dense_dimensions'])} dense outputs.

## Full Experiment 2 summary

{_markdown_table(summary, ['configuration', 'dense_dimensions', 'effective_bits', 'stored_floats', 'rmse_median', 'rmse_p95'])}

## Experiment 1 comparison

{baseline_text}

## What remains unanswered?

These are geometry-only representation ceilings. They do not yet show sparse point/power refitting quality, rendered audibility, or inferability from reference audio. The stock entries remain provisional modal-name geometries.
"""
    output_path.write_text(content, encoding="utf-8")


def run_experiment2(
    catalog_path: Path,
    codebook_path: Path,
    output_dir: Path,
    *,
    baseline_path: Path | None = None,
    resolution: int = 1024,
    beam_width: int = 32,
    base_widths: Iterable[int] = BASE_WIDTHS,
    residual_widths: Iterable[int] = RESIDUAL_WIDTHS,
    depths: Iterable[int] = DEPTHS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_curve_dataset(catalog_path, resolution=resolution)
    stock_names, stock = load_stock_curves(codebook_path, resolution)
    if len(stock) != 15:
        raise ValueError(f"Experiment 2 requires 15 stock entries, found {len(stock)}")

    split = {
        "seed": SEED,
        "train_instances": int(len(dataset.train_indices)),
        "validation_instances": int(len(dataset.validation_indices)),
        "train_authors": int(dataset.frame.iloc[dataset.train_indices]["author_id"].nunique()),
        "validation_authors": int(dataset.frame.iloc[dataset.validation_indices]["author_id"].nunique()),
        "stock_names": stock_names,
        "stock_provenance": "provisional_modal_name",
    }
    (output_dir / "split_summary.json").write_text(json.dumps(split, indent=2), encoding="utf-8")

    all_results: list[pd.DataFrame] = []
    all_usage: list[pd.DataFrame] = []
    chains: dict[tuple[int, int], StackedChain] = {}
    for base_width in base_widths:
        for residual_width in residual_widths:
            print(f"Training shared chain B={base_width}, K={residual_width}", flush=True)
            chain = train_shared_chain(
                dataset,
                stock,
                base_width=base_width,
                residual_width=residual_width,
                max_depth=max(depths),
            )
            chains[(base_width, residual_width)] = chain
            chain.save(output_dir / "codebooks" / f"shared_b{base_width}_k{residual_width}")
            previous_results: pd.DataFrame | None = None
            for depth in depths:
                print(f"Evaluating shared B={base_width}, K={residual_width}, L={depth}", flush=True)
                results, usage = evaluate_chain(
                    dataset,
                    chain,
                    depth=depth,
                    use_gains=False,
                    beam_width=beam_width,
                    fallback_results=previous_results,
                )
                all_results.append(results)
                all_usage.append(usage)
                previous_results = results

    shared_results = pd.concat(all_results, ignore_index=True)
    shared_summary = summarize(shared_results)
    finalists = pareto_finalists(shared_summary)
    finalists.to_csv(output_dir / "shared_finalists.csv", index=False)

    for finalist in finalists.itertuples(index=False):
        shared = chains[(int(finalist.base_width), int(finalist.residual_width))]
        depth = int(finalist.depth)
        variants = [
            train_conditional_chain(dataset, shared, kind="topology"),
            train_conditional_chain(dataset, shared, kind="base"),
        ]
        for variant in variants:
            variant.save(
                output_dir
                / "codebooks"
                / f"{variant.strategy}_b{variant.base_width}_k{variant.residual_width}"
            )
        for variant in [shared, *variants]:
            for gains in (False, True):
                if variant is shared and not gains:
                    continue  # already included above
                print(
                    f"Evaluating finalist {variant.strategy} B={variant.base_width}, "
                    f"K={variant.residual_width}, L={depth}, gains={gains}",
                    flush=True,
                )
                results, usage = evaluate_chain(
                    dataset, variant, depth=depth, use_gains=gains, beam_width=beam_width
                )
                all_results.append(results)
                all_usage.append(usage)

    results = pd.concat(all_results, ignore_index=True)
    usage = pd.concat(all_usage, ignore_index=True)
    summary = summarize(results)
    topology = topology_summary(results)
    results.to_csv(output_dir / "stacked_results.csv", index=False)
    summary.to_csv(output_dir / "stacked_summary.csv", index=False)
    topology.to_csv(output_dir / "stacked_topology_summary.csv", index=False)
    usage.to_csv(output_dir / "stacked_code_usage.csv", index=False)
    _plots(summary, output_dir)
    write_findings(
        dataset,
        summary,
        topology,
        finalists,
        output_dir / "EXPERIMENT_2_FINDINGS.md",
        baseline_path,
    )
    print(summary.to_string(index=False))
    return results, summary

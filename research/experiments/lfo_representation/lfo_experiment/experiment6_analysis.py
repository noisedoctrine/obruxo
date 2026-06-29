"""Richer analytics and visualizations for Experiment 6 outputs."""

from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _safe_name(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def _markdown_table(frame: pd.DataFrame, columns: list[str], limit: int = 15) -> str:
    if frame.empty:
        return "_No rows._"
    table = frame[columns].head(limit).copy()
    for column in table.columns:
        if pd.api.types.is_float_dtype(table[column]):
            table[column] = table[column].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in table.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(map(str, row)) + " |")
    return "\n".join(lines)


def _captioned_image(path: str, caption: str) -> str:
    return f"![{caption}]({path})\n\n_{caption}_"


def _phase_family(value: str) -> str:
    if value == "direct_grid":
        return "grid"
    if "additive" in value:
        return "additive"
    if "switch" in value:
        return "switch"
    if "partition" in value:
        return "partition"
    if "topology" in value:
        return "topology"
    if "shared" in value:
        return "shared"
    return value


def _plot_table_labels(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["strategy"] = out["candidate"].map(_phase_family)
    out["short_label"] = out["candidate"].astype(str).str.replace("phase_", "", regex=False)
    out.loc[out["family"] == "direct_grid", "short_label"] = out.loc[out["family"] == "direct_grid", "candidate"]
    return out


def _coverage_column(thresholds: pd.DataFrame, metric: str, threshold: float, subset: str = "all") -> pd.Series:
    rows = thresholds[
        (thresholds["metric"] == metric)
        & (thresholds["subset"] == subset)
        & np.isclose(thresholds["threshold"].astype(float), threshold)
    ]
    return rows.set_index("configuration")["coverage"]


def _rank01(series: pd.Series, *, lower_is_better: bool) -> pd.Series:
    values = series.astype(float)
    if values.nunique(dropna=True) <= 1:
        return pd.Series(np.zeros(len(values)), index=series.index)
    rank = values.rank(method="average", pct=True)
    return rank if lower_is_better else 1.0 - rank


def decision_matrix(summary: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    frame = summary.copy()
    frame["rmse_under_0.02"] = frame["configuration"].map(_coverage_column(thresholds, "rmse", 0.02))
    frame["rmse_under_0.05"] = frame["configuration"].map(_coverage_column(thresholds, "rmse", 0.05))
    frame["all_nodes_under_0.02"] = frame["configuration"].map(_coverage_column(thresholds, "all_nodes", 0.02))
    frame["all_nodes_under_0.05"] = frame["configuration"].map(_coverage_column(thresholds, "all_nodes", 0.05))
    components = pd.DataFrame(index=frame.index)
    components["tail_rmse_rank"] = _rank01(frame["rmse_p95"], lower_is_better=True)
    components["median_rmse_rank"] = _rank01(frame["rmse_median"], lower_is_better=True)
    components["node_tail_rank"] = _rank01(frame["node_max_error_p95"], lower_is_better=True)
    components["rmse_coverage_rank"] = _rank01(frame["rmse_under_0.02"], lower_is_better=False)
    components["node_coverage_rank"] = _rank01(frame["all_nodes_under_0.02"], lower_is_better=False)
    components["dense_rank"] = _rank01(frame["dense_outputs"], lower_is_better=True)
    components["storage_rank"] = _rank01(frame["stored_bytes_float32"], lower_is_better=True)
    # Lower score is better. This is not a winner selector; it is a way to sort
    # the discussion table so obviously interesting candidates rise to the top.
    frame["balanced_discussion_score"] = (
        0.22 * components["tail_rmse_rank"]
        + 0.12 * components["median_rmse_rank"]
        + 0.22 * components["node_tail_rank"]
        + 0.12 * components["rmse_coverage_rank"]
        + 0.12 * components["node_coverage_rank"]
        + 0.12 * components["dense_rank"]
        + 0.08 * components["storage_rank"]
    )
    for column in components:
        frame[column] = components[column]
    return frame.sort_values(["balanced_discussion_score", "dense_outputs", "rmse_p95"]).reset_index(drop=True)


def factor3_grid_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    grids = summary[summary["family"] == "direct_grid"].copy()
    if grids.empty:
        return pd.DataFrame()
    grids["grid_width"] = grids["candidate"].str.extract(r"(\d+)").astype(int)
    rows = []
    for resolution, group in grids.groupby("eval_resolution"):
        by_width = group.set_index("grid_width")
        for triplet, powerish in ((48, 32), (96, 64), (192, 128)):
            if triplet not in by_width.index or powerish not in by_width.index:
                continue
            a = by_width.loc[triplet]
            b = by_width.loc[powerish]
            rows.append(
                {
                    "eval_resolution": resolution,
                    "factor3_grid": f"Grid{triplet}",
                    "comparison_grid": f"Grid{powerish}",
                    "extra_outputs": int(a.dense_outputs - b.dense_outputs),
                    "rmse_p95_delta": float(a.rmse_p95 - b.rmse_p95),
                    "rmse_p95_relative": float((a.rmse_p95 - b.rmse_p95) / max(float(b.rmse_p95), 1e-12)),
                    "node_p95_delta": float(a.node_max_error_p95 - b.node_max_error_p95),
                    "node_p95_relative": float((a.node_max_error_p95 - b.node_max_error_p95) / max(float(b.node_max_error_p95), 1e-12)),
                    "rmse_under_0.02_delta": float(a.get("rmse_under_0.02", np.nan) - b.get("rmse_under_0.02", np.nan)),
                    "all_nodes_under_0.02_delta": float(a.get("all_nodes_under_0.02", np.nan) - b.get("all_nodes_under_0.02", np.nan)),
                }
            )
    return pd.DataFrame(rows)


def topology_gap(topology: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for configuration, group in topology.groupby("configuration"):
        if len(group) < 2:
            continue
        p95 = group.set_index("topology")["rmse_p95"]
        node = group.set_index("topology")["node_max_error_p95"]
        rows.append(
            {
                "configuration": configuration,
                "topology_rmse_p95_gap": float(p95.max() - p95.min()),
                "worst_topology_rmse": str(p95.idxmax()),
                "best_topology_rmse": str(p95.idxmin()),
                "topology_node_p95_gap": float(node.max() - node.min()),
                "worst_topology_node": str(node.idxmax()),
                "best_topology_node": str(node.idxmin()),
            }
        )
    return pd.DataFrame(rows).sort_values(["topology_rmse_p95_gap", "topology_node_p95_gap"], ascending=False)


def subset_breakdown(per_shape: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for configuration, group in per_shape.groupby("configuration"):
        for label, members in (
            ("all", group),
            ("stock_hint", group[group["stock_name_hint"].astype(bool)]),
            ("custom_ish", group[~group["stock_name_hint"].astype(bool)]),
            ("gate_pulse_heavy", group[
                (group["topology"] == "discontinuous")
                | group["shape_name"].astype(str).str.lower().str.contains("gate|pulse|square|stair|trance|shuffle", regex=True)
            ]),
        ):
            if members.empty:
                continue
            rows.append(
                {
                    "configuration": configuration,
                    "subset": label,
                    "shapes": int(len(members)),
                    "rmse_median": float(members.rmse.median()),
                    "rmse_p95": float(members.rmse.quantile(0.95)),
                    "node_max_error_p95": float(members.node_max_error.quantile(0.95)),
                    "rmse_under_0.02": float(np.mean(members.rmse <= 0.02)),
                    "all_nodes_under_0.02": float(np.mean(members.node_max_error <= 0.02)),
                }
            )
    return pd.DataFrame(rows)


def worst_cases(per_shape: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    rows = []
    for configuration, group in per_shape.groupby("configuration"):
        for metric in ("rmse", "node_max_error"):
            selected = group.sort_values(metric, ascending=False).head(top_n).copy()
            selected["worst_metric"] = metric
            selected["worst_rank"] = np.arange(1, len(selected) + 1)
            rows.append(
                selected[
                    [
                        "configuration",
                        "worst_metric",
                        "worst_rank",
                        "dataset_index",
                        "preset_id",
                        "shape_name",
                        "topology",
                        "stock_name_hint",
                        "rmse",
                        "max_abs_error",
                        "node_max_error",
                        "node_probe_count",
                        "duplicate_x_probe_count",
                    ]
                ]
            )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def node_rmse_disagreement(per_shape: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for configuration, group in per_shape.groupby("configuration"):
        good_rmse = group["rmse"] <= 0.02
        bad_nodes = group["node_max_error"] > 0.05
        bad_rmse = group["rmse"] > 0.05
        good_nodes = group["node_max_error"] <= 0.02
        rows.append(
            {
                "configuration": configuration,
                "shapes": int(len(group)),
                "good_rmse_bad_nodes_share": float(np.mean(good_rmse & bad_nodes)),
                "bad_rmse_good_nodes_share": float(np.mean(bad_rmse & good_nodes)),
                "rmse_node_corr": float(group[["rmse", "node_max_error"]].corr().iloc[0, 1])
                if len(group) > 1
                else np.nan,
                "median_node_given_rmse_under_0.02": float(group.loc[good_rmse, "node_max_error"].median())
                if good_rmse.any()
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("good_rmse_bad_nodes_share", ascending=False)


def _heatmap(
    pivot: pd.DataFrame,
    output: Path,
    *,
    title: str,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    if pivot.empty:
        return
    fig_width = max(8, 0.18 * len(pivot.index) + 4)
    fig_height = max(4.5, 0.48 * len(pivot.columns) + 2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(pivot.T.to_numpy(dtype=float), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.index, rotation=75, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(pivot.columns)))
    ax.set_yticklabels([str(x) for x in pivot.columns], fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("Configuration")
    fig.colorbar(image, ax=ax, shrink=0.82)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_visuals(
    summary: pd.DataFrame,
    thresholds: pd.DataFrame,
    topology: pd.DataFrame,
    per_shape: pd.DataFrame,
    matrix: pd.DataFrame,
    factor3: pd.DataFrame,
    output_dir: Path,
) -> None:
    analytics_dir = output_dir / "analytics"
    analytics_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    panels = [
        ("rmse_p95", "P95 RMSE"),
        ("node_max_error_p95", "P95 node max error"),
        ("rmse_under_0.02", "Share RMSE <= 0.02"),
        ("all_nodes_under_0.02", "Share all nodes <= 0.02"),
    ]
    for ax, (metric, label) in zip(axes.flat, panels):
        for family, group in matrix.groupby("family"):
            ax.scatter(group["dense_outputs"], group[metric], label=family, alpha=0.78)
        ax.set_xlabel("Dense outputs")
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
        ax.set_title(label)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Experiment 6 quality/complexity dashboard", y=0.995)
    fig.tight_layout()
    fig.savefig(analytics_dir / "quality_complexity_dashboard.png", dpi=180)
    plt.close(fig)

    rmse_cov = thresholds[(thresholds.subset == "all") & (thresholds.metric == "rmse")]
    rmse_pivot = rmse_cov.pivot(index="configuration", columns="threshold", values="coverage")
    order = matrix.set_index("configuration").sort_values(["dense_outputs", "rmse_p95"]).index
    rmse_pivot = rmse_pivot.reindex([x for x in order if x in rmse_pivot.index])
    rmse_pivot.to_csv(analytics_dir / "rmse_threshold_heatmap_table.csv")
    _heatmap(rmse_pivot, analytics_dir / "rmse_threshold_heatmap.png", title="RMSE threshold coverage", vmin=0, vmax=1)

    node_cov = thresholds[(thresholds.subset == "all") & (thresholds.metric == "all_nodes")]
    node_pivot = node_cov.pivot(index="configuration", columns="threshold", values="coverage")
    node_pivot = node_pivot.reindex([x for x in order if x in node_pivot.index])
    node_pivot.to_csv(analytics_dir / "node_threshold_heatmap_table.csv")
    _heatmap(node_pivot, analytics_dir / "node_threshold_heatmap.png", title="All-node threshold coverage", vmin=0, vmax=1)

    topo_pivot = topology.pivot(index="configuration", columns="topology", values="rmse_p95")
    topo_pivot = topo_pivot.reindex([x for x in order if x in topo_pivot.index])
    topo_pivot.to_csv(analytics_dir / "topology_p95_heatmap_table.csv")
    _heatmap(topo_pivot, analytics_dir / "topology_p95_heatmap.png", title="P95 RMSE by topology", cmap="magma")

    fig, ax = plt.subplots(figsize=(10, 6))
    direct = matrix[matrix.family == "direct_grid"].copy()
    if not direct.empty:
        direct["grid_width"] = direct["candidate"].str.extract(r"(\d+)").astype(int)
        for resolution, group in direct.groupby("eval_resolution"):
            group = group.sort_values("grid_width")
            ax.plot(group.grid_width, group.rmse_p95, marker="o", label=f"RMSE P95 eval {resolution}")
            ax.plot(group.grid_width, group.node_max_error_p95, marker="s", linestyle="--", label=f"Node P95 eval {resolution}")
        for width in (48, 96, 192):
            if width in set(direct.grid_width):
                ax.axvline(width, color="gray", alpha=0.18)
                ax.text(width, ax.get_ylim()[1] * 0.96, "x3", rotation=90, va="top", ha="right", fontsize=8)
    ax.set_title("Direct-grid baselines: powers of two vs factor-of-3 grids")
    ax.set_xlabel("Grid width")
    ax.set_ylabel("Error")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(analytics_dir / "factor3_grid_curve.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 7))
    for family, group in matrix.groupby("family"):
        size = np.clip(np.sqrt(group["stored_bytes_float32"].astype(float) + 1.0) * 1.8, 20, 450)
        ax.scatter(group["rmse_p95"], group["node_max_error_p95"], s=size, alpha=0.65, label=family)
    ax.set_xlabel("P95 RMSE")
    ax.set_ylabel("P95 node max error")
    ax.set_title("Node preservation vs sampled-curve reconstruction")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(analytics_dir / "node_vs_rmse_scatter.png", dpi=180)
    plt.close(fig)

    if not per_shape.empty:
        sample = per_shape.copy()
        fig, ax = plt.subplots(figsize=(10, 7))
        for family, group in sample.groupby("family"):
            ax.scatter(group["rmse"], group["node_max_error"], s=10, alpha=0.22, label=family)
        ax.axvline(0.02, color="gray", linestyle="--", linewidth=1)
        ax.axhline(0.05, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("Per-shape RMSE")
        ax.set_ylabel("Per-shape max node error")
        ax.set_title("Per-shape RMSE/node disagreement")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(analytics_dir / "per_shape_rmse_node_disagreement.png", dpi=180)
        plt.close(fig)

    if not factor3.empty:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        labels = [f"{row.factor3_grid} vs {row.comparison_grid}\n{row.eval_resolution}" for row in factor3.itertuples()]
        x = np.arange(len(factor3))
        ax.bar(x - 0.18, factor3["rmse_p95_relative"], width=0.36, label="RMSE P95 relative delta")
        ax.bar(x + 0.18, factor3["node_p95_relative"], width=0.36, label="Node P95 relative delta")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("Relative delta, lower is better")
        ax.set_title("Does a factor-of-3 grid buy accuracy?")
        ax.grid(axis="y", alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(analytics_dir / "factor3_relative_deltas.png", dpi=180)
        plt.close(fig)


def write_ablation_visuals(
    summary: pd.DataFrame,
    thresholds: pd.DataFrame,
    topology: pd.DataFrame,
    subsets: pd.DataFrame,
    disagreement: pd.DataFrame,
    output_dir: Path,
) -> dict[str, str]:
    """Write focused ablation plots for reading the Experiment 6 report."""
    analytics_dir = output_dir / "analytics"
    ablation_dir = analytics_dir / "ablations"
    ablation_dir.mkdir(parents=True, exist_ok=True)
    frame = _plot_table_labels(summary)
    plots: dict[str, str] = {}
    efficiency, marginal = parameter_efficiency_tables(summary)

    def save(fig, name: str) -> str:
        path = ablation_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        rel = f"analytics/ablations/{name}"
        plots[name.removesuffix(".png")] = rel
        return rel

    # 1. Whole ablation map.
    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    colors = {
        "grid": "#555555",
        "shared": "#4C78A8",
        "topology": "#72B7B2",
        "switch": "#F58518",
        "partition": "#B279A2",
        "additive": "#E45756",
    }
    eval1920 = frame[frame.eval_resolution == 1920].copy()
    for strategy, group in eval1920.groupby("strategy"):
        ax.scatter(
            group["dense_outputs"],
            group["rmse_p95"],
            s=np.clip(np.sqrt(group["stored_floats"].astype(float) + 1) * 0.42, 30, 420),
            alpha=0.74,
            label=strategy,
            color=colors.get(strategy, None),
            edgecolors="white",
            linewidths=0.5,
        )
    interesting = eval1920[
        eval1920["configuration"].isin(
            [
                "grid96_eval1920",
                "grid192_eval1920",
                "phase_additive_k4_d4_bw32_eval1920",
                "phase_additive_k8_d4_bw32_eval1920",
                "phase_additive_k12_d4_bw32_eval1920",
                "phase_additive_k16_d4_bw32_eval1920",
                "phase_shared_d4_bw32_eval1920",
                "phase_topology_d4_bw32_eval1920",
                "phase_switch_2_d4_bw32_eval1920",
                "phase_partition_s8_d4_bw32_eval1920",
            ]
        )
    ]
    for row in interesting.itertuples():
        ax.annotate(row.short_label.replace("additive_", "add_"), (row.dense_outputs, row.rmse_p95), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_title("Experiment 6 ablation map: output budget vs tail reconstruction")
    ax.set_xlabel("Dense outputs the future model would emit")
    ax.set_ylabel("Held-out P95 RMSE, lower is better")
    ax.grid(alpha=0.22)
    ax.legend(ncols=3, fontsize=8)
    save(fig, "ablation_map_eval1920.png")

    # 2. Direct grids, explicitly showing factor-of-3 widths.
    direct = frame[frame.family == "direct_grid"].copy()
    if not direct.empty:
        direct["grid_width"] = direct["candidate"].str.extract(r"(\d+)").astype(int)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True)
        for resolution, group in direct.groupby("eval_resolution"):
            group = group.sort_values("grid_width")
            axes[0].plot(group.grid_width, group.rmse_p95, marker="o", label=f"eval {resolution}")
            axes[1].plot(group.grid_width, group["rmse_under_0.02"], marker="o", label=f"eval {resolution}")
        for ax in axes:
            for width in (48, 96, 192):
                ax.axvline(width, color="gray", alpha=0.16)
                ax.text(width, ax.get_ylim()[1], "×3", ha="right", va="top", rotation=90, fontsize=8, color="gray")
            ax.grid(alpha=0.22)
            ax.set_xlabel("Grid points")
        axes[0].set_title("Direct-grid tail error")
        axes[0].set_ylabel("P95 RMSE")
        axes[1].set_title("Direct-grid threshold coverage")
        axes[1].set_ylabel("Share RMSE <= 0.02")
        axes[0].legend(fontsize=8)
        save(fig, "direct_grid_width_ablation.png")

    # 2b. Parameter efficiency: performance per model output.
    if not eval1920.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
        for strategy, group in eval1920.groupby("strategy"):
            axes[0].scatter(
                group["dense_outputs"],
                group["rmse_p95"],
                s=70,
                alpha=0.72,
                color=colors.get(strategy, None),
                label=strategy,
                edgecolors="white",
                linewidths=0.5,
            )
            axes[1].scatter(
                group["stored_bytes_float32"] / (1024 * 1024),
                group["rmse_p95"],
                s=70,
                alpha=0.72,
                color=colors.get(strategy, None),
                label=strategy,
                edgecolors="white",
                linewidths=0.5,
            )
        frontier = eval1920.sort_values(["dense_outputs", "rmse_p95"])
        best = []
        best_p95 = np.inf
        for row in frontier.itertuples():
            if row.rmse_p95 < best_p95:
                best.append(row)
                best_p95 = row.rmse_p95
        if best:
            axes[0].plot([x.dense_outputs for x in best], [x.rmse_p95 for x in best], color="black", linewidth=1.2, alpha=0.7, label="output frontier")
        for row in efficiency.itertuples():
            label = row.short_label.replace("phase_", "").replace("_bw32", "").replace("_d4", "_8st")
            axes[0].annotate(label, (row.dense_outputs, row.rmse_p95), fontsize=7, xytext=(4, 3), textcoords="offset points")
            axes[1].annotate(label, (row.stored_mb_float32, row.rmse_p95), fontsize=7, xytext=(4, 3), textcoords="offset points")
        axes[0].set_title("Model-output efficiency")
        axes[0].set_xlabel("Dense outputs")
        axes[0].set_ylabel("P95 RMSE, lower is better")
        axes[1].set_title("Decoder storage cost")
        axes[1].set_xlabel("Stored dictionary size, MiB float32")
        axes[1].set_ylabel("P95 RMSE, lower is better")
        for ax in axes:
            ax.grid(alpha=0.22)
        axes[0].legend(ncols=3, fontsize=8)
        fig.suptitle("Parameter efficiency vs performance")
        save(fig, "parameter_efficiency_frontier_eval1920.png")

    # 3. Additive width/depth.
    additive = frame[(frame.eval_resolution == 1920) & frame.candidate.str.contains("phase_additive", regex=False)].copy()
    if not additive.empty:
        additive["k"] = additive["candidate"].str.extract(r"_k(\d+)_").astype(int)
        additive["actual_stages"] = additive["depth"].astype(int)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
        metrics = [
            ("rmse_p95", "P95 RMSE"),
            ("rmse_under_0.02", "Share RMSE <= 0.02"),
            ("all_nodes_under_0.02", "Share all nodes <= 0.02"),
        ]
        for ax, (metric, label) in zip(axes, metrics):
            for k, group in additive.groupby("k"):
                group = group.sort_values("actual_stages")
                ax.plot(group["actual_stages"], group[metric], marker="o", label=f"k{k}")
            ax.set_title(label)
            ax.set_xlabel("Actual residual stages")
            ax.set_xticks(sorted(additive["actual_stages"].unique()))
            ax.grid(alpha=0.22)
        axes[0].set_ylabel("Lower is better")
        axes[1].set_ylabel("Higher is better")
        axes[2].set_ylabel("Higher is better")
        axes[0].legend(title="Residual width", fontsize=8)
        fig.suptitle("Additive codebook ablation: width and actual residual depth")
        save(fig, "additive_width_depth_ablation_eval1920.png")

    # 3b. Marginal efficiency of additive steps.
    if not marginal.empty:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
        for ax, axis_name, title in (
            (axes[0], "depth", "Depth steps"),
            (axes[1], "width_at_8_stages", "Width steps at 8 stages"),
        ):
            part = marginal[marginal["axis"] == axis_name].sort_values("p95_improvement_per_output", ascending=True)
            y = np.arange(len(part))
            ax.barh(y, part["p95_improvement_per_output"] * 1000, color="#E45756" if axis_name == "width_at_8_stages" else "#4C78A8")
            ax.set_yticks(y)
            ax.set_yticklabels(part["step"], fontsize=8)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("P95 RMSE improvement per 1k dense outputs")
            ax.set_title(title)
            ax.grid(axis="x", alpha=0.22)
        fig.suptitle("Marginal parameter efficiency: what each extra output bought")
        save(fig, "additive_marginal_efficiency_eval1920.png")

    # 4. Organization strategy at the deepest tested setting.
    selected = frame[
        (frame.eval_resolution == 1920)
        & (
            frame["configuration"].isin(
                [
                    "phase_shared_d4_bw32_eval1920",
                    "phase_topology_d4_bw32_eval1920",
                    "phase_switch_1_d4_bw32_eval1920",
                    "phase_switch_2_d4_bw32_eval1920",
                    "phase_switch_3_d4_bw32_eval1920",
                    "phase_partition_s8_d4_bw32_eval1920",
                    "phase_partition_taper_12_9_6_3_d4_bw32_eval1920",
                    "phase_additive_k4_d4_bw32_eval1920",
                    "phase_additive_k8_d4_bw32_eval1920",
                    "phase_additive_k12_d4_bw32_eval1920",
                    "phase_additive_k16_d4_bw32_eval1920",
                ]
            )
        )
    ].copy()
    if not selected.empty:
        selected = selected.sort_values("rmse_p95")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.7), sharey=True)
        labels = (
            selected["short_label"]
            .str.replace("phase_", "", regex=False)
            .str.replace("_bw32", "", regex=False)
            .str.replace("_d4", "_8st", regex=False)
        )
        y = np.arange(len(selected))
        axes[0].barh(y, selected["rmse_p95"], color=[colors.get(x, "#777777") for x in selected["strategy"]])
        axes[1].barh(y, selected["all_nodes_under_0.02"], color=[colors.get(x, "#777777") for x in selected["strategy"]])
        axes[0].set_yticks(y)
        axes[0].set_yticklabels(labels, fontsize=8)
        axes[0].invert_yaxis()
        axes[0].set_title("Tail reconstruction")
        axes[0].set_xlabel("P95 RMSE, lower is better")
        axes[1].set_title("Node coverage")
        axes[1].set_xlabel("Share all nodes <= 0.02")
        for ax in axes:
            ax.grid(axis="x", alpha=0.22)
        fig.suptitle("Codebook organization ablation at 8 actual residual stages")
        save(fig, "organization_strategy_d4_eval1920.png")

    # 5. Threshold coverage for key candidates.
    key_configs = [
        "grid96_eval1920",
        "grid192_eval1920",
        "phase_additive_k4_d4_bw32_eval1920",
        "phase_additive_k8_d4_bw32_eval1920",
        "phase_additive_k12_d4_bw32_eval1920",
        "phase_additive_k16_d4_bw32_eval1920",
    ]
    cov = thresholds[(thresholds.subset == "all") & thresholds.configuration.isin(key_configs)].copy()
    if not cov.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharex=False)
        for metric, ax in (("rmse", axes[0]), ("all_nodes", axes[1])):
            part = cov[cov.metric == metric]
            for configuration, group in part.groupby("configuration"):
                group = group.sort_values("threshold")
                label = configuration.replace("phase_", "").replace("_bw32_eval1920", "").replace("_eval1920", "")
                ax.plot(group["threshold"], group["coverage"], marker="o", label=label)
            ax.set_xscale("log")
            ax.set_ylim(-0.02, 1.02)
            ax.grid(alpha=0.22)
            ax.set_xlabel("Threshold")
            ax.set_ylabel("Coverage")
            ax.set_title("RMSE coverage" if metric == "rmse" else "All-node coverage")
        axes[0].legend(fontsize=7, ncols=2)
        fig.suptitle("Threshold coverage: structured tail wins vs grid node coverage")
        save(fig, "threshold_coverage_key_candidates.png")

    # 6. Custom-ish subset.
    custom = subsets[
        (subsets.subset == "custom_ish")
        & subsets.configuration.isin(
            [
                "grid96_eval1920",
                "grid192_eval1920",
                "phase_shared_d4_bw32_eval1920",
                "phase_topology_d4_bw32_eval1920",
                "phase_switch_2_d4_bw32_eval1920",
                "phase_additive_k4_d4_bw32_eval1920",
                "phase_additive_k8_d4_bw32_eval1920",
                "phase_additive_k12_d4_bw32_eval1920",
                "phase_additive_k16_d4_bw32_eval1920",
            ]
        )
    ].copy()
    if not custom.empty:
        custom["label"] = custom["configuration"].str.replace("phase_", "", regex=False).str.replace("_bw32_eval1920", "", regex=False).str.replace("_eval1920", "", regex=False)
        custom = custom.sort_values("rmse_p95")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
        y = np.arange(len(custom))
        axes[0].barh(y, custom["rmse_p95"], color="#E45756")
        axes[1].barh(y, custom["all_nodes_under_0.02"], color="#4C78A8")
        axes[0].set_yticks(y)
        axes[0].set_yticklabels(custom["label"], fontsize=8)
        axes[0].invert_yaxis()
        axes[0].set_xlabel("P95 RMSE")
        axes[0].set_title("Custom-ish reconstruction")
        axes[1].set_xlabel("Share all nodes <= 0.02")
        axes[1].set_title("Custom-ish node preservation")
        for ax in axes:
            ax.grid(axis="x", alpha=0.22)
        fig.suptitle("Custom-ish subset: where grids still matter")
        save(fig, "custom_ish_key_candidates.png")

    # 7. Topology tails for chosen candidates.
    topo_key = topology[
        topology.configuration.isin(
            [
                "grid192_eval1920",
                "phase_shared_d4_bw32_eval1920",
                "phase_topology_d4_bw32_eval1920",
                "phase_switch_2_d4_bw32_eval1920",
                "phase_additive_k8_d4_bw32_eval1920",
                "phase_additive_k12_d4_bw32_eval1920",
                "phase_additive_k16_d4_bw32_eval1920",
            ]
        )
    ].copy()
    if not topo_key.empty:
        topo_key["label"] = topo_key["configuration"].str.replace("phase_", "", regex=False).str.replace("_bw32_eval1920", "", regex=False).str.replace("_eval1920", "", regex=False)
        pivot = topo_key.pivot(index="label", columns="topology", values="rmse_p95")
        pivot = pivot.reindex(pivot.mean(axis=1).sort_values().index)
        fig, ax = plt.subplots(figsize=(10.5, 5.5))
        x = np.arange(len(pivot))
        width = 0.24
        for i, topology_name in enumerate(pivot.columns):
            ax.bar(x + (i - 1) * width, pivot[topology_name], width=width, label=topology_name)
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("P95 RMSE")
        ax.set_title("Topology tail behavior for key candidates")
        ax.grid(axis="y", alpha=0.22)
        ax.legend(fontsize=8)
        save(fig, "topology_tail_key_candidates.png")

    # 8. RMSE/node disagreement.
    key_disagreement = disagreement[
        disagreement.configuration.isin(
            [
                "grid96_eval1920",
                "grid192_eval1920",
                "phase_shared_d4_bw32_eval1920",
                "phase_topology_d4_bw32_eval1920",
                "phase_switch_2_d4_bw32_eval1920",
                "phase_additive_k4_d4_bw32_eval1920",
                "phase_additive_k8_d4_bw32_eval1920",
                "phase_additive_k12_d4_bw32_eval1920",
                "phase_additive_k16_d4_bw32_eval1920",
            ]
        )
    ].copy()
    if not key_disagreement.empty:
        key_disagreement["label"] = key_disagreement["configuration"].str.replace("phase_", "", regex=False).str.replace("_bw32_eval1920", "", regex=False).str.replace("_eval1920", "", regex=False)
        key_disagreement = key_disagreement.sort_values("good_rmse_bad_nodes_share", ascending=False)
        fig, ax = plt.subplots(figsize=(11.5, 5.2))
        x = np.arange(len(key_disagreement))
        ax.bar(x, key_disagreement["good_rmse_bad_nodes_share"], color="#B279A2")
        ax.set_xticks(x)
        ax.set_xticklabels(key_disagreement["label"], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Share of shapes")
        ax.set_title("Good sampled RMSE but bad editor-node match")
        ax.grid(axis="y", alpha=0.22)
        save(fig, "rmse_node_disagreement_key_candidates.png")

    return plots


def parameter_efficiency_tables(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return compact efficiency tables for eval1920 candidates."""
    frame = _plot_table_labels(summary[summary.eval_resolution == 1920].copy())
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    baseline = frame.loc[frame["configuration"] == "grid32_eval1920", "rmse_p95"]
    baseline_value = float(baseline.iloc[0]) if len(baseline) else float(frame["rmse_p95"].max())
    frame["p95_gain_vs_grid32"] = baseline_value - frame["rmse_p95"]
    frame["p95_gain_per_dense_output"] = frame["p95_gain_vs_grid32"] / frame["dense_outputs"].clip(lower=1)
    frame["stored_mb_float32"] = frame["stored_bytes_float32"] / (1024 * 1024)
    candidates = frame[
        frame["configuration"].isin(
            [
                "grid96_eval1920",
                "grid192_eval1920",
                "phase_additive_k4_d4_bw32_eval1920",
                "phase_additive_k8_d4_bw32_eval1920",
                "phase_additive_k12_d4_bw32_eval1920",
                "phase_additive_k16_d4_bw32_eval1920",
                "phase_shared_d4_bw32_eval1920",
                "phase_topology_d4_bw32_eval1920",
                "phase_switch_2_d4_bw32_eval1920",
                "phase_partition_s8_d4_bw32_eval1920",
            ]
        )
    ].copy()
    candidates = candidates.sort_values(["rmse_p95", "dense_outputs"])

    additive = frame[frame["candidate"].str.contains("phase_additive", regex=False)].copy()
    rows = []
    if not additive.empty:
        additive["k"] = additive["candidate"].str.extract(r"_k(\d+)_").astype(int)
        additive["actual_stages"] = additive["depth"].astype(int)
        for k, group in additive.groupby("k"):
            group = group.sort_values("actual_stages")
            previous = None
            for row in group.itertuples():
                if previous is not None:
                    delta_outputs = float(row.dense_outputs - previous.dense_outputs)
                    delta_p95 = float(previous.rmse_p95 - row.rmse_p95)
                    rows.append(
                        {
                            "axis": "depth",
                            "step": f"k{k}: {int(previous.actual_stages)} -> {int(row.actual_stages)} stages",
                            "from_configuration": previous.configuration,
                            "to_configuration": row.configuration,
                            "extra_outputs": delta_outputs,
                            "p95_improvement": delta_p95,
                            "p95_improvement_per_output": delta_p95 / delta_outputs if delta_outputs else np.nan,
                        }
                    )
                previous = row
        deepest = additive[additive["actual_stages"] == additive["actual_stages"].max()].sort_values("k")
        previous = None
        for row in deepest.itertuples():
            if previous is not None:
                delta_outputs = float(row.dense_outputs - previous.dense_outputs)
                delta_p95 = float(previous.rmse_p95 - row.rmse_p95)
                rows.append(
                    {
                        "axis": "width_at_8_stages",
                        "step": f"k{int(previous.k)} -> k{int(row.k)} at 8 stages",
                        "from_configuration": previous.configuration,
                        "to_configuration": row.configuration,
                        "extra_outputs": delta_outputs,
                        "p95_improvement": delta_p95,
                        "p95_improvement_per_output": delta_p95 / delta_outputs if delta_outputs else np.nan,
                    }
                )
            previous = row
    marginal = pd.DataFrame(rows).sort_values(["axis", "p95_improvement_per_output"], ascending=[True, False])
    return candidates, marginal


def write_report(
    output_dir: Path,
    matrix: pd.DataFrame,
    factor3: pd.DataFrame,
    topo_gap: pd.DataFrame,
    subsets: pd.DataFrame,
    disagreement: pd.DataFrame,
    ablation_plots: dict[str, str],
    efficiency: pd.DataFrame,
    marginal_efficiency: pd.DataFrame,
) -> None:
    analytics_dir = output_dir / "analytics"
    best_balanced = matrix.head(12)
    efficient_tail = matrix.sort_values(["rmse_p95", "dense_outputs"]).head(12)
    efficient_node = matrix.sort_values(["node_max_error_p95", "dense_outputs"]).head(12)
    custom = subsets[subsets.subset == "custom_ish"].sort_values(["rmse_p95", "node_max_error_p95"]).head(12)
    def img(key: str, caption: str) -> str:
        path = ablation_plots.get(key)
        return "" if not path else _captioned_image(path, caption)

    content = f"""# Experiment 6 Analytics

This is the visual readout for the Experiment 6 sweep. The point is not to crown a winner with a fake-objective scepter; it is to see which compromises are real.

## Read this first

The structured additive codebooks are doing the best job on sampled-curve tail error. Direct grids are still stubbornly useful for custom-ish shapes and editor-node preservation. So the shape of the answer is probably hybrid: structured codebook for search reduction, plus a small direct/grid residual or fallback for cases where nodes matter more than code compactness.

Important naming footgun: the additive config suffixes are not the same as actual residual stages. In this run, additive `d2` means 4 actual stages, `d3` means 6, and `d4` means 8. The deepest tested additive variants are therefore the 8-stage variants.

{img('ablation_map_eval1920', 'Overall ablation map at eval1920. Additive candidates form the useful structured frontier; Grid192 remains a strong dense fallback. Marker size reflects stored dictionary size.')}

## Parameter efficiency vs performance

This is the missing lens if we only stare at the best RMSE. Dense outputs are what the future model has to emit; stored floats are what the decoder/codebook has to carry. Those are different costs, and we should keep them separate.

{img('parameter_efficiency_frontier_eval1920', 'Parameter efficiency at eval1920. The left panel is model-output cost; the right panel is decoder dictionary storage.')}

{_markdown_table(efficiency, ['configuration', 'dense_outputs', 'categorical_logits', 'continuous_scalars', 'effective_index_bits', 'stored_mb_float32', 'rmse_p95', 'p95_gain_per_dense_output'], limit=12)}

The marginal view is the useful one for the additive family. Depth steps are still buying meaningful tail-error reduction; the k12 -> k16 width step is much less compelling than going deeper would likely be.

{img('additive_marginal_efficiency_eval1920', 'Marginal additive efficiency. Depth is not saturated; width has clearer diminishing returns at the 8-stage setting.')}

{_markdown_table(marginal_efficiency, ['axis', 'step', 'extra_outputs', 'p95_improvement', 'p95_improvement_per_output'], limit=16)}

## Direct grid ablation

Grid is the "just predict sampled points" baseline. The factor-of-3 widths are not cosmetic: Grid48/Grid96/Grid192 consistently improve tail RMSE over the adjacent power-of-two-ish grids.

{img('direct_grid_width_ablation', 'Direct-grid width ablation. Factor-of-3 grids buy real RMSE coverage, while node preservation remains a separate issue.')}

{_markdown_table(factor3, ['eval_resolution', 'factor3_grid', 'comparison_grid', 'extra_outputs', 'rmse_p95_delta', 'rmse_p95_relative', 'node_p95_delta', 'node_p95_relative'])}

## Additive ablation

Additive means the oracle applies both shared and topology-specific residual corrections. This is the family that matters most from this run. Depth is doing at least as much work as width here: 4 -> 6 -> 8 actual stages keeps improving, and we should not treat depth as saturated. Width starts to flatten around k12 at the deepest tested setting, but depth needs a follow-up sweep.

{img('additive_width_depth_ablation_eval1920', 'Additive width/depth ablation at eval1920. Focus on the 8-stage points: k12 is the current practical default, k16 is the tested upper-bound, and depth has not clearly saturated.')}

## Codebook organization ablation

Shared, topology, switch, and partition are offline decoder/codebook organizations, not learned predictors yet. This comparison is most useful at the deepest tested setting, because shallow stacks are mixing together organization quality and insufficient depth. Additive wins because it does not force common structure and topology-specific structure to compete for the same slot.

{img('organization_strategy_d4_eval1920', 'Codebook organization ablation at 8 actual residual stages. Additive dominates switch/partition/shared/topology on reconstruction tail while staying in a plausible output budget.')}

## Threshold coverage

P95 is useful, but coverage makes the tradeoff easier to feel. The structured candidates put many more shapes below RMSE 0.02. Grids keep a stronger node-coverage story.

{img('threshold_coverage_key_candidates', 'Threshold coverage for key candidates. Structured codebooks win sampled-curve coverage; grids hold onto node coverage.')}

## Custom-ish subset

This is the warning light. On custom-ish shapes, direct grids remain very competitive or better. That does not invalidate the structured codebook; it says we should not pretend the codebook alone is the whole LFO representation.

{img('custom_ish_key_candidates', 'Custom-ish subset. Grid192 is still hard to beat here, especially for node preservation.')}

{_markdown_table(custom, ['configuration', 'subset', 'shapes', 'rmse_median', 'rmse_p95', 'node_max_error_p95', 'rmse_under_0.02', 'all_nodes_under_0.02'])}

## Topology behavior

Discontinuous and smooth tails are where simple grids and some structured variants get exposed. Additive is less brittle than the pure shared/topology/switch families, but topology-specific failure modes are still visible.

{img('topology_tail_key_candidates', 'Topology tail behavior for key candidates. This helps separate real robustness from doing well on the easy continuous cases.')}

## RMSE/node disagreement

This is the most important caveat for editor-state modeling: good sampled RMSE does not automatically mean the original editor nodes are plausible. The structured methods can reconstruct the sound-ish curve while missing node-level details.

{img('rmse_node_disagreement_key_candidates', 'Good sampled RMSE but bad editor-node match. This is why a direct residual/fallback remains attractive.')}

{_markdown_table(disagreement, ['configuration', 'good_rmse_bad_nodes_share', 'bad_rmse_good_nodes_share', 'rmse_node_corr', 'median_node_given_rmse_under_0.02'])}

## Balanced discussion table

Lower `balanced_discussion_score` is better, but this score is only a sorting aid. It mixes tail RMSE, node preservation, threshold coverage, dense outputs, and storage.

{_markdown_table(best_balanced, ['configuration', 'family', 'dense_outputs', 'stored_floats', 'rmse_median', 'rmse_p95', 'rmse_under_0.02', 'node_max_error_p95', 'all_nodes_under_0.02', 'balanced_discussion_score'])}

## Tail-error leaders

{_markdown_table(efficient_tail, ['configuration', 'family', 'dense_outputs', 'rmse_median', 'rmse_p95', 'node_max_error_p95', 'rmse_under_0.02', 'all_nodes_under_0.02'])}

## Node-preservation leaders

{_markdown_table(efficient_node, ['configuration', 'family', 'dense_outputs', 'rmse_p95', 'node_max_error_p95', 'all_nodes_under_0.02'])}

## Topology sensitivity

Large gaps here mean a candidate is uneven across smooth/continuous/discontinuous shapes.

{_markdown_table(topo_gap, ['configuration', 'topology_rmse_p95_gap', 'worst_topology_rmse', 'topology_node_p95_gap', 'worst_topology_node'])}

## Extra visuals

- `analytics/quality_complexity_dashboard.png`
- `analytics/rmse_threshold_heatmap.png`
- `analytics/node_threshold_heatmap.png`
- `analytics/topology_p95_heatmap.png`
- `analytics/factor3_grid_curve.png`
- `analytics/factor3_relative_deltas.png`
- `analytics/node_vs_rmse_scatter.png`
- `analytics/per_shape_rmse_node_disagreement.png`

## Current recommendation

Use additive k12 with 8 actual residual stages as the default structured candidate, additive k8 with 8 stages as the compact candidate, and additive k16 with 8 stages as the tested upper-bound structured reference. Do not freeze depth yet: run a deeper additive sweep before finalizing the production codebook. Keep Grid96/Grid192 in the design conversation as direct residual/fallback candidates, not as embarrassments to be swept under the rug.
"""
    (output_dir / "EXPERIMENT_6_FINDINGS.md").write_text(content, encoding="utf-8")


def run_experiment6_analysis(output_dir: Path) -> dict[str, Path]:
    output_dir = output_dir.resolve()
    summary_path = output_dir / "summary.csv"
    thresholds_path = output_dir / "threshold_coverage.csv"
    topology_path = output_dir / "topology_summary.csv"
    per_shape_path = output_dir / "per_shape_results.csv"
    missing = [path for path in (summary_path, thresholds_path, topology_path, per_shape_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Experiment 6 result files: " + ", ".join(str(path) for path in missing))

    analytics_dir = output_dir / "analytics"
    analytics_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(summary_path)
    thresholds = pd.read_csv(thresholds_path)
    topology = pd.read_csv(topology_path)
    per_shape = pd.read_csv(per_shape_path, low_memory=False)

    matrix = decision_matrix(summary, thresholds)
    factor3 = factor3_grid_deltas(matrix)
    topo = topology_gap(topology)
    subsets = subset_breakdown(per_shape)
    worst = worst_cases(per_shape)
    disagreement = node_rmse_disagreement(per_shape)
    efficiency, marginal_efficiency = parameter_efficiency_tables(summary)

    matrix.to_csv(analytics_dir / "decision_matrix.csv", index=False)
    factor3.to_csv(analytics_dir / "factor3_grid_deltas.csv", index=False)
    topo.to_csv(analytics_dir / "topology_sensitivity.csv", index=False)
    subsets.to_csv(analytics_dir / "subset_breakdown.csv", index=False)
    worst.to_csv(analytics_dir / "worst_cases.csv", index=False)
    disagreement.to_csv(analytics_dir / "node_rmse_disagreement.csv", index=False)
    efficiency.to_csv(analytics_dir / "parameter_efficiency_key_candidates.csv", index=False)
    marginal_efficiency.to_csv(analytics_dir / "additive_marginal_efficiency.csv", index=False)

    write_visuals(summary, thresholds, topology, per_shape, matrix, factor3, output_dir)
    ablation_plots = write_ablation_visuals(summary, thresholds, topology, subsets, disagreement, output_dir)
    write_report(
        output_dir,
        matrix,
        factor3,
        topo,
        subsets,
        disagreement,
        ablation_plots,
        efficiency,
        marginal_efficiency,
    )
    return {
        "analytics_dir": analytics_dir,
        "report": output_dir / "EXPERIMENT_6_FINDINGS.md",
        "decision_matrix": analytics_dir / "decision_matrix.csv",
    }


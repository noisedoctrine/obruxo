#!/usr/bin/env python3
"""Generate the Experiment 8 findings report and report-specific plots."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("artifacts/additive_finalization_8_screen")
ANALYTICS = ROOT / "analytics"
REPORTS = Path("reports")
REPORT = REPORTS / "experiment-08-findings.md"
PLOTS = REPORTS / "images" / "experiment-08"
CONFIG_PATTERN = re.compile(r"W(?P<W>\d+)D(?P<D>\d+)\s+(?P<modifier>\S+)\s+(?P<clip>\S+)")


def _format_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float | np.floating):
        return f"{float(value):.6g}"
    return str(value)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame is None or len(frame) == 0:
        return "_No rows found._"
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.to_numpy():
        lines.append("| " + " | ".join(_format_value(value) for value in row) + " |")
    return "\n".join(lines)


def _modifier_indicators(summary: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    labels = summary["modifier_label"].astype(str)
    has_gain = labels.isin(["phase_gain", "phase_gain_offset"]).astype(int)
    has_offset = labels.isin(["phase_offset", "phase_gain_offset"]).astype(int)
    return has_gain, has_offset


def _add_head_accounting(summary: pd.DataFrame) -> pd.DataFrame:
    """Add the intended downstream output-head accounting.

    This intentionally does not reuse the older dense_outputs column. The old
    column came from serialized decoder fields and legacy implementation
    details; this one is the interface the model has to predict.
    """
    frame = summary.copy()
    width = frame["residual_width"].astype(int)
    depth = frame["residual_depth"].astype(int)
    has_gain, has_offset = _modifier_indicators(frame)
    modifier_families = 1 + has_gain + has_offset
    frame["I_phase"] = 1
    frame["I_gain"] = has_gain
    frame["I_offset"] = has_offset
    frame["cat_logits"] = 32 + width * depth
    frame["scalar_outputs"] = (depth + 1) * modifier_families
    frame["head_outputs"] = frame["cat_logits"] + frame["scalar_outputs"]
    frame["optional_modifier_outputs"] = (depth + 1) * (has_gain + has_offset)
    return frame


def _config_label(row: pd.Series) -> str:
    return (
        f"W{int(row.residual_width)}D{int(row.residual_depth)} "
        f"{row.modifier_label} {row.residual_clip_policy}"
    )


def _parse_config(label: str) -> dict[str, int | str]:
    match = CONFIG_PATTERN.match(label)
    if match is None:
        return {"W": -1, "D": -1, "modifier": label, "clip": ""}
    values = match.groupdict()
    return {
        "W": int(values["W"]),
        "D": int(values["D"]),
        "modifier": str(values["modifier"]),
        "clip": str(values["clip"]),
    }


def _move_label(row: pd.Series) -> str:
    before = _parse_config(str(row.from_config))
    after = _parse_config(str(row.to_config))
    move = str(row.move)
    if move == "depth":
        return f"D {before['D']} -> {after['D']} @ W{before['W']}"
    if move == "width":
        return f"W {before['W']} -> {after['W']} @ D{before['D']}"
    if move == "gain":
        return f"gain @ W{before['W']}D{before['D']}"
    if move == "offset":
        return f"offset @ W{before['W']}D{before['D']}"
    if move == "offset_after_gain":
        return f"offset after gain @ W{before['W']}D{before['D']}"
    if move == "clipping":
        return f"clip policy @ W{before['W']}D{before['D']}"
    return f"{move}: {row.from_config} -> {row.to_config}"


def _save_bar(frame: pd.DataFrame, x: str, ys: list[tuple[str, str]], title: str, ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    labels = frame[x].astype(str).tolist()
    positions = np.arange(len(frame))
    width = 0.35 if len(ys) == 2 else 0.25
    for index, (column, label) in enumerate(ys):
        ax.bar(positions + (index - (len(ys) - 1) / 2) * width, frame[column], width, label=label)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_metric_panels(
    frame: pd.DataFrame,
    x: str,
    title: str,
    path: Path,
    baseline_column: str | None = None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))
    labels = frame[x].astype(str).tolist()
    positions = np.arange(len(frame))
    metrics = [("rmse_median", "Median RMSE", "#2563eb"), ("rmse_p95", "P95 RMSE", "#f97316")]
    for ax, (metric, ylabel, color) in zip(axes, metrics, strict=True):
        values = frame[metric].astype(float).to_numpy()
        ax.plot(positions, values, color=color, marker="o", linewidth=1.8, markersize=6)
        if baseline_column is not None and baseline_column in frame:
            baseline_rows = frame[frame[baseline_column].astype(bool)]
            if not baseline_rows.empty:
                ax.axhline(float(baseline_rows.iloc[0][metric]), color="#111827", linewidth=1.0, linestyle="--")
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(axis="y", alpha=0.25)
        span = values.max() - values.min()
        if span > 0:
            ax.set_ylim(max(0.0, values.min() - span * 0.35), values.max() + span * 0.45)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_scatter(
    frame: pd.DataFrame,
    x: str,
    y: str,
    color: str,
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
    note: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    scatter = ax.scatter(frame[x], frame[y], c=frame[color], cmap="viridis", s=70, alpha=0.85)
    for _, row in frame.iterrows():
        ax.annotate(
            f"W{int(row.residual_width)}D{int(row.residual_depth)}",
            (row[x], row[y]),
            fontsize=7,
            xytext=(4, 3),
            textcoords="offset points",
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    if note:
        ax.text(
            0.02,
            0.98,
            note,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            color="#475569",
            bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.86, "boxstyle": "round,pad=0.25"},
        )
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label(color)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_size_metric_pair(
    frame: pd.DataFrame,
    color: str,
    family: str,
    color_label: str,
    title: str,
    path: Path,
    note: str | None = None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2), sharex=True)
    metrics = [("rmse_median", "Median RMSE"), ("rmse_p95", "P95 RMSE")]
    vmin = float(frame[color].min())
    vmax = float(frame[color].max())
    cmap = plt.get_cmap("viridis")
    denominator = vmax - vmin if vmax > vmin else 1.0
    scatter = None
    for ax, (metric, ylabel) in zip(axes, metrics, strict=True):
        for family_value, group in frame.groupby(family):
            ordered = group.sort_values(["head_outputs", "residual_width", "residual_depth"])
            normalized = (float(family_value) - vmin) / denominator if family == color else 0.42
            line_color = cmap(normalized) if family == color else "#94a3b8"
            ax.plot(
                ordered["head_outputs"],
                ordered[metric],
                color=line_color,
                linewidth=1.4,
                alpha=0.34,
                zorder=1,
            )
        scatter = ax.scatter(
            frame["head_outputs"],
            frame[metric],
            c=frame[color],
            cmap="viridis",
            s=74,
            alpha=0.86,
            vmin=vmin,
            vmax=vmax,
            zorder=2,
        )
        for _, row in frame.iterrows():
            ax.annotate(
                f"W{int(row.residual_width)}D{int(row.residual_depth)}",
                (row["head_outputs"], row[metric]),
                fontsize=7,
                xytext=(4, 3),
                textcoords="offset points",
            )
        ax.set_title(ylabel)
        ax.set_xlabel("head outputs per LFO")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        if note:
            ax.text(
                0.02,
                0.98,
                note,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                color="#475569",
                bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.86, "boxstyle": "round,pad=0.25"},
            )
    fig.suptitle(title)
    if scatter is not None:
        colorbar = fig.colorbar(scatter, ax=axes, pad=0.02)
        colorbar.set_label(color_label)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_size_heatmap_pair(frame: pd.DataFrame, path: Path) -> None:
    plot = frame.copy()
    metrics = [("rmse_median", "Median RMSE"), ("rmse_p95", "P95 RMSE")]
    widths = sorted(plot["residual_width"].astype(int).unique())
    depths = sorted(plot["residual_depth"].astype(int).unique())
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), sharey=True)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#f8fafc")
    for ax, (metric, title) in zip(axes, metrics, strict=True):
        pivot = plot.pivot(index="residual_width", columns="residual_depth", values=metric).reindex(index=widths, columns=depths)
        colored = pivot.copy()
        if 4 in colored.columns:
            colored[4] = np.nan
        finite = colored.to_numpy(dtype=float)
        valid = finite[np.isfinite(finite)]
        image = ax.imshow(
            finite,
            cmap=cmap,
            aspect="auto",
            vmin=float(valid.min()),
            vmax=float(valid.max()),
        )
        for row_index, width in enumerate(widths):
            for col_index, depth in enumerate(depths):
                value = pivot.loc[width, depth]
                if pd.isna(value):
                    continue
                if depth == 4:
                    text = "off\nscale"
                    color = "#64748b"
                else:
                    text = f"{float(value):.4f}"
                    color = "white" if float(value) > float(valid.mean()) else "#111827"
                ax.text(col_index, row_index, text, ha="center", va="center", fontsize=7, color=color)
        ax.set_title(title)
        ax.set_xticks(np.arange(len(depths)))
        ax.set_xticklabels([f"D{depth}" for depth in depths])
        ax.set_yticks(np.arange(len(widths)))
        ax.set_yticklabels([f"W{width}" for width in widths])
        ax.set_xlabel("residual depth")
        ax.grid(False)
        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label("RMSE")
    axes[0].set_ylabel("residual width")
    fig.suptitle("Experiment 8 size screen heatmaps; D4 annotated, not color-scaled")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_runtime_pair(frame: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2), sharey=True)
    panels = [
        ("residual_width", "residual_width", "Runtime by width family"),
        ("residual_depth", "residual_depth", "Runtime by depth family"),
    ]
    for ax, (color, family, title) in zip(axes, panels, strict=True):
        vmin = float(frame[color].min())
        vmax = float(frame[color].max())
        denominator = vmax - vmin if vmax > vmin else 1.0
        cmap = plt.get_cmap("viridis")
        for family_value, group in frame.groupby(family):
            ordered = group.sort_values(["head_outputs", "residual_width", "residual_depth"])
            normalized = (float(family_value) - vmin) / denominator if family == color else 0.42
            ax.plot(
                ordered["head_outputs"],
                ordered["elapsed_seconds_total"],
                color=cmap(normalized) if family == color else "#94a3b8",
                linewidth=1.4,
                alpha=0.34,
                zorder=1,
            )
        scatter = ax.scatter(
            frame["head_outputs"],
            frame["elapsed_seconds_total"],
            c=frame[color],
            cmap="viridis",
            s=74,
            alpha=0.86,
            vmin=vmin,
            vmax=vmax,
            zorder=2,
        )
        for _, row in frame.iterrows():
            ax.annotate(
                f"W{int(row.residual_width)}D{int(row.residual_depth)}",
                (row["head_outputs"], row["elapsed_seconds_total"]),
                fontsize=7,
                xytext=(4, 3),
                textcoords="offset points",
            )
        ax.set_title(title)
        ax.set_xlabel("head outputs per LFO")
        ax.grid(alpha=0.25)
        colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
        colorbar.set_label(color)
    axes[0].set_ylabel("elapsed seconds")
    fig.suptitle("Experiment 8 runtime scaling")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_pareto(
    frame: pd.DataFrame,
    frontier: pd.DataFrame,
    y: str,
    title: str,
    ylabel: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    scatter = ax.scatter(frame["head_outputs"], frame[y], c=frame["residual_depth"], cmap="viridis", s=70, alpha=0.72)
    ordered = frontier.sort_values("head_outputs")
    ax.plot(ordered["head_outputs"], ordered[y], color="#1f2937", linewidth=1.5, marker="o", label="Pareto frontier")
    for _, row in ordered.iterrows():
        ax.annotate(
            f"W{int(row.residual_width)}D{int(row.residual_depth)}\n{row.modifier_label}",
            (row["head_outputs"], row[y]),
            fontsize=7,
            xytext=(4, 3),
            textcoords="offset points",
        )
    ax.set_title(title)
    ax.set_xlabel("head outputs per LFO")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend()
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("residual_depth")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_output_accounting(frame: pd.DataFrame, path: Path) -> None:
    ordered = frame.sort_values(["head_outputs", "residual_width", "residual_depth"])
    labels = [f"W{int(row.residual_width)}D{int(row.residual_depth)}" for _, row in ordered.iterrows()]
    positions = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.bar(positions, ordered["cat_logits"], label="categorical logits")
    ax.bar(
        positions,
        ordered["scalar_outputs"],
        bottom=ordered["cat_logits"],
        label="continuous scalars",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_title("Experiment 8 analytic output-head accounting")
    ax.set_xlabel("configuration")
    ax.set_ylabel("head outputs per LFO")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _representative_cost_moves(moves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    finite = moves[moves["delta_head_outputs"] > 0].copy()
    finite["label"] = finite.apply(_move_label, axis=1)
    for move in ["gain", "offset", "offset_after_gain"]:
        subset = finite[finite["move"] == move]
        if not subset.empty:
            rows.append(subset.sort_values("label").iloc[0])
    depth = finite[finite["move"] == "depth"].copy()
    if not depth.empty:
        depth["W"] = depth["from_config"].map(lambda value: int(_parse_config(str(value))["W"]))
        for _, group in depth.groupby("W"):
            rows.append(group.sort_values("delta_head_outputs").iloc[0])
    width = finite[finite["move"] == "width"].sort_values("delta_head_outputs")
    if not width.empty:
        rows.extend([width.iloc[0], width.iloc[len(width) // 2], width.iloc[-1]])
    if not rows:
        return finite
    return pd.DataFrame(rows).drop_duplicates(subset=["move", "from_config", "to_config"])


def _save_marginal_cost(moves: pd.DataFrame, path: Path) -> None:
    ordered = _representative_cost_moves(moves).sort_values(["delta_head_outputs", "label"]).copy()
    fig, ax = plt.subplots(figsize=(8.8, max(4.4, 0.42 * len(ordered))))
    positions = np.arange(len(ordered))
    colors = ordered["move"].map(
        {
            "gain": "#16a34a",
            "offset": "#ca8a04",
            "offset_after_gain": "#a16207",
            "depth": "#2563eb",
            "width": "#7c3aed",
        }
    ).fillna("#64748b")
    ax.barh(positions, ordered["delta_head_outputs"], color=colors)
    ax.set_yticks(positions)
    ax.set_yticklabels(ordered["label"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("delta head outputs")
    ax.set_title("Representative marginal output-head costs")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_marginal_cost_value_pair(moves: pd.DataFrame, path: Path) -> None:
    representative = _representative_cost_moves(moves).copy()
    if representative.empty:
        return
    representative["label"] = representative.apply(_move_label, axis=1)
    representative = representative.sort_values(["delta_head_outputs", "label"])
    positions = np.arange(len(representative))
    colors = representative["move"].map(
        {
            "gain": "#16a34a",
            "offset": "#ca8a04",
            "offset_after_gain": "#a16207",
            "depth": "#2563eb",
            "width": "#7c3aed",
        }
    ).fillna("#64748b")
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.2, max(4.6, 0.44 * len(representative))),
        sharey=True,
        gridspec_kw={"width_ratios": [0.92, 1.22]},
    )
    axes[0].barh(positions, representative["delta_head_outputs"], color=colors)
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels(representative["label"], fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("extra head outputs")
    axes[0].set_title("Cost")
    axes[0].grid(axis="x", alpha=0.25)

    bar_height = 0.36
    axes[1].barh(
        positions - bar_height / 2,
        representative["median_efficiency"],
        bar_height,
        color="#2563eb",
        label="median",
    )
    axes[1].barh(
        positions + bar_height / 2,
        representative["p95_efficiency"],
        bar_height,
        color="#f97316",
        label="P95",
    )
    axes[1].axvline(0.0, color="#111827", linewidth=0.8)
    axes[1].set_xlabel("delta RMSE / extra head output; more negative is better")
    axes[1].set_title("Value")
    axes[1].grid(axis="x", alpha=0.25)
    axes[1].legend(loc="lower right")
    axes[1].text(
        0.02,
        0.98,
        "left = improves\nright = worsens",
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        fontsize=8,
        color="#475569",
        bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.86, "boxstyle": "round,pad=0.25"},
    )
    fig.suptitle("Marginal output-head tradeoffs from controlled Experiment 8 moves")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_marginal_efficiency(moves: pd.DataFrame, path: Path) -> None:
    finite = moves[moves["delta_head_outputs"] > 0].copy()
    if finite.empty:
        return
    finite["label"] = finite.apply(_move_label, axis=1)
    finite = finite.sort_values(["p95_efficiency", "median_efficiency", "label"]).head(12)
    height = max(4.8, 0.46 * len(finite))
    positions = np.arange(len(finite))
    fig, ax = plt.subplots(figsize=(9.6, height))
    bar_height = 0.36
    ax.barh(positions - bar_height / 2, finite["median_efficiency"], bar_height, label="median RMSE per output")
    ax.barh(positions + bar_height / 2, finite["p95_efficiency"], bar_height, label="P95 RMSE per output")
    ax.axvline(0.0, color="#111827", linewidth=0.8)
    ax.set_yticks(positions)
    ax.set_yticklabels(finite["label"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("delta RMSE / delta head outputs; more negative is better")
    ax.set_title("Best marginal quality value per output")
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_zero_cost_moves(moves: pd.DataFrame, path: Path) -> None:
    zero = moves[moves["delta_head_outputs"] == 0].copy()
    if zero.empty:
        return
    zero["label"] = zero.apply(_move_label, axis=1)
    zero = zero.sort_values(["delta_p95_rmse", "delta_median_rmse"])
    positions = np.arange(len(zero))
    fig, ax = plt.subplots(figsize=(10.5, max(3.8, 0.48 * len(zero))))
    bar_height = 0.36
    ax.barh(positions - bar_height / 2, zero["delta_median_rmse"], bar_height, label="median delta")
    ax.barh(positions + bar_height / 2, zero["delta_p95_rmse"], bar_height, label="P95 delta")
    ax.axvline(0.0, color="#111827", linewidth=0.8)
    ax.set_yticks(positions)
    ax.set_yticklabels(zero["label"], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("delta RMSE; negative is better")
    ax.set_title("Experiment 8 zero-output-cost decoder policy moves")
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _pareto_frontier(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    best = float("inf")
    ordered = frame.sort_values(["head_outputs", metric, "rmse_p95", "rmse_median"])
    for _, row in ordered.iterrows():
        if float(row[metric]) < best:
            rows.append(row)
            best = float(row[metric])
    return pd.DataFrame(rows)


def _frontier_table(frontier: pd.DataFrame) -> pd.DataFrame:
    table = frontier[
        [
            "residual_width",
            "residual_depth",
            "modifier_label",
            "residual_clip_policy",
            "cat_logits",
            "scalar_outputs",
            "head_outputs",
            "predicted_outputs",
            "rmse_median",
            "rmse_p95",
            "elapsed_seconds_total",
        ]
    ].copy()
    table.columns = [
        "W",
        "D",
        "modifier",
        "clip_policy",
        "cat_logits",
        "scalar_outputs",
        "head_outputs",
        "serialized_fields",
        "median_rmse",
        "p95_rmse",
        "seconds",
    ]
    return table


def _add_move(rows: list[dict[str, object]], move: str, before: pd.Series, after: pd.Series) -> None:
    output_delta = float(after.head_outputs - before.head_outputs)
    median_delta = float(after.rmse_median - before.rmse_median)
    p95_delta = float(after.rmse_p95 - before.rmse_p95)
    rows.append(
        {
            "move": move,
            "from_config": _config_label(before),
            "to_config": _config_label(after),
            "delta_head_outputs": output_delta,
            "delta_median_rmse": median_delta,
            "delta_p95_rmse": p95_delta,
            "median_efficiency": median_delta / output_delta if output_delta else np.nan,
            "p95_efficiency": p95_delta / output_delta if output_delta else np.nan,
        }
    )


def _build_marginal_moves(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for _, group in summary.groupby(["residual_depth", "modifier_label", "residual_clip_policy"], sort=False):
        ordered = group.sort_values("residual_width")
        for index in range(len(ordered) - 1):
            _add_move(rows, "width", ordered.iloc[index], ordered.iloc[index + 1])

    for _, group in summary.groupby(["residual_width", "modifier_label", "residual_clip_policy"], sort=False):
        ordered = group.sort_values("residual_depth")
        for index in range(len(ordered) - 1):
            _add_move(rows, "depth", ordered.iloc[index], ordered.iloc[index + 1])

    for _, group in summary.groupby(["residual_width", "residual_depth", "residual_clip_policy"], sort=False):
        by_modifier = {str(row.modifier_label): row for _, row in group.iterrows()}
        if "phase_only" in by_modifier and "phase_gain" in by_modifier:
            _add_move(rows, "gain", by_modifier["phase_only"], by_modifier["phase_gain"])
        if "phase_only" in by_modifier and "phase_offset" in by_modifier:
            _add_move(rows, "offset", by_modifier["phase_only"], by_modifier["phase_offset"])
        if "phase_gain" in by_modifier and "phase_gain_offset" in by_modifier:
            _add_move(rows, "offset_after_gain", by_modifier["phase_gain"], by_modifier["phase_gain_offset"])

    for _, group in summary.groupby(["residual_width", "residual_depth", "modifier_label"], sort=False):
        by_clip = {str(row.residual_clip_policy): row for _, row in group.iterrows()}
        if "final_only" in by_clip and "intermediate_m11_final_01" in by_clip:
            _add_move(rows, "clipping", by_clip["final_only"], by_clip["intermediate_m11_final_01"])

    moves = pd.DataFrame(rows)
    if not moves.empty:
        moves = moves.sort_values(["move", "from_config", "to_config"]).reset_index(drop=True)
    return moves


def _build_tables(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    phase = summary[(summary["modifier_label"] == "phase_only") & (summary["residual_clip_policy"] == "final_only")].copy()
    anchor = summary[
        (summary["residual_width"] == 12)
        & (summary["residual_depth"] == 16)
        & (summary["residual_clip_policy"] == "final_only")
    ].copy()
    clip = summary[
        (summary["residual_width"] == 12)
        & (summary["residual_depth"] == 16)
        & (summary["modifier_label"] == "phase_gain")
    ].copy()

    size_table = phase.sort_values(["rmse_p95", "rmse_median"]).head(12)[
        [
            "residual_width",
            "residual_depth",
            "cat_logits",
            "scalar_outputs",
            "head_outputs",
            "predicted_outputs",
            "rmse_median",
            "rmse_p95",
            "rmse_p99",
            "elapsed_seconds_total",
            "estimated_peak_memory_mb",
        ]
    ].copy()
    size_table.columns = [
        "W",
        "D",
        "cat_logits",
        "scalar_outputs",
        "head_outputs",
        "serialized_fields",
        "median_rmse",
        "p95_rmse",
        "p99_rmse",
        "seconds",
        "est_mem_mb",
    ]

    if not anchor.empty:
        order = ["phase_only", "phase_gain", "phase_offset", "phase_gain_offset"]
        anchor["modifier_label"] = pd.Categorical(anchor["modifier_label"], categories=order, ordered=True)
        anchor = anchor.sort_values("modifier_label")
        base = anchor[anchor["modifier_label"].astype(str) == "phase_only"].iloc[0]
        modifier_table = anchor[
            [
                "modifier_label",
                "cat_logits",
                "scalar_outputs",
                "head_outputs",
                "predicted_outputs",
                "rmse_median",
                "rmse_p95",
                "rmse_p99",
                "elapsed_seconds_total",
            ]
        ].copy()
        modifier_table["median_delta_vs_phase_only"] = modifier_table["rmse_median"] - base["rmse_median"]
        modifier_table["p95_delta_vs_phase_only"] = modifier_table["rmse_p95"] - base["rmse_p95"]
        modifier_table.columns = [
            "modifier",
            "cat_logits",
            "scalar_outputs",
            "head_outputs",
            "serialized_fields",
            "median_rmse",
            "p95_rmse",
            "p99_rmse",
            "seconds",
            "median_delta",
            "p95_delta",
        ]
    else:
        modifier_table = pd.DataFrame()

    if not clip.empty:
        order = ["final_only", "intermediate_m11_final_01"]
        clip["residual_clip_policy"] = pd.Categorical(clip["residual_clip_policy"], categories=order, ordered=True)
        clip = clip.sort_values("residual_clip_policy")
        final = clip[clip["residual_clip_policy"].astype(str) == "final_only"].iloc[0]
        clipping_table = clip[
            [
                "residual_clip_policy",
                "cat_logits",
                "scalar_outputs",
                "head_outputs",
                "predicted_outputs",
                "rmse_median",
                "rmse_p95",
                "rmse_p99",
                "elapsed_seconds_total",
            ]
        ].copy()
        clipping_table["median_delta_vs_final_only"] = clipping_table["rmse_median"] - final["rmse_median"]
        clipping_table["p95_delta_vs_final_only"] = clipping_table["rmse_p95"] - final["rmse_p95"]
        clipping_table.columns = [
            "clip_policy",
            "cat_logits",
            "scalar_outputs",
            "head_outputs",
            "serialized_fields",
            "median_rmse",
            "p95_rmse",
            "p99_rmse",
            "seconds",
            "median_delta",
            "p95_delta",
        ]
    else:
        clipping_table = pd.DataFrame()

    moves = _build_marginal_moves(summary)
    median_frontier = _pareto_frontier(summary, "rmse_median")
    p95_frontier = _pareto_frontier(summary, "rmse_p95")
    return (
        size_table,
        modifier_table,
        clipping_table,
        moves,
        _frontier_table(median_frontier),
        _frontier_table(p95_frontier),
    )


def _write_report(summary: pd.DataFrame) -> None:
    phase = summary[(summary["modifier_label"] == "phase_only") & (summary["residual_clip_policy"] == "final_only")].copy()
    size_table, modifier_table, clipping_table, moves, median_frontier_table, p95_frontier_table = _build_tables(summary)
    best_median = phase.sort_values(["rmse_median", "rmse_p95"]).iloc[0]
    best_p95 = phase.sort_values(["rmse_p95", "rmse_median"]).iloc[0]
    compact_deep = phase[(phase["residual_width"] == 8) & (phase["residual_depth"] == 32)].iloc[0]
    best_reference = phase[(phase["residual_width"] == 16) & (phase["residual_depth"] == 32)].iloc[0]
    memory_min = float(phase["estimated_peak_memory_mb"].min()) if "estimated_peak_memory_mb" in phase else float("nan")
    memory_max = float(phase["estimated_peak_memory_mb"].max()) if "estimated_peak_memory_mb" in phase else float("nan")

    zero_cost = moves[moves["delta_head_outputs"] == 0].sort_values("delta_p95_rmse").head(1)
    if zero_cost.empty:
        zero_cost_sentence = "No zero-output-cost decoder moves were available in this screen."
    else:
        zero_row = zero_cost.iloc[0]
        zero_cost_sentence = (
            f"The zero-output-cost decoder-policy move was `{_move_label(zero_row)}`: "
            f"median RMSE delta {zero_row.delta_median_rmse:.6g}, "
            f"P95 RMSE delta {zero_row.delta_p95_rmse:.6g}."
        )

    summary_lines = [
        (
            f"Best median RMSE is W{int(best_median.residual_width)}D{int(best_median.residual_depth)} "
            f"at {best_median.rmse_median:.6g} median / {best_median.rmse_p95:.6g} P95."
        ),
        (
            f"Best P95 RMSE is W{int(best_p95.residual_width)}D{int(best_p95.residual_depth)} "
            f"at {best_p95.rmse_p95:.6g} P95 / {best_p95.rmse_median:.6g} median."
        ),
        (
            f"W8D32 is the compact deep reference: P95 {compact_deep.rmse_p95:.6g} versus "
            f"W16D32 P95 {best_reference.rmse_p95:.6g}, with {int(compact_deep.head_outputs)} head outputs "
            f"versus {int(best_reference.head_outputs)}."
        ),
    ]
    if not modifier_table.empty:
        best_modifier = modifier_table.sort_values(["p95_rmse", "median_rmse"]).iloc[0]
        summary_lines.append(
            f"At W12D16, modifier winner by P95 is `{best_modifier.modifier}`: "
            f"median {best_modifier.median_rmse:.6g}, P95 {best_modifier.p95_rmse:.6g}."
        )
    if not clipping_table.empty:
        clipping_winner = clipping_table.sort_values(["p95_rmse", "median_rmse"]).iloc[0]
        summary_lines.append(
            f"For W12D16 phase+gain, clipping winner is `{clipping_winner.clip_policy}`: "
            f"median {clipping_winner.median_rmse:.6g}, P95 {clipping_winner.p95_rmse:.6g}."
        )

    report = f"""# Experiment 8 Findings

Experiment 8 is a 120-point, beam-4 screen on a fixed 1/3 train/validation sample. `W` is residual codebook width and `D` is actual residual-layer count. Phase is always enabled.

## Questions

- How much quality comes from width vs depth?
- Where is the useful `W x D` size band under a small screen budget?
- What is parameter-efficient in terms of output-head size the downstream model has to emit?
- Do gain and/or offset help once phase is always present?
- Does intermediate `[-1, 1]` clipping help before the final `[0, 1]` clip?

## Executive Read

{chr(10).join(f"- {line}" for line in summary_lines)}

The size screen strongly favors depth. The best configurations all use high `D`, and narrow deep models beat wide shallow models at comparable or smaller output-head size. Width helps, but only after enough residual layers are present.

## Output-Head Accounting

The corrected model-facing output-head size is:

```text
head_outputs = 32 + W*D + (D + 1) * (I_phase + I_gain + I_offset)
```

For Experiment 8, phase is mandatory:

```text
head_outputs = 33 + D(W + 1) + (D + 1)(I_gain + I_offset)
```

This is the intended interface cost: `32` base categorical choices, `W*D` residual categorical choices, and one scalar at the base plus one scalar per residual layer for each enabled modifier family. Optional gain and optional offset each cost `D + 1` outputs. Phase is included in the baseline and is not treated as a free design knob in the Experiment 8 plots.

`serialized_fields` is kept only as a storage/decoder count. It is not the neural output-head burden, because a categorical code index is emitted by a softmax over its codebook.

![Output accounting](images/experiment-08/experiment8_output_head_accounting.png)

## Size Screen

![Median and P95 heatmaps](images/experiment-08/experiment8_size_heatmap_pair.png)

D4 is kept in the tables and source CSV, but omitted from the scatter/ablation plots because its error is too far above the useful comparison range.

![Size screen colored by width](images/experiment-08/experiment8_size_rmse_vs_head_outputs_by_width.png)

![Size screen colored by depth](images/experiment-08/experiment8_size_rmse_vs_head_outputs_by_depth.png)

Top size jobs by P95:

{_markdown_table(size_table)}

## Modifier Screen

At W12D16, gain/offset were tested on top of phase. The deltas below are versus phase-only at the same W/D. Gain and offset have equal structural head cost, but their empirical quality deltas are separate facts.

![Modifier RMSE](images/experiment-08/experiment8_modifier_rmse_panels.png)

{_markdown_table(modifier_table)}

The modifier result is not a broad endorsement of more continuous outputs. Offset alone and gain+offset both degrade P95 relative to phase-only. Gain alone is effectively tied with phase-only in this final-only setting; its value appears mainly in the clipping test.

## Clipping Screen

Intermediate clipping was tested only for W12D16 with phase+gain. It has zero output-head cost, so its quality-per-output ratio is undefined rather than merely large.

![Clipping RMSE](images/experiment-08/experiment8_clipping_rmse_panels.png)

{_markdown_table(clipping_table)}

Intermediate `[-1, 1]` clipping improves both median and P95 versus phase+gain with final-only clipping in this screen.

## Marginal Cost And Value

Output cost is analytic. Quality value is empirical, measured with controlled finite differences between rows that differ by one design move where the screen contains such a pair. In the paired chart below, the left panel is the cost of asking the model to emit more outputs; the right panel is the observed RMSE change per added output. More negative is better.

![Marginal cost and value](images/experiment-08/experiment8_marginal_cost_value_pair.png)

{zero_cost_sentence}

The full row-level marginal table is saved as `analytics/marginal_efficiency.csv` for auditability, but it is not embedded here because the all-pairs version is visually noisy.

## Runtime And Memory

![Runtime scaling](images/experiment-08/experiment8_runtime_scaling_pair.png)

The measured elapsed time rises mostly with total residual work: more residual layers and wider dictionaries both cost. Estimated peak memory was not a decision driver in this screen: the phase-only size jobs ranged from {memory_min:.1f} MB to {memory_max:.1f} MB.

## Pareto Frontiers

For SOTA ML-style comparison, this is better treated as a rate-distortion / Pareto problem than as AIC or BIC. AIC/BIC require a likelihood model and count fitted statistical parameters; here the relevant cost is the output head the downstream model must emit.

The Pareto frontiers below keep only configurations where no smaller output-head model has equal or better error. Median and P95 are intentionally separate because they answer different modeling questions.

![Median output-head Pareto](images/experiment-08/experiment8_pareto_median_head_outputs.png)

Median RMSE output-head frontier:

{_markdown_table(median_frontier_table)}

![P95 output-head Pareto](images/experiment-08/experiment8_pareto_p95_head_outputs.png)

P95 RMSE output-head frontier:

{_markdown_table(p95_frontier_table)}

The efficient direction is not simply fewer outputs; deep enough models sharply reduce error. The practical trade is W8D32 versus W16D32: W8D32 is much cheaper in head outputs and runtime, while W16D32 is the quality leader.

## Working Recommendation

- Carry forward `topology_balanced_common_then_tail` with phase always enabled.
- Treat depth as the primary quality lever for the next experiment.
- Use W16D32 as the current best-quality reference from this screen.
- Keep W8D32 as the parameter-efficient deep-narrow reference.
- Include intermediate clipping with phase+gain in the next targeted test, because it is the only modifier/clipping variant that clearly improved W12D16.
- Do not carry offset forward unless a later targeted reason appears; it degraded P95 here.

## Files

- `analytics/summary.csv`
- `analytics/marginal_efficiency.csv`
- `analytics/results.csv`
- `analytics/thresholds.csv`
- `analytics/topology.csv`
- `analytics/usage.csv`
- `analytics/construction.csv`
- `analytics/paths.csv`
- `analytics/plots/`
"""
    REPORTS.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report, encoding="utf-8")


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    summary = _add_head_accounting(pd.read_csv(ANALYTICS / "summary.csv"))
    phase = summary[(summary["modifier_label"] == "phase_only") & (summary["residual_clip_policy"] == "final_only")].copy()
    anchor = summary[
        (summary["residual_width"] == 12)
        & (summary["residual_depth"] == 16)
        & (summary["residual_clip_policy"] == "final_only")
    ].copy()
    clip = summary[
        (summary["residual_width"] == 12)
        & (summary["residual_depth"] == 16)
        & (summary["modifier_label"] == "phase_gain")
    ].copy()
    moves = _build_marginal_moves(summary)
    moves.to_csv(ANALYTICS / "marginal_efficiency.csv", index=False)

    phase_sorted = phase.sort_values(["head_outputs", "residual_width"])
    phase_size_sorted = phase[phase["residual_depth"] != 4].sort_values(["head_outputs", "residual_width"])
    size_note = "D4 omitted; off-scale"
    _save_size_heatmap_pair(phase, PLOTS / "experiment8_size_heatmap_pair.png")
    _save_size_metric_pair(
        phase_size_sorted,
        "residual_width",
        "residual_width",
        "residual_width",
        "Experiment 8 size screen: RMSE vs output-head size, colored by width",
        PLOTS / "experiment8_size_rmse_vs_head_outputs_by_width.png",
        note=size_note,
    )
    _save_size_metric_pair(
        phase_size_sorted,
        "residual_depth",
        "residual_depth",
        "residual_depth",
        "Experiment 8 size screen: RMSE vs output-head size, colored by depth",
        PLOTS / "experiment8_size_rmse_vs_head_outputs_by_depth.png",
        note=size_note,
    )
    if not anchor.empty:
        order = ["phase_only", "phase_gain", "phase_offset", "phase_gain_offset"]
        anchor["modifier_label"] = pd.Categorical(anchor["modifier_label"], categories=order, ordered=True)
        anchor_plot = anchor.sort_values("modifier_label").copy()
        anchor_plot["is_baseline"] = anchor_plot["modifier_label"].astype(str) == "phase_only"
        _save_metric_panels(
            anchor_plot,
            "modifier_label",
            "W12D16 modifier screen",
            PLOTS / "experiment8_modifier_rmse_panels.png",
            baseline_column="is_baseline",
        )
    if not clip.empty:
        order = ["final_only", "intermediate_m11_final_01"]
        clip["residual_clip_policy"] = pd.Categorical(clip["residual_clip_policy"], categories=order, ordered=True)
        clip_plot = clip.sort_values("residual_clip_policy").copy()
        clip_plot["is_baseline"] = clip_plot["residual_clip_policy"].astype(str) == "final_only"
        _save_metric_panels(
            clip_plot,
            "residual_clip_policy",
            "W12D16 phase+gain clipping screen",
            PLOTS / "experiment8_clipping_rmse_panels.png",
            baseline_column="is_baseline",
        )
    _save_runtime_pair(phase_sorted, PLOTS / "experiment8_runtime_scaling_pair.png")
    _save_output_accounting(phase_sorted, PLOTS / "experiment8_output_head_accounting.png")
    _save_marginal_cost_value_pair(moves, PLOTS / "experiment8_marginal_cost_value_pair.png")
    _save_pareto(
        summary,
        _pareto_frontier(summary, "rmse_median"),
        "rmse_median",
        "Experiment 8 median RMSE output-head Pareto frontier",
        "median RMSE",
        PLOTS / "experiment8_pareto_median_head_outputs.png",
    )
    _save_pareto(
        summary,
        _pareto_frontier(summary, "rmse_p95"),
        "rmse_p95",
        "Experiment 8 P95 RMSE output-head Pareto frontier",
        "P95 RMSE",
        PLOTS / "experiment8_pareto_p95_head_outputs.png",
    )
    _write_report(summary)
    print(REPORT)


if __name__ == "__main__":
    main()

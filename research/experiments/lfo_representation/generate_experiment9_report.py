#!/usr/bin/env python3
"""Generate the Experiment 9 findings report and report-specific plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("artifacts/additive_finalization_9_screen")
EXPERIMENT8_ROOT = Path("artifacts/additive_finalization_8_screen")
ANALYTICS = ROOT / "analytics"
EXPERIMENT8_ANALYTICS = EXPERIMENT8_ROOT / "analytics"
REPORTS = Path("reports")
REPORT = REPORTS / "experiment-09-findings.md"
PLOTS = REPORTS / "images" / "experiment-09"

SECTION_ORDER = ["9A", "9B", "9C", "9D", "9D_ref"]
SCOPE_ORDER = ["base_only", "residuals_only", "base_and_residuals"]
MODULATION_ORDER = ["phase_gain", "phase_offset", "phase_gain_offset"]
NORMALIZATION_ORDER = ["raw", "range_normalized"]
CLIP_ORDER = [
    "final_only",
    "unipolar_guard_each_layer",
    "bipolar_guard_each_layer",
    "headroom_guard_025_each_layer",
    "headroom_guard_050_each_layer",
    "base_unipolar_residual_bipolar",
    "residual_depth_limiter_025",
    "residual_depth_limiter_050",
]
SNAP_ORDER = [
    "none",
    "data_snap_rails",
    "data_snap_dyadic_1",
    "data_snap_dyadic_2",
    "data_snap_dyadic_2_triadic_1",
]
PERFECT_EPS = 0.02
EXPECTED_JOB_COUNT = 39
W8_BUDGET_ANCHORS = [24, 32, 48, 64]


def _format_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float | np.floating):
        return f"{float(value):.6g}"
    return str(value)


def _markdown_table(frame: pd.DataFrame, *, max_rows: int | None = None) -> str:
    if frame is None or frame.empty:
        return "_No rows found._"
    if max_rows is not None:
        frame = frame.head(max_rows)
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.to_numpy():
        lines.append("| " + " | ".join(_format_value(value) for value in row) + " |")
    return "\n".join(lines)


def _ordered(frame: pd.DataFrame, column: str, order: list[str]) -> pd.DataFrame:
    result = frame.copy()
    result[column] = pd.Categorical(result[column].astype(str), categories=order, ordered=True)
    return result.sort_values(column)


def _add_accounting(summary: pd.DataFrame) -> pd.DataFrame:
    frame = summary.copy()
    frame["head_outputs"] = pd.to_numeric(frame.get("dense_outputs", 0), errors="coerce")
    frame["serialized_fields"] = pd.to_numeric(frame.get("predicted_outputs", 0), errors="coerce")
    for column in (
        "experiment9_section",
        "target_scope",
        "affine_modulation",
        "normalization_label",
        "decoder_hygiene_policy",
        "snap_policy",
        "budget_source",
    ):
        if column not in frame:
            frame[column] = ""
        frame[column] = frame[column].fillna("").astype(str)
    for column in (
        "train_rmse_median",
        "train_rmse_p95",
        "train_rmse_p99",
        "validation_rmse_median",
        "validation_rmse_p95",
        "validation_rmse_p99",
        "generalization_gap_p95",
        "gain_median",
        "gain_p95",
        "offset_abs_median",
        "offset_abs_p95",
        "gain_under_eps_rate",
        "snap_anchor_count",
        "snap_radius_median",
        "snap_changed_value_rate",
        "snap_mean_abs_delta",
        "all_eval_points_under_0.02",
        "perfect_lfo_rate_eps_0.02",
        "perfect_lfo_percent_eps_0.02",
        "perfect_lfo_count_eps_0.02",
        "budget_anchor_width",
        "budget_anchor_depth",
        "budget_anchor_head_outputs",
        "budget_actual_head_outputs",
    ):
        if column not in frame:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame["perfect_lfo_rate_eps_0.02"].isna().all() and "all_eval_points_under_0.02" in frame:
        frame["perfect_lfo_rate_eps_0.02"] = frame["all_eval_points_under_0.02"]
    frame["perfect_lfo_percent_eps_0.02"] = frame["perfect_lfo_rate_eps_0.02"] * 100.0
    if "shapes" in frame:
        frame["perfect_lfo_count_eps_0.02"] = frame["perfect_lfo_rate_eps_0.02"] * pd.to_numeric(frame["shapes"], errors="coerce")
    return frame


def _color_for_section(section: str) -> str:
    return {"9A": "#2563eb", "9B": "#16a34a", "9C": "#f97316", "9D": "#9333ea", "9D_ref": "#64748b"}.get(section, "#64748b")


def _load_experiment8_budget_references() -> pd.DataFrame:
    summary_path = EXPERIMENT8_ANALYTICS / "summary.csv"
    if not summary_path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(summary_path)
    required = {"residual_width", "residual_depth", "modifier_label", "residual_clip_policy"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    refs = frame[
        (pd.to_numeric(frame["residual_width"], errors="coerce") == 8)
        & (pd.to_numeric(frame["residual_depth"], errors="coerce").isin(W8_BUDGET_ANCHORS))
        & (frame["modifier_label"].astype(str) == "phase_only")
        & (frame["residual_clip_policy"].astype(str) == "final_only")
    ].copy()
    if refs.empty:
        return refs
    refs["experiment9_section"] = "9D_ref"
    refs["budget_source"] = "experiment8_reused"
    refs["budget_anchor_width"] = 8
    refs["budget_anchor_depth"] = pd.to_numeric(refs["residual_depth"], errors="coerce")
    refs["budget_anchor_head_outputs"] = refs["dense_outputs"]
    refs["budget_actual_head_outputs"] = refs["dense_outputs"]
    refs["perfect_lfo_rate_eps_0.02"] = pd.to_numeric(refs.get("all_eval_points_under_0.02", np.nan), errors="coerce")
    return refs


def _with_budget_references(summary: pd.DataFrame) -> pd.DataFrame:
    refs = _add_accounting(_load_experiment8_budget_references())
    if refs.empty:
        return summary
    for column in summary.columns:
        if column not in refs:
            refs[column] = np.nan
    for column in refs.columns:
        if column not in summary:
            summary[column] = np.nan
    return pd.concat([summary, refs[summary.columns]], ignore_index=True, sort=False)


def _save_9a_affine_grid(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        return
    metrics = [("rmse_median", "Median RMSE"), ("rmse_p95", "P95 RMSE")]
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.6), sharex=True, sharey=True)
    for row_index, (metric, metric_label) in enumerate(metrics):
        values = frame[metric].dropna()
        vmin = float(values.min()) if not values.empty else 0.0
        vmax = float(values.max()) if not values.empty else 1.0
        for col_index, normalization in enumerate(NORMALIZATION_ORDER):
            ax = axes[row_index, col_index]
            subset = frame[frame["normalization_label"].astype(str) == normalization]
            pivot = (
                subset.pivot_table(
                    index="target_scope",
                    columns="affine_modulation",
                    values=metric,
                    aggfunc="min",
                )
                .reindex(index=SCOPE_ORDER, columns=MODULATION_ORDER)
            )
            image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis")
            for r, scope in enumerate(SCOPE_ORDER):
                for c, modulation in enumerate(MODULATION_ORDER):
                    value = pivot.loc[scope, modulation]
                    if pd.isna(value):
                        continue
                    color = "white" if float(value) > (vmin + vmax) * 0.5 else "#111827"
                    ax.text(c, r, f"{float(value):.4f}", ha="center", va="center", fontsize=8, color=color)
            ax.set_title(f"{metric_label}, {normalization}")
            ax.set_xticks(np.arange(len(MODULATION_ORDER)))
            ax.set_xticklabels(MODULATION_ORDER, rotation=20, ha="right")
            ax.set_yticks(np.arange(len(SCOPE_ORDER)))
            ax.set_yticklabels(SCOPE_ORDER)
            if col_index == 0:
                ax.set_ylabel("affine target scope")
            colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            colorbar.set_label("RMSE")
    fig.suptitle("Experiment 9A affine and range-normalization screen")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_policy_metric_panels(frame: pd.DataFrame, x: str, order: list[str], title: str, path: Path) -> None:
    if frame.empty:
        return
    plot = _ordered(frame, x, order)
    labels = plot[x].astype(str).tolist()
    positions = np.arange(len(plot))
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
    for ax, metric, label, color in (
        (axes[0], "rmse_median", "Median RMSE", "#2563eb"),
        (axes[1], "rmse_p95", "P95 RMSE", "#f97316"),
    ):
        ax.plot(positions, plot[metric], color=color, marker="o", linewidth=1.8, markersize=6)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=24, ha="right", fontsize=8)
        ax.set_title(label)
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.25)
        values = plot[metric].astype(float).to_numpy()
        span = float(np.nanmax(values) - np.nanmin(values)) if len(values) else 0.0
        if span > 0:
            ax.set_ylim(max(0.0, float(np.nanmin(values)) - span * 0.35), float(np.nanmax(values)) + span * 0.45)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_train_validation(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty or frame["train_rmse_p95"].isna().all():
        return
    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    for section, group in frame.groupby("experiment9_section"):
        ax.scatter(
            group["train_rmse_p95"],
            group["validation_rmse_p95"],
            label=str(section),
            color=_color_for_section(str(section)),
            s=70,
            alpha=0.82,
        )
    limit = float(np.nanmax(frame[["train_rmse_p95", "validation_rmse_p95"]].to_numpy())) * 1.05
    ax.plot([0.0, limit], [0.0, limit], color="#111827", linewidth=1.0, linestyle="--", label="train = validation")
    ax.set_xlim(0.0, limit)
    ax.set_ylim(0.0, limit)
    ax.set_xlabel("train P95 RMSE")
    ax.set_ylabel("validation P95 RMSE")
    ax.set_title("Experiment 9 train vs validation P95")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_head_outputs(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0), sharex=True)
    for ax, metric, title in (
        (axes[0], "rmse_median", "Median RMSE"),
        (axes[1], "rmse_p95", "P95 RMSE"),
    ):
        for section, group in frame.groupby("experiment9_section"):
            ax.scatter(
                group["head_outputs"],
                group[metric],
                label=str(section),
                color=_color_for_section(str(section)),
                s=70,
                alpha=0.78,
            )
        ax.set_title(title)
        ax.set_xlabel("head outputs per LFO")
        ax.set_ylabel(title)
        ax.grid(alpha=0.25)
    axes[1].legend(title="section")
    fig.suptitle("Experiment 9 quality vs output-head burden")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_perfect_lfo_rate(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty or frame["perfect_lfo_rate_eps_0.02"].isna().all():
        return
    plot = frame.copy()
    plot["label"] = (
        plot["experiment9_section"].astype(str)
        + " "
        + plot.get("modifier_label", pd.Series("", index=plot.index)).astype(str)
    )
    plot = plot.sort_values(["experiment9_section", "perfect_lfo_rate_eps_0.02"], ascending=[True, False])
    fig, ax = plt.subplots(figsize=(12.2, max(4.8, 0.26 * len(plot))))
    colors = [_color_for_section(str(section)) for section in plot["experiment9_section"]]
    positions = np.arange(len(plot))
    ax.barh(positions, plot["perfect_lfo_percent_eps_0.02"], color=colors, alpha=0.86)
    ax.set_yticks(positions)
    ax.set_yticklabels(plot["label"].tolist(), fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(f"LFOs with every sampled point within +/-{PERFECT_EPS:g} (%)")
    ax.set_title("Experiment 9 perfect sampled-curve reconstruction rate")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_budget_equivalence(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        return
    subset = frame[frame["experiment9_section"].isin(["9D", "9D_ref"])].copy()
    if subset.empty:
        return
    subset["residual_width"] = pd.to_numeric(subset["residual_width"], errors="coerce")
    subset["residual_depth"] = pd.to_numeric(subset["residual_depth"], errors="coerce")
    subset["budget_anchor_depth"] = pd.to_numeric(subset["budget_anchor_depth"], errors="coerce")
    subset = subset.dropna(subset=["residual_width", "budget_anchor_depth"])
    if subset.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), sharex=True)
    metrics = [
        ("rmse_median", "Median RMSE"),
        ("rmse_p95", "P95 RMSE"),
        ("perfect_lfo_percent_eps_0.02", f"Perfect LFOs +/-{PERFECT_EPS:g} (%)"),
    ]
    palette = {4: "#2563eb", 6: "#16a34a", 8: "#64748b"}
    for ax, (metric, title) in zip(axes, metrics):
        for width, group in subset.groupby("residual_width"):
            group = group.sort_values("budget_anchor_depth")
            style = "--" if int(width) == 8 else "-"
            marker = "s" if int(width) == 8 else "o"
            ax.plot(
                group["budget_anchor_depth"],
                group[metric],
                label=f"W{int(width)}",
                color=palette.get(int(width), "#9333ea"),
                marker=marker,
                linestyle=style,
                linewidth=1.8,
            )
            for _, row in group.iterrows():
                if pd.notna(row.get("residual_depth")) and int(row.residual_width) != 8:
                    ax.annotate(
                        f"D{int(row.residual_depth)}",
                        (row.budget_anchor_depth, row[metric]),
                        fontsize=7,
                        xytext=(3, 3),
                        textcoords="offset points",
                    )
        ax.set_title(title)
        ax.set_xlabel("W8 equivalent anchor depth")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("error")
    axes[2].set_ylabel("coverage")
    axes[2].legend(title="tested width")
    fig.suptitle("Experiment 9D equivalent output-head budget screen")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_snap_diagnostics(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        return
    plot = _ordered(frame, "snap_policy", SNAP_ORDER)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
    positions = np.arange(len(plot))
    axes[0].plot(positions, plot["rmse_median"], color="#2563eb", marker="o", label="median")
    axes[0].plot(positions, plot["rmse_p95"], color="#f97316", marker="o", label="P95")
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(plot["snap_policy"].astype(str).tolist(), rotation=24, ha="right", fontsize=8)
    axes[0].set_ylabel("RMSE")
    axes[0].set_title("Snap policy quality")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()
    axes[1].scatter(plot["snap_changed_value_rate"], plot["rmse_p95"], s=70, color="#f97316", alpha=0.82)
    for _, row in plot.iterrows():
        axes[1].annotate(str(row.snap_policy), (row.snap_changed_value_rate, row.rmse_p95), fontsize=8, xytext=(4, 3), textcoords="offset points")
    axes[1].set_xlabel("snap changed-value rate")
    axes[1].set_ylabel("P95 RMSE")
    axes[1].set_title("Snap activity vs P95")
    axes[1].grid(alpha=0.25)
    fig.suptitle("Experiment 9C Snap Schwarzschild Radius")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _report_tables(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    columns = [
        "target_scope",
        "affine_modulation",
        "normalization_label",
        "head_outputs",
        "rmse_median",
        "rmse_p95",
        "perfect_lfo_percent_eps_0.02",
        "train_rmse_p95",
        "generalization_gap_p95",
        "gain_under_eps_rate",
    ]
    table_9a = summary[summary["experiment9_section"] == "9A"].sort_values(["rmse_p95", "rmse_median"])
    table_9a = table_9a[[column for column in columns if column in table_9a.columns]]

    columns_9b = [
        "decoder_hygiene_policy",
        "head_outputs",
        "rmse_median",
        "rmse_p95",
        "perfect_lfo_percent_eps_0.02",
        "train_rmse_p95",
        "generalization_gap_p95",
    ]
    table_9b = summary[summary["experiment9_section"] == "9B"].copy()
    if not table_9b.empty:
        table_9b = _ordered(table_9b, "decoder_hygiene_policy", CLIP_ORDER)
        table_9b = table_9b[[column for column in columns_9b if column in table_9b.columns]]

    columns_9c = [
        "snap_policy",
        "snap_anchor_count",
        "snap_radius_median",
        "snap_changed_value_rate",
        "snap_mean_abs_delta",
        "rmse_median",
        "rmse_p95",
        "perfect_lfo_percent_eps_0.02",
    ]
    table_9c = summary[summary["experiment9_section"] == "9C"].copy()
    if not table_9c.empty:
        table_9c = _ordered(table_9c, "snap_policy", SNAP_ORDER)
        table_9c = table_9c[[column for column in columns_9c if column in table_9c.columns]]
    columns_9d = [
        "budget_source",
        "residual_width",
        "residual_depth",
        "budget_anchor_width",
        "budget_anchor_depth",
        "head_outputs",
        "budget_anchor_head_outputs",
        "rmse_median",
        "rmse_p95",
        "perfect_lfo_percent_eps_0.02",
    ]
    table_9d = summary[summary["experiment9_section"].isin(["9D", "9D_ref"])].copy()
    if not table_9d.empty:
        table_9d = table_9d.sort_values(["budget_anchor_depth", "residual_width"])
        table_9d = table_9d[[column for column in columns_9d if column in table_9d.columns]]
    return table_9a, table_9b, table_9c, table_9d


def _write_report(summary: pd.DataFrame) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    if summary.empty:
        REPORT.write_text("# Experiment 9 Findings\n\nNo completed jobs yet.\n", encoding="utf-8")
        return

    table_9a, table_9b, table_9c, table_9d = _report_tables(summary)
    best = summary.sort_values(["rmse_p95", "rmse_median"]).iloc[0]
    best_9a = table_9a.iloc[0] if not table_9a.empty else None
    best_9b_source = summary[summary["experiment9_section"] == "9B"].sort_values(["rmse_p95", "rmse_median"])
    best_9c_source = summary[summary["experiment9_section"] == "9C"].sort_values(["rmse_p95", "rmse_median"])
    best_9b = best_9b_source.iloc[0] if not best_9b_source.empty else None
    best_9c = best_9c_source.iloc[0] if not best_9c_source.empty else None
    completed = int((summary["experiment9_section"] != "9D_ref").sum())

    lines = [
        f"Completed jobs in current analytics: {completed}/{EXPECTED_JOB_COUNT} excluding reused Experiment 8 reference rows.",
        (
            f"Overall best P95 is `{best.modifier_label}` in section `{best.experiment9_section}`: "
            f"median {best.rmse_median:.6g}, P95 {best.rmse_p95:.6g}, "
            f"perfect-rate {best['perfect_lfo_percent_eps_0.02']:.3g}%."
        ),
    ]
    if best_9a is not None:
        lines.append(
            f"9A best affine row: `{best_9a.target_scope}` / `{best_9a.affine_modulation}` / "
            f"`{best_9a.normalization_label}` with P95 {best_9a.rmse_p95:.6g}."
        )
    if best_9b is not None:
        lines.append(f"9B best decoder hygiene policy: `{best_9b.decoder_hygiene_policy}` with P95 {best_9b.rmse_p95:.6g}.")
    if best_9c is not None:
        lines.append(f"9C best snap policy: `{best_9c.snap_policy}` with P95 {best_9c.rmse_p95:.6g}.")

    report = f"""# Experiment 9 Findings

Experiment 9 is a quick fixed-budget W8D16 screen at 120-point evaluation resolution, beam 4, fixed 1/3 sample, and phase always enabled.

## Questions

- Do per-layer gain/offset scalars help when applied to base, residuals, or both?
- Does range normalization make those scalars useful?
- Which synth-style clipping or limiting policy should be the phase-only decoder baseline?
- Do data-derived snap anchors improve final output cheaply?
- At equivalent output-head budget, do very narrow/deep W4/W6 residual stacks beat W8 references?
- Do train and validation errors move together, or are any variants just fitting the train sample?

The third primary quality metric is `perfect_lfo_rate_eps_0.02`: the share of LFOs whose every sampled evaluation point is within +/-0.02 of the target curve. This is stricter than RMSE and different from editor-node preservation.

## Executive Read

{chr(10).join(f"- {line}" for line in lines)}

## 9A Affine And Normalization

![9A affine grid](images/experiment-09/experiment9_9a_affine_grid.png)

The heatmaps separate median and P95, and compare raw versus range-normalized targets on the same color scale for each metric.

{_markdown_table(table_9a, max_rows=8)}

## 9B Decoder Hygiene

![9B decoder hygiene](images/experiment-09/experiment9_9b_decoder_hygiene_panels.png)

{_markdown_table(table_9b)}

## 9C Snap Schwarzschild Radius

![9C snap diagnostics](images/experiment-09/experiment9_9c_snap_diagnostics.png)

{_markdown_table(table_9c)}

## 9D Equivalent Budget Narrow-Depth Screen

![9D equivalent budget](images/experiment-09/experiment9_9d_budget_equivalence.png)

W4 and W6 jobs are run at depths whose phase-only output-head count is closest to W8D24, W8D32, W8D48, and W8D64. W8D24 and W8D32 reference rows are reused from Experiment 8 analytics when available; W8D48 and W8D64 are budget anchors only unless those rows are later produced.

{_markdown_table(table_9d)}

## Perfect Reconstruction Rate

![Perfect LFO rate](images/experiment-09/experiment9_perfect_lfo_rate.png)

`perfect_lfo_rate_eps_0.02` is the fraction of validation LFOs with `max_abs_error <= 0.02` over the sampled evaluation grid.

## Train Vs Validation

![Train vs validation](images/experiment-09/experiment9_train_vs_validation_p95.png)

The diagonal is train P95 equals validation P95. Rows far above it are variants that look materially better on train than validation.

## Output-Head Accounting

![Head outputs](images/experiment-09/experiment9_head_outputs_vs_rmse.png)

`head_outputs` is categorical logits plus continuous scalar outputs per LFO. In 9A, gain/offset cost depends on whether the scalar family applies to base, residual layers, or both; clipping and snap policies do not add model outputs.

## Files

- `analytics/summary.csv`
- `analytics/results.csv`
- `analytics/thresholds.csv`
- `analytics/topology.csv`
- `analytics/usage.csv`
- `analytics/construction.csv`
- `analytics/paths.csv`
- `analytics/plots/`
"""
    REPORT.write_text(report, encoding="utf-8")


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    summary_path = ANALYTICS / "summary.csv"
    if not summary_path.exists():
        _write_report(pd.DataFrame())
        print(REPORT)
        return
    summary = _add_accounting(pd.read_csv(summary_path))
    summary.to_csv(ANALYTICS / "summary.csv", index=False)
    report_summary = _with_budget_references(summary.copy())
    section_9a = report_summary[report_summary["experiment9_section"] == "9A"].copy()
    section_9b = report_summary[report_summary["experiment9_section"] == "9B"].copy()
    section_9c = report_summary[report_summary["experiment9_section"] == "9C"].copy()
    section_9d = report_summary[report_summary["experiment9_section"].isin(["9D", "9D_ref"])].copy()
    _save_9a_affine_grid(section_9a, PLOTS / "experiment9_9a_affine_grid.png")
    _save_policy_metric_panels(
        section_9b,
        "decoder_hygiene_policy",
        CLIP_ORDER,
        "Experiment 9B decoder hygiene screen",
        PLOTS / "experiment9_9b_decoder_hygiene_panels.png",
    )
    _save_snap_diagnostics(section_9c, PLOTS / "experiment9_9c_snap_diagnostics.png")
    _save_budget_equivalence(section_9d, PLOTS / "experiment9_9d_budget_equivalence.png")
    _save_perfect_lfo_rate(report_summary[report_summary["experiment9_section"] != "9D_ref"], PLOTS / "experiment9_perfect_lfo_rate.png")
    _save_train_validation(report_summary, PLOTS / "experiment9_train_vs_validation_p95.png")
    _save_head_outputs(report_summary, PLOTS / "experiment9_head_outputs_vs_rmse.png")
    _write_report(report_summary)
    print(REPORT)


if __name__ == "__main__":
    main()

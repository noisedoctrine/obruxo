#!/usr/bin/env python3
"""Generate the Experiment 9 findings report and report-specific plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator
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
AFFINE_PLOT_ORDER = ["phase_only", "phase_gain", "phase_offset", "phase_gain_offset"]
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
FOCUSED_CLIP_ORDER = [
    "final_only",
    "unipolar_guard_each_layer",
    "bipolar_guard_each_layer",
    "base_unipolar_residual_bipolar",
]
SNAP_ORDER = [
    "none",
    "data_snap_rails",
    "data_snap_dyadic_1",
    "data_snap_dyadic_2",
    "data_snap_dyadic_2_triadic_1",
]
SCOPE_LABELS = {
    "base_only": "Base only",
    "residuals_only": "Residuals only",
    "base_and_residuals": "Base + residuals",
}
MODULATION_LABELS = {
    "phase_only": "None",
    "phase_gain": "Gain",
    "phase_offset": "Offset",
    "phase_gain_offset": "Gain + offset",
}
NORMALIZATION_LABELS = {
    "raw": "Raw",
    "range_normalized": "Range normalized",
}
CLIP_LABELS = {
    "final_only": "Final clip only",
    "unipolar_guard_each_layer": "Per-layer [0, 1]",
    "bipolar_guard_each_layer": "Per-layer [-1, 1]",
    "headroom_guard_025_each_layer": "Headroom +/-0.25",
    "headroom_guard_050_each_layer": "Headroom +/-0.50",
    "base_unipolar_residual_bipolar": "Base [0,1], residual [-1,1]",
    "residual_depth_limiter_025": "Clamp residual +/-0.25",
    "residual_depth_limiter_050": "Clamp residual +/-0.50",
}
SNAP_LABELS = {
    "none": "No snap",
    "data_snap_rails": "Rails",
    "data_snap_dyadic_1": "Rails + 1/2",
    "data_snap_dyadic_2": "Quarters",
    "data_snap_dyadic_2_triadic_1": "Quarters + thirds",
}
SECTION_LABELS = {
    "9A": "9A affine",
    "9B": "9B clipping",
    "9C": "9C snap",
    "9D": "9D narrow/deep",
    "9D_ref": "Exp 8 W8 ref",
}
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


def _display_label(value: object, labels: dict[str, str]) -> str:
    return labels.get(str(value), str(value))


def _topology_conditioned(frame: pd.DataFrame) -> pd.Series:
    if "topology_dependency" in frame:
        values = frame["topology_dependency"]
        if values.dtype == bool:
            return values.fillna(False).astype(bool)
        return values.astype(str).str.lower().isin(["true", "1", "yes"])
    if "construction_strategy" in frame:
        return frame["construction_strategy"].astype(str).str.contains("topology")
    return pd.Series(False, index=frame.index)


def _flattened_residual_logits(width: pd.Series, depth: pd.Series, topology_conditioned: pd.Series) -> pd.Series:
    shared_layers = (depth + 1) // 2
    topology_layers = depth // 2
    flattened_topology = width * (shared_layers + 3 * topology_layers)
    shared_only = width * depth
    return flattened_topology.where(topology_conditioned.astype(bool), shared_only)


def _phase_only_head_outputs(width: pd.Series | float | int, depth: pd.Series | float | int) -> pd.Series | float:
    if isinstance(width, pd.Series) or isinstance(depth, pd.Series):
        width_series = pd.to_numeric(width, errors="coerce")
        depth_series = pd.to_numeric(depth, errors="coerce")
        return 32 + _flattened_residual_logits(width_series, depth_series, pd.Series(True, index=width_series.index)) + (depth_series + 1)
    width_int = int(width)
    depth_int = int(depth)
    shared_layers = (depth_int + 1) // 2
    topology_layers = depth_int // 2
    return float(32 + width_int * (shared_layers + 3 * topology_layers) + depth_int + 1)


def _add_accounting(summary: pd.DataFrame) -> pd.DataFrame:
    frame = summary.copy()
    width = pd.to_numeric(frame.get("residual_width", np.nan), errors="coerce")
    depth = pd.to_numeric(frame.get("residual_depth", np.nan), errors="coerce")
    topology_conditioned = _topology_conditioned(frame)
    frame["residual_logits"] = _flattened_residual_logits(width, depth, topology_conditioned)
    if "continuous_scalars" in frame:
        frame["scalar_outputs"] = pd.to_numeric(frame["continuous_scalars"], errors="coerce")
    else:
        frame["scalar_outputs"] = pd.to_numeric(frame.get("continuous_outputs", np.nan), errors="coerce")
    frame["head_outputs"] = 32 + frame["residual_logits"] + frame["scalar_outputs"]
    if "categorical_logits" in frame and "continuous_scalars" in frame:
        frame["legacy_dense_outputs"] = (
            pd.to_numeric(frame["categorical_logits"], errors="coerce")
            + pd.to_numeric(frame["continuous_scalars"], errors="coerce")
        )
    else:
        frame["legacy_dense_outputs"] = pd.to_numeric(frame.get("dense_outputs", np.nan), errors="coerce")
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
        "implementation_anchor_head_outputs",
        "head_delta_vs_anchor",
        "wd_product",
        "phase_scalar_outputs",
        "residual_logits",
        "scalar_outputs",
        "legacy_dense_outputs",
        "elapsed_seconds_total",
    ):
        if column not in frame:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["wd_product"] = width * depth
    frame["phase_scalar_outputs"] = depth + 1
    anchor_width = pd.to_numeric(frame.get("budget_anchor_width", np.nan), errors="coerce")
    anchor_depth = pd.to_numeric(frame.get("budget_anchor_depth", np.nan), errors="coerce")
    frame["implementation_anchor_head_outputs"] = _phase_only_head_outputs(anchor_width, anchor_depth)
    frame["head_delta_vs_anchor"] = frame["head_outputs"] - frame["implementation_anchor_head_outputs"]
    budget_rows = anchor_depth.notna() & (anchor_depth > 0)
    frame.loc[budget_rows, "budget_anchor_head_outputs"] = frame.loc[budget_rows, "implementation_anchor_head_outputs"]
    frame.loc[budget_rows, "budget_actual_head_outputs"] = frame.loc[budget_rows, "head_outputs"]
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
    corrected_heads = _phase_only_head_outputs(refs["budget_anchor_width"], refs["budget_anchor_depth"])
    refs["head_outputs"] = corrected_heads
    refs["budget_anchor_head_outputs"] = corrected_heads
    refs["budget_actual_head_outputs"] = corrected_heads
    refs["implementation_anchor_head_outputs"] = corrected_heads
    refs["head_delta_vs_anchor"] = 0
    refs["perfect_lfo_rate_eps_0.02"] = pd.to_numeric(refs.get("all_eval_points_under_0.02", np.nan), errors="coerce")
    return refs


def _with_9a_phase_only_baseline(summary: pd.DataFrame) -> pd.DataFrame:
    baseline = summary[
        (summary["experiment9_section"].astype(str) == "9B")
        & (summary["decoder_hygiene_policy"].astype(str) == "bipolar_guard_each_layer")
    ]
    if baseline.empty:
        return summary
    template = baseline.iloc[0]
    rows = []
    for scope in SCOPE_ORDER:
        for normalization in NORMALIZATION_ORDER:
            row = template.copy()
            row["experiment9_section"] = "9A"
            row["target_scope"] = scope
            row["affine_modulation"] = "phase_only"
            row["normalization_label"] = normalization
            row["modifier_label"] = "phase_only_baseline"
            rows.append(row)
    return pd.concat([summary, pd.DataFrame(rows)], ignore_index=True, sort=False)


def _with_budget_references(summary: pd.DataFrame) -> pd.DataFrame:
    refs = _add_accounting(_load_experiment8_budget_references())
    if refs.empty:
        return summary
    ref_width = pd.to_numeric(refs["residual_width"], errors="coerce")
    ref_depth = pd.to_numeric(refs["residual_depth"], errors="coerce")
    ref_heads = _phase_only_head_outputs(ref_width, ref_depth)
    refs["head_outputs"] = ref_heads
    refs["implementation_anchor_head_outputs"] = ref_heads
    refs["head_delta_vs_anchor"] = 0
    missing_in_refs = [column for column in summary.columns if column not in refs]
    if missing_in_refs:
        refs = pd.concat([refs, pd.DataFrame(np.nan, index=refs.index, columns=missing_in_refs)], axis=1)
    missing_in_summary = [column for column in refs.columns if column not in summary]
    if missing_in_summary:
        summary = pd.concat([summary, pd.DataFrame(np.nan, index=summary.index, columns=missing_in_summary)], axis=1)
    return pd.concat([summary, refs[summary.columns]], ignore_index=True, sort=False)


def _save_9a_affine_grid(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        return
    metrics = [("rmse_median", "Median RMSE"), ("rmse_p95", "P95 RMSE")]
    row_keys: list[tuple[str, str]] = []
    row_labels: list[str] = []
    for scope in SCOPE_ORDER:
        for normalization in NORMALIZATION_ORDER:
            row_keys.append((scope, normalization))
            row_labels.append(f"{SCOPE_LABELS[scope]}\n{NORMALIZATION_LABELS[normalization]}")
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.35), sharey=True)
    for ax, (metric, metric_label) in zip(axes, metrics):
        matrix = np.full((len(row_keys), len(AFFINE_PLOT_ORDER)), np.nan)
        percent_matrix = np.full_like(matrix, np.nan)
        for r, (scope, normalization) in enumerate(row_keys):
            subset = frame[
                (frame["target_scope"].astype(str) == scope)
                & (frame["normalization_label"].astype(str) == normalization)
            ]
            baseline_values = subset[subset["affine_modulation"].astype(str) == "phase_only"][metric]
            baseline = float(baseline_values.iloc[0]) if not baseline_values.empty else np.nan
            for c, modulation in enumerate(AFFINE_PLOT_ORDER):
                value = subset[subset["affine_modulation"].astype(str) == modulation][metric]
                if not value.empty:
                    delta = float(value.iloc[0]) - baseline
                    matrix[r, c] = delta
                    if baseline and np.isfinite(baseline):
                        percent_matrix[r, c] = 100.0 * delta / baseline
        max_abs = float(np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else 1.0
        norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
        image = ax.imshow(matrix, aspect="auto", norm=norm, cmap="RdBu_r")
        for r in range(matrix.shape[0]):
            for c in range(matrix.shape[1]):
                value = matrix[r, c]
                if np.isnan(value):
                    continue
                pct = percent_matrix[r, c]
                text = f"{value:+.4f}\n({pct:+.0f}%)" if np.isfinite(pct) else f"{value:+.4f}"
                color = "#111827" if abs(value) < max_abs * 0.55 else "white"
                ax.text(c, r, text, ha="center", va="center", fontsize=7, color=color)
        ax.set_title(f"{metric_label} delta vs baseline")
        ax.set_xticks(np.arange(len(AFFINE_PLOT_ORDER)))
        ax.set_xticklabels([MODULATION_LABELS[item] for item in AFFINE_PLOT_ORDER], rotation=0)
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels)
        ax.set_xlabel("extra scalar family")
        ax.grid(False)
        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label("RMSE delta; blue is better")
    axes[0].set_ylabel("where scalars apply / target normalization")
    fig.suptitle("9A: per-layer affine scalars at fixed W8D16")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_policy_metric_panels(frame: pd.DataFrame, x: str, order: list[str], title: str, path: Path) -> None:
    if frame.empty:
        return
    plot = _ordered(frame, x, order)
    plot = plot[plot[x].astype(str).isin(FOCUSED_CLIP_ORDER)].copy()
    labels = [_display_label(value, CLIP_LABELS) for value in plot[x]]
    positions = np.arange(len(plot))
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.1))
    for ax, metric, label, color in (
        (axes[0], "rmse_median", "Median RMSE", "#2563eb"),
        (axes[1], "rmse_p95", "P95 RMSE", "#f97316"),
    ):
        ax.barh(positions, plot[metric], color=color, alpha=0.86)
        ax.set_yticks(positions)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(label)
        ax.set_xlabel(label)
        ax.grid(axis="x", alpha=0.25)
        values = plot[metric].astype(float).to_numpy()
        span = float(np.nanmax(values) - np.nanmin(values)) if len(values) else 0.0
        if span > 0:
            ax.set_xlim(max(0.0, float(np.nanmin(values)) - span * 0.35), float(np.nanmax(values)) + span * 0.45)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_train_validation(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty or frame["train_rmse_p95"].isna().all():
        return
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for section, group in frame.groupby("experiment9_section"):
        ax.scatter(
            group["train_rmse_p95"],
            group["validation_rmse_p95"],
            label=_display_label(section, SECTION_LABELS),
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


def _save_generalization_gap(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty or frame["generalization_gap_p95"].isna().all():
        return
    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    for section, group in frame.groupby("experiment9_section"):
        ax.scatter(
            group["head_outputs"],
            group["generalization_gap_p95"],
            label=_display_label(section, SECTION_LABELS),
            color=_color_for_section(str(section)),
            s=52,
            alpha=0.82,
        )
    ax.axhline(0.0, color="#111827", linewidth=0.9, linestyle="--")
    ax.set_xlabel("deployed head outputs per LFO")
    ax.set_ylabel("validation P95 - train P95")
    ax.set_title("Generalization gap vs output-head size")
    ax.grid(alpha=0.25)
    ax.legend(title="screen", fontsize=7, title_fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
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
                label=_display_label(section, SECTION_LABELS),
                color=_color_for_section(str(section)),
                s=70,
                alpha=0.78,
            )
        ax.set_title(title)
        ax.set_xlabel("head outputs per LFO")
        ax.set_ylabel(title)
        ax.grid(alpha=0.25)
    axes[1].legend(title="screen")
    fig.suptitle("Experiment 9 quality vs output-head burden")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_runtime_summary(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty or "elapsed_seconds_total" not in frame:
        return
    data = frame[frame["elapsed_seconds_total"].notna()].copy()
    if data.empty:
        return
    data["elapsed_minutes"] = pd.to_numeric(data["elapsed_seconds_total"], errors="coerce") / 60.0
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.8))

    section_9a = data[data["experiment9_section"] == "9A"].copy()
    baseline_9a = data[
        (data["experiment9_section"] == "9B")
        & (data["decoder_hygiene_policy"].astype(str) == "bipolar_guard_each_layer")
    ]
    ax = axes[0, 0]
    if not section_9a.empty and not baseline_9a.empty:
        baseline_minutes = float(baseline_9a.iloc[0]["elapsed_minutes"])
        row_keys = [(scope, normalization) for scope in SCOPE_ORDER for normalization in NORMALIZATION_ORDER]
        row_labels = [f"{SCOPE_LABELS[scope]}\n{NORMALIZATION_LABELS[normalization]}" for scope, normalization in row_keys]
        matrix = np.full((len(row_keys), len(MODULATION_ORDER)), np.nan)
        for r, (scope, normalization) in enumerate(row_keys):
            subset = section_9a[
                (section_9a["target_scope"].astype(str) == scope)
                & (section_9a["normalization_label"].astype(str) == normalization)
            ]
            for c, modulation in enumerate(MODULATION_ORDER):
                value = subset[subset["affine_modulation"].astype(str) == modulation]["elapsed_minutes"]
                if not value.empty:
                    matrix[r, c] = float(value.iloc[0]) - baseline_minutes
        max_abs = float(np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else 1.0
        norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
        image = ax.imshow(matrix, aspect="auto", norm=norm, cmap="RdBu_r")
        for r in range(matrix.shape[0]):
            for c in range(matrix.shape[1]):
                if np.isfinite(matrix[r, c]):
                    ax.text(c, r, f"{matrix[r, c]:+.2f}", ha="center", va="center", fontsize=7)
        ax.set_xticks(np.arange(len(MODULATION_ORDER)))
        ax.set_xticklabels([MODULATION_LABELS[item] for item in MODULATION_ORDER])
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=8)
        ax.set_title("9A affine vs phase-only [-1,1]")
        ax.set_xlabel("extra scalar family")
        ax.set_ylabel("scope / normalization")
        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label("delta minutes")

    def runtime_bars(
        ax: plt.Axes,
        plot: pd.DataFrame,
        label_column: str,
        baseline_filter: pd.Series,
        labels: dict[str, str],
        title: str,
        *,
        neutral: bool = False,
    ) -> None:
        if plot.empty or not baseline_filter.any():
            return
        baseline_minutes = float(plot[baseline_filter].iloc[0]["elapsed_minutes"])
        bars = plot.copy()
        bars["runtime_delta_minutes"] = bars["elapsed_minutes"] - baseline_minutes
        bars["display_label"] = bars[label_column].map(lambda value: _display_label(value, labels))
        bars = bars.sort_values("runtime_delta_minutes")
        colors = (
            ["#94a3b8"] * len(bars)
            if neutral
            else ["#2563eb" if value < 0 else "#f97316" if value > 0 else "#64748b" for value in bars["runtime_delta_minutes"]]
        )
        positions = np.arange(len(bars))
        ax.barh(positions, bars["runtime_delta_minutes"], color=colors, alpha=0.85)
        ax.axvline(0.0, color="#111827", linewidth=0.9)
        ax.set_yticks(positions)
        ax.set_yticklabels(bars["display_label"], fontsize=8)
        ax.set_xlabel("delta minutes vs baseline")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)

    section_9b = _ordered(data[data["experiment9_section"] == "9B"].copy(), "decoder_hygiene_policy", CLIP_ORDER)
    runtime_bars(
        axes[0, 1],
        section_9b,
        "decoder_hygiene_policy",
        section_9b["decoder_hygiene_policy"].astype(str) == "final_only",
        CLIP_LABELS,
        "9B clipping vs final clip only (noise)",
        neutral=True,
    )
    axes[0, 1].text(
        0.02,
        0.96,
        "same operation count\nbars reflect job noise, not policy speed",
        transform=axes[0, 1].transAxes,
        fontsize=8,
        color="#374151",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.85, "pad": 3.0},
    )

    section_9c = _ordered(data[data["experiment9_section"] == "9C"].copy(), "snap_policy", SNAP_ORDER)
    runtime_bars(
        axes[1, 0],
        section_9c,
        "snap_policy",
        section_9c["snap_policy"].astype(str) == "none",
        SNAP_LABELS,
        "9C snap vs no snap",
    )

    ax = axes[1, 1]
    section_9d = data[data["experiment9_section"].isin(["9D", "9D_ref"])].copy()
    if not section_9d.empty:
        refs = section_9d[section_9d["experiment9_section"] == "9D_ref"].set_index("budget_anchor_depth")
        rows = []
        for _, row in section_9d[section_9d["experiment9_section"] == "9D"].iterrows():
            anchor_depth = row.get("budget_anchor_depth")
            if pd.isna(anchor_depth) or anchor_depth not in refs.index:
                continue
            baseline_minutes = float(refs.loc[anchor_depth]["elapsed_minutes"])
            rows.append(
                {
                    "label": f"W{int(row.residual_width)}D{int(row.residual_depth)} vs W8D{int(anchor_depth)}",
                    "delta": float(row.elapsed_minutes - baseline_minutes),
                }
            )
        bars = pd.DataFrame(rows)
        if not bars.empty:
            bars = bars.sort_values("delta")
            positions = np.arange(len(bars))
            ax.barh(positions, bars["delta"], color="#9333ea", alpha=0.82)
            ax.axvline(0.0, color="#111827", linewidth=0.9)
            ax.set_yticks(positions)
            ax.set_yticklabels(bars["label"], fontsize=8)
            ax.set_xlabel("delta minutes vs W8 anchor")
        ax.set_title("9D narrow/deep vs W8 reference")
        ax.grid(axis="x", alpha=0.25)
        ax.text(
            0.02,
            0.03,
            "W8D48/W8D64 runtime baselines were not run.",
            transform=ax.transAxes,
            fontsize=8,
            color="#374151",
        )

    fig.suptitle("Experiment 9 runtime impact by tested dimension")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_perfect_lfo_rate(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty or frame["perfect_lfo_rate_eps_0.02"].isna().all():
        return
    plot = frame.copy()
    fig, ax = plt.subplots(figsize=(5.8, 3.7))
    for section, group in plot.groupby("experiment9_section"):
        ax.scatter(
            group["perfect_lfo_percent_eps_0.02"],
            group["rmse_p95"],
            label=_display_label(section, SECTION_LABELS),
            color=_color_for_section(str(section)),
            s=46,
            alpha=0.82,
        )
    best_by_section = plot.sort_values("rmse_p95").groupby("experiment9_section", as_index=False).head(1)
    snap_refs = plot[
        (plot["experiment9_section"].astype(str) == "9C")
        & plot.get("snap_policy", pd.Series("", index=plot.index)).astype(str).isin(["none", "data_snap_rails"])
    ]
    interesting = pd.concat(
        [
            plot.sort_values("perfect_lfo_percent_eps_0.02", ascending=False).head(2),
            plot.sort_values("rmse_p95", ascending=True).head(2),
            best_by_section,
            snap_refs,
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["configuration"], keep="first")
    for annotation_i, (_, row) in enumerate(interesting.iterrows()):
        label = str(row.get("modifier_label", ""))
        if str(row.get("experiment9_section")) == "9C":
            label = _display_label(row.get("snap_policy"), SNAP_LABELS)
        elif str(row.get("experiment9_section")) == "9B":
            label = "Best clipping"
        elif str(row.get("experiment9_section")) == "9A":
            label = "Best affine"
        elif str(row.get("experiment9_section")) == "9D":
            label = f"W{int(row.residual_width)}D{int(row.residual_depth)}"
        x_offset = -46 if float(row["perfect_lfo_percent_eps_0.02"]) > 92.0 else 4
        y_offset = 5 if annotation_i % 2 == 0 else -10
        ax.annotate(
            label,
            (row["perfect_lfo_percent_eps_0.02"], row["rmse_p95"]),
            fontsize=6,
            xytext=(x_offset, y_offset),
            textcoords="offset points",
        )
    ax.set_xlabel(f"perfect LFOs within +/-{PERFECT_EPS:g} at every sampled point (%)")
    ax.set_ylabel("P95 RMSE")
    ax.set_title("Strict pass rate versus tail error")
    ax.invert_yaxis()
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=8)
    ax.legend(title="screen", fontsize=7, title_fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
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
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.8))
    positions = np.arange(len(plot))
    labels = [_display_label(value, SNAP_LABELS) for value in plot["snap_policy"]]
    panels = [
        ("rmse_median", "Median RMSE", "#2563eb"),
        ("rmse_p95", "P95 RMSE", "#f97316"),
        ("perfect_lfo_percent_eps_0.02", f"Perfect LFOs +/-{PERFECT_EPS:g} (%)", "#16a34a"),
    ]
    for ax, (metric, title, color) in zip(axes, panels):
        ax.bar(positions, plot[metric], color=color, alpha=0.86)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=24, ha="right", fontsize=8)
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        values = plot[metric].astype(float).to_numpy()
        span = float(np.nanmax(values) - np.nanmin(values)) if len(values) else 0.0
        if span > 0 and metric != "perfect_lfo_percent_eps_0.02":
            ax.set_ylim(max(0.0, float(np.nanmin(values)) - span * 0.35), float(np.nanmax(values)) + span * 0.45)
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
        "wd_product",
        "phase_scalar_outputs",
        "budget_anchor_width",
        "budget_anchor_depth",
        "head_outputs",
        "implementation_anchor_head_outputs",
        "head_delta_vs_anchor",
        "rmse_median",
        "rmse_p95",
        "perfect_lfo_percent_eps_0.02",
    ]
    table_9d = summary[summary["experiment9_section"].isin(["9D", "9D_ref"])].copy()
    if not table_9d.empty:
        table_9d = table_9d.sort_values(["budget_anchor_depth", "residual_width"])
        table_9d = table_9d[[column for column in columns_9d if column in table_9d.columns]]
        table_9d = table_9d.rename(
            columns={
                "implementation_anchor_head_outputs": "anchor_head_outputs",
                "head_delta_vs_anchor": "head_delta_vs_anchor",
            }
        )
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
    section_9b = summary[summary["experiment9_section"] == "9B"].copy()
    section_9c = summary[summary["experiment9_section"] == "9C"].copy()
    section_9d = summary[summary["experiment9_section"].isin(["9D", "9D_ref"])].copy()
    final_9b = section_9b[section_9b["decoder_hygiene_policy"] == "final_only"]
    bipolar_9b = section_9b[section_9b["decoder_hygiene_policy"] == "bipolar_guard_each_layer"]
    unipolar_9b = section_9b[section_9b["decoder_hygiene_policy"] == "unipolar_guard_each_layer"]
    snap_none = section_9c[section_9c["snap_policy"] == "none"]
    snap_rails = section_9c[section_9c["snap_policy"] == "data_snap_rails"]
    budget_w4 = section_9d[(section_9d["experiment9_section"] == "9D") & (section_9d["residual_width"] == 4)]
    budget_w6 = section_9d[(section_9d["experiment9_section"] == "9D") & (section_9d["residual_width"] == 6)]
    runtime_rows = summary[(summary["experiment9_section"] != "9D_ref") & summary["elapsed_seconds_total"].notna()].copy()
    runtime_9a = runtime_rows[runtime_rows["experiment9_section"] == "9A"].copy()
    runtime_9b = runtime_rows[runtime_rows["experiment9_section"] == "9B"].copy()
    runtime_9c = runtime_rows[runtime_rows["experiment9_section"] == "9C"].copy()
    runtime_9d = summary[summary["experiment9_section"].isin(["9D", "9D_ref"]) & summary["elapsed_seconds_total"].notna()].copy()
    baseline_9a = runtime_9b[runtime_9b["decoder_hygiene_policy"] == "bipolar_guard_each_layer"]
    baseline_9b = runtime_9b[runtime_9b["decoder_hygiene_policy"] == "final_only"]
    baseline_9c = runtime_9c[runtime_9c["snap_policy"] == "none"]

    def runtime_delta_range(rows: pd.DataFrame, baseline: pd.DataFrame) -> tuple[float, float]:
        if rows.empty or baseline.empty:
            return float("nan"), float("nan")
        deltas = pd.to_numeric(rows["elapsed_seconds_total"], errors="coerce") / 60.0 - float(baseline.iloc[0]["elapsed_seconds_total"] / 60.0)
        return float(deltas.min()), float(deltas.max())

    runtime_9a_min, runtime_9a_max = runtime_delta_range(runtime_9a, baseline_9a)
    runtime_9b_min, runtime_9b_max = runtime_delta_range(runtime_9b, baseline_9b)
    runtime_9c_min, runtime_9c_max = runtime_delta_range(runtime_9c, baseline_9c)
    runtime_rails_delta = (
        float((snap_rails.iloc[0]["elapsed_seconds_total"] - snap_none.iloc[0]["elapsed_seconds_total"]) / 60.0)
        if not snap_rails.empty and not snap_none.empty
        else float("nan")
    )

    def runtime_9d_delta_rows() -> pd.DataFrame:
        if runtime_9d.empty:
            return pd.DataFrame()
        refs = runtime_9d[runtime_9d["experiment9_section"] == "9D_ref"].set_index("budget_anchor_depth")
        rows = []
        for _, row in runtime_9d[runtime_9d["experiment9_section"] == "9D"].iterrows():
            anchor_depth = row.get("budget_anchor_depth")
            if pd.isna(anchor_depth) or anchor_depth not in refs.index:
                continue
            rows.append(
                {
                    "label": f"W{int(row.residual_width)}D{int(row.residual_depth)} vs W8D{int(anchor_depth)}",
                    "delta_minutes": float((row.elapsed_seconds_total - refs.loc[anchor_depth].elapsed_seconds_total) / 60.0),
                }
            )
        return pd.DataFrame(rows)

    runtime_9d_deltas = runtime_9d_delta_rows()
    runtime_9d_min, runtime_9d_max = (
        (float(runtime_9d_deltas["delta_minutes"].min()), float(runtime_9d_deltas["delta_minutes"].max()))
        if not runtime_9d_deltas.empty
        else (float("nan"), float("nan"))
    )

    def metric_delta(row: pd.DataFrame, baseline: pd.DataFrame, column: str) -> float:
        if row.empty or baseline.empty:
            return float("nan")
        return float(row.iloc[0][column] - baseline.iloc[0][column])

    lines = [
        f"Completed jobs in current analytics: {completed}/{EXPECTED_JOB_COUNT} excluding reused Experiment 8 reference rows.",
        (
            f"Overall best P95 is `{best.modifier_label}` in section `{best.experiment9_section}`: "
            f"median {best.rmse_median:.6g}, P95 {best.rmse_p95:.6g}, "
            f"perfect-rate {best['perfect_lfo_percent_eps_0.02']:.3g}%."
        ),
        (
            "The strongest new signal is the equivalent-budget narrow/deep result: W4/W6 at larger depth "
            "substantially beat the W8D24 and W8D32 Experiment 8 references on P95 at similar output-head size."
        ),
        (
            "Clipping strategy is not settled by a single metric: unipolar per-layer clipping has the best 9B P95, "
            "while bipolar per-layer clipping has the best 9B median and is identical to base-unipolar/residual-bipolar here."
        ),
        (
            "Snap is not a safe default. Rails improve P95 relative to no snap, but the strict perfect-LFO rate collapses, "
            "so snapping is behaving like a tail repair that damages many otherwise close curves."
        ),
        (
            f"Runtime should be read structurally, not as tiny per-policy timing claims. 9B clipping variants have the same "
            f"operation count, so their small deltas are noise/cache/order effects. Real runtime pressure comes from "
            f"affine base+residual scoring and narrow/deep stacks: 9A spans {runtime_9a_min:+.2f} to {runtime_9a_max:+.2f} min, "
            f"and measured 9D W4/W6 rows add {runtime_9d_min:+.2f} to {runtime_9d_max:+.2f} min vs available W8 anchors."
        ),
    ]
    if best_9a is not None:
        lines.append(
            f"9A best affine row: {SCOPE_LABELS.get(str(best_9a.target_scope), best_9a.target_scope)} / "
            f"{MODULATION_LABELS.get(str(best_9a.affine_modulation), best_9a.affine_modulation)} / "
            f"{NORMALIZATION_LABELS.get(str(best_9a.normalization_label), best_9a.normalization_label)} "
            f"with P95 {best_9a.rmse_p95:.6g}."
        )
    if best_9b is not None:
        lines.append(
            f"9B best clipping strategy: "
            f"{CLIP_LABELS.get(str(best_9b.decoder_hygiene_policy), best_9b.decoder_hygiene_policy)} "
            f"with P95 {best_9b.rmse_p95:.6g}."
        )
    if best_9c is not None:
        lines.append(
            f"9C best snap policy: {SNAP_LABELS.get(str(best_9c.snap_policy), best_9c.snap_policy)} "
            f"with P95 {best_9c.rmse_p95:.6g}."
        )

    report = f"""# Experiment 9 Findings

Experiment 9 is a quick fixed-budget W8D16 screen at 120-point evaluation resolution, beam 4, fixed 1/3 sample, and phase always enabled.

## Questions

- Do per-layer gain/offset scalars help when applied to base, residuals, or both?
- Does range normalization make those scalars useful?
- Which clipping or limiting policy should be the phase-only decoder baseline?
- Do data-derived snap anchors improve final output cheaply?
- At equivalent output-head budget, do very narrow/deep W4/W6 residual stacks beat W8 references?
- Do train and validation errors move together, or are any variants just fitting the train sample?

The third primary quality metric is `perfect_lfo_rate_eps_0.02`: the share of LFOs whose every sampled evaluation point is within +/-0.02 of the target curve. This is stricter than RMSE and different from editor-node preservation.

Experiment 9 is not a replacement for Experiment 8's size screen. It answers three follow-up questions from that report: whether gain/offset become useful under better normalization, whether zero-output clipping choices can improve W8D16, and whether very narrow/deep stacks are more output-head efficient than the W8 references.

## Executive Read

{chr(10).join(f"- {line}" for line in lines)}

## 9A Affine And Normalization

This section asks whether extra continuous scalar outputs are worth paying for at fixed W8D16. The comparison varies three things: where the scalar applies, which scalar family is emitted, and whether the target is raw or range-normalized before choosing the code.

The first column is the no-extra-scalar baseline: phase-only W8D16 with the same per-layer `[-1, 1]` clipping strategy. It is repeated down the rows so every challenger is visually compared against the same baseline. The remaining columns are the optional scalar family beyond mandatory phase: gain, offset, or gain+offset. The rows show where those extra scalars are active: base only, residual layers only, or both. Raw means the original target/residual is encoded directly; range normalized means each target/residual is normalized before code selection and denormalized after reconstruction.

![9A affine grid](images/experiment-09/experiment9_9a_affine_grid.png)

The useful 9A result is narrower than "more scalars help." Applying gain to both base and residual layers is the best affine row by P95. Range normalization gives excellent medians for some rows, but the P95 winner remains raw. Offset remains suspect: the best offset-containing rows trail the best gain-only row on P95 despite paying the same or larger output-head cost. Many gain slots are effectively no-ops even in the better rows, so the scalar family is not being used uniformly across layers.

## 9B Clipping Strategy

This section keeps the model output head fixed and changes only the reconstruction rule between layers. These are zero-output-cost decoder policies: the downstream model emits the same code and phase outputs for every row. The question is whether clipping the running prefix after each residual layer is better than clipping only the final output.

![9B clipping strategy](images/experiment-09/experiment9_9b_decoder_hygiene_panels.png)

The plot focuses on the final-only baseline plus the three strategies worth discussing. The omitted headroom and residual-limiter rows were worse enough that they are not useful visual comparisons here. "Final clip only" clips once to `[0, 1]` at the end. "Per-layer [0, 1]" and "Per-layer [-1, 1]" clip the running prefix after each residual layer. "Base [0,1], residual [-1,1]" clips the base reconstruction to the unipolar output range, then allows bipolar residual accumulation.

Relative to final-only clipping, unipolar per-layer clipping improves P95 by {metric_delta(unipolar_9b, final_9b, "rmse_p95"):.6g} and raises perfect-LFO rate by {metric_delta(unipolar_9b, final_9b, "perfect_lfo_percent_eps_0.02"):.6g} percentage points. Bipolar clipping gives the best median in this section, but does not win P95. The headroom and residual-limiter variants do not justify replacing the simpler per-layer clip policies.

## 9C Snap Schwarzschild Radius

This section keeps the W8D16 phase-only reconstruction fixed and changes only a final-output snap step. The snap anchors are inferred from training data and applied after the final clip. No prefix snapping is tested here.

![9C snap diagnostics](images/experiment-09/experiment9_9c_snap_diagnostics.png)

Rails means the snap grid `{{0, 1}}`. The denser grids add `1/2`, then quarters, then thirds. For each anchor, the training data estimates a snap radius from values already near that anchor: collect points within `0.08`, take the 80th percentile distance, then clamp the radius to `[0.0075, 0.04]`. In this run every supported grid reports median radius `0.04`, so the learned radius hit the upper clamp rather than discovering a narrow natural basin.

The snap screen shows why P95 alone is not enough. Rails reduce P95 by {metric_delta(snap_rails, snap_none, "rmse_p95"):.6g} versus no snap, but increase median RMSE by {metric_delta(snap_rails, snap_none, "rmse_median"):.6g} and reduce perfect-LFO rate by {metric_delta(snap_rails, snap_none, "perfect_lfo_percent_eps_0.02"):.6g} percentage points. That is not harmless cleanup; it is an aggressive final-output correction. If snap returns, it should be gated or learned, not a blanket default. The radius saturation is another warning: the current estimator is too willing to snap a broad neighborhood.

## 9D Equivalent Budget Narrow-Depth Screen

![9D equivalent budget](images/experiment-09/experiment9_9d_budget_equivalence.png)

W4 and W6 jobs were scheduled as equivalent-budget checks against W8D24, W8D32, W8D48, and W8D64 anchors. After correcting the deployed accounting to flatten topology-conditioned dictionaries, the completed depths are not always the nearest possible depth under the new formula. The `head_delta_vs_anchor` column shows the actual mismatch. W8D24 and W8D32 reference rows are reused from Experiment 8 analytics when available; W8D48 and W8D64 are budget anchors only unless those rows are later produced.

{_markdown_table(table_9d)}

This section is the strongest argument for depth as the next representation lever, but it is not a simple constant-`W x D` test. A phase-only topology-conditioned chain pays:

```text
head_outputs = 32 base logits + sum(layer_codebook_size) + (D + 1) phase scalars
```

For shared residual layers, `layer_codebook_size = W`. For topology-conditioned residual layers, the deployment interface flattens the topology-specific dictionaries, so `layer_codebook_size = 3W`. The model emits one categorical index for the layer and does not separately predict topology. Exact equality is also impossible in some rows because depth is restricted to even values. Future budget-matched runs should use this corrected formula directly; this report keeps the completed rows and shows their corrected budget deltas.

At the W8D24-equivalent and W8D32-equivalent budgets, the reused W8 references have better medians but worse P95 than W4/W6. That is a useful split: W8's wider per-layer alphabet captures common/easy curves slightly better, while W4/W6 get more sequential refinement steps and do better on the tail. The W4 and W6 lines are close enough that width is not the main story inside this narrow/deep band; depth and sequential correction are.

## Perfect Reconstruction Rate

![Perfect LFO rate](images/experiment-09/experiment9_perfect_lfo_rate.png)

`perfect_lfo_rate_eps_0.02` is the fraction of validation LFOs with `max_abs_error <= 0.02` over the sampled evaluation grid.

This metric is intentionally unforgiving: one bad sampled point makes the whole LFO fail. The chart pairs it with P95 because the two metrics catch different failure modes. The snap policies are the clearest example: rails improve P95, but their perfect-LFO rate is far below the no-snap baseline. The narrow/deep 9D rows are the opposite pattern: they improve tail error and push the strict pass rate upward together.

The y-axis is inverted so the upper-right corner is good: more perfectly reconstructed LFOs and lower P95 error.

## Train Vs Validation

![Train vs validation](images/experiment-09/experiment9_train_vs_validation_p95.png)

![Generalization gap](images/experiment-09/experiment9_generalization_gap_vs_head_outputs.png)

The diagonal is train P95 equals validation P95. Rows below it have validation P95 lower than train P95; rows above it would be the suspicious "better on train than validation" cases. Most rows here sit below the diagonal, so this screen does not look like a conventional overfitting story. The richer decoders that lose are more likely losing because the construction/scoring objective is misaligned with the validation metric or because the decoder policy perturbs already-good curves.

## Runtime

![Runtime](images/experiment-09/experiment9_runtime.png)

`elapsed_seconds_total` is coarse wall time per completed job. It includes training, validation scoring, checkpoint/report writes, and cache effects for that job, so it is not a kernel-level XPU benchmark. The useful comparison is the incremental runtime impact inside each scenario.

The runtime baseline changes by section. 9A is compared against the W8D16 phase-only per-layer `[-1, 1]` row, because that is the matching no-extra-scalar policy. 9B is compared against final-only clipping. 9C is compared against no snap. 9D is compared against the W8 reference runtime only where that reference was actually run; W8D48 and W8D64 remain quality anchors without runtime baselines.

The pattern is sharper this way, but only for dimensions that change the amount of work. The 9B clipping rows all perform the same kind of clamp operation; `[0, 1]` and `[-1, 1]` are not fundamentally different, so the measured {runtime_9b_min:+.2f} to {runtime_9b_max:+.2f} minute spread should be treated as job-level noise from cache state, run order, process startup, checkpoint writes, and background load. It is evidence that clipping policy is cheap, not evidence that one clamp range is faster.

9C snap policies add a final snap pass, so some positive runtime delta is plausible, but the measured spread is still small: {runtime_9c_min:+.2f} to {runtime_9c_max:+.2f} minutes versus no snap. Rails specifically cost {runtime_rails_delta:+.2f} minutes and damaged perfect-LFO rate, so they are not attractive despite the tail-RMSE improvement. 9A is the first section with a more believable runtime signal: residual-only and base-only affine rows are modest, while base+residual gain/gain+offset rows expand scoring enough to create the largest W8D16 runtime hits. 9D adds real runtime versus W8 where measured, but it is the only runtime increase here that also buys a large P95 and perfect-LFO improvement.

## Output-Head Accounting

![Head outputs](images/experiment-09/experiment9_head_outputs_vs_rmse.png)

`head_outputs` is the deployed model-facing output burden: categorical logits plus continuous scalar outputs per LFO. In 9A, gain/offset cost depends on whether the scalar family applies to base, residual layers, or both. Clipping and snap policies do not add model outputs.

Topology-conditioned stages are accounted as flattened dictionaries, not as a separate topology prediction followed by a local code prediction. The useful comparison is output-head budget, not `W x D` alone, because deeper chains also require more phase scalars and topology-conditioned layers cost `3W` logits. For the downstream model, the important distinction is that 9B and 9C policies are zero-output decoder choices, while 9A affine variants add scalar outputs. The equivalent-budget 9D result suggests a more promising way to spend output-head budget: keep the alphabet narrow and buy more residual layers.

## Working Recommendation

- Carry forward narrow/deep phase-only stacks as the main Experiment 10 candidate, with W4/W6 depths chosen by output-head budget rather than by matching W8D names.
- Keep per-layer clipping in the candidate set, but choose between unipolar and bipolar based on whether P95 or median/perfect-LFO rate is the priority.
- Do not carry blanket snap forward as a default; only revisit snap as a gated or learned post-process.
- Treat phase+gain on base and residuals as the only affine variant worth a small follow-up; offset and range normalization did not earn broad expansion here.
- Keep `perfect_lfo_rate_eps_0.02` beside median and P95 in future reports because it caught failures that P95 alone made look attractive.

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
    plot_summary = _with_9a_phase_only_baseline(report_summary.copy())
    section_9a = plot_summary[plot_summary["experiment9_section"] == "9A"].copy()
    section_9b = report_summary[report_summary["experiment9_section"] == "9B"].copy()
    section_9c = report_summary[report_summary["experiment9_section"] == "9C"].copy()
    section_9d = report_summary[report_summary["experiment9_section"].isin(["9D", "9D_ref"])].copy()
    _save_9a_affine_grid(section_9a, PLOTS / "experiment9_9a_affine_grid.png")
    _save_policy_metric_panels(
        section_9b,
        "decoder_hygiene_policy",
        CLIP_ORDER,
        "Experiment 9B clipping strategy screen",
        PLOTS / "experiment9_9b_decoder_hygiene_panels.png",
    )
    _save_snap_diagnostics(section_9c, PLOTS / "experiment9_9c_snap_diagnostics.png")
    _save_budget_equivalence(section_9d, PLOTS / "experiment9_9d_budget_equivalence.png")
    _save_perfect_lfo_rate(report_summary[report_summary["experiment9_section"] != "9D_ref"], PLOTS / "experiment9_perfect_lfo_rate.png")
    _save_train_validation(report_summary, PLOTS / "experiment9_train_vs_validation_p95.png")
    _save_generalization_gap(report_summary, PLOTS / "experiment9_generalization_gap_vs_head_outputs.png")
    _save_head_outputs(report_summary, PLOTS / "experiment9_head_outputs_vs_rmse.png")
    _save_runtime_summary(report_summary, PLOTS / "experiment9_runtime.png")
    _write_report(report_summary)
    print(REPORT)


if __name__ == "__main__":
    main()

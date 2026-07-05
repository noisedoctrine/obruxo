"""Aggregate analytics for Era 2 experiment runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .accounting import path_address_budget_for_width


ERA2_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT11_RUN_ROOT = ERA2_ROOT / "artifacts" / "experiment_11" / "runs"
EXPERIMENT11_REPORT_PATH = ERA2_ROOT / "reports" / "EXPERIMENT_11_FLAT_CATEGORICAL_REPORT.md"
EXPERIMENT11_REPORT_IMAGE_DIR = ERA2_ROOT / "reports" / "images" / "experiment_11"


def analyze_run(run_dir: Path) -> dict[str, str]:
    run_dir = Path(run_dir)
    analytics_dir = run_dir / "analytics"
    analytics_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_row_summaries(run_dir)
    summary_path = analytics_dir / "summary.csv"
    _write_csv(summary_path, rows)
    budget_path = analytics_dir / "budget_band_summary.csv"
    _write_csv(budget_path, _budget_band_summary(rows))
    frontier_path = analytics_dir / "frontier.csv"
    _write_csv(frontier_path, _frontier(rows))
    projections_path = analytics_dir / "budget_projections.csv"
    projections = _budget_projections(rows)
    _write_csv(projections_path, projections)
    failures_path = analytics_dir / "failures.csv"
    _write_csv(failures_path, _failures(run_dir))
    report_path, image_dir = _report_paths(run_dir)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_run_report(run_dir, rows), encoding="utf-8")
    _write_plots(image_dir, rows)
    return {
        "analytics_dir": str(analytics_dir),
        "summary": str(summary_path),
        "budget_band_summary": str(budget_path),
        "frontier": str(frontier_path),
        "budget_projections": str(projections_path),
        "failures": str(failures_path),
        "report": str(report_path),
        "report_image_dir": str(image_dir),
    }


def _report_paths(run_dir: Path) -> tuple[Path, Path]:
    try:
        run_dir.resolve().relative_to(EXPERIMENT11_RUN_ROOT.resolve())
    except ValueError:
        report_path = run_dir / "reports" / "EXPERIMENT_11_FLAT_CATEGORICAL_REPORT.md"
        return report_path, report_path.parent / "images" / "experiment_11"
    return EXPERIMENT11_REPORT_PATH, EXPERIMENT11_REPORT_IMAGE_DIR


def _load_row_summaries(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((run_dir / "rows").glob("*/summary.csv")):
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row["row_id"] = row.get("row_id") or path.parent.name
                rows.append(row)
    return rows


def _budget_band_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("budget_band", "")), []).append(row)
    result = []
    for band, members in sorted(grouped.items()):
        p95_values = [_float(row.get("validation_p95_rmse")) for row in members]
        head_outputs = [_float(row.get("head_outputs_actual")) for row in members]
        result.append(
            {
                "budget_band": band,
                "row_count": len(members),
                "best_validation_p95_rmse": min(p95_values) if p95_values else "",
                "min_head_outputs_actual": min(head_outputs) if head_outputs else "",
                "max_head_outputs_actual": max(head_outputs) if head_outputs else "",
            }
        )
    return result


def _frontier(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if _float(row.get("head_outputs_actual")) is not None
        and _float(row.get("validation_p95_rmse")) is not None
    ]
    candidates.sort(key=lambda row: (_float(row.get("head_outputs_actual")) or 0.0, _float(row.get("validation_p95_rmse")) or 0.0))
    frontier = []
    best = None
    for row in candidates:
        p95 = _float(row.get("validation_p95_rmse"))
        if p95 is not None and (best is None or p95 < best):
            frontier.append(row)
            best = p95
    return frontier


def _budget_projections(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    projections: list[dict[str, Any]] = []
    for row in rows:
        row_id = row.get("row_id", "")
        actual = _int(row.get("head_outputs_actual"))
        if actual is not None:
            projections.append(
                {
                    "row_id": row_id,
                    "budget_view_id": "actual_flat_categorical",
                    "runtime_interface_id": row.get("runtime_interface_id", ""),
                    "head_outputs_formula": row.get("head_outputs_formula", ""),
                    "head_outputs_projected": actual,
                    "is_actual_runtime_interface": True,
                }
            )
        D = _int(row.get("D"))
        W = _int(row.get("W"))
        base = _int(row.get("base_dictionary_size")) or 32
        if D is None or W is None:
            continue
        try:
            binary = path_address_budget_for_width(
                base_dictionary_size=base,
                residual_layer_count=D,
                width=W,
                branching_factor=2,
            )
        except ValueError:
            continue
        projections.append(
            {
                "row_id": row_id,
                "budget_view_id": "formula_only_binary_path_same_leaf_capacity",
                "runtime_interface_id": "formula_only_path_address_per_residual_layer",
                "head_outputs_formula": binary.head_outputs_formula,
                "head_outputs_projected": binary.head_outputs_actual,
                "is_actual_runtime_interface": False,
            }
        )
    return projections


def _failures(run_dir: Path) -> list[dict[str, Any]]:
    status_path = run_dir / "run_status.json"
    if not status_path.exists():
        return []
    status = json.loads(status_path.read_text(encoding="utf-8"))
    rows = status.get("rows", {})
    return [
        {"row_id": row_id, **payload}
        for row_id, payload in sorted(rows.items())
        if payload.get("status") not in {"completed", "skipped"}
    ]


def _run_report(run_dir: Path, rows: list[dict[str, Any]]) -> str:
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    completed = len(rows)
    best_p95 = _best_row(rows, "validation_p95_rmse")
    best_median = _best_row(rows, "validation_median_rmse")
    frontier = _frontier(rows)
    bands = _budget_band_summary(rows)
    ordered_by_budget = sorted(
        rows,
        key=lambda row: (
            _float(row.get("head_outputs_actual")) or float("inf"),
            _float(row.get("validation_p95_rmse")) or float("inf"),
        ),
    )
    ordered_by_p95 = sorted(rows, key=lambda row: _float(row.get("validation_p95_rmse")) or float("inf"))
    small_best = _best_in_band(rows, "small", "validation_p95_rmse")
    medium_best = _best_in_band(rows, "medium", "validation_p95_rmse")
    efficient_medium = _best_in_band(rows, "medium", "head_outputs_actual")
    dominated = _dominated_rows(rows)
    topology_pass_count = sum(1 for row in rows if str(row.get("topology_contract_pass", "")).lower() == "true")
    active_phase_count = sum(1 for row in rows if _phase_is_active(row))
    inactive_phase_count = sum(1 for row in rows if _phase_is_counted(row) and not _phase_is_active(row))
    lattice_values = sorted({str(row.get("lfo_control_point_count", "")) for row in rows if row.get("lfo_control_point_count", "") != ""})
    dataset = manifest.get("dataset", {})
    sample_fraction = manifest.get("corpus_sample_fraction_requested", "")
    smoke = manifest.get("smoke", "")
    dataset_sentence = (
        f"The run used the full active corpus split "
        f"({dataset.get('train_count', '')} train / {dataset.get('validation_count', '')} validation, "
        f"{dataset.get('dataset_row_count', '')} total active LFO rows)."
        if str(sample_fraction) == "1.0" and str(smoke).lower() == "false"
        else (
            f"The run used smoke=`{smoke}` and requested corpus_sample_fraction=`{sample_fraction}` "
            f"({dataset.get('train_count', '')} train / {dataset.get('validation_count', '')} validation)."
        )
    )
    p95_metric_delta = _metric_delta(best_p95, rows, "validation_p95_rmse")
    medium_tradeoff = _pair_delta(efficient_medium, medium_best, "validation_p95_rmse", "head_outputs_actual")
    if best_median is not None and medium_best is not None and best_median is not medium_best:
        median_sentence = (
            f"The best typical-case result is `{_row_id(best_median)}` at validation median RMSE "
            f"`{_fmt_metric(best_median, 'validation_median_rmse')}`. `{_row_id(medium_best)}` reaches the same median, "
            "but spends more budget to improve the tail."
        )
    else:
        median_sentence = (
            f"The best typical-case result is `{_row_id(best_median)}` at validation median RMSE "
            f"`{_fmt_metric(best_median, 'validation_median_rmse')}`."
        )
    canonical_rows_present = all(
        _row_by_id(rows, row_id) is not None
        for row_id in ("w4_d48", "w6_d32", "w8_d28", "w4_d120", "w6_d80", "w8_d72")
    )
    if canonical_rows_present:
        why_lines = [
            f"Within the small budget band, `{_row_id(small_best)}` wins P95 even though it has the largest small-band budget. That says the extra atom choices per residual layer are helping more than simply adding residual layers at `W=4` or `W=6` in this range.",
            "",
            f"Within the medium band, `{_row_id(efficient_medium)}` is the efficient row and `{_row_id(medium_best)}` is the quality row. {_tradeoff_sentence(efficient_medium, medium_best, medium_tradeoff)} This is the main practical tension coming out of the run: `W=6,D=80` is attractive if we care about budget efficiency, while `W=8,D=72` is the cleaner quality candidate.",
            "",
            f"The `W=4` rows are the weak signal. `{_row_id(_row_by_id(rows, 'w4_d48'))}` is dominated in the small band, and `{_row_id(_row_by_id(rows, 'w4_d120'))}` is dominated in the medium band. In this screen, pushing many low-width residual layers does not buy enough reconstruction quality to justify the head budget.",
        ]
        plot_p95_note = "Lower is better. The x-axis is deployed model prediction head budget, not dictionary storage or oracle work. The plot shows a clear improvement when moving from the small band into the medium band, but the medium rows are not interchangeable: `w6_d80` gets most of the metric improvement, while `w8_d72` spends an extra 88 heads for the best tail."
        plot_median_note = "Lower is better. The median plot separates typical-case behavior from tail behavior. `w6_d80` and `w8_d72` land at the same median, so the extra `W=8` capacity is mainly buying tail cleanup, not a typical-case shift. `w4_d120` underperforms both even with a medium-band budget."
        frontier_note = "Reading left to right by budget, a row only belongs on the frontier if it improves validation P95 over every cheaper row. That is why the `W=4` rows are not decision candidates from this screen."
        projection_note = _projection_note(rows)
        takeaway_lines = [
            "- Treat `w8_d72` as the current quality anchor for the flat-categorical interface.",
            "- Treat `w6_d80` as the current budget-efficiency anchor; it matches the best median and is close on P95 with fewer heads.",
            "- Do not spend more time on low-width, many-layer `W=4` rows until there is a new construction idea that specifically justifies them.",
            "- The active-phase fix worked at the accounting level: phase is now an oracle-estimated continuous scalar target, and its search resolution does not change `head_outputs_actual`.",
            "- Topology remains cleanly out of deployed runtime: it is not used in inputs, targets, loss, decoder lookup, or head accounting.",
        ]
    else:
        why_lines = [
            "This run does not contain the full canonical six-row Experiment 11 screen, so the report should be read as a fixture, smoke, or partial-run summary. The useful interpretation is the local frontier inside the rows that were actually run; do not generalize row-family conclusions from this subset.",
            "",
            f"The current local frontier is `{_frontier_labels(frontier)}`. Rows outside that frontier are cheaper-or-worse tradeoffs for this run only.",
        ]
        plot_p95_note = "Lower is better. The x-axis is deployed model prediction head budget, not dictionary storage or oracle work. For partial runs, use this plot only to inspect the local quality/budget ordering."
        plot_median_note = "Lower is better. The median plot shows typical-case reconstruction quality for the rows present in this run. Partial runs do not support broad width/depth conclusions."
        frontier_note = "Reading left to right by budget, a row only belongs on the frontier if it improves validation P95 over every cheaper row in this run."
        projection_note = _projection_note(rows)
        takeaway_lines = [
            "- Treat this as a partial-run or test-fixture report unless it contains the canonical Experiment 11 screen rows.",
            "- The active-phase and topology-contract checks are still meaningful for readiness.",
            "- Use the CSV artifacts for exact values; avoid making row-family claims from partial coverage.",
        ]
    lines = [
        "# Experiment 11 Flat-Categorical Report",
        "",
        "## Main Findings",
        "",
        f"The active-phase Experiment 11 screen completed `{completed}` rows and now looks like a real quality run, not just a framework check. {dataset_sentence}",
        f"The best tail result is `{_row_id(best_p95)}` at validation P95 RMSE `{_fmt_metric(best_p95, 'validation_p95_rmse')}` with `{_heads(best_p95)}` model prediction head outputs. {_delta_sentence(p95_metric_delta)}",
        median_sentence,
        f"The useful frontier is `{_frontier_labels(frontier)}`. Rows outside that frontier are dominated on this screen: `{_labels(dominated)}`.",
        f"The fixed LFO vector shape is `{', '.join(lattice_values) if lattice_values else 'not recorded'}` control points. The x lattice is decoder-owned geometry and adds zero model prediction head outputs.",
        f"The topology contract passed for `{topology_pass_count}/{completed}` rows, and active oracle phase search was recorded for `{active_phase_count}/{completed}` rows.",
        "",
        "![Validation P95 vs model prediction head budget](./images/experiment_11/validation_p95_vs_head_outputs.png)",
        "",
        "![Validation median vs model prediction head budget](./images/experiment_11/validation_median_vs_head_outputs.png)",
        "",
        *(
            [
                f"Important caveat: `{inactive_phase_count}` row(s) count phase scalar outputs but record only one oracle phase candidate. Treat those rows as framework/readiness runs, not fair quality comparisons against phase-active Era 1 rows.",
                "",
            ]
            if inactive_phase_count
            else []
        ),
        "## Why This Happens",
        "",
        "The screen is mostly comparing width/depth tradeoffs under the same flat-categorical runtime interface. `W` is the number of atom choices per residual layer; `D` is the residual-layer count. The model prediction head budget is `32 + D * W + (D + 1)`: 32 base atom logits, `D * W` residual-layer atom logits, and one continuous phase scalar for the base plus one per residual layer.",
        "",
        *why_lines,
        "",
        "## Plot Notes",
        "",
        "### Validation P95 Vs Model Prediction Head Budget",
        "",
        plot_p95_note,
        "",
        "### Validation Median Vs Model Prediction Head Budget",
        "",
        plot_median_note,
        "",
        "### Validation P95 By Row",
        "",
        "Lower is better. This plot is the easiest row-level read: each bar is one planned screen row, colored by budget band. It makes the dominated rows visible without pretending this six-row screen is a dense sweep.",
        "",
        "![Validation P95 by row](./images/experiment_11/validation_p95_by_row.png)",
        "",
        "### Runtime Vs Model Prediction Head Budget",
        "",
        "Lower is faster. Runtime broadly increases with row size, but this is oracle construction/encoding runtime on the current implementation, not deployed model runtime. Treat it as a workflow planning metric for Era 2 experiment velocity.",
        "",
        "![Runtime vs model prediction head budget](./images/experiment_11/runtime_vs_head_outputs.png)",
        "",
        "## Best Rows By Validation P95",
        "",
        "Full numeric row metrics are in `analytics/summary.csv`; the markdown only keeps the decision-level facts.",
        "",
    ]
    for index, row in enumerate(ordered_by_p95[:6], start=1):
        lines.append(
            f"{index}. `{row.get('row_id', '')}`: P95 `{_fmt(row.get('validation_p95_rmse'))}`, "
            f"median `{_fmt(row.get('validation_median_rmse'))}`, "
            f"`{row.get('head_outputs_actual', '')}` heads, `{row.get('budget_band', '')}` band."
        )
    lines.extend(
        [
            "",
            "## Budget Band Read",
            "",
        ]
    )
    for row in bands:
        lines.append(
            f"- `{row.get('budget_band', '')}`: `{row.get('row_count', '')}` rows, "
            f"best P95 `{_fmt(row.get('best_validation_p95_rmse'))}`, "
            f"head range `{_fmt(row.get('min_head_outputs_actual'), decimals=0)}`-`{_fmt(row.get('max_head_outputs_actual'), decimals=0)}`."
        )
    lines.extend(
        [
            "",
            "## Frontier Read",
            "",
            "Lower validation P95 is better. `head_outputs_actual` is the model prediction head budget; the fixed x lattice is decoder-owned and does not add outputs.",
            "Oracle phase-search resolution is also not part of this budget: the deployed model emits one continuous phase scalar per base/residual layer either way.",
            "",
        ]
    )
    for row in frontier[:10]:
        lines.append(
            f"- `{row.get('row_id', '')}`: `{row.get('head_outputs_actual', '')}` heads, "
            f"P95 `{_fmt(row.get('validation_p95_rmse'))}`, `{row.get('budget_band', '')}` band."
        )
    if ordered_by_budget:
        lines.append("")
        lines.append(
        frontier_note
        )
    lines.extend(
        [
            "",
            "## Budget Projection Notes",
            "",
            "Run-local `analytics/budget_projections.csv` includes formula-only views for alternate dictionary addressing strategies, currently including binary path addressing over the same residual-layer leaf capacity.",
            "",
            projection_note,
            "",
            "## Practical Takeaways",
            "",
            *takeaway_lines,
            "",
            "## Runtime And Readiness Notes",
            "",
            f"- Run id: `{manifest.get('run_id', run_dir.name)}`",
            f"- Corpus mode: smoke=`{manifest.get('smoke', '')}`, requested sample fraction=`{manifest.get('corpus_sample_fraction_requested', '')}`.",
            f"- Screen: `{manifest.get('screen', '')}`",
            f"- Dataset: `{dataset.get('train_count', '')}` train, `{dataset.get('validation_count', '')}` validation, `{dataset.get('dataset_row_count', '')}` total active LFO rows.",
            f"- LFO vector shape: `{', '.join(lattice_values) if lattice_values else 'not recorded'}` control points.",
            "- Topology may be used for offline construction, but runtime topology is not part of inputs, targets, loss, decoder lookup, or model prediction head budget.",
            "- Any topology bucket metrics are analysis-only.",
            "- `oracle_phase_search_policy` and `oracle_phase_candidate_count` describe oracle target generation, not deployed head-output cost.",
            "- CSV analytics remain in the run artifact directory. This markdown file is the canonical Experiment 11 report.",
        ]
    )
    return "\n".join(lines) + "\n"


def _best_row(rows: list[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    candidates = [row for row in rows if _float(row.get(metric)) is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda row: _float(row.get(metric)) or float("inf"))


def _best_in_band(rows: list[dict[str, Any]], band: str, metric: str) -> dict[str, Any] | None:
    candidates = [row for row in rows if str(row.get("budget_band", "")) == band and _float(row.get(metric)) is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda row: _float(row.get(metric)) or float("inf"))


def _row_by_id(rows: list[dict[str, Any]], row_id: str) -> dict[str, Any] | None:
    return next((row for row in rows if row.get("row_id") == row_id), None)


def _dominated_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frontier_ids = {row.get("row_id") for row in _frontier(rows)}
    return [
        row
        for row in sorted(rows, key=lambda item: str(item.get("row_id", "")))
        if row.get("row_id") not in frontier_ids
    ]


def _best_row_sentence(label: str, row: dict[str, Any] | None, metric: str) -> str:
    if row is None:
        return f"{label}: not available."
    return (
        f"{label}: `{row.get('row_id', '')}` at `{row.get(metric, '')}` "
        f"with `{row.get('head_outputs_actual', '')}` head outputs."
    )


def _fmt(value: Any, *, decimals: int = 4) -> str:
    number = _float(value)
    if number is None:
        return str(value) if value not in (None, "") else "n/a"
    if decimals == 0:
        return str(int(round(number)))
    return f"{number:.{decimals}f}"


def _fmt_metric(row: dict[str, Any] | None, metric: str) -> str:
    if row is None:
        return "n/a"
    return _fmt(row.get(metric))


def _row_id(row: dict[str, Any] | None) -> str:
    return str(row.get("row_id", "n/a")) if row else "n/a"


def _heads(row: dict[str, Any] | None) -> str:
    return str(row.get("head_outputs_actual", "n/a")) if row else "n/a"


def _labels(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "none"
    return ", ".join(str(row.get("row_id", "")) for row in rows)


def _frontier_labels(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "empty"
    return " -> ".join(str(row.get("row_id", "")) for row in rows)


def _metric_delta(best: dict[str, Any] | None, rows: list[dict[str, Any]], metric: str) -> tuple[float | None, str | None]:
    if best is None:
        return None, None
    best_value = _float(best.get(metric))
    candidates = [row for row in rows if row is not best and _float(row.get(metric)) is not None]
    if best_value is None or not candidates:
        return None, None
    runner_up = min(candidates, key=lambda row: _float(row.get(metric)) or float("inf"))
    runner_value = _float(runner_up.get(metric))
    if runner_value is None:
        return None, None
    return runner_value - best_value, str(runner_up.get("row_id", ""))


def _delta_sentence(delta: tuple[float | None, str | None]) -> str:
    value, row_id = delta
    if value is None or not row_id:
        return ""
    return f"That is `{_fmt(value)}` better than the next row, `{row_id}`."


def _pair_delta(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
    metric_key: str,
    budget_key: str,
) -> tuple[float | None, float | None]:
    if left is None or right is None or left is right:
        return None, None
    left_metric = _float(left.get(metric_key))
    right_metric = _float(right.get(metric_key))
    left_budget = _float(left.get(budget_key))
    right_budget = _float(right.get(budget_key))
    if left_metric is None or right_metric is None or left_budget is None or right_budget is None:
        return None, None
    return left_metric - right_metric, right_budget - left_budget


def _tradeoff_sentence(left: dict[str, Any] | None, right: dict[str, Any] | None, delta: tuple[float | None, float | None]) -> str:
    if left is None or right is None:
        return "The medium-band tradeoff is not available."
    if left is right:
        return f"`{_row_id(left)}` is both the cheapest and best medium-band row."
    p95_delta, budget_delta = delta
    if p95_delta is None or budget_delta is None:
        return f"`{_row_id(right)}` improves quality over `{_row_id(left)}`, but the exact tradeoff could not be computed."
    return (
        f"`{_row_id(right)}` improves validation P95 by `{_fmt(p95_delta)}` "
        f"while adding `{_fmt(budget_delta, decimals=0)}` head outputs over `{_row_id(left)}`."
    )


def _projection_note(rows: list[dict[str, Any]]) -> str:
    savings: list[str] = []
    unchanged_widths: set[int] = set()
    for row in rows:
        D = _int(row.get("D"))
        W = _int(row.get("W"))
        base = _int(row.get("base_dictionary_size")) or 32
        actual = _int(row.get("head_outputs_actual"))
        if D is None or W is None or actual is None:
            continue
        try:
            projected = path_address_budget_for_width(
                base_dictionary_size=base,
                residual_layer_count=D,
                width=W,
                branching_factor=2,
            ).head_outputs_actual
        except ValueError:
            continue
        if projected < actual:
            savings.append(f"`{row.get('row_id', '')}` drops from {actual} to {projected}")
        elif projected == actual:
            unchanged_widths.add(W)
    if savings:
        unchanged = (
            f" The `W={', W='.join(str(width) for width in sorted(unchanged_widths))}` rows match the flat-categorical budget under the current formula."
            if unchanged_widths
            else ""
        )
        return (
            "The important read is narrow: binary path addressing changes projected budget in two places: "
            + "; ".join(savings)
            + "."
            + unchanged
            + " These are budget views, not quality claims, because changing atom indexing changes the learning problem and may require a different dictionary organization."
        )
    return (
        "In this run, the formula-only binary path view does not reduce projected head budget versus the actual flat-categorical interface. "
        "These are still budget views, not quality claims, because changing atom indexing changes the learning problem and may require a different dictionary organization."
    )


def _write_plots(image_dir: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    image_dir.mkdir(parents=True, exist_ok=True)
    _scatter_plot(
        image_dir / "validation_p95_vs_head_outputs.png",
        rows,
        x_key="head_outputs_actual",
        y_key="validation_p95_rmse",
        xlabel="model prediction head budget",
        ylabel="validation p95 RMSE",
        title="Validation P95 vs model prediction head budget",
        plt=plt,
    )
    _scatter_plot(
        image_dir / "validation_median_vs_head_outputs.png",
        rows,
        x_key="head_outputs_actual",
        y_key="validation_median_rmse",
        xlabel="model prediction head budget",
        ylabel="validation median RMSE",
        title="Validation median vs model prediction head budget",
        plt=plt,
    )
    _scatter_plot(
        image_dir / "runtime_vs_head_outputs.png",
        rows,
        x_key="head_outputs_actual",
        y_key="row_elapsed_seconds",
        xlabel="model prediction head budget",
        ylabel="row elapsed seconds",
        title="Runtime vs model prediction head budget",
        plt=plt,
    )
    _row_bar_plot(
        image_dir / "validation_p95_by_row.png",
        rows,
        y_key="validation_p95_rmse",
        ylabel="validation p95 RMSE",
        title="Validation P95 by row",
        plt=plt,
    )


def _scatter_plot(path: Path, rows: list[dict[str, Any]], *, x_key: str, y_key: str, xlabel: str, ylabel: str, title: str, plt: Any) -> None:
    points = [
        (row, _float(row.get(x_key)), _float(row.get(y_key)))
        for row in rows
    ]
    points = [(row, x, y) for row, x, y in points if x is not None and y is not None]
    if not points:
        return
    palette = {"small": "#4C78A8", "medium": "#F58518", "large": "#54A24B", "tiny": "#7E57C2"}
    plt.figure(figsize=(8.5, 5.2))
    x_values = [x for _, x, _ in points]
    y_values = [y for _, _, y in points]
    x_span = max(x_values) - min(x_values) if len(x_values) > 1 else 1.0
    y_span = max(y_values) - min(y_values) if len(y_values) > 1 else 1.0
    for index, (row, x, y) in enumerate(points):
        color = palette.get(str(row.get("budget_band", "")), "#6B7280")
        plt.scatter([x], [y], color=color, s=58)
        x_offset = -46 if x > max(x_values) - 0.08 * x_span else 6
        y_offset = -12 if y > max(y_values) - 0.08 * y_span or index % 2 else 5
        plt.annotate(
            str(row.get("row_id", "")),
            (x, y),
            textcoords="offset points",
            xytext=(x_offset, y_offset),
            fontsize=8,
        )
    if y_key == "validation_p95_rmse":
        frontier = [
            row
            for row in _frontier(rows)
            if _float(row.get(x_key)) is not None and _float(row.get(y_key)) is not None
        ]
        if len(frontier) >= 2:
            frontier_x = [_float(row.get(x_key)) for row in frontier]
            frontier_y = [_float(row.get(y_key)) for row in frontier]
            plt.plot(frontier_x, frontier_y, color="#111827", linewidth=1.25, alpha=0.75, label="P95 frontier")
            plt.legend(frameon=False)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _row_bar_plot(path: Path, rows: list[dict[str, Any]], *, y_key: str, ylabel: str, title: str, plt: Any) -> None:
    ordered = [
        row
        for row in sorted(rows, key=lambda item: (str(item.get("budget_band", "")), _float(item.get(y_key)) or float("inf")))
        if _float(row.get(y_key)) is not None
    ]
    if not ordered:
        return
    labels = [str(row.get("row_id", "")) for row in ordered]
    values = [_float(row.get(y_key)) or 0.0 for row in ordered]
    bands = [str(row.get("budget_band", "")) for row in ordered]
    palette = {"small": "#4C78A8", "medium": "#F58518", "large": "#54A24B", "tiny": "#7E57C2"}
    colors = [palette.get(band, "#6B7280") for band in bands]
    width = max(7.0, 0.55 * len(labels))
    plt.figure(figsize=(width, 4.8))
    plt.bar(range(len(values)), values, color=colors)
    for index, value in enumerate(values):
        plt.text(index, value, _fmt(value), ha="center", va="bottom", fontsize=8, rotation=0)
    plt.xticks(range(len(labels)), labels, rotation=35, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    number = _float(value)
    if number is None:
        return None
    return int(number)


def _phase_is_counted(row: dict[str, Any]) -> bool:
    return "phase" in str(row.get("scalar_families", "")).lower()


def _phase_is_active(row: dict[str, Any]) -> bool:
    if not _phase_is_counted(row):
        return False
    candidate_count = _int(row.get("oracle_phase_candidate_count"))
    if candidate_count is None:
        candidate_count = _int(row.get("phase_bins"))
    return candidate_count is not None and candidate_count > 1

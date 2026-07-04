"""Aggregate analytics for Era 2 experiment runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


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
    failures_path = analytics_dir / "failures.csv"
    _write_csv(failures_path, _failures(run_dir))
    report_path = analytics_dir / "run_report.md"
    report_path.write_text(_run_report(run_dir, rows), encoding="utf-8")
    _write_plots(analytics_dir / "images", rows)
    return {
        "analytics_dir": str(analytics_dir),
        "summary": str(summary_path),
        "budget_band_summary": str(budget_path),
        "frontier": str(frontier_path),
        "failures": str(failures_path),
        "run_report": str(report_path),
    }


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
    topology_pass_count = sum(1 for row in rows if str(row.get("topology_contract_pass", "")).lower() == "true")
    lattice_values = sorted({str(row.get("lfo_control_point_count", "")) for row in rows if row.get("lfo_control_point_count", "") != ""})
    lines = [
        "# Experiment 11 Run Report",
        "",
        "## Main Findings",
        "",
        f"This run completed `{completed}` topology-free flat-categorical rows for `{manifest.get('profile', '')}`.",
        f"The LFO vector shape recorded by completed rows is `{', '.join(lattice_values) if lattice_values else 'not recorded'}` control points.",
        f"Topology contract passed for `{topology_pass_count}/{completed}` completed rows.",
        "",
        _best_row_sentence("Best validation P95", best_p95, "validation_p95_rmse"),
        _best_row_sentence("Best validation median", best_median, "validation_median_rmse"),
        f"The Pareto frontier has `{len(frontier)}` row(s) when sorting by model prediction head budget and validation P95.",
        "",
        "## Best Rows By Validation P95",
        "",
        "| row | budget band | head outputs | validation p95 RMSE | validation median RMSE | elapsed seconds |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    ordered = sorted(rows, key=lambda row: _float(row.get("validation_p95_rmse")) or float("inf"))
    for row in ordered[:10]:
        lines.append(
            "| {row_id} | {band} | {heads} | {p95} | {median} | {elapsed} |".format(
                row_id=row.get("row_id", ""),
                band=row.get("budget_band", ""),
                heads=row.get("head_outputs_actual", ""),
                p95=row.get("validation_p95_rmse", ""),
                median=row.get("validation_median_rmse", ""),
                elapsed=row.get("row_elapsed_seconds", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Budget Band Read",
            "",
            "| budget band | rows | best validation P95 | min head outputs | max head outputs |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in bands:
        lines.append(
            "| {band} | {count} | {p95} | {min_heads} | {max_heads} |".format(
                band=row.get("budget_band", ""),
                count=row.get("row_count", ""),
                p95=row.get("best_validation_p95_rmse", ""),
                min_heads=row.get("min_head_outputs_actual", ""),
                max_heads=row.get("max_head_outputs_actual", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Frontier Read",
            "",
            "Lower validation P95 is better. `head_outputs_actual` is the model prediction head budget; the fixed x lattice is decoder-owned and does not add outputs.",
            "",
            "| row | head outputs | validation p95 RMSE | budget band |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for row in frontier[:10]:
        lines.append(
            "| {row_id} | {heads} | {p95} | {band} |".format(
                row_id=row.get("row_id", ""),
                heads=row.get("head_outputs_actual", ""),
                p95=row.get("validation_p95_rmse", ""),
                band=row.get("budget_band", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Runtime And Readiness Notes",
            "",
            f"- Run id: `{manifest.get('run_id', run_dir.name)}`",
            f"- Screen: `{manifest.get('screen', '')}`",
            f"- Profile: `{manifest.get('profile', '')}`",
            "- Runtime topology is not part of targets, loss, decoder lookup, or model prediction head budget.",
            "- Any topology bucket metrics are analysis-only.",
            "- Analytics CSVs and plots are emitted beside this report.",
        ]
    )
    return "\n".join(lines) + "\n"


def _best_row(rows: list[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    candidates = [row for row in rows if _float(row.get(metric)) is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda row: _float(row.get(metric)) or float("inf"))


def _best_row_sentence(label: str, row: dict[str, Any] | None, metric: str) -> str:
    if row is None:
        return f"{label}: not available."
    return (
        f"{label}: `{row.get('row_id', '')}` at `{row.get(metric, '')}` "
        f"with `{row.get('head_outputs_actual', '')}` head outputs."
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
    points = [(_float(row.get(x_key)), _float(row.get(y_key))) for row in rows]
    points = [(x, y) for x, y in points if x is not None and y is not None]
    if not points:
        return
    x, y = zip(*points)
    plt.figure(figsize=(7, 4.5))
    plt.scatter(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
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
    plt.xticks(range(len(labels)), labels, rotation=35, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
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

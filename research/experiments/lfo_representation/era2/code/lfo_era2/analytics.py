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
    lines = [
        "# Experiment 10 Run Report",
        "",
        f"- Run id: `{manifest.get('run_id', run_dir.name)}`",
        f"- Screen: `{manifest.get('screen', '')}`",
        f"- Profile: `{manifest.get('profile', '')}`",
        f"- Completed rows: `{len(rows)}`",
        "",
        "This report summarizes topology-free flat-categorical Experiment 10 rows. "
        "Any topology bucket metrics are analysis-only and do not participate in runtime targets, loss, decoder lookup, or model prediction head budget.",
        "",
        "## Best Rows By Validation P95",
        "",
        "| row | budget band | head outputs | validation p95 RMSE | validation median RMSE |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    ordered = sorted(rows, key=lambda row: _float(row.get("validation_p95_rmse")) or float("inf"))
    for row in ordered[:10]:
        lines.append(
            "| {row_id} | {band} | {heads} | {p95} | {median} |".format(
                row_id=row.get("row_id", ""),
                band=row.get("budget_band", ""),
                heads=row.get("head_outputs_actual", ""),
                p95=row.get("validation_p95_rmse", ""),
                median=row.get("validation_median_rmse", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Valid comparisons are inside the topology-free flat-categorical family unless later screens add other runtime interfaces.",
            "- The primary capacity axis is `head_outputs_actual`, the model prediction head budget.",
        ]
    )
    return "\n".join(lines) + "\n"


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

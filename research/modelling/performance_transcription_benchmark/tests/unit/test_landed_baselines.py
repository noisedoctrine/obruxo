from __future__ import annotations

import json
from pathlib import Path

from obruxo_performance_benchmark.report import (
    _landed_basic_pitch_quality,
    _landed_basic_pitch_runtime,
    _markdown,
)

ROOT = Path(__file__).resolve().parents[2]


def test_landed_quality_is_exposed_as_both_required_views() -> None:
    views = _landed_basic_pitch_quality(
        {
            "aggregate": {
                "pair_count": 4,
                "successful_pair_count": 3,
                "failed_pair_count": 1,
                "micro": {"onset_pitch": {"f1": 0.25}},
            },
        }
    )
    assert sorted(views) == ["failure_penalized", "success_only"]
    assert views["success_only"]["coverage"] == 0.75
    assert views["failure_penalized"]["failed_pairs"] == 1
    assert views["success_only"]["aggregate"]["micro"]["onset_pitch"]["f1"] == 0.25


def test_landed_runtime_preserves_route_failure_without_global_fallback() -> None:
    runtime = _landed_basic_pitch_runtime(
        {
            "config": {"routes": {"inference": ["pytorch_cpu"]}},
            "inference": [{"route": "pytorch_cpu", "status": "ok"}],
            "training": [{"route": "pytorch_xpu", "status": "parity_failed"}],
        }
    )
    assert runtime["status"] == "measured"
    assert runtime["routes"][1]["status"] == "parity_failed"
    assert runtime["route_failures"] == [{"route": "pytorch_xpu", "status": "parity_failed"}]


def test_comparison_report_distinguishes_baseline_from_blocked_comparison() -> None:
    report = json.loads((ROOT / "reports" / "model_comparison.json").read_text(encoding="utf-8"))
    markdown = _markdown(report)
    assert "incomplete_alternatives_unavailable" in markdown
    assert "What was actually executed" in markdown
    assert "What could not be executed" in markdown
    assert "Adapter implementation scope" in markdown
    assert "implemented pinned-official adapter path" in markdown
    assert "The intended comparative benchmark remains incomplete" in markdown
    assert "No ranking, quality estimate, cost estimate" in markdown
    assert "measured_corrected_fp32_performance" in markdown
    assert "Historical pre-fix/default result" in markdown
    assert "Bounded diagnostic result (corrected)" in markdown
    assert "Corrected measured result" in markdown
    assert "corrected FP32 startup, throughput, end-to-end, and resource measurements remain `not_run`" not in markdown
    assert "including the OpenVINO GPU parity failure" not in markdown

    execution = next(model for model in report["models"] if model["model_id"] == "basic_pitch")["execution"]
    assert execution["measurement_status"]["post_fix_parity"] == "passed"
    assert execution["measurement_status"]["post_fix_timing"] == "measured"
    assert execution["openvino_parity_history"]["routes"][-1]["status"] == "parity_failed"
    assert execution["openvino_precision_diagnostic"]["status"] == "parity_passed"

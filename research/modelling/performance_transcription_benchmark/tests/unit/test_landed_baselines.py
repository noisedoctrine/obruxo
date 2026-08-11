from __future__ import annotations

from obruxo_performance_benchmark.report import (
    _landed_basic_pitch_quality,
    _landed_basic_pitch_runtime,
)


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

from __future__ import annotations

import pytest
from obruxo_performance_benchmark.report import (
    ReportPrivacyError,
    sanitize_public_report,
)


def test_public_serializer_rejects_private_identifiers_and_paths() -> None:
    with pytest.raises(ReportPrivacyError):
        sanitize_public_report({"pair_id": "pair-12345678"})
    with pytest.raises(ReportPrivacyError):
        sanitize_public_report({"source": "C:\\Users\\example\\datasets\\private.wav"})


def test_public_serializer_accepts_public_model_identity() -> None:
    value = sanitize_public_report(
        {
            "model": "MuScriptor/muscriptor-small",
            "sha256": "a" * 64,
            "status": "unavailable",
        }
    )
    assert value["status"] == "unavailable"


def test_public_serializer_omits_per_preset_aggregate() -> None:
    value = sanitize_public_report(
        {
            "quality": {
                "aggregate": {
                    "per_preset": {"private-preset": {"f1": 0.5}},
                    "micro": {"onset_pitch": {"f1": 0.25}},
                }
            }
        }
    )
    assert "per_preset" not in value["quality"]["aggregate"]
    assert value["quality"]["aggregate"]["micro"]["onset_pitch"]["f1"] == 0.25

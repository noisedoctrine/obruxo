from __future__ import annotations

import json

import pytest

from obruxo_data.errors import Diagnostic, Severity, ValidationError, ValidationReport
from obruxo_data.hashing import canonical_json, canonical_sha256
from obruxo_data.render import RendererCapabilities


def test_canonical_json_is_stable_and_rejects_non_finite_values() -> None:
    assert canonical_json({"b": 2, "a": "é"}) == '{"a":"é","b":2}'
    assert canonical_sha256({"b": 2, "a": 1}) == canonical_sha256({"a": 1, "b": 2})
    with pytest.raises(ValueError):
        canonical_json({"value": float("nan")})


def test_validation_report_is_structured() -> None:
    report = ValidationReport((Diagnostic("bad.value", Severity.ERROR, "bad value", parameter="volume"),))
    assert not report.valid
    assert json.loads(json.dumps(report.to_dict()))["diagnostics"][0]["parameter"] == "volume"
    with pytest.raises(ValidationError):
        report.require_valid()


def test_renderer_capabilities_round_trip() -> None:
    capabilities = RendererCapabilities(control_changes=frozenset({1, 64}))
    assert RendererCapabilities.from_dict(capabilities.to_dict()) == capabilities

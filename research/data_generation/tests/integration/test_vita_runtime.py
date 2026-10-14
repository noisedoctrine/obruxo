from __future__ import annotations

import json
from pathlib import Path

import pytest

from obruxo_data.vital import VitalPreset, VitalSchema


pytestmark = [pytest.mark.integration, pytest.mark.vita]
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.mark.parametrize("version", ["99999.9.9", "1.0.8"])
def test_canonical_fixture_round_trips_through_pinned_vita(version) -> None:
    pytest.importorskip("vita")
    document = VitalPreset.init().to_dict()
    document["synth_version"] = version
    preset = VitalPreset(document, VitalSchema.load())
    preset.set_raw("osc_1_level", 0.25)
    report = preset.validate(runtime=True)
    assert report.valid, report.to_dict()
    canonicalizations = [item for item in report.diagnostics if item.code == "vital.runtime.canonicalization"]
    assert [item.pointer for item in canonicalizations] == ["/settings/sample/samples"]


def test_missing_scalar_fixture_canonicalizes_explicitly() -> None:
    vita = pytest.importorskip("vita")
    source = (FIXTURES / "vital" / "missing_scalar_canonicalization.vital").read_text(encoding="utf-8")
    synth = vita.Synth()
    assert synth.load_json(source)
    canonical = json.loads(synth.to_json())
    assert canonical["settings"]["osc_2_level"] == VitalPreset.init().get_raw("osc_2_level")
    assert VitalPreset(canonical, VitalSchema.load()).validate().valid

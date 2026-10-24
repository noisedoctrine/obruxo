import copy

import pytest
import torch

from obruxo_data.hashing import canonical_sha256
from obruxo_training import paired_report_recovery as recovery
from obruxo_training.paired_report_recovery import normalized_audit, scoped_plan


def test_identity_bridge_changes_only_the_declared_metadata_field():
    plan = {"limits": {"wall_seconds": 21600, "archive_bytes": 4000}, "numeric_source": "frozen"}
    manifest = {"limits": {"wall_seconds": 86400, "archive_bytes": 5000}, "operational_archive_reserve_bytes": 10}
    audio = torch.tensor([[.1, -.2], [.3, .4]], dtype=torch.float32)
    saved = {"contract": {"plan_id": canonical_sha256(scoped_plan(plan, manifest)), "recipe": {"state": "same"}, "identity": "same"},
             "audio": audio, "rows": {"spectral": {"scores": {"spectral": .3}, "candidate": [1, 2], "receipt": "same"}}}
    before = copy.deepcopy(saved)
    normalized = normalized_audit(saved, plan, manifest)
    assert normalized["contract"]["plan_id"] == canonical_sha256(plan)
    assert normalized["contract"]["recipe"] == before["contract"]["recipe"]
    assert normalized["contract"]["identity"] == before["contract"]["identity"]
    assert normalized["rows"] == before["rows"]
    assert torch.equal(normalized["audio"], before["audio"])
    assert saved["contract"] == before["contract"]


@pytest.mark.parametrize("alias", ["unknown", "numeric"])
def test_identity_bridge_rejects_an_unrecognized_cached_plan(alias):
    plan, manifest = {"limits": {"wall_seconds": 1}}, {"limits": {"wall_seconds": 2, "archive_bytes": 4}, "operational_archive_reserve_bytes": 1}
    saved = {"contract": {"plan_id": canonical_sha256(plan) if alias == "numeric" else alias}}
    with pytest.raises(ValueError, match="declared operational plan"):
        normalized_audit(saved, plan, manifest)


def test_report_only_execution_blocks_training_and_restores_its_scopes(monkeypatch):
    calls = []
    class Session:
        def render(self, target, performance, phase, label):
            calls.append((target, performance, phase, label))
            return "cached-or-qualified"
    def fit(*args, **kwargs):
        calls.append((args, kwargs))
        return "final-state-restored"
    monkeypatch.setattr(recovery.fit, "PairedSession", Session)
    monkeypatch.setattr(recovery.fit, "fit_pairs", fit)
    with recovery.report_only_execution():
        session = recovery.fit.PairedSession()
        with pytest.raises(RuntimeError, match="cannot render new training"):
            session.render(None, None, "training", "gradient")
        assert session.render(None, None, "qualification", "fit") == "cached-or-qualified"
        with pytest.raises(ValueError, match="earlier checkpoint"):
            recovery.fit.fit_pairs(None, None, None, None, {"updates": [1, 2]}, {"step": 1, "trace": [1]})
        assert recovery.fit.fit_pairs(None, None, None, None, {"updates": [1, 2]}, {"step": 2, "trace": [1, 2]}) == "final-state-restored"
    assert recovery.fit.PairedSession is Session and recovery.fit.fit_pairs is fit
    assert len(calls) == 2

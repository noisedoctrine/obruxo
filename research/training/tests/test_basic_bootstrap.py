import json
from time import time

import pytest

pytest.importorskip("torch")

from obruxo_training import basic_bootstrap as bootstrap  # noqa: E402
from obruxo_training.pilot import PilotPool  # noqa: E402
from obruxo_training.profiled_sampler import ProfiledSampler  # noqa: E402
from test_profiled_bootstrap import ProfileRendererFactory  # noqa: E402


def test_frozen_preflight_is_train_only_costed_and_detects_changed_plan(tmp_path, monkeypatch):
    pool = PilotPool.create(24, seed=0)
    qualification = tmp_path / "qualification.json"
    qualification.write_text(json.dumps({"renderer_identity": {"fixture": True}}))
    calls = []
    monkeypatch.setattr(bootstrap, "require_qualification", lambda report, **kwargs: calls.append(kwargs))
    output = tmp_path / "frozen"
    result = bootstrap.run_preflight(pool, qualification, output)
    assert result == bootstrap.run_preflight(pool, qualification, output)
    assert not (output / "run.json").exists()
    plan = json.loads((output / "plan.json").read_text())
    assert calls == [{"preroll_frames": 32768}] * 2
    assert plan["planned_requests"]["total"] == 8 + 8 * 10 == 88
    assert plan["config"]["steps"] == plan["limits"]["optimizer_updates"] == 800
    assert plan["config"]["checkpoint_every"] == 50
    assert len({row["state_id"] for row in plan["sources"]}) == 8
    assert all(pool.manifest.by_state[row["state_id"]].split == "train" for row in plan["sources"])
    plan["config"]["learning_rate"] *= 2
    (output / "plan.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="frozen bootstrap plan changed"):
        bootstrap.run_preflight(pool, qualification, output, execute=True)
    assert not (output / "run.json").exists()


def test_audit_charges_before_renderer_start_and_refuses_budget_or_expired_original_deadline(tmp_path):
    plan = {"limits": {"wall_seconds": 1800, "training_attempts": 1}}
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    (tmp_path / "run.json").write_text(json.dumps({"started_unix": time()}))
    pool = PilotPool.create(24, seed=0)
    request = ProfiledSampler(pool, preroll_frames=32768).recipe_at(0).render_request()
    renderer = bootstrap.AuditedFactory(str(tmp_path), "training", ProfileRendererFactory())()
    result = renderer.render(request)
    assert result.provenance.request_id == request.request_id
    assert len(bootstrap.rows(tmp_path / "attempts.jsonl")) == 1
    with pytest.raises(RuntimeError, match="attempt budget"):
        renderer.render(request)
    (tmp_path / "run.json").write_text(json.dumps({"started_unix": time() - 1801}))
    with pytest.raises(RuntimeError, match="original wall-time budget"):
        renderer.render(request)
    assert len(bootstrap.rows(tmp_path / "attempts.jsonl")) == 1

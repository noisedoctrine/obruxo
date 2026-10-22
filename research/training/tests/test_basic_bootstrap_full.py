import json

import pytest

pytest.importorskip("torch")

from obruxo_data.hashing import canonical_sha256, sha256_bytes  # noqa: E402
from obruxo_training import basic_bootstrap, basic_bootstrap_full  # noqa: E402
from obruxo_training.pilot import PilotPool  # noqa: E402


@pytest.fixture(scope="module")
def pool():
    return PilotPool.create(256, seed=0).expand(512)


def prior_evidence(tmp_path, pool):
    root = tmp_path / "prior"
    root.mkdir()
    plan = {"pool_id": pool.pool_id}
    summary = {"complete": True, "preflight_passed": True, "plan_id": canonical_sha256(plan), "attempts": 88}
    (root / "plan.json").write_text(json.dumps(plan))
    (root / "summary.json").write_text(json.dumps(summary))
    (root / "fit-000").mkdir()
    for name in ("checkpoint.pt", "checkpoint.initial.pt", "model.pt"):
        (root / "fit-000" / name).write_bytes(b"fixture")
    audit = {"plan_id": summary["plan_id"], "model_optimizer_batch_and_global_rng_exact": True, "replayed_updates": 800,
             "native_renders_added": 0, "all_eighty_audit_requests_reconstructed": True,
             "evidence_sha256": {path.relative_to(root).as_posix(): sha256_bytes(path.read_bytes()) for path in root.rglob("*") if path.is_file()}}
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(audit))
    return root, path


def test_full_plan_preserves_small_sources_reserves_held_out_data_and_costs_recovery(tmp_path, pool, monkeypatch):
    prior, audit = prior_evidence(tmp_path, pool)
    qualification = tmp_path / "qualification.json"
    qualification.write_text(json.dumps({"renderer_identity": {"fixture": True}}))
    monkeypatch.setattr(basic_bootstrap, "require_qualification", lambda *_args, **_kwargs: None)
    plan = basic_bootstrap_full.make_plan(pool, qualification, prior, audit)
    small = basic_bootstrap.make_plan(pool, qualification)
    assert plan["sources"][:8] == small["sources"]
    assert len({row["state_id"] for row in plan["sources"]}) == 404
    assert all(pool.manifest.by_state[row["state_id"]].split == "train" for row in plan["sources"])
    assert plan["planned_requests"] == {"training": 404, "evaluation": 80, "total": 484}
    assert plan["limits"]["training_attempts"] + plan["limits"]["evaluation_attempts"] == 496
    assert plan["config"]["steps"] == plan["limits"]["optimizer_updates"] == 16000
    assert plan["config"]["checkpoint_every"] == 250 and plan["config"]["preroll_frames"] == 32768
    assert plan["config"]["prediction_scope"] == "basic" and plan["gates"]["parameter_fit"]["continuous_mae_below"] == .06
    result = basic_bootstrap_full.run_full(pool, qualification, prior, audit, tmp_path / "frozen")
    assert result == {"plan_id": canonical_sha256(plan), "executed": False}
    assert not (tmp_path / "frozen/run.json").exists()


def test_full_plan_refuses_failed_or_changed_preflight_before_qualification(tmp_path, pool, monkeypatch):
    prior, audit = prior_evidence(tmp_path, pool)

    def unexpected(*_args, **_kwargs):
        raise AssertionError("qualification must not run before the prior evidence checks")

    monkeypatch.setattr(basic_bootstrap, "make_plan", unexpected)
    saved = json.loads((prior / "summary.json").read_text())
    (prior / "summary.json").write_text(json.dumps({**saved, "preflight_passed": False}))
    with pytest.raises(ValueError, match="completed small fit"):
        basic_bootstrap_full.make_plan(pool, tmp_path / "absent.json", prior, audit)
    (prior / "summary.json").write_text(json.dumps(saved))
    (prior / "fit-000/model.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="evidence changed"):
        basic_bootstrap_full.make_plan(pool, tmp_path / "absent.json", prior, audit)

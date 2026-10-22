from dataclasses import asdict
import json
from time import time

import pytest

from obruxo_data.hashing import canonical_sha256, sha256_bytes
from obruxo_training import varied_input_check as check
from obruxo_training.pilot import PilotPool
from obruxo_training.profiled_sampler import ProfiledSampler
from obruxo_training.trainer import TrainConfig
from test_stream import FakeRendererFactory


def test_plan_freezes_training_only_balanced_stress_cases_and_audited_parity(tmp_path, monkeypatch):
    monkeypatch.setattr("obruxo_training.native_fit.require_qualification", lambda *args, **kwargs: None)
    pool = PilotPool.create(96, seed=0)
    prior = tmp_path / "prior"
    (prior / "fit-000").mkdir(parents=True)
    old = {"pool_id": pool.pool_id, "config": asdict(TrainConfig(prediction_scope="basic", preroll_frames=32768))}
    (prior / "plan.json").write_text(json.dumps(old))
    (prior / "summary.json").write_text(json.dumps({"plan_id": canonical_sha256(old), "complete": True, "preflight_passed": True}))
    sampler = ProfiledSampler(pool, preroll_frames=32768, policy="pairs")
    receipts = [{"sequence_id": i, "recipe_id": sampler.recipe_at(i).recipe_id, "status": "accepted",
                 "provenance": {"request_id": sampler.recipe_at(i).render_request().request_id},
                 "qa": {"audio_float32_sha256": "fixture"}} for i in range(4)]
    (prior / "fit-000/renders.jsonl").write_text("\n".join(json.dumps(row) for row in receipts))
    audit_path, qualification_path = tmp_path / "audit.json", tmp_path / "qualification.json"
    audit = {"plan_id": canonical_sha256(old), "model_optimizer_batch_and_global_rng_exact": True, "replayed_updates": 16000,
             "native_renders_added": 0, "evidence_sha256": {name: sha256_bytes((prior / name).read_bytes())
                                                           for name in ("plan.json", "summary.json", "fit-000/renders.jsonl")}}
    audit_path.write_text(json.dumps(audit))
    qualification_path.write_text(json.dumps({"renderer_identity": {"fixture": True}}))
    plan = check.make_plan(pool, qualification_path, prior, audit_path)
    varied = plan["requests"][:72]
    assert len(plan["requests"]) == plan["planned_native_requests"] == plan["limits"]["native_attempts"] == 76
    assert plan["optimizer_updates"] == 0 and not plan["test_split_used"]
    assert len({row["state_id"] for row in varied}) == 18
    assert len({row["request_id"] for row in varied}) == 36
    assert len({(tuple(row["category"]), row["pattern"]) for row in varied if row["pattern"] != "held"}) == 18
    assert all(pool.manifest.by_state[row["state_id"]].split == "train" for row in varied)
    output = tmp_path / "frozen"
    assert not check.run_check(plan, output)["executed"]
    assert not (output / "run.json").exists()
    changed = {**plan, "optimizer_updates": 1}
    with pytest.raises(ValueError, match="frozen varied-input plan changed"):
        check.run_check(changed, output, execute=True)
    (prior / "fit-000/renders.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="evidence changed"):
        check.make_plan(pool, qualification_path, prior, audit_path)


def tiny_plan():
    request = ProfiledSampler(PilotPool.create(12, seed=0), preroll_frames=32768).recipe_at(0).render_request()
    spec = {"kind": "varied", "request": request.to_dict(), "request_id": request.request_id}
    audio = FakeRendererFactory()().render(request).audio
    return {"requests": [spec, spec, {**spec, "kind": "historical_parity", "expected_audio_sha256": sha256_bytes(audio.tobytes())}],
            "limits": {"native_attempts": 3, "wall_seconds": 60, "process_tree_bytes": 4 * 1024**3, "archive_bytes": 64 * 1024**2},
            "renderer_identity": {"renderer_id": request.renderer_id, "backend_version": "test-fixture", "engine_fingerprint": "fixture"}}


def test_native_check_uses_actual_audio_hashes_and_recovers_without_new_requests(tmp_path):
    plan = tiny_plan()
    summary = check.run_check(plan, tmp_path / "run", execute=True, renderer_factory=FakeRendererFactory())
    assert summary["passed"] and summary["accepted"] == 3 and summary["repeat_pairs"] == 1
    assert summary["all_historical_hashes_exact"] and summary["optimizer_updates"] == 0
    summary2 = check.run_check(plan, tmp_path / "run", execute=True, resume=True, renderer_factory=FakeRendererFactory("crash"))
    assert summary2["passed"] and len(check.rows(tmp_path / "run/attempts.jsonl")) == 3


def test_charged_failed_requests_and_original_deadline_cannot_be_erased(tmp_path):
    plan, output = tiny_plan(), tmp_path / "failed"
    with pytest.raises(RuntimeError, match="audio QA"):
        check.run_check(plan, output, execute=True, renderer_factory=FakeRendererFactory("silence"))
    assert len(check.rows(output / "attempts.jsonl")) == len(check.rows(output / "renders.jsonl")) == 1
    with pytest.raises(ValueError, match="failed request history"):
        check.run_check(plan, output, execute=True, resume=True)
    fresh = tmp_path / "expired"
    check.run_check(plan, fresh)
    (fresh / "run.json").write_text(json.dumps({"started_unix": time() - 61}))
    with pytest.raises(RuntimeError, match="original deadline"):
        check.run_check(plan, fresh, execute=True, resume=True)
    assert not (fresh / "attempts.jsonl").exists()
    (fresh / "run.json").write_text(json.dumps({"started_unix": time()}))
    check.append(fresh / "attempts.jsonl", {"index": 0, "request_id": plan["requests"][0]["request_id"]})
    with pytest.raises(ValueError, match="unresolved charged attempt"):
        check.run_check(plan, fresh, execute=True, resume=True)

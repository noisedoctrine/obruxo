import copy
import json
import os
import shutil

import psutil
import pytest
import torch

from obruxo_data.hashing import canonical_sha256
from obruxo_training import basic_direct_fit as fit
from obruxo_training import final_fit_completion as completion
from obruxo_training.audit import write_json
from obruxo_training.basic_direct_fit_audit import verify
from obruxo_training.conditioned_bridge import read
from obruxo_training.native_fit import digest
from obruxo_training.paired_native_audit import equal_tree
from obruxo_training.paired_native_evidence import native_ledger
from test_basic_direct_fit import FakeSession, completed  # noqa: F401
from test_native_acceptance_fit import accepted  # noqa: F401


def test_process_loss_requires_explicit_certificate_and_identical_accepted_repeat():
    identity = {"phase": "qualification", "attempt": 0, "label": "final:000:spectral", "request_id": "request",
                "source_sha256": "source", "controlled_sha256": "controlled"}
    loss = {**identity, "observation_evidence_sha256": "frozen-observation"}
    events = [{"event": "attempt", **identity}, {"event": "outcome", **loss, "status": "unobserved", "error_type": "ProcessLostBeforeReceipt"},
              {"event": "attempt", **identity, "attempt": 1},
              {"event": "outcome", **identity, "attempt": 1, "status": "accepted", "renderer_identity": {"simulation": True}}]
    with pytest.raises(ValueError, match="unsupported"):
        native_ledger(events, {"simulation": True})
    _attempts, _outcomes, recovery = native_ledger(events, {"simulation": True}, known_process_losses=[loss])
    assert recovery == [{"kind": "process_loss", "phase": "qualification", "unobserved_attempt": 0, "accepted_attempt": 1, "request_id": "request"}]
    for change in ("qa", "certificate", "repeat", "training"):
        modified, declared = copy.deepcopy(events), copy.deepcopy(loss)
        if change == "qa":
            modified[1]["qa"] = {}
        elif change == "certificate":
            modified[1]["observation_evidence_sha256"] = "forged"
        elif change == "repeat":
            modified = modified[:2]
        else:
            for row in modified:
                row["phase"] = "training"
            declared["phase"] = "training"
        with pytest.raises(ValueError, match="process.loss|frozen"):
            native_ledger(modified, {"simulation": True}, known_process_losses=[declared])


def test_live_original_process_is_refused():
    with pytest.raises(ValueError, match="still live"):
        completion.require_gone({"process_id": os.getpid(), "process_created_unix": psutil.Process().create_time()})


@pytest.fixture(scope="module")
def recovered(accepted, tmp_path_factory):  # noqa: F811
    original, finished, _prior = accepted
    root = tmp_path_factory.mktemp("final-only-completion")
    source, output, cache = root / "interrupted", root / "completed", root / "features.pt"
    shutil.copytree(finished, source)
    cache.write_bytes(b"simulation input cache")
    plan = {**original.plan, "source": {}, "input_cache_sha256": digest(cache), "initial_file_sha256": digest(original.initial_path)}
    write_json(source / "plan.json", plan)
    plan_id = canonical_sha256(plan)
    for name in ("checkpoint.pt", "prefix-checkpoint.pt"):
        saved = torch.load(source / name, map_location="cpu", weights_only=True)
        saved["plan_id"] = plan_id
        torch.save(saved, source / name)
    prefix = read(source / "prefix-verification.json")
    prefix.update(plan_id=plan_id, checkpoint_file_sha256=digest(source / "prefix-checkpoint.pt"))
    write_json(source / "prefix-verification.json", prefix)
    write_json(source / "run.json", {"plan_id": plan_id, "started_unix": 1})
    write_json(source / "service-start-001.json", {"mode": "execute", "process_id": 2147483647, "process_created_unix": 0})
    for name in ("summary.json", "diagnostic-parameter.pt", "diagnostic-spectral.pt"):
        (source / name).unlink()
    rows = fit.lines(source / "finals.jsonl")
    (source / "finals.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows[:3]), encoding="utf-8")
    events = fit.lines(source / "renders.jsonl")
    kept = [row for row in events if row["phase"] == "training" or row["attempt"] < 3 or (row["attempt"] == 3 and row["event"] == "attempt")]
    (source / "renders.jsonl").write_text("".join(json.dumps(row) + "\n" for row in kept), encoding="utf-8")
    executions = fit.lines(source / "executions.jsonl")
    (source / "executions.jsonl").write_text("".join(json.dumps({**row, "plan_id": plan_id}) + "\n" for row in executions), encoding="utf-8")

    def inputs(_parent, _bridge, _cache, current_plan):
        result = copy.copy(original)
        result.plan = current_plan
        return result

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(completion, "Inputs", inputs)
        patch.setattr(completion, "PairedSession", FakeSession)
        prepared = completion.run(source, original.parent, original.initial_path.parent, cache, output)
        before = completion.source_manifest(source)
        result = completion.run(source, original.parent, original.initial_path.parent, cache, output, mode="execute")
        assert result["complete"] and result["native_receipts_resolved"]
        assert completion.source_manifest(source) == before
        ctx = inputs(None, None, None, read(output / "plan.json"))
        proof = verify(ctx, output)
    return original, source, output, prepared, proof


def test_completion_only_renders_missing_finals_and_preserves_training_state(recovered):
    _inputs, source, output, prepared, proof = recovered
    assert prepared["experimental_training_updates"] == 0
    assert prepared["observation"]["training_model_adam_rng_replay_exact"]
    assert proof["comparison_waveforms_rescored"] == 24 and proof["experimental_training_updates_added"] == 0
    assert proof["original_budget_passed"] is False and proof["original_peak_process_tree_rss_bytes"] is None
    assert len(proof["resolved_process_losses"]) == 1 and not proof["qualified_for_release"]
    assert "process-loss-observation.json" in proof["evidence_sha256"]
    old = torch.load(source / "checkpoint.pt", map_location="cpu", weights_only=True)
    new = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert all(equal_tree(old[key], new[key]) for key in ("models", "optimizers", "rng", "trace"))
    assert new["step"] == old["step"] == 4
    old_events, new_events = fit.lines(source / "renders.jsonl"), fit.lines(output / "renders.jsonl")
    added = [row for row in new_events[len(old_events):] if row["event"] == "attempt"]
    assert len(added) == prepared["observation"]["missing_comparisons"] == 13
    assert all(row["phase"] == "qualification" for row in added)
    for name in ("probes.jsonl", "updates.jsonl"):
        assert (source / name).read_bytes() == (output / name).read_bytes()


def test_completion_rejects_duplicate_execution_and_forged_loss_declaration(recovered):
    _inputs, _source, output, _prepared, _proof = recovered
    plan = read(output / "plan.json")
    with pytest.raises(ValueError, match="second migration"):
        completion.migrate(plan, output)
    changed = copy.deepcopy(plan)
    changed["final_only_completion"]["known_process_losses"][0]["request_id"] = "forged"
    with pytest.raises(ValueError, match="certificate"):
        completion.verify_origin(changed, output)


def test_completion_rejects_changed_saved_training_state(recovered, tmp_path):
    _inputs, _source, output, _prepared, _proof = recovered
    changed = tmp_path / "changed-completion"
    shutil.copytree(output, changed)
    checkpoint = torch.load(changed / "checkpoint.pt", map_location="cpu", weights_only=True)
    checkpoint["models"]["spectral"]["correction.2.bias"][0] += .1
    torch.save(checkpoint, changed / "checkpoint.pt")
    with pytest.raises(ValueError, match="training tensors"):
        completion.verify_origin(read(changed / "plan.json"), changed)

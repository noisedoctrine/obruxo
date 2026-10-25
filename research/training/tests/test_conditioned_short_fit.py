import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from obruxo_data.midi import Performance  # noqa: E402
from obruxo_data.render.qa import audio_float32_sha256  # noqa: E402
from obruxo_training import paired_native_audit as audit  # noqa: E402
from obruxo_training import paired_native_fit as fit  # noqa: E402
from obruxo_training.audit import write_json  # noqa: E402
from obruxo_training.codec import PilotCodec  # noqa: E402
from obruxo_training.conditioned_basic import make_conditioned  # noqa: E402
from obruxo_training.conditioned_fit_policy import learning_access, require_frozen  # noqa: E402
from obruxo_training.conditioned_short_audit import CELLS, short_summary  # noqa: E402
from obruxo_training.conditioned_short_fit import audit_cells, charge_execution, select_recipes, storage_bound  # noqa: E402
from obruxo_training.features import tensor_digest  # noqa: E402
from obruxo_training.model import PilotInverse, batch_features  # noqa: E402
from obruxo_training.paired_native_step import reciprocal_updates  # noqa: E402
from obruxo_training.paired_performance_objective import INPUT_TRANSFER  # noqa: E402
from obruxo_training.profiled_sampler import profile_metadata  # noqa: E402
from test_paired_native_fit import FakeSession, Frontend, waveform  # noqa: E402


@pytest.fixture
def completed(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    torch.manual_seed(12)
    codec, frontend = PilotCodec(), Frontend()
    model = make_conditioned(PilotInverse().eval(), codec)
    bridge = tmp_path / "bridge"
    (bridge / "fit-000").mkdir(parents=True)
    initial, retrieval = bridge / "fit-000/model.pt", tmp_path / "retrieval.pt"
    torch.save({"model": model.state_dict(), "basic_pitch": frontend.basic_pitch.state_dict(),
                "conditioning_training_policy": model.conditioning_training_policy}, initial)
    features = frontend(None)
    torch.save({"replay": [{"features": features, "target": codec.default_target().to_dict()}]}, retrieval)
    performances = [Performance(bpm=value).to_dict() for value in (100, 120, 140)]
    recipe = {"state_id": "fixture", "target": codec.default_target().to_dict(), "input_views": "all", "performances": performances,
              "source_audio_sha256": [audio_float32_sha256(waveform(codec.default_target(), Performance.from_dict(value))) for value in performances],
              "source_feature_sha256": [tensor_digest(features)] * 3, "source_recipe_ids": ["input-0", "input-1", "input-2"]}
    cache = tmp_path / "features.pt"
    torch.save({"replay": [{"recipe_id": recipe["source_recipe_ids"][view], "state_id": recipe["state_id"], "target": recipe["target"],
                           "features": features, "audio_sha256": recipe["source_audio_sha256"][view]} for view in range(3)]}, cache)
    plan = {"format": "simulation_not_native_quality_evidence", "training": [recipe], "validation": [], "fit_audit_cells": audit_cells([recipe], [recipe]),
            "regression_state_ids": ["fixture"], "training_view_roles": ["original_input", "wide_held_input", "wide_alternate_input"],
            "updates": reciprocal_updates(1), "production_training": {"forward": "unperturbed_argmax", "parameter_logits": "raw", "scales": [32., 29.]},
            "prefix_check_updates": 1, "optimizer": {"learning_rate": .001, "seed": 0}, "reconstruction_objective": INPUT_TRANSFER,
            "rng_initialization": "python_numpy_and_torch_from_optimizer_seed", "limits": {"optimizer_attempts": 3, "training_attempts": 256,
                "qualification_attempts": 32, "wall_seconds": 600, "process_tree_bytes": 4 * 1024**3, "archive_bytes": 256 * 1024**2,
                "replay_bytes": 32 * 1024**2, "replay_records": 8}, "expected_renderer_identity": {"fixture": True},
            "initial_file_sha256": fit.digest(initial), "retrieval_file_sha256": fit.digest(retrieval), **profile_metadata(32768)}
    def restore(saved):
        restored = PilotInverse(conditioning="cross_attention")
        restored.load_state_dict(saved["model"])
        return restored, Frontend(saved["basic_pitch"]), codec
    def export(network, front, _codec, _pool, **_kwargs):
        return {"model": network.state_dict(), "basic_pitch": front.basic_pitch.state_dict()}
    monkeypatch.setattr(fit, "restore_model", restore)
    monkeypatch.setattr(audit, "restore_model", restore)
    monkeypatch.setattr(fit, "inference_state", export)
    monkeypatch.setattr(fit, "PairedSession", FakeSession)
    monkeypatch.setattr(fit, "MemoryMonitor", lambda: SimpleNamespace(peak_bytes=1024**2, errors=0, start=lambda: None, close=lambda: None))
    pool, output = SimpleNamespace(codec=codec, pool_id="fixture"), tmp_path / "fit"
    fit.run_plan(pool, initial, retrieval, output, plan)
    assert fit.run_plan(pool, initial, retrieval, output, plan, execute=True, input_cache_path=cache)["complete"]
    return output, pool, bridge, cache, plan, model


def test_conditioned_runner_and_full_auditor_keep_frozen_weights_and_score_transfer(completed):
    output, pool, bridge, cache, plan, initial = completed
    result = audit.verify(output, pool, bridge, None, None, cache, None, None, expected_plan=plan)
    assert result["offline_training_replay_exact"] and result["comparison_waveforms_rescored"] == 21
    assert result["native_requests_added"] == result["optimizer_updates_added"] == 0
    assert len(result["fit_result"]["groups"]) == 6
    saved = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert learning_access(saved["trace"])["pass"]
    for arm in ("parameter", "spectral"):
        model = copy.deepcopy(initial).train()
        model.load_state_dict(saved["models"][arm])
        require_frozen(model)
        assert all(torch.equal(initial.state_dict()[name], model.state_dict()[name]) for name in initial.conditioning_training_policy["frozen_names"])
        values = batch_features([{"features": Frontend()(None)}])
        before = fit.predict_candidates({"trained": model}, pool.codec, Frontend()(None))
        raw = model(values)
        model.eval()
        other = model(values)
        assert torch.equal(raw["continuous"], other["continuous"])
        assert all(torch.equal(a, b) for a, b in zip(raw["categorical"], other["categorical"], strict=True))
        assert before == fit.predict_candidates({"trained": model}, pool.codec, Frontend()(None))


def test_conditioned_checkpoint_resume_keeps_mask_adam_and_rng_exact(completed, monkeypatch):
    source, pool, bridge, cache, plan, _initial = completed
    output = source.parent / "recovered"
    initial, retrieval = bridge / "fit-000/model.pt", bridge.parent / "retrieval.pt"
    fit.run_plan(pool, initial, retrieval, output, plan)
    original, failed = FakeSession.render, False
    def render(session, candidate, performance, phase, label):
        nonlocal failed
        if not failed and label.startswith("gradient:1:1:"):
            failed = True
            identity = session.identity(candidate, performance, phase, label)
            session._append({"event": "attempt", **identity})
            receipt = {"event": "outcome", **identity, "status": "error", "error_type": "RenderTimeout", "renderer_identity": session.renderer_identity}
            session._append(receipt)
            return None, receipt
        return original(session, candidate, performance, phase, label)
    monkeypatch.setattr(FakeSession, "render", render)
    with pytest.raises(RuntimeError, match="not accepted"):
        fit.run_plan(pool, initial, retrieval, output, plan, execute=True, input_cache_path=cache)
    assert torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)["step"] == 1
    assert fit.run_plan(pool, initial, retrieval, output, plan, execute=True, resume=True, input_cache_path=cache)["complete"]
    result = audit.verify(output, pool, bridge, None, None, cache, None, None, expected_plan=plan)
    assert result["offline_training_replay_exact"] and len(result["recovered_timeouts"]) == 1
    assert result["interruption_checkpoints_replayed"] == [1]


@pytest.mark.parametrize("change", ["input_view", "groups", "frozen_export", "policy_export", "source_features"])
def test_short_auditor_rejects_changed_input_gradients_and_frozen_export(completed, change):
    output, pool, bridge, cache, plan, _initial = completed
    if change == "input_view":
        path = output / "fit_cases.json"
        rows = json.loads(path.read_text())
        rows[1]["input_view"] = 1
        write_json(path, rows)
    elif change == "groups":
        path = output / "checkpoint.pt"
        saved = torch.load(path, map_location="cpu", weights_only=True)
        saved["trace"][1]["correction_gradients"]["spectral"]["conditioning_attention"] = 0.
        torch.save(saved, path)
    elif change == "source_features":
        saved = torch.load(cache, map_location="cpu", weights_only=True)
        saved["replay"][0]["features"]["pitch"][0] += 1
        torch.save(saved, cache)
    else:
        path = output / "diagnostic-spectral.pt"
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if change == "frozen_export":
            saved["model"]["continuous.bias"][0] += 1
        else:
            del saved["conditioning_training_policy"]
        torch.save(saved, path)
    with pytest.raises(ValueError, match="changed|reconstruct|differs"):
        audit.verify(output, pool, bridge, None, None, cache, None, None, expected_plan=plan)


def test_selection_coverage_and_calculated_storage_bound_use_only_existing_training_states():
    parent = json.loads((Path(__file__).parents[1] / "reports/data/input_transfer_native_fit_plan.json").read_text())
    selected, regression = select_recipes(parent)
    assert len(selected) == 12 and len(regression) == 32
    for a in range(3):
        for b in range(2):
            assert [r["state_id"] for r in selected if r["target"]["categorical"] == [a, b]] == sorted(
                r["state_id"] for r in regression if r["target"]["categorical"] == [a, b])[:2]
    cells = audit_cells(selected, regression)
    assert len(cells) == 104 and sum(row["training_index"] is not None for row in cells) == 84
    assert len({(r["recipe"]["state_id"], r["input_view"], r["playback_view"]) for r in cells}) == 104
    assert set((r["input_view"], r["playback_view"]) for r in cells) == set(CELLS)
    bound = storage_bound(selected, cells, PilotCodec().contract.tail_seconds)
    assert bound["estimated_archive_upper_bound_bytes"] < 2 * 1024**3
    assert bound["candidate_waveforms_maximum"] == 312


def test_fit_gate_keeps_six_groups_and_original_regression_separate():
    recipe = {"state_id": "a"}
    cells = audit_cells([recipe], [recipe])
    scores = {"initial": {"spectral": 1., "mrstft": 1.}, "parameter": {"spectral": .9, "mrstft": .9}, "spectral": {"spectral": .8, "mrstft": .8}}
    cases = [{**cell, "state_id": "a", "view": cell["playback_view"], "repeat_exact": True,
              "rows": {name: {"scores": dict(value)} for name, value in scores.items()}} for cell in cells]
    gradients = {arm: {group: 1. for group in ("conditioning_attention", "conditioning_gain", "conditioning_projection", "performance_position",
                  "performance_projection", "timbre_position", "timbre_projection")} for arm in ("parameter", "spectral")}
    plan = {"fit_audit_cells": cells, "regression_state_ids": ["a"], "training_view_roles": ["original", "held", "alternate"]}
    trace = [{"correction_gradients": gradients}] * 2
    assert short_summary(cases, plan, trace)["pass"]
    cases[1]["rows"]["spectral"]["scores"]["spectral"] = 1.1
    result = short_summary(cases, plan, trace)
    assert not result["pass"] and not result["groups"]["original_transfer"]["pass"]
    assert result["original_regression"]["pass"]


def test_short_execution_limit_and_recovery_reason_cannot_be_overridden(tmp_path):
    plan = {"limits": {"native_executions": 2}}
    write_json(tmp_path / "plan.json", plan)
    charge_execution(tmp_path, plan, resume=False)
    write_json(tmp_path / "run.json", {"started_unix": 1.})
    summary = {"complete": False, "failure": "render failed", "within_resource_limits": True}
    write_json(tmp_path / "summary.json", summary)
    path = tmp_path / "renders.jsonl"
    path.write_text(json.dumps({"event": "outcome", "status": "error", "error_type": "AudioRejected"}) + "\n")
    with pytest.raises(ValueError, match="terminal known native timeout"):
        charge_execution(tmp_path, plan, resume=True)
    path.write_text(json.dumps({"event": "outcome", "status": "error", "error_type": "RenderTimeout"}) + "\n")
    charge_execution(tmp_path, plan, resume=True)
    with pytest.raises(ValueError, match="allocation exhausted"):
        charge_execution(tmp_path, plan, resume=True)

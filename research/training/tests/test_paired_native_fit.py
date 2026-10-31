import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
from render_fixtures import LifecycleReceipts

torch = pytest.importorskip("torch")

from obruxo_data.hashing import canonical_sha256, sha256_bytes  # noqa: E402
from obruxo_data.midi import Performance  # noqa: E402
from obruxo_data.render.base import RenderRequest  # noqa: E402
from obruxo_data.render.qa import audio_float32_sha256  # noqa: E402
from obruxo_training import paired_native_audit as audit  # noqa: E402
from obruxo_training import paired_native_fit as fit  # noqa: E402
from obruxo_training.audit import write_json  # noqa: E402
from obruxo_training.codec import PilotCodec  # noqa: E402
from obruxo_training.features import PITCH_SIZE, TIMBRE_SHAPE, tensor_digest  # noqa: E402
from obruxo_training.model import PilotInverse  # noqa: E402
from obruxo_training.native_fit import audit_summary as actual_summary  # noqa: E402
from obruxo_training.paired_native_evidence import native_ledger  # noqa: E402
from obruxo_training.paired_native_step import reciprocal_updates  # noqa: E402
from obruxo_training.paired_performance_objective import INPUT_TRANSFER  # noqa: E402
from obruxo_training.profiled_sampler import profile_metadata  # noqa: E402
from obruxo_training.reproducibility import fixed_generator_phases  # noqa: E402


def waveform(candidate, performance):
    offset = int(canonical_sha256(performance.to_dict())[:4], 16) % 50
    x = np.arange(8192, dtype=np.float64) / 44100
    values = candidate.continuous
    wave = (.05 + .1 * values[0]) * np.sin(2 * np.pi * (180 + offset + 20 * candidate.categorical[0]) * x)
    return np.stack((wave * (1.1 - values[1]), wave * (.1 + values[1])), -1).astype(np.float32)


class Frontend:
    def __init__(self, state=None):
        self.basic_pitch = torch.nn.Linear(1, 1)
        if state:
            self.basic_pitch.load_state_dict(state)
    def __call__(self, _audio):
        return {"timbre": torch.zeros(*TIMBRE_SHAPE), "pitch": torch.zeros(PITCH_SIZE)}
    def require_frozen(self):
        pass


class FakeSession(LifecycleReceipts, fit.PairedSession):
    def identity(self, candidate, performance, phase, label):
        self.check_limits()
        number = sum(row["event"] == "attempt" and row["phase"] == phase for row in self.events)
        assert number < self.plan["limits"][f"{phase}_attempts"]
        preset = self.codec.decode(candidate)
        request = RenderRequest(preset_json=fixed_generator_phases(preset).decode(), performance=performance,
                                renderer_id=self.codec.contract.renderer_id, tail_seconds=self.codec.contract.tail_seconds, preroll_frames=32768)
        identity = {"phase": phase, "label": label, "attempt": number, "request_id": request.request_id,
                    "source_sha256": sha256_bytes(preset), "controlled_sha256": sha256_bytes(request.preset_json.encode())}
        return identity
    def render(self, candidate, performance, phase, label):
        identity = self.identity(candidate, performance, phase, label)
        self._append({"event": "attempt", **identity})
        audio = waveform(candidate, performance)
        self.renderer_identity = self.plan["expected_renderer_identity"]
        receipt = {"event": "outcome", **identity, "status": "accepted", "renderer_identity": self.renderer_identity,
                   "qa": {"audio_float32_sha256": audio_float32_sha256(audio)}}
        self._append(receipt)
        return audio, receipt


@pytest.fixture(params=[False, True], ids=["transfer", "input-and-transfer"])
def completed(tmp_path, monkeypatch, request):
    torch.set_num_threads(2)
    torch.manual_seed(12)
    codec, model, frontend = PilotCodec(), PilotInverse(), Frontend()
    bootstrap = tmp_path / "bootstrap"
    (bootstrap / "fit-000").mkdir(parents=True)
    artifact = {"model": copy.deepcopy(model.state_dict()), "basic_pitch": copy.deepcopy(frontend.basic_pitch.state_dict())}
    initial = bootstrap / "fit-000/model.pt"
    torch.save(artifact, initial)
    record_features = frontend(None)
    retrieval = bootstrap / "fit-000/checkpoint.pt"
    torch.save({"replay": [{"features": record_features, "target": codec.default_target().to_dict()}]}, retrieval)
    performances = [Performance(bpm=value).to_dict() for value in (100, 120, 140)]
    recipe = {"state_id": "fixture", "target": codec.default_target().to_dict(), "input_views": "all", "performances": performances,
              "source_audio_sha256": [audio_float32_sha256(waveform(codec.default_target(), Performance.from_dict(value))) for value in performances],
              "source_feature_sha256": [tensor_digest(record_features)] * 3}
    development = {"state_id": "dev-fixture", "target": codec.default_target().to_dict(), "performances": performances[:2]}
    plan = {"format": "software_simulation_not_native_quality_evidence", "training": [recipe], "validation": [development],
            "training_view_roles": ["original_input", "wide_held_input", "wide_alternate_input"], "updates": reciprocal_updates(1),
            "production_training": {"forward": "unperturbed_argmax", "parameter_logits": "raw", "scales": [32., 29.]},
            "prefix_check_updates": 1, "optimizer": {"learning_rate": .0001, "seed": 0}, "limits": {"optimizer_attempts": 6, "training_attempts": 128,
                "qualification_attempts": 40, "wall_seconds": 600, "process_tree_bytes": 4 * 1024**3, "archive_bytes": 256 * 1024**2,
                "replay_bytes": 32 * 1024**2, "replay_records": 8}, "expected_renderer_identity": {"fixture": True},
            "initial_file_sha256": fit.digest(initial), "retrieval_file_sha256": fit.digest(retrieval), **profile_metadata(32768)}
    if request.param:
        plan.update(reconstruction_objective=INPUT_TRANSFER, rng_initialization="python_numpy_and_torch_from_optimizer_seed")
        plan["limits"]["training_attempts"] = 256
    pool = SimpleNamespace(codec=codec, pool_id="fixture")
    def restore(saved):
        restored, front = PilotInverse(), Frontend(saved["basic_pitch"])
        restored.load_state_dict(saved["model"])
        return restored, front, codec
    def export(network, front, _codec, _pool, **_kwargs):
        return {"model": network.state_dict(), "basic_pitch": front.basic_pitch.state_dict()}
    # Exercise the conditional development branch without inventing a passing sound result.
    def coverage_summary(cases, frozen, *, validation):
        result = actual_summary(cases, frozen, validation=validation)
        return result if validation else {**result, "pass": True}
    monkeypatch.setattr(fit, "restore_model", restore)
    monkeypatch.setattr(audit, "restore_model", restore)
    monkeypatch.setattr(fit, "inference_state", export)
    monkeypatch.setattr(fit, "PairedSession", FakeSession)
    monkeypatch.setattr(fit, "MemoryMonitor", lambda: SimpleNamespace(peak_bytes=1024**2, errors=0, start=lambda: None, close=lambda: None))
    monkeypatch.setattr(fit, "audit_summary", coverage_summary)
    monkeypatch.setattr(audit, "audit_summary", coverage_summary)
    monkeypatch.setattr(audit, "make_plan", lambda *_args, **_kwargs: copy.deepcopy(plan))
    output = tmp_path / "fit"
    fit.run_plan(pool, initial, retrieval, output, plan)
    result = fit.run_plan(pool, initial, retrieval, output, plan, execute=True)
    assert result["complete"] and not result["matched_comparison_passed"]
    return output, pool, bootstrap


def test_complete_simulated_runner_replays_both_models_and_rescores_all_audio(completed):
    output, pool, bootstrap = completed
    result = audit.verify(output, pool, bootstrap, None, None, None, None, None)
    assert result["offline_training_replay_exact"] and result["all_hard_requests_reconstructed"]
    assert result["comparison_waveforms_rescored"] == 19  # Nine fit predictions and ten development predictions.
    assert not result["matched_comparison_passed"]
    assert result["native_requests_added"] == result["optimizer_updates_added"] == 0
    assert (output / "progress.json").exists()
    prefix = json.loads((output / "prefix-verification.json").read_text())
    assert prefix["replayed_updates"] == 1 and prefix["native_requests_added"] == 0
    assert len(list((output / "comparison_audio").glob("*.pt"))) <= 19


@pytest.mark.parametrize("change", ("model", "optimizer", "loss", "waveform", "score", "export", "receipt", "prefix", "decision", "performance"))
def test_paired_auditor_rejects_changed_model_measurement_and_audio_evidence(completed, change):
    output, pool, bootstrap = completed
    if change == "performance":
        path = output / "measurements.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["view"] = (rows[0]["view"] + 1) % 3
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    if change in ("model", "optimizer"):
        path = output / "checkpoint.pt"
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if change == "model":
            next(iter(saved["models"]["spectral"].values())).view(-1)[0] += 1
        else:
            next(iter(saved["optimizers"]["spectral"]["state"].values()))["exp_avg"].view(-1)[0] += 1
        torch.save(saved, path)
    elif change == "loss":
        path = output / "measurements.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["loss"] += 1
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    elif change == "waveform":
        next((output / "comparison_audio").glob("*.pt")).write_bytes(b"changed")
    elif change == "score":
        cases = json.loads((output / "fit_cases.json").read_text())
        cases[0]["rows"]["spectral"]["scores"]["mrstft"] += 1
        saved = torch.load(output / "audits/fit-000-0.pt", map_location="cpu", weights_only=True)
        saved["rows"] = cases[0]["rows"]
        torch.save(saved, output / "audits/fit-000-0.pt")
        write_json(output / "fit_cases.json", cases)
    elif change == "export":
        path = output / "diagnostic-spectral.pt"
        saved = torch.load(path, map_location="cpu", weights_only=True)
        saved["production_training"]["parameter_logits"] = "scaled"
        torch.save(saved, path)
    elif change == "receipt":
        path = output / "renders.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[1]["controlled_sha256"] = "changed"
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    elif change == "prefix":
        path = output / "prefix-verification.json"
        saved = json.loads(path.read_text())
        saved["training_state_sha256"] = "changed"
        write_json(path, saved)
    elif change == "decision":
        path = output / "summary.json"
        saved = json.loads(path.read_text())
        saved["matched_comparison_passed"] = True
        write_json(path, saved)
    with pytest.raises((ValueError, RuntimeError), match="changed|reconstruct|differs|reproduce"):
        audit.verify(output, pool, bootstrap, None, None, None, None, None)


def test_native_timeout_can_only_resolve_to_a_later_identical_accepted_request():
    common = {"phase": "training", "label": "gradient:0:0:0", "request_id": "request", "source_sha256": "source", "controlled_sha256": "controlled"}
    error = {"event": "outcome", **common, "attempt": 0, "status": "error", "error_type": "RenderTimeout", "renderer_identity": None}
    accepted = {"event": "outcome", **common, "attempt": 1, "status": "accepted", "renderer_identity": {"fixture": True}, "qa": {}}
    events = [{"event": "attempt", **common, "attempt": 0}, error, {"event": "attempt", **common, "attempt": 1}, accepted]
    assert len(native_ledger(events, {"fixture": True})[2]) == 1
    events[-1] = {**accepted, "request_id": "different"}
    with pytest.raises(ValueError, match="identical|differs"):
        native_ledger(events, {"fixture": True})


def test_full_runner_recovers_a_counted_timeout_without_advancing_the_failed_pair(completed, monkeypatch):
    source, pool, bootstrap = completed
    plan = json.loads((source / "plan.json").read_text())
    output = source.parent / "recovered"
    initial, retrieval = bootstrap / "fit-000/model.pt", bootstrap / "fit-000/checkpoint.pt"
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
        fit.run_plan(pool, initial, retrieval, output, plan, execute=True)
    saved = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert saved["step"] == 1
    assert fit.run_plan(pool, initial, retrieval, output, plan, execute=True, resume=True)["complete"]
    verified = audit.verify(output, pool, bootstrap, None, None, None, None, None)
    assert verified["offline_training_replay_exact"] and len(verified["recovered_timeouts"]) == 1
    assert verified["interrupted_optimizer_attempts_replayed"] == [1]
    assert verified["interruption_checkpoints_replayed"] == [1]
    assert not verified["all_native_accepted"] and verified["native_receipts_resolved"]


def test_full_paired_plan_preserves_request_budget_and_adds_rescorable_audio(monkeypatch):
    from pathlib import Path
    frozen = json.loads((Path(__file__).parents[1] / "reports/data/balanced_native_fit_plan.json").read_text())
    monkeypatch.setattr(fit, "category_balanced_entries", lambda *_args: [])
    monkeypatch.setattr(fit, "expand_training", lambda *_args: ([SimpleNamespace(state_id=row["state_id"]) for row in frozen["training"]], None))
    monkeypatch.setattr(fit, "compose_plan", lambda *_args: copy.deepcopy(frozen))
    plan = fit.compose_paired_plan(SimpleNamespace(codec=PilotCodec()), {}, {"replay": []}, [32., 29.])
    assert len(plan["updates"]) == plan["optimizer"]["updates_per_arm"] == 128
    assert sum(len(row["examples"]) for row in plan["updates"]) == 256
    assert plan["optimizer"]["forward_microbatch_size"] == 1 and plan["optimizer"]["batch_size"] == 2
    assert sum(plan["limits"][key] for key in ("training_attempts", "qualification_attempts")) == 7120
    assert plan["planned_maximum_requests"] == frozen["planned_maximum_requests"]
    assert plan["training"] == frozen["training"] and plan["validation"] == frozen["validation"]
    assert plan["storage_bound"]["candidate_waveforms_maximum"] == 936
    assert plan["storage_bound"]["estimated_archive_upper_bound_bytes"] == 4088393207 < plan["limits"]["archive_bytes"]
    assert plan["production_training"]["parameter_logits"] == "raw" and "category_training" not in plan


@pytest.mark.parametrize("change", ("failed", "audio_not_rescored", "gradients_not_replayed"))
def test_plan_loader_rejects_unqualified_native_evidence_before_loading_training_data(monkeypatch, change):
    direction = {"qualified_production_direction": True, "candidate_audio_rescored": 72,
                 "all_model_gradients_and_local_steps_reconstructed": True, "all_requests_reconstructed": True}
    if change == "failed":
        direction["qualified_production_direction"] = False
    elif change == "audio_not_rescored":
        direction["candidate_audio_rescored"] = 71
    else:
        direction["all_model_gradients_and_local_steps_reconstructed"] = False
    monkeypatch.setattr(fit, "verify_screen", lambda *_args: direction)
    with pytest.raises(ValueError, match="qualification"):
        fit.make_plan(None, None, None, None, None, None, None)

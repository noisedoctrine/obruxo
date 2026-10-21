import copy
import json
from pathlib import Path
from time import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from obruxo_training.codec import PilotCodec
from obruxo_training.features import PITCH_SIZE, TIMBRE_SHAPE, tensor_digest
from obruxo_training.model import PilotInverse
from obruxo_training.native_fit import audit_case, audit_summary, fit_models, require_qualification
from obruxo_training.performances import initial_bank
from obruxo_training.spectral import SpectralTargets
from obruxo_data.render.qa import audio_float32_sha256


def test_qualification_is_checked_before_inverse_use():
    from obruxo_data.hashing import canonical_sha256
    path = Path(__file__).parents[1] / "reports/data/native_gradient_summary.json"
    report = json.loads(path.read_text())
    require_qualification(report)
    with pytest.raises(ValueError, match="render profile"):
        require_qualification(report, preroll_frames=32768)
    relabelled = copy.deepcopy(report)
    relabelled["plan"]["render_profile"] = {"preroll_frames": 32768}
    relabelled["renderer_identity"]["settings"]["preroll_frames"] = 32768
    relabelled["plan_id"] = canonical_sha256(relabelled["plan"])
    with pytest.raises(ValueError, match="renderer source"):
        require_qualification(relabelled, preroll_frames=32768)
    for key in ("qualified_native_gradient_path", "all_native_accepted", "within_resource_limits", "all_requests_reconstructed"):
        changed = copy.deepcopy(report)
        changed[key] = False
        with pytest.raises(ValueError, match="qualification"):
            require_qualification(changed)
    changed = copy.deepcopy(report)
    changed["cases"].pop()
    with pytest.raises(ValueError, match="qualification"):
        require_qualification(changed)


def test_fit_and_validation_gates_use_both_performances_and_complete_denominators():
    recipes = [{"state_id": str(index)} for index in range(2)]
    plan = {"training": recipes, "validation": recipes}
    cases = []
    for recipe in recipes:
        for view in (0, 1):
            rows = {name: {"scores": {"spectral": value, "mrstft": value, "tail_error": .01}}
                    for name, value in (("initial", 1.), ("parameter", 1.), ("spectral", .8), ("template", 2.), ("retrieval", 1.2))}
            cases.append({"state_id": recipe["state_id"], "view": view, "repeat_exact": True, "rows": rows})
    assert audit_summary(cases, plan, validation=False)["pass"]
    assert audit_summary(cases, plan, validation=True)["pass"]
    assert not audit_summary(cases[:-1], plan, validation=True)["pass"]
    assert not audit_summary(cases[:-1] + [cases[0]], plan, validation=False)["pass"]
    cases[-1]["rows"]["spectral"]["scores"]["tail_error"] = 1.
    assert not audit_summary(cases, plan, validation=True)["pass"]
    cases[-1]["repeat_exact"] = False
    assert not audit_summary(cases, plan, validation=False)["pass"]


class FixtureSession:
    def __init__(self, codec, plan):
        self.codec, self.plan, self.calls = codec, plan, []

    def render(self, candidate, performance, phase, label):
        wave = (np.sin(np.arange(8192) / 10) * .002 * (1 + sum(candidate.continuous) + .2 * sum(candidate.categorical))).astype(np.float32)
        audio = np.stack((wave, wave * .8), axis=1)
        self.calls.append(label)
        return audio, {"qa": {"audio_float32_sha256": audio_float32_sha256(audio)}}


def test_actual_matched_models_resume_exactly_and_optimizer_attempts_are_charged(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(4)
    codec, base = PilotCodec(), PilotInverse()
    features = {"timbre": torch.rand(*TIMBRE_SHAPE), "pitch": torch.rand(PITCH_SIZE)}
    recipe = {"target": codec.default_target().to_dict(), "performances": [initial_bank()[0].performance().to_dict()] * 2}
    plan = {"training": [recipe], "updates": [{"state": 0, "view": i % 2} for i in range(4)],
            "optimizer": {"learning_rate": .0001, "seed": 0}, "limits": {"optimizer_attempts": 6}}
    audio, _ = FixtureSession(codec, plan).render(codec.default_target(), None, "training", "target")
    class Records:
        def check_limits(self):
            pass

        def __getitem__(self, index):
            measured = SpectralTargets()(torch.from_numpy(audio).T[None])
            return {"features": features, "targets": [measured, measured]}

    def models():
        return {name: copy.deepcopy(base) for name in ("parameter", "spectral")}

    expected, interrupted = models(), models()
    expected_dir, resumed_dir = tmp_path / "expected", tmp_path / "resumed"
    expected_dir.mkdir()
    resumed_dir.mkdir()
    expected_trace = fit_models(expected, Records(), FixtureSession(codec, plan), expected_dir, plan)
    class PlannedStop(Exception):
        pass
    def stop(state):
        if state["step"] == 2:
            raise PlannedStop()
    with pytest.raises(PlannedStop):
        fit_models(interrupted, Records(), FixtureSession(codec, plan), resumed_dir, plan, after_step=stop)
    saved = torch.load(resumed_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
    resumed = models()
    assert fit_models(resumed, Records(), FixtureSession(codec, plan), resumed_dir, plan, saved) == expected_trace
    assert all(tensor_digest(resumed[name].state_dict()) == tensor_digest(expected[name].state_dict()) for name in expected)
    assert len((resumed_dir / "updates.jsonl").read_text().splitlines()) == 4
    saved["plan_id"] = "changed"
    with pytest.raises(ValueError, match="contract"):
        fit_models(models(), Records(), FixtureSession(codec, plan), resumed_dir, plan, saved)


def test_audit_reuses_committed_renders_and_refuses_changed_candidates(tmp_path):
    torch.set_num_threads(2)
    codec = PilotCodec()
    target = codec.default_target()
    recipe = {"state_id": "fixture", "target": target.to_dict(), "performances": [initial_bank()[0].performance().to_dict()] * 2}
    session = FixtureSession(codec, {"format": "audit_fixture"})
    def provider(_audio):
        return {"initial": target, "parameter": target, "spectral": target}
    first, _ = audit_case(session, tmp_path, "fixture", recipe, 0, provider, {"models": "fixed"}, repeat=True)
    count = len(session.calls)
    second, _ = audit_case(session, tmp_path, "fixture", recipe, 0, provider, {"models": "fixed"}, repeat=True)
    assert first == second and len(session.calls) == count == 5
    with pytest.raises(ValueError, match="predictions changed"):
        audit_case(session, tmp_path, "fixture", recipe, 0, lambda _audio: {"other": target}, {"models": "fixed"}, repeat=True)


def test_exception_summary_retains_completed_updates(monkeypatch, tmp_path):
    from obruxo_training import native_fit
    initial, retrieval = tmp_path / "initial.pt", tmp_path / "retrieval.pt"
    torch.save({}, initial)
    torch.save({}, retrieval)
    plan = {"initial_file_sha256": native_fit.digest(initial), "retrieval_file_sha256": native_fit.digest(retrieval),
            "limits": {"wall_seconds": 60, "process_tree_bytes": 1024, "archive_bytes": 16 * 1024**2}}
    output = tmp_path / "run"
    output.mkdir()
    (output / "plan.json").write_text(json.dumps(plan))
    class Monitor:
        peak_bytes, errors = 1, 0
        def start(self):
            pass
        def close(self):
            pass
    def failed_fit(*_args, after_step=None):
        after_step({"trace": [{"step": 1}]})
        after_step({"trace": [{"step": 1}, {"step": 2}]})
        raise RuntimeError("fixture render rejected")
    codec = PilotCodec()
    monkeypatch.setattr(native_fit, "make_plan", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(native_fit, "restore_model", lambda *_args: (torch.nn.Linear(1, 1), object(), codec))
    monkeypatch.setattr(native_fit, "load_training_checkpoint", lambda *_args: {"replay": []})
    monkeypatch.setattr(native_fit, "MemoryMonitor", Monitor)
    monkeypatch.setattr(native_fit, "NativeSession", lambda *_args: SimpleNamespace(started=time(), events=[], renderer_identity={}))
    monkeypatch.setattr(native_fit, "FitRecords", lambda *_args: object())
    monkeypatch.setattr(native_fit, "fit_models", failed_fit)
    with pytest.raises(RuntimeError, match="fixture render rejected"):
        native_fit.run_fit(SimpleNamespace(codec=codec), initial, retrieval, tmp_path / "qualification.json", output, execute=True)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["committed_updates_per_arm"] == 2 and not summary["complete"]
    assert summary["failure"] == "RuntimeError: fixture render rejected"


def test_cold_target_cache_is_rejected_even_when_recipe_and_file_hash_match(tmp_path):
    from obruxo_data.hashing import canonical_sha256
    from obruxo_training.native_fit import FitRecords, digest
    from obruxo_training.render_profile import basic_render_profile
    recipe = {"state_id": "fixture", "target": PilotCodec().default_target().to_dict()}
    directory = tmp_path / "records"
    directory.mkdir()
    path = directory / "000.pt"
    torch.save({"recipe_id": canonical_sha256(recipe)}, path)  # Legacy record has no warm-profile declaration.
    (tmp_path / "records.json").write_text(json.dumps({"0": {"file": path.name, "sha256": digest(path)}}))
    plan = {"training": [recipe], "render_profile": basic_render_profile(), "limits": {"archive_bytes": 2 * 1024**3}}
    records = FitRecords(tmp_path, plan, None, SimpleNamespace(check_limits=lambda: None))
    with pytest.raises(ValueError, match="different render profile"):
        records[0]

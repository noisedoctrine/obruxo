import copy
import json
from pathlib import Path

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
    path = Path(__file__).parents[1] / "reports/data/native_gradient_summary.json"
    report = json.loads(path.read_text())
    require_qualification(report)
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

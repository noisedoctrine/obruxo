import copy
from types import SimpleNamespace

import pytest
import torch

from obruxo_training import basic_direct_fit as fit
from obruxo_training.audit import write_json
from obruxo_training.basic_direct_fit_audit import verify
from obruxo_training.features import tensor_digest
from obruxo_training.native_acceptance import SPEC
from obruxo_training.native_acceptance_fit import compatible_prior, make_plan
from obruxo_training.paired_native_audit import equal_tree
from test_basic_direct_fit import FakeSession, completed, receipt  # noqa: F401


@pytest.fixture(scope="module")
def accepted(completed, tmp_path_factory):  # noqa: F811
    original, source = completed
    prior = torch.load(source / "checkpoint.pt", map_location="cpu", weights_only=True)
    prefix = torch.load(source / "prefix-checkpoint.pt", map_location="cpu", weights_only=True)
    inputs = copy.copy(original)
    inputs.plan = {**copy.deepcopy(original.plan), "native_acceptance": copy.deepcopy(SPEC),
                   "gates": {**original.plan["gates"], "minimum_native_accepted_updates": 3},
                   "expected_parameter_model_sha256": tensor_digest(prior["models"]["parameter"]),
                   "expected_parameter_prefix_model_sha256": tensor_digest(prefix["models"]["parameter"])}
    output = tmp_path_factory.mktemp("native-accepted-fit")
    write_json(output / "plan.json", inputs.plan)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(fit, "PairedSession", FakeSession)
        result = fit.execute(inputs, output)
    assert result["complete"] and result["committed_updates_per_arm"] == 4
    return inputs, output, source


def test_complete_measured_acceptance_fit_replays_exactly_and_keeps_matched_control(accepted):
    inputs, output, source = accepted
    proof = verify(inputs, output)
    assert proof["offline_training_replay_exact"] and proof["comparison_waveforms_rescored"] == 24
    saved = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
    prior = torch.load(source / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert equal_tree(prior["models"]["parameter"], saved["models"]["parameter"])
    assert equal_tree(prior["optimizers"]["parameter"], saved["optimizers"]["parameter"])
    assert all(int(state["step"]) == 4 for state in saved["optimizers"]["spectral"]["state"].values())
    assert any(row["native_acceptance"]["accepted"] for row in saved["trace"])
    for row in saved["trace"]:
        update = row["native_acceptance"]
        assert update["before_objective"] == row["losses"]["spectral"]
        assert update["committed_objective"] <= update["before_objective"]
        assert update["logical_trial_evaluations"] <= 12 and update["maximum_physical_trial_queries"] <= 18
        if update["accepted"]:
            assert update["before_objective"] - update["committed_objective"] > 1e-6
    compatibility = compatible_prior(inputs, source)
    assert compatibility["prior_model_adam_rng_replay_exact"] and compatibility["replayed_optimizer_applications"] == 8


def test_timeout_during_actual_trial_recovers_both_models_adam_and_rng_exactly(accepted, tmp_path, monkeypatch):
    inputs, uninterrupted, _source = accepted
    output = tmp_path / "trial-timeout"
    output.mkdir()
    write_json(output / "plan.json", inputs.plan)
    original_render, original_accept = FakeSession.render, fit.accept_update
    in_trial, failed = False, False

    def accept(*args):
        nonlocal in_trial
        in_trial = True
        try:
            return original_accept(*args)
        finally:
            in_trial = False

    def render(session, candidate, performance, phase, label):
        nonlocal failed
        if in_trial and not failed and label.startswith("probe:002:"):
            failed = True
            attempt = sum(row["event"] == "attempt" and row["phase"] == phase for row in session.events)
            _audio, normal = receipt(session.codec, session.plan, candidate, performance, phase, label, attempt)
            identity = {key: normal[key] for key in ("phase", "label", "attempt", "request_id", "source_sha256", "controlled_sha256")}
            session._append({"event": "attempt", **identity})
            session._append({"event": "outcome", **identity, "status": "error", "error_type": "RenderTimeout"})
            raise RuntimeError("simulated trial RenderTimeout")
        return original_render(session, candidate, performance, phase, label)

    monkeypatch.setattr(fit, "accept_update", accept)
    monkeypatch.setattr(FakeSession, "render", render)
    monkeypatch.setattr(fit, "PairedSession", FakeSession)
    with pytest.raises(RuntimeError, match="RenderTimeout"):
        fit.execute(inputs, output)
    assert torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)["step"] == 1
    fit.execute(inputs, output, resume=True)
    proof = verify(inputs, output)
    assert len(proof["recovered_timeouts"]) == 1 and proof["offline_training_replay_exact"]
    expected = torch.load(uninterrupted / "checkpoint.pt", map_location="cpu", weights_only=True)
    recovered = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert all(equal_tree(expected[key], recovered[key]) for key in ("models", "optimizers", "rng"))
    assert len(fit.lines(output / "updates.jsonl")) == 5


def test_changed_acceptance_contract_is_refused_before_loading_observations():
    inputs = SimpleNamespace(plan={"native_acceptance": {**SPEC, "maximum_native_trials": 5}})
    with pytest.raises(ValueError, match="fixed sound-only"):
        fit.step({}, {}, inputs, None, 0)


def test_unverified_native_fit_source_is_refused_before_training_inputs(tmp_path):
    prior, checked = tmp_path / "prior", tmp_path / "native"
    prior.mkdir()
    checked.mkdir()
    for path in (prior / "plan.json", checked / "plan.json", checked / "verification.json"):
        write_json(path, {})
    write_json(prior / "verification.json", {"verified": False})
    with pytest.raises(ValueError, match="audited prior learning"):
        make_plan(None, None, None, None, None, prior, checked)

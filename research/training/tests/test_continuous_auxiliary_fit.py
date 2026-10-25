import copy

import pytest
import torch

from obruxo_training import basic_direct_fit as fit
from obruxo_training.audit import write_json
from obruxo_training.basic_direct_fit_audit import decisions, verify
from obruxo_training.basic_direct_fit_step import matched_step
from obruxo_training.continuous_auxiliary import SPEC
from obruxo_training.continuous_auxiliary_fit import make_plan
from obruxo_training.features import tensor_digest
from obruxo_training.paired_native_audit import equal_tree
from test_basic_direct_fit import FakeSession, completed  # noqa: F401


def test_full_auxiliary_fit_replays_and_keeps_parameter_model_and_adam_identical(completed, tmp_path, monkeypatch):  # noqa: F811
    original, source = completed
    prior = torch.load(source / "checkpoint.pt", map_location="cpu", weights_only=True)
    prefix = torch.load(source / "prefix-checkpoint.pt", map_location="cpu", weights_only=True)
    inputs = copy.copy(original)
    inputs.plan = {**copy.deepcopy(original.plan), "continuous_auxiliary": copy.deepcopy(SPEC),
                   "expected_parameter_model_sha256": tensor_digest(prior["models"]["parameter"]),
                   "expected_parameter_prefix_model_sha256": tensor_digest(prefix["models"]["parameter"]),
                   "earlier_sound_reference": {"means": {"spectral": 100., "mrstft": 100.}, "mrstft_p90": 100.}}
    output = tmp_path / "auxiliary"
    output.mkdir()
    write_json(output / "plan.json", inputs.plan)
    monkeypatch.setattr(fit, "PairedSession", FakeSession)
    result = fit.execute(inputs, output)
    assert result["complete"] and result["result"]["earlier_sound_preservation"]["pass"]
    audit = verify(inputs, output)
    assert audit["offline_training_replay_exact"] and audit["comparison_waveforms_rescored"] == 24
    saved = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert equal_tree(prior["models"]["parameter"], saved["models"]["parameter"])
    assert equal_tree(prior["optimizers"]["parameter"], saved["optimizers"]["parameter"])
    assert not equal_tree(prior["models"]["spectral"], saved["models"]["spectral"])
    for row in saved["trace"]:
        assert row["continuous_auxiliary_weight"] == 1 and row["continuous_auxiliary_loss"] >= 0
        assert row["losses"]["spectral"] == pytest.approx(row["native_sound_loss"] + row["continuous_auxiliary_loss"])


def test_previous_sound_gains_cannot_be_lost_even_if_original_preservation_would_pass(completed):  # noqa: F811
    inputs, source = completed
    saved = torch.load(source / "checkpoint.pt", map_location="cpu", weights_only=True)
    store = fit.SoundStore(source, inputs)
    old = decisions(inputs, store, saved["trace"])
    proposed = copy.copy(inputs)
    proposed.plan = {**copy.deepcopy(inputs.plan), "earlier_sound_reference": {
        "means": {metric: .99 * old["original_regression"]["means"]["spectral"][metric] for metric in ("spectral", "mrstft")},
        "mrstft_p90": .99 * old["original_regression"]["mrstft_p90"]["spectral"]}}
    result = decisions(proposed, store, saved["trace"])
    assert not result["earlier_sound_preservation"]["pass"] and not result["pass"]


def test_changed_matched_control_stops_before_final_rendering(completed, tmp_path, monkeypatch):  # noqa: F811
    original, source = completed
    inputs = copy.copy(original)
    inputs.plan = {**copy.deepcopy(original.plan), "expected_parameter_model_sha256": "0" * 64}
    models, _optimizers = fit.initialize(inputs)
    saved = torch.load(source / "checkpoint.pt", map_location="cpu", weights_only=True)
    for arm, model in models.items():
        model.load_state_dict(saved["models"][arm])
    monkeypatch.setattr(fit, "train", lambda *_args: (models, saved["trace"]))
    monkeypatch.setattr(fit, "collect_finals", lambda *_args: pytest.fail("mismatched parameter control must not render final comparisons"))
    monkeypatch.setattr(fit, "PairedSession", FakeSession)
    output = tmp_path / "wrong-control"
    output.mkdir()
    write_json(output / "plan.json", inputs.plan)
    with pytest.raises(ValueError, match="matched parameter-control model"):
        fit.execute(inputs, output)


def test_auxiliary_weight_cannot_change_silently():
    altered = {**SPEC, "weight": 2.}
    with pytest.raises(ValueError, match="fixed paired contract"):
        matched_step({}, {}, None, [], None, 0, 48, continuous_auxiliary=altered)


def test_unverified_prior_fit_is_refused_before_loading_training_inputs(tmp_path):
    prior, diagnosis = tmp_path / "prior", tmp_path / "diagnosis"
    prior.mkdir()
    diagnosis.mkdir()
    write_json(prior / "verification.json", {"verified": False})
    write_json(prior / "plan.json", {})
    write_json(diagnosis / "summary.json", {})
    write_json(diagnosis / "plan.json", {})
    with pytest.raises(ValueError, match="audited direct fit"):
        make_plan(None, None, None, None, None, prior, diagnosis)

import copy
import json

import pytest
import torch

from obruxo_training import paired_native_step as pair
from obruxo_training.paired_learning_diagnosis import feature_sensitivity, replay_alignment, source_hashes, summarize
from test_paired_native_training import inputs as paired_inputs
from obruxo_training.paired_native_training import fit_pairs

inputs = paired_inputs


@pytest.mark.parametrize("inputs", [True], indirect=True)
def test_gradient_observation_preserves_every_trace_and_both_models_adam_and_rng(tmp_path, inputs):
    plan, records, session, fresh, _calls = inputs
    plan["training"][0]["state_id"] = "training-only"
    baseline = fresh()
    initial = copy.deepcopy(baseline["spectral"])
    fit_pairs(baseline, records, session, tmp_path, plan)
    saved = torch.load(tmp_path / "checkpoint.pt", map_location="cpu", weights_only=True)
    measured = [json.loads(line) for line in (tmp_path / "measurements.jsonl").read_text().splitlines()]
    models, gradients = replay_alignment(initial, session.codec, plan, records, measured, saved)
    assert len(gradients) == len(plan["updates"])
    assert all(-1 <= row["all"]["cosine"] <= 1 for row in gradients)
    assert set(gradients[0]) == {"timbre", "pitch", "fusion", "continuous", "categorical", "all"}
    features = feature_sensitivity({"initial": initial, **models}, session.codec, plan, {0: records[0]})
    report = summarize(gradients, features)
    assert len(features) == 3 * (1 + 2 * 3)
    assert all(row["state_id"] == "training-only" for row in features)
    assert len(report["feature_groups"]) == len(features)


@pytest.mark.parametrize("inputs", [True], indirect=True)
@pytest.mark.parametrize("change", ["performance", "missing_component", "final_model"])
def test_diagnosis_rejects_changed_components_or_final_state(tmp_path, inputs, change):
    plan, records, session, fresh, _calls = inputs
    initial = fresh()["spectral"]
    fit_pairs(fresh(), records, session, tmp_path, plan)
    saved = torch.load(tmp_path / "checkpoint.pt", map_location="cpu", weights_only=True)
    measured = [json.loads(line) for line in (tmp_path / "measurements.jsonl").read_text().splitlines()]
    if change == "performance":
        measured[0]["view"] = (measured[0]["view"] + 1) % 3
    elif change == "missing_component":
        measured.pop()
    else:
        next(iter(saved["models"]["spectral"].values())).view(-1)[0] += 1
    original = pair.native_reconstruction_loss
    with pytest.raises(ValueError, match="differs|incomplete|reproduce"):
        replay_alignment(initial, session.codec, plan, records, measured, saved)
    assert pair.native_reconstruction_loss is original


def test_unverified_parent_is_refused_before_reading_model_or_native_data(tmp_path):
    (tmp_path / "verification.json").write_text(json.dumps({"verified": False}))
    with pytest.raises(ValueError, match="independently verified"):
        source_hashes(tmp_path, tmp_path / "missing-model.pt")

import copy

import pytest
import torch

from obruxo_training.audit import write_json
from obruxo_training.basic_direct_fit import initialize
from obruxo_training.conditioned_bridge import read
from obruxo_training.features import tensor_digest
from obruxo_training.trainable_basic_fit import check_input_learning, optimizer_digest, parameter_rehearsal, source_code
from obruxo_training.trainer import capture_rng
from obruxo_training.paired_native_audit import equal_tree
from test_basic_direct_fit import completed  # noqa: F401
from test_trainable_basic import trained  # noqa: F401


def test_control_rehearsal_exactly_matches_the_full_sound_trial_without_changing_initializer(trained):  # noqa: F811
    inputs, output, _source = trained
    before, rng = inputs.initial_path.read_bytes(), capture_rng("cpu")
    expected = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
    prefix = torch.load(output / "prefix-checkpoint.pt", map_location="cpu", weights_only=True)
    result = parameter_rehearsal(inputs)
    assert result["model_sha256"] == tensor_digest(expected["models"]["parameter"])
    assert result["optimizer_sha256"] == optimizer_digest(expected["optimizers"]["parameter"])
    assert result["prefix"]["model_sha256"] == tensor_digest(prefix["models"]["parameter"])
    assert result["prefix"]["optimizer_sha256"] == optimizer_digest(prefix["optimizers"]["parameter"])
    assert result["losses"] == [row["losses"]["parameter"] for row in expected["trace"]]
    assert result["software_parameter_control_optimizer_applications"] == 4
    assert result["experimental_candidate_updates"] == result["native_requests"] == 0
    assert before == inputs.initial_path.read_bytes() and equal_tree(rng, capture_rng("cpu"))


def test_input_learning_check_rejects_unchanged_branches_and_wrong_control_adam(trained):  # noqa: F811
    inputs, output, _source = trained
    models, optimizers = initialize(inputs)
    with pytest.raises(ValueError, match="both copied input branches"):
        check_input_learning(inputs, models, optimizers)
    saved = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
    for arm, model in models.items():
        model.load_state_dict(saved["models"][arm])
        optimizers[arm].load_state_dict(saved["optimizers"][arm])
    checked = copy.copy(inputs)
    checked.plan = {**inputs.plan, "expected_parameter_optimizer_sha256": "wrong"}
    with pytest.raises(ValueError, match="Adam state"):
        check_input_learning(checked, models, optimizers)
    prefix = read(output / "prefix-verification.json")
    assert all(value > 0 for groups in prefix["input_branch_updates"].values() for value in groups.values())


def test_training_code_change_outside_declared_adapters_is_refused():
    with pytest.raises(ValueError, match="numerical prediction, loss or renderer"):
        source_code({"source": {"native_inverse.py": "wrong"}})


def test_unaudited_bridge_is_refused_before_loading_training_inputs(tmp_path):
    from obruxo_training import trainable_basic_fit as module
    bridge, prior = tmp_path / "bridge", tmp_path / "prior"
    bridge.mkdir()
    prior.mkdir()
    for path in (bridge / "plan.json", prior / "plan.json", prior / "verification.json"):
        write_json(path, {})
    write_json(bridge / "summary.json", {"initialization_gradient_reference_recovery_passed": False})
    with pytest.raises(ValueError, match="qualified untrained bridge"):
        module.inputs_bound(None, bridge, None, None, prior)

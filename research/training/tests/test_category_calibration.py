import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from obruxo_data.hashing import canonical_sha256, sha256_bytes
from obruxo_training.category_calibration import file_digest, load_calibration
from obruxo_training.codec import PilotCodec
from obruxo_training.features import PITCH_SIZE, TIMBRE_SHAPE
from obruxo_training.model import loss_terms
from obruxo_training.native_fit import matched_step


def calibration_fixture(tmp_path):
    source = Path(__file__).parents[1] / "reports/data/category_signal_summary.json"
    report = json.loads(source.read_text())
    initial, qualification = tmp_path / "initial.pt", tmp_path / "qualification.json"
    initial.write_bytes(b"fixture initial weights")
    qualification.write_bytes(b"fixture qualification")
    pool = SimpleNamespace(pool_id="fixture-pool", to_dict=lambda: {"fixture": True})
    encoded = (json.dumps({"pool_id": pool.pool_id, "pool": pool.to_dict()}, ensure_ascii=True,
                          allow_nan=False, sort_keys=True, indent=2) + "\n").encode()
    report["plan"].update(pool_sha256=sha256_bytes(encoded), initial_sha256=file_digest(initial), qualification_sha256=file_digest(qualification))
    report["plan_id"] = canonical_sha256(report["plan"])
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(report))
    return path, report, pool, initial, qualification


def test_calibration_is_bound_to_actual_inputs_and_reconstructs_fixed_scales(tmp_path):
    path, report, pool, initial, qualification = calibration_fixture(tmp_path)
    result = load_calibration(path, pool, initial, qualification)
    assert result["scales"] == report["fixed_scales"]
    assert result["loss_arms"] == ["parameter", "spectral"]
    assert result["inference"] == "raw_argmax_unchanged" and not result["initial_weights_changed"]
    assert result["diagnosis_file_sha256"] == file_digest(path)
    with pytest.raises(ValueError, match="different pool"):
        load_calibration(path, SimpleNamespace(pool_id="other", to_dict=pool.to_dict), initial, qualification)
    initial.write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="different pool"):
        load_calibration(path, pool, initial, qualification)


def test_unqualified_or_modified_calibration_cannot_enter_training(tmp_path):
    path, report, pool, initial, qualification = calibration_fixture(tmp_path)
    for field in ("passed", "historical_replay_unchanged", "all_saved_derivatives_reconstructed", "actual_training_points_verified"):
        changed = copy.deepcopy(report)
        changed[field] = False
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="preflight"):
            load_calibration(path, pool, initial, qualification)
    changed = copy.deepcopy(report)
    changed["fixed_scales"][0] += 1
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="scales differ"):
        load_calibration(path, pool, initial, qualification)


class TinyHeads(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.controls = torch.nn.Parameter(torch.zeros(1, 9))
        self.voices = torch.nn.Parameter(torch.tensor([[0., 20., 40.]]))
        self.filter = torch.nn.Parameter(torch.tensor([[0., 40.]]))

    def forward(self, _features):
        return {"continuous": self.controls.sigmoid(), "categorical": [self.voices * 1., self.filter * 1.]}


def test_both_losses_use_calibrated_scores_and_report_gradients_at_raw_model_outputs():
    torch.manual_seed(1)
    codec, base = PilotCodec(), TinyHeads()
    models = {name: copy.deepcopy(base) for name in ("parameter", "spectral")}
    optimizers = {name: torch.optim.Adam(model.parameters(), lr=.0001) for name, model in models.items()}
    features = {"timbre": torch.zeros(*TIMBRE_SHAPE), "pitch": torch.zeros(PITCH_SIZE)}
    before = base(None)
    expected = loss_terms({**before, "categorical": [scores / 20 for scores in before["categorical"]]}, [codec.default_target()], codec)["loss"]
    def evaluate(_index, candidate):
        return sum(value * value for value in candidate.continuous) + .2 * sum(candidate.categorical)
    result = matched_step(models, optimizers, features, codec.default_target(), codec, evaluate, 0, 4,
                          category_training={"scales": [20., 20.]})
    assert result["losses"]["parameter"] == pytest.approx(float(expected.detach()))
    assert all(value > 0 for value in result["gradients"]["category_l1"])
    assert result["gradients"]["category_scaled_l1"] == pytest.approx([20 * value for value in result["gradients"]["category_l1"]])
    assert result["gradient_queries"] <= 22
    assert sum(scores.shape[-1] for scores in models["spectral"](None)["categorical"]) + 9 == 14

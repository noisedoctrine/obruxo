from types import SimpleNamespace

import pytest
import torch
from obruxo_training.codec import PilotCodec
from obruxo_training.features import PITCH_SIZE, TIMBRE_SHAPE
from obruxo_training.model import PilotInverse, batch_features
from obruxo_training.paired_native_audit import equal_tree
from obruxo_training.production_controls import ProductionControlAdapter
from obruxo_training.shared_auxiliary_run import train
from obruxo_training.shared_basic_auxiliary import ARMS, calibrate, initialize
from obruxo_training.trainable_basic import COUNTS, TrainableBasic


@pytest.fixture
def synthetic():
    torch.set_num_threads(2)
    torch.manual_seed(31)
    rows = []
    for state in range(32):
        for view in range(3):
            rows.append({"state_id": str(state), "performance": [36. + state + view, 100., .3 + .3 * view],
                         "features": {"timbre": torch.rand(TIMBRE_SHAPE) + view * .1, "pitch": torch.rand(PITCH_SIZE)}})
    features = {index: row["features"] for index, row in enumerate(rows[:4])}
    codec = PilotCodec()
    inputs = SimpleNamespace(model=TrainableBasic(PilotInverse()), codec=codec,
        adapter=ProductionControlAdapter(codec, scales=[1., 1.]), features=features,
        plan={"updates": [{"ordinal": i} for i in range(3)], "fit_audit_cells": [{"key": i} for i in range(4)]},
        cases=[{"rows": {"initial": {"scores": {"spectral": .5}}}} for _ in range(4)])
    inputs.key = lambda cell: cell["key"]
    inputs.predict = lambda model, cell: model(batch_features([{"features": features[cell["key"]]}]))
    inputs.observations = lambda _update: [[0, 1], [1, 0], [2], [3]]

    class Store:
        def probe(self, _step, _observation, index, patch):
            # Analytic test function; these are never represented as native sound measurements.
            value = torch.tensor(patch.continuous, dtype=torch.float32)
            return .1 + float((value - (.2 + .03 * index)).square().mean()) + .02 * sum(patch.categorical)

    return inputs, rows, Store()


def test_matched_models_start_equal_and_first_update_is_sound_only(synthetic):
    inputs, rows, store = synthetic
    models, optimizers = initialize(inputs)
    assert equal_tree(models[ARMS[0]].state_dict(), models[ARMS[1]].state_dict())
    assert equal_tree(optimizers[ARMS[0]].state_dict(), optimizers[ARMS[1]].state_dict())
    models, saved = train(inputs, rows, store, stop_after=1)
    assert equal_tree(saved["models"][ARMS[0]], saved["models"][ARMS[1]])
    assert equal_tree(saved["optimizers"][ARMS[0]], saved["optimizers"][ARMS[1]])
    assert saved["calibration"] is None
    assert all(model.require_frozen() == COUNTS for model in models.values())


def test_actual_training_step_recovers_both_models_adam_calibration_trace_and_rng(synthetic):
    inputs, rows, store = synthetic
    _models, expected = train(inputs, rows, store, stop_after=3)
    _models, first = train(inputs, rows, store, stop_after=1)
    _models, recovered = train(inputs, rows, store, saved=first, stop_after=3)
    assert equal_tree(expected, recovered)
    assert expected["calibration"]["sound_encoder_norm"] > 0
    assert expected["calibration"]["unscaled_note_encoder_norm"] > 0
    assert expected["trace"][1]["arms"]["sound_auxiliary"]["coefficient"] > 0
    assert not equal_tree(expected["models"][ARMS[0]], expected["models"][ARMS[1]])


def test_calibration_rejects_a_disconnected_training_signal():
    encoder = torch.nn.Linear(3, 2)
    value = encoder(torch.ones(1, 3))
    with pytest.raises(ValueError, match="nonzero finite"):
        calibrate(value.sum(), value.sum() * 0., encoder)

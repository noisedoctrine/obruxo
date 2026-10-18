from dataclasses import replace
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from obruxo_training.codec import PilotCodec  # noqa: E402
from obruxo_training.evaluate import load_evaluation_checkpoint  # noqa: E402
from obruxo_training.features import PITCH_SIZE, TIMBRE_SHAPE  # noqa: E402
from obruxo_training.full_scalar import FullScalarCodec  # noqa: E402
from obruxo_training.model import PilotInverse  # noqa: E402
from obruxo_training.pilot import PilotPool  # noqa: E402
from obruxo_training.trainer import TrainConfig, load_model, train_pilot  # noqa: E402
from test_stream import FakeRendererFactory  # noqa: E402


def test_basic_has_only_fourteen_outputs_and_rejects_broad_pool_before_work(tmp_path, monkeypatch):
    model = PilotInverse(PilotCodec())
    predicted = model({"timbre": torch.zeros(1, *TIMBRE_SHAPE), "pitch": torch.zeros(1, PITCH_SIZE)})
    assert predicted["continuous"].shape == (1, 9)
    assert [item.shape for item in predicted["categorical"]] == [(1, 3), (1, 2)]
    assert sum(parameter.numel() for parameter in model.parameters()) == 355806
    monkeypatch.setattr("obruxo_training.trainer.features_for", lambda *args: pytest.fail("frontend must not load"))
    with pytest.raises(ValueError, match="basic prediction scope"):
        train_pilot(SimpleNamespace(codec=FullScalarCodec()), tmp_path / "rejected", config=TrainConfig(prediction_scope="basic"))
    assert not (tmp_path / "rejected").exists()
    with pytest.raises(ValueError, match="prediction scope"):
        TrainConfig(prediction_scope="unknown")


def test_basic_scope_keeps_legacy_features_and_export_but_pins_recovery(tmp_path):
    pool = PilotPool.create(12, seed=3)
    legacy = TrainConfig(steps=2, replay_capacity=2, batch_size=2, checkpoint_every=1, render_budget=4, cpu_threads=2)
    train_pilot(pool, tmp_path / "source", config=legacy, renderer_factory=FakeRendererFactory(), stop_after=0)
    source = tmp_path / "source/checkpoint.pt"
    assert "prediction_scope" not in torch.load(source, weights_only=True)["run_contract"]["training"]
    config = replace(legacy, prediction_scope="basic")

    def run(name, **kwargs):
        return train_pilot(pool, tmp_path / name, config=config, renderer_factory=FakeRendererFactory("crash"), **kwargs)

    complete = run("full", seed_replay=source)
    run("partial", seed_replay=source, stop_after=1)
    resumed = run("resumed", resume=tmp_path / "partial/checkpoint.pt")
    assert complete["model_state_sha256"] == resumed["model_state_sha256"]
    assert complete["final_metrics"] == resumed["final_metrics"]
    assert complete["cursor"] == resumed["cursor"]
    assert complete["feature_seconds_this_segment"] == resumed["feature_seconds_this_segment"] == 0
    saved, _, _, _ = load_evaluation_checkpoint(tmp_path / "full/checkpoint.pt", pool)
    assert saved["run_contract"]["training"]["prediction_scope"] == "basic"
    model, _, codec = load_model(tmp_path / "full/model.pt")
    assert len(codec.continuous) == model.continuous.out_features == 9
    assert [head.out_features for head in model.categorical] == [3, 2]
    with pytest.raises(ValueError, match="contract changed"):
        train_pilot(pool, tmp_path / "changed", config=legacy, resume=tmp_path / "partial/checkpoint.pt")

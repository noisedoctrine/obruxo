import torch
import json

from obruxo_training.full_scalar import FullScalarCodec
from obruxo_training.pilot import PilotPool
from obruxo_training.stream import StreamConfig
from obruxo_training.trainer import TrainConfig, train_pilot
from test_stream import FakeRendererFactory


def test_full_scalar_optimizer_and_rng_resume_exactly_with_frozen_variant_splits(tmp_path):
    pool = PilotPool.create(12, seed=2, codec=FullScalarCodec())
    config = TrainConfig(steps=4, replay_capacity=4, batch_size=2, render_budget=4, cpu_threads=1,
                         checkpoint_every=1, sampling_policy="state_round_robin")
    stream = StreamConfig(prefetch_clips=1, worker_jobs=2)

    def train(name, **kwargs):
        return train_pilot(pool, tmp_path / name, config=config, stream_config=stream, renderer_factory=FakeRendererFactory(), **kwargs)

    complete = train("complete")
    train("interrupted", stop_after=2)
    resumed = train("resumed", resume=tmp_path / "interrupted/checkpoint.pt")
    assert complete["model_state_sha256"] == resumed["model_state_sha256"] and complete["cursor"] == resumed["cursor"]
    left = torch.load(tmp_path / "complete/checkpoint.pt", weights_only=True)
    right = torch.load(tmp_path / "resumed/checkpoint.pt", weights_only=True)
    assert left["run_contract"] == right["run_contract"]
    assert torch.equal(left["batch_rng"], right["batch_rng"]) and torch.equal(left["rng"]["torch"], right["rng"]["torch"])
    for index, state in left["optimizer"]["state"].items():
        assert all(torch.equal(value, right["optimizer"]["state"][index][name]) for name, value in state.items())


def test_full_scalar_cli_selects_pool_renderer_and_writes_an_audio_only_bundle(tmp_path, monkeypatch):
    from obruxo_data.render.base import FULL_PRESET_RENDERER_ID
    from obruxo_training.cli import main
    seen = []

    def fake_factory(_path=None, *, renderer_id):
        seen.append(renderer_id)
        return FakeRendererFactory()

    monkeypatch.setattr("obruxo_training.stream.NativeRendererFactory", fake_factory)
    pool_dir, output = tmp_path / "pool", tmp_path / "training"
    assert main(["full-scalar-pool", "--count", "12", "--output", str(pool_dir)]) == 0
    # This deliberately short run cannot pass the strict tiny-overfit gate.
    assert main(["train", "--pool", str(pool_dir / "pool.json"), "--output", str(output), "--steps", "1",
                 "--replay-capacity", "2", "--batch-size", "1", "--render-budget", "2", "--cpu-threads", "1"]) == 2
    assert seen and set(seen) == {FULL_PRESET_RENDERER_ID}
    assert json.loads((output / "summary.json").read_text())["complete"]
    saved = torch.load(output / "model.pt", weights_only=True)
    assert saved["input"] == "audio_only" and saved["contract"]["renderer_id"] == FULL_PRESET_RENDERER_ID

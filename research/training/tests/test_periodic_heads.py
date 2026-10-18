from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from obruxo_training.evaluate import load_evaluation_checkpoint  # noqa: E402
from obruxo_training.features import PITCH_SIZE, TIMBRE_SHAPE  # noqa: E402
from obruxo_training.full_scalar import FullScalarCodec  # noqa: E402
from obruxo_training.model import PilotInverse, cyclic_head_from_spec, model_spec  # noqa: E402
from obruxo_training.pilot import PilotPool  # noqa: E402
from obruxo_training.trainer import TrainConfig, load_model, train_pilot  # noqa: E402
from test_stream import FakeRendererFactory  # noqa: E402


def features():
    return {"timbre": torch.zeros((1, *TIMBRE_SHAPE)), "pitch": torch.zeros((1, PITCH_SIZE))}


def test_periodic_gradient_crosses_zero_instead_of_saturating():
    codec = FullScalarCodec()
    index = codec.cyclic_indices[0]
    results = {}
    for policy in ("sigmoid", "periodic"):
        model = PilotInverse(codec, cyclic_head=policy)
        with torch.no_grad():
            model.continuous.bias[index] = 0. if policy == "periodic" else torch.tensor(.0001).logit()
        optimizer = torch.optim.SGD([model.continuous.bias], lr=.5)
        phase = model(features())["continuous"][0, index]
        error = (phase - .8 + .5).remainder(1.) - .5
        error.square().backward()
        optimizer.step()
        phase = model(features())["continuous"][0, index]
        results[policy] = float(((phase - .8 + .5).remainder(1.) - .5).square().detach())
    assert results["periodic"] < 1e-10
    assert results["sigmoid"] > .039


def test_periodic_outputs_preserve_other_heads_and_wrap_any_finite_phase():
    codec = FullScalarCodec()
    torch.manual_seed(4)
    legacy = PilotInverse(codec)
    torch.manual_seed(4)
    periodic = PilotInverse(codec, cyclic_head="periodic")
    other = [index for index in range(len(codec.continuous)) if index not in codec.cyclic_indices]
    left, right = legacy(features()), periodic(features())
    assert torch.equal(left["continuous"][:, other], right["continuous"][:, other])
    assert all(torch.equal(a, b) for a, b in zip(left["categorical"], right["categorical"], strict=True))
    for value in (-3.2, -.01, 0., .99, 1.01, 9.3):
        with torch.no_grad():
            periodic.continuous.bias[list(codec.cyclic_indices)] = value
        phases = periodic(features())["continuous"][:, codec.cyclic_indices]
        assert torch.all((phases >= 0) & (phases < 1))
        torch.testing.assert_close(phases, torch.full_like(phases, value % 1), atol=1e-6, rtol=0)
    assert cyclic_head_from_spec(model_spec(codec)) == "sigmoid"
    assert cyclic_head_from_spec(model_spec(codec, cyclic_head="periodic")) == "periodic"
    with pytest.raises(ValueError):
        TrainConfig(cyclic_head="unknown")
    with pytest.raises(ValueError):
        cyclic_head_from_spec({"cyclic_outputs": "unknown"})


def test_periodic_training_reuses_legacy_features_exports_and_recovers_exactly(tmp_path):
    pool = PilotPool.create(12, seed=3, codec=FullScalarCodec())
    legacy = TrainConfig(steps=2, replay_capacity=2, batch_size=2, checkpoint_every=1, render_budget=4, cpu_threads=2)
    train_pilot(pool, tmp_path / "source", config=legacy, renderer_factory=FakeRendererFactory(), stop_after=0)
    source = tmp_path / "source/checkpoint.pt"
    assert "cyclic_head" not in torch.load(source, weights_only=True)["run_contract"]["training"]
    config = replace(legacy, cyclic_head="periodic")

    def run(name, **kwargs):
        return train_pilot(pool, tmp_path / name, config=config, renderer_factory=FakeRendererFactory("crash"), **kwargs)

    complete = run("full", seed_replay=source)
    run("partial", seed_replay=source, stop_after=1)
    resumed = run("resumed", resume=tmp_path / "partial/checkpoint.pt")
    assert complete["model_state_sha256"] == resumed["model_state_sha256"]
    assert complete["final_metrics"] == resumed["final_metrics"]
    assert complete["totals"]["render_attempts"] == 0
    model, _, _ = load_model(tmp_path / "full/model.pt")
    assert model.cyclic_head == "periodic"
    _, evaluated, _, _ = load_evaluation_checkpoint(tmp_path / "full/checkpoint.pt", pool)
    assert evaluated.cyclic_head == "periodic"
    assert torch.equal(model(features())["continuous"], evaluated(features())["continuous"])
    with pytest.raises(ValueError, match="contract changed"):
        train_pilot(pool, tmp_path / "wrong-head", config=legacy, resume=tmp_path / "partial/checkpoint.pt")

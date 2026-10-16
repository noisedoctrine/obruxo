from dataclasses import replace
import json

import pytest

torch = pytest.importorskip("torch")

from obruxo_training.codec import PilotCodec  # noqa: E402
from obruxo_training.evaluate import evaluate_comparison, load_evaluation_checkpoint  # noqa: E402
from obruxo_training.features import PITCH_SIZE, TIMBRE_SHAPE, tensor_digest  # noqa: E402
from obruxo_training.model import MODEL_SPEC, PilotInverse, conditioning_from_spec, loss_terms, model_spec  # noqa: E402
from obruxo_training.pilot import PilotPool  # noqa: E402
from obruxo_training.trainer import TrainConfig, load_model, train_pilot  # noqa: E402
from test_stream import FakeRendererFactory  # noqa: E402


def features():
    generator = torch.Generator().manual_seed(6)
    return {"timbre": torch.rand((2, *TIMBRE_SHAPE), generator=generator), "pitch": torch.rand((2, PITCH_SIZE), generator=generator)}


def test_default_model_contract_is_unchanged_and_initial_common_predictions_match():
    codec = PilotCodec()
    assert model_spec(codec) == MODEL_SPEC
    assert conditioning_from_spec(MODEL_SPEC) == "concat"
    torch.manual_seed(11)
    baseline = PilotInverse(codec)
    torch.manual_seed(11)
    attention = PilotInverse(codec, "cross_attention")
    assert all(torch.equal(value, attention.state_dict()[name]) for name, value in baseline.state_dict().items())
    values = features()
    left, right = baseline(values), attention(values)
    assert torch.equal(left["continuous"], right["continuous"])
    assert all(torch.equal(a, b) for a, b in zip(left["categorical"], right["categorical"], strict=True))
    assert conditioning_from_spec(model_spec(codec, "cross_attention")) == "cross_attention"
    with pytest.raises(ValueError):
        conditioning_from_spec({"conditioning": "unknown"})
    with pytest.raises(ValueError):
        TrainConfig(conditioning="unknown")


def test_performance_tokens_preserve_frozen_posterior_time_layout():
    model = PilotInverse(conditioning="cross_attention")
    values = features()
    values["pitch"] = torch.arange(2 * PITCH_SIZE).reshape(2, PITCH_SIZE).float() / 10000
    observed = []
    handle = model.performance_projection.register_forward_pre_hook(lambda module, args: observed.append(args[0].detach()))
    model(values)
    handle.remove()
    tokens = observed[0]
    assert tokens.shape == (2, 8, 440)
    for time in range(8):
        assert tokens[0, time, 0] == values["pitch"][0, time]
        assert tokens[0, time, 88] == values["pitch"][0, 88 * 8 + time]
        assert tokens[0, time, 176] == values["pitch"][0, (88 + 88) * 8 + time]


def test_zero_start_gate_receives_gradients_then_activates_conditioning_weights():
    codec = PilotCodec()
    model = PilotInverse(codec, "cross_attention")
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    values = features()
    targets = [codec.default_target()] * 2
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss_terms(model(values), targets, codec)["loss"].backward()
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
        if step == 0:
            assert model.conditioning_gain.grad != 0
        else:
            assert model.performance_projection[0].weight.grad.abs().sum() > 0
        optimizer.step()
    assert model.conditioning_gain != 0


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    pool = PilotPool.create(24, seed=0)
    root = tmp_path_factory.mktemp("conditioning")
    base = TrainConfig(steps=4, replay_capacity=3, batch_size=2, checkpoint_every=2, render_budget=3, cpu_threads=2)
    train_pilot(pool, root / "concat", config=base, renderer_factory=FakeRendererFactory())
    attention = replace(base, conditioning="cross_attention")
    train_pilot(pool, root / "full", config=attention, renderer_factory=FakeRendererFactory(), seed_replay=root / "concat/checkpoint.pt")
    def interrupt(step):
        if step == 3:
            raise InterruptedError("after committed optimizer step 2")
    with pytest.raises(InterruptedError):
        train_pilot(pool, root / "resume", config=attention, renderer_factory=FakeRendererFactory(),
                    seed_replay=root / "concat/checkpoint.pt", after_step=interrupt)
    train_pilot(pool, root / "recovered", config=attention, renderer_factory=FakeRendererFactory(), resume=root / "resume/checkpoint.pt")
    return pool, root


def test_conditioning_resume_export_and_frozen_frontend_are_exact(trained):
    pool, root = trained
    full, recovered = [torch.load(root / name / "checkpoint.pt", weights_only=True) for name in ("full", "recovered")]
    assert tensor_digest(full["model"]) == tensor_digest(recovered["model"])
    assert torch.equal(full["batch_rng"], recovered["batch_rng"])
    for index, state in full["optimizer"]["state"].items():
        for name, value in state.items():
            assert torch.equal(value, recovered["optimizer"]["state"][index][name])
    checked, model, frontend, _ = load_evaluation_checkpoint(root / "full/checkpoint.pt", pool)
    exported, exported_frontend, codec = load_model(root / "full/model.pt")
    assert model.conditioning == exported.conditioning == "cross_attention"
    assert tensor_digest(model.state_dict()) == tensor_digest(exported.state_dict())
    assert frontend.frozen_digest == exported_frontend.frozen_digest
    assert codec.to_dict() == pool.codec.to_dict()
    baseline = torch.load(root / "concat/checkpoint.pt", weights_only=True)
    assert "conditioning" not in baseline["run_contract"]["training"]
    assert checked["run_contract"]["training"]["conditioning"] == "cross_attention"


def test_architecture_comparison_requires_explicit_flag_and_same_codec_frontend(trained, tmp_path):
    pool, root = trained
    paths = {"concat": root / "concat/checkpoint.pt", "attention": root / "full/checkpoint.pt"}
    with pytest.raises(ValueError, match="share inference"):
        evaluate_comparison(pool, paths, paths["concat"], tmp_path / "blocked", count=1, renderer_factory=FakeRendererFactory())
    result = evaluate_comparison(pool, paths, paths["concat"], tmp_path / "allowed", count=1,
                                 renderer_factory=FakeRendererFactory(), allow_architecture_comparison=True)
    assert result["renders_accepted"] == result["renders_attempted"] == 12
    plan = json.loads((tmp_path / "allowed/plan.json").read_text())
    assert plan["architecture_comparison"]["attention"]["conditioning"] == "cross_attention_v1"
    assert not plan["test_split_evaluated"]

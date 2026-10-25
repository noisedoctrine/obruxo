import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from obruxo_data.hashing import canonical_sha256  # noqa: E402
from obruxo_data.midi import Performance  # noqa: E402
from obruxo_data.render.qa import audio_float32_sha256  # noqa: E402
from obruxo_training.codec import PilotCodec  # noqa: E402
from obruxo_training.features import PITCH_SIZE, TIMBRE_SHAPE, tensor_digest  # noqa: E402
from obruxo_training.model import PilotInverse  # noqa: E402
from obruxo_training.paired_native_step import reciprocal_updates  # noqa: E402
from obruxo_training.paired_native_training import fit_pairs  # noqa: E402
from obruxo_training.paired_performance_objective import INPUT_TRANSFER, evaluation_views  # noqa: E402
from obruxo_training.spectral import SpectralTargets  # noqa: E402


def sound(candidate, performance):
    x = np.arange(8192, dtype=np.float64) / 44100
    note_rate = int(canonical_sha256(performance.to_dict())[:4], 16) % 50
    wave = (.05 + .1 * candidate.continuous[0]) * np.sin(2 * np.pi * (100 + note_rate + 20 * candidate.categorical[0]) * x)
    return np.stack((wave * (1.1 - candidate.continuous[1]), wave * (.1 + candidate.continuous[1])), -1).astype(np.float32)


@pytest.fixture(params=[False, True], ids=["transfer", "input-and-transfer"])
def inputs(request):
    torch.set_num_threads(2)
    torch.manual_seed(5)
    codec, initial = PilotCodec(), PilotInverse()
    performances = [Performance(bpm=value).to_dict() for value in (100, 120, 140)]
    features = [{"timbre": torch.rand(*TIMBRE_SHAPE), "pitch": torch.rand(PITCH_SIZE)} for _ in range(3)]
    targets = [SpectralTargets()(torch.from_numpy(sound(codec.default_target(), Performance.from_dict(value))).T[None]) for value in performances]
    record = {"features": features, "targets": targets}
    records = type("Records", (), {"check_limits": lambda _self: None, "__getitem__": lambda _self, _index: record})()
    calls = []
    def render(candidate, performance, phase, label):
        assert phase == "training"
        calls.append(label)
        audio = sound(candidate, performance)
        return audio, {"request_id": canonical_sha256({"candidate": candidate.to_dict(), "performance": performance.to_dict()}),
                       "qa": {"audio_float32_sha256": audio_float32_sha256(audio)}}
    session = SimpleNamespace(codec=codec, render=render)
    plan = {"training": [{"target": codec.default_target().to_dict(), "performances": performances}], "updates": reciprocal_updates(1),
            "production_training": {"forward": "unperturbed_argmax", "parameter_logits": "raw", "scales": [32., 29.]},
            "optimizer": {"learning_rate": .0001, "seed": 0}, "limits": {"optimizer_attempts": 4}}
    if request.param:
        plan.update(reconstruction_objective=INPUT_TRANSFER, rng_initialization="python_numpy_and_torch_from_optimizer_seed")
    def fresh():
        return {name: copy.deepcopy(initial) for name in ("parameter", "spectral")}
    return plan, records, session, fresh, calls


def adam_tensors(saved):
    return {f"{arm}:{index}:{name}": value for arm, optimizer in saved["optimizers"].items()
            for index, state in optimizer["state"].items() for name, value in state.items() if torch.is_tensor(value)}


def test_checkpoint_resume_reproduces_both_models_adam_and_individual_measurements(tmp_path, inputs):
    plan, records, session, fresh, calls = inputs
    full, interrupted = tmp_path / "full", tmp_path / "interrupted"
    full.mkdir()
    interrupted.mkdir()
    expected = fit_pairs(fresh(), records, session, full, plan)
    full_saved = torch.load(full / "checkpoint.pt", map_location="cpu", weights_only=True)
    def pause_after_one(state):
        if state["step"] == 1:
            raise RuntimeError("simulated interruption after durable checkpoint")
    with pytest.raises(RuntimeError, match="durable checkpoint"):
        fit_pairs(fresh(), records, session, interrupted, plan, after_step=pause_after_one)
    saved = torch.load(interrupted / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert saved["step"] == 1
    assert fit_pairs(fresh(), records, session, interrupted, plan, saved) == expected
    recovered = torch.load(interrupted / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert all(tensor_digest(recovered["models"][name]) == tensor_digest(full_saved["models"][name]) for name in ("parameter", "spectral"))
    assert tensor_digest(adam_tensors(recovered)) == tensor_digest(adam_tensors(full_saved))
    assert (interrupted / "checkpoint.previous.pt").exists()
    measured = [json.loads(line) for line in (interrupted / "measurements.jsonl").read_text().splitlines()]
    assert {row["example"] for row in measured} == {0, 1}
    assert len(measured) == sum(row["gradient_queries"] for row in expected)
    for row in measured:
        example = plan["updates"][row["step"] - 1]["examples"][row["example"]]
        views = evaluation_views(plan, example)
        assert all(row[key] == example[key] for key in ("state", "input_view"))
        assert row["view"] == views[row["query"] % len(views)]
    for update in expected:
        for index, estimate in enumerate(update["estimates"]):
            rows = [row for row in measured if row["step"] == update["step"] and row["example"] == index]
            views = evaluation_views(plan, update["examples"][index])
            assert sum(row["loss"] for row in rows[:len(views)]) / len(views) == estimate["value"]
            for start in range(0, len(rows), len(views)):
                components = rows[start:start + len(views)]
                assert tuple(row["view"] for row in components) == views
                assert all(row["candidate"] == components[0]["candidate"] for row in components)


def test_budget_rejection_does_not_charge_or_render(tmp_path, inputs):
    plan, records, session, fresh, calls = inputs
    plan = copy.deepcopy(plan)
    plan["limits"]["optimizer_attempts"] = 0
    with pytest.raises(RuntimeError, match="budget exhausted"):
        fit_pairs(fresh(), records, session, tmp_path, plan)
    assert calls == [] and not (tmp_path / "updates.jsonl").exists()


def test_wrong_view_pair_rejected_before_any_render(tmp_path, inputs):
    plan, records, session, fresh, calls = inputs
    plan = copy.deepcopy(plan)
    plan["updates"][0]["examples"][1]["view"] = 1
    with pytest.raises(ValueError, match="reciprocal"):
        fit_pairs(fresh(), records, session, tmp_path, plan)
    assert calls == []


def test_attempt_ledger_cannot_skip_the_first_pair(tmp_path, inputs):
    plan, records, session, fresh, calls = inputs
    (tmp_path / "updates.jsonl").write_text(json.dumps({"attempt": 0, "step": 2, "examples": plan["updates"][1]["examples"]}) + "\n")
    with pytest.raises(ValueError, match="attempt ledger"):
        fit_pairs(fresh(), records, session, tmp_path, plan)
    assert calls == []


@pytest.mark.parametrize("inputs", [True], indirect=True)
def test_failed_second_performance_does_not_advance_either_model(tmp_path, inputs):
    plan, records, session, fresh, _calls = inputs
    models = fresh()
    before = {name: tensor_digest(model.state_dict()) for name, model in models.items()}
    original = session.render
    def fail_second(candidate, performance, phase, label):
        if label == "gradient:0:0:1":
            return None, {}
        return original(candidate, performance, phase, label)
    session.render = fail_second
    with pytest.raises(RuntimeError, match="not accepted"):
        fit_pairs(models, records, session, tmp_path, plan)
    assert {name: tensor_digest(model.state_dict()) for name, model in models.items()} == before
    assert not (tmp_path / "checkpoint.pt").exists()
    assert len((tmp_path / "updates.jsonl").read_text().splitlines()) == 1
    assert len((tmp_path / "measurements.jsonl").read_text().splitlines()) == 1

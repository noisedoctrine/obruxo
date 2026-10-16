from copy import deepcopy
from dataclasses import replace
import json

import pytest

torch = pytest.importorskip("torch")

from obruxo_training.candidate_data import CandidateConfig, collect_candidates, dataset_digest, training_panel, validate_dataset  # noqa: E402
from obruxo_training.candidate_learning import train_candidates  # noqa: E402
from obruxo_training.candidates import confirmed_improvement, propose_candidates  # noqa: E402
from obruxo_training.evaluate import evaluate_comparison, load_evaluation_checkpoint  # noqa: E402
from obruxo_training.features import tensor_digest  # noqa: E402
from obruxo_training.model import batch_features  # noqa: E402
from obruxo_training.performances import initial_bank  # noqa: E402
from obruxo_training.pilot import PilotPool  # noqa: E402
from obruxo_training.trainer import TrainConfig, load_model, train_pilot  # noqa: E402
from test_stream import FakeRendererFactory  # noqa: E402


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    pool = PilotPool.create(64, seed=0)
    output = tmp_path_factory.mktemp("candidate-source") / "source"
    train_pilot(pool, output, config=TrainConfig(steps=3, replay_capacity=4, batch_size=4, checkpoint_every=1, render_budget=4, cpu_threads=2),
                renderer_factory=FakeRendererFactory())
    checkpoint = output / "checkpoint.pt"
    saved, model, frontend, _ = load_evaluation_checkpoint(checkpoint, pool)
    return pool, checkpoint, saved, model, frontend


@pytest.fixture(scope="module")
def collected(source, tmp_path_factory):
    pool, checkpoint, saved, _, _ = source
    config = CandidateConfig(training_states=8, validation_states=1, steps=6, checkpoint_every=2, cpu_threads=2)
    root = tmp_path_factory.mktemp("candidate-data")
    # Controlled scorer exercises confirmed improvements; real-render accuracy is covered by the native screen.
    scorer = pytest.MonkeyPatch()
    values = iter([1., .5, .8, .9, 1., .5] * 8)
    scorer.setattr("obruxo_training.candidate_data.audio_distance", lambda *args: {"mrstft": next(values)})
    def interrupt(event):
        if event["completed"] == 2:
            raise InterruptedError("two committed candidate cases")
    try:
        with pytest.raises(InterruptedError):
            collect_candidates(pool, checkpoint, root / "data", config=config, renderer_factory=FakeRendererFactory(), progress=interrupt)
        prefix = torch.load(root / "data/data.pt", weights_only=True)
        assert len(prefix["cases"]) == 2 and len(prefix["history"]) == 14
        data, summary = collect_candidates(pool, checkpoint, root / "data", config=config, renderer_factory=FakeRendererFactory(), resume=True)
    finally:
        scorer.undo()
    assert summary["attempted"] == 56 and summary["confirmed_improvements"] == 8
    return root, config, data, saved


def test_proposals_are_bounded_legal_deterministic_and_audio_only(source):
    pool, _, saved, model, _ = source
    item = saved["replay"][0]
    with torch.no_grad():
        prediction = model(batch_features([item]))
        proposals = propose_candidates(prediction, item["features"], saved["replay"], pool.codec, seed=7)
        repeated = propose_candidates(prediction, item["features"], saved["replay"], pool.codec, seed=7)
    assert proposals == repeated and 1 <= len(proposals) <= 4 and next(iter(proposals)) == "model"
    assert len(set(proposals.values())) == len(proposals)
    for target in proposals.values():
        restored = pool.codec.encode(pool.codec.decode(target))
        assert restored.categorical == target.categorical and restored.continuous == pytest.approx(target.continuous)
    panel = training_panel(pool, saved, 8)
    assert {entry.state_id for entry in panel}.isdisjoint(item["state_id"] for item in saved["replay"])
    assert all(pool.manifest.by_state[entry.state_id].split == "train" for entry in panel)


def test_confirmation_requires_two_successful_independent_improvements():
    assert confirmed_improvement({"model": [1., 1.1], "retrieval": [.8, .9]}) == "retrieval"
    assert confirmed_improvement({"model": [1., 1.1], "retrieval": [.1, 1.2]}) is None
    assert confirmed_improvement({"model": [1., 1.1], "retrieval": [.8]}) is None
    assert confirmed_improvement({"model": [1., None], "retrieval": [.8, .8]}) is None
    assert confirmed_improvement({"model": [1., 1.1], "retrieval": [.8, float("nan")]}) is None
    assert confirmed_improvement({"model": [1., 1.1], "retrieval": [.96, .96]}) is None


def test_collected_targets_bind_recipes_qa_features_and_confirmation(source, collected):
    pool, _, saved, model, frontend = source
    _, _, data, _ = collected
    validate_dataset(data, pool, saved, frontend, model)
    tampered = deepcopy(data)
    tampered["records"][0]["features"]["timbre"][0, 0, 0] += 1
    with pytest.raises(ValueError, match="changed"):
        validate_dataset(tampered, pool, saved, frontend, model)
    tampered = deepcopy(data)
    tampered["records"][0]["performance_id"] = initial_bank()[4].performance_id
    tampered["sha256"] = dataset_digest(tampered)
    with pytest.raises(ValueError, match="held-out"):
        validate_dataset(tampered, pool, saved, frontend, model)
    tampered = deepcopy(data)
    tampered["cases"][0]["scores"]["retrieval"] = [.1, 2.]
    tampered["sha256"] = dataset_digest(tampered)
    with pytest.raises(ValueError, match="confirmed"):
        validate_dataset(tampered, pool, saved, frontend, model)


def test_matched_candidate_training_recovers_exactly_and_exports_audio_only(source, collected, tmp_path):
    pool, _, source_saved, _, _ = source
    _, config, data, _ = collected
    control = train_candidates(pool, source_saved, data, tmp_path / "control", config=config)
    complete = train_candidates(pool, source_saved, data, tmp_path / "complete", config=config, auxiliary_weight=config.candidate_weight)
    def interrupt(step):
        if step == 3:
            raise InterruptedError("after committed step 2")
    with pytest.raises(InterruptedError):
        train_candidates(pool, source_saved, data, tmp_path / "resumed", config=config, auxiliary_weight=config.candidate_weight, after_step=interrupt)
    resumed = train_candidates(pool, source_saved, data, tmp_path / "resumed", config=config, auxiliary_weight=config.candidate_weight, resume=True)
    assert complete["model_sha256"] == resumed["model_sha256"] != control["model_sha256"]
    assert complete["uses"] == resumed["uses"] == control["uses"]
    assert complete["batch_rng_sha256"] == resumed["batch_rng_sha256"] == control["batch_rng_sha256"]
    full, recovered = [torch.load(tmp_path / name / "checkpoint.pt", weights_only=True) for name in ("complete", "resumed")]
    for index, state in full["optimizer"]["state"].items():
        for name, value in state.items():
            assert torch.equal(value, recovered["optimizer"]["state"][index][name])
    checked, model, frontend, _ = load_evaluation_checkpoint(tmp_path / "complete/checkpoint.pt", pool)
    exported, exported_frontend, codec = load_model(tmp_path / "complete/model.pt")
    assert tensor_digest(model.state_dict()) == tensor_digest(exported.state_dict()) == complete["model_sha256"]
    assert frontend.frozen_digest == exported_frontend.frozen_digest
    assert checked["run_contract"]["inference_input"] == "audio_only" and codec.to_dict() == pool.codec.to_dict()
    with pytest.raises(ValueError, match="configuration"):
        train_candidates(pool, source_saved, data, tmp_path / "changed", config=replace(config, steps=7))


def test_clipped_source_never_becomes_a_candidate_training_record(source, tmp_path):
    pool, checkpoint, _, _, _ = source
    def interrupt(event):
        raise InterruptedError
    with pytest.raises(InterruptedError):
        collect_candidates(pool, checkpoint, tmp_path / "bad", config=CandidateConfig(training_states=8, validation_states=1),
                           renderer_factory=FakeRendererFactory("clipped"), progress=interrupt)
    data = torch.load(tmp_path / "bad/data.pt", weights_only=True)
    assert data["records"] == [] and not data["cases"][0]["source_accepted"]
    assert data["history"][0]["status"] == "rejected" and "audio.clipping" in data["history"][0]["diagnostic_codes"]


def test_native_comparison_enforces_attempt_cap_including_recovery(source, tmp_path):
    pool, checkpoint, _, _, _ = source
    checkpoints = {"first": checkpoint, "second": checkpoint}
    output = tmp_path / "limited"
    for resume in (False, True):
        with pytest.raises(RuntimeError, match="budget exhausted"):
            evaluate_comparison(pool, checkpoints, checkpoint, output, count=1, renderer_factory=FakeRendererFactory(),
                                render_attempt_limit=1, resume=resume)
    history = [json.loads(line) for line in (output / "renders.jsonl").read_text().splitlines()]
    assert len(history) == 1


def test_candidate_config_and_empty_confirmation_are_rejected():
    with pytest.raises(ValueError):
        CandidateConfig(candidate_weight=float("nan"))
    with pytest.raises(ValueError):
        CandidateConfig(batch_size=3)
    with pytest.raises(ValueError):
        confirmed_improvement({"retrieval": [.1, .2]})


def test_candidate_screen_connects_matched_arms_to_native_validation(source, collected, tmp_path, monkeypatch):
    from obruxo_training.candidate_screen import run_candidate_screen
    pool, checkpoint, _, _, _ = source
    _, config, data, _ = collected
    collection = {"attempted": len(data["history"]), "accepted": len(data["history"]), "confirmed_improvements": 8}
    monkeypatch.setattr("obruxo_training.candidate_screen.collect_candidates", lambda *args, **kwargs: (data, collection))
    result = run_candidate_screen(pool, checkpoint, tmp_path / "screen", config=config, renderer_factory=FakeRendererFactory())
    assert result["validation_renders_attempted"] == result["validation_renders_accepted"] == 12
    assert result["arms"]["control"]["uses"] == result["arms"]["render_selected"]["uses"]
    assert not result["oracle_candidate_selection_used_at_inference"] and not result["final_test_used"]
    assert json.loads((tmp_path / "screen/status.json").read_text())["complete"]


def test_candidate_feature_budget_is_checked_before_native_collection(source, tmp_path):
    pool, checkpoint, _, _, _ = source
    with pytest.raises(ValueError, match="feature budget"):
        collect_candidates(pool, checkpoint, tmp_path / "too-small", config=CandidateConfig(training_states=32, replay_mib=1),
                           renderer_factory=FakeRendererFactory("crash"))
    assert not (tmp_path / "too-small").exists()

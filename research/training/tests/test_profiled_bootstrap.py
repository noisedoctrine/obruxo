from dataclasses import replace
import json

import pytest

torch = pytest.importorskip("torch")

from obruxo_training.checkpoints import load_training_checkpoint  # noqa: E402
from obruxo_training.cli import main  # noqa: E402
from obruxo_training.evaluate import evaluate_pilot  # noqa: E402
from obruxo_training.pilot import PilotPool, PilotSampler  # noqa: E402
from obruxo_training.profiled_sampler import ProfiledSampler, profile_metadata, training_sampler  # noqa: E402
from obruxo_training.reproducibility import fixed_generator_phases  # noqa: E402
from obruxo_training.stream import OnlineRenderStream, StreamConfig  # noqa: E402
from obruxo_training.trainer import TrainConfig, train_pilot  # noqa: E402
from profile_renderer import ProfileRendererFactory  # noqa: E402


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("profiled")
    pool = PilotPool.create(24, seed=0)
    config = TrainConfig(steps=6, replay_capacity=3, batch_size=2, checkpoint_every=2, render_budget=3,
                         cpu_threads=2, sampling_policy="state_round_robin", prediction_scope="basic", preroll_frames=32768)

    def run(name, **kwargs):
        return train_pilot(pool, root / name, config=config, renderer_factory=ProfileRendererFactory(), **kwargs)

    full = run("full")

    def interrupt(step):
        if step == 3:
            raise InterruptedError("one update after the saved recovery point")

    with pytest.raises(InterruptedError):
        run("crash", after_step=interrupt)
    resumed = run("resumed", resume=root / "crash/checkpoint.pt")
    return root, pool, config, full, resumed


def test_profile_keeps_original_source_schedule_but_separates_request_recipe_and_cursor():
    pool = PilotPool.create(24, seed=0)
    legacy = PilotSampler(pool, seed=0, policy="state_round_robin")
    profiled = ProfiledSampler(pool, seed=0, policy="state_round_robin", preroll_frames=32768)
    assert legacy.epoch_size == profiled.epoch_size and legacy.sampler_id != profiled.sampler_id
    for sequence in range(2 * legacy.epoch_size):
        left, right = legacy.recipe_at(sequence), profiled.recipe_at(sequence)
        for key in ("state_id", "lineage_id", "raw_sha256", "canonical_sha256", "performance_id", "split", "sequence_id"):
            assert getattr(left, key) == getattr(right, key)
        original, controlled = left.render_request(), right.render_request()
        expected = replace(original, preset_json=fixed_generator_phases(original.preset_json.encode()).decode(), preroll_frames=32768)
        assert controlled.to_dict() == expected.to_dict()
        assert original.request_id != controlled.request_id and left.recipe_id != right.recipe_id
        assert controlled.performance == original.performance
    cursor = OnlineRenderStream(legacy, 1).state_dict()
    with pytest.raises(ValueError):
        OnlineRenderStream(profiled, 1, resume=cursor)
    assert type(training_sampler(pool, TrainConfig())) is PilotSampler
    assert profile_metadata(0) == {}


@pytest.mark.parametrize("frames", (False, 0.0, -1, 8192, 40960))
def test_profile_rejects_unqualified_values(frames):
    with pytest.raises(ValueError):
        TrainConfig(prediction_scope="basic", preroll_frames=frames)


def test_profiled_recovery_is_exact_and_export_declares_source_rendering(trained):
    root, _, _, full, resumed = trained
    assert full["model_state_sha256"] == resumed["model_state_sha256"]
    assert full["final_metrics"] == resumed["final_metrics"] and full["cursor"] == resumed["cursor"]
    assert resumed["feature_seconds_this_segment"] == resumed["fresh_accepted_this_segment"] == 0
    left, right = [load_training_checkpoint(root / name / "checkpoint.pt") for name in ("full", "resumed")]
    for index, state in left["optimizer"]["state"].items():
        assert all(torch.equal(value, right["optimizer"]["state"][index][name]) for name, value in state.items())
    assert torch.equal(left["batch_rng"], right["batch_rng"])
    assert torch.equal(left["rng"]["torch"], right["rng"]["torch"])
    for item in left["replay"]:
        assert all(item[name] == value for name, value in profile_metadata(32768).items())
    exported = torch.load(root / "full/model.pt", weights_only=True)
    assert all(exported[name] == value for name, value in profile_metadata(32768).items())
    assert exported["qualified_for_release"] is False
    from obruxo_training.evaluate import load_evaluation_checkpoint
    from obruxo_training.inference_profile import model_rendering
    from obruxo_training.trainer import inference_state
    _, pool, _, _, _ = trained
    _, restored, frontend, _ = load_evaluation_checkpoint(root / "full/checkpoint.pt", pool)
    assert model_rendering(restored) == profile_metadata(32768)
    assert all(inference_state(restored, frontend, pool.codec, pool.pool_id)[name] == value for name, value in profile_metadata(32768).items())


def test_profile_refuses_mixed_seed_resume_and_tampered_replay(trained, tmp_path):
    root, pool, config, _, _ = trained
    source = root / "full/checkpoint.pt"
    for name, kwargs in (("seed", {"seed_replay": source}), ("resume", {"resume": source})):
        with pytest.raises(ValueError, match="different render profile|contract changed"):
            train_pilot(pool, tmp_path / name, config=replace(config, preroll_frames=0), **kwargs)
    saved = load_training_checkpoint(source)
    saved["replay"][0].pop("render_phase_controls")
    torch.save(saved, tmp_path / "tampered.pt")
    with pytest.raises(ValueError, match="phase policy changed"):
        train_pilot(pool, tmp_path / "tampered", config=config, resume=tmp_path / "tampered.pt")
    with pytest.raises(ValueError, match="one fresh native worker"):
        train_pilot(pool, tmp_path / "batch", config=config, stream_config=StreamConfig(worker_jobs=2))


def test_profiled_validation_controls_every_render(trained, tmp_path):
    root, pool, _, _, _ = trained
    output = tmp_path / "validation"
    result = evaluate_pilot(pool, root / "full/checkpoint.pt", output, count=1, renderer_factory=ProfileRendererFactory())
    plan = json.loads((output / "plan.json").read_text())
    assert all(plan[name] == value for name, value in profile_metadata(32768).items())
    assert plan["evaluation_cpu_threads"] == 2
    rows = [json.loads(line) for line in (output / "renders.jsonl").read_text().splitlines()]
    assert len(rows) == 10 and all(row["status"] == "accepted" for row in rows)
    assert all(row["provenance"]["settings"]["preroll_frames"] == 32768 for row in rows)
    assert result["renders_accepted"] == result["renders_attempted"] == 10
    assert result["aggregates"]["source_repeat"]["reference"]["mean_mrstft"] == 0
    resumed = evaluate_pilot(pool, root / "full/checkpoint.pt", output, count=1,
                             renderer_factory=ProfileRendererFactory(), resume=True)
    assert resumed["renders_attempted_this_segment"] == 0 and resumed["aggregates"] == result["aggregates"]
    (output / "prediction-000.vital").write_bytes(b"changed")
    with pytest.raises(ValueError, match="prediction changed"):
        evaluate_pilot(pool, root / "full/checkpoint.pt", output, count=1, resume=True)


def test_basic_cli_defaults_to_qualified_profile_and_retains_explicit_legacy_option(tmp_path, monkeypatch):
    pool = PilotPool.create(24, seed=0)
    path = tmp_path / "pool.json"
    pool.save(path)
    calls = []

    def capture(_pool, _output, **kwargs):
        calls.append(kwargs["config"])
        return {"complete": True, "tiny_overfit_gate": True}

    monkeypatch.setattr("obruxo_training.trainer.train_pilot", capture)
    base = ["train", "--pool", str(path), "--output", str(tmp_path / "unused"), "--prediction-scope", "basic"]
    assert main(base) == main([*base, "--preroll-frames", "0"]) == 0
    assert [config.preroll_frames for config in calls] == [32768, 0]

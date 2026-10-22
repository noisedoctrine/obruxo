from dataclasses import replace
import copy

import pytest

torch = pytest.importorskip("torch")

from obruxo_training.checkpoints import load_training_checkpoint  # noqa: E402
from obruxo_training.pilot import PilotPool  # noqa: E402
from obruxo_training.features import AudioFeatures, tensor_digest  # noqa: E402
from obruxo_training.model import PilotInverse  # noqa: E402
from obruxo_training.profiled_sampler import training_sampler  # noqa: E402
from obruxo_training.trainer import TrainConfig, export_model, train_pilot  # noqa: E402
from obruxo_training.evaluate import load_evaluation_checkpoint  # noqa: E402
from test_profiled_bootstrap import ProfileRendererFactory  # noqa: E402


def test_varied_training_resume_matches_and_old_input_contract_cannot_be_substituted(tmp_path):
    pool = PilotPool.create(24, seed=0)
    config = TrainConfig(steps=6, replay_capacity=4, batch_size=4, checkpoint_every=2, render_budget=4, cpu_threads=2,
                         prediction_scope="basic", preroll_frames=32768, sampling_policy="paired_views", input_performances="varied")
    torch.manual_seed(99)
    frontend = AudioFeatures()
    initial_model = PilotInverse(pool.codec)
    initial_path = tmp_path / "initial.pt"
    export_model(initial_path, initial_model, frontend, pool.codec, pool.pool_id, preroll_frames=32768)

    def run(name, **kwargs):
        return train_pilot(pool, tmp_path / name, config=config, renderer_factory=ProfileRendererFactory(),
                           initial_model=None if "resume" in kwargs else initial_path, **kwargs)

    full = run("full")
    run("stopped", stop_after=2)
    resumed = run("resumed", resume=tmp_path / "stopped/checkpoint.pt")
    assert full["model_state_sha256"] == resumed["model_state_sha256"] and full["cursor"] == resumed["cursor"]
    left, right = [load_training_checkpoint(tmp_path / name / "checkpoint.pt") for name in ("full", "resumed")]
    assert left["run_contract"]["observation_contract"] == training_sampler(pool, config).observation_contract
    torch.testing.assert_close(left["optimizer"], right["optimizer"], rtol=0, atol=0)
    assert torch.equal(left["batch_rng"], right["batch_rng"])
    assert torch.equal(left["rng"]["torch"], right["rng"]["torch"])
    assert len({item["performance_id"] for item in left["replay"]}) == 4
    assert len({item["state_id"] for item in left["replay"]}) == 2
    initial = torch.load(tmp_path / "full/checkpoint.initial.pt", weights_only=True)["state"]
    assert tensor_digest(initial["model"]) == tensor_digest(initial_model.state_dict())
    assert initial["optimizer"]["state"] == {} and initial["run_contract"]["initial_model"]["model_state_sha256"] == tensor_digest(initial["model"])
    load_evaluation_checkpoint(tmp_path / "full/checkpoint.pt", pool)
    with pytest.raises(ValueError, match="contract changed"):
        train_pilot(pool, tmp_path / "old-inputs", config=replace(config, input_performances="initial"),
                    resume=tmp_path / "stopped/checkpoint.pt")
    source = torch.load(initial_path, weights_only=True)
    for field, value in (("training_pool_id", "other"), ("model_spec", {}), ("frontend", {}), ("render_profile", {"preroll_frames": 0})):
        changed = copy.deepcopy(source)
        changed[field] = value
        path = tmp_path / f"bad-{field}.pt"
        torch.save(changed, path)
        with pytest.raises(ValueError, match="initial model has incompatible"):
            train_pilot(pool, tmp_path / f"bad-{field}", config=config, initial_model=path)
        assert not (tmp_path / f"bad-{field}").exists()

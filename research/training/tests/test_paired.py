from dataclasses import dataclass, replace
import json

import numpy as np
import pytest

from obruxo_training.pilot import PilotPool, PilotSampler


@dataclass(frozen=True)
class VariedRendererFactory:
    def __call__(self):
        return VariedRenderer()


class VariedRenderer:
    def render(self, request):
        from test_stream import FakeRenderer
        result = FakeRenderer("valid").render(request)
        settings = json.loads(request.preset_json)["settings"]
        pitch = request.performance.note_spans()[0].pitch
        wave = .08 * np.sin(np.arange(len(result.audio)) * (.015 + settings["osc_1_level"] * .2 + pitch * .0001))
        result.audio[:, 0] = wave * (1 - settings["osc_1_pan"] * .5)
        result.audio[:, 1] = wave * (1 + settings["osc_1_pan"] * .5)
        result.audio[-request.sample_rate:] = 0
        return result


@dataclass(frozen=True)
class RejectOneFactory:
    request_id: str

    def __call__(self):
        return RejectOneRenderer(self.request_id)


class RejectOneRenderer(VariedRenderer):
    def __init__(self, request_id):
        self.request_id = request_id

    def render(self, request):
        result = super().render(request)
        if request.request_id == self.request_id:
            result.audio[0, 0] = 2.
        return result


@pytest.fixture(scope="module")
def pool():
    return PilotPool.create(24, seed=0)


def test_adjacent_views_cover_states_and_rotate_all_reference_performances(pool):
    sampler = PilotSampler(pool, policy="paired_views")
    states = {entry.state_id for entry in pool.manifest.assignments if entry.split == "train"}
    assert sampler.epoch_size == 2 * len(states)
    performances = {state: set() for state in states}
    for epoch in range(2):
        seen = []
        for position in range(0, sampler.epoch_size, 2):
            recipes = [sampler.recipe_at(epoch * sampler.epoch_size + position + offset) for offset in (0, 1)]
            assert recipes[0].state_id == recipes[1].state_id
            assert recipes[0].performance_id != recipes[1].performance_id
            assert all(recipe.split == "train" and recipe.performance_role == "reference" for recipe in recipes)
            seen.append(recipes[0].state_id)
            performances[recipes[0].state_id].update(recipe.performance_id for recipe in recipes)
        assert set(seen) == states and len(seen) == len(states)
    assert all(len(views) == 4 for views in performances.values())
    assert sampler.recipe_at(3).recipe_id == PilotSampler(pool, policy="paired_views").recipe_at(3).recipe_id
    assert sampler.sampler_id != PilotSampler(pool).sampler_id


def test_paired_batch_excludes_pending_views_and_has_distinct_negative_states():
    torch = pytest.importorskip("torch")
    from obruxo_training.paired import complete_pairs, paired_batch_indices
    items = [{"sequence_id": index, "state_id": str(index // 2 % 3), "performance_id": str(index % 2)} for index in range(9)]
    groups = complete_pairs(items)
    assert len(groups) == 3 and sum(map(len, groups.values())) == 4
    indices = paired_batch_indices(items, 6, torch.Generator().manual_seed(0))
    assert len({items[index]["state_id"] for index in indices}) == 3
    assert 8 not in indices
    assert all(items[a]["state_id"] == items[b]["state_id"] for a, b in zip(indices[::2], indices[1::2], strict=True))
    assert complete_pairs(items[1:2]) == {}
    with pytest.raises(ValueError, match="insufficient"):
        paired_batch_indices(items, 8, torch.Generator())
    items[1]["performance_id"] = items[0]["performance_id"]
    with pytest.raises(ValueError, match="distinct performances"):
        complete_pairs(items)


def test_identity_loss_rewards_pairs_resists_collapse_and_has_gradients():
    torch = pytest.importorskip("torch")
    from obruxo_training.paired import identity_terms
    good = torch.eye(4).repeat_interleave(2, dim=0)
    shuffled = good.clone()
    shuffled[1::2] = good[1::2].roll(1, 0)
    aligned, wrong = identity_terms(good), identity_terms(shuffled)
    collapsed = identity_terms(torch.ones_like(good))
    assert aligned["identity_loss"] < wrong["identity_loss"]
    assert aligned["identity_loss"] < collapsed["identity_loss"]
    assert aligned["identity_accuracy"] == 1 and collapsed["identity_accuracy"] == .25
    torch.testing.assert_close(identity_terms(good * 7)["identity_loss"], aligned["identity_loss"])
    representation = (good + torch.randn(8, 4) * .2).requires_grad_()
    identity_terms(representation)["identity_loss"].backward()
    assert torch.isfinite(representation.grad).all() and representation.grad[::2].abs().sum() > 0
    assert representation.grad[1::2].abs().sum() > 0
    with pytest.raises(ValueError, match="two complete"):
        identity_terms(torch.ones(2, 4))


def test_identity_diagnostics_retain_failures_and_expose_collapse():
    pytest.importorskip("torch")
    from obruxo_training.paired import representation_statistics
    rows = [{"representations": {"good": {"reference": row.tolist(), "transfer": row.tolist()},
                                  "collapsed": {"reference": [1.] * 4, "transfer": [1.] * 4}}} for row in np.eye(4)]
    rows.append({"failure": "reference_render_failed"})
    stats = representation_statistics(rows, ["good", "collapsed"])["models"]
    assert stats["good"]["planned"] == 5 and stats["good"]["scored"] == 4
    assert stats["good"]["failed_or_unscorable"] == 1 and stats["good"]["bidirectional_top1"] == 1
    assert stats["collapsed"]["bidirectional_top1"] == .25 and stats["collapsed"]["normalized_spread"] == 0
    assert stats["good"]["cosine_margin"] > stats["collapsed"]["cosine_margin"]


@pytest.mark.parametrize("changes", [{"identity_weight": float("nan")}, {"identity_weight": -1},
                                    {"identity_temperature": 0}, {"identity_weight": 1},
                                    {"sampling_policy": "paired_views", "batch_size": 3}])
def test_invalid_identity_training_contracts_are_rejected(changes):
    pytest.importorskip("torch")
    from obruxo_training.trainer import TrainConfig
    with pytest.raises(ValueError):
        TrainConfig(**changes)


def test_qualified_cache_excludes_broken_pairs_and_detects_tampering(pool, tmp_path):
    torch = pytest.importorskip("torch")
    from obruxo_training.features import AudioFeatures
    from obruxo_training.paired_features import prepare_paired_features, validate_cache
    from obruxo_training.stream import StreamConfig
    from obruxo_training.trainer import TrainConfig, train_pilot
    from test_stream import FakeRendererFactory
    sampler = PilotSampler(pool, policy="paired_views")
    rejected_request = sampler.recipe_at(1).render_request().request_id
    stream = StreamConfig(prefetch_clips=1, timeout_seconds=30)
    result = prepare_paired_features(pool, tmp_path / "features", states=3, stream_config=stream,
                                     renderer_factory=RejectOneFactory(rejected_request))
    assert result["attempted"] == 6 and result["accepted"] == 5 and result["retained_views"] == 4
    assert result["qualified_states"] == 2 and result["discarded_unpaired_accepted_views"] == 1
    path = tmp_path / "features/features.pt"
    saved = torch.load(path, weights_only=True)
    assert [item["sequence_id"] for item in saved["replay"]] == [2, 3, 4, 5]
    config = TrainConfig(mode="online", steps=2, replay_capacity=6, batch_size=4, fresh_examples=4, render_budget=6,
                         sampling_policy="paired_views", identity_weight=1., cpu_threads=2)
    trained = train_pilot(pool, tmp_path / "trained", config=config, stream_config=stream, seed_replay=path,
                          renderer_factory=FakeRendererFactory("crash"))
    assert trained["complete"] and trained["totals"]["render_attempts"] == 0 and trained["seeded_feature_examples"] == 4
    assert trained["cursor"]["next_sequence_id"] == 6
    saved["replay"][0]["features"]["pitch"][0] += .01
    with pytest.raises(ValueError, match="content changed"):
        validate_cache(saved, pool, sampler, AudioFeatures(), stream)


def test_paired_training_recovers_from_a_pending_view_and_reuses_features(pool, tmp_path):
    torch = pytest.importorskip("torch")
    from obruxo_training.features import tensor_digest
    from obruxo_training.trainer import TrainConfig, load_model, train_pilot
    config = TrainConfig(mode="online", steps=6, replay_capacity=8, batch_size=4, render_budget=8,
                         fresh_examples=8, checkpoint_every=1, cpu_threads=2, sampling_policy="paired_views", identity_weight=1.)

    def run(name, **kwargs):
        return train_pilot(pool, tmp_path / name, config=config, renderer_factory=VariedRendererFactory(), **kwargs)

    full = run("full")

    def interrupt(event):
        if event["stage"] == "render" and event["sequence_id"] == 4:
            raise InterruptedError("pending first view persisted before the paired batch")

    with pytest.raises(InterruptedError):
        run("crash", progress=interrupt)
    pending = torch.load(tmp_path / "crash/checkpoint.pt", weights_only=True)
    assert pending["cursor"]["next_sequence_id"] == 5
    resumed = run("resumed", resume=tmp_path / "crash/checkpoint.pt")
    assert full["model_state_sha256"] == resumed["model_state_sha256"]
    assert full["final_metrics"] == resumed["final_metrics"] and full["cursor"] == resumed["cursor"]
    assert full["totals"]["accepted"] == 8 and resumed["feature_seconds_this_segment"] >= 0
    from obruxo_training.paired_features import prepare_paired_features
    from obruxo_training.stream import StreamConfig
    recovered = prepare_paired_features(pool, tmp_path / "recovered-cache", states=4, stream_config=StreamConfig(prefetch_clips=1),
                                        recover_checkpoint=tmp_path / "full/checkpoint.pt", recover_log=tmp_path / "full/renders.jsonl")
    assert recovered["new_render_attempts"] == 0 and recovered["retained_views"] == 8 and recovered["origin"]["source_step"] == 6
    model, _, _ = load_model(tmp_path / "full/model.pt")
    assert tensor_digest(model.state_dict()) == full["model_state_sha256"]
    pending["replay"][0]["performance_id"] = "0" * 64
    torch.save(pending, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="paired replay performance"):
        run("bad", resume=tmp_path / "bad.pt")
    results = []
    for weight in (0., 1.):
        result = train_pilot(pool, tmp_path / f"seed-{weight}", config=replace(config, identity_weight=weight),
                             seed_replay=tmp_path / "full/checkpoint.pt", renderer_factory=VariedRendererFactory())
        assert result["totals"]["render_attempts"] == 0 and result["seeded_feature_examples"] == 8
        results.append(result)
    assert results[0]["model_state_sha256"] != results[1]["model_state_sha256"]
    from obruxo_training.evaluate import evaluate_comparison
    checkpoints = {"control": tmp_path / "seed-0.0/checkpoint.pt", "identity": tmp_path / "seed-1.0/checkpoint.pt"}
    output = tmp_path / "comparison"
    panel = [entry.state_id for entry in reversed(pool.entries) if pool.manifest.by_state[entry.state_id].split == "validation"][:2]

    def interrupt_evaluation(event):
        if event["case"] == 1:
            raise InterruptedError("resume after one committed representation pair")

    def compare(**kwargs):
        return evaluate_comparison(pool, checkpoints, tmp_path / "full/checkpoint.pt", output, count=2,
                                   renderer_factory=VariedRendererFactory(), identity_diagnostics=True, validation_states=panel, **kwargs)

    with pytest.raises(InterruptedError):
        compare(progress=interrupt_evaluation)
    comparison = compare(resume=True)
    assert comparison["completed_cases_reused"] == 1
    assert all(item["scored"] == 2 for item in comparison["identity_diagnostics"]["models"].values())
    assert all(len(case["representations"]["identity"]["transfer"]) == 128 for case in comparison["cases"])
    assert [case["state_id"] for case in comparison["cases"]] == panel
    with pytest.raises(ValueError, match="plan changed"):
        evaluate_comparison(pool, checkpoints, tmp_path / "full/checkpoint.pt", output, count=2,
                            renderer_factory=VariedRendererFactory(), resume=True)
    train_state = next(entry.state_id for entry in pool.entries if pool.manifest.by_state[entry.state_id].split == "train")
    with pytest.raises(ValueError, match="available cases"):
        evaluate_comparison(pool, checkpoints, tmp_path / "full/checkpoint.pt", tmp_path / "leak", count=1,
                            renderer_factory=VariedRendererFactory(), validation_states=[train_state])
    assert not (tmp_path / "leak").exists()

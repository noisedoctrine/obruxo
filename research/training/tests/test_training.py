from dataclasses import replace
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from obruxo_training.codec import PilotCodec, PilotTarget  # noqa: E402
from obruxo_training.cli import main  # noqa: E402
from obruxo_training.evaluate import audio_distance, curve_statistics, evaluate_comparison, evaluate_learning_curve, evaluate_pilot, retrieval_digest  # noqa: E402
from obruxo_training.features import AudioFeatures, tensor_digest  # noqa: E402
from obruxo_training.model import loss_terms  # noqa: E402
from obruxo_training.pilot import PilotPool, PilotSampler  # noqa: E402
from obruxo_training.stream import StreamConfig  # noqa: E402
from obruxo_training.trainer import TrainConfig, load_model, predict_audio, train_pilot  # noqa: E402
from test_stream import FakeRendererFactory  # noqa: E402


@pytest.fixture(scope="module")
def pool():
    return PilotPool.create(24, seed=0)


@pytest.fixture(scope="module")
def trained(pool, tmp_path_factory):
    root = tmp_path_factory.mktemp("trainer")
    config = TrainConfig(steps=8, replay_capacity=3, batch_size=2, checkpoint_every=2, render_budget=8, cpu_threads=2)
    stream_config = StreamConfig(prefetch_clips=1, timeout_seconds=20)

    def run(name, **kwargs):
        return train_pilot(pool, root / name, config=config, stream_config=stream_config,
                           renderer_factory=FakeRendererFactory(), **kwargs)

    full = run("full")

    def interrupt(step):
        if step == 5:
            raise InterruptedError("deliberate failure after the step-4 recovery point")

    with pytest.raises(InterruptedError):
        run("crash", after_step=interrupt)
    resumed = run("resumed", resume=root / "crash/checkpoint.pt")
    return root, config, stream_config, full, resumed


def test_interruption_replays_exact_model_optimizer_rng_and_consumption(trained):
    root, _, _, full, resumed = trained
    assert full["model_state_sha256"] == resumed["model_state_sha256"]
    assert full["final_metrics"] == resumed["final_metrics"]
    assert full["cursor"] == resumed["cursor"]
    assert full["frozen_weights_and_buffers_unchanged"]
    assert resumed["feature_seconds_this_segment"] == 0
    left, right = [torch.load(root / name / "checkpoint.pt", weights_only=True) for name in ("full", "resumed")]
    for index in left["optimizer"]["state"]:
        for name, tensor in left["optimizer"]["state"][index].items():
            assert torch.equal(tensor, right["optimizer"]["state"][index][name])
    assert torch.equal(left["batch_rng"], right["batch_rng"])
    assert torch.equal(left["rng"]["torch"], right["rng"]["torch"])
    assert (root / "crash/checkpoint.previous.pt").is_file()


def test_resume_refuses_changed_training_contract(pool, trained, tmp_path):
    root, config, stream_config, _, _ = trained
    with pytest.raises(ValueError, match="contract changed"):
        train_pilot(pool, tmp_path / "changed", config=replace(config, learning_rate=0.01), stream_config=stream_config,
                    resume=root / "crash/checkpoint.pt")


def test_resume_refuses_replay_target_or_cursor_tampering(pool, trained, tmp_path):
    root, config, stream_config, _, _ = trained
    saved = torch.load(root / "crash/checkpoint.pt", weights_only=True)
    saved["replay"][0]["state_id"] = "0" * 64
    torch.save(saved, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="training recipe"):
        train_pilot(pool, tmp_path / "bad", config=config, stream_config=stream_config, resume=tmp_path / "bad.pt")


def test_audio_only_export_roundtrip_and_frozen_statistics(trained):
    root, _, _, full, _ = trained
    model, frontend, codec = load_model(root / "full/model.pt")
    assert tensor_digest(model.state_dict()) == full["model_state_sha256"]
    audio = np.stack((np.sin(np.arange(44100) * .03), np.sin(np.arange(44100) * .05)), 1).astype(np.float32) * .1
    before = tensor_digest(frontend.basic_pitch.state_dict())
    frontend.train(True)
    target = predict_audio(model, frontend, codec, audio)
    assert codec.encode(codec.decode(target)).categorical == target.categorical
    assert tensor_digest(frontend.basic_pitch.state_dict()) == before
    assert not frontend.training and all(parameter.grad is None for parameter in frontend.parameters())
    features, swapped = frontend(audio), frontend(audio[:, ::-1])
    assert torch.equal(features["timbre"][0], swapped["timbre"][1])
    assert not torch.equal(features["timbre"][0], features["timbre"][1])


def test_masked_loss_excludes_inactive_controls_and_has_active_gradients():
    codec = PilotCodec()
    target = PilotTarget((.5,) * 9, (0, 0))
    values = torch.zeros(1, 9, requires_grad=True)
    logits = [torch.zeros(1, count, requires_grad=True) for count in (3, 2)]
    loss_terms({"continuous": values, "categorical": logits}, [target], codec)["loss"].backward()
    assert torch.equal(values.grad[0, 2:5], torch.zeros(3))
    assert torch.all(values.grad[0, [0, 1, 5, 6, 7, 8]] != 0)
    assert all(torch.isfinite(item.grad).all() and item.grad.abs().sum() > 0 for item in logits)


def test_online_checkpoint_after_ingestion_does_not_skip_or_repeat_a_view(pool, tmp_path):
    config = TrainConfig(mode="online", steps=3, replay_capacity=2, batch_size=1, checkpoint_every=1,
                         render_budget=5, cpu_threads=2)
    stream_config = StreamConfig(prefetch_clips=1)

    def crash(event):
        if event["stage"] == "render" and event["step"] == 1:
            raise InterruptedError("failure after render acknowledgement but before optimizer")

    def run(name, **kwargs):
        return train_pilot(pool, tmp_path / name, config=config, stream_config=stream_config,
                           renderer_factory=FakeRendererFactory(), **kwargs)

    full = run("full")
    with pytest.raises(InterruptedError):
        run("crash", progress=crash)
    resumed = run("resumed", resume=tmp_path / "crash/checkpoint.pt")
    assert full["model_state_sha256"] == resumed["model_state_sha256"]
    assert full["totals"]["accepted"] == resumed["totals"]["accepted"] == 3
    assert full["cursor"] == resumed["cursor"]


def test_bounded_refresh_rotates_states_and_resumes_after_ingestion(pool, tmp_path):
    config = TrainConfig(mode="online", steps=7, replay_capacity=3, batch_size=1, checkpoint_every=1,
                         render_budget=5, cpu_threads=2, sampling_policy="state_round_robin", updates_per_render=2,
                         fresh_examples=3)

    def crash(event):
        if event["stage"] == "render" and event["step"] == 2:
            raise InterruptedError("failure after a scheduled refresh")

    def run(name, **kwargs):
        return train_pilot(pool, tmp_path / name, config=config, renderer_factory=FakeRendererFactory(), **kwargs)

    full = run("full")
    with pytest.raises(InterruptedError):
        run("crash", progress=crash)
    resumed = run("resumed", resume=tmp_path / "crash/checkpoint.pt")
    assert full["model_state_sha256"] == resumed["model_state_sha256"]
    assert full["totals"]["accepted"] == resumed["totals"]["accepted"] == 3
    assert full["replay_distinct_states"] == 3 and full["fresh_example_target_reached"]
    assert full["cursor"] == resumed["cursor"]


def test_refresh_target_requires_sufficient_optimizer_steps():
    with pytest.raises(ValueError, match="step budget"):
        TrainConfig(mode="online", steps=4, batch_size=1, fresh_examples=3, updates_per_render=2)


def test_feature_seed_reuses_only_features_and_resets_training(pool, trained, tmp_path):
    root, config, stream_config, full, _ = trained
    result = train_pilot(pool, tmp_path / "seeded", config=config, stream_config=stream_config,
                         seed_replay=root / "full/checkpoint.pt", renderer_factory=FakeRendererFactory("crash"))
    assert result["model_state_sha256"] == full["model_state_sha256"]
    assert result["final_metrics"] == full["final_metrics"]
    assert result["totals"]["render_attempts"] == result["fresh_accepted_this_segment"] == 0
    assert result["seeded_feature_examples"] == result["totals"]["accepted"] == 3
    assert result["cursor"] == full["cursor"] and result["feature_seconds_this_segment"] == 0


def test_learning_seed_override_changes_learning_rngs_without_changing_replay(pool, trained, tmp_path):
    root, config, stream_config, _, _ = trained
    source = root / "full/checkpoint.pt"
    results = []
    for seed in (1, 2):
        result = train_pilot(pool, tmp_path / f"seed-{seed}", config=replace(config, learning_seed_override=seed),
                             stream_config=stream_config, seed_replay=source, renderer_factory=FakeRendererFactory())
        saved = torch.load(tmp_path / f"seed-{seed}/checkpoint.pt", weights_only=True)
        assert saved["run_contract"]["training"]["seed"] == config.seed
        assert saved["run_contract"]["training"]["learning_seed_override"] == seed
        results.append((result, saved))
    (first, first_saved), (second, second_saved) = results
    assert first["cursor"] == second["cursor"]
    assert [(item["recipe_id"], item["audio_sha256"]) for item in first_saved["replay"]] == [
        (item["recipe_id"], item["audio_sha256"]) for item in second_saved["replay"]]
    assert all(torch.equal(left["features"][name], right["features"][name])
               for left, right in zip(first_saved["replay"], second_saved["replay"], strict=True)
               for name in ("timbre", "pitch"))
    assert not torch.equal(first_saved["batch_rng"], second_saved["batch_rng"])
    assert first["fresh_accepted_this_segment"] == second["fresh_accepted_this_segment"] == 0
    assert first["totals"]["render_attempts"] == second["totals"]["render_attempts"] == 0
    assert first["model_state_sha256"] != second["model_state_sha256"]
    with pytest.raises(ValueError, match="contract changed"):
        train_pilot(pool, tmp_path / "changed-resume", config=replace(config, learning_seed_override=2),
                    stream_config=stream_config, resume=tmp_path / "seed-1/checkpoint.pt")


def test_learning_seed_override_preserves_legacy_contract_and_exact_recovery(pool, trained, tmp_path):
    root, config, stream_config, _, _ = trained
    source = root / "full/checkpoint.pt"
    legacy = torch.load(source, weights_only=True)
    assert "learning_seed_override" not in legacy["run_contract"]["training"]
    seeded = replace(config, learning_seed_override=7)

    def run(name, **kwargs):
        return train_pilot(pool, tmp_path / name, config=seeded, stream_config=stream_config,
                           seed_replay=source, renderer_factory=FakeRendererFactory(), **kwargs)

    def crash(step):
        if step == 2:
            raise InterruptedError("interrupt after the second optimizer update")

    full = run("full-learning-seed")
    with pytest.raises(InterruptedError):
        run("crash-learning-seed", after_step=crash)
    resumed = train_pilot(pool, tmp_path / "resumed-learning-seed", config=seeded, stream_config=stream_config,
                          resume=tmp_path / "crash-learning-seed/checkpoint.pt")
    assert full["model_state_sha256"] == resumed["model_state_sha256"]
    assert full["cursor"] == resumed["cursor"]


def test_learning_seed_override_cli_and_validation(pool, trained, tmp_path, monkeypatch):
    root, config, _, _, _ = trained
    with pytest.raises(ValueError, match="integer or None"):
        TrainConfig(learning_seed_override=True)
    pool.save(tmp_path / "pool.json")
    monkeypatch.setattr("obruxo_training.stream.NativeRendererFactory", lambda _=None: FakeRendererFactory())
    output = tmp_path / "cli-learning-seed"
    assert main(["train", "--pool", str(tmp_path / "pool.json"), "--output", str(output),
                 "--steps", str(config.steps), "--replay-capacity", str(config.replay_capacity),
                 "--batch-size", str(config.batch_size), "--render-budget", str(config.render_budget),
                 "--checkpoint-every", str(config.checkpoint_every), "--cpu-threads", "2", "--seed", "0",
                 "--timeout", "20", "--learning-seed-override", "9", "--seed-replay",
                 str(root / "full/checkpoint.pt")]) == 2  # The fixture's 8-step replay can fail the tiny-overfit gate.
    saved = torch.load(output / "checkpoint.pt", weights_only=True)
    assert saved["run_contract"]["training"]["learning_seed_override"] == 9
    assert saved["run_contract"]["training"]["seed"] == 0


def test_learning_seed_initialization_matches_across_populations(pool, tmp_path):
    config = TrainConfig(steps=2, replay_capacity=3, batch_size=2, checkpoint_every=2,
                         render_budget=8, cpu_threads=2, seed=0, learning_seed_override=11)
    stream = StreamConfig(prefetch_clips=1, timeout_seconds=20)
    for name, candidate_pool in (("one", pool), ("expanded", pool.expand(48))):
        train_pilot(candidate_pool, tmp_path / name, config=config, stream_config=stream,
                    renderer_factory=FakeRendererFactory(), stop_after=1)
    one_initial = torch.load(tmp_path / "one/checkpoint.previous.pt", weights_only=True)["model"]
    expanded_initial = torch.load(tmp_path / "expanded/checkpoint.previous.pt", weights_only=True)["model"]
    assert tensor_digest(one_initial) == tensor_digest(expanded_initial)


def test_seeded_online_recovery_preserves_prefix_and_new_render_cursor(pool, trained, tmp_path):
    root, _, stream_config, _, _ = trained
    config = TrainConfig(mode="online", steps=5, replay_capacity=5, batch_size=2, checkpoint_every=2,
                         render_budget=8, cpu_threads=2, fresh_examples=5)

    def run(name, **kwargs):
        return train_pilot(pool, tmp_path / name, config=config, stream_config=stream_config,
                           renderer_factory=FakeRendererFactory(), **kwargs)

    def crash(step):
        if step == 3:
            raise InterruptedError("interrupt seeded training after fresh ingestion")

    full = run("full", seed_replay=root / "full/checkpoint.pt")
    with pytest.raises(InterruptedError):
        run("crash", seed_replay=root / "full/checkpoint.pt", after_step=crash)
    resumed = run("resumed", resume=tmp_path / "crash/checkpoint.pt")
    assert full["model_state_sha256"] == resumed["model_state_sha256"]
    assert full["cursor"] == resumed["cursor"] and full["cursor"]["next_sequence_id"] == 5
    assert full["totals"]["render_attempts"] == resumed["totals"]["render_attempts"] == 2
    assert full["totals"]["accepted"] == 5 and full["seeded_feature_examples"] == 3
    assert resumed["fresh_accepted_this_segment"] == 0


def test_feature_seed_rejects_missing_prefix_and_changed_frontend(pool, trained, tmp_path):
    root, config, stream_config, _, _ = trained
    source = root / "full/checkpoint.pt"
    for change, error in (("prefix", "complete accepted recipe prefix"), ("frontend", "frozen frontend")):
        saved = torch.load(source, weights_only=True)
        if change == "prefix":
            saved["replay"].pop(0)
        else:
            next(iter(saved["basic_pitch"].values())).add_(1)
        path = tmp_path / f"{change}.pt"
        torch.save(saved, path)
        with pytest.raises(ValueError, match=error):
            train_pilot(pool, tmp_path / change, config=config, stream_config=stream_config, seed_replay=path)


def test_named_comparison_pins_external_retrieval_and_resumes_across_training_configs(pool, trained, tmp_path):
    root, config, stream_config, _, _ = trained
    source = root / "full/checkpoint.pt"
    candidate_config = replace(config, mode="online", steps=4, learning_rate=.002, replay_capacity=4, fresh_examples=4)
    train_pilot(pool, tmp_path / "candidate", config=candidate_config, stream_config=stream_config,
                seed_replay=source, renderer_factory=FakeRendererFactory())
    candidates = {"candidate": tmp_path / "candidate/checkpoint.pt", "previous": source}
    output = tmp_path / "comparison"
    result = evaluate_comparison(pool, candidates, source, output, count=1, renderer_factory=FakeRendererFactory())
    assert result["evaluation"] == "pilot_real_render_comparison_v2"
    assert result["retrieval_examples"] == 3  # Candidate retains four; the comparator must stay pinned to the source.
    assert result["renders_attempted"] == 12 and result["comparison"]["selected"] == "candidate"
    assert list(result["models"]) == ["candidate", "previous"]
    resumed = evaluate_comparison(pool, candidates, source, output, count=1, renderer_factory=FakeRendererFactory(), resume=True)
    assert resumed["completed_cases_reused"] == 1 and resumed["renders_attempted_this_segment"] == 0
    assert resumed["aggregates"] == result["aggregates"]
    with pytest.raises(ValueError, match="plan changed"):
        evaluate_comparison(pool, candidates, candidates["candidate"], output, count=1, resume=True)
    with pytest.raises(ValueError, match="plan changed"):
        evaluate_comparison(pool, dict(reversed(candidates.items())), source, output, count=1, resume=True)


def test_expanded_pool_cannot_resume_or_seed_parent_features(pool, trained, tmp_path):
    root, config, stream_config, _, _ = trained
    expanded = pool.expand(48)
    source = root / "full/checkpoint.pt"
    for option, message in (("resume", "contract changed"), ("seed_replay", "incompatible pool")):
        with pytest.raises(ValueError, match=message):
            train_pilot(expanded, tmp_path / option, config=config, stream_config=stream_config, **{option: source})
        assert not (tmp_path / option).exists()


def test_expanded_comparison_cli_keeps_parent_cases_retrieval_and_selected_training_identity(pool, trained, tmp_path, monkeypatch):
    root, config, stream_config, _, _ = trained
    expanded = pool.expand(48)
    source = root / "full/checkpoint.pt"
    old_states = {entry.state_id for entry in pool.entries}
    # Ensure this fixture actually trains on a newly added state.
    seed = next(seed for seed in range(20) if any(PilotSampler(expanded, seed=seed).recipe_at(i).state_id not in old_states
                                                for i in range(config.replay_capacity)))
    train_pilot(expanded, tmp_path / "candidate", config=replace(config, seed=seed), stream_config=stream_config,
                renderer_factory=FakeRendererFactory())
    checkpoint = tmp_path / "candidate/checkpoint.pt"
    pool.save(tmp_path / "parent.json")
    expanded.save(tmp_path / "expanded.json")
    monkeypatch.setattr("obruxo_training.stream.NativeRendererFactory", lambda _=None: FakeRendererFactory())
    output = tmp_path / "comparison"
    assert main(["compare", "--pool", str(tmp_path / "parent.json"), "--checkpoints", f"expanded={checkpoint}",
                 f"previous={source}", "--checkpoint-pools", f"expanded={tmp_path / 'expanded.json'}",
                 "--retrieval-checkpoint", str(source), "--output", str(output), "--count", "1"]) == 0
    result, plan = [json.loads((output / name).read_text()) for name in ("summary.json", "plan.json")]
    recipe = PilotSampler(pool, split="validation").recipe_at(0)
    assert plan["pool_id"] == result["pool_id"] == pool.pool_id
    assert plan["recipe_ids"] == [recipe.recipe_id]
    assert plan["cases"][0]["state_id"] == result["cases"][0]["state_id"] == recipe.state_id
    assert plan["cases"][0]["reference_performance_id"] == recipe.performance_id
    assert plan["cases"][0]["transfer_performance_id"] == result["cases"][0]["transfer_performance_id"]
    assert plan["candidate_order"] == result["candidate_order"] == ["expanded", "previous"]
    assert result["retrieval_sha256"] == retrieval_digest(torch.load(source, weights_only=True)["replay"])
    assert result["models"]["expanded"]["pool_expansion"] == expanded.expansion_from(pool)
    assert result["models"]["previous"]["pool_expansion"] is None
    assert result["retrieval_source"]["training_pool_id"] == pool.pool_id
    assert result["renders_attempted"] == result["renders_accepted"] == 12
    assert result["comparison"]["selected"] == "expanded"  # Synthetic render tie chooses the declared first candidate.
    exported = torch.load(output / "selected-model.pt", weights_only=True)
    assert exported["training_pool_id"] == expanded.pool_id
    candidate = torch.load(checkpoint, weights_only=True)
    assert any(item["state_id"] not in old_states for item in candidate["replay"])
    candidates = {"expanded": checkpoint, "previous": source}
    resumed = evaluate_comparison(pool, candidates, source, output, checkpoint_pools={"expanded": expanded},
                                  count=1, resume=True, renderer_factory=FakeRendererFactory("crash"))
    assert resumed["completed_cases_reused"] == 1 and resumed["renders_attempted_this_segment"] == 0
    assert resumed["aggregates"] == result["aggregates"] and not resumed["test_split_evaluated"]
    with pytest.raises(ValueError, match="different pool"):
        evaluate_comparison(pool, candidates, source, tmp_path / "missing-pool", count=1)
    with pytest.raises(ValueError, match="same seed"):
        evaluate_comparison(pool, candidates, source, tmp_path / "wrong-pool", checkpoint_pools={"expanded": replace(expanded, seed=1)})
    with pytest.raises(ValueError, match="identify candidate"):
        evaluate_comparison(pool, candidates, source, tmp_path / "typo", checkpoint_pools={"typo": expanded})
    assert not any((tmp_path / name).exists() for name in ("missing-pool", "wrong-pool", "typo"))


def test_named_comparison_rejects_unsafe_names_and_changed_renderer(pool, trained, tmp_path):
    root, _, _, _, _ = trained
    source = root / "full/checkpoint.pt"
    with pytest.raises(ValueError, match="safe, non-reserved"):
        evaluate_comparison(pool, {"../escape": source, "previous": source}, source, tmp_path / "bad-name")
    saved = torch.load(source, weights_only=True)
    saved["renderer_identity"]["backend_version"] = "changed"
    torch.save(saved, tmp_path / "changed.pt")
    with pytest.raises(ValueError, match="inference contracts and renderer"):
        evaluate_comparison(pool, {"changed": tmp_path / "changed.pt", "previous": source}, source, tmp_path / "bad-renderer")


def test_evaluator_counts_failures_against_planned_denominator(pool, trained, tmp_path):
    root, _, _, _, _ = trained
    result = evaluate_pilot(pool, root / "full/checkpoint.pt", tmp_path / "evaluation", count=1,
                            renderer_factory=FakeRendererFactory("silence"))
    assert result["renders_attempted"] == 1 and result["renders_accepted"] == 0
    assert result["aggregates"]["model"]["transfer"]["failed_or_unscorable"] == 1
    assert result["aggregates"]["train_retrieval"]["reference"]["scored"] == 0
    assert result["test_split_evaluated"] is False


def test_learning_curve_shares_sources_and_exports_selected_checkpoint(pool, trained, tmp_path):
    root, _, _, _, _ = trained
    result = evaluate_learning_curve(pool, [root / "crash/checkpoint.pt", root / "full/checkpoint.pt"],
                                     tmp_path / "curve", count=1, renderer_factory=FakeRendererFactory())
    assert result["renders_attempted"] == result["renders_accepted"] == 12
    assert all(result["aggregates"][name]["transfer"]["scored"] == 1 for name in result["aggregates"])
    assert result["learning_curve"]["selected"] == "step_4"  # Identical synthetic renders: earliest step breaks the tie.
    model, _, _ = load_model(tmp_path / "curve/selected-model.pt")
    assert tensor_digest(model.state_dict()) == result["models"]["step_4"]["model_state_sha256"]
    plan = json.loads((tmp_path / "curve/plan.json").read_text())
    assert plan["maximum_render_attempts"] == 12 and len(plan["recipe_ids"]) == 1
    assert plan["retrieval_sha256"] == result["retrieval_sha256"]
    assert result["test_split_evaluated"] is False


def test_learning_curve_rejects_changed_retrieval_and_duplicate_steps(pool, trained, tmp_path):
    root, _, _, _, _ = trained
    checkpoint = root / "full/checkpoint.pt"
    with pytest.raises(ValueError, match="increasing"):
        evaluate_learning_curve(pool, [checkpoint, checkpoint], tmp_path / "duplicate")
    saved = torch.load(checkpoint, weights_only=True)
    saved["replay"][0]["features"]["timbre"][0, 0, 0] += 1
    torch.save(saved, tmp_path / "changed.pt")
    with pytest.raises(ValueError, match="identical training retrieval"):
        evaluate_learning_curve(pool, [root / "crash/checkpoint.pt", tmp_path / "changed.pt"], tmp_path / "changed")
    assert not (tmp_path / "changed").exists()


@pytest.mark.parametrize("mode, expected_renders, selected", [("valid", 25, "step_4"), ("silence", 14, None)])
def test_curve_resume_verifies_committed_cases_and_retains_failures(pool, trained, tmp_path, mode, expected_renders, selected):
    root, _, _, _, _ = trained
    checkpoints = [root / "crash/checkpoint.pt", root / "full/checkpoint.pt"]
    output = tmp_path / "interrupted-curve"

    def interrupt(event):
        if event["case"] == 1:
            raise InterruptedError("interrupted after the second case's source render")

    with pytest.raises(InterruptedError):
        evaluate_learning_curve(pool, checkpoints, output, count=2, renderer_factory=FakeRendererFactory(mode), progress=interrupt)
    original = (output / "case-000.json").read_bytes()
    with pytest.raises(ValueError, match="plan changed"):
        evaluate_learning_curve(pool, checkpoints, output, count=1, resume=True)
    (output / "case-000.json").write_bytes(original + b" ")
    with pytest.raises(ValueError, match="case changed"):
        evaluate_learning_curve(pool, checkpoints, output, count=2, resume=True)
    (output / "case-000.json").write_bytes(original)
    result = evaluate_learning_curve(pool, checkpoints, output, count=2, resume=True, renderer_factory=FakeRendererFactory())
    assert (output / "case-000.json").read_bytes() == original
    assert result["completed_cases_reused"] == 1 and result["renders_attempted_this_segment"] == 12
    assert result["renders_attempted"] == expected_renders
    assert result["learning_curve"]["selected"] == selected
    assert result["elapsed_includes_uncommitted_interrupted_work"] is False
    assert result["aggregates"]["step_4"]["reference"]["failed_or_unscorable"] == (mode == "silence")


def test_curve_selection_requires_every_planned_case_and_bootstraps_paired_states():
    def score(value):
        return {role: {"mrstft": value} for role in ("reference", "transfer")}

    rows = [{"methods": {"step_1": score(0), "step_2": score(2), "train_retrieval": score(3)}},
            {"methods": {"step_1": {"reference": {"failure": "render_failed"}},
                         "step_2": score(4), "train_retrieval": score(5)}}]
    result = curve_statistics(rows, ["step_1", "step_2"])
    assert result["selected"] == "step_2"
    assert result["checkpoints"]["step_1"]["paired_roles_scored"] == 1
    comparison = result["checkpoints"]["step_2"]["against_train_retrieval"]
    assert comparison["paired"] == comparison["wins"] == 2
    assert comparison["mean_delta"] == -1 and comparison["ci95"] == [-1, -1]
    assert curve_statistics(rows, ["step_1"])["selected"] is None


def test_spectral_metric_detects_gain_and_stereo_changes():
    audio = np.stack((np.sin(np.arange(12000) * .04), np.sin(np.arange(12000) * .07)), 1).astype(np.float32) * .1
    assert audio_distance(audio, audio)["mrstft"] == 0
    assert audio_distance(audio, audio * .5)["mrstft"] > .1
    assert audio_distance(audio, audio[:, ::-1])["mrstft"] > .1


def test_feature_errors_and_failed_render_never_advance_a_saved_training_cursor(pool, tmp_path, monkeypatch):
    config = TrainConfig(steps=1, replay_capacity=1, batch_size=1, checkpoint_every=1, render_budget=1)

    def broken(*_):
        raise RuntimeError("feature extraction failed")

    monkeypatch.setattr(AudioFeatures, "forward", broken)
    with pytest.raises(RuntimeError, match="feature extraction failed"):
        train_pilot(pool, tmp_path / "broken", config=config, renderer_factory=FakeRendererFactory())
    saved = torch.load(tmp_path / "broken/checkpoint.pt", weights_only=True)
    assert saved["step"] == saved["cursor"]["next_sequence_id"] == 0 and saved["replay"] == []
    assert json.loads((tmp_path / "broken/status.json").read_text())["complete"] is False


def test_rejected_render_budget_stops_without_an_optimizer_update(pool, tmp_path):
    config = TrainConfig(steps=1, replay_capacity=1, batch_size=1, render_budget=1)
    with pytest.raises(RuntimeError, match="budget exhausted"):
        train_pilot(pool, tmp_path / "rejected", config=config, renderer_factory=FakeRendererFactory("silence"))
    saved = torch.load(tmp_path / "rejected/checkpoint.pt", weights_only=True)
    assert saved["step"] == 0 and saved["cursor"]["next_sequence_id"] == 1
    assert saved["replay"] == [] and saved["totals"]["rejected"] == 1


def test_offline_inference_cli_converts_pcm_and_refuses_overwrite(trained, tmp_path):
    from scipy.io import wavfile
    root, _, _, _, _ = trained
    audio = (3000 * np.sin(np.arange(12000) * .04)).astype(np.int16)
    wavfile.write(tmp_path / "input.wav", 44100, np.stack((audio, audio), 1))
    args = ["infer", "--model", str(root / "full/model.pt"), "--audio", str(tmp_path / "input.wav"),
            "--output", str(tmp_path / "result.vital")]
    assert main(args) == 0
    raw = (tmp_path / "result.vital").read_bytes()
    PilotCodec().encode(raw)
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 1 and (tmp_path / "result.vital").read_bytes() == raw

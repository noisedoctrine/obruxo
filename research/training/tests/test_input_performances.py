from collections import Counter, defaultdict
from dataclasses import replace
import json

import pytest

from obruxo_data.hashing import canonical_sha256
from obruxo_training.input_performances import PATTERNS, VariedSampler, performance_for
from obruxo_training.performance_eligibility import longest_held_seconds
from obruxo_training.performances import initial_bank
from obruxo_training.pilot import PilotPool
from obruxo_training.reproducibility import FULL_PHASE_CONTROL


def test_varied_input_pairs_are_complete_wide_informative_and_category_balanced():
    pool = PilotPool.create(96, seed=0)
    sampler = VariedSampler(pool)
    labels = {entry.state_id: entry.target.categorical for entry in pool.entries}
    groups = defaultdict(Counter)
    seen, pitches = Counter(), set()
    reserved = {entry.performance_id for entry in initial_bank()}
    for sequence in range(0, sampler.epoch_size, 2):
        first, second = sampler.recipe_at(sequence), sampler.recipe_at(sequence + 1)
        assert first.state_id == second.state_id and first.performance_id != second.performance_id
        assert first.input_pattern == "held" and second.input_pattern in PATTERNS[1:]
        assert first.contract_id == sampler.observation_contract["contract_id"] != pool.codec.contract.contract_id
        assert first.split == second.split == "train"
        for recipe in (first, second):
            request = recipe.render_request()
            assert recipe.performance_id == canonical_sha256(request.performance.to_dict())
            assert request.end_tick == request.performance.end_tick and request.preroll_frames == 32768
            assert recipe.performance_id not in reserved
            assert all(json.loads(request.preset_json)["settings"][key] == value for key, value in FULL_PHASE_CONTROL.items())
            pitches.update(event.data[0] for event in request.performance.canonical_events() if event.kind.value == "note_on")
        assert longest_held_seconds(first.render_request().performance) >= 2.
        seen[first.state_id] += 1
        groups[labels[second.state_id]][second.input_pattern] += 1
        with pytest.raises(ValueError, match="held-out"):
            replace(first, split="validation").render_request()
        with pytest.raises(ValueError, match="held-out"):
            replace(first, performance_role="transfer").render_request()
    assert len(seen) * 2 == sampler.epoch_size and set(seen.values()) == {1}
    assert min(pitches) >= 36 and max(pitches) <= 84 and len(pitches) > 24
    for counts in groups.values():
        values = [counts[pattern] for pattern in PATTERNS[1:]]
        assert max(values) - min(values) <= 1


def test_performance_generation_is_deterministic_and_not_tied_to_source_category():
    assert performance_for(0, "state", 0, 0, 0).to_dict() == performance_for(0, "state", 0, 0, 0).to_dict()
    assert performance_for(0, "state", 0, 0, 0).to_dict() != performance_for(0, "state", 1, 0, 0).to_dict()
    assert performance_for(0, "other", 0, 0, 0).to_dict() != performance_for(0, "state", 0, 0, 0).to_dict()


def test_varied_configuration_keeps_the_large_replay_allowance_basic_only():
    from obruxo_training.trainer import TrainConfig
    from obruxo_training.profiled_sampler import training_sampler
    config = TrainConfig(prediction_scope="basic", preroll_frames=32768, sampling_policy="paired_views", input_performances="varied",
                         replay_capacity=808, replay_mib=32, batch_size=32, render_budget=808)
    assert isinstance(training_sampler(PilotPool.create(12, seed=0), config), VariedSampler)
    assert 808 * (2 * 96 * 32 + 3521) * 4 < 32 * 1024**2
    with pytest.raises(ValueError, match="512 examples"):
        TrainConfig(replay_capacity=808, replay_mib=32, render_budget=808)
    with pytest.raises(ValueError, match="varied inputs require"):
        TrainConfig(input_performances="varied")


def test_varied_cli_passes_the_initial_model_and_explicit_observation_policy(tmp_path, monkeypatch):
    from obruxo_training.cli import main
    pool_path = tmp_path / "pool.json"
    PilotPool.create(24, seed=0).save(pool_path)
    initial_path, calls = tmp_path / "initial.pt", []

    def capture(_pool, _output, **kwargs):
        calls.append(kwargs)
        return {"complete": True, "tiny_overfit_gate": True}

    monkeypatch.setattr("obruxo_training.trainer.train_pilot", capture)
    assert main(["train", "--pool", str(pool_path), "--output", str(tmp_path / "unused"), "--prediction-scope", "basic",
                 "--sampling-policy", "paired_views", "--input-performances", "varied", "--initial-model", str(initial_path)]) == 0
    assert calls[0]["initial_model"] == initial_path
    assert calls[0]["config"].input_performances == "varied" and calls[0]["config"].preroll_frames == 32768

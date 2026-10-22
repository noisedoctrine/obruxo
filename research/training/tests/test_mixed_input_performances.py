from dataclasses import replace

import pytest
import torch

from obruxo_training.mixed_input_performances import MixedSampler
from obruxo_training.paired import complete_pairs, paired_batch_indices
from obruxo_training.pilot import PilotPool
from obruxo_training.trainer import TrainConfig


def test_mixed_pairs_keep_all_three_origins_and_half_the_draws_rehearse_original_inputs():
    sampler = MixedSampler(PilotPool.create(24, seed=0))
    recipes = [sampler.recipe_at(index) for index in range(sampler.epoch_size)]
    rows = [{"sequence_id": row.sequence_id, "state_id": row.state_id, "performance_id": row.performance_id} for row in recipes]
    assert len(complete_pairs(rows)) * 4 == sampler.epoch_size
    assert all(len(pairs) == 2 for pairs in complete_pairs(rows).values())
    for offset in range(0, len(recipes), 4):
        group = recipes[offset:offset + 4]
        assert len({recipe.state_id for recipe in group}) == 1
        assert group[0].origin_recipe_id == group[2].origin_recipe_id
        assert len({recipe.performance_id for recipe in group}) == 3
        for recipe in group:
            parent = sampler.original if recipe.origin == "original" else sampler.wide
            original = parent.recipe_at(recipe.origin_sequence_id)
            assert recipe.origin_recipe_id == original.recipe_id
            assert recipe.render_request().to_dict() == original.render_request().to_dict()
    generator = torch.Generator().manual_seed(0)
    selected_views = set()
    for _step in range(20):
        indices = paired_batch_indices(rows, 8, generator)
        assert sum(recipes[index].origin == "original" for index in indices) == 4
        selected_views.update(recipes[index].input_view for index in indices)
    assert selected_views == {0, 1, 2, 3}
    with pytest.raises(ValueError, match="held-out"):
        replace(recipes[0], split="validation").render_request()


def test_mixed_capacity_is_basic_only_and_requires_fixed_qualified_pairs():
    config = TrainConfig(prediction_scope="basic", preroll_frames=32768, sampling_policy="paired_views", input_performances="mixed",
                         replay_capacity=1616, replay_mib=64, batch_size=32, render_budget=1616)
    assert config.replay_capacity == 1616
    with pytest.raises(ValueError, match="512 examples"):
        TrainConfig(replay_capacity=1616, render_budget=1616)
    with pytest.raises(ValueError, match="mixed inputs require"):
        replace(config, mode="online")

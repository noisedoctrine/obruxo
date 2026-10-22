import pytest

from obruxo_training.evaluate import training_recipes
from obruxo_training.input_performances import VariedSampler
from obruxo_training.pilot import PilotPool


def test_explicit_training_cases_select_distinct_states_independently_of_retrieval_rows():
    sampler = VariedSampler(PilotPool.create(24, seed=0))
    replay = [{"sequence_id": index} for index in range(4)]
    assert len({recipe.state_id for recipe in training_recipes(sampler, replay, 2)}) == 1
    selected = training_recipes(sampler, replay, 2, [2, 0])
    assert [recipe.sequence_id for recipe in selected] == [2, 0]
    assert len({recipe.state_id for recipe in selected}) == 2
    for ids in ([0, 1], [0, 0], [True, 2], [-1, 2], [0, sampler.epoch_size], [0]):
        with pytest.raises(ValueError, match="explicit training"):
            training_recipes(sampler, replay, 2, ids)

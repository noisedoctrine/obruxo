from types import SimpleNamespace

import pytest

from obruxo_training.codec import PilotCodec, PilotTarget
from release_probe import nearest_pitch, replace_release, require_training_replay

torch = pytest.importorskip("torch")


def test_pitch_neighbor_ignores_timbre_and_breaks_ties_by_retained_order():
    features = {"pitch": torch.tensor([1., 2.]), "timbre": torch.tensor([100.])}
    replay = [{"features": {"pitch": torch.tensor(pitch), "timbre": torch.tensor([timbre])}}
              for pitch, timbre in [([5., 6.], 100.), ([1., 2.], -100.), ([1., 2.], 100.)]]
    assert nearest_pitch(features, replay) == (1, 0.0)


@pytest.mark.parametrize("split", ["validation", "test"])
def test_retrieval_rejects_held_out_candidates(split):
    manifest = SimpleNamespace(by_state={"a": SimpleNamespace(split="train"), "b": SimpleNamespace(split=split)})
    with pytest.raises(ValueError, match="training split"):
        require_training_replay([{"state_id": "a"}, {"state_id": "b"}], manifest)


def test_retrieval_rejects_duplicate_states():
    manifest = SimpleNamespace(by_state={"a": SimpleNamespace(split="train")})
    with pytest.raises(ValueError, match="one retained view"):
        require_training_replay([{"state_id": "a"}, {"state_id": "a"}], manifest)


@pytest.mark.parametrize("categories", [(0, 0), (1, 1), (2, 0)])
def test_release_intervention_preserves_all_other_controls(categories):
    codec = PilotCodec()
    before = codec.canonical_target(PilotTarget((0.4,) * 9, categories))
    after = replace_release(codec, before, 0.7)
    assert codec.encode(codec.decode(after)).continuous == pytest.approx(after.continuous)
    assert after.categorical == before.categorical
    assert after.continuous[:-1] == before.continuous[:-1]
    assert after.continuous[-1] == pytest.approx(0.7)

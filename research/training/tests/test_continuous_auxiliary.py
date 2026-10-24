import pytest
import torch

from obruxo_training.codec import PilotCodec, PilotTarget
from obruxo_training.continuous_auxiliary import SPEC, masked_continuous_loss, paired_continuous_loss


def test_source_inactive_controls_receive_no_label_gradient():
    codec = PilotCodec()
    target = PilotTarget((.4,) * 9, (0, 0))
    raw = torch.full((1, 9), .6, requires_grad=True)
    value = masked_continuous_loss(raw, [target], codec)
    value.backward()
    assert float(value.detach()) == pytest.approx(.04)
    assert raw.grad[0, [2, 3, 4]].tolist() == [0., 0., 0.]
    assert raw.grad[0, [0, 1, 5, 6, 7, 8]].tolist() == pytest.approx([.4 / 6] * 6)


def test_pair_weights_preserve_reference_sound_objective_and_do_not_supervise_categories():
    codec = PilotCodec()
    target = PilotTarget((.4,) * 9, (2, 1))
    predictions = [{"continuous": torch.full((1, 9), value, requires_grad=True),
                    "categorical": [torch.randn(1, 3, requires_grad=True), torch.randn(1, 2, requires_grad=True)]} for value in (.6, .7)]
    value = paired_continuous_loss(predictions, [target, target], codec)
    value.backward()
    assert float(value.detach()) == pytest.approx(.25 * .04 + .5 * .09)
    assert all(row["continuous"].grad is not None and all(scores.grad is None for scores in row["categorical"]) for row in predictions)
    assert SPEC["weight"] == 1 and SPEC["dense_outputs"] == 14 and not SPEC["label_inputs"]


@pytest.mark.parametrize("invalid", [float("nan"), -1., 2.])
def test_invalid_continuous_prediction_is_refused(invalid):
    with pytest.raises(ValueError, match="bounded finite"):
        masked_continuous_loss(torch.full((1, 9), invalid), [PilotCodec().default_target()], PilotCodec())


def test_unrelated_patches_cannot_be_combined_as_a_reciprocal_pair():
    codec = PilotCodec()
    target = codec.default_target()
    other = PilotTarget((.5,) * 9, (1, 1))
    with pytest.raises(ValueError, match="reciprocal"):
        paired_continuous_loss([{}, {}], [target, other], codec)

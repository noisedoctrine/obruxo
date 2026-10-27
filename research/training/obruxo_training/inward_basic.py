"""Keep legal basic outputs while allowing clipped proposals to learn inward."""

import torch

from .features import tensor_digest
from .trainable_basic import TrainableBasic, export_trainable
from .unison_choice import SCALES

POLICY = {"format": "inward_basic_output_gradient_v1", "forward": "unchanged_clamp_to_zero_one",
          "inside_and_boundary_backward": "identity", "below_zero_backward": "pass_negative_gradient_only",
          "above_one_backward": "pass_positive_gradient_only", "native_derivative": "unchanged_bounded_finite_difference",
          "production_format": "trainable_basic_inputs_v1", "additional_parameters": 0, "dense_outputs": 14}


class InwardClamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, proposal):
        ctx.save_for_backward(proposal)
        return proposal.clamp(0., 1.)

    @staticmethod
    def backward(ctx, gradient):
        proposal, = ctx.saved_tensors
        inside = (proposal >= 0.) & (proposal <= 1.)
        inward = ((proposal < 0.) & (gradient < 0.)) | ((proposal > 1.) & (gradient > 0.))
        return torch.where(inside | inward, gradient, 0.)


class InwardBasic(TrainableBasic):
    def forward(self, features):
        with torch.no_grad():
            baseline = self.base(features)
        delta = self.correction(features)
        return {**baseline, "continuous": InwardClamp.apply(baseline["continuous"] + delta[:, :9]),
                "categorical": [baseline["categorical"][0] / SCALES[0] + delta[:, 9:12],
                                baseline["categorical"][1] / SCALES[1] + delta[:, 12:14]]}


def training_model(model):
    if type(model) is not TrainableBasic:
        raise ValueError("inward training requires the existing trainable basic initializer")
    result = InwardBasic(model.base, seed=model.policy["seed"])
    result.load_state_dict(model.state_dict())
    result.require_frozen()
    if result.policy != model.policy or tensor_digest(result.state_dict()) != tensor_digest(model.state_dict()):
        raise ValueError("inward wrapping changed existing weights or frozen policy")
    return result.eval()


def export_inward(model, frontend, codec, pool_id):
    if type(model) is not InwardBasic:
        raise ValueError("inward export requires its declared training policy")
    model.require_frozen()
    inference = TrainableBasic(model.base, seed=model.policy["seed"])
    inference.load_state_dict(model.state_dict())
    inference.require_frozen()
    result = export_trainable(inference, frontend, codec, pool_id)
    result["training_gradient_policy"] = POLICY
    return result

"""Learn audio features and direct corrections while preserving the basic interface."""

import copy

import torch
from torch import nn

from .basic_direct import SPEC as DIRECT_SPEC, BasicDirect
from .codec import PilotCodec
from .features import tensor_digest
from .inference_profile import artifact_rendering, bind_rendering, model_rendering
from .trainer import inference_state, restore_model
from .unison_choice import SCALES

FORMAT = "trainable_basic_inputs_v1"
COUNTS = {"total_parameters": 613788, "frozen_parameters": 355806, "trainable_parameters": 257982, "dense_outputs": 14}
SPEC = {**DIRECT_SPEC, "format": FORMAT, "input_branch": "copied_trainable_stereo_cnn_and_pitch_projection",
        "initialization": "copied_accepted_features_zero_final_layer_v1", "additional_output_heads": 0}


class InputProjection(nn.Module):
    def __init__(self, direct):
        super().__init__()
        self.timbre = copy.deepcopy(direct.base.timbre).requires_grad_(True)
        self.pitch = copy.deepcopy(direct.base.pitch).requires_grad_(True)
        self.hidden = copy.deepcopy(direct.correction[0])

    def forward(self, features):
        return self.hidden(torch.cat((self.timbre(features["timbre"]), self.pitch(features["pitch"])), dim=1))


class TrainableBasic(nn.Module):
    def __init__(self, base, *, seed=0):
        super().__init__()
        direct = BasicDirect(base, seed=seed)
        self.base, self.policy = direct.base, copy.deepcopy(direct.policy)
        self.correction = nn.Sequential(InputProjection(direct), direct.correction[1], direct.correction[2])
        bind_rendering(self, model_rendering(base))
        self.require_frozen()

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def require_frozen(self):
        actual = {"total_parameters": sum(value.numel() for value in self.parameters()),
                  "frozen_parameters": sum(value.numel() for value in self.base.parameters()),
                  "trainable_parameters": sum(value.numel() for value in self.correction.parameters()),
                  "dense_outputs": self.correction[2].out_features}
        if (actual != COUNTS or tensor_digest(self.base.state_dict()) != self.policy["base_sha256"]
                or any(value.requires_grad or value.grad is not None for value in self.base.parameters())
                or any(not value.requires_grad for value in self.correction.parameters())):
            raise ValueError("trainable basic inputs changed their frozen baseline, mask or output budget")
        return actual

    def forward(self, features):
        with torch.no_grad():
            baseline = self.base(features)
        delta = self.correction(features)
        return {**baseline, "continuous": (baseline["continuous"] + delta[:, :9]).clamp(0., 1.),
                "categorical": [baseline["categorical"][0] / SCALES[0] + delta[:, 9:12],
                                baseline["categorical"][1] / SCALES[1] + delta[:, 12:14]]}


def input_gradients(model):
    model.require_frozen()
    result = {}
    for name in ("timbre", "pitch", "hidden"):
        parameters = tuple(getattr(model.correction[0], name).parameters())
        if any(value.grad is None or not torch.isfinite(value.grad).all() for value in parameters):
            raise ValueError("missing or nonfinite trainable input gradient")
        result[name] = sum(float(value.grad.double().abs().sum()) for value in parameters)
    return result


def export_trainable(model, frontend, codec, pool_id):
    if type(model) is not TrainableBasic or type(codec) is not PilotCodec:
        raise ValueError("trainable input export requires the basic model and codec")
    model.require_frozen()
    return {"format": FORMAT, "spec": copy.deepcopy(SPEC), "policy": copy.deepcopy(model.policy),
            "base": inference_state(model.base, frontend, codec, pool_id), "correction": model.correction.state_dict(),
            "model_sha256": tensor_digest(model.state_dict()), **model_rendering(model), "qualified_for_release": False}


def restore_trainable(saved):
    if saved["format"] != FORMAT or saved["spec"] != SPEC or saved["qualified_for_release"]:
        raise ValueError("trainable input export contract changed")
    base, frontend, codec = restore_model(saved["base"])
    if type(codec) is not PilotCodec or model_rendering(base) != artifact_rendering(saved):
        raise ValueError("trainable input codec or rendering profile changed")
    model = TrainableBasic(base, seed=saved["policy"]["seed"])
    model.correction.load_state_dict(saved["correction"])
    if model.policy != saved["policy"] or tensor_digest(model.state_dict()) != saved["model_sha256"]:
        raise ValueError("trainable input weights or policy changed")
    model.require_frozen()
    return model.eval(), frontend, codec

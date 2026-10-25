"""A direct unison-score correction with the earlier predictor entirely frozen."""

import copy

import torch
from torch import nn

from .codec import PilotCodec
from .features import tensor_digest
from .inference_profile import artifact_rendering, bind_rendering, model_rendering
from .trainer import inference_state, restore_model

FORMAT = "basic_independent_unison_choice_v1"
SCALES = (32.73040780087214, 29.015704035654633)
COUNTS = {"total_parameters": 382561, "frozen_parameters": 355806, "trainable_parameters": 26755, "dense_outputs": 14}
SPEC = {"format": FORMAT, "input_width": 832, "hidden_width": 32, "outputs": 3, "activation": "GELU",
        "base_unison_scale": SCALES[0], "adapter_scales": [1., SCALES[1]], "initialization": "zero_final_layer_v1"}


class UnisonChoice(nn.Module):
    def __init__(self, base, *, seed=0):
        super().__init__()
        if base.conditioning != "concat" or base.cyclic_head != "sigmoid" or type(seed) is not int:
            raise ValueError("unison correction requires the original basic predictor and an integer seed")
        self.base = copy.deepcopy(base).eval().requires_grad_(False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.correction = nn.Sequential(nn.Linear(832, 32), nn.GELU(), nn.Linear(32, 3))
        nn.init.zeros_(self.correction[2].weight)
        nn.init.zeros_(self.correction[2].bias)
        self.policy = {"base_sha256": tensor_digest(self.base.state_dict()), "seed": seed}
        bind_rendering(self, model_rendering(base))
        self.require_frozen()

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def require_frozen(self):
        frozen = list(self.base.parameters())
        actual = {"frozen_parameters": sum(value.numel() for value in frozen),
                  "trainable_parameters": sum(value.numel() for value in self.correction.parameters())}
        actual.update(total_parameters=sum(value.numel() for value in self.parameters()), dense_outputs=self.base.continuous.out_features
                      + sum(head.out_features for head in self.base.categorical))
        if (actual != COUNTS or any(value.requires_grad or value.grad is not None for value in frozen)
                or any(not value.requires_grad for value in self.correction.parameters())
                or tensor_digest(self.base.state_dict()) != self.policy["base_sha256"]):
            raise ValueError("unison correction changed the frozen predictor, mask or output budget")
        return actual

    def forward(self, features):
        with torch.no_grad():
            prediction = self.base(features)
            controls = torch.cat((self.base.timbre(features["timbre"]), self.base.pitch(features["pitch"])), dim=1)
        scores = prediction["categorical"][0] / SCALES[0] + self.correction(controls.detach())
        return {**prediction, "categorical": [scores, prediction["categorical"][1]]}


def export_unison(model, frontend, codec, pool_id):
    if type(codec) is not PilotCodec:
        raise ValueError("unison correction requires the basic codec")
    model.require_frozen()
    return {"format": FORMAT, "spec": copy.deepcopy(SPEC), "policy": copy.deepcopy(model.policy),
            "base": inference_state(model.base, frontend, codec, pool_id), "correction": model.correction.state_dict(),
            "model_sha256": tensor_digest(model.state_dict()), **model_rendering(model), "qualified_for_release": False}


def restore_unison(saved):
    if saved["format"] != FORMAT or saved["spec"] != SPEC or saved["qualified_for_release"]:
        raise ValueError("unison correction export contract changed")
    base, frontend, codec = restore_model(saved["base"])
    if type(codec) is not PilotCodec or model_rendering(base) != artifact_rendering(saved):
        raise ValueError("unison correction export changed its codec or rendering profile")
    model = UnisonChoice(base, seed=saved["policy"]["seed"])
    model.correction.load_state_dict(saved["correction"])
    if model.policy != saved["policy"] or tensor_digest(model.state_dict()) != saved["model_sha256"]:
        raise ValueError("unison correction export weights or policy changed")
    model.require_frozen()
    return model.eval(), frontend, codec


def measured_choice_loss(prediction, costs, *, temperature=1.):
    scores = prediction["categorical"][0]
    values = torch.as_tensor(costs, dtype=scores.dtype, device=scores.device)
    if (scores.ndim != 2 or scores.shape != values.shape or scores.shape[-1] != 3 or not torch.isfinite(scores).all()
            or not torch.isfinite(values).all() or (values < 0).any()
            or type(temperature) not in (int, float) or not 0 < temperature <= 1):
        raise ValueError("expected finite nonnegative measured three-choice costs and a valid temperature")
    return ((scores / temperature).softmax(-1) * values).sum(-1).mean()

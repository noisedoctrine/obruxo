"""Direct corrections to the existing nine controls and five category scores."""

import copy
import math

import torch
from torch import nn

from .codec import PilotCodec
from .features import tensor_digest
from .inference_profile import artifact_rendering, bind_rendering, model_rendering
from .trainer import inference_state, restore_model
from .unison_choice import SCALES

FORMAT = "basic_direct_correction_v1"
COUNTS = {"total_parameters": 382924, "frozen_parameters": 355806, "trainable_parameters": 27118, "dense_outputs": 14}
SPEC = {"format": FORMAT, "input_width": 832, "hidden_width": 32, "outputs": 14, "activation": "GELU",
        "continuous_transform": "clamp_base_normalized_plus_offset_to_zero_one", "base_category_scales": list(SCALES),
        "adapter_scales": [1., 1.], "initialization": "zero_final_layer_v1"}
REHEARSAL = {"reference_records": 32, "paired_weights": [.25, .5], "reference_weights": [.125, .125],
             "reference_excess_multiplier": 1., "reference_selection": "two_original_records_in_round_robin_per_update",
             "reference_loss": "current_sound_loss_plus_relu_current_minus_accepted_initial_sound_loss"}


class BasicDirect(nn.Module):
    def __init__(self, base, *, seed=0):
        super().__init__()
        if base.conditioning != "concat" or base.cyclic_head != "sigmoid" or type(seed) is not int:
            raise ValueError("direct correction requires the original basic predictor and an integer seed")
        self.base = copy.deepcopy(base).eval().requires_grad_(False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.correction = nn.Sequential(nn.Linear(832, 32), nn.GELU(), nn.Linear(32, 14))
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
        actual = {"total_parameters": sum(value.numel() for value in self.parameters()),
                  "frozen_parameters": sum(value.numel() for value in self.base.parameters()),
                  "trainable_parameters": sum(value.numel() for value in self.correction.parameters()),
                  "dense_outputs": self.correction[2].out_features}
        if (actual != COUNTS or tensor_digest(self.base.state_dict()) != self.policy["base_sha256"]
                or any(value.requires_grad or value.grad is not None for value in self.base.parameters())
                or any(not value.requires_grad for value in self.correction.parameters())):
            raise ValueError("direct correction changed its frozen predictor, mask or output budget")
        return actual

    def forward(self, features):
        with torch.no_grad():
            baseline = self.base(features)
            controls = torch.cat((self.base.timbre(features["timbre"]), self.base.pitch(features["pitch"])), dim=1)
        delta = self.correction(controls.detach())
        continuous = (baseline["continuous"] + delta[:, :9]).clamp(0., 1.)
        logits = [baseline["categorical"][0] / SCALES[0] + delta[:, 9:12], baseline["categorical"][1] / SCALES[1] + delta[:, 12:14]]
        return {**baseline, "continuous": continuous, "categorical": logits}


def reference_loss(current, baseline):
    if (current.ndim != 0 or not torch.isfinite(current) or current < 0 or type(baseline) not in (int, float)
            or not math.isfinite(baseline) or baseline < 0):
        raise ValueError("reference rehearsal requires a finite scalar sound loss and nonnegative baseline")
    return current + (current - baseline).relu()


def rehearsal_indices(step, records=32):
    if type(step) is not int or step < 0 or records != REHEARSAL["reference_records"]:
        raise ValueError("reference rehearsal needs a nonnegative update and the 32 original records")
    return (2 * step % records, (2 * step + 1) % records)


def export_direct(model, frontend, codec, pool_id):
    if type(model) is not BasicDirect:
        from .trainable_basic import export_trainable
        return export_trainable(model, frontend, codec, pool_id)
    if type(codec) is not PilotCodec:
        raise ValueError("direct correction requires the basic codec")
    model.require_frozen()
    return {"format": FORMAT, "spec": copy.deepcopy(SPEC), "policy": copy.deepcopy(model.policy),
            "base": inference_state(model.base, frontend, codec, pool_id), "correction": model.correction.state_dict(),
            "model_sha256": tensor_digest(model.state_dict()), **model_rendering(model), "qualified_for_release": False}


def restore_direct(saved):
    if saved["format"] == "trainable_basic_inputs_v1":
        from .trainable_basic import restore_trainable
        return restore_trainable(saved)
    if saved["format"] != FORMAT or saved["spec"] != SPEC or saved["qualified_for_release"]:
        raise ValueError("direct correction export contract changed")
    base, frontend, codec = restore_model(saved["base"])
    if type(codec) is not PilotCodec or model_rendering(base) != artifact_rendering(saved):
        raise ValueError("direct correction codec or rendering profile changed")
    model = BasicDirect(base, seed=saved["policy"]["seed"])
    model.correction.load_state_dict(saved["correction"])
    if model.policy != saved["policy"] or tensor_digest(model.state_dict()) != saved["model_sha256"]:
        raise ValueError("direct correction weights or policy changed")
    model.require_frozen()
    return model.eval(), frontend, codec

"""Fixed train-only score scaling and an observation-only historical gradient check."""

from contextlib import contextmanager
import copy
import math

import torch

from .model import batch_features, predicted_targets
from .spectral_controls import temperature_at


def fit_scales(logits):
    if (len(logits) != 2 or any(scores.ndim != 2 or scores.shape[1] != size or not len(scores)
                               or not torch.isfinite(scores).all() for scores, size in zip(logits, (3, 2), strict=True))
            or len(logits[0]) != len(logits[1])):
        raise ValueError("calibration requires finite training rows for both basic category heads")
    rows = [scores.detach().double() for scores in logits]
    return tuple(max(1., float((scores - scores.mean(-1, keepdim=True)).square().mean().sqrt())) for scores in rows)


def scaled_logits(logits, scales):
    if len(logits) != 2 or len(scales) != 2 or any(not math.isfinite(scale) or scale < 1. for scale in scales):
        raise ValueError("expected two finite fixed scales at least one")
    return [scores / scale for scores, scale in zip(logits, scales, strict=True)]


def conditional_score_gradient(scores, noise, costs, *, scale, temperature):
    raw = scores.detach().clone().requires_grad_(True)
    soft = ((raw / scale + noise) / temperature).softmax(-1)
    expected = (soft * raw.new_tensor([costs])).sum()
    return torch.autograd.grad(expected, raw)[0]


class SignalObserver:
    def __init__(self, codec, *, check_limits):
        self.codec, self.check_limits = codec, check_limits
        self.scales, self.points, self.initial = None, [], []

    def calibrate(self, model, records):
        frozen = copy.deepcopy(model).eval()
        with torch.no_grad(), torch.random.fork_rng(devices=[]):
            predictions = [frozen(batch_features([{"features": records[index]}])) for index in sorted(records)]
            self.scales = fit_scales([torch.cat([row["categorical"][head] for row in predictions]) for head in (0, 1)])
            for row in predictions:
                calibrated = {**row, "categorical": scaled_logits(row["categorical"], self.scales)}
                before, after = predicted_targets(row, self.codec)[0], predicted_targets(calibrated, self.codec)[0]
                if before != after or self.codec.decode(before) != self.codec.decode(after):
                    raise ValueError("score scaling changed an initial hard patch")
                self.initial.append({"raw_logits": [scores[0].tolist() for scores in row["categorical"]],
                                     "hard_patch_unchanged": True})

    @contextmanager
    def __call__(self, step, models, records, committed):
        self.check_limits()
        if self.scales is None:
            self.calibrate(models["spectral"], records)
        observations, expected_rng = [], None

        def observe(_module, _inputs, prediction):
            nonlocal expected_rng
            actual_rng = torch.get_rng_state().clone()
            generator = torch.Generator().set_state(actual_rng)
            estimate = committed["estimates"][0]
            if prediction["continuous"].detach()[0].tolist() != estimate["anchor"]["continuous"]:
                raise ValueError("observed controls differ from the recorded native point")
            for head, scores in enumerate(prediction["categorical"]):
                raw = scores.detach()
                noise = -torch.empty_like(raw, memory_format=torch.legacy_contiguous_format).exponential_(generator=generator).log()
                if int((raw + noise).argmax(-1).item()) != estimate["anchor"]["categorical"][head]:
                    raise ValueError("local Gumbel draw differs from the actual recorded hard choice")
                costs = estimate["categorical"][head]
                gradients = [conditional_score_gradient(raw, noise, costs, scale=scale, temperature=temperature_at(step, 128))
                             for scale in (1., self.scales[head])]
                original, calibrated = [float(value.abs().sum()) for value in gradients]
                recorded = committed["gradients"]["category_l1"][head]
                if not math.isclose(original, recorded, rel_tol=1e-5, abs_tol=1e-30):
                    raise ValueError("baseline score gradient differs from the actual training trace")
                selected = estimate["anchor"]["categorical"][head]
                observations.append({"head": head, "raw_logits": raw[0].tolist(), "noise": noise[0].tolist(),
                    "conditional_losses": costs, "selected": selected, "original_gradient_l1": recorded,
                    "calibrated_gradient_l1": calibrated,
                    "better_alternative": min(costs) < costs[selected] - 1e-7,
                    "raw_top_two_margin": float(raw.topk(2).values.diff(dim=-1).abs().item())})
            expected_rng = generator.get_state()
            if not torch.equal(actual_rng, torch.get_rng_state()):
                raise ValueError("observation changed the actual sampler RNG")

        with models["spectral"].register_forward_hook(observe):
            yield
        if len(observations) != 2 or not torch.equal(expected_rng, torch.get_rng_state()):
            raise ValueError("observation did not reproduce exactly two native Gumbel draws")
        self.points.append({"step": step + 1, "heads": observations})
        self.check_limits()

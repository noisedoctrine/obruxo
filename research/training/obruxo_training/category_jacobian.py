"""Check the category sampler's full derivative matrix without new native queries."""

import math

import torch

from .category_signal import scaled_logits
from .spectral_controls import BasicControlAdapter

JACOBIAN_GATES = {"matrix_norm_floor": 1e-8, "fraction_above_floor": .99, "median_norm_floor": 1e-5,
                  "weak_raw_median_threshold": 1e-5, "weak_median_gain": 10., "healthy_median_gain": .5}


def softmax_jacobian(scores, noise, scale, temperature):
    probability = ((scores / scale + noise) / temperature).softmax(-1)[0]
    return (probability.diag() - probability[:, None] * probability[None, :]) / (scale * temperature)


def check_jacobians(continuous, logits, scales, *, draws=16, seed=0):
    """Exercise actual hard Gumbel choices; the matrices cover every linear cost table.

    This qualifies derivative connectivity only. It does not measure native category
    sensitivity, choose a model, or establish perceptual improvement.
    """
    if type(draws) is not int or not 1 <= draws <= 16 or type(seed) is not int:
        raise ValueError("category derivative check needs one through sixteen draws and an integer seed")
    if continuous.device.type != "cpu":
        raise ValueError("category derivative qualification uses CPU float32")
    adapter = BasicControlAdapter()
    adapter(continuous, logits, sample=False)
    adapter(continuous, scaled_logits(logits, scales), sample=False)
    if any(scores.requires_grad for scores in logits):
        raise ValueError("provide detached frozen model scores")
    points = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for row in range(len(continuous)):
            for temperature in (1., .5):
                for draw in range(draws):
                    raw = [scores[row:row + 1].clone().requires_grad_(True) for scores in logits]
                    before = torch.get_rng_state().clone()
                    choice = adapter(continuous[row:row + 1].detach(), scaled_logits(raw, scales), temperature=temperature, sample=True)
                    generator = torch.Generator().set_state(before)
                    noises = [-torch.empty_like(scores).exponential_(generator=generator).log() for scores in raw]
                    if not torch.equal(generator.get_state(), torch.get_rng_state()):
                        raise ValueError("installed Gumbel sampler consumed unexpected randomness")
                    heads = []
                    for head, (scores, noise, scale, offset) in enumerate(zip(raw, noises, scales, (9, 12), strict=True)):
                        selected = choice[:, offset:offset + scores.shape[1]]
                        expected_choice = ((scores.detach() / scale + noise) / temperature).argmax(-1)
                        if not torch.equal(selected.detach().argmax(-1), expected_choice):
                            raise ValueError("actual hard choice differs from the reconstructed Gumbel draw")
                        actual = torch.stack([torch.autograd.grad(selected[0, index], scores, retain_graph=True)[0][0]
                                              for index in range(scores.shape[1])])
                        expected = softmax_jacobian(scores.detach(), noise, scale, temperature)
                        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-8)
                        original = softmax_jacobian(scores.detach(), noise, 1., temperature)
                        if not torch.isfinite(actual).all() or not torch.isfinite(original).all():
                            raise ValueError("nonfinite category derivative")
                        heads.append({"head": head, "noise": noise[0].tolist(), "hard_choice": int(expected_choice.item()),
                                      "jacobian": actual.tolist(), "original_norm": float(original.double().norm()),
                                      "scaled_norm": float(actual.double().norm()), "maximum_formula_error": float((actual - expected).abs().max())})
                    points.append({"row": row, "temperature": temperature, "draw": draw, "heads": heads})
    heads = []
    for head in (0, 1):
        temperatures = []
        for temperature in (1., .5):
            selected = [point["heads"][head] for point in points if point["temperature"] == temperature]
            original = torch.tensor([point["original_norm"] for point in selected], dtype=torch.float64)
            scaled = torch.tensor([point["scaled_norm"] for point in selected], dtype=torch.float64)
            original_median, scaled_median = float(original.quantile(.5)), float(scaled.quantile(.5))
            gain = scaled_median / original_median if original_median else None
            connected_fraction = float((scaled >= JACOBIAN_GATES["matrix_norm_floor"]).double().mean())
            required_gain = JACOBIAN_GATES["weak_median_gain"] if original_median < JACOBIAN_GATES["weak_raw_median_threshold"] else JACOBIAN_GATES["healthy_median_gain"]
            passed = (bool((scaled > 0).all()) and connected_fraction >= JACOBIAN_GATES["fraction_above_floor"]
                      and scaled_median >= JACOBIAN_GATES["median_norm_floor"] and scaled_median >= original_median * required_gain)
            temperatures.append({"temperature": temperature, "points": len(selected), "original_median_norm": original_median,
                                 "scaled_median_norm": scaled_median, "median_gain": gain, "fraction_above_connectivity_floor": connected_fraction,
                                 "exact_zero_points": int((scaled == 0).sum()), "pass": passed})
        heads.append({"head": head, "temperatures": temperatures, "pass": all(item["pass"] for item in temperatures)})
    return {"points": points, "heads": heads, "passed": all(head["pass"] for head in heads),
            "all_derivatives_finite": all(math.isfinite(item["scaled_norm"]) for point in points for item in point["heads"]),
            "gates": JACOBIAN_GATES.copy(), "native_requests": 0, "optimizer_updates": 0}

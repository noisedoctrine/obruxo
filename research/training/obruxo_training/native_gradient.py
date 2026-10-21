"""Sound-loss gradients estimated from legal native renders, not a learned forward model."""

from dataclasses import dataclass
import math

import torch

from .codec import CATEGORICAL, PilotCodec, PilotTarget


def native_gradient_spec():
    return {"format": "basic_native_loss_gradient_v1", "continuous_step": .025,
            "continuous": "bounded_two_point_difference_divided_by_actual_span_in_normalized_units",
            "inactive_continuous": "zero", "categories": "conditional_native_loss_for_each_option",
            "backward": "local_linear_estimate_on_raw_controls_and_hard_straight_through_choices",
            "forward": "measured_loss_of_selected_canonical_patch", "maximum_native_queries": 22,
            "exact_plugin_derivative": False, "category_interactions": "not_estimated", "output_size": 14}


@dataclass(frozen=True)
class NativeGradient:
    anchor: PilotTarget
    value: float
    continuous: tuple
    categorical: tuple
    queries: int

    def to_dict(self):
        return {"anchor": self.anchor.to_dict(), "value": self.value, "continuous": list(self.continuous),
                "categorical": [list(row) for row in self.categorical], "queries": self.queries}


def measure_native_gradient(codec, anchor, evaluate):
    """Evaluate one example serially. `evaluate` supplies the same sound loss for each legal patch."""
    if type(codec) is not PilotCodec:
        raise ValueError("native gradient requires the exact basic codec")
    codec.validate_target(anchor)
    step, cache = native_gradient_spec()["continuous_step"], {}

    def score(candidate):
        candidate = codec.canonical_target(candidate)
        if candidate not in cache:
            value = float(evaluate(candidate))
            if not math.isfinite(value) or value < 0:
                raise ValueError("native sound loss must be finite and nonnegative")
            cache[candidate] = value
        return cache[candidate]

    base = score(anchor)
    gradient = []
    for index, active in enumerate(codec.continuous_mask(anchor)):
        if not active:
            gradient.append(0.)
            continue
        lower, upper = list(anchor.continuous), list(anchor.continuous)
        lower[index], upper[index] = max(0., lower[index] - step), min(1., upper[index] + step)
        low = score(PilotTarget(tuple(lower), anchor.categorical))
        high = score(PilotTarget(tuple(upper), anchor.categorical))
        gradient.append((high - low) / (upper[index] - lower[index]))
    categorical = []
    for head, (_, options) in enumerate(CATEGORICAL):
        losses = []
        for option in range(len(options)):
            choices = list(anchor.categorical)
            choices[head] = option
            losses.append(score(PilotTarget(anchor.continuous, tuple(choices))))
        categorical.append(tuple(losses))
    return NativeGradient(anchor, base, tuple(gradient), tuple(categorical), len(cache))


def measured_loss(continuous, choices, estimate):
    """Exact measured forward value, estimated backward; one example, explicit render boundary.

    Pass raw controls, not adapter-canonicalized values: the category loss tables already
    include inactive-default jumps. Applying that blend's derivative again double-counts them.
    Choices may come from the existing adapter's hard Gumbel-softmax output.
    """
    if (continuous.shape != (1, 9) or continuous.dtype != torch.float32
            or not torch.isfinite(continuous).all() or len(choices) != 2):
        raise ValueError("expected one finite float32 basic control row and two choice heads")
    anchor = continuous.new_tensor([estimate.anchor.continuous])
    if not torch.equal(continuous.detach(), anchor):
        raise ValueError("gradient estimate is stale for these raw controls")
    coefficients = (*estimate.continuous, *(value for row in estimate.categorical for value in row), estimate.value)
    if (len(estimate.continuous) != 9 or tuple(map(len, estimate.categorical)) != (3, 2)
            or not all(math.isfinite(value) for value in coefficients)):
        raise ValueError("invalid native gradient coefficients")
    loss = continuous.new_tensor(estimate.value)
    loss = loss + ((continuous - continuous.detach()) * continuous.new_tensor([estimate.continuous])).sum()
    for head, (choice, size) in enumerate(zip(choices, (3, 2), strict=True)):
        expected = torch.nn.functional.one_hot(torch.tensor([estimate.anchor.categorical[head]], device=continuous.device), size)
        if (choice.shape != (1, size) or choice.dtype != continuous.dtype or choice.device != continuous.device
                or not torch.equal(choice.detach(), expected.to(continuous.dtype))):
            raise ValueError("choices must match the hard patch that was measured")
        loss = loss + ((choice - choice.detach()) * choice.new_tensor([estimate.categorical[head]])).sum()
    return loss


def guided_candidates(codec, estimate):
    """One frozen local step for qualification; no source parameters or best-of-search."""
    continuous = torch.tensor([estimate.anchor.continuous], dtype=torch.float32, requires_grad=True)
    logits, choices = [], []
    for selected, size in zip(estimate.anchor.categorical, (3, 2), strict=True):
        hard = torch.nn.functional.one_hot(torch.tensor([selected]), size).float()
        scores = (.1 * hard).requires_grad_()
        soft = scores.softmax(-1)
        logits.append(scores)
        choices.append(hard - soft.detach() + soft)
    loss = measured_loss(continuous, choices, estimate)
    loss.backward()
    with torch.no_grad():
        gradient = continuous.grad
        updated = (continuous - .05 * gradient / gradient.abs().max().clamp_min(1e-12)).clamp(0, 1)
        categories = tuple(int((scores - scores.grad / scores.grad.abs().max().clamp_min(1e-12)).argmax(-1).item())
                           for scores in logits)
    values = tuple(updated[0].tolist())
    return {"continuous": codec.canonical_target(PilotTarget(values, estimate.anchor.categorical)),
            "joint": codec.canonical_target(PilotTarget(values, categories))}, {
                "continuous_gradient_l1": float(continuous.grad.abs().sum()),
                "category_gradient_l1": [float(scores.grad.abs().sum()) for scores in logits]}

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


def native_gradient_queries(codec, anchor):
    """Plan every independent query in the original order, deduplicating canonical patches."""
    if type(codec) is not PilotCodec:
        raise ValueError("native gradient requires the exact basic codec")
    codec.validate_target(anchor)
    step, queries, positions = native_gradient_spec()["continuous_step"], [], {}

    def add(candidate):
        candidate = codec.canonical_target(candidate)
        if candidate not in positions:
            positions[candidate] = len(queries)
            queries.append(candidate)
        return positions[candidate]

    add(anchor)
    spans = []
    for index, active in enumerate(codec.continuous_mask(anchor)):
        if not active:
            spans.append(None)
            continue
        lower, upper = list(anchor.continuous), list(anchor.continuous)
        lower[index], upper[index] = max(0., lower[index] - step), min(1., upper[index] + step)
        low = add(PilotTarget(tuple(lower), anchor.categorical))
        high = add(PilotTarget(tuple(upper), anchor.categorical))
        spans.append((low, high, upper[index] - lower[index]))
    categorical = []
    for head, (_, options) in enumerate(CATEGORICAL):
        losses = []
        for option in range(len(options)):
            choices = list(anchor.categorical)
            choices[head] = option
            losses.append(add(PilotTarget(anchor.continuous, tuple(choices))))
        categorical.append(tuple(losses))
    return tuple(queries), tuple(spans), tuple(categorical)


def gradient_from_values(anchor, plan, values):
    queries, spans, categories = plan
    values = tuple(float(value) for value in values)
    if len(values) != len(queries) or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("native sound loss must be finite and nonnegative for every query")
    continuous = tuple(0. if span is None else (values[span[1]] - values[span[0]]) / span[2] for span in spans)
    categorical = tuple(tuple(values[index] for index in row) for row in categories)
    return NativeGradient(anchor, values[0], continuous, categorical, len(queries))


def measure_native_gradient(codec, anchor, evaluate=None, *, evaluate_many=None):
    if evaluate is None and evaluate_many is None:
        raise ValueError("native gradient needs a scalar or bulk evaluator")
    plan = native_gradient_queries(codec, anchor)
    values = evaluate_many(plan[0]) if evaluate_many is not None else [evaluate(query) for query in plan[0]]
    return gradient_from_values(anchor, plan, values)


def measured_loss(continuous, choices, estimate):
    return measured_losses(continuous, choices, [estimate])[0]


def measured_losses(continuous, choices, estimates):
    """Exact measured forward values and estimated backwards for an arbitrary example batch.

    Pass raw controls, not adapter-canonicalized values: the category loss tables already
    include inactive-default jumps. Applying that blend's derivative again double-counts them.
    Choices may come from the existing adapter's hard Gumbel-softmax output.
    """
    batch = len(estimates)
    if (not batch or continuous.shape != (batch, 9) or continuous.dtype != torch.float32
            or not torch.isfinite(continuous).all() or len(choices) != 2):
        raise ValueError("expected finite float32 basic control rows and two choice heads")
    anchor = continuous.new_tensor([estimate.anchor.continuous for estimate in estimates])
    if not torch.equal(continuous.detach(), anchor):
        raise ValueError("gradient estimate is stale for these raw controls")
    for estimate in estimates:
        coefficients = (*estimate.continuous, *(value for row in estimate.categorical for value in row), estimate.value)
        if (len(estimate.continuous) != 9 or tuple(map(len, estimate.categorical)) != (3, 2)
                or not all(math.isfinite(value) for value in coefficients)):
            raise ValueError("invalid native gradient coefficients")
    loss = continuous.new_tensor([estimate.value for estimate in estimates])
    loss = loss + ((continuous - continuous.detach()) * continuous.new_tensor([estimate.continuous for estimate in estimates])).sum(1)
    for head, (choice, size) in enumerate(zip(choices, (3, 2), strict=True)):
        expected = torch.nn.functional.one_hot(torch.tensor([estimate.anchor.categorical[head] for estimate in estimates], device=continuous.device), size)
        if (choice.shape != (batch, size) or choice.dtype != continuous.dtype or choice.device != continuous.device
                or not torch.equal(choice.detach(), expected.to(continuous.dtype))):
            raise ValueError("choices must match the hard patch that was measured")
        loss = loss + ((choice - choice.detach()) * choice.new_tensor([estimate.categorical[head] for estimate in estimates])).sum(1)
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

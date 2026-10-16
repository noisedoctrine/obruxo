"""Small, legal candidate sets from audio-only predictions and train-only retrieval."""

import torch

from .codec import PilotTarget
from .model import predicted_targets


PROPOSAL_SPEC = {"proposal": "model_retrieval_category_continuous_v1", "maximum_candidates": 4,
                 "category": "second_choice_of_lowest_probability_margin_active_head",
                 "continuous": "three_active_controls_random_sign_step_0.15", "duplicates": "keep_first"}


def propose_candidates(prediction, features, replay, codec, *, seed=0):
    from .evaluate import nearest_training_target
    if prediction["continuous"].shape[0] != 1 or type(seed) is not int:
        raise ValueError("candidate proposal needs one audio prediction and an integer seed")
    target = predicted_targets(prediction, codec)[0]
    proposals = {"model": target, "retrieval": nearest_training_target(features, replay)}
    active = codec.categorical_mask(target) if hasattr(codec, "categorical_mask") else [True] * len(codec.categorical)
    choices = []
    for index, (logits, enabled) in enumerate(zip(prediction["categorical"], active, strict=True)):
        if enabled and logits.shape[1] > 1:
            probabilities, indices = logits[0].detach().cpu().softmax(0).sort(descending=True, stable=True)
            choices.append((float(probabilities[0] - probabilities[1]), index, int(indices[1])))
    if choices:
        _, index, alternative = min(choices)
        categories = list(target.categorical)
        categories[index] = alternative
        proposals["category"] = codec.canonical_target(PilotTarget(target.continuous, tuple(categories)))
    rng = torch.Generator().manual_seed(seed)
    active = [index for index, enabled in enumerate(codec.continuous_mask(target)) if enabled]
    positions = torch.randperm(len(active), generator=rng)[:3].tolist()
    signs = torch.randint(2, (len(positions),), generator=rng).mul(2).sub(1).tolist()
    values = list(target.continuous)
    for position, sign in zip(positions, signs, strict=True):
        index = active[position]
        values[index] = min(1., max(0., values[index] + .15 * sign))
    proposals["continuous"] = codec.canonical_target(PilotTarget(tuple(values), target.categorical))
    unique, seen = {}, set()
    for name, value in proposals.items():
        value = codec.canonical_target(value)
        if value not in seen:
            codec.validate_target(value)
            unique[name] = value
            seen.add(value)
    return unique


def confirmed_improvement(scores, incumbent="model", *, relative_margin=.05):
    """A held-out second render must beat both incumbent draws by the declared margin."""
    if not 0 <= relative_margin < 1 or incumbent not in scores:
        raise ValueError("confirmation needs an incumbent and a valid relative margin")
    baseline = scores[incumbent]
    if len(baseline) != 2 or any(value is None or not torch.isfinite(torch.tensor(value)) or value < 0 for value in baseline):
        return None
    qualified = [(max(values), name) for name, values in scores.items() if name != incumbent and len(values) == 2
                 and all(value is not None and torch.isfinite(torch.tensor(value)) and value >= 0 for value in values)
                 and max(values) < (1 - relative_margin) * min(baseline)]
    return min(qualified)[1] if qualified else None

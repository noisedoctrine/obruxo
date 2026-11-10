"""Connect an inverse model's outputs to a native sound-only evaluator."""

import torch

from .codec import PilotTarget
from .native_gradient import gradient_from_values, measured_losses, native_gradient_queries


def native_reconstruction_loss(prediction, adapter, evaluate=None, *, temperature=1., evaluate_many=None, reduction="mean"):
    """Plan independent native queries together, then attach their measured backward paths.

    `evaluate(example_index, canonical_patch)` returns the measured sound loss for that
    example's target audio and evaluator MIDI. Neither source parameters nor MIDI enter
    the inverse model here. A training runner must enforce qualification and query caps.
    """
    raw, logits = prediction["continuous"], prediction["categorical"]
    if raw.ndim != 2 or not len(raw):
        raise ValueError("native reconstruction needs a nonempty prediction batch")
    if evaluate is None and evaluate_many is None:
        raise ValueError("native reconstruction needs a scalar or bulk evaluator")
    if reduction not in {"mean", "none"}:
        raise ValueError("unsupported native loss reduction")
    # Preserve per-example Gumbel draw order across execution batch sizes and recovery.
    encoded = torch.cat([adapter(raw[index:index + 1], [head[index:index + 1] for head in logits],
                                 temperature=temperature, sample=True) for index in range(len(raw))])
    plans, anchors, queries = [], [], []
    for index in range(len(raw)):
        choices = (encoded[index:index + 1, 9:12], encoded[index:index + 1, 12:])
        # This is the explicit native boundary; measured_loss supplies the backward path.
        anchor = PilotTarget(tuple(raw[index].detach().cpu().tolist()),
                             tuple(int(choice.detach().argmax(-1).item()) for choice in choices))
        plan = native_gradient_queries(adapter.codec, anchor)
        anchors.append(anchor)
        plans.append(plan)
        queries.extend((index, patch) for patch in plan[0])
    values = list(evaluate_many(queries) if evaluate_many is not None else (evaluate(index, patch) for index, patch in queries))
    if len(values) != len(queries):
        raise ValueError("native evaluator returned the wrong query count")
    estimates, offset = [], 0
    for index, (anchor, plan) in enumerate(zip(anchors, plans, strict=True)):
        count = len(plan[0])
        estimate = gradient_from_values(anchor, plan, values[offset:offset + count])
        offset += count
        estimates.append(estimate)
    result = measured_losses(raw, (encoded[:, 9:12], encoded[:, 12:]), estimates)
    return (result.mean() if reduction == "mean" else result), estimates

"""Connect an inverse model's outputs to a native sound-only evaluator."""

import torch

from .codec import PilotTarget
from .native_gradient import measure_native_gradient, measured_loss


def native_reconstruction_loss(prediction, adapter, evaluate, *, temperature=1.):
    """Measure serially, then backpropagate the mean local loss through all 14 outputs.

    `evaluate(example_index, canonical_patch)` returns the measured sound loss for that
    example's target audio and evaluator MIDI. Neither source parameters nor MIDI enter
    the inverse model here. A training runner must enforce qualification and query caps.
    """
    raw, logits = prediction["continuous"], prediction["categorical"]
    if raw.ndim != 2 or not len(raw):
        raise ValueError("native reconstruction needs a nonempty prediction batch")
    encoded = adapter(raw, logits, temperature=temperature, sample=True)
    losses, estimates = [], []
    for index in range(len(raw)):
        choices = (encoded[index:index + 1, 9:12], encoded[index:index + 1, 12:])
        # This is the explicit native boundary; measured_loss supplies the backward path.
        anchor = PilotTarget(tuple(raw[index].detach().cpu().tolist()),
                             tuple(int(choice.detach().argmax(-1).item()) for choice in choices))
        estimate = measure_native_gradient(adapter.codec, anchor, lambda patch: evaluate(index, patch))
        losses.append(measured_loss(raw[index:index + 1], choices, estimate))
        estimates.append(estimate)
    return torch.stack(losses).mean(), estimates

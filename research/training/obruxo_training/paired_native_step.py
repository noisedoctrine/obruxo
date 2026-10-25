"""Matched updates keep original and wider observations together in one minibatch."""

import torch

from .codec import PilotCodec
from .conditioned_fit_policy import gradient_groups, require_frozen
from .model import batch_features, loss_terms
from .native_inverse import native_reconstruction_loss
from .production_controls import ProductionControlAdapter
from .spectral_controls import temperature_at


def reciprocal_updates(count, *, seed=0):
    if type(count) is not int or count < 1 or type(seed) is not int:
        raise ValueError("paired updates need a positive state count and integer seed")
    pairs = [(index, view) for index in range(count) for view in (1, 2)]
    order = torch.randperm(len(pairs), generator=torch.Generator().manual_seed(seed)).tolist()
    return [{"examples": [{"state": pairs[index][0], "input_view": 0, "view": pairs[index][1]},
                          {"state": pairs[index][0], "input_view": pairs[index][1], "view": 0}]} for index in order]


def pair_prediction(model, features):
    # Match production's singleton forward exactly; average independent observations in the loss.
    rows = [model(batch_features([{"features": value}])) for value in features]
    return {"continuous": torch.cat([row["continuous"] for row in rows]),
            "categorical": [torch.cat([row["categorical"][head] for row in rows]) for head in (0, 1)]}


def matched_pair_step(models, optimizers, features, targets, codec, evaluate, step, steps, *, scales):
    if type(codec) is not PilotCodec or len(features) != 2 or len(targets) != 2 or targets[0] != targets[1]:
        raise ValueError("paired native update requires two observations of the same basic patch")
    for target in targets:
        codec.validate_target(target)
    adapter = ProductionControlAdapter(codec, scales=scales)
    temperature = temperature_at(step, steps)
    for optimizer in optimizers.values():
        optimizer.zero_grad(set_to_none=True)
    predictions = {name: pair_prediction(models[name], features) for name in ("parameter", "spectral")}
    # The matched control always receives raw logits. Scales belong only to the adapter's backward path.
    parameter = loss_terms(predictions["parameter"], targets, codec)
    spectral = predictions["spectral"]
    for value in (spectral["continuous"], *spectral["categorical"]):
        value.retain_grad()
    production = adapter.export(spectral["continuous"], spectral["categorical"])
    sound_loss, estimates = native_reconstruction_loss(spectral, adapter, evaluate, temperature=temperature)
    if len(estimates) != 2 or [codec.canonical_target(row.anchor) for row in estimates] != production:
        raise ValueError("paired native measurements differ from the two production predictions")
    losses = {"parameter": parameter["loss"], "spectral": sound_loss}
    for name, loss in losses.items():
        if not torch.isfinite(loss):
            raise ValueError("nonfinite paired inverse objective")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(models[name].parameters(), 5.)
        if not torch.isfinite(norm):
            raise ValueError("nonfinite paired inverse gradient")
    gradients = {"continuous_l1": spectral["continuous"].grad.abs().sum(-1).tolist(),
                 "category_l1": [scores.grad.abs().sum(-1).tolist() for scores in spectral["categorical"]]}
    correction = {name: gradient_groups(model) for name, model in models.items() if hasattr(model, "conditioning_training_policy")}
    # Do not advance either optimizer if a native callback or either backward pass fails.
    for name in ("parameter", "spectral"):
        optimizers[name].step()
        require_frozen(models[name])
    return {"losses": {name: float(value.detach()) for name, value in losses.items()}, "gradients": gradients,
            **({"correction_gradients": correction} if correction else {}),
            "gradient_queries": sum(row.queries for row in estimates), "estimates": [row.to_dict() for row in estimates],
            "production_patches": [target.to_dict() for target in production], "temperature": temperature}

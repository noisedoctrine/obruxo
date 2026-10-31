"""One matched sound/label update with two paired and two rehearsed observations."""

import torch

from .basic_direct import REHEARSAL, reference_loss
from .batching import BatchConfig, concatenate_predictions, prediction_row
from .continuous_auxiliary import SPEC as AUXILIARY_SPEC, paired_continuous_loss
from .model import batch_features, loss_terms
from .native_inverse import native_reconstruction_loss
from .production_controls import ProductionControlAdapter
from .spectral_controls import temperature_at


def gradients(model):
    model.require_frozen()
    groups = {"hidden": 0., "continuous": 0., "unison": 0., "filter": 0.}
    for parameter in model.correction.parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise ValueError("missing or nonfinite direct-correction gradient")
    groups["hidden"] = sum(float(p.grad.double().abs().sum()) for p in model.correction[0].parameters())
    for name, section in (("continuous", slice(0, 9)), ("unison", slice(9, 12)), ("filter", slice(12, 14))):
        groups[name] = sum(float(p.grad[section].double().abs().sum()) for p in model.correction[2].parameters())
    return groups


def predict(model, features):
    return predict_many(model, [features], batch_size=1)[0]


def predict_many(model, features, *, batch_size=32):
    BatchConfig(model=batch_size)
    rows = []
    for start in range(0, len(features), batch_size):
        rows.extend(_predict_chunk(model, features[start:start + batch_size]))
    return rows


def _predict_chunk(model, features):
    batch = batch_features([{"features": feature} for feature in features])
    delta = []
    hook = model.correction.register_forward_hook(lambda _module, _args, output: delta.append(output.detach()))
    try:
        result = model(batch)
    finally:
        hook.remove()
    with torch.no_grad():
        proposal = model.base(batch)["continuous"] + delta[0][:, :9]
    return [(prediction_row(result, index), {"minimum_proposal": float(row.min()), "maximum_proposal": float(row.max()),
             "below_zero": int((row < 0).sum()), "above_one": int((row > 1).sum()), "controls": row.numel()})
            for index, row in enumerate(proposal)]


def matched_step(models, optimizers, codec, observations, evaluate, step, steps, *, norm_cap=5., continuous_auxiliary=None,
                 evaluate_many=None, model_batch_size=32):
    if continuous_auxiliary is not None and continuous_auxiliary != AUXILIARY_SPEC:
        raise ValueError("continuous auxiliary differs from its fixed paired contract")
    if len(observations) != 4 or [len(row["cells"]) for row in observations] != [2, 2, 1, 1]:
        raise ValueError("direct fitting needs a reciprocal pair and two singleton references")
    if observations[0]["target"] != observations[1]["target"]:
        raise ValueError("reciprocal observations must describe the same patch")
    adapter = ProductionControlAdapter(codec, scales=[1., 1.])
    weights = REHEARSAL["paired_weights"] + REHEARSAL["reference_weights"]
    temperature = temperature_at(step, steps)
    for model in models.values():
        model.require_frozen()
    for optimizer in optimizers.values():
        optimizer.zero_grad(set_to_none=True)
    predictions, clamping = {}, {}
    for arm, model in models.items():
        rows = predict_many(model, [row["features"] for row in observations], batch_size=model_batch_size)
        predictions[arm], clamping[arm] = [row[0] for row in rows], [row[1] for row in rows]
    parameter = [loss_terms(prediction, [row["target"]], codec)["loss"]
                 for prediction, row in zip(predictions["parameter"], observations, strict=True)]
    prediction = concatenate_predictions(predictions["spectral"])
    values, measured = native_reconstruction_loss(prediction, adapter, evaluate, temperature=temperature,
                                                  evaluate_many=evaluate_many, reduction="none")
    production = adapter.export(prediction["continuous"], prediction["categorical"])
    sound, estimates, patches = list(values.unbind()), [], []
    for estimate, patch in zip(measured, production, strict=True):
        if codec.canonical_target(estimate.anchor) != patch:
            raise ValueError("direct native measurements differ from production")
        estimates.append(estimate.to_dict())
        patches.append(patch.to_dict())
    penalized = sound[:2] + [reference_loss(sound[i], observations[i]["baseline"]) for i in (2, 3)]
    losses = {"parameter": sum(w * value for w, value in zip(weights, parameter, strict=True)),
              "spectral": sum(w * value for w, value in zip(weights, penalized, strict=True))}
    native_sound = losses["spectral"]
    auxiliary = None
    if continuous_auxiliary is not None:
        auxiliary = paired_continuous_loss(predictions["spectral"][:2], [row["target"] for row in observations[:2]], codec)
        losses["spectral"] = native_sound + AUXILIARY_SPEC["weight"] * auxiliary
    norms, access = {}, {}
    for arm, loss in losses.items():
        if not torch.isfinite(loss):
            raise ValueError("nonfinite direct fitting objective")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(models[arm].correction.parameters(), norm_cap)
        if not torch.isfinite(norm):
            raise ValueError("nonfinite direct fitting gradient")
        norms[arm], access[arm] = float(norm), gradients(models[arm])
    # Rendering and both backward passes must succeed before either optimizer advances.
    for arm in ("parameter", "spectral"):
        optimizers[arm].step()
        models[arm].require_frozen()
    result = {"temperature": temperature, "losses": {arm: float(value.detach()) for arm, value in losses.items()},
            "observation_sound_losses": [float(value.detach()) for value in sound],
            "reference_penalized_losses": [float(value.detach()) for value in penalized[2:]],
            "gradient_norms": norms, "correction_gradients": access, "clamping": clamping,
            "estimates": estimates, "production_patches": patches, "gradient_queries": sum(row["queries"] for row in estimates)}
    if auxiliary is not None:
        result.update(native_sound_loss=float(native_sound.detach()), continuous_auxiliary_loss=float(auxiliary.detach()), continuous_auxiliary_weight=AUXILIARY_SPEC["weight"])
    return result

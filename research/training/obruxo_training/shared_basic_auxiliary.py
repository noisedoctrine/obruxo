"""Compare native sound learning with one training-only pitch/duration adversary."""

import copy
import math

import torch

from .basic_direct import REHEARSAL, reference_loss
from .basic_direct_fit_step import gradients, predict_many
from .batching import BatchConfig, concatenate_predictions
from .branch_adversary import note_targets, pair_indices
from .fitted_note_head import FittedNoteHead
from .inward_basic import training_model
from .native_acceptance import accept_update, capture
from .native_inverse import native_reconstruction_loss
from .spectral_controls import temperature_at

ARMS = ("sound_only", "sound_auxiliary")
SPEC = {"format": "shared_basic_sound_note_auxiliary_v1", "training_indices": [2, 8, 11], "epochs": 8,
        "updates_per_arm": 48, "learning_rate": .001, "gradient_norm_cap": 5., "seed": 0,
        "auxiliary_batch_patches": 32, "auxiliary_batch_seed": 101, "calibration_update": 2,
        "note_to_sound_encoder_gradient_ratio": 1., "coefficient_limits": [.001, 1000.], "ramp_updates": 8,
        "head_refit": "before_each_update_on_training_patches_only", "primary": "unchanged_native_sound_and_reference_rehearsal",
        "acceptance": "unchanged_actual_sound_decrease_only", "additional_inference_outputs": 0}


@torch.no_grad()
def encode(encoder, rows):
    return torch.cat([encoder(torch.stack([row["features"]["timbre"] for row in rows[start:start + 64]]))
                      for start in range(0, len(rows), 64)])


def initialize(inputs):
    models = {arm: training_model(copy.deepcopy(inputs.model)).train() for arm in ARMS}
    optimizers = {arm: torch.optim.Adam(model.correction.parameters(), lr=SPEC["learning_rate"]) for arm, model in models.items()}
    return models, optimizers


def calibrate(sound, note, encoder):
    parameters = tuple(encoder.parameters())
    primary = torch.autograd.grad(sound, parameters, retain_graph=True)
    auxiliary = torch.autograd.grad(note, parameters, retain_graph=True)
    a, b = (float(torch.cat([value.flatten() for value in gradients]).double().norm()) for gradients in (primary, auxiliary))
    if not all(math.isfinite(value) and value > 0 for value in (a, b)):
        raise ValueError("sound/auxiliary calibration needs nonzero finite timbre gradients")
    coefficient = SPEC["note_to_sound_encoder_gradient_ratio"] * a / b
    lower, upper = SPEC["coefficient_limits"]
    if not lower <= coefficient <= upper:
        raise ValueError("auxiliary coefficient is outside its declared limits")
    return {"coefficient": coefficient, "sound_encoder_norm": a, "unscaled_note_encoder_norm": b}


def sound_objective(inputs, model, cell_lists, evaluate, index, evaluate_many=None):
    predictions, observations, clamping = [], [], []
    features = [inputs.features[inputs.key(inputs.plan["fit_audit_cells"][cells[0]])] for cells in cell_lists]
    rows = predict_many(model, features, batch_size=BatchConfig.from_plan(inputs.plan).model)
    for cells, (prediction, bounds) in zip(cell_lists, rows, strict=True):
        predictions.append(prediction)
        clamping.append(bounds)
        observations.append({"baseline": inputs.cases[cells[0]]["rows"]["initial"]["scores"]["spectral"]})
    values, measured = native_reconstruction_loss(concatenate_predictions(predictions), inputs.adapter, evaluate,
        temperature=temperature_at(index, len(inputs.plan["updates"])), evaluate_many=evaluate_many, reduction="none")
    values = list(values.unbind())
    estimates = [estimate.to_dict() for estimate in measured]
    penalized = values[:2] + [reference_loss(values[i], observations[i]["baseline"]) for i in (2, 3)]
    weights = REHEARSAL["paired_weights"] + REHEARSAL["reference_weights"]
    loss = sum(weight * value for weight, value in zip(weights, penalized, strict=True))
    return loss, estimates, observations, clamping


def step(inputs, store, models, optimizers, auxiliary_rows, index, calibration):
    cell_lists = inputs.observations(inputs.plan["updates"][index])
    result = {"step": index + 1, "arms": {}}
    for arm, model in models.items():
        optimizer = optimizers[arm]
        optimizer.zero_grad(set_to_none=True)
        before = capture(inputs, model, anchor_cells=[inputs.plan["fit_audit_cells"][cells[0]] for cells in cell_lists])

        def evaluate(observation, patch, arm=arm):
            costs = [store.probe(index + 1, f"{arm}-{observation}", cell, patch) for cell in cell_lists[observation]]
            return sum(costs) / len(costs)

        evaluate_many = (lambda requests: store.evaluate_many(index + 1, [f"{arm}-{i}" for i in range(len(cell_lists))], cell_lists, requests)) if hasattr(store, "evaluate_many") else None
        loss, estimates, observations, clamping = sound_objective(inputs, model, cell_lists, evaluate, index, evaluate_many)
        note, coefficient = None, 0.
        if arm == "sound_auxiliary":
            encoder = model.correction[0].timbre
            head = FittedNoteHead()
            head.fit(encode(encoder, auxiliary_rows), note_targets(auxiliary_rows))
            by_state = {}
            for position, row in enumerate(auxiliary_rows):
                by_state.setdefault(row["state_id"], []).append(position)
            selected = pair_indices(by_state, torch.Generator().manual_seed(SPEC["auxiliary_batch_seed"] + index),
                                    patches=SPEC["auxiliary_batch_patches"])
            batch = [auxiliary_rows[position] for position in selected]
            representation = encoder(torch.stack([row["features"]["timbre"] for row in batch]))
            note = (head(representation, coefficient=1.) - note_targets(batch).double()).square().mean()
            if index + 1 == SPEC["calibration_update"]:
                calibration = calibrate(loss, note, encoder)
            if index + 1 >= SPEC["calibration_update"]:
                if calibration is None:
                    raise ValueError("missing the fixed training-only auxiliary calibration")
                coefficient = calibration["coefficient"] * min(1., (index + 1) / SPEC["ramp_updates"])
        total = loss if not coefficient else loss + coefficient * note
        if not torch.isfinite(total):
            raise ValueError("nonfinite shared sound objective")
        total.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.correction.parameters(), SPEC["gradient_norm_cap"], error_if_nonfinite=True)
        access = gradients(model)
        optimizer.step()
        accepted = accept_update(inputs, model, before, cell_lists, observations,
                                 {"estimates": estimates, "losses": {"spectral": float(loss.detach())}}, evaluate, evaluate_many=evaluate_many)
        result["arms"][arm] = {"sound_loss": float(loss.detach()), "note_mse": None if note is None else float(note.detach()),
            "coefficient": coefficient, "gradient_norm": float(norm), "gradients": access, "clamping": clamping,
            "estimates": estimates, "acceptance": accepted}
    return result, calibration

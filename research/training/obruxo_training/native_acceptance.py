"""Accept bounded Adam proposals only after measuring their actual native sound."""

import math

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from .basic_direct import REHEARSAL, reference_loss

SPEC = {"format": "actual_native_batch_acceptance_v1", "control_change_cap": .025, "control_cap_tolerance": 1e-8,
        "cap_search_factors": [2.**-i for i in range(9)], "maximum_native_trials": 3, "minimum_objective_decrease": 1e-6,
        "objective": "original_weighted_native_sound_and_accepted_initial_reference_hinges_float32",
        "refused_weights": "restore_pre_update_exactly", "adam_moments": "advance_once_per_scheduled_update_including_refusals",
        "gradient_estimates": "fresh_at_each_committed_state", "individual_sound_monotonicity": False, "auxiliary_labels": False}


def objective(costs, baselines):
    if len(costs) != 4 or len(baselines) != 4 or any(not math.isfinite(value) or value < 0 for value in costs):
        raise ValueError("acceptance requires four finite nonnegative measured sound costs")
    values = [torch.tensor(value, dtype=torch.float32) for value in costs]
    penalized = values[:2] + [reference_loss(values[i], baselines[i]) for i in (2, 3)]
    weights = REHEARSAL["paired_weights"] + REHEARSAL["reference_weights"]
    return float(sum(weight * value for weight, value in zip(weights, penalized, strict=True)))


@torch.no_grad()
def bank(inputs, model, cells):
    predictions = [inputs.predict(model, cell) for cell in cells]
    continuous = torch.cat([row["continuous"] for row in predictions])
    if continuous.shape != (len(cells), 9) or not torch.isfinite(continuous).all():
        raise ValueError("invalid finite control bank for native acceptance")
    choices = [[int(value.argmax(-1)) for value in row["categorical"]] for row in predictions]
    return continuous, choices


def capture(inputs, model):
    model.require_frozen()
    unique = {inputs.key(cell): cell for cell in inputs.plan["fit_audit_cells"]}
    cells = list(unique.values())
    continuous, choices = bank(inputs, model, cells)
    return {"weights": parameters_to_vector(model.correction.parameters()).detach().clone(), "cells": cells,
            "keys": list(unique), "continuous": continuous, "choices": choices}


def accept_update(inputs, model, before, cell_lists, observations, result, evaluate):
    """Keep measured improvements; restore weights on refusal or any exception.

    Adam has already advanced once. This function never alters its moments or
    applies another optimizer update. Trial audio is measured through the same
    durable native-probe store as the fresh gradient measurements.
    """
    parameters = tuple(model.correction.parameters())
    proposed = parameters_to_vector(parameters).detach().clone()
    displacement = proposed - before["weights"]
    estimates = result["estimates"]
    costs, baselines = [row["value"] for row in estimates], [row["baseline"] for row in observations]
    cap_checks, trials, kept = [], [], False
    try:
        previous = objective(costs, baselines)
        if previous != result["losses"]["spectral"]:
            raise ValueError("acceptance changed the original float32 sound objective")
        for cells, estimate in zip(cell_lists, estimates, strict=True):
            key = inputs.key(inputs.plan["fit_audit_cells"][cells[0]])
            index = before["keys"].index(key)
            if (before["continuous"][index].tolist() != estimate["anchor"]["continuous"]
                    or before["choices"][index] != estimate["anchor"]["categorical"]):
                raise ValueError("acceptance reached a stale native measurement anchor")

        @torch.no_grad()
        def apply(scale):
            candidate = proposed if scale == 1 else before["weights"] + scale * displacement
            vector_to_parameters(candidate, parameters)
            continuous, _choices = bank(inputs, model, before["cells"])
            return float((continuous - before["continuous"]).abs().max())

        initial_scale = None
        for scale in SPEC["cap_search_factors"]:
            maximum = apply(scale)
            cap_checks.append({"scale": scale, "maximum_control_change": maximum})
            if maximum <= SPEC["control_change_cap"] + SPEC["control_cap_tolerance"]:
                initial_scale = scale
                break
        if initial_scale is not None:
            for trial in range(SPEC["maximum_native_trials"]):
                scale = initial_scale / 2**trial
                maximum = apply(scale)
                if maximum > SPEC["control_change_cap"] + SPEC["control_cap_tolerance"]:
                    trials.append({"scale": scale, "maximum_control_change": maximum, "status": "control_cap_refused"})
                    continue
                with torch.no_grad():
                    predictions = [inputs.predict(model, inputs.plan["fit_audit_cells"][cells[0]]) for cells in cell_lists]
                    patches = [inputs.adapter.export(row["continuous"], row["categorical"])[0] for row in predictions]
                    measured = [evaluate(index, patch) for index, patch in enumerate(patches)]
                after = objective(measured, baselines)
                kept = previous - after > SPEC["minimum_objective_decrease"]
                trials.append({"scale": scale, "maximum_control_change": maximum, "status": "accepted" if kept else "sound_refused",
                               "measured_sound_costs": measured, "objective": after, "patches": [patch.to_dict() for patch in patches]})
                if kept:
                    break
        return {"format": SPEC["format"], "accepted": kept, "scale": trials[-1]["scale"] if kept else 0.,
                "before_objective": previous, "committed_objective": trials[-1]["objective"] if kept else previous,
                "cap_checks": cap_checks, "trials": trials, "bank_inputs": len(before["cells"]),
                "logical_trial_evaluations": 4 * sum("objective" in row for row in trials),
                "maximum_physical_trial_queries": sum(map(len, cell_lists)) * sum("objective" in row for row in trials),
                "adam_moments_advanced": True}
    finally:
        if not kept:
            with torch.no_grad():
                vector_to_parameters(before["weights"], parameters)
        model.require_frozen()

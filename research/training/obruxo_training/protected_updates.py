"""Project an Adam proposal to protect each sampled sound, then guard its outputs."""

from itertools import combinations

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from .update_diagnosis import movement, predict_bank

SPEC = {"format": "sampled_sound_protected_adam_v1", "constraints": "four_individual_raw_sound_gradients_dot_displacement_nonpositive",
        "projection": "closest_euclidean_parameter_displacement", "continuous_bank_change_cap": .025,
        "backtracking_factors": [2.**-i for i in range(9)], "estimated_change_tolerance": 1e-8,
        "native_sound_acceptance_qualified": False, "auxiliary_labels": False}


def project_direction(proposal, gradients):
    """Solve a tiny active-set projection; zero is always feasible."""
    rows, direction = torch.stack(gradients).double(), proposal.double()
    if rows.ndim != 2 or direction.ndim != 1 or rows.shape[1] != len(direction) or not 1 <= len(rows) <= 4:
        raise ValueError("protect one to four matching sound gradients")
    if not torch.isfinite(rows).all() or not torch.isfinite(direction).all():
        raise ValueError("protected proposal must be finite")
    norms = rows.norm(dim=1)
    rows = rows[norms > 0] / norms[norms > 0, None]
    tolerance = 1e-10 * max(float(direction.norm()), 1e-12)
    best, distance = torch.zeros_like(direction), float(direction.square().sum())
    for count in range(len(rows) + 1):
        for indices in combinations(range(len(rows)), count):
            if indices:
                active = rows[list(indices)]
                gram, rhs = active @ active.T, active @ direction
                multipliers = torch.linalg.lstsq(gram, rhs, rcond=1e-12, driver="gelsd").solution
                if (multipliers < -tolerance).any():
                    continue
                candidate = direction - active.T @ multipliers
            else:
                candidate = direction
            if len(rows) and (rows @ candidate > tolerance).any():
                continue
            cost = float((candidate - direction).square().sum())
            if cost <= distance:
                best, distance = candidate, cost
    return best.to(proposal), {"proposal_norm": float(direction.norm()), "projected_norm": float(best.norm()),
                              "projection_distance": distance**.5, "nonzero_sound_gradients": len(rows)}


def loss_gradients(losses, model):
    parameters = tuple(model.correction.parameters())
    return [torch.cat([gradient.detach().flatten() for gradient in torch.autograd.grad(loss, parameters, retain_graph=True)]) for loss in losses]


def select_proposal(model, before, proposal, gradients, inputs, observations, estimates):
    """Inspect temporary candidates and restore the incoming model in all cases.

    These guards use local native estimates. They cannot replace actual native
    after-update checks or establish sound quality for a new training trajectory.
    """
    parameters = tuple(model.correction.parameters())
    incoming = parameters_to_vector(parameters).detach().clone()
    projected, projection = project_direction(proposal, gradients)
    tolerance = SPEC["estimated_change_tolerance"]
    try:
        with torch.no_grad():
            vector_to_parameters(before, parameters)
            initial = predict_bank(inputs, model)
            for scale in SPEC["backtracking_factors"]:
                candidate = before + scale * projected
                vector_to_parameters(candidate, parameters)
                after = predict_bank(inputs, model)
                changes = [movement(initial[inputs.key(inputs.plan["fit_audit_cells"][cells[0]])],
                                    after[inputs.key(inputs.plan["fit_audit_cells"][cells[0]])], estimate, inputs.codec)
                           for cells, estimate in zip(observations, estimates, strict=True)]
                maximum = max(abs(b - a) for key in initial for a, b in zip(initial[key]["continuous"], after[key]["continuous"], strict=True))
                if (maximum <= SPEC["continuous_bank_change_cap"] + tolerance
                        and all(row["estimated_sound_change"] <= tolerance for row in changes)
                        and any(row["estimated_sound_change"] < -tolerance for row in changes)):
                    return candidate.clone(), {**projection, "accepted": True, "scale": scale, "maximum_bank_control_change": maximum, "changes": changes}
        changes = [movement(initial[inputs.key(inputs.plan["fit_audit_cells"][cells[0]])],
                            initial[inputs.key(inputs.plan["fit_audit_cells"][cells[0]])], estimate, inputs.codec)
                   for cells, estimate in zip(observations, estimates, strict=True)]
        return before.clone(), {**projection, "accepted": False, "scale": 0., "maximum_bank_control_change": 0., "changes": changes}
    finally:
        with torch.no_grad():
            vector_to_parameters(incoming, parameters)

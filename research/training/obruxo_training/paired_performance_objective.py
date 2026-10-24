"""One inferred patch must explain its input and reciprocal performances."""

INPUT_TRANSFER = "mean_input_and_transfer_v1"


def evaluation_views(plan, example):
    objective = plan.get("reconstruction_objective")
    if objective is None:
        return (example["view"],)
    if objective != INPUT_TRANSFER:
        raise ValueError("unknown paired reconstruction objective")
    views = (example["input_view"], example["view"])
    if any(type(view) is not int or view not in (0, 1, 2) for view in views) or views[0] == views[1]:
        raise ValueError("input and reciprocal performances must be distinct valid views")
    return views


def include_input_performance(plan):
    if "reconstruction_objective" in plan:
        raise ValueError("preserve the already declared paired objective")
    planned = dict(plan["planned_maximum_requests"])
    planned["gradient_probes"] *= 2
    planned["total"] = sum(value for name, value in planned.items() if name != "total")
    allowance = 256 * 1024**2
    storage = {**plan["storage_bound"], "additional_component_ledger_allowance_bytes": allowance,
               "estimated_archive_upper_bound_bytes": plan["storage_bound"]["estimated_archive_upper_bound_bytes"] + allowance}
    limits = {**plan["limits"], "wall_seconds": 36000, "archive_bytes": 5 * 1024**3,
              "training_attempts": planned["training_targets"] + (len(plan["updates"]) + 4) * 2 * 2 * 22 + 32}
    if storage["estimated_archive_upper_bound_bytes"] > limits["archive_bytes"]:
        raise ValueError("input-and-transfer evidence exceeds its declared archive ceiling")
    return {**plan, "format": "paired_input_transfer_inverse_fit_v1", "reconstruction_objective": INPUT_TRANSFER,
            "performance_weights": [0.5, 0.5], "rng_initialization": "python_numpy_and_torch_from_optimizer_seed",
            "planned_maximum_requests": planned, "limits": limits, "storage_bound": storage}


def seed_external_streams(plan):
    if plan.get("reconstruction_objective") != INPUT_TRANSFER:
        return
    import random
    import numpy as np
    if plan["rng_initialization"] != "python_numpy_and_torch_from_optimizer_seed":
        raise ValueError("paired external RNG initialization changed")
    seed = plan["optimizer"]["seed"]
    random.seed(seed)
    np.random.seed(seed)

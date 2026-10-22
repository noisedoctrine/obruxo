"""Describe a verified basic fit's coverage and errors without new audio or learning."""

from collections import Counter

import numpy as np

from obruxo_data.hashing import canonical_sha256


METHODS = ("initial", "parameter", "spectral")


def active(target, index):
    voices, filtered = target["categorical"]
    return (index != 2 or voices != 0) and (index not in (3, 4) or filtered != 0)


def describe(values):
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"count": 0}
    if not np.isfinite(array).all():
        raise ValueError("nonfinite saved measurements")
    return {"count": len(array), "mean": float(array.mean()), "median": float(np.median(array)),
            "p90": float(np.quantile(array, .9)), "minimum": float(array.min()), "maximum": float(array.max())}


def score_groups(cases):
    groups = {}
    for view in (0, 1):
        rows = [case for case in cases if case["view"] == view]
        if not rows:
            continue
        methods = {method: {metric: describe([row["methods"][method]["scores"][metric] for row in rows])
                            for metric in ("mrstft", "spectral", "tail_error")} for method in METHODS}
        groups[str(view)] = {"states": len(rows), "methods": methods,
            "improved_vs_initial": sum(row["methods"]["spectral"]["scores"]["mrstft"] < row["methods"]["initial"]["scores"]["mrstft"] for row in rows)}
    return groups


def control_errors(cases, recipes, controls):
    rows = [case for case in cases if case["view"] == 0]
    result = []
    for index, control in enumerate(controls):
        enabled = [case for case in rows if active(recipes[case["state_id"]]["target"], index)]
        source = [recipes[case["state_id"]]["target"]["continuous"][index] for case in enabled]
        methods = {}
        for method in METHODS:
            predicted = [case["methods"][method]["candidate"]["continuous"][index] for case in enabled]
            errors = np.asarray(predicted) - np.asarray(source)
            methods[method] = {"absolute_error": describe(np.abs(errors)), "signed_error": describe(errors),
                               "near_bound_count": sum(value <= .02 or value >= .98 for value in predicted)}
        result.append({"name": control["name"], "source_active_states": len(enabled),
                       "source_values": describe(source), "methods": methods})
    return result


def coverage(entries, controls):
    counts = Counter(tuple(entry["target"]["categorical"]) for entry in entries)
    continuous = []
    for index, control in enumerate(controls):
        values = [entry["target"]["continuous"][index] for entry in entries if active(entry["target"], index)]
        continuous.append({"name": control["name"], "source_active_values": describe(values),
                           "equal_width_fifths": np.histogram(values, bins=np.linspace(0, 1, 6))[0].tolist()})
    return {"states": len(entries), "category_counts": [{"indices": list(key), "count": value} for key, value in sorted(counts.items())],
            "continuous": continuous}


def nearest_target(query, entries):
    target = query["target"]
    indices = [index for index in range(9) if active(target, index)]
    options = [entry for entry in entries if entry["target"]["categorical"] == target["categorical"]]
    if not options:
        return None
    distances = [(float(np.sqrt(np.mean([(entry["target"]["continuous"][index] - target["continuous"][index]) ** 2
                                        for index in indices]))), entry["state_id"]) for entry in options]
    distance, state_id = min(distances)
    return {"state_id": state_id, "normalized_control_rmse": distance}


def diagnose(report, pool):
    plan = report["plan"]
    required = ("complete", "offline_training_replay_exact", "cached_targets_and_inputs_rescored", "hard_audit_patches_reconstructed")
    if not all(report[key] for key in required) or report["failure"] is not None or report["plan_id"] != canonical_sha256(plan):
        raise ValueError("parent fit has not passed its evidence audit")
    if pool["pool_id"] != canonical_sha256(pool["pool"]) or pool["pool_id"] != plan["pool_id"] or pool["pool"]["codec"] != plan["codec"]:
        raise ValueError("pool identity or codec differs from audited fit")
    controls = plan["codec"]["continuous"]
    if len(controls) != 9 or [len(head["values"]) for head in plan["codec"]["categorical"]] != [3, 2]:
        raise ValueError("diagnosis requires the basic 14-output scope")
    splits = {row["state_id"]: row["split"] for row in pool["pool"]["splits"]["assignments"]}
    entries = {row["state_id"]: row for row in pool["pool"]["entries"]}
    training = [entry for state, entry in entries.items() if splits[state] == "train"]
    fitted, validation = plan["training"], plan["validation"]
    for name, recipes, expected_split in (("fit", fitted, "train"), ("validation", validation, "validation")):
        identities = [row["state_id"] for row in recipes]
        cases = report[f"{name}_cases"]
        if (len(set(identities)) != len(identities) or any(splits[state] != expected_split for state in identities)
                or any(row["target"] != entries[row["state_id"]]["target"] for row in recipes)
                or len(cases) != 2 * len(recipes)
                or {(row["state_id"], row["view"]) for row in cases} != {(state, view) for state in identities for view in (0, 1)}):
            raise ValueError("split, target or complete case membership changed")
        by_state = {row["state_id"]: row for row in cases if row["view"] == 0}
        for case in cases:
            if not case["repeat_exact"] or any(case["methods"][method]["candidate"] != by_state[case["state_id"]]["methods"][method]["candidate"] for method in METHODS):
                raise ValueError("source repeats or fixed cross-performance predictions changed")
    recipes = {row["state_id"]: row for row in (*fitted, *validation)}
    categories = sorted({tuple(row["target"]["categorical"]) for row in fitted + validation})
    panels = {}
    for name in ("fit", "validation"):
        cases = report[f"{name}_cases"]
        panels[name] = {"scores": score_groups(cases), "control_errors": control_errors(cases, recipes, controls),
            "category_groups": [{"source_indices": list(category), "scores": score_groups([case for case in cases
                if tuple(recipes[case["state_id"]]["target"]["categorical"]) == category])} for category in categories]}
    distances = [{"state_id": row["state_id"], "fitted_32": nearest_target(row, fitted), "all_training": nearest_target(row, training)} for row in validation]
    return {"format": "basic_fit_coverage_v1", "parent_plan_id": report["plan_id"], "pool_id": pool["pool_id"],
            "descriptive_only": True, "causal_claim": False, "native_requests": 0, "optimizer_updates": 0, "test_split_used": False,
            "definitions": {"control_errors": "normalized source-active values; one prediction per state; not perceptual scores",
                "near_bound": "prediction <=0.02 or >=0.98; descriptive, not a pass threshold",
                "nearest_target": "source-active normalized-control RMSE within identical source categories; not audio similarity",
                "coverage": "fitted states compared with existing training-only population; no test target statistics"},
            "coverage": {"fitted": coverage(fitted, controls), "all_training": coverage(training, controls),
                "fitted_fraction": len(fitted) / len(training), "validation_nearest_targets": distances,
                "nearest_distance_summary": {name: describe([row[name]["normalized_control_rmse"] for row in distances if row[name]])
                                             for name in ("fitted_32", "all_training")}}, "panels": panels}

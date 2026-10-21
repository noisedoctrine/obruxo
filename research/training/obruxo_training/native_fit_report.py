"""Descriptive per-sound evidence for the matched native inverse comparison."""

import numpy as np


def terminal_interruption(outcomes):
    failures = [row for row in outcomes if row["status"] != "accepted"]
    if len(failures) != 1 or failures[0] != outcomes[-1]:
        raise ValueError("expected one terminal native interruption")
    failure = failures[0]
    if failure["status"] == "rejected":
        return failure
    if failure["status"] == "error" and failure.get("error_type") == "RenderTimeout" and "qa" not in failure:
        return failure
    raise ValueError("unsupported native interruption; do not classify it as an audio rejection")


def paired_comparison(cases, *, resamples=2000):
    """Resample whole source states, retaining their paired reference/transfer observations."""
    by_state = {}
    for case in cases:
        views = by_state.setdefault(case["state_id"], {})
        if case["view"] in views or case["view"] not in (0, 1):
            raise ValueError("duplicate or unsupported comparison view")
        views[case["view"]] = case
    if not by_state or any(set(views) != {0, 1} for views in by_state.values()):
        raise ValueError("paired comparison requires both views for every source state")
    states = sorted(by_state)
    rng = np.random.default_rng(0)
    selections = rng.integers(0, len(states), size=(resamples, len(states)))
    groups = {}
    for view, role in ((0, "reference"), (1, "transfer")):
        selected = [by_state[state][view] for state in states]
        comparisons = {}
        for baseline in ("initial", "parameter"):
            candidate = np.array([row["rows"]["spectral"]["scores"]["mrstft"] for row in selected])
            control = np.array([row["rows"][baseline]["scores"]["mrstft"] for row in selected])
            delta = candidate - control
            if not np.isfinite(candidate).all() or not np.isfinite(control).all() or (candidate < 0).any() or (control < 0).any():
                raise ValueError("comparison scores must be finite and nonnegative")
            comparisons[baseline] = {"planned": len(states), "improved": int((delta < -1e-7).sum()),
                                     "mean_delta": float(delta.mean()), "median_delta": float(np.median(delta)),
                                     "delta_standard_deviation": float(delta.std(ddof=1)) if len(delta) > 1 else None,
                                     "mean_control": float(control.mean()),
                                     "mean_reduction_fraction": float(-delta.mean() / control.mean()) if control.mean() > 0 else None,
                                     "mean_delta_95_percent_interval": np.quantile(delta[selections].mean(1), [.025, .975]).tolist()}
        groups[role] = comparisons
    return {"groups": groups, "independent_source_states": len(states), "bootstrap_resamples": resamples,
            "bootstrap_seed": 0, "unit": "whole_source_state_with_both_views", "descriptive_not_an_additional_pass_gate": True}


def public_cases(cases):
    return [{"state_id": case["state_id"], "view": case["view"], "repeat_exact": case["repeat_exact"],
             "source_audio_sha256": case["source_receipt"]["qa"]["audio_float32_sha256"],
             "methods": {name: {"scores": row["scores"], "candidate": row["candidate"],
                                 "request_id": row["receipt"]["request_id"],
                                 "audio_sha256": row["receipt"]["qa"]["audio_float32_sha256"]}
                         for name, row in case["rows"].items()}} for case in cases]

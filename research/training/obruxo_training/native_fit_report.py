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


def recovered_timeouts(outcomes):
    recovered = []
    fields = ("phase", "label", "request_id", "source_sha256", "controlled_sha256", "renderer_identity")
    for index, row in enumerate(outcomes):
        if row["status"] == "accepted":
            continue
        if row["status"] != "error" or row.get("error_type") != "RenderTimeout" or "qa" in row:
            raise ValueError("a native failure is not a recoverable timeout")
        matches = [later for later in outcomes[index + 1:] if later["status"] == "accepted"
                   and all(later.get(name) == row.get(name) and name in row for name in fields)]
        if not matches:
            raise ValueError("timeout has no later accepted render of the identical request")
        recovered.append({"failed_attempt": row["attempt"], "accepted_attempt": matches[0]["attempt"],
                          "phase": row["phase"], "request_id": row["request_id"]})
    return recovered


def learning_diagnostics(trace, fit_cases, categorical):
    """Separate native category sensitivity from gradients reaching the actual logits."""
    if not trace:
        raise ValueError("learning diagnostics require committed updates")

    def statistics(values):
        values = np.asarray(values, dtype=np.float64)
        if not len(values) or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("gradient diagnostics require finite nonnegative measurements")
        return {"min": float(values.min()), "median": float(np.median(values)), "mean": float(values.mean()),
                "max": float(values.max()), "exact_zero_steps": int((values == 0).sum())}

    groups = {}
    for index, category in enumerate(categorical):
        gradients = [row["gradients"]["category_l1"][index] for row in trace]
        spans = [max(row["estimates"][0]["categorical"][index]) - min(row["estimates"][0]["categorical"][index]) for row in trace]
        samples = [row["estimates"][0]["anchor"]["categorical"][index] for row in trace]
        groups[category["name"]] = {"score_gradient_l1": statistics(gradients), "native_loss_span": statistics(spans),
            "steps_with_native_response_but_exact_zero_score_gradient": sum(span > 0 and gradient == 0 for span, gradient in zip(spans, gradients, strict=True)),
            "sampled_choices": {str(value): samples.count(choice) for choice, value in enumerate(category["values"])}}
    selected = [case for case in fit_cases if case["view"] == 0]
    if len({case["state_id"] for case in selected}) != len(selected):
        raise ValueError("duplicate training states in export diagnostics")
    exports = {}
    for method in ("initial", "parameter", "spectral"):
        exports[method] = {}
        for index, category in enumerate(categorical):
            choices = [case["rows"][method]["candidate"]["categorical"][index] for case in selected]
            exports[method][category["name"]] = {
                "counts": {str(value): choices.count(choice) for choice, value in enumerate(category["values"])},
                "changed_from_initial": sum(choice != case["rows"]["initial"]["candidate"]["categorical"][index]
                                            for choice, case in zip(choices, selected, strict=True))}
    return {"committed_updates": len(trace), "training_states": len(selected),
            "continuous_output_gradient_l1": statistics([row["gradients"]["continuous_l1"] for row in trace]),
            "categorical": groups, "hard_export_category_usage": exports,
            "claim": "learning_signal_and_category_usage_diagnostics_not_a_quality_gate"}


def paired_comparison(cases, *, resamples=2000):
    """Resample whole source states, retaining their paired reference/transfer observations."""
    by_state = {}
    for case in cases:
        views = by_state.setdefault(case["state_id"], {})
        key = case.get("case_id", case["state_id"])
        observations = views.setdefault(case["view"], {})
        if key in observations or case["view"] not in (0, 1):
            raise ValueError("duplicate or unsupported comparison view")
        observations[key] = case
    if not by_state or any(set(views) != {0, 1} or set(views[0]) != set(views[1]) for views in by_state.values()):
        raise ValueError("paired comparison requires both views for every source state")
    states = sorted(by_state)
    rng = np.random.default_rng(0)
    selections = rng.integers(0, len(states), size=(resamples, len(states)))
    groups = {}
    for view, role in ((0, "reference"), (1, "transfer")):
        selected = [list(by_state[state][view].values()) for state in states]
        comparisons = {}
        for baseline in ("initial", "parameter"):
            candidate = np.array([np.mean([row["rows"]["spectral"]["scores"]["mrstft"] for row in rows]) for rows in selected])
            control = np.array([np.mean([row["rows"][baseline]["scores"]["mrstft"] for row in rows]) for rows in selected])
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


def development_comparison(cases, panels=None):
    if not panels:
        return paired_comparison(cases)
    return {"panels": {name: paired_comparison([row for row in cases if row.get("case_id", row["state_id"]) in ids])
                       for name, ids in panels.items()}, "panels_share_source_states": True}


def public_cases(cases):
    return [{"state_id": case["state_id"], **({"case_id": case["case_id"]} if "case_id" in case else {}),
             "view": case["view"], "repeat_exact": case["repeat_exact"],
             "source_audio_sha256": case["source_receipt"]["qa"]["audio_float32_sha256"],
             "methods": {name: {"scores": row["scores"], "candidate": row["candidate"],
                                 "request_id": row["receipt"]["request_id"],
                                 "audio_sha256": row["receipt"]["qa"]["audio_float32_sha256"]}
                         for name, row in case["rows"].items()}} for case in cases]

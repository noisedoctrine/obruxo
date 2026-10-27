"""Bounded sound-only fitting of one basic patch, without an inverse network."""

import copy
import math

import torch

from .codec import PilotTarget

SPEC = {"updates": 32, "category_rounds": [0, 8, 16, 24], "probe_step": .025, "learning_rate": .03,
        "move_cap": .025, "trial_scales": [1., .5, .25], "minimum_improvement": 1e-6, "refusal_limit": 8,
        "optimizer": "Adam", "betas": [.9, .999], "epsilon": 1e-8, "inactive_stored_values": "preserve",
        "refused_move": "restore_controls_keep_advanced_Adam", "category_ties": "prefer_current_then_lexicographic"}


class DirectPatch:
    def __init__(self, codec, initial):
        codec.validate_target(initial)
        self.codec = codec
        self.controls = torch.nn.Parameter(torch.tensor(initial.continuous, dtype=torch.float32))
        self.categories = initial.categorical
        self.optimizer = torch.optim.Adam([self.controls], lr=SPEC["learning_rate"], betas=tuple(SPEC["betas"]), eps=SPEC["epsilon"])
        self.refusals = 0

    def target(self, controls=None, categories=None):
        values = self.controls.detach() if controls is None else controls
        return self.codec.canonical_target(PilotTarget(tuple(values.tolist()), self.categories if categories is None else categories))

    def snapshot(self):
        return {"controls": self.controls.detach().clone(), "categories": list(self.categories),
                "optimizer": copy.deepcopy(self.optimizer.state_dict()), "refusals": self.refusals}

    def restore(self, saved):
        self.codec.validate_target(PilotTarget(tuple(saved["controls"].tolist()), tuple(saved["categories"])))
        with torch.no_grad():
            self.controls.copy_(saved["controls"])
        self.categories = tuple(saved["categories"])
        self.optimizer.load_state_dict(copy.deepcopy(saved["optimizer"]))
        self.refusals = saved["refusals"]


def finite_cost(evaluate, candidate, role):
    value = evaluate(candidate, role)
    if value is None and role == "trial":
        return None
    if value is None or not math.isfinite(value) or value < 0:
        raise ValueError("direct gradient/category/anchor costs must be valid, finite and nonnegative")
    return float(value)


def category_round(patch, evaluate):
    before = patch.categories
    current = finite_cost(evaluate, patch.target(), "anchor")
    choices = [before] + [(voices, filtered) for voices in range(3) for filtered in range(2) if (voices, filtered) != before]
    rows = [{"categories": list(choice), "loss": finite_cost(evaluate, patch.target(categories=choice), "category")} for choice in choices]
    best = min(rows, key=lambda row: row["loss"])
    accepted = best["loss"] < current - SPEC["minimum_improvement"]
    if accepted:
        patch.categories = tuple(best["categories"])
        patch.refusals = 0
    return {"before": list(before), "after": list(patch.categories), "accepted": accepted, "rows": rows}


def continuous_step(patch, evaluate):
    before = patch.controls.detach().clone()
    anchor = patch.target()
    baseline = finite_cost(evaluate, anchor, "anchor")
    active = patch.codec.continuous_mask(anchor)
    gradient, probes = [], []
    for index, enabled in enumerate(active):
        if not enabled:
            gradient.append(0.)
            continue
        lower, upper = before.clone(), before.clone()
        lower[index] = max(0., float(before[index]) - SPEC["probe_step"])
        upper[index] = min(1., float(before[index]) + SPEC["probe_step"])
        low = finite_cost(evaluate, patch.target(lower), "gradient")
        high = finite_cost(evaluate, patch.target(upper), "gradient")
        span = float(upper[index]) - float(lower[index])
        gradient.append((high - low) / span)
        probes.append({"control": index, "lower": float(lower[index]), "upper": float(upper[index]), "losses": [low, high]})
    patch.optimizer.zero_grad(set_to_none=True)
    patch.controls.grad = torch.tensor(gradient, dtype=torch.float32)
    if not torch.isfinite(patch.controls.grad).all():
        raise ValueError("nonfinite direct finite-difference gradient")
    patch.optimizer.step()
    with torch.no_grad():
        move = (patch.controls - before).clamp(-SPEC["move_cap"], SPEC["move_cap"])
        move.masked_fill_(~torch.tensor(active), 0.)
        patch.controls.copy_(before)
    trials, accepted, committed = [], False, baseline
    for scale in SPEC["trial_scales"]:
        proposal = (before + scale * move).clamp(0., 1.)
        over_cap = (proposal.double() - before.double()).abs() > SPEC["move_cap"]
        proposal = torch.where(over_cap, torch.nextafter(proposal, before), proposal)
        value = finite_cost(evaluate, patch.target(proposal), "trial")
        improves = value is not None and value < baseline - SPEC["minimum_improvement"]
        trials.append({"scale": scale, "candidate": patch.target(proposal).to_dict(), "loss": value,
                       "status": "invalid_audio" if value is None else "accepted" if improves else "sound_refused"})
        if improves:
            with torch.no_grad():
                patch.controls.copy_(proposal)
            accepted, committed = True, value
            break
    patch.refusals = 0 if accepted else patch.refusals + 1
    return {"anchor": anchor.to_dict(), "before_loss": baseline, "after_loss": committed, "gradient": gradient,
            "probes": probes, "trials": trials, "accepted": accepted, "maximum_move": float((patch.controls.detach() - before).abs().max()),
            "stored_controls": patch.controls.detach().tolist(), "categories": list(patch.categories), "refusals": patch.refusals}


def step(patch, index, evaluate):
    if type(index) is not int or not 0 <= index < SPEC["updates"] or patch.refusals >= SPEC["refusal_limit"]:
        raise ValueError("direct update exceeds its frozen step or refusal allocation")
    categories = category_round(patch, evaluate) if index in SPEC["category_rounds"] else None
    return {"step": index, "category_round": categories, "continuous": continuous_step(patch, evaluate)}


def quality(initial, final, parameter):
    if len(initial) != 2 or len(final) != 2 or len(parameter) != 2:
        raise ValueError("direct quality requires both declared performances")
    def average(rows):
        return sum(row["spectral"] for row in rows) / 2
    gates = {"joint_at_least_25_percent_better": average(final) <= .75 * average(initial),
             "joint_no_worse_than_parameter": average(final) <= average(parameter),
             "each_performance_at_least_10_percent_better": all(b["spectral"] <= .9 * a["spectral"] for a, b in zip(initial, final, strict=True)),
             "mrstft_improves_both": all(b["mrstft"] < a["mrstft"] for a, b in zip(initial, final, strict=True))}
    return {"initial_joint": average(initial), "final_joint": average(final), "parameter_joint": average(parameter), "gates": gates, "pass": all(gates.values())}

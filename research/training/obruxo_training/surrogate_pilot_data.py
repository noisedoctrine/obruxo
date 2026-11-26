"""Freeze matched patch families, exact MIDI, and bounded control perturbations."""

from collections import defaultdict
from itertools import chain
import json

from obruxo_data.hashing import canonical_sha256

from .codec import PilotTarget
from .pilot import seeded_random
from .temporal_proxy_data import training_performance


def _ordered_roots(pool, split, excluded):
    assignments = {row["state_id"]: row for row in pool["splits"]["assignments"]}
    groups = defaultdict(list)
    for row in pool["entries"]:
        state = row["state_id"]
        assignment = assignments[state]
        if assignment["split"] == split and assignment["lineage_id"] == state and state not in excluded:
            groups[tuple(row["target"]["categorical"])].append(row)
    groups = [sorted(rows, key=lambda row: canonical_sha256(("surrogate-pilot-v1", split, row["state_id"])))
              for _, rows in sorted(groups.items())]
    return [rows[i] for i in range(max(map(len, groups))) for rows in groups if i < len(rows)]


def _expand_roots(rows, codec, wet_count, salt):
    permutations = []
    for field, choices in codec.categorical[2:]:
        values = ([1] * wet_count + [0] * (len(rows) - wet_count) if field == "reverb_on"
                  else [i % len(choices) for i in range(len(rows))])
        seeded_random(salt, field).shuffle(values)
        permutations.append(values)
    result = []
    for index, row in enumerate(rows):
        original = PilotTarget.from_dict(row["target"])
        rng = seeded_random(salt, row["state_id"])
        target = codec.canonical(PilotTarget((*original.continuous, *(rng.uniform(.15, .85) for _ in range(4))),
                                             (*original.categorical, *(values[index] for values in permutations))))
        result.append({**row, "target": target.to_dict()})
    return result


def _midi_banks(excluded):
    banks, seen = {}, set(excluded)
    for role, seed in (("seen", 172), ("new", 173)):
        bank = []
        for style in range(4):
            variants = []
            candidate = 0
            while len(variants) < 4:
                perf = training_performance(seed, candidate, style).to_dict()
                identity = canonical_sha256(perf)
                if identity not in seen:
                    seen.add(identity)
                    variants.append({"midi_id": identity, "performance": perf})
                candidate += 1
            bank.append(variants)
        banks[role] = bank
    return banks


def perturb(codec, target, pair_index, root):
    active, categorical = codec.masks(target.categorical)
    count = len(active) + len(categorical)
    for offset in range(count):
        index = (pair_index + offset) % count
        values, choices = list(target.continuous), list(target.categorical)
        if index < len(active) and active[index]:
            step = .10 if seeded_random(root, pair_index).randrange(2) else -.10
            values[index] += step if 0 <= values[index] + step <= 1 else -step
            return codec.canonical(PilotTarget(tuple(values), tuple(choices))), codec.continuous[index][0]
        if index >= len(active) and categorical[index - len(active)]:
            position = index - len(active)
            name, options = codec.categorical[position]
            # Keep the declared wet proportion on both members of the training pairs.
            if name == "reverb_on":
                continue
            choices[position] = (choices[position] + 1 + (pair_index // count) % (len(options) - 1)) % len(options)
            return codec.canonical(PilotTarget(tuple(values), tuple(choices))), name
    raise ValueError("patch has no active control available for perturbation")


def make_cohorts(pool_path, reservation_path, basic, expanded, *, smoke=False):
    source = json.loads(pool_path.read_text(encoding="utf-8"))
    pool = source["pool"]
    if canonical_sha256(pool) != source["pool_id"] or pool["codec"] != basic.basic.to_dict():
        raise ValueError("basic source pool identity or codec differs")
    reservation = json.loads(reservation_path.read_text(encoding="utf-8"))
    excluded = {row["lineage_id"] for row in reservation["cases"]}
    per_stage, evaluation_count = (4, 4) if smoke else (256, 64)
    train = _ordered_roots(pool, "train", excluded)[:2 * per_stage]
    evaluation = _ordered_roots(pool, "validation", excluded)[:evaluation_count]
    if len(train) != 2 * per_stage or len(evaluation) != evaluation_count:
        raise ValueError("insufficient eligible independent patch roots")
    if len({row["state_id"] for row in chain(train, evaluation)}) != len(train) + len(evaluation):
        raise ValueError("patch lineage crosses pilot splits")
    reserved_midi = {canonical_sha256(row["performance"]) for row in reservation["performances"]}
    midi = _midi_banks(reserved_midi)
    stages = {}
    for label, codec, roots, eval_roots in (
            ("basic", basic, train[:per_stage], evaluation),
            ("expanded", expanded, _expand_roots(train[per_stage:], expanded, per_stage // 2, "training"),
             _expand_roots(evaluation, expanded, evaluation_count // 4, "evaluation"))):
        pairs, cases = [], []
        for index, row in enumerate(roots):
            original = PilotTarget.from_dict(row["target"])
            for view in range(4):
                pair_index = index * 4 + view
                changed, control = perturb(codec, original, pair_index, row["state_id"])
                pairs.append({**row, **midi["seen"][view][(index // 6 + view) % 4], "changed": changed.to_dict(),
                              "pair_id": f"{label}-{pair_index:04d}", "control": control})
        seen_ids = {row["midi_id"] for row in pairs}
        eval_banks = {"seen": [[item for item in style if item["midi_id"] in seen_ids] for style in midi["seen"]],
                      "new": midi["new"]}
        for index, row in enumerate(eval_roots):
            for role, bank in eval_banks.items():
                for view in range(2):
                    style = (index // 6 + 2 * view + (role == "new")) % 4
                    variant = (index // 24 + index % 6 + view) % len(bank[style])
                    cases.append({**row, **bank[style][variant], "midi_role": role,
                                  "case_id": f"{label}-{len(cases):04d}"})
        stages[label] = {"pairs": pairs, "cases": cases, "codec": codec.spec()}
    return {"pool_id": source["pool_id"], "reservation_id": reservation["reservation_id"], "midi": midi, "stages": stages}

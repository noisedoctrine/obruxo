"""Category-independent performances for the next matched basic sound fit."""

from collections import defaultdict

from obruxo_data.hashing import canonical_sha256

from .codec import PilotCodec
from .features import tensor_digest
from .input_performances import PATTERNS, performance_for
from .pilot import seeded_random


def groups_for(pool, entries, split):
    if type(pool.codec) is not PilotCodec or split not in {"train", "validation"}:
        raise ValueError("balanced sound records require the exact basic codec and reserved test split")
    known = {entry.state_id: entry.target for entry in pool.entries}
    if len({entry.state_id for entry in entries}) != len(entries):
        raise ValueError("balanced sound records need unique states")
    groups = defaultdict(list)
    for entry in entries:
        if known.get(entry.state_id) != entry.target or pool.manifest.by_state[entry.state_id].split != split:
            raise ValueError("balanced sound record contains a changed or wrong-split source")
        groups[entry.target.categorical].append(entry.state_id)
    return groups


def shuffled_groups(groups, seed, purpose):
    for category, values in sorted(groups.items()):
        states = sorted(values)
        seeded_random("basic_balanced_sound_performances_v1", seed, purpose, category).shuffle(states)
        yield category, states


def training_records(pool, entries, *, seed=0):
    modes = {}
    for _category, states in shuffled_groups(groups_for(pool, entries, "train"), seed, "training"):
        modes.update({state: 1 + index % 3 for index, state in enumerate(states)})
    return [{"state_id": entry.state_id, "target": entry.target.to_dict(),
             "performances": [performance_for(seed, entry.state_id, 1, view, mode).to_dict() for view, mode in enumerate((0, modes[entry.state_id]))],
             "input_patterns": ["held", PATTERNS[modes[entry.state_id]]]} for entry in entries]


def counterbalanced_validation_records(pool, entries, *, seed=0):
    groups = groups_for(pool, entries, "validation")
    if len(groups) != 6 or any(len(states) != 2 for states in groups.values()):
        raise ValueError("counterbalanced development requires two states in each of six categories")
    modes = {}
    for _category, states in shuffled_groups(groups, seed, "counterbalanced_development"):
        modes.update({state: (2 * index, 2 * index + 1) for index, state in enumerate(states)})
    records = []
    for entry in entries:
        for case, mode in enumerate(modes[entry.state_id]):
            pair = (mode, (mode + 2) % 4)
            records.append({"state_id": entry.state_id, "target": entry.target.to_dict(), "counterbalance_case": case,
                            "performances": [performance_for(seed, entry.state_id, 2, 2 * case + view, pattern).to_dict()
                                             for view, pattern in enumerate(pair)], "input_patterns": [PATTERNS[pattern] for pattern in pair]})
    return records


def cached_training_records(pool, entries, cache, sampler):
    """Use the actual audited original/held/alternate recordings, excluding duplicate replay weights."""
    groups_for(pool, entries, "train")
    selected = defaultdict(dict)
    for item in cache["replay"]:
        recipe = sampler.recipe_at(item["sequence_id"])
        if recipe.input_view in (0, 1, 3):
            selected[item["state_id"]][recipe.input_view] = (recipe, item)
    records = []
    for entry in entries:
        rows = [selected[entry.state_id][view] for view in (0, 1, 3)]
        if any(item["target"] != entry.target.to_dict() or item["recipe_id"] != recipe.recipe_id for recipe, item in rows):
            raise ValueError("sound training cache differs from its source recipes")
        records.append({"state_id": entry.state_id, "target": entry.target.to_dict(), "input_views": "all",
                        "performances": [recipe.render_request().performance.to_dict() for recipe, _ in rows],
                        "source_recipe_ids": [recipe.recipe_id for recipe, _ in rows],
                        "source_audio_sha256": [item["audio_sha256"] for _, item in rows],
                        "source_feature_sha256": [tensor_digest(item["features"]) for _, item in rows]})
    return records


def cross_view_updates(count, *, seed=0):
    import torch
    rng = torch.Generator().manual_seed(seed)
    return [{"state": int(index), "input_view": source, "view": target} for source, target in ((0, 1), (1, 0), (0, 2), (2, 0))
            for index in torch.randperm(count, generator=rng)]


def development_panels(pool, entries):
    from .performances import initial_bank
    original = [{"state_id": entry.state_id, "target": entry.target.to_dict(),
                 "performances": [initial_bank()[offset + index % 4].performance().to_dict() for offset in (0, 4)]}
                for index, entry in enumerate(entries)]
    added = counterbalanced_validation_records(pool, entries)
    for recipe in added:
        recipe["case_id"] = canonical_sha256({"state_id": recipe["state_id"], "case": recipe["counterbalance_case"], "panel": "counterbalanced"})
    return original + added, {"original": [row["state_id"] for row in original], "counterbalanced": [row["case_id"] for row in added]}

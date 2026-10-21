"""Expand the basic training subset using training targets alone."""

from collections import Counter
from itertools import product
from math import fsum, sqrt

from .codec import PilotCodec


def expand_training(pool, original):
    if type(pool.codec) is not PilotCodec or len(original) != 32 or len({entry.state_id for entry in original}) != 32:
        raise ValueError("coverage expansion requires the original 32 basic states")
    if any(pool.manifest.by_state[entry.state_id].split != "train" for entry in original):
        raise ValueError("coverage selection must use training states only")
    categories = list(product(*(range(len(options)) for _name, options in pool.codec.categorical)))
    quotas = {category: 64 // len(categories) + int(index < 64 % len(categories)) for index, category in enumerate(categories)}
    original_ids = {entry.state_id for entry in original}
    remaining = [entry for entry in pool.entries if pool.manifest.by_state[entry.state_id].split == "train" and entry.state_id not in original_ids]
    selected, additions = list(original), []
    counts = Counter(entry.target.categorical for entry in selected)
    if any(counts[category] > quota for category, quota in quotas.items()):
        raise ValueError("original subset exceeds balanced 64-state category quotas")
    while len(selected) < 64:
        category = min(categories, key=lambda key: (counts[key] - quotas[key], key))
        candidates = [entry for entry in remaining if entry.target.categorical == category]
        if not candidates:
            raise ValueError("not enough training states to fill coverage quotas")
        references = [entry for entry in selected if entry.target.categorical == category]
        if not references:
            raise ValueError("original subset must include every basic category")
        mask = pool.codec.continuous_mask(references[0].target)
        indices = [index for index, enabled in enumerate(mask) if enabled]

        def nearest(entry):
            return min((fsum((entry.target.continuous[index] - reference.target.continuous[index]) ** 2 for index in indices) / len(indices),
                        reference.state_id) for reference in references)

        distances = [(nearest(entry), entry) for entry in candidates]
        (squared, neighbor), chosen = min(distances, key=lambda row: (-row[0][0], row[1].state_id))
        selected.append(chosen)
        remaining.remove(chosen)
        counts[category] += 1
        additions.append({"state_id": chosen.state_id, "category_indices": list(category),
                          "nearest_selected_state_id": neighbor, "normalized_control_rmse": sqrt(squared)})
    return selected, {"rule": "retain_32_then_category_constrained_maximin_active_controls_v1",
        "source_split": "train", "development_data_used": False, "original_state_ids": [entry.state_id for entry in original],
        "category_quotas": [{"indices": list(category), "count": quotas[category]} for category in categories],
        "distance": "same_category_source_active_normalized_control_rmse", "tie_break": "lexicographic_state_id",
        "additions": additions}

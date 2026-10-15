"""Training-only paired views and a bidirectional identity contrastive objective."""

import torch
from torch.nn import functional as F


def complete_pairs(items):
    """Group complete scheduled pairs by state; exclude pending or QA-broken pairs."""
    by_sequence = {item["sequence_id"]: index for index, item in enumerate(items)}
    if len(by_sequence) != len(items):
        raise ValueError("paired replay has duplicate sequences")
    groups = {}
    for sequence, left in sorted(by_sequence.items()):
        if sequence % 2 or sequence + 1 not in by_sequence:
            continue
        right = by_sequence[sequence + 1]
        first, second = items[left], items[right]
        if first["state_id"] != second["state_id"] or first["performance_id"] == second["performance_id"]:
            raise ValueError("paired views need the same state and distinct performances")
        groups.setdefault(first["state_id"], []).append((left, right))
    return groups


def paired_batch_indices(items, batch_size, generator):
    groups = list(complete_pairs(items).values())
    if len(groups) < batch_size // 2:
        raise ValueError("paired replay has insufficient distinct states for the configured batch")
    selected = torch.randperm(len(groups), generator=generator)[:batch_size // 2].tolist()
    return [index for group in selected for index in groups[group][int(torch.randint(len(groups[group]), (), generator=generator))]]


def identity_terms(representation, temperature=.1):
    """Adjacent positive pairs; all other states in the batch are cross-view negatives."""
    if representation.ndim != 2 or len(representation) < 4 or len(representation) % 2:
        raise ValueError("identity loss needs at least two complete state pairs")
    normalized = F.normalize(representation, dim=1)
    similarities = normalized[::2] @ normalized[1::2].T
    labels = torch.arange(len(similarities), device=representation.device)
    logits = similarities / temperature
    loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
    accuracy = ((logits.argmax(1) == labels).float().mean() + (logits.argmax(0) == labels).float().mean()) / 2
    different = ~torch.eye(len(similarities), dtype=torch.bool, device=representation.device)
    return {"identity_loss": loss, "identity_accuracy": accuracy, "identity_positive_cosine": similarities.diag().mean(),
            "identity_negative_cosine": similarities[different].mean()}


def representation_statistics(rows, names):
    """Cross-performance retrieval over held-out states, with failure denominators."""
    result = {}
    for name in names:
        complete = [row["representations"][name] for row in rows if "reference" in row.get("representations", {}).get(name, {})
                    and "transfer" in row["representations"][name]]
        count = len(complete)
        summary = {"planned": len(rows), "scored": count, "failed_or_unscorable": len(rows) - count}
        if count:
            left, right = [F.normalize(torch.tensor([item[role] for item in complete], dtype=torch.float32), dim=1)
                           for role in ("reference", "transfer")]
            similarities = left @ right.T
            labels = torch.arange(count)
            summary.update(matched_cosine=float(similarities.diag().mean()), chance_top1=1 / count,
                           bidirectional_top1=float(((similarities.argmax(1) == labels).float().mean()
                                                     + (similarities.argmax(0) == labels).float().mean()) / 2),
                           normalized_spread=float(torch.cat((left, right)).std(dim=0, correction=0).mean()))
            if count >= 2:
                summary["different_state_cosine"] = float(similarities[~torch.eye(count, dtype=torch.bool)].mean())
                summary["cosine_margin"] = summary["matched_cosine"] - summary["different_state_cosine"]
        result[name] = summary
    return {"protocol": "heldout_state_reference_to_reserved_transfer_retrieval_v1", "models": result,
            "limit": "descriptive_state_identity_and_collapse_diagnostics_not_model_selection"}

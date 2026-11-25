"""Score all 256 V7 patch-token positions on the frozen Issue 167 galleries."""

from pathlib import Path

import numpy as np

from frozen_retrieval.core import mixed_gallery_midi_id
from run_issue167_v7 import RUN as ISSUE167_RUN, read_json, read_rows, write_json
from run_issue169_tokens import RUN, SHARD_SIZE, cohort, status


ROOT = Path(__file__).resolve().parent
FEATURES = RUN / "patch-token-sweep"
PATCHES = 256
DIMENSION = 384
BATCH_SIZE = 4
REPLICATES = 2000
METRICS = ("recall_at_1", "recall_at_5", "recall_at_10")


def token_store(receipts):
    cache = RUN / "tokens" / read_json(RUN / "token-cache.json")["cache_id"]
    request_ids = sorted(receipts)
    identity = read_json(cache / "identity.json")
    if identity["rows"] != [[key, receipts[key]["audio_sha256"]] for key in request_ids]:
        raise ValueError("Token cache does not match the verified Issue 167 WAVs")
    FEATURES.mkdir(parents=True, exist_ok=True)
    store_path = FEATURES / "tokens.npy"
    if store_path.exists():
        array = np.load(store_path, mmap_mode="r+", allow_pickle=False)
        if array.shape != (len(request_ids), PATCHES, DIMENSION):
            raise ValueError("Cached patch token store has the wrong shape")
        return request_ids, array
    array = np.lib.format.open_memmap(store_path, mode="w+", dtype=np.float32,
                                      shape=(len(request_ids), PATCHES, DIMENSION))
    for start in range(0, len(request_ids), SHARD_SIZE):
        end = min(start + SHARD_SIZE, len(request_ids))
        shard = np.load(cache / f"{start:05d}.npy", mmap_mode="r", allow_pickle=False)
        array[start:end] = shard[:, 4:]
        status("Prepare patch token store", end, len(request_ids))
    array.flush()
    return request_ids, array


def episode_indices(condition, episodes, candidates, requests, request_by_id):
    rows = [row for row in episodes if row["condition"] == condition]
    query_ids, gallery_ids, targets = [], [], []
    for row in rows:
        candidate_set = candidates[row["candidate_set_id"]]
        states = candidate_set["state_ids"]
        query_ids.append(requests[(row["target_state_id"], row["query_midi_id"])])
        if condition == "cross_midi_common_gallery":
            midi_ids = [row["gallery_midi_id"]] * len(states)
        else:
            midi_ids = [mixed_gallery_midi_id(row, state_id, candidates) for state_id in states]
        gallery_ids.append([requests[(state_id, midi)] for state_id, midi in zip(states, midi_ids, strict=True)])
        targets.append(states.index(row["target_state_id"]))
    return rows, np.array([request_by_id[key] for key in query_ids]), np.array(
        [[request_by_id[key] for key in row] for row in gallery_ids]), np.array(targets)


def rank_batch(scores, targets):
    # scores are [batch, candidate, token]; exactly tied candidates share rank credit.
    batch = scores.shape[0]
    rows = np.arange(batch)
    target_scores = scores[rows, targets, :]
    higher = (scores > target_scores[:, None, :] + 1e-12).sum(axis=1)
    tied = (np.abs(scores - target_scores[:, None, :]) <= 1e-12).sum(axis=1)
    result = {}
    for k in (1, 5, 10):
        result[f"recall_at_{k}"] = np.minimum(np.maximum(k - higher, 0), tied) / tied
    return result


def score(condition, rows, query_indices, gallery_indices, targets, tokens):
    import torch

    if not torch.xpu.is_available():
        raise RuntimeError("The pinned Intel XPU is unavailable for the 256-position score pass")
    device = torch.device("xpu")
    outputs = {metric: np.empty((len(rows), PATCHES), dtype=np.float32) for metric in METRICS}
    status(f"Score {condition}", 0, len(rows))
    for start in range(0, len(rows), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(rows))
        query = torch.from_numpy(np.array(tokens[query_indices[start:end]])).to(device)
        gallery = torch.from_numpy(np.array(tokens[gallery_indices[start:end]])).to(device)
        query = query / query.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        gallery = gallery / gallery.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        similarities = (gallery * query[:, None, :, :]).sum(dim=-1).cpu().numpy()
        metrics = rank_batch(similarities, targets[start:end])
        for key, value in metrics.items():
            outputs[key][start:end] = value
        if end % 200 == 0 or end == len(rows):
            status(f"Score {condition}", end, len(rows))
    return outputs


def clustered_intervals(rows, values, baseline):
    families = sorted({row["target_family_id"] for row in rows})
    performances = sorted({row["query_midi_id"] for row in rows})
    if len(rows) != len(families) * len(performances):
        raise ValueError("Patch token results do not form a complete family by performance grid")
    f_index = {key: index for index, key in enumerate(families)}
    p_index = {key: index for index, key in enumerate(performances)}
    matrices = np.full((PATCHES, len(families), len(performances)), np.nan, dtype=np.float32)
    base = np.full((len(families), len(performances)), np.nan, dtype=np.float32)
    for index, row in enumerate(rows):
        f, p = f_index[row["target_family_id"]], p_index[row["query_midi_id"]]
        matrices[:, f, p] = values[index]
        base[f, p] = baseline[index]
    if not np.isfinite(matrices).all() or not np.isfinite(base).all():
        raise ValueError("Patch token results contain incomplete bootstrap cells")
    rng = np.random.default_rng(152)
    family_counts = np.stack([np.bincount(rng.integers(len(families), size=len(families)), minlength=len(families))
                              for _ in range(REPLICATES)]).astype(np.float32)
    performance_counts = np.stack([np.bincount(rng.integers(len(performances), size=len(performances)), minlength=len(performances))
                                   for _ in range(REPLICATES)]).astype(np.float32)
    draws = np.empty((REPLICATES, PATCHES), dtype=np.float32)
    delta = matrices - base[None]
    for replicate, (fw, pw) in enumerate(zip(family_counts, performance_counts, strict=True)):
        draws[replicate] = np.einsum("f,pfm,m->p", fw, matrices, pw, optimize=True) / (len(families) * len(performances))
    delta_draws = np.empty_like(draws)
    for replicate, (fw, pw) in enumerate(zip(family_counts, performance_counts, strict=True)):
        delta_draws[replicate] = np.einsum("f,pfm,m->p", fw, delta, pw, optimize=True) / (len(families) * len(performances))
    return {"estimate": matrices.mean(axis=(1, 2)).tolist(),
            "lower_95": np.quantile(draws, 0.025, axis=0).tolist(),
            "upper_95": np.quantile(draws, 0.975, axis=0).tolist(),
            "delta_vs_original": delta.mean(axis=(1, 2)).tolist(),
            "delta_lower_95": np.quantile(delta_draws, 0.025, axis=0).tolist(),
            "delta_upper_95": np.quantile(delta_draws, 0.975, axis=0).tolist(),
            "family_clusters": len(families), "performance_clusters": len(performances), "replicates": REPLICATES}


def publish(result):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = ROOT / "reports" / "VITAL_JEPA_V7_PATCH_TOKEN_SWEEP.md"
    figure = ROOT / "reports" / "figures" / "issue169_v7_patch_tokens.png"
    results_path = ROOT / "reports" / "issue169_v7_patch_tokens.json"
    common, mixed = result["cross_midi_common_gallery"], result["cross_midi_mixed_gallery"]
    best = int(np.argmax(common["recall_at_1"]["estimate"]))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for ax, data, title, field, cmap, vmin, vmax in (
        (axes[0], common["recall_at_1"]["estimate"], "Common gallery R@1", "recall", "viridis", 0, 0.45),
        (axes[1], mixed["recall_at_1"]["estimate"], "Mixed gallery R@1", "recall", "viridis", 0, 0.45),
        (axes[2], common["recall_at_1"]["delta_vs_original"], "Common R@1 change vs V7", "delta", "coolwarm", -0.15, 0.15),
    ):
        matrix = np.asarray(data).reshape(8, 32)
        image = ax.imshow(matrix, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set(title=title, xlabel="Time patch", ylabel="Mel patch")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes:
        ax.scatter(best % 32, best // 32, marker="x", s=55, color="white", linewidths=1.5)
    fig.suptitle("Vital JEPA V7 patch token positions for preset retrieval", fontsize=14, fontweight="bold")
    fig.savefig(figure, dpi=160)
    plt.close(fig)

    indices = sorted(range(PATCHES), key=lambda i: common["recall_at_1"]["estimate"][i], reverse=True)
    lines = ["# Vital JEPA V7 individual patch tokens", "",
             f"The strongest patch position was **mel {best // 32}, time {best % 32}**. Its common-gallery Recall@1 was "
             f"**{common['recall_at_1']['estimate'][best]:.1%}** (95% interval "
             f"{common['recall_at_1']['lower_95'][best]:.1%} to {common['recall_at_1']['upper_95'][best]:.1%}), "
             f"a paired change of {common['recall_at_1']['delta_vs_original'][best] * 100:+.1f} percentage points "
             f"({common['recall_at_1']['delta_lower_95'][best] * 100:+.1f} to "
             f"{common['recall_at_1']['delta_upper_95'][best] * 100:+.1f}).", "",
             "Each of the 256 patch positions is evaluated independently as a 384-D vector, comparing the same grid position across query and gallery recordings. "
             "The 6,450 queries, 129 candidates, and two gallery assignments are reused from Issue 167.", "",
             "![Patch token retrieval heatmaps](figures/issue169_v7_patch_tokens.png)", "",
             "## Highest common-gallery positions", "",
             "| Rank | Mel patch | Time patch | R@1 | R@5 | R@10 | Mixed R@1 | Paired common R@1 change vs original |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for rank, idx in enumerate(indices[:12], 1):
        vals = [common[f"recall_at_{k}"]["estimate"][idx] for k in (1, 5, 10)]
        delta = common["recall_at_1"]
        lines.append(f"| {rank} | {idx // 32} | {idx % 32} | {vals[0]:.1%} | {vals[1]:.1%} | {vals[2]:.1%} | "
                     f"{mixed['recall_at_1']['estimate'][idx]:.1%} | {delta['delta_vs_original'][idx] * 100:+.1f} "
                     f"[{delta['delta_lower_95'][idx] * 100:+.1f}, {delta['delta_upper_95'][idx] * 100:+.1f}] pp |")
    lines.extend(["", "## Limits", "",
                  "A position indexes a frequency by time patch in the model's flattened 8 by 32 grid. All tokens have attended to the full input through the transformer, so an individual position does not isolate only its local spectrogram patch. "
                  "The score measures preset retrieval on this cohort, not listener-rated timbre similarity. Family and performance cluster intervals describe this cohort; possible training-set parent overlap remains as reported in Issue 167.", "",
                  "[Full per-position results](issue169_v7_patch_tokens.json) · [Representation sweep](VITAL_JEPA_V7_REPRESENTATION_SWEEP.md)", ""])
    report.write_text("\n".join(lines), encoding="utf-8")
    write_json(results_path, result)


def main():
    episodes, candidates, requests, receipts = cohort()
    request_ids, tokens = token_store(receipts)
    request_by_id = {request: index for index, request in enumerate(request_ids)}
    original_rows = read_rows(ISSUE167_RUN / "episode-ranks.private.jsonl")
    baseline = {condition: {metric: {row["episode_id"]: row[metric] for row in original_rows
                                     if row["condition"] == condition and row["arm"] == "vital_jepa_v7:clip"}
                            for metric in METRICS}
                for condition in ("cross_midi_common_gallery", "cross_midi_mixed_gallery")}
    result = {}
    for condition in baseline:
        rows, query_indices, gallery_indices, targets = episode_indices(condition, episodes, candidates, requests, request_by_id)
        scores = score(condition, rows, query_indices, gallery_indices, targets, tokens)
        result[condition] = {}
        for metric, matrix in scores.items():
            base = np.array([baseline[condition][metric][row["episode_id"]] for row in rows], dtype=np.float32)
            result[condition][metric] = clustered_intervals(rows, matrix, base)
    publish(result)
    status("Patch token sweep ready", 1, 1, state="complete")


if __name__ == "__main__":
    main()

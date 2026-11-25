"""Choose V7 patch-token retrieval settings on training performances and score held-out performances."""

import argparse
import json
import random
from pathlib import Path

import numpy as np

from frozen_retrieval.evaluation import _evaluate_condition
from run_issue167_v7 import RUN as ISSUE167_RUN, read_rows, write_json
from run_issue169_patch_token_sweep import episode_indices, score
from run_issue169_tokens import RUN, cohort, status


ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
OUTPUT = REPORTS / "issue169_v7_patch_oos.json"
PRIVATE = RUN / "patch-oos-ranks.private.jsonl"
CONDITIONS = ("cross_midi_common_gallery", "cross_midi_mixed_gallery")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.report_only:
        publish(json.loads(OUTPUT.read_text(encoding="utf-8")))
        return
    episodes, candidates, requests, receipts = cohort()
    request_ids = sorted(receipts)
    request_index = {request: index for index, request in enumerate(request_ids)}
    cache_id = json.loads((RUN / "token-cache.json").read_text(encoding="utf-8"))["cache_id"]
    identity = json.loads((RUN / "tokens" / cache_id / "identity.json").read_text(encoding="utf-8"))
    if identity["rows"] != [[key, receipts[key]["audio_sha256"]] for key in request_ids]:
        raise ValueError("Token cache does not match the verified Issue 167 audio receipts")
    tokens = np.load(RUN / "patch-token-sweep" / "tokens.npy", mmap_mode="r", allow_pickle=False)
    if tokens.shape != (len(request_ids), 256, 384):
        raise ValueError("Cached patch-token array has an unexpected shape")

    condition = CONDITIONS[0]
    rows, query_idx, gallery_idx, targets = episode_indices(condition, episodes, candidates, requests, request_index)
    score_cache = RUN / "patch-oos-common-r1.npy"
    if score_cache.is_file():
        common_r1 = np.load(score_cache, mmap_mode="r", allow_pickle=False)
        if common_r1.shape != (len(rows), 256):
            raise ValueError("Cached token scores have an unexpected shape")
    else:
        common_r1 = score(condition, rows, query_idx, gallery_idx, targets, tokens)["recall_at_1"]
        np.save(score_cache, common_r1, allow_pickle=False)
    evaluated = {condition: (rows, {"recall_at_1": common_r1})}

    performances = sorted({row["query_midi_id"] for row in episodes if row["condition"] == CONDITIONS[0]})
    shuffled = performances.copy()
    random.Random(169).shuffle(shuffled)
    folds = [set(shuffled[index:index + 10]) for index in range(0, len(shuffled), 10)]
    if len(folds) != 5 or any(len(fold) != 10 for fold in folds):
        raise ValueError("Expected five complete folds of ten MIDI performances")

    original_rows = {row["episode_id"]: row for row in read_rows(ISSUE167_RUN / "episode-ranks.private.jsonl")
                     if row["arm"] == "vital_jepa_v7:clip"}
    fold_results = []
    position_counts = np.zeros(256, dtype=np.int32)
    out_of_fold = {condition: [] for condition in CONDITIONS}
    with PRIVATE.open("w", encoding="utf-8", newline="\n") as stream:
        for fold_number, heldout in enumerate(folds, 1):
            common_rows, common_scores = evaluated[CONDITIONS[0]]
            train_indices = [i for i, row in enumerate(common_rows) if row["query_midi_id"] not in heldout]
            val_indices = [i for i, row in enumerate(common_rows) if row["query_midi_id"] in heldout]
            if len(train_indices) != 5160 or len(val_indices) != 1290:
                raise ValueError("Fold does not contain the expected 40/10 performance split")
            r1 = common_scores["recall_at_1"]
            token_order = np.argsort(-r1[train_indices].mean(axis=0), kind="stable")[:8].tolist()
            position_counts[token_order] += 1

            train_episodes = [common_rows[i] for i in train_indices]
            best = None
            for method in ("mean", "concat"):
                for n in range(1, 9):
                    selected = token_order[:n]
                    vectors = (tokens[:, selected, :].mean(axis=1) if method == "mean"
                               else tokens[:, selected, :].reshape(len(request_ids), -1))
                    features = {request: {"candidate": vector}
                                for request, vector in zip(request_ids, vectors, strict=True)}
                    scored = _evaluate_condition(train_episodes, candidates, requests, features, "candidate",
                                                 CONDITIONS[0], {})
                    train_r1 = float(np.mean([row["recall_at_1"] for row in scored]))
                    choice = {"method": method, "n": n, "positions": selected, "train_common_r1": train_r1}
                    if best is None or (train_r1, method == "mean", -n) > (best["train_common_r1"], best["method"] == "mean", -best["n"]):
                        best = choice

            fold_scores = {}
            for condition in CONDITIONS:
                test_episodes = [row for row in episodes if row["condition"] == condition
                                 and row["query_midi_id"] in heldout]
                selected = best["positions"]
                vectors = (tokens[:, selected, :].mean(axis=1) if best["method"] == "mean"
                           else tokens[:, selected, :].reshape(len(request_ids), -1))
                features = {request: {"candidate": vector}
                            for request, vector in zip(request_ids, vectors, strict=True)}
                scored = _evaluate_condition(test_episodes, candidates, requests, features, "candidate", condition, {})
                fold_scores[condition] = float(np.mean([row["recall_at_1"] for row in scored]))
                out_of_fold[condition].extend(scored)
                for row in scored:
                    stream.write(json.dumps({"fold": fold_number, "arm": f"{best['method']}_{best['n']}",
                                             "condition": condition, **row}, separators=(",", ":")) + "\n")

            fold_results.append({"fold": fold_number, "heldout_midi_ids": sorted(heldout), "train_episodes": len(train_indices),
                                 "test_episodes": len(val_indices), "selected_method": best["method"],
                                 "selected_n": best["n"], "selected_positions": best["positions"],
                                 "selected_mel_time": [[index // 32, index % 32] for index in best["positions"]],
                                 "training_common_r1": best["train_common_r1"], "heldout_r1": fold_scores})
            status(f"Completed held-out fold {fold_number}/5", fold_number, 5)

    pooled = {}
    for condition in CONDITIONS:
        rows = out_of_fold[condition]
        baseline = [original_rows[row["episode_id"]]["recall_at_1"] for row in rows]
        pooled[condition] = {"episodes": len(rows), "r1": float(np.mean([row["recall_at_1"] for row in rows])),
                             "original_v7_r1_same_queries": float(np.mean(baseline)),
                             "difference_vs_original_v7": float(np.mean([row["recall_at_1"] - base
                                                                          for row, base in zip(rows, baseline, strict=True)]))}
    result = {"protocol": "five-fold grouped out-of-fold selection; 40 MIDI performances select top eight positions, mean/concat and N by common-gallery R@1; ten held-out MIDI performances score the selected configuration",
              "seed": 169, "performance_count": len(performances), "folds": fold_results,
              "pooled_out_of_fold": pooled,
              "position_selection_frequency": [{"index": int(i), "mel": int(i) // 32, "time": int(i) % 32,
                                                  "folds_selected": int(position_counts[i])}
                                                for i in np.argsort(-position_counts, kind="stable") if position_counts[i]],
              "selection_counts": {"mean": sum(row["selected_method"] == "mean" for row in fold_results),
                                   "concat": sum(row["selected_method"] == "concat" for row in fold_results),
                                   "n": {str(n): sum(row["selected_n"] == n for row in fold_results) for n in range(1, 9)}}}
    write_json(OUTPUT, result)
    publish(result)
    status("Out-of-sample token selection ready", 5, 5, state="complete")


def publish(result):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prior = json.loads((REPORTS / "issue169_v7_patch_topn.json").read_text(encoding="utf-8"))["results"]
    conditions = CONDITIONS
    common, mixed = (result["pooled_out_of_fold"][condition] for condition in conditions)
    in_sample = [prior[condition]["mean_8"]["recall_at_1"]["estimate"] for condition in conditions]
    original = [result["pooled_out_of_fold"][condition]["original_v7_r1_same_queries"] for condition in conditions]
    oof = [row["r1"] for row in (common, mixed)]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), layout="constrained")
    labels = ("Common", "Mixed")
    x = np.arange(2)
    width = 0.24
    for offset, values, label, color in ((-width, original, "Original V7 pooling", "#6c7a89"),
                                         (0, in_sample, "Top 8, selected and scored on same data", "#d08a26"),
                                         (width, oof, "Held-out selection", "#087b70")):
        axes[0].bar(x + offset, np.array(values) * 100, width, label=label, color=color)
    axes[0].set(title="Pooled Recall@1", ylabel="Exact preset retrieval (%)", xticks=x, xticklabels=labels)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", alpha=0.2)
    for condition, color, label in ((conditions[0], "#087b70", "Common"), (conditions[1], "#ab5a29", "Mixed")):
        scores = [fold["heldout_r1"][condition] * 100 for fold in result["folds"]]
        axes[1].plot(range(1, 6), scores, marker="o", linewidth=1.8, color=color, label=label)
    axes[1].set(title="Held-out score by fold", xlabel="Held-out fold", ylabel="Exact preset Recall@1 (%)", xticks=range(1, 6))
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False)
    figure = REPORTS / "figures/issue169_v7_patch_oos.png"
    fig.savefig(figure, dpi=160, bbox_inches="tight")
    plt.close(fig)

    lines = ["# V7 patch-token selection on held-out MIDI performances", "",
             f"Selecting patch positions, pooling method, and N using only other performances retained **{common['r1']:.1%} common-gallery** "
             f"and **{mixed['r1']:.1%} mixed-gallery** Recall@1 across held-out performances. That matches the same-data top-eight result "
             "closely, so the token-selection finding carries over to unseen MIDI performances in this benchmark.", "",
             "This checks whether choosing patch tokens on one set of MIDI performances carries over to different performances. "
             "Each fold uses 40 of the 50 performances to rank positions, then choose mean or concatenation and N (1–8) by common-gallery Recall@1. "
             "The selected setting is scored on the remaining 10 performances. The encoder and audio cache are reused; no audio was rendered.", "",
             "![Held-out and same-data token selection results](figures/issue169_v7_patch_oos.png)", "",
             "## Held-out results", "",
             "| Gallery | Out-of-fold R@1 | Original V7 on same queries | Difference | Episodes |",
             "| --- | ---: | ---: | ---: | ---: |"]
    names = {"cross_midi_common_gallery": "Common", "cross_midi_mixed_gallery": "Mixed"}
    for condition, row in result["pooled_out_of_fold"].items():
        lines.append(f"| {names[condition]} | {row['r1']:.1%} | {row['original_v7_r1_same_queries']:.1%} | "
                     f"{row['difference_vs_original_v7'] * 100:+.1f} pp | {row['episodes']:,} |")
    lines.extend(["", "## Fold-by-fold choices", "",
                  "Positions use mel:time coordinates in the 8 × 32 patch grid.", "",
                  "| Fold | Held-out MIDI count | Chosen method | N | Positions (mel:time) | Train common R@1 | Held-out common R@1 | Held-out mixed R@1 |",
                  "| ---: | ---: | --- | ---: | --- | ---: | ---: | ---: |"])
    for fold in result["folds"]:
        positions = ", ".join(f"{mel}:{time}" for mel, time in fold["selected_mel_time"])
        lines.append(f"| {fold['fold']} | {len(fold['heldout_midi_ids'])} | {fold['selected_method']} | {fold['selected_n']} | {positions} | "
                     f"{fold['training_common_r1']:.1%} | {fold['heldout_r1']['cross_midi_common_gallery']:.1%} | "
                     f"{fold['heldout_r1']['cross_midi_mixed_gallery']:.1%} |")
    lines.extend(["", "## How often positions were selected", "",
                  "| Mel patch | Time patch | Folds selected |", "| ---: | ---: | ---: |"])
    for row in result["position_selection_frequency"]:
        lines.append(f"| {row['mel']} | {row['time']} | {row['folds_selected']} / 5 |")
    lines.extend(["", "## Interpretation", "",
                  "The pooled held-out score estimates the complete selection procedure, including choosing positions, pooling method, and N. "
                  "The split holds query performances out; gallery references follow the benchmark definition and can reuse MIDI performances across folds. "
                  "So this tests whether selection generalizes to unseen query performances within this preset set, not to a fully separate audio corpus, new presets, or listening similarity.", "",
                  "Position frequencies show whether the same patches are repeatedly chosen. Five folds provide a useful check but limited evidence for fine-grained uncertainty; fold results are shown individually.", "",
                  "[Machine-readable results](issue169_v7_patch_oos.json) · [Earlier same-cohort top-N sweep](VITAL_JEPA_V7_PATCH_TOPN.md)", ""])
    (REPORTS / "VITAL_JEPA_V7_PATCH_OOS.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

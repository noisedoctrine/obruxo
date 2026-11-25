"""Score each top-N V7 patch-token pooling choice on five held-out MIDI folds."""

import json
from pathlib import Path

import numpy as np

from frozen_retrieval.evaluation import _evaluate_condition
from run_issue169_patch_token_sweep import episode_indices
from run_issue169_tokens import RUN, cohort, status
from run_issue167_v7 import write_json


ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
RESULT_PATH = REPORTS / "issue169_v7_patch_oos.json"
FIGURE = REPORTS / "figures/issue169_v7_patch_oos_topn.png"
CONDITIONS = ("cross_midi_common_gallery", "cross_midi_mixed_gallery")
CONDITION_SHORT = {CONDITIONS[0]: "common", CONDITIONS[1]: "mixed"}


def main():
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    episodes, candidates, requests, receipts = cohort()
    request_ids = sorted(receipts)
    request_index = {request: index for index, request in enumerate(request_ids)}
    cache_id = json.loads((RUN / "token-cache.json").read_text(encoding="utf-8"))["cache_id"]
    identity = json.loads((RUN / "tokens" / cache_id / "identity.json").read_text(encoding="utf-8"))
    if identity["rows"] != [[key, receipts[key]["audio_sha256"]] for key in request_ids]:
        raise ValueError("Token cache does not match the verified Issue 167 audio receipts")
    tokens = np.load(RUN / "patch-token-sweep" / "tokens.npy", mmap_mode="r", allow_pickle=False)
    common_rows, _, _, _ = episode_indices(CONDITIONS[0], episodes, candidates, requests, request_index)
    token_r1 = np.load(RUN / "patch-oos-common-r1.npy", mmap_mode="r", allow_pickle=False)
    if token_r1.shape != (len(common_rows), 256):
        raise ValueError("Cached individual-token scores do not align with the retrieval episodes")
    row_index_by_midi = {}
    for index, row in enumerate(common_rows):
        row_index_by_midi.setdefault(row["query_midi_id"], []).append(index)

    curves = {f"{method}_{n}": {CONDITION_SHORT[condition]: [] for condition in CONDITIONS}
              for method in ("mean", "concat") for n in range(1, 9)}
    for fold in result["folds"]:
        heldout = set(fold["heldout_midi_ids"])
        train_indices = [i for midi, indices in row_index_by_midi.items() if midi not in heldout for i in indices]
        order = np.argsort(-token_r1[train_indices].mean(axis=0), kind="stable")[:8].tolist()
        test_episodes = {condition: [row for row in episodes if row["condition"] == condition
                                     and row["query_midi_id"] in heldout]
                         for condition in CONDITIONS}
        for method in ("mean", "concat"):
            for n in range(1, 9):
                selected = order[:n]
                vectors = (tokens[:, selected, :].mean(axis=1) if method == "mean"
                           else tokens[:, selected, :].reshape(len(request_ids), -1))
                features = {request: {"candidate": vector}
                            for request, vector in zip(request_ids, vectors, strict=True)}
                key = f"{method}_{n}"
                for condition in CONDITIONS:
                    rows = _evaluate_condition(test_episodes[condition], candidates, requests, features,
                                               "candidate", condition, {})
                    value = float(np.mean([row["recall_at_1"] for row in rows]))
                    curves[key][CONDITION_SHORT[condition]].append(value)
        status(f"Scored held-out top-N curves for fold {fold['fold']}/5", fold["fold"], 5)

    summaries = {key: {condition: {"fold_scores": scores, "mean": float(np.mean(scores)),
                                  "min": float(np.min(scores)), "max": float(np.max(scores))}
                       for condition, scores in by_condition.items()}
                 for key, by_condition in curves.items()}
    result["crossval_top_n_curves"] = {
        "protocol": "Each fold ranks positions on its 40 training MIDI performances; mean and concatenated top-N vectors are evaluated on its ten held-out query performances.",
        "error_bars": "minimum to maximum held-out fold score; not a confidence interval",
        "results": summaries}
    write_json(RESULT_PATH, result)
    publish(result)
    status("Out-of-sample top-N curves ready", 5, 5, state="complete")


def publish(result):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = result["crossval_top_n_curves"]["results"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, layout="constrained")
    baseline = {"common": result["pooled_out_of_fold"][CONDITIONS[0]]["original_v7_r1_same_queries"],
                "mixed": result["pooled_out_of_fold"][CONDITIONS[1]]["original_v7_r1_same_queries"]}
    for ax, condition, title in zip(axes, ("common", "mixed"), ("Common gallery", "Mixed gallery"), strict=True):
        for method, color, marker in (("mean", "#087b70", "o"), ("concat", "#ab5a29", "s")):
            values = [data[f"{method}_{n}"][condition]["mean"] * 100 for n in range(1, 9)]
            lower = [data[f"{method}_{n}"][condition]["min"] * 100 for n in range(1, 9)]
            upper = [data[f"{method}_{n}"][condition]["max"] * 100 for n in range(1, 9)]
            ax.errorbar(range(1, 9), values,
                        yerr=[[value - lo for value, lo in zip(values, lower, strict=True)],
                              [hi - value for value, hi in zip(values, upper, strict=True)]],
                        marker=marker, capsize=3, linewidth=1.8, color=color,
                        label=f"{method.title()} pooling, held-out fold mean and range")
        ax.axhline(baseline[condition] * 100, color="#657384", linestyle="--", linewidth=1.2,
                   label="Original V7 pooling")
        ax.set(title=title, xlabel="Number of top-ranked patch tokens", xticks=range(1, 9),
               ylabel="Exact preset Recall@1 (%)", ylim=(0, 70))
        ax.grid(axis="y", alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("V7 top-N patch-token retrieval on held-out MIDI performances", fontsize=13, fontweight="bold")
    fig.savefig(FIGURE, dpi=160, bbox_inches="tight")
    plt.close(fig)

    lines = ["# Held-out V7 patch-token results by token count", "",
             "The chart reports Recall@1 averaged over five held-out folds for each token count. Error bars show the lowest to highest fold, not a confidence interval.", "",
             "Each fold ranks token positions using 40 MIDI performances, then evaluates the top N on the remaining 10. "
             "This is done independently for mean pooling and concatenation.", "",
             "![Held-out Recall@1 by number of top-ranked tokens](figures/issue169_v7_patch_oos_topn.png)", "",
             "The dashed line is the original V7 pooling score on the same query episodes. Galleries follow the benchmark definition and may reuse MIDI performances across folds.", "",
             "| N | Mean common R@1 (fold mean; min–max) | Mean mixed R@1 (fold mean; min–max) | Concat common R@1 (fold mean; min–max) | Concat mixed R@1 (fold mean; min–max) |",
             "| ---: | ---: | ---: | ---: | ---: |"]
    for n in range(1, 9):
        values = [data[f"{method}_{n}"][condition] for method, condition in
                  (("mean", "common"), ("mean", "mixed"), ("concat", "common"), ("concat", "mixed"))]
        cells = [f"{row['mean']:.1%} ({row['min']:.1%}–{row['max']:.1%})" for row in values]
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    lines.extend(["", "## Reading the result", "",
                  "The bars show variation across five specific folds. They are ranges, not statistical confidence intervals. "
                  "The held-out result estimates performance on new query performances within this preset set; it does not establish performance on a separate audio corpus, new presets, or listener-rated timbre similarity.", "",
                  "[Machine-readable fold results](issue169_v7_patch_oos.json) · [Earlier same-cohort sweep](VITAL_JEPA_V7_PATCH_TOPN.md)", ""])
    (REPORTS / "VITAL_JEPA_V7_PATCH_OOS_CURVES.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

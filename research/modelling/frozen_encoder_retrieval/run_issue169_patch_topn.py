"""Compare mean-pooled and concatenated top V7 patch tokens on Issue 167 galleries."""

import argparse
import json
from pathlib import Path

import numpy as np

from frozen_retrieval.core import paired_two_way_cluster_bootstrap, two_way_cluster_bootstrap
from frozen_retrieval.evaluation import _evaluate_condition
from run_issue167_v7 import RUN as ISSUE167_RUN, ROOT, read_json, read_rows, write_json
from run_issue169_tokens import RUN, cohort, status


REPORTS = ROOT / "reports"
RANK_PATH = RUN / "patch-topn-ranks.private.jsonl"
CONDITIONS = ("cross_midi_common_gallery", "cross_midi_mixed_gallery")
METRICS = ("recall_at_1", "recall_at_5", "recall_at_10")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    episodes, candidates, requests, receipts = cohort()
    request_ids = sorted(receipts)
    cache_id = read_json(RUN / "token-cache.json")["cache_id"]
    cache = RUN / "patch-token-sweep" / "tokens.npy"
    identity = json.loads((RUN / "tokens" / cache_id / "identity.json").read_text(encoding="utf-8"))
    if identity["rows"] != [[key, receipts[key]["audio_sha256"]] for key in request_ids]:
        raise ValueError("Patch tokens do not match the verified Issue 167 audio receipts")
    tokens = np.load(cache, mmap_mode="r", allow_pickle=False)
    patch_results = read_json(REPORTS / "issue169_v7_patch_tokens.json")["cross_midi_common_gallery"]["recall_at_1"]
    token_order = sorted(range(256), key=lambda i: patch_results["estimate"][i], reverse=True)[:8]
    old_rows = [row for row in read_rows(ISSUE167_RUN / "episode-ranks.private.jsonl")
                if row["arm"] == "vital_jepa_v7:clip"]
    old_by_condition = {condition: [row for row in old_rows if row["condition"] == condition] for condition in CONDITIONS}
    if args.report_only:
        results = read_json(REPORTS / "issue169_v7_patch_topn.json")["results"]
        report(REPORTS / "figures/issue169_v7_patch_topn.png", token_order, results)
        return
    results = {condition: {} for condition in CONDITIONS}
    status("Score top token combinations", 0, 16)
    with RANK_PATH.open("w", encoding="utf-8") as stream:
        arm_number = 0
        for count in range(1, 9):
            indices = token_order[:count]
            for method in ("mean", "concat"):
                key = f"{method}_{count}"
                vectors = tokens[:, indices, :].mean(axis=1) if method == "mean" else tokens[:, indices, :].reshape(len(request_ids), -1)
                features = {request: {key: vector} for request, vector in zip(request_ids, vectors, strict=True)}
                for condition in CONDITIONS:
                    rows = _evaluate_condition(episodes, candidates, requests, features, key, condition, {})
                    for row in rows:
                        tied = row["tie_count"]
                        higher = row["rank_expected"] - (tied + 1) / 2
                        row["recall_at_10"] = min(max(10 - higher, 0), tied) / tied
                        stream.write(json.dumps({"arm": key, "condition": condition, **row}, separators=(",", ":")) + "\n")
                    old = old_by_condition[condition]
                    results[condition][key] = {
                        "dimensions": int(vectors.shape[1]),
                        **{metric: two_way_cluster_bootstrap(rows, metric=metric, seed=152) for metric in METRICS},
                        "r1_change_vs_original": paired_two_way_cluster_bootstrap(rows, old, metric="recall_at_1", seed=152),
                    }
                arm_number += 1
                status("Score top token combinations", arm_number, 16)

    plot_path = REPORTS / "figures/issue169_v7_patch_topn.png"
    write_json(REPORTS / "issue169_v7_patch_topn.json", {"top_patch_indices": token_order, "results": results,
                                                         "selection_metric": "common-gallery individual-token Recall@1",
                                                         "selection_cohort": "same 6,450 Issue 167 episodes", "pooling": "arithmetic mean",
                                                         "dimensions": {"mean": 384, "concat_per_token": 384}})
    report(plot_path, token_order, results)


def report(plot_path, token_order, results):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    baseline = 0.2910077519379845
    common, mixed = (results[condition] for condition in CONDITIONS)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, layout="constrained")
    for ax, (condition, data, title) in zip(axes, ((CONDITIONS[0], common, "Common gallery"),
                                                   (CONDITIONS[1], mixed, "Mixed gallery")), strict=True):
        for method, color in (("mean", "#087b70"), ("concat", "#ab5a29")):
            estimates = [data[f"{method}_{n}"]["recall_at_1"]["estimate"] * 100 for n in range(1, 9)]
            lower = [data[f"{method}_{n}"]["r1_change_vs_original"]["lower_95"] * 100 for n in range(1, 9)]
            upper = [data[f"{method}_{n}"]["r1_change_vs_original"]["upper_95"] * 100 for n in range(1, 9)]
            ax.errorbar(range(1, 9), estimates, yerr=[[est - baseline * 100 - lo for est, lo in zip(estimates, lower, strict=True)],
                                                       [baseline * 100 + hi - est for est, hi in zip(estimates, upper, strict=True)]],
                        marker="o", capsize=3, linewidth=1.8, color=color, label=method.title())
        ax.axhline(baseline * 100, color="#657384", linestyle="--", label="Original V7 pooling")
        ax.axhline(59.4 if condition == CONDITIONS[0] else 55.0, color="#d08a26", linestyle=":", label="CLAP")
        ax.axhline(81.0 if condition == CONDITIONS[0] else 80.9, color="#8745a8", linestyle=":", label="CIR")
        ax.set(title=title, xlabel="Number of top-ranked patch tokens", xticks=range(1, 9), ylabel="Exact preset Recall@1 (%)")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Combining the eight highest-scoring individual patch positions", fontsize=13, fontweight="bold")
    fig.savefig(plot_path, dpi=155, bbox_inches="tight")
    plt.close(fig)

    lines = ["# Combining the top V7 patch tokens", "",
             "The eight positions were selected by their individual common-gallery Recall@1 on these same 6,450 queries. "
             "The combined results are exploratory and likely optimistic. A separate cohort is needed to confirm the token ranking.", "",
             f"Position order, from strongest to eighth strongest: {', '.join(f'{index // 32}:{index % 32}' for index in token_order)} (mel:time).", "",
             "![Mean and concatenated retrieval results](figures/issue169_v7_patch_topn.png)", "",
             "| N | Mean dimensions | Mean common R@1 / 5 / 10 | Mean mixed R@1 | Concat dimensions | Concat common R@1 / 5 / 10 | Concat mixed R@1 |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for n in range(1, 9):
        mean, concat = common[f"mean_{n}"], common[f"concat_{n}"]
        fmt = lambda row: " / ".join(f"{row[f'recall_at_{k}']['estimate']:.1%}" for k in (1, 5, 10))
        lines.append(f"| {n} | {mean['dimensions']} | {fmt(mean)} | {mixed[f'mean_{n}']['recall_at_1']['estimate']:.1%} | "
                     f"{concat['dimensions']} | {fmt(concat)} | {mixed[f'concat_{n}']['recall_at_1']['estimate']:.1%} |")
    lines.extend(["", "## Interpretation", "",
                  "Each token is a contextual output at one mel/time position after full self-attention. The combined vectors compare the same selected positions between recordings. "
                  "Mean pooling keeps 384 dimensions. Concatenation preserves all selected token outputs and increases dimension by 384 per token. Cosine scoring normalizes each vector.", "",
                  "The individual positions were selected and the combinations evaluated on the same cohort, so the apparent best N should be treated as a screening result. "
                  "The score measures exact preset retrieval and cannot alone establish which combination produces the closest-sounding alternate preset.", "",
                  "[Machine-readable results](issue169_v7_patch_topn.json) · [Per-position sweep](VITAL_JEPA_V7_PATCH_TOKEN_SWEEP.md)", ""])
    (REPORTS / "VITAL_JEPA_V7_PATCH_TOPN.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

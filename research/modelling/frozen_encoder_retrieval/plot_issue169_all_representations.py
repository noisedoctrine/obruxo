"""Rank every evaluated Issue 169 representation by full-cohort common-gallery Recall@1."""

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
RESULTS = REPORTS / "issue169_v7_all_representations.json"
CSV = REPORTS / "issue169_v7_all_representations.csv"
FIGURE = REPORTS / "figures/issue169_v7_all_representations.png"
PDF = REPORTS / "issue169_v7_all_representations.pdf"

COLORS = {"V7 fixed": "#2878a5", "External baseline": "#7251a3", "Global token": "#548c55",
          "Top-N mean": "#d39025", "Top-N concat": "#bb5a32"}


def main():
    retrieval = json.loads((REPORTS / "issue169_v7_retrieval_results.json").read_text(encoding="utf-8"))
    topn = json.loads((REPORTS / "issue169_v7_patch_topn.json").read_text(encoding="utf-8"))
    token_order = topn["top_patch_indices"]
    condition = "cross_midi_common_gallery"
    entries = []

    fixed = retrieval["retrieval"][condition]
    dimensions = {key: int(value[1]) for key, value in retrieval["representations"].items()}
    for key, metadata in retrieval["representations"].items():
        if key.startswith("global_token_"):
            continue
        entries.append({"name": metadata[0], "dimensions": dimensions[key], "r1": fixed[key]["metrics"]["recall_at_1"]["estimate"],
                        "group": "V7 fixed", "selection": "Representation fixed before scoring"})

    for key, label, model in (("cir_ast:official_clip", "CIR", "cir_ast"),
                              ("clap_music:official_clip", "CLAP", "clap_music")):
        metadata = json.loads((ROOT / "outputs/issue167-v7/dataset/features" / model / "extraction.latest.json").read_text(encoding="utf-8"))
        dim = int(metadata["dimensions_by_variant"]["official_clip"])
        entries.append({"name": label, "dimensions": dim, "r1": fixed[key]["metrics"]["recall_at_1"]["estimate"],
                        "group": "External baseline", "selection": "Previously evaluated baseline"})

    for key, metadata in retrieval["representations"].items():
        if not key.startswith("global_token_"):
            continue
        token_number = int(key.rsplit("_", 1)[1]) + 1
        entries.append({"name": f"Global token {token_number}", "dimensions": int(metadata[1]),
                        "r1": fixed[key]["metrics"]["recall_at_1"]["estimate"], "group": "Global token",
                        "selection": "Individual token"})

    topn_candidates = []
    for n in range(1, 9):
        variants = {method: topn["results"][condition][f"{method}_{n}"] for method in ("mean", "concat")}
        method = max(variants, key=lambda name: variants[name]["recall_at_1"]["estimate"])
        row = variants[method]
        topn_candidates.append({"name": "Top 1 token (mean = concat)" if n == 1 else f"Top {n} tokens, {method}",
                                "dimensions": int(row["dimensions"]), "r1": row["recall_at_1"]["estimate"],
                                "group": "Top-N mean" if method == "mean" else "Top-N concat",
                                "selection": "Highest full-cohort common-gallery Recall@1 among all tested top-N variants"})
    entries.append(max(topn_candidates, key=lambda row: row["r1"]))

    entries.sort(key=lambda row: (-row["r1"], row["name"]))
    for rank, entry in enumerate(entries, 1):
        entry["rank"] = rank
    payload = {"metric": "common-gallery Recall@1", "queries": 6450, "preset_families": 129,
               "midi_performances": 50, "representation_count": len(entries), "token_order": token_order,
               "selection_note": "Only the single best top-N variant is included; it was selected on this same cohort. The individual patch-position sweep is omitted from this chart and is reported separately.",
               "representations": entries}
    RESULTS.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with CSV.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("rank", "name", "dimensions", "r1", "r1_percent", "group", "selection"))
        writer.writeheader()
        for entry in entries:
            writer.writerow({**entry, "r1_percent": round(entry["r1"] * 100, 3)})
    plot(entries)
    write_report(payload)


def plot(entries):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(entries) <= 40:
        fig, ax = plt.subplots(figsize=(13, max(7, len(entries) * 0.42)), layout="constrained")
        y = np.arange(len(entries))
        ax.barh(y, [row["r1"] * 100 for row in entries], height=0.72,
                color=[COLORS[row["group"]] for row in entries], edgecolor="none")
        ax.set(yticks=y, yticklabels=[f"{row['name']} ({row['dimensions']}-D)" for row in entries],
               xlabel="Common-gallery exact preset Recall@1 (%)", xlim=(0, 87))
        ax.invert_yaxis()
        ax.tick_params(axis="y", labelsize=9, length=0, pad=5)
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
        baseline = next(row["r1"] for row in entries if row["name"] == "Original V7 pooling") * 100
        ax.axvline(baseline, color="#657384", linestyle="--", linewidth=1.2)
        for y_pos, row in enumerate(entries):
            ax.text(row["r1"] * 100 + 0.5, y_pos, f"{row['r1']:.1%}", va="center", fontsize=8)
        present_groups = {row["group"] for row in entries}
        handles = [plt.Rectangle((0, 0), 1, 1, color=color, label=label)
                   for label, color in COLORS.items() if label in present_groups]
        fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 1.04))
        fig.suptitle(f"Representation ranking with the single best top-N token result\n"
                     f"{len(entries)} representations · 6,450 queries · dashed line: original V7 pooling",
                     y=1.16, fontsize=14, fontweight="bold")
        fig.savefig(FIGURE, dpi=180, bbox_inches="tight")
        fig.savefig(PDF, bbox_inches="tight")
        plt.close(fig)
        return

    split = (len(entries) + 1) // 2
    pages = (entries[:split], entries[split:])
    fig, axes = plt.subplots(1, 2, figsize=(24, 29), sharex=True, layout="constrained")
    baseline = next(row["r1"] for row in entries if row["name"] == "Original V7 pooling") * 100
    split_rank = len(pages[0])
    for ax, rows, part in zip(axes, pages, (f"Ranks 1–{split_rank}", f"Ranks {split_rank + 1}–{len(entries)}"), strict=True):
        y = np.arange(len(rows))
        ax.barh(y, [row["r1"] * 100 for row in rows], height=0.78,
                color=[COLORS[row["group"]] for row in rows], edgecolor="none")
        ax.set(yticks=y, yticklabels=[f"{row['name']} ({row['dimensions']}-D)" for row in rows],
               title=part, xlabel="Common-gallery exact preset Recall@1 (%)", xlim=(0, 87))
        ax.invert_yaxis()
        ax.tick_params(axis="y", labelsize=6.3, length=0, pad=2)
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
        ax.axvline(baseline, color="#657384", linestyle="--", linewidth=1.1)
        for y_pos, row in enumerate(rows):
            ax.text(row["r1"] * 100 + 0.35, y_pos, f"{row['r1']:.1%}", va="center", fontsize=5.7)
    present_groups = {row["group"] for row in entries}
    handles = [plt.Rectangle((0, 0), 1, 1, color=color, label=label)
               for label, color in COLORS.items() if label in present_groups]
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.008))
    fig.suptitle(f"All {len(entries)} tested representations ranked by full-cohort Recall@1\n"
                 "6,450 queries · 129 preset families · 50 MIDI performances · dashed line: original V7 pooling",
                 y=1.035, fontsize=16, fontweight="bold")
    fig.savefig(FIGURE, dpi=180, bbox_inches="tight")
    fig.savefig(PDF, bbox_inches="tight")
    plt.close(fig)


def write_report(payload):
    lines = ["# V7 representation ranking with the best top-N token result", "",
             "The chart compares the fixed V7 representations and baselines, four individual global tokens, and the single best top-N patch-token representation by common-gallery Recall@1 on all 6,450 Issue 167 queries (129 preset families × 50 MIDI performances). "
             "Each bar label includes its vector size. Higher is better.", "",
             "![All tested representations ranked by Recall@1](figures/issue169_v7_all_representations.png)", "",
             "The top-token combinations contribute only their single best tested choice: N=7 with mean pooling (384-D). The 256 individual patch positions are left out here; see the separate [position sweep](VITAL_JEPA_V7_PATCH_TOKEN_SWEEP.md).", "",
             "The top patch positions and top-N winner were selected using the same full cohort; their full-cohort scores are exploratory. The five-fold held-out results are available in [the token-count report](VITAL_JEPA_V7_PATCH_OOS_CURVES.md).", "",
             f"[All {payload['representation_count']} ranked values as CSV](issue169_v7_all_representations.csv) · "
             "[Machine-readable results](issue169_v7_all_representations.json) · [Full-size PDF](issue169_v7_all_representations.pdf)", ""]
    (REPORTS / "VITAL_JEPA_V7_ALL_REPRESENTATIONS.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

"""Derive Issue 169 representations from the retained tokens and score retrieval."""

import argparse
import json
from time import sleep

import numpy as np

from frozen_retrieval.core import digest, paired_two_way_cluster_bootstrap, two_way_cluster_bootstrap
from frozen_retrieval.evaluation import _evaluate_condition
from run_issue167_v7 import ROOT, RUN as PREVIOUS_RUN, read_json, read_rows, write_json
from run_issue169_tokens import RUN, SHARD_SIZE, array_to_file, checkpoint, cohort, load_shard, status


CONDITIONS = {"cross_midi_common_gallery": "Common gallery", "cross_midi_mixed_gallery": "Mixed gallery"}
METRICS = ("recall_at_1", "recall_at_5", "recall_at_10")
BASELINE = "original_v7"
ARMS = {"contrastive": ("Contrastive projection", 128), "control": ("Control contrastive projection", 128),
        "both_projections": ("Both projections concatenated", 256), "global_concat": ("Global tokens concatenated", 1536),
        "global_token_0": ("Global token 1", 384), "global_token_1": ("Global token 2", 384),
        "global_token_2": ("Global token 3", 384), "global_token_3": ("Global token 4", 384),
        "patch_mean": ("Patch token mean", 384), "global_mean": ("Global token mean", 384),
        "acoustic": ("Acoustic descriptor predictions", 15), "global_both": ("Global tokens plus both projections", 1792),
        "patch_pyramid": ("Patch temporal pyramid", 6144), BASELINE: ("Original V7 pooling", 384)}
ANCHORS = {"cir_ast:official_clip": "CIR", "clap_music:official_clip": "CLAP"}
SAVED_ARMS = {"vital_jepa_v7:clip": BASELINE, **ANCHORS}
RELEASE = "https://huggingface.co/duckking032/vital-jepa-v7/tree/main"
HEADS = {"contrastive": "aux_projection", "control": "control_delta_projection", "acoustic": "acoustic_feature_head"}


def load_head(torch, weights, name, output_size):
    # The parameterized layers follow the saved indices. The stateless activation is a local choice.
    head = torch.nn.Sequential(torch.nn.LayerNorm(384), torch.nn.Linear(384, 384), torch.nn.GELU(),
                               torch.nn.Linear(384, output_size))
    head.load_state_dict({key.removeprefix(name + "."): value for key, value in weights.items()
                          if key.startswith(name + ".")}, strict=True)
    return head.eval()


def temporal_pyramid(patches, valid_steps):
    # Patch embedding flattens the mel axis first, with time as the fastest index.
    frames = patches.reshape(len(patches), 8, 32, 384).mean(axis=1, dtype=np.float64)
    result = np.empty((len(frames), 6144), dtype=np.float32)
    for count in np.unique(valid_steps):
        selection = valid_steps == count
        valid = frames[selection]
        parts = [valid[:, i * count // 8:(i + 1) * count // 8] for i in range(8)]
        moments = [stat for part in parts for stat in (part.mean(axis=1), part.std(axis=1))]
        result[selection] = np.concatenate(moments, axis=1)
    return result


def derive(receipts):
    import torch

    cache = RUN / "tokens" / read_json(RUN / "token-cache.json")["cache_id"]
    identity = read_json(cache / "identity.json")
    request_ids = sorted(receipts)
    path, entry, checkpoint_sha = checkpoint()
    if identity["rows"] != [[key, receipts[key]["audio_sha256"]] for key in request_ids]:
        raise ValueError("Token cache does not match the Issue 167 WAV receipts")
    if cache.name != digest(identity) or identity["checkpoint_sha256"] != checkpoint_sha:
        raise ValueError("Token cache does not match the locked checkpoint")
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    device = identity["device"]
    heads = {key: load_head(torch, artifact["model"], name, ARMS[key][1]) for key, name in HEADS.items()}
    config = artifact["config"]
    config_model = config["model"]
    widths = {name: list(weight.shape) for name, weight in artifact["model"].items()
              if name.endswith("weight") and ("_head." in name or "_projection." in name)}
    if config_model["global_tokens"] != 4 or config["features"]["n_mels"] // config["patching"]["patch_mels"] != 8:
        raise ValueError("Checkpoint token layout differs from the benchmark's recorded geometry")
    sample_rate = config["data"]["sample_rate"]
    hop = config["features"]["hop_length"]
    patch_frames = config["patching"]["patch_frames"]
    sample_counts = np.ceil(np.array([receipts[key]["duration_seconds"] for key in request_ids]) * sample_rate).astype(int)
    valid_steps = np.minimum(32, (sample_counts // hop + 1) // patch_frames)
    if (valid_steps < 8).any():
        raise ValueError("A source clip has fewer than eight full time patches")

    vectors = {key: np.empty((len(request_ids), dim), dtype=np.float32) for key, (_, dim) in ARMS.items()}
    status("Derive clip representations", 0, len(request_ids))
    with torch.inference_mode():
        for start in range(0, len(request_ids), SHARD_SIZE):
            end = min(start + SHARD_SIZE, len(request_ids))
            shard = load_shard(cache / f"{start:05d}.npy", end - start)
            tokens = torch.from_numpy(np.array(shard)).to(device)
            global_mean = tokens[:, :4].mean(dim=1)
            patch_mean = tokens[:, 4:].mean(dim=1)
            pooled = (global_mean + patch_mean).cpu()
            vectors[BASELINE][start:end] = pooled.numpy()
            vectors["global_mean"][start:end] = global_mean.cpu().numpy()
            vectors["patch_mean"][start:end] = patch_mean.cpu().numpy()
            vectors["global_concat"][start:end] = shard[:, :4].reshape(end - start, -1)
            for token_index in range(4):
                vectors[f"global_token_{token_index}"][start:end] = shard[:, token_index]
            vectors["patch_pyramid"][start:end] = temporal_pyramid(shard[:, 4:], valid_steps[start:end])
            for key, head in heads.items():
                vectors[key][start:end] = head(pooled).numpy()
            vectors["both_projections"][start:end] = np.concatenate((vectors["contrastive"][start:end],
                                                                       vectors["control"][start:end]), axis=1)
            vectors["global_both"][start:end] = np.concatenate((vectors["global_concat"][start:end],
                                                                  vectors["contrastive"][start:end],
                                                                  vectors["control"][start:end]), axis=1)
            status("Derive clip representations", end, len(request_ids))
    vector_dir = RUN / "vectors"
    vector_dir.mkdir(exist_ok=True)
    for key, array in vectors.items():
        if not np.isfinite(array).all() or (np.linalg.norm(array, axis=1) <= 1e-12).any():
            raise ValueError(f"Invalid vectors for {key}")
        array_to_file(vector_dir / f"{key}.npy", array)
    head_input = "final global mean plus final patch mean"
    write_json(vector_dir / "identity.json", {"token_cache_id": cache.name, "request_ids": request_ids,
                                              "head_input": head_input, "head_activation": "GELU"})
    provenance = {"checkpoint_sha256": checkpoint_sha, "checkpoint_revision": entry["revision"], "step": artifact["step"],
                  "head_weight_shapes": widths, "token_cache_id": cache.name, "token_shape": identity["token_shape"],
                  "preprocessing": identity["preprocessing"], "pyramid_valid_time_steps": [int(valid_steps.min()), int(valid_steps.max())],
                  "token_device": device, "head_input": head_input, "head_activation": "GELU"}
    return request_ids, vectors, provenance


def score(request_ids, vectors, episodes, candidates, requests):
    previous = {condition: {arm: [] for arm in SAVED_ARMS} for condition in CONDITIONS}
    for row in read_rows(PREVIOUS_RUN / "episode-ranks.private.jsonl"):
        previous[row["condition"]][row["arm"]].append(row)
    aggregate = {condition: {} for condition in CONDITIONS}
    replay = {}
    rank_path = RUN / "episode-ranks.private.jsonl"
    order = (BASELINE, *(key for key in ARMS if key != BASELINE), *ANCHORS)
    status("Score cross-MIDI retrieval", 0, len(order))
    with rank_path.with_suffix(".tmp").open("w", encoding="utf-8") as stream:
        for index, key in enumerate(order, 1):
            if key in vectors:
                features = {rid: {key: row} for rid, row in zip(request_ids, vectors[key], strict=True)}
            for condition in CONDITIONS:
                if key in ANCHORS:
                    rows = previous[condition][key]
                else:
                    rows = _evaluate_condition(episodes, candidates, requests, features, key, condition, {})
                    for row in rows:
                        tied = row["tie_count"]
                        higher = row["rank_expected"] - (tied + 1) / 2
                        row["recall_at_10"] = min(max(10 - higher, 0), tied) / tied
                if len(rows) != 6450 or {row["gallery_size"] for row in rows} != {129}:
                    raise ValueError("Unexpected retrieval denominator")
                old = previous[condition]["vital_jepa_v7:clip"]
                metrics = {m: two_way_cluster_bootstrap(rows, metric=m, seed=152) for m in METRICS}
                if key == BASELINE:
                    old_by_id = {row["episode_id"]: row for row in old}
                    replay[condition] = {m: sum(row[m] != old_by_id[row["episode_id"]][m] for row in rows) for m in METRICS}
                    if any(count > 1 for count in replay[condition].values()):
                        raise ValueError(f"Original V7 replay differs from Issue 167: {replay[condition]}")
                paired = paired_two_way_cluster_bootstrap(rows, old, metric=METRICS[0], seed=152)
                aggregate[condition][key] = {"metrics": metrics, "paired_r1_vs_original": paired}
                if key in vectors:
                    fields = ("episode_id", "query_midi_id", "target_family_id", "gallery_size", "rank_expected", "tie_count", *METRICS)
                    for row in rows:
                        record = {**{field: row[field] for field in fields}, "arm": key, "condition": condition}
                        stream.write(json.dumps(record, separators=(",", ":")) + "\n")
            status("Score cross-MIDI retrieval", index, len(order))
    rank_path.with_suffix(".tmp").replace(rank_path)
    return aggregate, replay


def plot(aggregate, image_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = sorted(ARMS, key=lambda key: aggregate[next(iter(CONDITIONS))][key]["metrics"][METRICS[0]]["estimate"])
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.8), sharey=True, gridspec_kw={"width_ratios": (1.4, 1)})
    colors = ["#657384" if key == BASELINE else "#087b70" for key in keys]
    for (condition, name), (ax, dx) in zip(CONDITIONS.items(), axes, strict=True):
        rows = aggregate[condition]
        y = np.arange(len(keys))
        r1 = np.array([rows[key]["metrics"][METRICS[0]]["estimate"] for key in keys]) * 100
        delta, lo, hi = (np.array([rows[key]["paired_r1_vs_original"][metric] for key in keys]) * 100
                         for metric in ("difference", "lower_95", "upper_95"))
        ax.barh(y, r1, color=colors, height=0.64)
        ax.bar_label(ax.containers[0], fmt="%.1f%%", padding=4, fontsize=9)
        for (key, label), color in zip(ANCHORS.items(), ("#8a46a8", "#c25d1f"), strict=True):
            anchor = rows[key]["metrics"][METRICS[0]]["estimate"] * 100
            ax.axvline(anchor, ls="--", color=color, linewidth=1.4, label=f"{label} {anchor:.1f}%")
        ax.set(xlim=(0, 100), xlabel="Exact preset ranked first (%)", title=name, yticks=y,
               yticklabels=[ARMS[key][0] for key in keys])
        ax.legend(loc="lower right", frameon=False, fontsize=8)
        dx.axvline(0, color="#89939e", linewidth=1.0)
        dx.hlines(y, lo, hi, color=colors, linewidth=2)
        dx.scatter(delta, y, c=colors, s=22, zorder=3)
        dx.set(xlabel="Change in Recall@1 (percentage points)", title="Difference versus original V7 (95% interval)")
        for panel in (ax, dx):
            panel.grid(axis="x", alpha=0.18)
            panel.set_axisbelow(True)
            panel.spines[["top", "right", "left"]].set_visible(False)
    fig.suptitle("Vital JEPA V7 representations for exact preset retrieval", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96), pad=2)
    fig.savefig(image_path, dpi=155, bbox_inches="tight")
    plt.close(fig)


def publish(aggregate, replay, provenance):
    reports = ROOT / "reports"
    image_path = reports / "figures/issue169_v7_retrieval.png"
    plot(aggregate, image_path)
    common, mixed = tuple(aggregate.values())
    best = max((key for key in ARMS if key != BASELINE), key=lambda key: common[key]["metrics"][METRICS[0]]["estimate"])
    r1 = lambda rows, key: rows[key]["metrics"][METRICS[0]]["estimate"]
    diff = common[best]["paired_r1_vs_original"]
    comparison = (f"The best tested alternative on the common gallery was **{ARMS[best][0]}** "
                  f"at **{r1(common, best):.1%}** versus **{r1(common, BASELINE):.1%}** for the original V7 pooling. "
                  f"The paired change was **{diff['difference'] * 100:+.1f} percentage points** "
                  f"(95% interval {diff['lower_95'] * 100:+.1f} to {diff['upper_95'] * 100:+.1f}).")
    lines = ["# Vital JEPA V7 representation sweep", "", comparison, "",
             "Each protocol searched 129 candidates for 6,450 queries from 129 preset families and 50 distinct MIDI performances. "
             "All scores use the same Issue 167 recordings and gallery assignments. Higher recall is better.", "",
             "| Representation | Dimensions | Common R@1 / 5 / 10 | Mixed R@1 / 5 / 10 |",
             "| --- | ---: | ---: | ---: |"]
    for key, (label, dim) in {**ARMS, **{key: (label, "—") for key, label in ANCHORS.items()}}.items():
        scores = [" / ".join(f"{rows[key]['metrics'][metric]['estimate']:.1%}" for metric in METRICS) for rows in (common, mixed)]
        lines.append(f"| {label} | {dim} | {scores[0]} | {scores[1]} |")
    lines.extend(["", "![Recall@1 and paired changes](figures/issue169_v7_retrieval.png)", "", "## Evidence limits", "",
                  "The intervals resample preset families and MIDI performances and describe this cohort. "
                  "The Issue 167 corpus audit found 96 of 129 preset families matched parent hashes in V7 source corpora. "
                  "Exact training split membership remains unavailable. This is an exploratory comparison of frozen representations. "
                  "The substitute and sensitivity stages await the candidate selection.", "",
                  "## Execution and reproducibility", "",
                  "All retained WAVs passed their receipt hash checks. One encoder pass per WAV produced the cached 260 × 384 FP32 tensors. "
                  "The original pooling was replayed and compared with the Issue 167 per-query recall. CIR and CLAP reuse their Issue 167 ranks.", "",
                  "The checkpoint contains `aux_projection` and `control_delta_projection` with 128 outputs each, and distinct acoustic feature "
                  "and delta heads with 15 outputs each. The contrastive objectives and per-sound acoustic evaluation in the model lineage "
                  "support trying the contrastive and acoustic feature heads on one clip. The release leaves their exact V7 input and activation undocumented. "
                  "Both probes use the global mean plus patch mean described for V5, with the saved LayerNorm and linear layers and a local GELU choice. "
                  "[V5 model card](https://huggingface.co/duckking032/vital-jepa-v5)", "",
                  "The control projection is stored as `control_delta_projection`. Control ID, sign, magnitude and effect heads accept its "
                  "128-dimensional output by shape, and the released config uses mutation pairs. The release provides no forward code that establishes "
                  "whether this projection takes one clip or a mutation pair. To complete the requested retrieval rows, we apply it to each clip's "
                  "global-plus-patch mean. Treat it and both concatenations containing it as diagnostic only, since that single-clip input is unverified. "
                  "Pitch and articulation heads take 384 values "
                  "and are consistent with clip predictions. The release lists a spatial loss but the checkpoint contains no named spatial head. "
                  "The acoustic delta head likewise has no verified standalone input.", "",
                  "The requested eight representations are all included, along with a patch temporal pyramid because the existing encoders "
                  "also report that pooling. The temporal pyramid averages the eight mel patches at each time step and uses the prior eight-bin mean and population "
                  "standard deviation recipe. The bins use complete time patches up to the source duration (capped by the model's patch grid). "
                  "The plain patch mean uses the entire fixed grid, as in the original V7 pooling. Cached tokens have already attended across the model input.", "",
                  "No Vital controls are predicted or edited. [Release materials](" + RELEASE + ") · "
                  "[Issue 167 comparison](VITAL_JEPA_V7_COMPARISON.md) · [Machine-readable results](issue169_v7_retrieval_results.json)", ""])
    (reports / "VITAL_JEPA_V7_REPRESENTATION_SWEEP.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(reports / "issue169_v7_retrieval_results.json", {"retrieval": aggregate, "original_v7_replay_query_differences": replay,
                                                               "representations": ARMS, "provenance": provenance,
                                                               "diagnostic_only": ["control", "both_projections", "global_both"],
                                                               "decision": "await candidate selection"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for-tokens", action="store_true")
    args = parser.parse_args()
    while args.wait_for_tokens:
        state = read_json(RUN / "status.json")
        if state["phase"] == "Token cache ready" and state["state"] == "complete":
            break
        if state["state"] == "failed":
            raise RuntimeError("Token worker failed. See tokens.stderr.log.")
        import psutil

        if not psutil.pid_exists(state["pid"]):
            raise RuntimeError("Token worker exited without completing the cache")
        sleep(3)
    try:
        episodes, candidates, requests, receipts = cohort()
        request_ids, vectors, provenance = derive(receipts)
        aggregate, replay = score(request_ids, vectors, episodes, candidates, requests)
        status("Publish retrieval report", 0, 1)
        publish(aggregate, replay, provenance)
    except Exception:
        status("Retrieval analysis failed", 0, 1, state="failed")
        raise
    status("Retrieval ready for review", 1, 1, state="complete")


if __name__ == "__main__":
    main()

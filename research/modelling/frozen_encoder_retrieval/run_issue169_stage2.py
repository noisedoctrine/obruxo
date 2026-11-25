"""Evaluate the selected V7 representations on frozen acoustic substitutes and preset/MIDI sensitivity."""

import json
from datetime import datetime, timezone
import os
from pathlib import Path
import tempfile

import numpy as np

from frozen_retrieval.core import digest
from frozen_retrieval.encoders import file_sha256, list_clip_cache
from frozen_retrieval.evaluation import _evaluate_condition
from frozen_retrieval.issue161_substitution import SEED, challenger_reference_choices, verify_triplet_wavs
from frozen_retrieval.rendering import accepted_receipts
from frozen_retrieval.sensitivity import (BOOTSTRAP_SEED, _paired_effect_summary, normalize_embedding_grid,
                                           summarize_arm)
from run_issue167_v7 import (CHECKPOINTS, DATASET, OUTPUTS, ROOT, read_json, read_rows, write_json)
from run_issue169_tokens import RUN as ISSUE169_RUN, cohort
from run_issue169_patch_token_sweep import episode_indices
from run_issue164_normalization import _metric_summary


OUT = OUTPUTS / "issue169-v7-stage2"
STATUS = OUT / "status.json"
LOG = OUT / "stage2.log"
REPORTS = ROOT / "reports"
CONDITIONS = ("cross_midi_common_gallery", "cross_midi_mixed_gallery")
ARMS = ("top7_mean", "contrastive", "patch_mean", "original_v7", "cir", "clap")
NAMES = {"top7_mean": "Top-7 patch-token mean", "contrastive": "Contrastive projection",
         "patch_mean": "Patch-token mean", "original_v7": "Original V7 pooling", "cir": "CIR", "clap": "CLAP"}
RETRIEVAL_KEYS = {"top7_mean": ("topn", "mean_7"), "contrastive": ("v7", "contrastive"),
                  "patch_mean": ("v7", "patch_mean"), "original_v7": ("v7", "original_v7"),
                  "cir": ("v7", "cir_ast:official_clip"), "clap": ("v7", "clap_music:official_clip")}
VARIANTS = {"vital_jepa_v7": ("top7_mean", "contrastive", "patch_mean", "original_v7"),
            "cir_ast": ("official_clip",), "clap_music": ("official_clip",)}
MODEL_FOR_ARM = {"top7_mean": "vital_jepa_v7", "contrastive": "vital_jepa_v7", "patch_mean": "vital_jepa_v7",
                 "original_v7": "vital_jepa_v7", "cir": "cir_ast", "clap": "clap_music"}
VARIANT_FOR_ARM = {"top7_mean": "top7_mean", "contrastive": "contrastive", "patch_mean": "patch_mean",
                   "original_v7": "original_v7", "cir": "official_clip", "clap": "official_clip"}


def status(phase, completed, total, *, state="running", detail=""):
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload = {"phase": phase, "completed": completed, "total": total, "failures": int(state == "failed"),
               "updated_at": stamp, "state": state, "pid": os.getpid(), "detail": detail}
    scratch = STATUS.with_suffix(".tmp")
    scratch.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    scratch.replace(STATUS)
    with LOG.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(f"{stamp} {phase} {completed}/{total} {state} {detail}\n")
    print(f"{stamp} {phase} {completed}/{total} {state} {detail}", flush=True)


def load_vectors(request_ids, receipts):
    identity = read_json(ISSUE169_RUN / "vectors" / "identity.json")
    if identity["request_ids"] != request_ids:
        raise ValueError("Derived V7 vectors are not aligned with the verified Issue 167 requests")
    cache_id = read_json(ISSUE169_RUN / "token-cache.json")["cache_id"]
    token_identity = read_json(ISSUE169_RUN / "tokens" / cache_id / "identity.json")
    if token_identity["rows"] != [[key, receipts[key]["audio_sha256"]] for key in request_ids]:
        raise ValueError("Token cache differs from the accepted Issue 167 WAV receipts")
    features = {}
    for name in ("contrastive", "patch_mean", "original_v7"):
        array = np.load(ISSUE169_RUN / "vectors" / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        if array.shape != (len(request_ids), 384 if name != "contrastive" else 128) or not np.isfinite(array).all():
            raise ValueError(f"Invalid cached {name} vector array")
        features[name] = array
    tokens = np.load(ISSUE169_RUN / "patch-token-sweep" / "tokens.npy", mmap_mode="r", allow_pickle=False)
    if tokens.shape != (len(request_ids), 256, 384):
        raise ValueError("Cached V7 patch-token array has the wrong shape")
    top_positions = read_json(REPORTS / "issue169_v7_patch_oos.json")["folds"][0]["selected_positions"][:7]
    features["top7_mean"] = tokens[:, top_positions, :].mean(axis=1)

    feature_root = DATASET / "features"
    lock = read_json(CHECKPOINTS / "checkpoint-lock.json")
    for arm, model in (("cir", "cir_ast"), ("clap", "clap_music")):
        directory = feature_root / model
        manifest = read_rows(directory / "feature_manifest.private.jsonl")
        by_request = {row["request_id"]: row for row in manifest}
        if set(by_request) != set(request_ids) or any(by_request[key]["audio_sha256"] != receipts[key]["audio_sha256"] for key in request_ids):
            raise ValueError(f"Cached {model} embeddings differ from the accepted Issue 167 WAV set")
        cache = read_json(directory / "extraction.latest.json")
        loaded = list_clip_cache(directory, model, feature_ids={row["feature_id"] for row in manifest},
                                 expected_checkpoint_sha256=cache["checkpoint_sha256"])
        variant = "official_clip"
        features[arm] = np.stack([loaded[request][variant] for request in request_ids])
        if features[arm].shape != (len(request_ids), int(cache["dimensions_by_variant"][variant])):
            raise ValueError(f"Cached {model} embeddings have the wrong shape")
    for arm, matrix in features.items():
        if matrix.dtype != np.float32 or not np.isfinite(matrix).all() or np.any(np.linalg.norm(matrix, axis=1) <= 1e-12):
            raise ValueError(f"Invalid or zero-norm {arm} embeddings")
    return features, tokens, lock


def build_grids(features, request_ids, request_index, preset_ids, midi_ids, request_by_pair):
    grids = {}
    for arm in ARMS:
        matrix = features[arm]
        grid = np.empty((len(preset_ids), len(midi_ids), matrix.shape[1]), dtype=np.float32)
        for i, state_id in enumerate(preset_ids):
            for j, midi_id in enumerate(midi_ids):
                grid[i, j] = matrix[request_index[request_by_pair[(state_id, midi_id)]]]
        grids[arm] = grid
    return grids


def select_substitutes(grids, triplets, preset_ids, family_by_state, midi_ids):
    state_index = {state: i for i, state in enumerate(preset_ids)}
    midi_index = {midi: i for i, midi in enumerate(midi_ids)}
    output = {}
    for arm, grid in grids.items():
        unit = grid / np.linalg.norm(grid, axis=-1, keepdims=True).clip(1e-12)
        rows = []
        for triplet in triplets["triplets"]:
            midi1, midi2 = (role["midi_id"] for role in triplet["roles"][:2])
            similarities = unit[:, midi_index[midi1]] @ unit[:, midi_index[midi2]].T
            for target_index, target_id in enumerate(preset_ids):
                eligible = [i for i, state in enumerate(preset_ids) if family_by_state[state] != family_by_state[target_id]]
                if len(eligible) != 128:
                    raise ValueError("Substitute pool must contain 128 presets from other families")
                chosen = min(eligible, key=lambda i: (-float(similarities[target_index, i]), preset_ids[i]))
                rows.append({"triplet_index": triplet["triplet_index"], "target_family_id": family_by_state[target_id],
                             "target_state_id": target_id, "candidate_state_id": preset_ids[chosen], "candidate_rank": 1,
                             "cosine_similarity": float(similarities[target_index, chosen]), "eligible_candidates": len(eligible),
                             "midi1_pattern": triplet["roles"][0]["pattern"], "midi2_pattern": triplet["roles"][1]["pattern"],
                             "midi3_pattern": triplet["roles"][2]["pattern"], "midi3_register": triplet["roles"][2]["register"],
                             "midi3_articulation": triplet["roles"][2]["articulation"]})
        if len(rows) != 129 * len(triplets["triplets"]):
            raise ValueError("Substitute selection did not cover every preset and triplet")
        output[arm] = rows
    return output


def acoustic_results(choices, triplets, preset_ids, family_by_state, matrices):
    triplet_count = len(triplets["triplets"])
    refs = challenger_reference_choices(preset_ids, family_by_state, matrices, list(range(triplet_count)), seed=SEED)
    index_by_state = {state: i for i, state in enumerate(preset_ids)}
    rng = np.random.default_rng(SEED)
    family_draws = rng.integers(len(preset_ids), size=(2000, len(preset_ids)), dtype=np.int16)
    triplet_draws = rng.integers(triplet_count, size=(2000, triplet_count), dtype=np.int8)

    def score(rows):
        errors = np.full((len(preset_ids), triplet_count), np.nan)
        ranks = np.full_like(errors, np.nan)
        for triplet_index, target, candidate in rows:
            target_index, candidate_index = index_by_state[target], index_by_state[candidate]
            candidates = np.delete(matrices[triplet_index][target_index], target_index)
            value = matrices[triplet_index][target_index, candidate_index]
            errors[target_index, triplet_index] = value
            ranks[target_index, triplet_index] = np.sum(candidates < value - 1e-8) + (np.sum(np.abs(candidates - value) <= 1e-8) + 1) / 2
        if not np.isfinite(errors).all():
            raise ValueError("Acoustic scoring did not cover all 1,548 target/triplet choices")
        return errors, ranks

    summary = lambda values: _metric_summary(values, family_draws, triplet_draws)
    oracle, _ = score((triplet, target, row["oracle_state_id"]) for (triplet, target), row in refs.items())
    random, _ = score((triplet, target, row["random_state_id"]) for (triplet, target), row in refs.items())
    scores = {arm: score((row["triplet_index"], row["target_state_id"], row["candidate_state_id"]) for row in rows)
              for arm, rows in choices.items()}
    return {"matrix_triplets_reused": triplet_count, "matrix_triplets_rebuilt": 0,
            "oracle": summary(oracle), "random": summary(random),
            "arms": {arm: {"error": summary(error), "excess_over_oracle": summary(error - oracle), "rank": summary(rank)}
                     for arm, (error, rank) in scores.items()},
            "paired_error_differences": {arm: {base: summary(scores[arm][0] - scores[base][0])
                                               for base in ARMS if base != arm}
                                         for arm in ARMS}}


def run_sensitivity(grids, model_caches, retrieval_r1, preset_ids, midi_ids, family_ids, family_by_state):
    import torch
    import frozen_retrieval.sensitivity as sensitivity

    if not torch.xpu.is_available():
        raise RuntimeError("The pinned Intel XPU is required for the frozen sensitivity analysis")
    arm_grids = {arm: torch.from_numpy(grids[arm].copy()).to("xpu") for arm in ARMS}
    memory_grids = {arm: arm_grids[arm] for arm in ARMS}
    original_loader = sensitivity.load_embedding_grid

    def load_memory_grid(cache_audit, model_name, variant, *, device):
        arm = next(name for name in ARMS if MODEL_FOR_ARM[name] == model_name and VARIANT_FOR_ARM[name] == variant)
        return normalize_embedding_grid(memory_grids[arm].to(device))

    sensitivity.load_embedding_grid = load_memory_grid
    audit = {"model_caches": model_caches, "preset_ids": preset_ids, "midi_ids": midi_ids, "family_ids": family_ids,
             "family_by_state": family_by_state, "common_cells": len(preset_ids) * len(midi_ids),
             "retrieval_r1": retrieval_r1, "expected_arms": [f"{model}:{variant}" for model, variants in VARIANTS.items() for variant in variants],
             "cache_entry_counts": [len(preset_ids) * len(midi_ids)], "source_split": "validation",
             "performance_source_split": "train", "reserved_split_access": "none"}
    # The standard analyzer is reused unchanged; only its cache reader is redirected to the verified in-memory grids.
    completed = iter(range(1, len(ARMS) + 1))
    try:
        result = sensitivity.analyze_cache(audit, device_name="xpu", bootstrap_replicates=1000,
                                           bootstrap_combinations=32768, variants_by_model=VARIANTS,
                                           progress=lambda message: status("Preset/MIDI sensitivity", next(completed), len(ARMS), detail=message))
    finally:
        sensitivity.load_embedding_grid = original_loader
    del arm_grids, memory_grids
    return result


def publish(result):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    retrieval = result["retrieval"]
    acoustic = result["acoustic"]
    sensitivity = result["sensitivity"]
    figure = REPORTS / "figures/issue169_v7_stage2.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    labels = [NAMES[arm] for arm in ARMS]
    arms = list(ARMS)
    axes[0].barh(labels, [retrieval[arm]["common_r1"] * 100 for arm in arms], color="#2878a5")
    axes[0].invert_yaxis()
    axes[0].set(title="Exact preset retrieval", xlabel="Common-gallery Recall@1 (%)", xlim=(0, 100))
    errors = [acoustic["arms"][arm]["error"] for arm in arms]
    means = [row["mean"] for row in errors]
    lower = [row["clustered_95_ci"][0] for row in errors]
    upper = [row["clustered_95_ci"][1] for row in errors]
    axes[1].barh(labels, means, color="#d39025")
    axes[1].errorbar(means, np.arange(len(arms)), xerr=[np.array(means) - lower, np.array(upper) - means],
                     fmt="none", ecolor="#252525", capsize=3)
    axes[1].invert_yaxis()
    axes[1].set(title="Held-out substitute quality (lower is better)", xlabel="Active-RMS-matched MR-STFT error")
    fig.savefig(figure, dpi=170, bbox_inches="tight")
    plt.close(fig)

    lines = ["# Issue 169 Stage 2: acoustic substitutes and sensitivity", "",
             "Of the three selected V7 representations, the top-7 patch-token mean ranked best for exact preset retrieval and had the lowest substitute error, narrowly ahead of the contrastive projection and patch-token mean. CIR remained stronger on both exact retrieval and the spectral substitute score. Across all six representations, preset changes usually moved embeddings farther than MIDI changes.", "",
             "Retrieval covers 6,450 queries from 129 presets × 50 MIDI performances. Substitute selection covers 129 preset families across 12 fixed MIDI triplets (1,548 selections per representation): MIDI 1 and 2 select an alternative from 128 different-family presets, and MIDI 3 scores it. Lower active-RMS-matched MR-STFT error means a closer spectral match on MIDI 3. Intervals resample preset families and triplets for acoustic error, and preset families and MIDI performances for sensitivity.", "",
             "| Representation | Dimensions | Common R@1 | Mixed R@1 | Substitute error (95% interval) | Mean rank / 128 |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for arm in ARMS:
        error = acoustic["arms"][arm]["error"]
        lo, hi = error["clustered_95_ci"]
        lines.append(f"| {NAMES[arm]} | {result['dimensions'][arm]} | {retrieval[arm]['common_r1']:.1%} | {retrieval[arm]['mixed_r1']:.1%} | "
                     f"{error['mean']:.3f} [{lo:.3f}, {hi:.3f}] | {acoustic['arms'][arm]['rank']['mean']:.1f} |")
    lines += ["", f"Best available substitute: {acoustic['oracle']['mean']:.3f}. Fixed random choice: {acoustic['random']['mean']:.3f}.", "",
              "![Retrieval and substitute results](figures/issue169_v7_stage2.png)", "",
              "## Paired substitute differences", "",
              "Negative error difference means the first representation selected a closer-sounding substitute than the comparator.", "",
              "| Selected representation | Comparator | Error difference (95% interval) |", "| --- | --- | ---: |"]
    for selected in ("top7_mean", "contrastive", "patch_mean"):
        for comparator in ("original_v7", "cir", "clap"):
            row = acoustic["paired_error_differences"][selected][comparator]
            lo, hi = row["clustered_95_ci"]
            lines.append(f"| {NAMES[selected]} | {NAMES[comparator]} | {row['mean']:+.3f} [{lo:+.3f}, {hi:+.3f}] |")
    lines += ["", "## Preset and MIDI sensitivity", "",
              "Preset-dominant share is the portion of anchored comparisons where changing the preset moved the embedding farther than changing the MIDI. Higher share means preset identity more often dominates performance changes; it does not by itself guarantee good substitute retrieval.", "",
              "| Representation | Preset distance | MIDI distance | Normalized margin | Preset-dominant share | Variance from preset / MIDI / interaction |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for arm in ARMS:
        row = sensitivity["arms"][f"{MODEL_FOR_ARM[arm]}:{VARIANT_FOR_ARM[arm]}"]
        ci = row["confidence_intervals_95"]
        shares = row["variance_decomposition"]["shares"]
        share_ci = ci.get("preset_dominant_fraction")
        share_text = f"{row['preset_dominant_fraction']:.1%}"
        if share_ci:
            share_text += f" [{share_ci[0]:.1%}, {share_ci[1]:.1%}]"
        lines.append(f"| {NAMES[arm]} | {row['mean_preset_change_distance']:.3f} [{ci['mean_preset_distance'][0]:.3f}, {ci['mean_preset_distance'][1]:.3f}] | "
                     f"{row['mean_midi_change_distance']:.3f} [{ci['mean_midi_distance'][0]:.3f}, {ci['mean_midi_distance'][1]:.3f}] | "
                     f"{row['mean_normalized_margin']:+.3f} [{ci['mean_margin'][0]:+.3f}, {ci['mean_margin'][1]:+.3f}] | {share_text} | "
                     f"{shares['preset']:.1%} / {shares['midi']:.1%} / {shares['interaction']:.1%} |")
    lines += ["", "## What this supports", "",
              "Among V7 candidates, carry the top-7 patch-token mean forward if choosing one for the next V7-specific evaluation; retain CIR as the stronger external reference. The top-7 error interval overlaps those for the contrastive projection, patch-token mean, original V7, and CLAP, so the small error differences do not establish that it sounds better. These are frozen-embedding comparisons, not patch reconstruction or listening tests. The acoustic score is a spectral proxy. "
              "The split and five-fold retrieval review support using unseen query performances, but 96 of 129 evaluation preset families have parent hashes in the published V7 source corpora, and exact training membership is unavailable. Treat these results as exploratory for unseen presets.", "",
              "No audio was rendered and no encoder was run in this stage. The three selected V7 representations were derived from the verified cached encoder outputs; CIR, CLAP, original V7 vectors, triplets, WAV receipts, and all 12 active-RMS acoustic matrices were reused.", "",
              "No Vital controls are predicted or edited. [Canonical frozen-control inventory](vital_control_inventory.jsonl) · "
              "[Retrieval stage](VITAL_JEPA_V7_REPRESENTATION_SWEEP.md) · [Machine-readable Stage 2 results](issue169_v7_stage2_results.json)", ""]
    (REPORTS / "VITAL_JEPA_V7_STAGE2.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(REPORTS / "issue169_v7_stage2_results.json", result)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    status("Verify cached inputs", 0, 6)
    episodes, candidates, requests, receipts = cohort()
    request_ids = sorted(receipts)
    if len(request_ids) != 6450:
        raise ValueError("Expected the complete accepted 6,450-record Issue 167 cohort")
    request_index = {request: index for index, request in enumerate(request_ids)}
    features, tokens, lock = load_vectors(request_ids, receipts)
    episodes_path = DATASET / "episodes_qa_qualified.private.jsonl"
    episodes = read_rows(episodes_path)
    plan = read_rows(DATASET / "render_plan.private.jsonl")
    preset_ids = sorted({state for state, _ in requests})
    midi_ids = sorted({midi for _, midi in requests})
    family_by_state = {row["state_id"]: row["family_id"] for row in plan if row["state_id"] in preset_ids}
    if (len(preset_ids), len(midi_ids), len(set(family_by_state.values()))) != (129, 50, 129):
        raise ValueError("Preset/performance cohort or family count changed")
    retrieval_json = read_json(REPORTS / "issue169_v7_retrieval_results.json")
    topn_json = read_json(REPORTS / "issue169_v7_patch_topn.json")
    retrieval = {}
    for arm, (source, key) in RETRIEVAL_KEYS.items():
        if source == "topn":
            common = topn_json["results"][CONDITIONS[0]][key]["recall_at_1"]["estimate"]
            mixed = topn_json["results"][CONDITIONS[1]][key]["recall_at_1"]["estimate"]
        elif source == "v7" and key in {"contrastive", "patch_mean", "original_v7"}:
            common = retrieval_json["retrieval"][CONDITIONS[0]][key]["metrics"]["recall_at_1"]["estimate"]
            mixed = retrieval_json["retrieval"][CONDITIONS[1]][key]["metrics"]["recall_at_1"]["estimate"]
        else:
            common = retrieval_json["retrieval"][CONDITIONS[0]][key]["metrics"]["recall_at_1"]["estimate"]
            mixed = retrieval_json["retrieval"][CONDITIONS[1]][key]["metrics"]["recall_at_1"]["estimate"]
        retrieval[arm] = {"common_r1": common, "mixed_r1": mixed}
    dimensions = {arm: int(features[arm].shape[1]) for arm in ARMS}
    grids = build_grids(features, request_ids, request_index, preset_ids, midi_ids, requests)
    status("Verify cached inputs", 1, 6, detail="6,450 embeddings per arm")

    triplets = read_json(OUTPUTS / "issue161-audible-substitute" / "triplets.private.json")
    prior_triplets = read_json(OUTPUTS / "issue167-v7" / "triplets.private.json")
    if digest(prior_triplets) != digest(triplets):
        raise ValueError("Issue 167 acoustic matrices use a different triplet definition")
    if len(triplets["triplets"]) != 12:
        raise ValueError("The fixed 12-triplet Issue 161 cohort is required")
    triplet_midi_ids = {role["midi_id"] for row in triplets["triplets"] for role in row["roles"]}
    if not triplet_midi_ids.issubset(set(midi_ids)):
        raise ValueError("Triplet performances are missing from the verified cached vectors")
    choices = select_substitutes(grids, triplets, preset_ids, family_by_state, midi_ids)
    prior_choices = read_json(OUTPUTS / "issue167-v7" / "substitution-choices.private.json")["vital_jepa_v7:clip"]
    by_key = {(row["triplet_index"], row["target_state_id"]): row for row in prior_choices}
    for row in choices["original_v7"]:
        previous = by_key[(row["triplet_index"], row["target_state_id"])]
        if row["candidate_state_id"] != previous["candidate_state_id"] or abs(row["cosine_similarity"] - previous["cosine_similarity"]) > 1e-5:
            raise ValueError("Recomputed original V7 substitute choices differ from the frozen Issue 167 choices")
    write_json(OUT / "substitution-choices.private.json", choices)
    status("Select preset alternatives", 2, 6, detail="1,548 selections per representation")

    audit = {"preset_ids": preset_ids, "family_by_state": family_by_state, "request_by_pair": requests,
             "receipts": receipts, "preset_count": len(preset_ids)}
    triplet_midi3 = {row["roles"][2]["midi_id"] for row in triplets["triplets"]}
    audio, _ = verify_triplet_wavs(DATASET, audit, plan, triplets, midi_ids_override=triplet_midi3)
    matrices_path = OUTPUTS / "issue167-v7" / "active-rms-matrices.private.npz"
    with np.load(matrices_path, allow_pickle=False) as archive:
        expected_keys = [f"active_triplet_{i:02d}" for i in range(12)]
        if set(archive.files) != set(expected_keys):
            raise ValueError("Issue 167 acoustic matrix archive does not contain the exact 12 triplets")
        matrices = {i: archive[key].copy() for i, key in enumerate(expected_keys)}
    if any(matrix.shape != (129, 129) or not np.isfinite(matrix).all() or np.any(np.diag(matrix) != 0)
           for matrix in matrices.values()):
        raise ValueError("Issue 167 active-RMS matrix is incomplete or invalid")
    acoustic = acoustic_results(choices, triplets, preset_ids, family_by_state, matrices)
    status("Score acoustic substitutes", 3, 6, detail="reused all 12 Issue 167 matrices")

    model_caches = {}
    checkpoint_lock = read_json(CHECKPOINTS / "checkpoint-lock.json")
    checkpoint_lock["models"].update(read_json(ROOT / "issue167-checkpoint-lock.json")["models"])
    for model, variants in VARIANTS.items():
        model_caches[model] = {"directory": OUT, "rows_by_request": {},
            "dimensions_by_variant": {variant: dimensions[arm] for arm in ARMS
                                      if MODEL_FOR_ARM[arm] == model for variant in [VARIANT_FOR_ARM[arm]]},
            "checkpoint_sha256": checkpoint_lock["models"][model]["sha256"],
            "source_revision": checkpoint_lock["models"][model].get("revision"),
            "loader_revision": checkpoint_lock["models"][model].get("loader_revision", "cached feature adapter"),
            "tap": "cached frozen representation", "pooling": "per-arm pooling recorded in result metadata",
            "preprocessing": "verified Issue 167 audio receipts", "cache_entries": len(request_ids)}
    retrieval_r1 = {f"{MODEL_FOR_ARM[arm]}:{VARIANT_FOR_ARM[arm]}": retrieval[arm]["common_r1"] for arm in ARMS}
    families = sorted(set(family_by_state.values()))
    sensitivity = run_sensitivity(grids, model_caches, retrieval_r1, preset_ids, midi_ids, families, family_by_state)
    status("Preset/MIDI sensitivity", 4, 6, detail="completed 6 representation arms")

    result = {"protocol": {"stage": 2, "selected_representations": list(NAMES),
                            "retrieval_cohort": {"queries": 6450, "preset_families": 129, "midi_performances": 50},
                            "substitute_protocol": "12 fixed triplets; first two MIDI performances select among 128 different-family presets; third MIDI scores; active-RMS-matched MR-STFT",
                            "sensitivity_protocol": "exhaustive anchored comparison on 129 x 50 embeddings; crossed family x MIDI bootstrap",
                            "triplet_matrix_archive_sha256": file_sha256(matrices_path),
                            "triplet_definition_sha256": digest(triplets),
                            "triplets_reused": 12, "audio_rendered": 0, "encoder_inference_calls": 0},
              "dimensions": dimensions, "retrieval": retrieval, "acoustic": acoustic, "sensitivity": sensitivity}
    publish(result)
    status("Report ready", 6, 6, state="complete", detail="Stage 2 report and plots saved")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        status("Stage 2 failed", 0, 6, state="failed", detail=f"{type(error).__name__}: {error}")
        raise

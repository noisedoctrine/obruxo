"""Score Issue #167 on the regenerated frozen cohort and publish the comparison."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import sleep

import numpy as np

from frozen_retrieval.core import MODEL_REGISTRY, digest, paired_two_way_cluster_bootstrap, two_way_cluster_bootstrap
from frozen_retrieval.encoders import file_sha256, list_clip_cache
from frozen_retrieval.evaluation import _evaluate_condition
from frozen_retrieval.feature_stage import extract_dataset
from frozen_retrieval.issue161_substitution import (SEED, active_rms_matched_mrstft_matrix, challenger_reference_choices,
    load_scoring_midi, retrieve_alternatives, verify_triplet_wavs)
from frozen_retrieval.normalization import measure_active_rms
from frozen_retrieval.rendering import accepted_receipts
from frozen_retrieval.sensitivity import analyze_cache
from run_issue164_normalization import _metric_summary
from run_issue167_v7 import CHECKPOINTS, DATASET, MODELS, OUTPUTS, ROOT, RUN, SOURCE, STATUS, read_json, read_rows, status, write_json


BASELINE = "cir_ast:official_clip"
V7 = "vital_jepa_v7:clip"
VARIANTS = {key: (value,) for key, value in MODELS.items()}
CONDITIONS = ("cross_midi_common_gallery", "cross_midi_mixed_gallery")


def _wait_for_render():
    import psutil

    while True:
        state = read_json(STATUS)
        if state["state"] in {"failed", "incomplete"}:
            raise RuntimeError("The matched render stage requires recovery. See render status and logs.")
        if state["phase"] != "Render" or state["state"] == "complete":
            if read_json(RUN / "render-comparison.json")["accepted_requests"] != 6450:
                raise RuntimeError("The render stage did not retain the full frozen cohort")
            return
        if not psutil.pid_exists(state["pid"]):
            raise RuntimeError("Render worker exited before completing the frozen cohort")
        sleep(5)


def _load_cohort(lock_path):
    episodes = read_rows(DATASET / "episodes_qa_qualified.private.jsonl")
    candidates = read_json(DATASET / "candidate_sets.render_qualified.private.json")["candidate_sets"]
    plan = read_rows(DATASET / "render_plan.private.jsonl")
    pairs = {(row["target_state_id"], row["query_midi_id"]) for row in episodes if row["condition"] == CONDITIONS[0]}
    presets = sorted({preset for preset, _ in pairs})
    midis = sorted({midi for _, midi in pairs})
    assert (len(presets), len(midis), len(pairs)) == (129, 50, 6450)
    requests = {(row["state_id"], row["midi_id"]): row["request_id"] for row in plan if (row["state_id"], row["midi_id"]) in pairs}
    families = {row["state_id"]: row["family_id"] for row in plan if row["state_id"] in presets}
    receipts = accepted_receipts(DATASET)
    audit = {"preset_ids": presets, "midi_ids": midis, "family_ids": sorted(set(families.values())), "family_by_state": families,
             "common_cells": len(pairs), "request_by_pair": requests, "receipts": receipts, "source_split": "validation",
             "performance_source_split": "train", "reserved_split_access": "none", "model_caches": {}, "retrieval_r1": {},
             "cache_entry_counts": [len(receipts)], "expected_arms": [f"{model}:{variant}" for model, variant in MODELS.items()]}
    retrieval = {condition: {} for condition in CONDITIONS}
    model_rows = {condition: {} for condition in CONDITIONS}
    with (RUN / "episode-ranks.private.jsonl").open("w", encoding="utf-8") as private_rows:
        for index, (model, variant) in enumerate(MODELS.items()):
            phase = f"Embed {MODEL_REGISTRY[model]['name']}"
            status(phase, 0, len(receipts))
            extraction = extract_dataset(DATASET, model, CHECKPOINTS, lock_path, device="xpu",
                                         progress=lambda done, failures: status(phase, done, len(receipts), failures))
            directory = DATASET / "features" / model
            manifest = {row["request_id"]: row for row in read_rows(directory / "feature_manifest.private.jsonl")}
            if extraction["failure_count"] or set(manifest) != receipts.keys():
                raise RuntimeError(f"{model} is missing embeddings from the accepted render cohort")
            features = list_clip_cache(directory, model, feature_ids={row["feature_id"] for row in manifest.values()},
                                       expected_checkpoint_sha256=extraction["checkpoint_sha256"])
            if set(features) != set(manifest) or any(not np.isfinite(row[variant]).all() or np.linalg.norm(row[variant]) <= 1e-12
                                                     for row in features.values()):
                raise RuntimeError(f"{model} cache has missing or invalid embeddings")
            if any(row["audio_sha256"] != receipts[key]["audio_sha256"] for key, row in manifest.items()):
                raise RuntimeError(f"{model} embedding cache differs from the matched render receipts")
            audit["model_caches"][model] = {**extraction, "directory": directory, "rows_by_request": manifest}
            arm = f"{model}:{variant}"
            status("Cross-MIDI retrieval", index, len(MODELS))
            for condition in CONDITIONS:
                rows = _evaluate_condition(episodes, candidates, requests, features, variant, condition, {})
                if len(rows) != 6450 or {row["gallery_size"] for row in rows} != {129}:
                    raise RuntimeError("Retrieval denominator differs from the frozen galleries")
                for row in rows:
                    tied = row["tie_count"]
                    higher = row["rank_expected"] - (tied + 1) / 2
                    row["recall_at_10"] = min(max(10 - higher, 0), tied) / tied
                    private_rows.write(json.dumps({**row, "arm": arm}, separators=(",", ":")) + "\n")
                model_rows[condition][arm] = rows
                retrieval[condition][arm] = {metric: two_way_cluster_bootstrap(rows, metric=metric, seed=152)
                                             for metric in ("recall_at_1", "recall_at_5", "recall_at_10")}
            audit["retrieval_r1"][arm] = float(np.mean([row["recall_at_1"] for row in model_rows[CONDITIONS[0]][arm]]))
    paired = {condition: {arm: {metric: paired_two_way_cluster_bootstrap(model_rows[condition][V7], rows,
                                                                        metric=metric, seed=152)
                                for metric in ("recall_at_1", "recall_at_5", "recall_at_10")}
                          for arm, rows in model_rows[condition].items() if arm != V7} for condition in CONDITIONS}
    return audit, plan, {"metrics": retrieval, "v7_minus_comparator": paired}


def _acoustic_scores(audit, plan, triplets):
    picks = retrieve_alternatives(audit, triplets, variants_by_model=VARIANTS)
    write_json(RUN / "substitution-choices.private.json", picks)
    presets, families = audit["preset_ids"], audit["family_by_state"]
    old_receipts = accepted_receipts(SOURCE)
    original_dir = OUTPUTS / "issue164-active-rms"
    old_matrix_path = original_dir / "active_rms_acoustic_matrices.private.npz"
    contract = read_json(original_dir / "active_rms_acoustic_contract.private.json")
    held_out = [triplet["roles"][2]["midi_id"] for triplet in triplets["triplets"]]
    requests = audit["request_by_pair"]
    old_hashes = {requests[(preset, midi)]: old_receipts[requests[(preset, midi)]]["audio_sha256"]
                  for midi in held_out for preset in presets}
    if (file_sha256(old_matrix_path) != contract["matrix_archive_sha256"] or digest(triplets) != contract["triplets_sha256"] or
            digest(old_hashes) != contract["scoring_wav_hashes_sha256"]):
        raise RuntimeError("Historical active-RMS matrix archive differs from its frozen contract")
    audio, _ = verify_triplet_wavs(DATASET, audit, plan, triplets, midi_ids_override=set(held_out))
    matrices, reused = {}, 0
    with np.load(old_matrix_path, allow_pickle=False) as old, TemporaryDirectory(prefix="stft-", dir=RUN) as scratch:
        for index, midi in enumerate(held_out):
            status("Acoustic substitution", index, len(held_out))
            if all(audio[(preset, midi)]["receipt"]["audio_sha256"] == old_hashes[requests[(preset, midi)]] for preset in presets):
                matrices[index] = old[f"active_triplet_{index:02d}"]
                reused += 1
                continue
            clips = load_scoring_midi(audio, presets, midi)
            levels = [measure_active_rms(clip, 44_100)["active_rms"] for clip in clips]
            matrices[index], _ = active_rms_matched_mrstft_matrix(clips, levels, output_dir=Path(scratch), batch_size=8,
                                                                 pair_batch_size=64, device_name="xpu")
    np.savez_compressed(RUN / "active-rms-matrices.private.npz", **{f"active_triplet_{i:02d}": matrix for i, matrix in matrices.items()})
    refs = challenger_reference_choices(presets, families, matrices, list(range(len(held_out))), seed=SEED)
    index_by_state = {state: index for index, state in enumerate(presets)}
    rng = np.random.default_rng(SEED)
    family_draws = rng.integers(len(presets), size=(2000, len(presets)), dtype=np.int16)
    triplet_draws = rng.integers(len(held_out), size=(2000, len(held_out)), dtype=np.int8)

    def score(rows):
        errors = np.full((len(presets), len(held_out)), np.nan)
        ranks = np.full_like(errors, np.nan)
        for triplet, target, chosen in rows:
            target_index, choice_index = index_by_state[target], index_by_state[chosen]
            candidates = np.delete(matrices[triplet][target_index], target_index)
            value = matrices[triplet][target_index, choice_index]
            errors[target_index, triplet] = value
            ranks[target_index, triplet] = np.sum(candidates < value - 1e-8) + (np.sum(np.abs(candidates - value) <= 1e-8) + 1) / 2
        if not np.isfinite(errors).all():
            raise RuntimeError("Acoustic scores do not cover all 1,548 frozen choices")
        return errors, ranks

    summary = lambda values: _metric_summary(values, family_draws, triplet_draws)
    oracle, _ = score((triplet, target, row["oracle_state_id"]) for (triplet, target), row in refs.items())
    random, _ = score((triplet, target, row["random_state_id"]) for (triplet, target), row in refs.items())
    values = {arm: score((row["triplet_index"], row["target_state_id"], row["candidate_state_id"]) for row in rows)
              for arm, rows in picks.items()}
    return {"matrix_triplets_reused": reused, "matrix_triplets_rebuilt": len(held_out) - reused,
            "oracle": summary(oracle), "random": summary(random),
            "arms": {arm: {"error": summary(error), "excess_over_oracle": summary(error - oracle), "rank": summary(rank)}
                     for arm, (error, rank) in values.items()},
            "v7_minus_comparator": {arm: summary(values[V7][0] - error) for arm, (error, _) in values.items() if arm != V7}}


def _resources(lock_path, receipts):
    sample = read_json(OUTPUTS / "model-efficiency-benchmark/sample.private.json")
    for record in sample["records"]:
        receipt = receipts[record["request_id"]]
        if (record["wav_relative_path"] != receipt["wav_relative_path"] or
                abs(record["duration_seconds"] - receipt["duration_seconds"]) > 1e-6):
            raise RuntimeError("Resource sample differs from the original fixed recording list or durations")
    sample_path = RUN / "resource-sample.private.json"
    write_json(sample_path, sample)
    results = {}
    for index, model in enumerate(MODELS):
        status("Resource benchmark", index, len(MODELS))
        destination = RUN / f"resource-{model}.private.json"
        subprocess.run([sys.executable, "-B", "-u", str(ROOT / "run_efficiency_benchmark.py"), "--worker", model,
                        "--dataset", str(DATASET), "--sample-manifest", str(sample_path), "--checkpoint-root", str(CHECKPOINTS),
                        "--lock", str(lock_path), "--raw-output", str(destination)], check=True)
        results[model] = read_json(destination)["result"]
        if not results[model]["all_inputs_processed"]:
            raise RuntimeError(f"{model} did not complete all four warm-ups and 384 timed calls")
    return results


def _publish(result):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report_dir = ROOT / "reports"
    report_path = report_dir / "VITAL_JEPA_V7_COMPARISON.md"
    json_path = report_dir / "issue167_v7_results.json"
    figure_path = report_dir / "figures/issue167_v7_comparison.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    names = {V7: "Vital JEPA V7", BASELINE: "CIR", "clap_music:official_clip": "CLAP"}
    common, mixed = (result["retrieval"]["metrics"][condition] for condition in CONDITIONS)
    acoustic, sensitivity, resources = (result[key] for key in ("acoustic", "sensitivity", "resources"))
    render = result["render"]
    best_retrieval = max(names, key=lambda arm: common[arm]["recall_at_1"]["estimate"])
    best_acoustic = min(names, key=lambda arm: acoustic["arms"][arm]["error"]["mean"])
    v7_r1 = common[V7]["recall_at_1"]["estimate"]
    v7_error = acoustic["arms"][V7]["error"]["mean"]
    paired = result["retrieval"]["v7_minus_comparator"][CONDITIONS[0]]
    worse_both = all(delta["recall_at_1"]["upper_95"] < 0 and acoustic["v7_minus_comparator"][arm]["clustered_95_ci"][0] > 0
                     for arm, delta in paired.items())
    if worse_both:
        recommendation = "Defer V7. Both comparators lead on paired retrieval and substitute error in this cohort."
    elif best_retrieval == V7 or best_acoustic == V7:
        recommendation = "Include V7 in an exploratory training arm if a parent-disjoint reconstruction evaluation is available."
    else:
        recommendation = "Defer V7 pending parent-disjoint evidence of an advantage for inverse-model training."
    lines = ["# Vital JEPA V7 compared with CIR and CLAP", "",
             f"V7 retrieved the exact preset first for **{v7_r1:.1%}** of common-gallery queries and had a mean substitute error of "
             f"**{v7_error:.3f}**. {names[best_retrieval]} had the highest Recall@1 and {names[best_acoustic]} had the lowest substitute error. "
             f"{recommendation}", "",
             "Retrieval used 129 preset families and 50 distinct MIDI performances, for 6,450 queries and 129 candidates per gallery protocol. "
             "Substitution used 1,548 choices across 12 triplets and 36 distinct MIDI performances. Lower active-RMS-matched MR-STFT means "
             "a closer spectral match on the withheld performance.", "",
             "| Encoder | Common R@1 (95% CI) | R@5 | R@10 | Mixed R@1 / 5 / 10 | Substitute error (95% CI) | Substitute mean rank / 128 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm, name in names.items():
        first, other = common[arm]["recall_at_1"], mixed[arm]
        error = acoustic["arms"][arm]["error"]
        lo, hi = error["clustered_95_ci"]
        mixed_recall = " / ".join(f"{other[f'recall_at_{k}']['estimate']:.1%}" for k in (1, 5, 10))
        lines.append(f"| {name} | {first['estimate']:.1%} [{first['lower_95']:.1%}, {first['upper_95']:.1%}] | "
                     f"{common[arm]['recall_at_5']['estimate']:.1%} | {common[arm]['recall_at_10']['estimate']:.1%} | "
                     f"{mixed_recall} | {error['mean']:.3f} [{lo:.3f}, {hi:.3f}] | {acoustic['arms'][arm]['rank']['mean']:.1f} |")
    lines += ["", f"The best available substitute under the scorer averaged {acoustic['oracle']['mean']:.3f}. "
              f"The fixed random choice averaged {acoustic['random']['mean']:.3f}.", ""]
    for arm, delta in paired.items():
        r1, error = delta["recall_at_1"], acoustic["v7_minus_comparator"][arm]
        low, high = error["clustered_95_ci"]
        lines.append(f"- V7 vs. {names[arm]} had a paired Recall@1 difference of {100 * r1['difference']:+.1f} pp "
                     f"[{100 * r1['lower_95']:+.1f}, {100 * r1['upper_95']:+.1f}] and substitute error difference of "
                     f"{error['mean']:+.3f} [{low:+.3f}, {high:+.3f}].")
    lines += ["", "## Preset and MIDI sensitivity", "",
              "Preset dominance is the share of anchored comparisons where changing the preset moved the embedding farther than changing the MIDI.", "",
              "| Encoder | Preset distance | MIDI distance | Normalized margin | Preset-dominant share | Preset / MIDI / interaction variance |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for arm, name in names.items():
        point = sensitivity["arms"][arm]
        shares = point["variance_decomposition"]["shares"]
        variance = " / ".join(f"{shares[key]:.1%}" for key in ("preset", "midi", "interaction"))
        lines.append(f"| {name} | {point['mean_preset_change_distance']:.3f} | {point['mean_midi_change_distance']:.3f} | "
                     f"{point['mean_normalized_margin']:+.3f} | {point['preset_dominant_fraction']:.1%} | {variance} |")
    lines += ["", f"![Retrieval and substitute comparison](figures/{figure_path.name})", "",
              "## Evidence limits", "",
              "In each of the two published V7 source corpora, 96 of the 129 evaluation families matched a parent preset SHA-256. "
              "Published manifests lack exact training split membership. The inherited V4 source corpus also had 96 parent matches. "
              "The intervals resample preset families and MIDI performances for retrieval, and preset families and triplets for substitution. "
              "They cover sampling variation in this cohort. Reconstruction and listening quality remain unmeasured. "
              "[Published parent manifests](https://huggingface.co/datasets/duckking032/vital/tree/08e84bcd0b4f8ee90e493d26780853e025e81df6)", "",
              "## Execution and reproducibility", "",
              "This is a frozen representation benchmark with zero predicted or editable Vital controls. "
              "[Canonical Vital inventory](vital_control_inventory.jsonl)", "",
              f"Recovered {render['accepted_requests']:,} matched recordings using the original renderer, assignments and QA. "
              f"{render['original_sha256_matches']:,} matched the historical WAV hashes. All encoders used the same recovered WAVs. "
              f"The acoustic scorer reused {acoustic['matrix_triplets_reused']} compatible triplet matrices and rebuilt {acoustic['matrix_triplets_rebuilt']}.", "",
              "V7 used the SHA-locked Apache-2.0 step 16,000 checkpoint. The local 384-dimensional adapter uses 22.05 kHz stereo, a leading "
              "six-second crop or zero padding, energy-preserving mid/side, Hann/HTK log-mel features, pre-norm GELU and residual mean pooling "
              "of global and patch tokens from the online encoder. Framing, mel conventions, transformer settings and pooling are local "
              "implementation choices based on the released config and tensor layout. [Pinned release](https://huggingface.co/duckking032/vital-jepa-v7/commit/e5ebdfb02dd107c15125b12c1bb4af03a7e31592)", "",
              "Resource measurement used the original 128-recording list, the Intel XPU, FP32, batch size 1, four warm-ups and three timed passes per encoder. "
              "V7's mel frontend runs on the CPU. Its encoder runs on the XPU. Runtime versions and checkpoint hashes are in the result JSON.", "",
              "| Encoder | Median / P90 latency (ms) | Throughput (audio s / wall s) | Peak process RAM (GiB) | Peak XPU allocation (GiB) |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for arm, name in names.items():
        resource = resources[arm.split(":")[0]]
        timing = resource["aggregate"]
        lines.append(f"| {name} | {timing['latency_median_ms']:.1f} / {timing['latency_p90_ms']:.1f} | "
                     f"{timing['audio_seconds_per_wall_second']:.1f} | {resource['peak_process_rss_bytes'] / 2**30:.2f} | "
                     f"{resource['peak_xpu_allocated_bytes'] / 2**30:.2f} |")
    lines += ["", f"[Machine-readable results]({json_path.name}) · [Benchmark commands](../README.md#issue-167-vital-jepa-v7)", "",
              "The run retains its assignments, WAV receipts, embeddings, acoustic score matrices and timing rows under the ignored Issue #167 output directory. "
              "The temporary STFT tensors and render timing probe WAVs were removed after their measurements were recorded.", ""]
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    axes[0].bar(list(names.values()), [100 * common[arm]["recall_at_1"]["estimate"] for arm in names])
    axes[0].set(ylabel="Recall@1 (%)", title="Exact preset retrieval", ylim=(0, 100))
    axes[1].bar(list(names.values()), [acoustic["arms"][arm]["error"]["mean"] for arm in names])
    axes[1].set(ylabel="Active-RMS-matched MR-STFT", title="Held-out substitute error")
    figure.savefig(figure_path, dpi=170)
    plt.close(figure)
    write_json(json_path, {**result, "recommendation": recommendation})
    report_path.write_text("\n".join(lines), encoding="utf-8")
    summary = report_dir / "FROZEN_ENCODER_RETRIEVAL_RESULTS.md"
    contents = summary.read_text(encoding="utf-8")
    link = f"[Issue #167 compares Vital JEPA V7 with CIR and CLAP]({report_path.name})."
    if link not in contents:
        summary.write_text(contents.replace("## Results and recommendation", f"## Results and recommendation\n\n{link}", 1), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for-render", action="store_true")
    args = parser.parse_args()
    if args.wait_for_render:
        _wait_for_render()
    render = read_json(RUN / "render-comparison.json")
    if render["accepted_requests"] != 6450:
        raise RuntimeError("Frozen-cohort render must complete before scoring")
    try:
        lock = read_json(CHECKPOINTS / "checkpoint-lock.json")
        lock["models"].update(read_json(ROOT / "issue167-checkpoint-lock.json")["models"])
        lock_path = RUN / "checkpoint-lock.private.json"
        write_json(lock_path, lock)
        audit, plan, retrieval = _load_cohort(lock_path)
        status("Preset and MIDI sensitivity", 0, len(MODELS))
        completed = iter(range(1, len(MODELS) + 1))
        sensitivity = analyze_cache(audit, device_name="xpu", variants_by_model=VARIANTS,
            progress=lambda message: (print(message, flush=True), status("Preset and MIDI sensitivity", next(completed), len(MODELS))))
        triplets = read_json(OUTPUTS / "issue161-audible-substitute/triplets.private.json")
        write_json(RUN / "triplets.private.json", triplets)
        acoustic = _acoustic_scores(audit, plan, triplets)
        resources = _resources(lock_path, audit["receipts"])
        status("Publish report", 0, 1)
        _publish({"retrieval": retrieval, "sensitivity": sensitivity, "acoustic": acoustic, "resources": resources,
                  "render": render, "training_corpus_overlap": {"evaluation_families": 129, "exact_parent_hash_matches": 96,
                      "dataset_revision": "08e84bcd0b4f8ee90e493d26780853e025e81df6", "train_split_membership": "unresolved"}})
        status("Publish report", 1, 1, state="complete")
    except Exception:
        state = read_json(STATUS)
        status(state["phase"], state["completed"], state["total"], state="failed", failures=max(state["failures"], 1))
        raise


if __name__ == "__main__":
    main()

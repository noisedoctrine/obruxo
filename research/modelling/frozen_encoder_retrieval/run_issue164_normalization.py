from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
sys.path[:0] = [str(ROOT), str(REPO / "research/data_generation"), str(REPO / "research/training"),
               str(REPO / "research/modelling/basic_pitch")]

from frozen_retrieval.core import MODEL_REGISTRY, canonical_json, digest, paired_two_way_cluster_bootstrap  # noqa: E402
from frozen_retrieval.dataset import validate_dataset_manifests  # noqa: E402
from frozen_retrieval.encoders import (dependency_status, file_sha256, load_adapter, load_checkpoint_lock,
    resolve_checkpoint, resolve_source_checkout)  # noqa: E402
from frozen_retrieval.evaluation import evaluate_dataset  # noqa: E402
from frozen_retrieval.issue161_substitution import (  # noqa: E402
    MODEL_ORDER, SEED as ISSUE161_SEED, active_rms_matched_mrstft_matrix,
    challenger_reference_choices, load_scoring_midi, retrieve_alternatives, verify_triplet_wavs,
)
from frozen_retrieval.normalization import (  # noqa: E402
    ACTIVE_FLOOR_DB, MAX_NORMALIZED_PEAK, MIN_ACTIVE_RMS, NORMALIZATION_VERSION, WINDOW_SECONDS,
    measure_active_rms, normalize_active_rms,
)
from frozen_retrieval.sensitivity import analyze_cache, verify_embedding_cache, _jsonl  # noqa: E402


DATASET = ROOT / "outputs" / "issue152-validation-train-midi-v3-50"
PRIVATE = ROOT / "outputs" / "issue164-active-rms"
NORMALIZED_ROOT = DATASET / "features_active_rms"
CHECKPOINT_ROOT = ROOT / "local_checkpoints"
LOCK_PATH = CHECKPOINT_ROOT / "checkpoint-lock.json"
PUBLIC_RESULT = ROOT / "reports" / "issue164_active_rms_results.json"
REPORT = ROOT / "reports" / "FROZEN_ENCODER_RETRIEVAL_RESULTS.md"
TARGET_FILE = PRIVATE / "preflight.private.json"
WAV_AUDIT_FILE = PRIVATE / "wav_identity.private.jsonl"
ACTIVE_MATRIX_FILE = PRIVATE / "active_rms_acoustic_matrices.private.npz"
ACTIVE_MATRIX_CONTRACT = PRIVATE / "active_rms_acoustic_contract.private.json"
TRIPLET_FILE = PRIVATE / "triplets.private.json"
CHOICES_FILE = PRIVATE / "retrieval_choices.private.json"
ISSUE161_DIR = ROOT / "outputs" / "issue161-audible-substitute"
ISSUE155_DIR = ROOT / "outputs" / "issue155-validation"


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _target_from_preflight(path: Path = TARGET_FILE) -> float:
    record = _read_json(path)
    if record.get("normalization", {}).get("version") != NORMALIZATION_VERSION:
        raise ValueError("issue #164 target file uses a different normalization contract")
    return float(record["normalization"]["target_active_rms"])


def _common_wav_rows(dataset: Path, cache_audit: dict) -> list[dict]:
    plans = {row["request_id"]: row for row in _jsonl(dataset / "render_plan.private.jsonl")}
    rows = []
    for (state_id, midi_id), request_id in sorted(cache_audit["request_by_pair"].items()):
        receipt = cache_audit["receipts"][request_id]
        plan = plans.get(request_id)
        if not plan or not receipt.get("accepted"):
            raise ValueError("common-cohort WAV is missing its accepted request or receipt")
        wav = dataset / receipt["wav_relative_path"]
        rows.append({"state_id": state_id, "midi_id": midi_id, "request_id": request_id,
                     "audio_sha256": receipt["audio_sha256"], "wav_path": wav,
                     "wav_relative_path": receipt["wav_relative_path"], "plan": plan})
    if len(rows) != 6_450:
        raise ValueError(f"expected the existing 6,450-WAV matrix; found {len(rows)}")
    return rows


def _accepted_wav_paths(dataset: Path, cache_audit: dict) -> dict[tuple[str, str], Path]:
    paths = {}
    for pair, request_id in cache_audit["request_by_pair"].items():
        receipt = cache_audit["receipts"].get(request_id)
        relative_path = receipt.get("wav_relative_path") if receipt else None
        if not receipt or not receipt.get("accepted") or not relative_path:
            raise ValueError(f"accepted common-cohort request {request_id} has no WAV path")
        paths[pair] = dataset / relative_path
    if len(paths) != len(cache_audit["request_by_pair"]):
        raise ValueError("accepted common-cohort WAV path map is incomplete")
    return paths


def _latency_medians(value) -> list[float]:
    found = []
    if isinstance(value, dict):
        if isinstance(value.get("latency_median_ms"), (float, int)):
            found.append(float(value["latency_median_ms"]))
        else:
            for child in value.values():
                found.extend(_latency_medians(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_latency_medians(child))
    return found


def _benchmark_normalizer(paths: list[Path], target: float) -> dict:
    from scipy.io import wavfile

    if len(paths) < 3:
        raise ValueError("normalizer benchmark needs representative 4–10 second stereo WAVs")
    selected = paths[:5]
    samples = []
    for path in selected:
        rate, audio = wavfile.read(path)
        if rate != 44_100 or audio.ndim != 2 or audio.shape[1] != 2 or not 4 <= len(audio) / rate <= 10:
            continue
        samples.append((rate, audio))
    if len(samples) < 3:
        raise ValueError("normalizer benchmark sample did not contain enough 4–10 second stereo clips")
    # First invocation compiles the Numba signatures; compilation is excluded from timed samples.
    for rate, audio in samples:
        normalize_active_rms(audio, rate, target)
    timings = []
    for rate, audio in samples:
        elapsed = []
        for _ in range(5):
            started = time.perf_counter()
            normalize_active_rms(audio, rate, target)
            elapsed.append(time.perf_counter() - started)
        timings.append({"duration_seconds": len(audio) / rate,
                        "median_seconds": statistics.median(elapsed),
                        "samples": len(audio), "channels": int(audio.shape[1])})
    benchmark_path = ROOT / "reports" / "model_efficiency_benchmark.json"
    encoder_latency = _latency_medians(_read_json(benchmark_path).get("models", {})) if benchmark_path.is_file() else []
    per_clip_ms = [row["median_seconds"] * 1000 for row in timings]
    return {"numba_compilation_included": False,
            "measured_operation": "active-RMS scan plus one-scalar gain application, excluding WAV read",
            "sample_count": len(timings), "timed_repeats_per_clip": 5,
            "normalization_median_ms": statistics.median(per_clip_ms),
            "normalization_min_ms": min(per_clip_ms), "normalization_max_ms": max(per_clip_ms),
            "normalization_audio_seconds_per_wall_second": statistics.median(
                row["duration_seconds"] / row["median_seconds"] for row in timings),
            "per_clip": timings,
            "issue160_adapter_latency_median_ms": ({"min": min(encoder_latency), "median": statistics.median(encoder_latency),
                                                       "max": max(encoder_latency), "samples": len(encoder_latency)}
                                                      if encoder_latency else None),
            "overhead_as_fraction_of_median_issue160_adapter_latency": (
                statistics.median(per_clip_ms) / statistics.median(encoder_latency)
                if encoder_latency else None)}


def preflight(dataset: Path = DATASET, checkpoint_root: Path = CHECKPOINT_ROOT) -> dict:
    from scipy.io import wavfile

    dataset = dataset.resolve(strict=True)
    checkpoint_root = checkpoint_root.resolve(strict=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    progress = lambda message: print(message, flush=True)
    audit = verify_embedding_cache(dataset, checkpoint_root, progress=progress)
    lock = load_checkpoint_lock(checkpoint_root / "checkpoint-lock.json")
    checkpoint_reuse = {}
    for model_name in MODEL_ORDER:
        progress(f"Verifying locked checkpoint bytes and source tree for {model_name}.")
        checkpoint, entry, checkpoint_digest = resolve_checkpoint(model_name, checkpoint_root, lock)
        source_root = resolve_source_checkout(checkpoint_root, entry)
        runtime = dependency_status(model_name)
        locked_runtime = entry["runtime_versions"]
        mismatches = {name: {"expected": expected, "actual": runtime.get(name, {}).get("version", "missing")}
                      for name, expected in locked_runtime.items()
                      if name in runtime and runtime[name]["version"] != expected}
        if model_name != "basic_pitch_trunk" and mismatches:
            raise RuntimeError(f"installed runtime differs from the locked environment for {model_name}: {mismatches}")
        checkpoint_reuse[model_name] = {"checkpoint_relative_path": checkpoint.relative_to(checkpoint_root).as_posix(),
            "checkpoint_sha256": checkpoint_digest, "source_revision": entry["revision"],
            "loader_revision": entry["loader_revision"],
            "source_root_relative_path": source_root.relative_to(checkpoint_root).as_posix() if source_root else None,
            "runtime_version_mismatches_in_bench_environment": mismatches}
    dataset_summary = _read_json(dataset / "dataset_summary.private.json")
    validate_dataset_manifests(dataset, dataset_summary)
    input_rows = _common_wav_rows(dataset, audit)

    triplets = _read_json(ISSUE161_DIR / "triplets.private.json")
    choices = _read_json(ISSUE161_DIR / "retrieval_choices.private.json")
    matrices_path = ISSUE161_DIR / "acoustic_matrices.private.npz"
    matrices = np.load(matrices_path, allow_pickle=False)
    expected_arms = {f"{model}:{variant}" for model, variants in audit["model_caches"].items()
                     for variant in audit["model_caches"][model]["dimensions_by_variant"]}
    if (triplets.get("seed") != ISSUE161_SEED or len(triplets.get("triplets", [])) != 12 or
            triplets.get("coverage", {}).get("unique_selected_performances") != 36 or
            set(choices) != expected_arms):
        raise ValueError("existing #161 triplet or retrieval-choice artifacts do not match the frozen contract")
    for arm, rows in choices.items():
        if len(rows) != 1_548 or any(row.get("eligible_candidates") != 128 for row in rows):
            raise ValueError(f"existing #161 retrieval choices are incomplete for {arm}")
    matrix_keys = {f"{prefix}_triplet_{index:02d}" for prefix in ("raw", "gain_matched") for index in range(12)}
    if set(matrices.files) != matrix_keys or any(matrices[key].shape != (129, 129) for key in matrix_keys):
        raise ValueError("existing #161 raw and whole-clip gain-matched score matrices are incomplete")
    prior_wav = _read_json(ISSUE161_DIR / "wav_verification.private.json")
    if prior_wav.get("selected_matrix_wavs_verified") != 4_644:
        raise ValueError("existing #161 selected MIDI triplet WAV audit is incomplete")
    issue155_rows = _jsonl(ISSUE155_DIR / "gallery_assignment_episode_metrics.private.jsonl")
    if len(issue155_rows) != 3 * 6_450 * 11:
        raise ValueError("existing #155 three-offset per-query baseline is missing or incomplete")
    issue157 = _read_json(ROOT / "reports" / "issue157_sensitivity_results.json")
    if set(issue157.get("arms", {})) != expected_arms:
        raise ValueError("existing #157 sensitivity results do not contain the same eleven representation arms")

    _write_json(TRIPLET_FILE, triplets)
    _write_json(CHOICES_FILE, choices)

    active_rms_values, peak_values, crest_values, durations = [], [], [], []
    source_rows = []
    for index, row in enumerate(input_rows, start=1):
        if file_sha256(row["wav_path"]) != row["audio_sha256"]:
            raise ValueError(f"cohort WAV hash differs from its accepted render receipt: {row['request_id']}")
        sample_rate, audio = wavfile.read(row["wav_path"])
        if (sample_rate != 44_100 or audio.dtype != np.float32 or audio.ndim != 2 or audio.shape[1] != 2 or
                not np.isfinite(audio).all()):
            raise ValueError(f"cohort WAV differs from the accepted stereo float32 contract: {row['request_id']}")
        measured = measure_active_rms(audio, sample_rate)
        active_rms = measured["active_rms"]
        if active_rms < MIN_ACTIVE_RMS:
            raise ValueError(f"cohort WAV is below the active-RMS floor: {row['request_id']}")
        active_rms_values.append(active_rms)
        peak_values.append(measured["peak_sample"])
        crest_values.append(measured["peak_to_active_rms"])
        duration = len(audio) / sample_rate
        durations.append(duration)
        source_rows.append({"request_id": row["request_id"], "audio_sha256": row["audio_sha256"],
            "sample_rate": sample_rate, "frames": len(audio), "duration_seconds": duration,
            "active_rms": active_rms, "peak_sample": measured["peak_sample"],
            "peak_to_active_rms": measured["peak_to_active_rms"],
            "source_peak_above_unit_range": bool(measured["peak_sample"] > 1.0),
            "active_window_count": measured["active_window_count"],
            "window_size_samples": measured["window_size_samples"]})
        if index % 500 == 0 or index == len(input_rows):
            progress(f"Verified source WAV identity and active RMS: {index:,}/{len(input_rows):,}.")

    max_crest = max(crest_values)
    target = MAX_NORMALIZED_PEAK / max_crest * (1.0 - 1e-6)
    if not math.isfinite(target) or target <= 0:
        raise ValueError("the shared target active-RMS level is not feasible for the cohort")
    durations_and_paths = sorted((duration, row["wav_path"]) for duration, row in zip(durations, input_rows, strict=True))
    sample_paths = [durations_and_paths[min(len(durations_and_paths) - 1, int(q * (len(durations_and_paths) - 1)))][1]
                    for q in (0.05, 0.25, 0.50, 0.75, 0.95)]
    benchmark = _benchmark_normalizer(sample_paths, target)
    normalized_peaks = np.asarray(peak_values) * target / np.asarray(active_rms_values)
    if float(normalized_peaks.max()) > MAX_NORMALIZED_PEAK:
        raise ArithmeticError("cohort-wide fixed target violates the shared peak ceiling")
    if TARGET_FILE.is_file():
        previous = _read_json(TARGET_FILE)
        if (previous.get("source_manifest_sha256") != dataset_summary.get("manifest_sha256") or
                previous.get("cohort_identity_sha256") != digest([(row["request_id"], row["audio_sha256"]) for row in input_rows]) or
                previous.get("normalization", {}).get("target_active_rms") != target):
            raise ValueError("existing issue #164 preflight target or WAV cohort identity has drifted")
    with WAV_AUDIT_FILE.open("w", encoding="utf-8", newline="\n") as stream:
        for row in source_rows:
            stream.write(canonical_json(row) + "\n")
    summary = {"schema": "issue164_active_rms_preflight_v1", "source_manifest_sha256": dataset_summary.get("manifest_sha256"),
        "cohort_identity_sha256": digest([(row["request_id"], row["audio_sha256"]) for row in input_rows]),
        "reuse_audit": {"source_wavs": 6_450, "source_wavs_rehashed_and_measured": 6_450,
            "source_wavs_modified": 0, "original_feature_caches_preserved": True,
            "original_feature_entries_per_family": audit["cache_entry_counts"],
            "original_cached_representation_arms": len(audit["expected_arms"]),
            "issue152_evaluation_reused": True, "issue155_three_offset_cases_reused": len(issue155_rows),
            "issue157_original_sensitivity_result_reused": True,
            "issue161_original_retrieval_choices_reused": sum(len(rows) for rows in choices.values()),
            "issue161_raw_score_matrices_reused": 12,
            "issue161_whole_clip_gain_matrices_not_used_as_active_rms": 12,
            "issue161_triplets_reused": 12, "issue161_selected_unique_midis": 36,
            "issue161_selected_wavs_verified_by_existing_receipt_audit": 4_644,
            "normalized_spectral_intermediates_available": False,
            "new_active_rms_spectral_matrix_pairs": 12 * 129 * 129,
            "locked_checkpoints_validated": checkpoint_reuse},
        "cohort": {"preset_states": 129, "preset_families": 129, "midi_performances": 50,
                   "wav_count": len(input_rows), "source_split": dataset_summary.get("source_split"),
                   "performance_source_split": dataset_summary.get("performance_source_split"),
                   "reserved_split_access": dataset_summary.get("reserved_split_access")},
        "source_audio_distribution": {"active_rms_dbfs": {"p05": float(20*np.log10(np.quantile(active_rms_values, .05))),
                "p50": float(20*np.log10(np.quantile(active_rms_values, .50))),
                "p95": float(20*np.log10(np.quantile(active_rms_values, .95)))},
            "sample_peak": {"p50": float(np.quantile(peak_values, .5)), "p95": float(np.quantile(peak_values, .95)),
                            "maximum": float(max(peak_values)), "above_unit_range_count": int(np.count_nonzero(np.asarray(peak_values) > 1.0))},
            "peak_to_active_rms": {"p50": float(np.quantile(crest_values, .5)),
                "p95": float(np.quantile(crest_values, .95)), "p99": float(np.quantile(crest_values, .99)),
                "maximum": float(max_crest),
                "maximum_db": float(20 * np.log10(max_crest))},
            "duration_seconds": {"minimum": float(min(durations)), "median": float(np.median(durations)),
                                 "maximum": float(max(durations))}},
        "normalization": {"version": NORMALIZATION_VERSION, "window_seconds": WINDOW_SECONDS,
            "window_size_samples_at_44100_hz": 882, "window_hop_samples": 1,
            "zero_extension": "windows at every integer start overlapping first through last nonzero sample",
            "active_gate_db_below_peak": ACTIVE_FLOOR_DB, "hard_dbfs_gate": None,
            "target_active_rms": float(target), "target_active_rms_dbfs": float(20 * np.log10(target)),
            "target_selection": "0.95 divided by maximum cohort peak/active-RMS ratio, minus a 1e-6 relative margin",
            "normalized_peak_maximum": float(normalized_peaks.max()), "normalized_peak_ceiling": MAX_NORMALIZED_PEAK,
            "out_of_range_source_samples": "reported; common target guarantees no normalized peak exceeds 0.95",
            "clipping_or_limiter": False, "near_zero_floor": MIN_ACTIVE_RMS},
        "normalizer_benchmark": benchmark,
        "artifacts": {"normalized_cache_root": str(NORMALIZED_ROOT), "wav_identity_manifest": str(WAV_AUDIT_FILE),
                      "source_receipt_manifest_sha256": dataset_summary.get("manifest_sha256")}}
    _write_json(TARGET_FILE, summary)
    progress(f"Preflight passed. Pinned target active RMS {target:.8f} ({20*np.log10(target):.2f} dBFS); "
             f"max normalized peak {normalized_peaks.max():.6f}.")
    return summary


def run_normalized_evaluation(dataset: Path = DATASET, *, feature_root: Path = NORMALIZED_ROOT) -> dict:
    result = evaluate_dataset(dataset, feature_root=feature_root, result_root=feature_root,
                              hard_negative_source=dataset)
    print(f"Evaluated {len(result['available_arms'])} normalized representation arms across "
          f"{result['common_episode_count_across_available_arms']:,} common episodes.", flush=True)
    return result


def _paired_results(original_rows: list[dict], normalized_rows: list[dict], metrics: tuple[str, ...]) -> dict:
    original_by_id = {row["episode_id"]: row for row in original_rows}
    normalized_by_id = {row["episode_id"]: row for row in normalized_rows}
    if original_by_id.keys() != normalized_by_id.keys() or not original_by_id:
        raise ValueError("paired original and normalized episode rows do not align")
    return {metric: paired_two_way_cluster_bootstrap(normalized_rows, original_rows,
        metric=metric, seed=164, replicates=2_000) for metric in metrics}


def _retrieval_comparison(original_evaluation: dict, normalized_evaluation: dict,
                          original_rows: list[dict], normalized_rows: list[dict]) -> dict:
    arms = sorted(original_evaluation["available_arms"])
    if set(arms) != set(normalized_evaluation["available_arms"]):
        raise ValueError("original and normalized retrieval results do not have the same eleven arms")
    results = {}
    for arm in arms:
        results[arm] = {}
        for condition in ("cross_midi_common_gallery", "cross_midi_mixed_gallery"):
            old = [row for row in original_rows if row.get("arm") == arm and row.get("condition") == condition]
            new = [row for row in normalized_rows if row.get("arm") == arm and row.get("condition") == condition]
            if len(old) != len(new):
                raise ValueError(f"original and normalized episode counts differ for {arm}/{condition}")
            results[arm][condition] = {"original": original_evaluation["model_results"][arm][condition],
                "normalized": normalized_evaluation["model_results"][arm][condition],
                "paired_normalized_minus_original": _paired_results(old, new, ("recall_at_1", "recall_at_5", "mrr"))}
            for key in ("same_midi_negative_win", "acoustic_hard_negative_win"):
                left = [row for row in old if row.get(key) is not None]
                right = [row for row in new if row.get(key) is not None]
                if len(left) != len(right):
                    raise ValueError(f"paired {key} diagnostic differs for {arm}/{condition}")
                if left:
                    results[arm][condition][f"paired_{key}_normalized_minus_original"] = _paired_results(left, right, (key,))[key]
    return results


def _metric_summary(values: np.ndarray, family_draws: np.ndarray, triplet_draws: np.ndarray) -> dict:
    estimate = float(values.mean())
    draws = values[family_draws[:, :, None], triplet_draws[:, None, :]].mean(axis=(1, 2))
    return {"mean": estimate, "clustered_95_ci": [float(value) for value in np.quantile(draws, [0.025, 0.975])],
            "family_clusters": int(values.shape[0]), "triplet_clusters": int(values.shape[1])}


def _substitution_2x2(original_choices: dict, normalized_choices: dict, raw_matrices: dict[int, np.ndarray],
                       active_matrices: dict[int, np.ndarray], triplets: dict, preset_ids: list[str],
                       family_by_state: dict, *, replicates: int = 2_000) -> dict:
    if len(preset_ids) != 129 or len(set(preset_ids)) != 129 or not set(preset_ids).issubset(family_by_state):
        raise ValueError("the three-MIDI comparison requires the exact 129-state retrieval cohort")
    family_by_state = {state_id: family_by_state[state_id] for state_id in preset_ids}
    family_ids = sorted(set(family_by_state.values()))
    if len(family_ids) != 129:
        raise ValueError("the three-MIDI bootstrap must use the 129 retrieval-family clusters only")
    family_index = {value: index for index, value in enumerate(family_ids)}
    state_index = {value: index for index, value in enumerate(preset_ids)}
    triplet_count = len(triplets["triplets"])
    seed = np.random.default_rng(ISSUE161_SEED)
    family_draws = seed.integers(len(family_ids), size=(replicates, len(family_ids)), dtype=np.int16)
    triplet_draws = seed.integers(triplet_count, size=(replicates, triplet_count), dtype=np.int8)
    random_reference = challenger_reference_choices(preset_ids, family_by_state, raw_matrices,
        list(range(triplet_count)), seed=ISSUE161_SEED)
    matrix_groups = {"original_level": raw_matrices, "active_rms_matched": active_matrices}
    results = {"scoring_conditions": {}, "paired_normalized_retrieval_minus_original": {}}
    choice_arrays, rank_arrays = {}, {}
    for score_name, matrices in matrix_groups.items():
        oracle = np.zeros((len(family_ids), triplet_count), dtype=np.float64)
        random = np.zeros_like(oracle)
        for triplet_index, matrix in matrices.items():
            for target_id in preset_ids:
                family_id = family_by_state[target_id]
                fi, ti = family_index[family_id], triplet_index
                eligible = [state for state in preset_ids if family_by_state[state] != family_id]
                best = min(eligible, key=lambda candidate: (float(matrix[state_index[target_id], state_index[candidate]]), candidate))
                oracle[fi, ti] = float(matrix[state_index[target_id], state_index[best]])
                random_id = random_reference[(triplet_index, target_id)]["random_state_id"]
                random[fi, ti] = float(matrix[state_index[target_id], state_index[random_id]])
        group = {"oracle": _metric_summary(oracle, family_draws, triplet_draws),
                 "fixed_random_reference": _metric_summary(random, family_draws, triplet_draws),
                 "retrieval_conditions": {}}
        for condition_name, choices in (("original_input", original_choices), ("normalized_input", normalized_choices)):
            per_arm = {}
            for arm, rows in choices.items():
                selected = np.zeros((len(family_ids), triplet_count), dtype=np.float64)
                ranks = np.zeros_like(selected)
                for row in rows:
                    ti, target, candidate = row["triplet_index"], row["target_state_id"], row["candidate_state_id"]
                    fi = family_index[row["target_family_id"]]
                    matrix = matrices[ti]
                    errors = np.asarray([matrix[state_index[target], state_index[state]] for state in preset_ids
                                         if family_by_state[state] != row["target_family_id"]], dtype=np.float64)
                    value = float(matrix[state_index[target], state_index[candidate]])
                    ranks[fi, ti] = int(np.sum(errors < value - 1e-8)) + (int(np.sum(np.abs(errors - value) <= 1e-8)) + 1) / 2
                    selected[fi, ti] = value
                per_arm[arm] = {"selected_substitute_error": _metric_summary(selected, family_draws, triplet_draws),
                    "excess_over_metric_specific_oracle": _metric_summary(selected - oracle, family_draws, triplet_draws),
                    "rank_among_128_alternatives": _metric_summary(ranks, family_draws, triplet_draws)}
                choice_arrays[(score_name, condition_name, arm)] = selected
                rank_arrays[(score_name, condition_name, arm)] = ranks
            group["retrieval_conditions"][condition_name] = per_arm
        for condition_name in ("original_input", "normalized_input"):
            selected_mean = np.stack([choice_arrays[(score_name, condition_name, arm)] for arm in sorted(original_choices)]).mean(axis=0)
            rank_mean = np.stack([rank_arrays[(score_name, condition_name, arm)] for arm in sorted(original_choices)]).mean(axis=0)
            group["retrieval_conditions"][condition_name]["macro_average_across_11_arms"] = {
                "selected_substitute_error": _metric_summary(selected_mean, family_draws, triplet_draws),
                "excess_over_metric_specific_oracle": _metric_summary(selected_mean - oracle, family_draws, triplet_draws),
                "rank_among_128_alternatives": _metric_summary(rank_mean, family_draws, triplet_draws),
                "arms_averaged_equally": len(original_choices)}
        results["scoring_conditions"][score_name] = group
    for score_name in matrix_groups:
        for arm in sorted(original_choices):
            delta = choice_arrays[(score_name, "normalized_input", arm)] - choice_arrays[(score_name, "original_input", arm)]
            results["paired_normalized_retrieval_minus_original"][score_name + ":" + arm] = _metric_summary(
                delta, family_draws, triplet_draws)
    results["design"] = {"input_conditions": ["original waveform before encoder preprocessing", "active-RMS normalized waveform before encoder preprocessing"],
        "acoustic_conditions": ["stored original-level MR-STFT matrix reused", "new active-RMS-matched MR-STFT matrix"],
        "triplets": triplet_count, "target_families": len(family_ids), "choices_per_arm_per_condition": 1_548,
        "alternatives_per_choice": 128, "bootstrap": "paired crossed resamples of 129 preset families and 12 triplets",
        "oracle": "recomputed separately under each acoustic score matrix",
        "random_alternative": "same deterministic seed-161 alternative IDs under both score matrices"}
    return results


def _read_matrix_archive(path: Path) -> dict[int, np.ndarray]:
    with np.load(path, allow_pickle=False) as arrays:
        result = {index: arrays[f"raw_triplet_{index:02d}"].astype(np.float32, copy=False) for index in range(12)}
    if any(matrix.shape != (129, 129) or not np.isfinite(matrix).all() for matrix in result.values()):
        raise ValueError("stored #161 original-level matrices are incomplete or invalid")
    return result


def _build_active_matrices(dataset: Path, audit: dict, triplets: dict, *, device: str,
                           output_path: Path = ACTIVE_MATRIX_FILE) -> tuple[dict[int, np.ndarray], dict]:
    expected_identity = {"cohort_identity_sha256": _read_json(TARGET_FILE)["cohort_identity_sha256"],
        "triplets_sha256": digest(triplets), "triplets": 12, "preset_families": 129,
        "scoring_wavs_loaded": 12 * 129,
        "scoring_unique_midis": len({row["roles"][2]["midi_id"] for row in triplets["triplets"]}),
        "active_rms_contract_shared_with_input_normalizer": True}
    if output_path.is_file():
        with np.load(output_path, allow_pickle=False) as arrays:
            expected_keys = {f"active_triplet_{index:02d}" for index in range(12)}
            if set(arrays.files) == expected_keys:
                matrices = {index: arrays[f"active_triplet_{index:02d}"].astype(np.float32, copy=False)
                            for index in range(12)}
                if all(matrix.shape == (129, 129) and np.isfinite(matrix).all() for matrix in matrices.values()):
                    contract = _read_json(ACTIVE_MATRIX_CONTRACT)
                    if all(contract.get(key) == value for key, value in expected_identity.items()):
                        return matrices, contract
    plans = _jsonl(dataset / "render_plan.private.jsonl")
    audio_rows, coverage = verify_triplet_wavs(dataset, audit, plans, triplets)
    matrices = {}
    contracts = []
    from frozen_retrieval.normalization import measure_active_rms

    for triplet_index, triplet in enumerate(triplets["triplets"]):
        midi3_id = triplet["roles"][2]["midi_id"]
        samples = load_scoring_midi(audio_rows, audit["preset_ids"], midi3_id)
        active = [measure_active_rms(audio, 44_100)["active_rms"] for audio in samples]
        matrix, contract = active_rms_matched_mrstft_matrix(samples, active,
            output_dir=PRIVATE / "active_stft_work", batch_size=8, pair_batch_size=64,
            device_name=device, progress=lambda message: print(f"Triplet {triplet_index:02d}: {message}", flush=True))
        matrices[triplet_index] = matrix
        contracts.append(contract)
        print(f"Scored active-RMS triplet {triplet_index + 1}/12 ({coverage['selected_unique_midis']} MIDI IDs total).", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **{f"active_triplet_{index:02d}": value for index, value in matrices.items()})
    contract = {"acoustic_contract": contracts[0], **expected_identity,
                "selected_unique_midis": coverage["selected_unique_midis"],
                "selected_matrix_wavs_verified": coverage["selected_matrix_wavs_verified"],
                "source_manifest_sha256": _read_json(TARGET_FILE)["source_manifest_sha256"]}
    _write_json(ACTIVE_MATRIX_CONTRACT, contract)
    return matrices, contract


def _issue155_comparison(feature_root: Path, output_dir: Path) -> dict:
    import run_issue155_validation as issue155

    cached_scores = output_dir / "gallery_assignment_episode_metrics.private.jsonl"
    reuse_cached_only = cached_scores.is_file()
    context = issue155._load_context(DATASET, feature_root=feature_root, load_features=not reuse_cached_only)
    normalized, data = issue155._score_assignments(context, (1, 7, 19), output_dir,
        reuse_cached_only=reuse_cached_only)
    baseline_rows = _jsonl(ISSUE155_DIR / "gallery_assignment_episode_metrics.private.jsonl")
    paired = {}
    for arm in sorted(context["arms"]):
        paired[arm] = {"per_assignment": {}}
        for offset in (1, 7, 19):
            old = [row for row in baseline_rows if row.get("arm") == arm and row.get("gallery_assignment_offset") == offset]
            new = [row for row in data["rows_by_arm"][arm] if row.get("gallery_assignment_offset") == offset]
            paired[arm]["per_assignment"][str(offset)] = issue155._paired_result(new, old,
                metrics=("recall_at_1", "recall_at_5", "mrr"))
        old_aggregate = issue155._collapse_assignments([row for row in baseline_rows if row.get("arm") == arm])
        paired[arm]["assignment_average"] = issue155._paired_result(data["aggregate_rows"][arm], old_aggregate,
            metrics=("recall_at_1", "recall_at_5", "mrr"))
    def macro_rows(arms: dict[str, list[dict]]) -> list[dict]:
        by_arm = {arm: {(row["target_state_id"], row["query_midi_id"]): row for row in rows}
                  for arm, rows in arms.items()}
        keys = set.intersection(*(set(rows) for rows in by_arm.values()))
        output = []
        for state_id, midi_id in sorted(keys):
            exemplar = by_arm[next(iter(by_arm))][(state_id, midi_id)]
            output.append({"episode_id": f"{state_id}|{midi_id}", "target_state_id": state_id,
                "target_family_id": exemplar["target_family_id"], "query_midi_id": midi_id,
                **{metric: float(np.mean([rows[(state_id, midi_id)][metric] for rows in by_arm.values()]))
                   for metric in ("recall_at_1", "recall_at_5", "mrr")}})
        return output
    original_aggregate = {arm: issue155._collapse_assignments(
        [row for row in baseline_rows if row.get("arm") == arm]) for arm in context["arms"]}
    macro_original, macro_normalized = macro_rows(original_aggregate), macro_rows(data["aggregate_rows"])
    macro_paired = issue155._paired_result(macro_normalized, macro_original,
        metrics=("recall_at_1", "recall_at_5", "mrr"))
    macro = {"original": {metric: float(np.mean([row[metric] for row in macro_original]))
                          for metric in ("recall_at_1", "recall_at_5", "mrr")},
        "normalized": {metric: float(np.mean([row[metric] for row in macro_normalized]))
                       for metric in ("recall_at_1", "recall_at_5", "mrr")},
        "paired_normalized_minus_original": macro_paired,
        "arms_averaged_equally": len(context["arms"])}
    return {"original_issue155_result_reused": True, "normalized_assignments": normalized,
            "paired_normalized_minus_original": paired, "macro_average_across_11_arms": macro}


def _issue157_comparison(dataset: Path, target: float, *, cache_audit: dict, device: str,
                         bootstrap_replicates: int, bootstrap_combinations: int) -> tuple[dict, dict]:
    original = _read_json(ROOT / "reports" / "issue157_sensitivity_results.json")
    normalized_path = NORMALIZED_ROOT / "issue157_sensitivity_results.private.json"
    expected_normalization = _read_json(TARGET_FILE)["normalization"]
    normalized = None
    if normalized_path.is_file():
        try:
            cached = _read_json(normalized_path)
            if (cached.get("analysis") == "issue157_preset_midi_sensitivity_v2" and
                    set(cached.get("arms", {})) == set(cache_audit["expected_arms"]) and
                    cached.get("input_normalization", {}).get("version") == NORMALIZATION_VERSION and
                    cached.get("input_normalization", {}).get("target_active_rms") == target):
                normalized = cached
                print("Reusing the verified normalized issue #157 sensitivity result.", flush=True)
        except (OSError, ValueError, TypeError):
            normalized = None
    if normalized is None:
        normalized = analyze_cache(cache_audit, device_name=device, bootstrap_replicates=bootstrap_replicates,
            bootstrap_combinations=bootstrap_combinations, progress=lambda message: print(message, flush=True))
        normalized["input_normalization"] = expected_normalization
        _write_json(normalized_path, normalized)
    paired_points = {arm: {metric: {"normalized": normalized["arms"][arm][metric],
        "original": original["arms"][arm][metric],
        "difference_normalized_minus_original": normalized["arms"][arm][metric] - original["arms"][arm][metric]}
        for metric in ("mean_normalized_margin", "preset_dominant_fraction", "mean_midi_change_distance",
                       "mean_preset_change_distance")}
        for arm in normalized["arms"]}
    return normalized, {"comparison_type": "paired full-cohort point estimates; intervals shown separately by condition",
                       "arms": paired_points}


def _load_normalized_choices(path: Path, cache_audit: dict, triplets: dict) -> dict | None:
    if not path.is_file():
        return None
    try:
        choices = _read_json(path)
        preset_ids = set(cache_audit["preset_ids"])
        family_by_state = cache_audit["family_by_state"]
        expected_keys = {(triplet["triplet_index"], state_id)
                         for triplet in triplets["triplets"] for state_id in preset_ids}
        if set(choices) != set(cache_audit["expected_arms"]):
            return None
        for arm, rows in choices.items():
            keys = {(row["triplet_index"], row["target_state_id"]) for row in rows}
            if len(rows) != len(expected_keys) or keys != expected_keys:
                return None
            for row in rows:
                target, candidate = row["target_state_id"], row["candidate_state_id"]
                if (candidate not in preset_ids or row.get("target_family_id") != family_by_state.get(target) or
                        family_by_state[candidate] == family_by_state[target] or row.get("eligible_candidates") != 128):
                    return None
        return choices
    except (OSError, ValueError, TypeError, KeyError):
        return None


def _gain_probe_result_row(case: dict, role: str, gain_db: float, **metrics) -> dict:
    episode_id = (f"gain164:{case['triplet_index']}:{case['target_state_id']}:"
                  f"{case['candidate_state_id']}:{role}:{int(gain_db)}")
    return {"episode_id": episode_id, "triplet_index": case["triplet_index"],
            "target_family_id": case["target_family_id"], "query_midi_id": case["query_midi_id"],
            "gain_db": gain_db, **metrics}


def _gain_sensitivity(dataset: Path, original_audit: dict, normalized_audit: dict,
                      original_choices: dict, normalized_choices: dict,
                      triplets: dict, target: float, *, checkpoint_root: Path, lock_path: Path,
                      models: tuple[str, ...] | None = None,
                      output: Path = PRIVATE / "gain_sensitivity.private.json") -> dict:
    selected_models = models or MODEL_ORDER
    expected_method = {"gain_db": [-6.0, 6.0], "roles": ["query", "selected_candidate"],
        "fixed_cases": 48, "families": 24, "triplets": 12, "selected_models": list(selected_models),
        "target_active_rms": target, "original_embedding_cache_reused": True,
        "normalized_embedding_cache_reused": True, "normalization_applied_after_gain_change": True}
    output_data = {"method": {**expected_method,
        "cohort_families": 129, "selection": "two seed-161 triplets each for 24 evenly spaced preset families; query and retrieval candidates from unchanged issue161 choices",
        "retrieval_pool": "same 128 different-family MIDI-2 alternatives as issue161", "completed_models": []}, "arms": {}}
    inference_calls = Counter()
    completed_models = set()
    if output.is_file():
        try:
            cached = _read_json(output)
            method = cached.get("method", {})
            if all(method.get(key) == value for key, value in expected_method.items()):
                output_data = cached
                inference_calls.update(method.get("new_embedding_calls_by_family", {}))
                completed_models = set(method.get("completed_models", []))
                print(f"Resuming gain sensitivity; reusing completed model families {sorted(completed_models)}.", flush=True)
        except (OSError, ValueError, TypeError):
            pass
    preset_ids = original_audit["preset_ids"]
    family_by_state = original_audit["family_by_state"]
    state_index = {state: index for index, state in enumerate(preset_ids)}
    triplets_by_index = {row["triplet_index"]: row for row in triplets["triplets"]}
    choice_lookup = {condition: {arm: {(row["triplet_index"], row["target_state_id"]): row["candidate_state_id"]
        for row in rows} for arm, rows in choices.items()}
        for condition, choices in (("original_input", original_choices), ("normalized_input", normalized_choices))}
    sample_cases = []
    ordered_families = sorted(original_audit["family_ids"])
    selected_family_numbers = {round(index * (len(ordered_families) - 1) / 23) for index in range(24)}
    for family_number, family_id in enumerate(ordered_families):
        if family_number not in selected_family_numbers:
            continue
        target_state = min(state for state in preset_ids if family_by_state[state] == family_id)
        for triplet_index in (family_number % 12, (family_number + 6) % 12):
            sample_cases.append({"triplet_index": triplet_index, "target_state_id": target_state,
                                 "target_family_id": family_id})
    if len(sample_cases) != 48 or len({row["triplet_index"] for row in sample_cases}) != 12:
        raise ValueError("gain probe must contain two triplets for each of 24 evenly spaced preset families")
    paths = _accepted_wav_paths(dataset, original_audit)
    lock = load_checkpoint_lock(lock_path)
    output_data["method"]["fixed_cases"] = len(sample_cases)
    output_data["method"]["completed_models"] = sorted(completed_models)
    expected_arms_by_model = {model: {f"{model}:{variant}" for variant in original_audit["model_caches"][model]["dimensions_by_variant"]}
                              for model in selected_models}
    completed_models = {model for model in completed_models
                        if expected_arms_by_model.get(model, set()).issubset(output_data["arms"])}
    for model_name in selected_models:
        if model_name in completed_models:
            print(f"Reusing completed gain sensitivity probes for {model_name}.", flush=True)
            continue
        checkpoint, entry, checkpoint_digest = resolve_checkpoint(model_name, checkpoint_root, lock)
        source_root = resolve_source_checkout(checkpoint_root, entry)
        adapter = load_adapter(model_name, checkpoint, entry, checkpoint_digest,
            checkpoint_root=checkpoint_root, device="xpu", source_root=source_root)
        for arm, variant in ((f"{model_name}:{variant}", variant)
                             for variant in original_audit["model_caches"][model_name]["dimensions_by_variant"]):
            conditions = {}
            for condition_name, active_target in (("original_input", None), ("normalized_input", target)):
                source_audit = original_audit if condition_name == "original_input" else normalized_audit
                vector_rows, decision_rows = {"query": [], "selected_candidate": []}, {"query": [], "selected_candidate": []}
                cases = []
                unique_sources = {"query": set(), "selected_candidate": set()}
                for case in sample_cases:
                    t = triplets_by_index[case["triplet_index"]]
                    midi1, midi2 = t["roles"][:2]
                    target_state, family_id = case["target_state_id"], case["target_family_id"]
                    candidate_state = choice_lookup[condition_name][arm][(case["triplet_index"], target_state)]
                    query_key, candidate_key = (target_state, midi1["midi_id"]), (candidate_state, midi2["midi_id"])
                    if query_key not in paths or candidate_key not in paths:
                        raise ValueError("gain probe selected a WAV outside the existing 6,450-cell matrix")
                    unique_sources["query"].add(query_key)
                    unique_sources["selected_candidate"].add(candidate_key)
                    cases.append({**case, "candidate_state_id": candidate_state,
                                  "query_key": query_key, "candidate_key": candidate_key,
                                  "midi2_id": midi2["midi_id"], "query_midi_id": midi1["midi_id"]})
                baseline_grids = {}
                for case in cases:
                    triplet_index, target_state = case["triplet_index"], case["target_state_id"]
                    if triplet_index in baseline_grids:
                        continue
                    t = triplets_by_index[triplet_index]
                    midi2 = t["roles"][1]["midi_id"]
                    vectors = {}
                    cache = source_audit["model_caches"][model_name]
                    for candidate in preset_ids:
                        request_id = source_audit["request_by_pair"][(candidate, midi2)]
                        row = cache["rows_by_request"][request_id]
                        with np.load(cache["directory"] / row["cache_path"], allow_pickle=False) as arrays:
                            vectors[candidate] = arrays[variant].astype(np.float32, copy=False)
                    baseline_grids[triplet_index] = vectors
                changed_by_source = {role: {} for role in unique_sources}
                for role in ("query", "selected_candidate"):
                    for source_key in sorted(unique_sources[role]):
                        source_path = paths[source_key]
                        request_id = original_audit["request_by_pair"][source_key]
                        original_audio_sha = original_audit["receipts"][request_id]["audio_sha256"]
                        if file_sha256(source_path) != original_audio_sha:
                            raise ValueError("gain-probe source WAV changed after the preflight receipt audit")
                        variants = {}
                        for gain_db in (-6.0, 6.0):
                            variants[gain_db] = adapter.extract_gain_variant(source_path, gain_db,
                                active_rms_target=active_target)
                            inference_calls[model_name] += 1
                        changed_by_source[role][source_key] = variants
                for case in cases:
                    case_key = (case["triplet_index"], case["target_state_id"])
                    gallery = baseline_grids[case["triplet_index"]]
                    vectors = {state: vector for state, vector in gallery.items()
                               if family_by_state[state] != case["target_family_id"]}
                    query_request = source_audit["request_by_pair"][case["query_key"]]
                    candidate_request = source_audit["request_by_pair"][case["candidate_key"]]
                    query_baseline = vectors.get(case["target_state_id"])
                    if query_baseline is None:
                        row = source_audit["model_caches"][model_name]["rows_by_request"][query_request]
                        with np.load(source_audit["model_caches"][model_name]["directory"] / row["cache_path"], allow_pickle=False) as arrays:
                            query_baseline = arrays[variant].astype(np.float32, copy=False)
                    candidate_baseline = vectors[case["candidate_state_id"]]
                    # Cosine helpers operate on the same common 128 alternatives used by issue #161.
                    def rank(query_vector, candidate_vectors):
                        q = np.asarray(query_vector, dtype=np.float64)
                        q /= max(float(np.linalg.norm(q)), 1e-12)
                        scores = {}
                        for candidate, vector in candidate_vectors.items():
                            v = np.asarray(vector, dtype=np.float64)
                            v /= max(float(np.linalg.norm(v)), 1e-12)
                            scores[candidate] = float(q @ v)
                        return min(scores, key=lambda state: (-scores[state], state))
                    original_winner = rank(query_baseline, vectors)
                    if original_winner != choice_lookup[condition_name][arm][case_key]:
                        raise ValueError(f"baseline candidate rank differs from saved choices for {arm}/{condition_name}")
                    for role, request_id, baseline_vector in (("query", query_request, query_baseline),
                                                               ("selected_candidate", candidate_request, candidate_baseline)):
                        for gain_db in (-6.0, 6.0):
                            changed = changed_by_source[role][case["query_key"] if role == "query" else case["candidate_key"]][gain_db][variant]
                            left, right = np.asarray(baseline_vector, dtype=np.float64), np.asarray(changed, dtype=np.float64)
                            distance = 1.0 - float(left @ right / (np.linalg.norm(left) * np.linalg.norm(right)))
                            current = dict(vectors)
                            if role == "query":
                                winner = rank(changed, current)
                            else:
                                current[case["candidate_state_id"]] = changed
                                winner = rank(query_baseline, current)
                            vector_rows[role].append(_gain_probe_result_row(case, role, gain_db,
                                embedding_distance=max(0.0, distance)))
                            decision_rows[role].append(_gain_probe_result_row(case, role, gain_db,
                                decision_changed=float(winner != original_winner)))
                role_results = {}
                for role in ("query", "selected_candidate"):
                    perturbed = vector_rows[role][-len(cases) * 2:]
                    changes = decision_rows[role]
                    role_results[role] = {"embedding_displacement_by_gain_db": {},
                        "retrieval_decision_change_by_gain_db": {}}
                    for gain_db in (-6.0, 6.0):
                        selected_vectors = [row for row in perturbed if row["gain_db"] == gain_db]
                        selected_changes = [row for row in changes if row["gain_db"] == gain_db]
                        role_results[role]["embedding_displacement_by_gain_db"][str(int(gain_db))] = {
                            "mean_cosine_distance": float(np.mean([row["embedding_distance"] for row in selected_vectors])),
                            "clustered_95_ci": evaluation_two_way_ci(selected_vectors, "embedding_distance")}
                        role_results[role]["retrieval_decision_change_by_gain_db"][str(int(gain_db))] = {
                            "rate": float(np.mean([row["decision_changed"] for row in selected_changes])),
                            "clustered_95_ci": evaluation_two_way_ci(selected_changes, "decision_changed")}
                conditions[condition_name] = role_results
            output_data["arms"][arm] = conditions
        print(f"Completed gain sensitivity probes for {model_name} ({inference_calls[model_name]:,} new embeddings).", flush=True)
        completed_models.add(model_name)
        output_data["method"]["completed_models"] = sorted(completed_models)
        output_data["method"]["new_embedding_calls_by_family"] = dict(inference_calls)
        output_data["method"]["new_embedding_calls"] = sum(inference_calls.values())
        output_data["method"]["new_inference_fraction_of_full_normalized_matrix"] = sum(inference_calls.values()) / (6_450 * 7)
        _write_json_atomic(output, output_data)
        print(f"Checkpointed gain sensitivity through {model_name}.", flush=True)
    output_data["method"]["new_embedding_calls_by_family"] = dict(inference_calls)
    output_data["method"]["new_embedding_calls"] = sum(inference_calls.values())
    output_data["method"]["new_inference_fraction_of_full_normalized_matrix"] = sum(inference_calls.values()) / (6_450 * 7)
    output_data["method"]["completed_models"] = sorted(completed_models)
    _write_json_atomic(output, output_data)
    return output_data


def _gain_probe(model: str, dataset: Path, checkpoint_root: Path, *, device: str) -> dict:
    target = _target_from_preflight()
    model_output = PRIVATE / f"gain_sensitivity_{model}.private.json"
    if model_output.is_file():
        return _read_json(model_output)
    original_audit = verify_embedding_cache(dataset, checkpoint_root, model_names=(model,),
        progress=lambda message: print(f"Original-cache reuse check: {message}", flush=True))
    normalized_audit = verify_embedding_cache(dataset, checkpoint_root, feature_root=NORMALIZED_ROOT,
        results_root=NORMALIZED_ROOT, normalization_target=target, model_names=(model,),
        progress=lambda message: print(f"Normalized-cache reuse check: {message}", flush=True))
    triplets = _read_json(TRIPLET_FILE)
    original_choices = _read_json(CHOICES_FILE)
    normalized_choices = _read_json(PRIVATE / "normalized_retrieval_choices.private.json")
    return _gain_sensitivity(dataset, original_audit, normalized_audit, original_choices,
        normalized_choices, triplets, target, checkpoint_root=checkpoint_root,
        lock_path=checkpoint_root / "checkpoint-lock.json", models=(model,), output=model_output)


def evaluation_two_way_ci(rows: list[dict], metric: str) -> dict:
    from frozen_retrieval.core import two_way_cluster_bootstrap

    return two_way_cluster_bootstrap(rows, metric=metric, seed=164, replicates=2_000)


def analyze(dataset: Path = DATASET, checkpoint_root: Path = CHECKPOINT_ROOT, *,
            device: str = "xpu", bootstrap_replicates: int = 1_000,
            bootstrap_combinations: int = 32_768) -> dict:
    target = _target_from_preflight()
    original_eval = _read_json(dataset / "evaluation.private.json")
    normalized_eval = _read_json(NORMALIZED_ROOT / "evaluation.private.json")
    original_rows = _jsonl(dataset / "episode_metrics.private.jsonl")
    normalized_rows = _jsonl(NORMALIZED_ROOT / "episode_metrics.private.jsonl")
    retrieval = _retrieval_comparison(original_eval, normalized_eval, original_rows, normalized_rows)
    original_audit = verify_embedding_cache(dataset, checkpoint_root,
        progress=lambda message: print(f"Original-cache reuse check: {message}", flush=True))
    norm_audit = verify_embedding_cache(dataset, checkpoint_root, feature_root=NORMALIZED_ROOT,
        results_root=NORMALIZED_ROOT, normalization_target=target,
        progress=lambda message: print(message, flush=True))
    sensitivity, sensitivity_compare = _issue157_comparison(dataset, target, cache_audit=norm_audit,
        device=device, bootstrap_replicates=bootstrap_replicates, bootstrap_combinations=bootstrap_combinations)
    _write_json(NORMALIZED_ROOT / "issue157_sensitivity_results.private.json", sensitivity)
    (NORMALIZED_ROOT / "issue157_sensitivity_results.private.json").write_text(
        json.dumps(sensitivity, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    triplets = _read_json(ISSUE161_DIR / "triplets.private.json")
    original_choices = _read_json(CHOICES_FILE)
    if set(original_choices) != set(norm_audit["expected_arms"]):
        raise ValueError("stored original issue #161 choices do not cover the normalized eleven-arm cache")
    normalized_choice_path = PRIVATE / "normalized_retrieval_choices.private.json"
    normalized_choices = _load_normalized_choices(normalized_choice_path, norm_audit, triplets)
    if normalized_choices is None:
        normalized_choices = retrieve_alternatives(norm_audit, triplets)
        _write_json(normalized_choice_path, normalized_choices)
    else:
        print("Reusing verified normalized issue #161 retrieval choices.", flush=True)
    active_matrices, active_contract = _build_active_matrices(dataset, norm_audit, triplets, device=device)
    raw_matrices = _read_matrix_archive(ISSUE161_DIR / "acoustic_matrices.private.npz")
    substitution = _substitution_2x2(original_choices, normalized_choices, raw_matrices, active_matrices,
        triplets, norm_audit["preset_ids"], norm_audit["family_by_state"])
    issue155 = _issue155_comparison(NORMALIZED_ROOT, PRIVATE / "issue155_assignments")
    standard_gain_path = PRIVATE / "gain_sensitivity_six_families.private.json"
    standard_gain = _gain_sensitivity(dataset, original_audit, norm_audit, original_choices,
        normalized_choices, triplets, target, checkpoint_root=checkpoint_root,
        lock_path=checkpoint_root / "checkpoint-lock.json",
        models=tuple(model for model in MODEL_ORDER if model != "basic_pitch_trunk"), output=standard_gain_path)
    basic_pitch_gain_path = PRIVATE / "gain_sensitivity_basic_pitch_trunk.private.json"
    native_site = Path.home() / "miniforge3" / "envs" / "obruxo-data-generation" / "Lib" / "site-packages"
    if not native_site.is_dir():
        raise FileNotFoundError(f"Basic Pitch's locked native package directory is missing: {native_site}")
    if not basic_pitch_gain_path.is_file():
        env = os.environ.copy()
        current_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(native_site) + (";" + current_pythonpath if current_pythonpath else "")
        subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "--stage", "gain-probe",
            "--model", "basic_pitch_trunk", "--dataset", str(dataset),
            "--checkpoint-root", str(checkpoint_root), "--device", device],
            cwd=REPO, env=env, check=True)
    basic_pitch_gain = _read_json(basic_pitch_gain_path)
    gain = {"method": {**standard_gain["method"],
            "selected_models": list(MODEL_ORDER),
            "completed_models": sorted(set(standard_gain["method"]["completed_models"]) |
                                        set(basic_pitch_gain["method"]["completed_models"])),
            "new_embedding_calls_by_family": {**standard_gain["method"]["new_embedding_calls_by_family"],
                **basic_pitch_gain["method"]["new_embedding_calls_by_family"]},
            "new_embedding_calls": standard_gain["method"]["new_embedding_calls"] + basic_pitch_gain["method"]["new_embedding_calls"]},
            "arms": {**standard_gain["arms"], **basic_pitch_gain["arms"]}}
    gain["method"]["new_inference_fraction_of_full_normalized_matrix"] = gain["method"]["new_embedding_calls"] / (6_450 * 7)
    _write_json(PRIVATE / "gain_sensitivity.private.json", gain)
    issue161_original = _read_json(ISSUE161_DIR / "aggregate_results.private.json")
    original157 = _read_json(ROOT / "reports" / "issue157_sensitivity_results.json")
    result = {"schema": "issue164_active_rms_comparison_v1",
        "normalization": _read_json(TARGET_FILE)["normalization"],
        "cohort": {"preset_states": 129, "preset_family_clusters": 129, "midi_performances": 50,
            "cross_midi_queries": 6_450, "triplet_cases_per_arm": 1_548,
            "independent_triplet_performances": 36, "reserved_split_access": "none"},
        "reuse_and_execution": {"preflight": _read_json(TARGET_FILE)["reuse_audit"],
            "original_normalized_caches_separated": True,
            "normalized_cache_root": "outputs/issue152-validation-train-midi-v3-50/features_active_rms",
            "source_wavs_modified": False, "new_vital_renders": 0, "model_training": False,
            "original_issue161_result_reused": True,
            "active_rms_matrix_contract": active_contract},
        "cross_midi_retrieval": retrieval,
        "issue155_three_comparison_midi_assignments": issue155,
        "issue157_preset_midi_sensitivity": {"normalized": sensitivity, "comparison_to_stored_original": sensitivity_compare},
        "three_midi_substitution_2x2": substitution,
        "gain_sensitivity": {"method": gain["method"],
            "results_path": "outputs/issue164-active-rms/gain_sensitivity.private.json",
            "summary": {arm: value for arm, value in gain["arms"].items()}},
        "historical_original_results": {"issue161": {"analysis": issue161_original.get("schema"),
            "triplets": issue161_original.get("triplets"), "original_score_metric": "original-level MR-STFT"},
            "issue157": {"analysis": original157.get("analysis"), "retrieval_scores_reused": True},
            "issue155": {"warning_reduction_pilot_repeated": False}}}
    _write_json(PUBLIC_RESULT, result)
    _write_issue164_reporting(result)
    print(f"Saved issue #164 aggregates to {PUBLIC_RESULT}.", flush=True)
    return result


def _arm_label(arm: str) -> str:
    family, variant = arm.split(":", 1)
    names = {"cir_ast": "CIR", "beats_iter3_plus": "BEATs", "clap_music": "CLAP",
             "dac_44khz": "DAC", "encodec_24khz": "EnCodec", "muq_mulan": "MuQ-MuLan",
             "basic_pitch_trunk": "Basic Pitch"}
    variants = {"official_clip": "clip", "masked_mean": "mean", "temporal_pyramid_8": "temporal"}
    return f"{names.get(family, family)} — {variants.get(variant, variant)}"


def _interval_text(estimate: float, interval, *, multiplier: float = 1.0, digits: int = 3) -> str:
    lo, hi = interval
    return f"{estimate * multiplier:.{digits}f} [{lo * multiplier:.{digits}f}, {hi * multiplier:.{digits}f}]"


def _issue164_report_section(result: dict) -> str:
    retrieval = result["cross_midi_retrieval"]
    substitution = result["three_midi_substitution_2x2"]
    active = substitution["scoring_conditions"]["active_rms_matched"]
    raw = substitution["scoring_conditions"]["original_level"]
    active_pairs = substitution["paired_normalized_retrieval_minus_original"]
    arms = sorted(retrieval)
    r1_effects, active_error_effects = [], []
    main_rows = []
    for arm in arms:
        common = retrieval[arm]["cross_midi_common_gallery"]
        mixed = retrieval[arm]["cross_midi_mixed_gallery"]
        r1_change = common["paired_normalized_minus_original"]["recall_at_1"]
        mixed_change = mixed["paired_normalized_minus_original"]["recall_at_1"]
        active_old = active["retrieval_conditions"]["original_input"][arm]
        active_new = active["retrieval_conditions"]["normalized_input"][arm]
        active_change = active_pairs[f"active_rms_matched:{arm}"]
        r1_effects.append(float(r1_change["difference"]))
        active_error_effects.append(float(active_change["mean"]))
        main_rows.append(f"| {_arm_label(arm)} | {100*common['original']['recall_at_1']:.1f}% | "
            f"{100*common['normalized']['recall_at_1']:.1f}% | "
            f"{_interval_text(r1_change['difference'], [r1_change['lower_95'],r1_change['upper_95']],multiplier=100,digits=1)} pp | "
            f"{_interval_text(mixed_change['difference'],[mixed_change['lower_95'],mixed_change['upper_95']],multiplier=100,digits=1)} pp | "
            f"{active_old['selected_substitute_error']['mean']:.3f} → {active_new['selected_substitute_error']['mean']:.3f} | "
            f"{active_old['rank_among_128_alternatives']['mean']:.1f} → {active_new['rank_among_128_alternatives']['mean']:.1f} | "
            f"{_interval_text(active_change['mean'], active_change['clustered_95_ci'],digits=3)} |")

    score_rows = []
    for score_name, score_data in (("Original-level MR-STFT", raw), ("Active-RMS-matched MR-STFT", active)):
        oracle = score_data["oracle"]
        random = score_data["fixed_random_reference"]
        for input_name in ("original_input", "normalized_input"):
            macro = score_data["retrieval_conditions"][input_name]["macro_average_across_11_arms"]
            score_rows.append(f"| {score_name} | {input_name.replace('_', ' ')} | "
                f"{_interval_text(macro['selected_substitute_error']['mean'],macro['selected_substitute_error']['clustered_95_ci'])} | "
                f"{_interval_text(macro['excess_over_metric_specific_oracle']['mean'],macro['excess_over_metric_specific_oracle']['clustered_95_ci'])} | "
                f"{_interval_text(oracle['mean'],oracle['clustered_95_ci'])} | "
                f"{_interval_text(random['mean'],random['clustered_95_ci'])} | "
                f"{_interval_text(macro['rank_among_128_alternatives']['mean'],macro['rank_among_128_alternatives']['clustered_95_ci'],digits=1)} |")

    issue155 = result["issue155_three_comparison_midi_assignments"]
    issue155_macro = issue155["macro_average_across_11_arms"]

    sensitivity = result["issue157_preset_midi_sensitivity"]["comparison_to_stored_original"]["arms"]
    sensitivity_fraction_deltas = [float(value["preset_dominant_fraction"]["difference_normalized_minus_original"])
                                    for value in sensitivity.values()]
    sensitivity_margin_deltas = [float(value["mean_normalized_margin"]["difference_normalized_minus_original"])
                                 for value in sensitivity.values()]

    gain_data = result["gain_sensitivity"]
    gain_rows = []
    for input_name in ("original_input", "normalized_input"):
        for role, role_label in (("query", "Query"), ("selected_candidate", "Selected candidate")):
            scenarios = [arm_data[input_name][role] for arm_data in gain_data["summary"].values()]
            decision_rates = [scenario["retrieval_decision_change_by_gain_db"][str(gain_db)]["rate"]
                              for scenario in scenarios for gain_db in (-6, 6)]
            embedding_distances = [scenario["embedding_displacement_by_gain_db"][str(gain_db)]["mean_cosine_distance"]
                                   for scenario in scenarios for gain_db in (-6, 6)]
            gain_rows.append(f"| {input_name.replace('_', ' ')} | {role_label} | "
                f"{100*np.mean(decision_rates):.1f}% (range {100*min(decision_rates):.1f}–{100*max(decision_rates):.1f}%) | "
                f"{np.mean(embedding_distances):.4f} |")

    r1_median, r1_min, r1_max = (float(np.median(r1_effects)), float(min(r1_effects)), float(max(r1_effects)))
    active_median, active_min, active_max = (float(np.median(active_error_effects)),
        float(min(active_error_effects)), float(max(active_error_effects)))
    target = result["normalization"]["target_active_rms"]
    target_db = result["normalization"]["target_active_rms_dbfs"]
    benchmark = _read_json(TARGET_FILE)["normalizer_benchmark"]
    reuse = result["reuse_and_execution"]["preflight"]
    return "\n".join([
        "<!-- issue164-active-rms-results:start -->",
        "### Issue #164 — Padding-invariant active-RMS re-evaluation", "",
        f"The paired input-normalization effect on cross-MIDI Recall@1 was arm-specific: the median change across 11 variants was {100*r1_median:+.1f} pp (range {100*r1_min:+.1f} to {100*r1_max:+.1f}; see intervals below). Under active-RMS-matched acoustic scoring, the median per-arm change in substitute error was {active_median:+.3f} (range {active_min:+.3f} to {active_max:+.3f}); lower error is closer audio. These medians describe the tested variants, not a pooled independent sample. Together these comparisons estimate how input level and evaluation-time candidate gain matching influence this cohort, while MR-STFT remains an acoustic proxy rather than a listener or pure-timbre judgment.", "",
        f"The shared target was {target:.8f} ({target_db:.2f} dBFS). The cohort is 129 preset families × 50 distinct MIDI performances (6,450 queries); #161 contributes 1,548 selections per arm from 129 families × 12 triplets and 36 distinct MIDI performances. Retrieval intervals resample preset-family and MIDI-performance clusters; substitution intervals resample preset family and whole triplet.", "",
        "| Frozen representation | Common-gallery original R@1 | Common-gallery normalized R@1 | Paired common-gallery ΔR@1 (95% CI) | Paired mixed-gallery ΔR@1 (95% CI) | Active-RMS-matched error (original → normalized input) | Active-RMS rank / 128 (original → normalized input) | Paired active-RMS error Δ (95% CI) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|", *main_rows, "",
        "The per-arm acoustic columns use active-RMS-matched MR-STFT. The four-row table below separates the two embedding-input conditions from the two score conditions: original-level MR-STFT keeps recorded levels, while active-RMS scoring scales each candidate B3 by active_RMS(A3)/active_RMS(B3). It never modifies the input audio or embeddings. Each score matrix has its own oracle and fixed-random reference:", "",
        "| Acoustic score | Embedding input | 11-arm macro selected error (95% CI) | Excess over metric-specific oracle (95% CI) | Oracle (95% CI) | Fixed random (95% CI) | Mean rank among 128 (95% CI) |",
        "|---|---|---:|---:|---:|---:|---:|", *score_rows, "",
        "Macro rows give equal weight to the 11 representation arms. The oracle is an evaluation-only choice made after the third MIDI is known; it never affects retrieval. Same-MIDI and acoustic-hard-negative win rates are also paired and recalculated. Per-arm values, paired uncertainty, and complete 2×2 scores are in the [machine-readable result](issue164_active_rms_results.json).", "",
        "#### Preset/MIDI sensitivity and gallery-MIDI robustness", "",
        f"Issue #155 reuses the three existing gallery-MIDI offsets and 6,450 queries per offset. Averaged equally over the 11 arms, Recall@1 changed from {100*issue155_macro['original']['recall_at_1']:.1f}% to {100*issue155_macro['normalized']['recall_at_1']:.1f}% (paired Δ {_interval_text(issue155_macro['paired_normalized_minus_original']['recall_at_1']['difference'],[issue155_macro['paired_normalized_minus_original']['recall_at_1']['lower_95'],issue155_macro['paired_normalized_minus_original']['recall_at_1']['upper_95']],multiplier=100,digits=1)} pp). Each query's three assignments were collapsed before crossed resampling; per-arm and per-offset metrics are in the machine result.", "",
        f"Issue #157 was recomputed from normalized embeddings and its original full-cohort aggregates were reused. Across the 11 arms, the median point-estimate change was {100*np.median(sensitivity_fraction_deltas):+.1f} pp for preset-dominant fraction (range {100*min(sensitivity_fraction_deltas):+.1f} to {100*max(sensitivity_fraction_deltas):+.1f}) and {np.median(sensitivity_margin_deltas):+.3f} for normalized margin. The original report's aggregate intervals and new normalized-condition intervals are retained in machine output; paired intervals are unavailable because #157 did not persist original anchor-level values.", "",
        f"The controlled ±6 dB probe perturbed query and selected-candidate audio for {gain_data['method']['fixed_cases']} fixed cases across {gain_data['method']['families']} evenly spaced families and all 12 triplets. The normalized condition reapplies the shared RMS target after the gain change. This table averages the 11 arms and two gain directions descriptively; per-arm crossed intervals are in the machine result.", "",
        "| Input condition | Perturbed role | Retrieval decision change, mean (range across arm/gain) | Mean embedding cosine distance |", "|---|---|---:|---:|", *gain_rows, "",
        "#### Interpretation, limits, and next experiment", "",
        "A change after input normalization shows that a common level factor affected the encoder's representation or its ranking on these clips. Active-RMS matching at evaluation time separately removes one candidate-gain difference from the acoustic score; remaining MR-STFT error still reflects pitch, timing, envelope, and spectral differences, not timbre alone. The 12 triplets and 24-family ±6 dB probe are exploratory; they do not cover other synthesizers, broader MIDI/performance diversity, listener similarity, or downstream patch reconstruction. The next decision should use a larger independently selected MIDI set and an actual reconstruction/listening evaluation if the intended endpoint is playable sound recovery.", "",
        "#### Execution and reproducibility", "",
        f"The preflight rehashed and measured all {reuse['source_wavs']:,} accepted stereo WAVs, validated every original cache and locked checkpoint, and confirmed that the prior #152 evaluation, #155 assignments, #157 aggregate, #161 choices, and original-level matrices were reusable. It found no reusable per-WAV #161 spectra, so only the 1,548 MIDI-3 WAVs were transformed into new active-RMS matrices ({reuse['new_active_rms_spectral_matrix_pairs']:,} ordered matrix cells); no rendering, training, or original-cache regeneration occurred.",
        f"The Numba implementation is O(N), padding invariant, and was benchmarked after compilation warm-up at {benchmark['normalization_median_ms']:.1f} ms median per clip (range {benchmark['normalization_min_ms']:.1f}–{benchmark['normalization_max_ms']:.1f} ms for five 5–10 s stereo clips), about {100*benchmark['overhead_as_fraction_of_median_issue160_adapter_latency']:.1f}% of the stored median adapter latency. The normalized cache contains 6,450 WAVs per family under `outputs/issue152-validation-train-midi-v3-50/features_active_rms/`; provenance records the fixed target, gain, input receipt hash, checkpoint, and preprocessing contract. Source audio and original feature caches were unchanged.", "",
        "<!-- issue164-active-rms-results:end -->", ""])


def _write_issue164_reporting(result: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    section = _issue164_report_section(result)
    report_text = REPORT.read_text(encoding="utf-8")
    start, end = "<!-- issue164-active-rms-results:start -->", "<!-- issue164-active-rms-results:end -->"
    if start in report_text and end in report_text:
        before, remainder = report_text.split(start, 1)
        _, after = remainder.split(end, 1)
        report_text = before + section + after
    else:
        anchor = "## Results and recommendation\n\n"
        if anchor not in report_text:
            raise ValueError("technical report is missing its results insertion point")
        report_text = report_text.replace(anchor, anchor + section + "\n", 1)
    REPORT.write_text(report_text, encoding="utf-8", newline="\n")

    arms = sorted(result["cross_midi_retrieval"])
    active_pairs = result["three_midi_substitution_2x2"]["paired_normalized_retrieval_minus_original"]
    labels = [_arm_label(arm) for arm in arms]
    r1, r1_lo, r1_hi, audio, audio_lo, audio_hi = [], [], [], [], [], []
    for arm in arms:
        effect = result["cross_midi_retrieval"][arm]["cross_midi_common_gallery"]["paired_normalized_minus_original"]["recall_at_1"]
        r1.append(100*effect["difference"]); r1_lo.append(100*(effect["difference"]-effect["lower_95"])); r1_hi.append(100*(effect["upper_95"]-effect["difference"]))
        audio_effect = active_pairs[f"active_rms_matched:{arm}"]
        audio.append(audio_effect["mean"]); audio_lo.append(audio_effect["mean"]-audio_effect["clustered_95_ci"][0]); audio_hi.append(audio_effect["clustered_95_ci"][1]-audio_effect["mean"])
    y = np.arange(len(arms))
    with plt.rc_context({"font.size": 8, "axes.titlesize": 11, "axes.labelsize": 9}):
        figure, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
        axes[0].errorbar(r1, y, xerr=[r1_lo, r1_hi], fmt="o", color="#176b87", capsize=2)
        axes[0].axvline(0, color="#555555", linestyle="--", linewidth=1)
        axes[0].set_title("Cross-MIDI Recall@1")
        axes[0].set_xlabel("Normalized input − original input (percentage points)")
        axes[0].xaxis.set_major_locator(MaxNLocator(6))
        axes[1].errorbar(audio, y, xerr=[audio_lo, audio_hi], fmt="o", color="#ad4b18", capsize=2)
        axes[1].axvline(0, color="#555555", linestyle="--", linewidth=1)
        axes[1].set_title("Active-RMS-matched substitute error")
        axes[1].set_xlabel("Normalized-input error − original-input error (MR-STFT units)")
        for axis in axes:
            axis.grid(axis="x", alpha=.22)
        axes[0].set_yticks(y, labels)
        axes[0].invert_yaxis()
        figure.suptitle("Paired effect of padding-invariant active-RMS input normalization", y=.99)
        figure.text(.5, .015, "95% crossed intervals: 129 preset families × 50 MIDI performances for retrieval; 129 families × 12 whole triplets for substitute error.", ha="center", fontsize=8)
        figure.tight_layout(rect=(0, .04, 1, .95))
    figures_dir = ROOT / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(figures_dir / "issue164_active_rms_paired_effects.svg", format="svg", bbox_inches="tight", facecolor="white")
    plt.close(figure)

    visual_path = ROOT / "reports" / "FROZEN_ENCODER_VISUAL_SUMMARY.md"
    visual_text = visual_path.read_text(encoding="utf-8")
    visual_section = "\n".join(["<!-- issue164-active-rms-summary:start -->",
        "## Does input loudness normalization change the comparison?", "",
        "![Paired Recall@1 and active-RMS acoustic error effects by encoder](figures/issue164_active_rms_paired_effects.svg)", "",
        f"A single fixed active-RMS target of {result['normalization']['target_active_rms']:.5f} ({result['normalization']['target_active_rms_dbfs']:.2f} dBFS) changed Recall@1 by a median {100*np.median([result['cross_midi_retrieval'][arm]['cross_midi_common_gallery']['paired_normalized_minus_original']['recall_at_1']['difference'] for arm in arms]):+.1f} points across the 11 arms. In the separate acoustic test, candidate gain was matched only while scoring A3 against B3; input normalization was applied before encoding. Lower active-RMS MR-STFT error is closer audio, though it remains sensitive to pitch, timing, and envelope as well as timbre.", "",
        "The chart shows paired changes with 95% intervals. The left panel isolates the effect of normalizing each encoder input on retrieval. The right panel holds active-RMS candidate matching fixed and compares which alternatives original versus normalized embeddings selected. This scoring-time gain match is different from input normalization and does not alter cached embeddings.",
        "<!-- issue164-active-rms-summary:end -->", ""])
    vstart, vend = "<!-- issue164-active-rms-summary:start -->", "<!-- issue164-active-rms-summary:end -->"
    if vstart in visual_text and vend in visual_text:
        before, remainder = visual_text.split(vstart, 1)
        _, after = remainder.split(vend, 1)
        visual_text = before + visual_section + after
    else:
        visual_text = visual_text.replace("## 1. Is recognising a preset the same as finding a good substitute?", visual_section + "\n## 1. Is recognising a preset the same as finding a good substitute?", 1)
    visual_path.write_text(visual_text, encoding="utf-8", newline="\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Re-evaluate frozen encoder caches with issue #164 active-RMS normalization")
    parser.add_argument("--stage", choices=("preflight", "evaluate", "analyze", "gain-probe"), required=True)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--checkpoint-root", type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument("--device", choices=("cpu", "cuda", "xpu"), default="xpu")
    parser.add_argument("--bootstrap-replicates", type=int, default=1_000)
    parser.add_argument("--bootstrap-combinations", type=int, default=32_768)
    parser.add_argument("--model", choices=MODEL_ORDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.stage == "preflight":
        result = preflight(args.dataset, args.checkpoint_root)
    elif args.stage == "evaluate":
        result = run_normalized_evaluation(args.dataset)
    elif args.stage == "gain-probe":
        if not args.model:
            raise ValueError("gain-probe stage requires --model")
        result = _gain_probe(args.model, args.dataset, args.checkpoint_root, device=args.device)
    else:
        result = analyze(args.dataset, args.checkpoint_root, device=args.device,
                         bootstrap_replicates=args.bootstrap_replicates,
                         bootstrap_combinations=args.bootstrap_combinations)
    print(json.dumps(result if args.stage == "preflight" else {"stage": args.stage, "complete": True},
                     sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

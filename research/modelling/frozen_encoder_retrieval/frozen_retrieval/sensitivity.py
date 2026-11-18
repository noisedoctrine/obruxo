from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from .core import MODEL_REGISTRY, digest
from .dataset import validate_dataset_manifests
from .rendering import QA_POLICY_ID, RENDER_POLICY_ID, render_plan_identity


EXPECTED_VARIANTS = {
    "cir_ast": ("official_clip",),
    "beats_iter3_plus": ("masked_mean", "temporal_pyramid_8"),
    "clap_music": ("official_clip",),
    "dac_44khz": ("masked_mean", "temporal_pyramid_8"),
    "encodec_24khz": ("masked_mean", "temporal_pyramid_8"),
    "muq_mulan": ("official_clip",),
    "basic_pitch_trunk": ("masked_mean", "temporal_pyramid_8"),
}
MODEL_ORDER = tuple(EXPECTED_VARIANTS)
PREPROCESSING_ID = "pcm_scale_stereo_mean_mono_resample_poly_no_gain_norm_v2"
ZERO_NORM_TOLERANCE = 1e-12
NEAR_ZERO_DENOMINATOR_TOLERANCE = 1e-6
NORMALIZED_MARGIN_EPSILON = 1e-8
TIE_TOLERANCE = 1e-6
BOOTSTRAP_SEED = 15720261116


def _jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _expected_dimensions(model_name: str) -> dict[str, int]:
    spec = MODEL_REGISTRY[model_name]
    return {variant: spec["dimension"] * (16 if variant == "temporal_pyramid_8" else 1)
            for variant in EXPECTED_VARIANTS[model_name]}


def _expected_pooling(model_name: str) -> str:
    return "native_clip_v1" if MODEL_REGISTRY[model_name]["kind"] == "clip" else "masked_mean_and_8bin_mean_std_v1"


def verify_embedding_cache(dataset_dir: Path, checkpoint_root: Path, *, feature_root: Path | None = None,
                           results_root: Path | None = None, normalization_target: float | None = None,
                           model_names: tuple[str, ...] | None = None, progress=None) -> dict:
    """Validate the frozen cache contract without opening source audio or loading an encoder."""
    dataset_dir = dataset_dir.resolve(strict=True)
    feature_root = (feature_root or dataset_dir / "features").resolve(strict=True)
    results_root = (results_root or dataset_dir).resolve(strict=True)
    summary = json.loads((dataset_dir / "dataset_summary.private.json").read_text(encoding="utf-8"))
    validate_dataset_manifests(dataset_dir, summary)
    if summary.get("source_split") != "validation" or summary.get("performance_source_split") != "train":
        raise ValueError("issue #157 requires the existing validation-preset/train-performance cohort")
    if summary.get("reserved_split_access") != "none" or summary.get("performance_identity_overlap_excluded") != 0:
        raise ValueError("dataset manifest does not preserve the reserved-source boundary")

    evaluation = json.loads((results_root / "evaluation.private.json").read_text(encoding="utf-8"))
    expected_arms = {f"{model}:{variant}" for model, variants in EXPECTED_VARIANTS.items() for variant in variants}
    if set(evaluation.get("available_arms", [])) != expected_arms:
        raise ValueError("existing evaluation does not contain the complete seven-family, eleven-arm cohort")
    if evaluation.get("held_out_reserved_sources_used") is not False:
        raise ValueError("existing evaluation does not affirm the held-out-source boundary")

    metric_rows = [row for row in _jsonl(results_root / "episode_metrics.private.jsonl")
                   if row.get("condition") == "cross_midi_common_gallery"]
    rows_by_arm = {arm: [row for row in metric_rows if row.get("arm") == arm] for arm in sorted(expected_arms)}
    if any(len(rows) != 6_450 for rows in rows_by_arm.values()):
        raise ValueError("the existing common-cohort evaluation is not 6,450 rows for every arm")
    anchor_rows = rows_by_arm[sorted(expected_arms)[0]]
    episode_ids = {row["episode_id"] for row in anchor_rows}
    cohort_pairs = {(row["target_state_id"], row["query_midi_id"]) for row in anchor_rows}
    if len(episode_ids) != 6_450 or len(cohort_pairs) != 6_450:
        raise ValueError("common-cohort episode identities are duplicated")
    for arm, rows in rows_by_arm.items():
        if {row["episode_id"] for row in rows} != episode_ids:
            raise ValueError(f"existing common-cohort episodes differ for {arm}")
    preset_ids = sorted({state_id for state_id, _ in cohort_pairs})
    midi_ids = sorted({midi_id for _, midi_id in cohort_pairs})
    if len(preset_ids) != 129 or len(midi_ids) != 50 or len(cohort_pairs) != len(preset_ids) * len(midi_ids):
        raise ValueError("common-cohort rows are not a complete 129 by 50 matrix")

    family_by_state = {row["state_id"]: row["family_id"] for row in _jsonl(dataset_dir / "presets.private.jsonl")}
    if not set(preset_ids).issubset(family_by_state):
        raise ValueError("common-cohort preset family metadata is incomplete")
    family_ids = {family_by_state[state_id] for state_id in preset_ids}
    if len(family_ids) != 129:
        raise ValueError("preset-family clusters are not one-to-one with the analyzed preset states")
    performances = {row["midi_id"]: row for row in _jsonl(dataset_dir / "performances.private.jsonl")}
    if any(midi_id not in performances or performances[midi_id].get("split") != "validation" or
           performances[midi_id].get("performance_source_split") != "train" for midi_id in midi_ids):
        raise ValueError("common MIDI identities do not match the held-out train-source cohort")

    plan_rows = _jsonl(dataset_dir / "render_plan.private.jsonl")
    plan_by_request = {row["request_id"]: row for row in plan_rows}
    plan_by_pair = {(row["state_id"], row["midi_id"]): row for row in plan_rows
                    if row.get("source_kind") == "newly_rendered_original_preset"}
    if len(plan_by_pair) != sum(row.get("source_kind") == "newly_rendered_original_preset" for row in plan_rows):
        raise ValueError("render plan contains duplicate preset-performance cells")
    receipts = {}
    for row in _jsonl(dataset_dir / "render_receipts.private.jsonl"):
        plan = plan_by_request.get(row.get("request_id"))
        if (not plan or row.get("render_policy_id") != RENDER_POLICY_ID or
                row.get("plan_identity") != render_plan_identity(plan)):
            raise ValueError("render receipt differs from the frozen request manifest")
        receipts[row["request_id"]] = row
    if any(row.get("qa_policy_id") != QA_POLICY_ID for row in receipts.values()):
        raise ValueError("render receipt QA policy is stale")
    if set(receipts) != set(plan_by_request):
        raise ValueError("render receipt ledger lacks a terminal record for a planned request")
    request_by_pair = {}
    for state_id, midi_id in cohort_pairs:
        plan = plan_by_pair.get((state_id, midi_id))
        receipt = receipts.get(plan["request_id"]) if plan else None
        if not plan or not receipt or not receipt.get("accepted") or receipt.get("audio_sha256") is None:
            raise ValueError("a required preset-performance cell lacks an accepted render receipt")
        request_by_pair[(state_id, midi_id)] = plan["request_id"]

    lock = json.loads((checkpoint_root / "checkpoint-lock.json").read_text(encoding="utf-8"))
    if lock.get("version") != 1 or not isinstance(lock.get("models"), dict):
        raise ValueError("local checkpoint lock is missing or unsupported")
    selected_models = MODEL_ORDER if model_names is None else tuple(model_names)
    if not selected_models or len(set(selected_models)) != len(selected_models) or not set(selected_models).issubset(MODEL_ORDER):
        raise ValueError("cache verification model selection must contain unique known encoder families")
    model_caches = {}
    shared_requests = None
    shared_pair_by_request = None
    shared_audio_by_request = None
    cache_entry_counts = set()
    for model_name in selected_models:
        model_dir = feature_root / model_name
        manifest_rows = _jsonl(model_dir / "feature_manifest.private.jsonl")
        extraction = json.loads((model_dir / "extraction.latest.json").read_text(encoding="utf-8"))
        lock_entry = lock["models"].get(model_name)
        expected_dimensions = _expected_dimensions(model_name)
        if not isinstance(lock_entry, dict) or not lock_entry.get("sha256"):
            raise ValueError(f"checkpoint provenance is missing for {model_name}")
        locked_sha = lock_entry["sha256"]
        if MODEL_REGISTRY[model_name].get("expected_sha256") not in (None, locked_sha):
            raise ValueError(f"locked checkpoint digest differs from the pinned registry for {model_name}")
        expected_input_count = 6_480 if normalization_target is None else 6_450
        if (extraction.get("model") != model_name or extraction.get("checkpoint_sha256") != locked_sha or
                extraction.get("failure_count") != 0 or extraction.get("input_render_count") != expected_input_count or
                extraction.get("frozen") is not True or extraction.get("eval_mode") is not True or
                extraction.get("dimensions_by_variant") != expected_dimensions):
            raise ValueError(f"extraction summary does not match the verified cache contract for {model_name}")
        if normalization_target is None:
            if extraction.get("preprocessing") != "stereo mean to mono; scipy polyphase resample; no loudness normalization":
                raise ValueError(f"extraction preprocessing differs from the frozen report for {model_name}")
            expected_preprocessing = PREPROCESSING_ID
        else:
            from .normalization import ACTIVE_FLOOR_DB, NORMALIZATION_VERSION, WINDOW_SECONDS

            expected_preprocessing = "pcm_scale_active_rms_stereo_mean_mono_resample_poly_v1"
            expected_normalization = {"version": NORMALIZATION_VERSION, "window_seconds": WINDOW_SECONDS,
                "active_floor_db": ACTIVE_FLOOR_DB, "target_active_rms": float(normalization_target),
                "channel_energy": "mean-square across original WAV channels",
                "placement": "before mono downmix and model-specific resampling"}
            if extraction.get("preprocessing") != "stereo active-RMS normalization before mono downmix and scipy polyphase resample" or \
                    extraction.get("normalization", {}).get("target_active_rms") != float(normalization_target) or \
                    extraction.get("normalization", {}).get("version") != NORMALIZATION_VERSION:
                raise ValueError(f"active-RMS preprocessing provenance differs for {model_name}")
        expected_extraction_pooling = ("official native clip embedding" if MODEL_REGISTRY[model_name]["kind"] == "clip"
                                       else "valid sequence frames: masked mean and 8 ordered bins of mean plus standard deviation")
        if extraction.get("pooling") != expected_extraction_pooling or extraction.get("precision") != "float32":
            raise ValueError(f"pooling or precision differs from the frozen report for {model_name}")
        if len(manifest_rows) != expected_input_count or len({row["request_id"] for row in manifest_rows}) != expected_input_count:
            raise ValueError(f"feature manifest is incomplete or duplicated for {model_name}")

        request_ids = set()
        pair_by_request = {}
        audio_by_request = {}
        for row in manifest_rows:
            request_id = row["request_id"]
            request_ids.add(request_id)
            feature_id = row["feature_id"]
            plan = plan_by_request.get(request_id)
            receipt = receipts.get(request_id)
            if (not plan or not receipt or not receipt.get("accepted") or
                    (row["state_id"], row["midi_id"]) != (plan["state_id"], plan["midi_id"]) or
                    row.get("excerpt_id") != plan.get("excerpt_id") or row.get("family_id") != plan.get("family_id") or
                    row.get("source_kind") != plan.get("source_kind") or row.get("split") != plan.get("split") or
                    row.get("audio_sha256") != receipt.get("audio_sha256")):
                raise ValueError(f"cached render identity differs from the accepted manifest for {model_name}")
            pair_by_request[request_id] = (row["state_id"], row["midi_id"])
            audio_by_request[request_id] = row["audio_sha256"]
            if row.get("dimensions") != expected_dimensions:
                raise ValueError(f"cached variant dimensions differ from the contract for {model_name}")
            metadata_path = model_dir / f"{feature_id}.json"
            archive_path = model_dir / f"{feature_id}.npz"
            relative_cache_path = Path(row.get("cache_path", ""))
            if (not metadata_path.is_file() or not archive_path.is_file() or relative_cache_path.is_absolute() or
                    ".." in relative_cache_path.parts or relative_cache_path != Path(archive_path.name)):
                raise ValueError(f"a manifested feature artifact is missing for {model_name}")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            identity = metadata.get("cache_identity", {})
            expected_local_configs = {name: {key: record[key] for key in ("repository_id", "revision", "sha256")}
                                      for name, record in lock_entry.get("local_configs", {}).items()}
            if (metadata.get("feature_id") != feature_id or digest(identity) != feature_id or
                    metadata.get("feature_sha256") != _file_sha256(archive_path) or
                    metadata.get("frozen") is not True or metadata.get("eval_mode") is not True or
                    identity.get("adapter_id") != "frozen_audio_adapters_v2" or
                    identity.get("model") != model_name or identity.get("checkpoint_sha256") != locked_sha or
                    identity.get("source_revision") != lock_entry.get("revision") or
                    identity.get("loader_revision") != lock_entry.get("loader_revision") or
                    identity.get("renderer_request_id") != request_id or
                    identity.get("audio_file_sha256") != row.get("audio_sha256") or
                    identity.get("tap") != MODEL_REGISTRY[model_name]["tap"] or
                    identity.get("sample_rate") != MODEL_REGISTRY[model_name]["sample_rate"] or
                    identity.get("preprocessing") != expected_preprocessing or
                    identity.get("pooling") != _expected_pooling(model_name) or
                    identity.get("precision") != "fp32" or
                    identity.get("local_config_pins") != expected_local_configs or
                    metadata.get("dimensions") != expected_dimensions):
                raise ValueError(f"feature provenance or integrity check failed for {model_name}")
            if normalization_target is not None:
                normalization = metadata.get("normalization", {})
                if (identity.get("input_normalization") != expected_normalization or
                        normalization.get("version") != NORMALIZATION_VERSION or
                        normalization.get("target_active_rms") != float(normalization_target) or
                        normalization.get("sample_rate") != 44_100 or
                        not np.isfinite(normalization.get("applied_gain", np.nan)) or
                        not np.isfinite(normalization.get("peak_to_active_rms", np.nan))):
                    raise ValueError(f"active-RMS cache provenance is incomplete for {model_name}")
            if model_name != "encodec_24khz" and identity.get("encoder_internal_normalization") != "none":
                raise ValueError(f"unexpected encoder-internal normalization for {model_name}")
            if model_name == "encodec_24khz" and identity.get("encoder_internal_normalization") not in {"none", "encodec_config_rms"}:
                raise ValueError("invalid EnCodec internal-normalization provenance")
            with np.load(archive_path, allow_pickle=False) as arrays:
                if set(arrays.files) != set(expected_dimensions):
                    raise ValueError(f"cached array variants differ from the contract for {model_name}")
                for variant, dimension in expected_dimensions.items():
                    vector = arrays[variant]
                    if vector.shape != (dimension,) or vector.dtype != np.float32:
                        raise ValueError(f"cached tensor shape or dtype differs from the contract for {model_name}")
                    if not np.isfinite(vector).all():
                        raise ValueError(f"cached {variant} values are nonfinite for {model_name}")
                    if float(np.linalg.norm(vector.astype(np.float64))) <= ZERO_NORM_TOLERANCE:
                        raise ValueError(f"cached {variant} contains a zero-norm vector for {model_name}")

        if shared_requests is None:
            shared_requests, shared_pair_by_request, shared_audio_by_request = request_ids, pair_by_request, audio_by_request
        elif (request_ids != shared_requests or pair_by_request != shared_pair_by_request or
              audio_by_request != shared_audio_by_request):
            raise ValueError(f"cached render matrix or WAV identity differs across encoder families at {model_name}")
        matrix_requests = {request_by_pair[pair] for pair in cohort_pairs}
        if not matrix_requests.issubset(request_ids):
            raise ValueError(f"the exact 129 by 50 cohort is incomplete for {model_name}")
        if normalization_target is not None and request_ids != matrix_requests:
            raise ValueError(f"active-RMS cache must contain exactly the common 129 by 50 cohort for {model_name}")
        cache_entry_counts.add(len(request_ids))
        model_caches[model_name] = {"directory": model_dir, "rows_by_request": {row["request_id"]: row for row in manifest_rows},
                                    "checkpoint_sha256": locked_sha, "source_revision": lock_entry["revision"],
                                    "loader_revision": lock_entry["loader_revision"], "tap": MODEL_REGISTRY[model_name]["tap"],
                                    "pooling": _expected_pooling(model_name), "preprocessing": PREPROCESSING_ID,
                                    "dimensions_by_variant": expected_dimensions, "cache_entries": len(manifest_rows)}
        if progress:
            progress(f"Verified {model_name}: {len(manifest_rows):,} cache entries, {len(expected_dimensions)} representation variants.")

    retrieval = {}
    for arm in sorted(expected_arms):
        retrieval[arm] = evaluation["model_results"][arm]["cross_midi_common_gallery"]["recall_at_1"]
    return {"dataset_dir": dataset_dir, "model_caches": model_caches, "preset_ids": preset_ids, "midi_ids": midi_ids,
            "family_ids": sorted(family_ids), "family_by_state": family_by_state,
            "request_by_pair": request_by_pair, "receipts": receipts, "retrieval_r1": retrieval,
            "expected_arms": expected_arms, "common_cells": len(cohort_pairs), "cache_entry_counts": sorted(cache_entry_counts),
            "verified_model_names": selected_models,
            "evaluation_arms": len(evaluation["available_arms"]), "source_split": summary["source_split"],
            "performance_source_split": summary["performance_source_split"], "reserved_split_access": summary["reserved_split_access"]}


def load_embedding_grid(cache_audit: dict, model_name: str, variant: str, *, device):
    import torch

    cache = cache_audit["model_caches"][model_name]
    dimension = cache["dimensions_by_variant"][variant]
    preset_lookup = {value: index for index, value in enumerate(cache_audit["preset_ids"])}
    midi_lookup = {value: index for index, value in enumerate(cache_audit["midi_ids"])}
    values = np.empty((len(preset_lookup), len(midi_lookup), dimension), dtype=np.float32)
    for (state_id, midi_id), request_id in cache_audit["request_by_pair"].items():
        row = cache["rows_by_request"][request_id]
        archive = cache["directory"] / row["cache_path"]
        with np.load(archive, allow_pickle=False) as arrays:
            values[preset_lookup[state_id], midi_lookup[midi_id]] = arrays[variant]
    grid = torch.from_numpy(values).to(device=device)
    del values
    return normalize_embedding_grid(grid)


def normalize_embedding_grid(grid):
    import torch

    finite_count = int((~torch.isfinite(grid)).sum().item())
    norms = torch.linalg.vector_norm(grid, dim=-1)
    zero_count = int((norms <= ZERO_NORM_TOLERANCE).sum().item())
    if finite_count or zero_count:
        raise ValueError(f"embedding matrix has nonfinite={finite_count}, zero_norm={zero_count}")
    grid.div_(norms.unsqueeze(-1))
    return grid, {"nonfinite_values": finite_count, "zero_norm_vectors": zero_count}


def cosine_distance_matrices(unit_grid):
    import torch

    same_preset = torch.bmm(unit_grid, unit_grid.transpose(1, 2)).clamp(-1.0, 1.0)
    by_midi = unit_grid.transpose(0, 1)
    same_midi = torch.bmm(by_midi, by_midi.transpose(1, 2)).clamp(-1.0, 1.0)
    return (1.0 - same_preset).clamp_min(0), (1.0 - same_midi).clamp_min(0)


def four_anchor_distances(within, between, preset_i, preset_j, midi_i, midi_j):
    """Return each orientation separately for broadcastable preset- and MIDI-pair indices."""
    import torch

    same_i = within[preset_i, midi_i, midi_j]
    same_j = within[preset_j, midi_i, midi_j]
    preset_at_i = between[midi_i, preset_i, preset_j]
    preset_at_j = between[midi_j, preset_i, preset_j]
    midi_change = torch.stack((same_i, same_i, same_j, same_j), dim=-1)
    preset_change = torch.stack((preset_at_i, preset_at_j, preset_at_i, preset_at_j), dim=-1)
    return midi_change, preset_change


def anchored_metrics_from_distances(d_midi, d_preset, *, denominator_tolerance=NEAR_ZERO_DENOMINATOR_TOLERANCE,
                                    margin_epsilon=NORMALIZED_MARGIN_EPSILON, tie_tolerance=TIE_TOLERANCE):
    """Calculate four orientation-specific anchored metrics without pre-averaging."""
    import torch

    if d_midi.shape != d_preset.shape:
        raise ValueError("MIDI-change and preset-change distance tensors must have the same shape")
    valid_ratio = d_preset > denominator_tolerance
    safe_denominator = d_preset.clamp_min(denominator_tolerance)
    ratio = torch.where(valid_ratio, d_midi / safe_denominator, torch.nan)
    difference = d_midi - d_preset
    margin = (d_preset - d_midi) / (d_preset + d_midi + margin_epsilon)
    delta = d_preset - d_midi
    tied = delta.abs() <= tie_tolerance
    preset_dominant = delta > tie_tolerance
    midi_dominant = delta < -tie_tolerance
    return {"ratio": ratio, "difference": difference, "margin": margin, "valid_ratio": valid_ratio,
            "tie": tied, "preset_dominant": preset_dominant, "midi_dominant": midi_dominant}


def _variance_components(unit_grid, *, chunk_presets: int = 16):
    import torch

    preset_count, midi_count, _ = unit_grid.shape
    grand = unit_grid.mean(dim=(0, 1))
    preset_means = unit_grid.mean(dim=1)
    midi_means = unit_grid.mean(dim=0)
    ss_preset = midi_count * (preset_means - grand).square().sum()
    ss_midi = preset_count * (midi_means - grand).square().sum()
    ss_total = torch.zeros((), dtype=unit_grid.dtype, device=unit_grid.device)
    ss_interaction = torch.zeros_like(ss_total)
    for start in range(0, preset_count, chunk_presets):
        stop = min(start + chunk_presets, preset_count)
        block = unit_grid[start:stop]
        centered = block - grand
        interaction = block - preset_means[start:stop, None, :] - midi_means[None, :, :] + grand
        ss_total += centered.square().sum()
        ss_interaction += interaction.square().sum()
    return ss_total, ss_preset, ss_midi, ss_interaction


def variance_decomposition(unit_grid, *, chunk_presets: int = 16) -> dict:
    """Compute balanced preset, performance, and interaction sums without a full residual copy."""
    ss_total, ss_preset, ss_midi, ss_interaction = _variance_components(unit_grid, chunk_presets=chunk_presets)
    total = float(ss_total.item())
    components = {"preset": float(ss_preset.item()), "midi": float(ss_midi.item()),
                  "interaction": float(ss_interaction.item())}
    if total <= 0:
        return {"status": "zero_total_variation", "ss_total": total,
                "ss_components": components, "shares": {key: None for key in components},
                "additivity_error": abs(sum(components.values()) - total)}
    error = abs(sum(components.values()) - total)
    if error > max(1e-3, total * 5e-4):
        raise ArithmeticError("preset, MIDI, and interaction sums do not add to total variation")
    shares = {key: value / total for key, value in components.items()}
    if abs(sum(shares.values()) - 1.0) > 5e-4:
        raise ArithmeticError("variance component shares do not sum to 100 percent")
    return {"status": "ok", "ss_total": total, "ss_components": components, "shares": shares,
            "additivity_error": error, "additivity_error_fraction": error / total}


def _quantiles(values, count: int, probabilities=(0.1, 0.25, 0.5, 0.75, 0.9)) -> dict:
    if count == 0:
        return {f"p{int(probability * 100):02d}": None for probability in probabilities}
    result = np.quantile(values[:count], probabilities, method="linear", overwrite_input=True)
    return {f"p{int(probability * 100):02d}": float(value) for probability, value in zip(probabilities, result, strict=True)}


def _sample_distinct_original_pairs(resampled_indices: np.ndarray, pair_first: np.ndarray,
                                    pair_second: np.ndarray, *, sample_count: int,
                                    generator: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Sample distinct source identities with pair probability proportional to multiplicity products."""
    counts = np.bincount(resampled_indices, minlength=max(pair_first.max(), pair_second.max()) + 1)
    pair_weights = counts[pair_first] * counts[pair_second]
    total_weight = int(pair_weights.sum())
    if total_weight == 0:
        raise ValueError("a crossed bootstrap replicate must contain at least two distinct source identities")
    positions = generator.choice(len(pair_weights), size=sample_count, p=pair_weights / total_weight)
    return pair_first[positions], pair_second[positions]


def _pair_bootstrap_distances(within, between, preset_draws: np.ndarray, midi_draws: np.ndarray, *,
                              replicates: int, combinations_per_replicate: int, device):
    import torch

    preset_count, midi_count = within.shape[:2]
    generator = np.random.default_rng(BOOTSTRAP_SEED + 1)
    preset_pair_first, preset_pair_second = np.triu_indices(preset_count, k=1)
    midi_pair_first, midi_pair_second = np.triu_indices(midi_count, k=1)
    output = {name: np.empty(replicates, dtype=np.float64) for name in
              ("mean_midi_distance", "mean_preset_distance", "mean_ratio", "mean_difference", "mean_margin",
               "preset_dominant_fraction", "midi_dominant_fraction", "tie_fraction")}
    for replicate in range(replicates):
        p1_np, p2_np = _sample_distinct_original_pairs(
            preset_draws[replicate], preset_pair_first, preset_pair_second,
            sample_count=combinations_per_replicate, generator=generator)
        m1_np, m2_np = _sample_distinct_original_pairs(
            midi_draws[replicate], midi_pair_first, midi_pair_second,
            sample_count=combinations_per_replicate, generator=generator)
        p1 = torch.as_tensor(p1_np, dtype=torch.long, device=device)
        p2 = torch.as_tensor(p2_np, dtype=torch.long, device=device)
        m1 = torch.as_tensor(m1_np, dtype=torch.long, device=device)
        m2 = torch.as_tensor(m2_np, dtype=torch.long, device=device)
        dm, dp = four_anchor_distances(within, between, p1, p2, m1, m2)
        metrics = anchored_metrics_from_distances(dm, dp)
        ratio_values = metrics["ratio"]
        valid = metrics["valid_ratio"]
        valid_count = valid.sum()
        ratio_mean = ratio_values.nan_to_num().sum() / valid_count.clamp_min(1)
        ratio_mean = torch.where(valid_count > 0, ratio_mean, torch.nan)
        summary = torch.stack((dm.mean(), dp.mean(), ratio_mean, metrics["difference"].mean(),
                               metrics["margin"].mean(), metrics["preset_dominant"].float().mean(),
                               metrics["midi_dominant"].float().mean(), metrics["tie"].float().mean()))
        values = summary.detach().cpu().numpy()
        for name, value in zip(output, values, strict=True):
            output[name][replicate] = value
    return output


def _variance_bootstrap(unit_grid, preset_draws: np.ndarray, midi_draws: np.ndarray) -> dict:
    import torch

    preset_count, midi_count, dimension = unit_grid.shape
    replicates = preset_draws.shape[0]
    samples = {name: np.empty(replicates, dtype=np.float64) for name in ("preset", "midi", "interaction")}
    flat = unit_grid.reshape(preset_count * midi_count, dimension)
    for replicate in range(replicates):
        preset_indices = torch.as_tensor(preset_draws[replicate], dtype=torch.long, device=unit_grid.device)
        midi_indices = torch.as_tensor(midi_draws[replicate], dtype=torch.long, device=unit_grid.device)
        flat_indices = (preset_indices[:, None] * midi_count + midi_indices[None, :]).reshape(-1)
        sample = flat.index_select(0, flat_indices).reshape(preset_count, midi_count, dimension)
        total, preset, midi, interaction = _variance_components(sample)
        share_tensor = torch.stack((preset, midi, interaction)) / total.clamp_min(1e-20)
        share_tensor = torch.where(total > 0, share_tensor, torch.full_like(share_tensor, math.nan))
        shares = share_tensor.detach().cpu().numpy()
        for name, value in zip(samples, shares, strict=True):
            samples[name][replicate] = value
        del sample
    return samples


def _interval(values: np.ndarray) -> list[float] | None:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return None
    return [float(value) for value in np.quantile(finite, [0.025, 0.975])]


def _paired_effect_summary(observed: float, baseline_observed: float, bootstrap_values: np.ndarray,
                           baseline_bootstrap_values: np.ndarray) -> dict:
    bootstrap_differences = bootstrap_values - baseline_bootstrap_values
    return {"difference": float(observed - baseline_observed),
            "confidence_interval_95": _interval(bootstrap_differences)}


def _distribution_histograms(ratio_values: np.ndarray, ratio_count: int, difference_values: np.ndarray,
                             margin_values: np.ndarray, anchor_count: int) -> dict:
    ratio_edges = np.geomspace(1e-4, 1e4, 201)
    ratios = ratio_values[:ratio_count]
    ratio_counts = np.histogram(ratios[(ratios >= ratio_edges[0]) & (ratios <= ratio_edges[-1])], bins=ratio_edges)[0]
    ratio_zero = int(np.count_nonzero(ratios == 0))
    ratio_underflow = int(np.count_nonzero((ratios > 0) & (ratios < ratio_edges[0])))
    ratio_overflow = int(np.count_nonzero(ratios > ratio_edges[-1]))
    delta_edges = np.linspace(-2.0, 2.0, 201)
    margin_edges = np.linspace(-1.0, 1.0, 201)
    deltas = difference_values[:anchor_count]
    margins = margin_values[:anchor_count]
    return {"ratio_log10_edges": ratio_edges.tolist(), "ratio_counts": ratio_counts.astype(int).tolist(),
            "ratio_zero": ratio_zero, "ratio_underflow": ratio_underflow, "ratio_overflow": ratio_overflow,
            "difference_edges": delta_edges.tolist(),
            "difference_counts": np.histogram(deltas, bins=delta_edges)[0].astype(int).tolist(),
            "margin_edges": margin_edges.tolist(),
            "margin_counts": np.histogram(margins, bins=margin_edges)[0].astype(int).tolist()}


def summarize_arm(unit_grid, *, chunk_preset_pairs: int = 64, temp_dir: Path, bootstrap_replicates: int,
                  bootstrap_combinations: int, preset_draws: np.ndarray, midi_draws: np.ndarray,
                  device) -> tuple[dict, dict]:
    import torch

    preset_count, midi_count, dimension = unit_grid.shape
    within, between = cosine_distance_matrices(unit_grid)
    preset_i, preset_j = torch.triu_indices(preset_count, preset_count, offset=1, device=device)
    midi_i, midi_j = torch.triu_indices(midi_count, midi_count, offset=1, device=device)
    pair_count = int(preset_i.numel() * midi_i.numel())
    anchor_count = pair_count * 4
    ratio_path, difference_path, margin_path = (temp_dir / f"metric_{name}.bin" for name in ("ratio", "difference", "margin"))
    ratio_values = np.memmap(ratio_path, dtype=np.float32, mode="w+", shape=(anchor_count,))
    difference_values = np.memmap(difference_path, dtype=np.float32, mode="w+", shape=(anchor_count,))
    margin_values = np.memmap(margin_path, dtype=np.float32, mode="w+", shape=(anchor_count,))
    offset = 0
    ratio_offset = 0
    ratio_sum = 0.0
    midi_sum = preset_sum = difference_sum = margin_sum = 0.0
    zero_denominator_count = both_near_zero_count = tie_count = preset_dominant_count = midi_dominant_count = 0
    nonfinite_count = 0
    preset_margin_sums = np.zeros(preset_count, dtype=np.float64)
    midi_margin_sums = np.zeros(midi_count, dtype=np.float64)
    preset_i_cpu = preset_i.cpu().numpy()
    preset_j_cpu = preset_j.cpu().numpy()
    midi_i_cpu = midi_i.cpu().numpy()
    midi_j_cpu = midi_j.cpu().numpy()
    for start in range(0, len(preset_i_cpu), chunk_preset_pairs):
        stop = min(start + chunk_preset_pairs, len(preset_i_cpu))
        pi = preset_i[start:stop]
        pj = preset_j[start:stop]
        dm, dp = four_anchor_distances(within, between, pi[:, None], pj[:, None], midi_i[None, :], midi_j[None, :])
        metrics = anchored_metrics_from_distances(dm, dp)
        ratio_np = metrics["ratio"].reshape(-1).cpu().numpy()
        ratio_valid = np.isfinite(ratio_np)
        ratio_np = ratio_np[ratio_valid]
        difference_np = metrics["difference"].reshape(-1).cpu().numpy()
        margin_np = metrics["margin"].reshape(-1).cpu().numpy()
        dm_np = dm.reshape(-1).cpu().numpy()
        dp_np = dp.reshape(-1).cpu().numpy()
        finite = np.isfinite(dm_np) & np.isfinite(dp_np) & np.isfinite(difference_np) & np.isfinite(margin_np)
        nonfinite_count += int((~finite).sum())
        if nonfinite_count:
            raise ValueError("cosine distance matrix produced nonfinite metric values")
        chunk_count = len(difference_np)
        difference_values[offset:offset + chunk_count] = difference_np
        margin_values[offset:offset + chunk_count] = margin_np
        valid_count = len(ratio_np)
        ratio_values[ratio_offset:ratio_offset + valid_count] = ratio_np
        ratio_offset += valid_count
        offset += chunk_count
        ratio_sum += float(ratio_np.sum(dtype=np.float64))
        midi_sum += float(dm_np.sum(dtype=np.float64))
        preset_sum += float(dp_np.sum(dtype=np.float64))
        difference_sum += float(difference_np.sum(dtype=np.float64))
        margin_sum += float(margin_np.sum(dtype=np.float64))
        ties = metrics["tie"].reshape(-1).cpu().numpy()
        preset_dominant = metrics["preset_dominant"].reshape(-1).cpu().numpy()
        midi_dominant = metrics["midi_dominant"].reshape(-1).cpu().numpy()
        zero_denominator_count += int((dp_np <= NEAR_ZERO_DENOMINATOR_TOLERANCE).sum())
        both_near_zero_count += int(((dp_np <= NEAR_ZERO_DENOMINATOR_TOLERANCE) &
                                     (dm_np <= NEAR_ZERO_DENOMINATOR_TOLERANCE)).sum())
        tie_count += int(ties.sum())
        preset_dominant_count += int(preset_dominant.sum())
        midi_dominant_count += int(midi_dominant.sum())
        margin_chunk = margin_np.reshape(stop - start, -1, 4)
        np.add.at(preset_margin_sums, preset_i_cpu[start:stop], margin_chunk[:, :, :2].sum(axis=(1, 2)))
        np.add.at(preset_margin_sums, preset_j_cpu[start:stop], margin_chunk[:, :, 2:].sum(axis=(1, 2)))
        np.add.at(midi_margin_sums, midi_i_cpu, margin_chunk[:, :, [0, 2]].sum(axis=(0, 2)))
        np.add.at(midi_margin_sums, midi_j_cpu, margin_chunk[:, :, [1, 3]].sum(axis=(0, 2)))
        del dm, dp, metrics

    if offset != anchor_count or preset_dominant_count + midi_dominant_count + tie_count != anchor_count:
        raise ArithmeticError("exhaustive anchor accounting is incomplete")
    ratio_values.flush()
    difference_values.flush()
    margin_values.flush()
    histograms = _distribution_histograms(ratio_values, ratio_offset, difference_values, margin_values, anchor_count)
    quantiles = {"ratio": _quantiles(ratio_values, ratio_offset),
                 "difference": _quantiles(difference_values, anchor_count),
                 "margin": _quantiles(margin_values, anchor_count)}
    point = {"dimension": dimension, "input_shape": [preset_count, midi_count, dimension],
             "unordered_2x2_combinations": pair_count, "anchored_observations": anchor_count,
             "mean_midi_change_distance": midi_sum / anchor_count,
             "mean_preset_change_distance": preset_sum / anchor_count,
             "mean_raw_difference_midi_minus_preset": difference_sum / anchor_count,
             "mean_ratio": ratio_sum / ratio_offset if ratio_offset else None,
             "mean_normalized_margin": margin_sum / anchor_count,
             "preset_dominant_fraction": preset_dominant_count / anchor_count,
             "midi_dominant_fraction": midi_dominant_count / anchor_count,
             "tie_fraction": tie_count / anchor_count,
             "preset_dominant_count": preset_dominant_count, "midi_dominant_count": midi_dominant_count,
             "tie_count": tie_count,
             "ratio_defined_count": ratio_offset, "near_zero_preset_distance_denominator_count": zero_denominator_count,
             "both_distances_near_zero_count": both_near_zero_count,
             "nonfinite_metric_values": nonfinite_count,
             "quantiles": quantiles,
             "per_preset_mean_margin_quantiles": _quantiles(preset_margin_sums / ((preset_count - 1) * (midi_count * (midi_count - 1) // 2) * 2),
                                                              preset_count),
             "per_midi_mean_margin_quantiles": _quantiles(midi_margin_sums / ((midi_count - 1) * (preset_count * (preset_count - 1) // 2) * 2),
                                                            midi_count),
             "histograms": histograms}
    variance = variance_decomposition(unit_grid)
    point["variance_decomposition"] = variance
    pair_bootstrap = _pair_bootstrap_distances(within, between, preset_draws, midi_draws,
                                                replicates=bootstrap_replicates,
                                                combinations_per_replicate=bootstrap_combinations, device=device)
    variance_bootstrap = _variance_bootstrap(unit_grid, preset_draws, midi_draws)
    point["confidence_intervals_95"] = {name: _interval(values) for name, values in pair_bootstrap.items()}
    point["variance_decomposition"]["confidence_intervals_95"] = {
        name: _interval(values) for name, values in variance_bootstrap.items()}
    bootstrap = {"relative_metrics": pair_bootstrap, "variance_shares": variance_bootstrap}
    del within, between, preset_i, preset_j, midi_i, midi_j
    del ratio_values, difference_values, margin_values
    return point, bootstrap


def analyze_cache(cache_audit: dict, *, device_name: str = "cpu", bootstrap_replicates: int = 1_000,
                  bootstrap_combinations: int = 32_768, chunk_preset_pairs: int = 64, progress=None) -> dict:
    import torch

    if bootstrap_replicates < 100 or bootstrap_combinations < 1_024:
        raise ValueError("bootstrap requires at least 100 replicates and 1,024 sampled 2x2 combinations")
    if device_name == "xpu" and (not hasattr(torch, "xpu") or not torch.xpu.is_available()):
        raise ValueError("XPU was selected but is not available")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was selected but is not available")
    if device_name not in {"cpu", "xpu", "cuda"}:
        raise ValueError("compute device must be cpu, xpu, or cuda")
    device = torch.device(device_name)
    if device_name == "cpu":
        torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    preset_draws = generator.integers(0, len(cache_audit["preset_ids"]),
                                      size=(bootstrap_replicates, len(cache_audit["preset_ids"])), dtype=np.int16)
    midi_draws = generator.integers(0, len(cache_audit["midi_ids"]),
                                    size=(bootstrap_replicates, len(cache_audit["midi_ids"])), dtype=np.int8)
    started = time.perf_counter()
    arm_results = {}
    arm_bootstrap = {}
    with tempfile.TemporaryDirectory(prefix="obruxo-issue157-") as temporary:
        temp_dir = Path(temporary)
        for model_name in MODEL_ORDER:
            for variant in EXPECTED_VARIANTS[model_name]:
                arm = f"{model_name}:{variant}"
                unit_grid, vector_checks = load_embedding_grid(cache_audit, model_name, variant, device=device)
                point, bootstrap = summarize_arm(unit_grid, temp_dir=temp_dir, bootstrap_replicates=bootstrap_replicates,
                                                bootstrap_combinations=bootstrap_combinations,
                                                preset_draws=preset_draws, midi_draws=midi_draws,
                                                chunk_preset_pairs=chunk_preset_pairs, device=device)
                point["vector_validation"] = vector_checks
                point["retrieval_recall_at_1"] = cache_audit["retrieval_r1"][arm]
                arm_results[arm] = point
                arm_bootstrap[arm] = bootstrap
                if progress:
                    progress(f"Scored {arm}: median ratio {point['quantiles']['ratio']['p50']:.3f}; "
                             f"mean margin {point['mean_normalized_margin']:+.3f}; "
                             f"preset share {point['variance_decomposition']['shares']['preset']:.1%}.")
                del unit_grid
                if device_name == "xpu":
                    torch.xpu.synchronize()
        del temp_dir
    baseline = "cir_ast:official_clip"
    paired = {}
    for arm, samples in arm_bootstrap.items():
        if arm == baseline:
            continue
        paired[arm] = {}
        for metric in ("mean_margin", "preset_dominant_fraction"):
            point_key = "mean_normalized_margin" if metric == "mean_margin" else "preset_dominant_fraction"
            paired[arm][metric] = _paired_effect_summary(
                arm_results[arm][point_key], arm_results[baseline][point_key],
                samples["relative_metrics"][metric], arm_bootstrap[baseline]["relative_metrics"][metric])
    for arm, point in arm_results.items():
        if point["anchored_observations"] != 40_454_400:
            raise ArithmeticError(f"full anchor population differs from the expected matrix for {arm}")
    if device_name == "xpu":
        torch.xpu.synchronize()
    return {"version": 2, "analysis": "issue157_preset_midi_sensitivity_v2",
            "cohort": {"preset_states": len(cache_audit["preset_ids"]), "preset_family_clusters": len(cache_audit["family_ids"]),
                       "midi_performances": len(cache_audit["midi_ids"]), "matrix_cells": cache_audit["common_cells"],
                       "source_preset_split": cache_audit["source_split"], "performance_source_split": cache_audit["performance_source_split"],
                       "reserved_split_access": cache_audit["reserved_split_access"],
                       "unordered_2x2_combinations": 10_113_600, "anchored_observations_per_arm": 40_454_400},
            "cache_validation": {"encoder_families": len(MODEL_ORDER), "representation_arms": len(cache_audit["expected_arms"]),
                                 "cache_entries_per_family": cache_audit["cache_entry_counts"],
                                 "verified_common_cells_per_arm": cache_audit["common_cells"],
                                 "all_arms_share_identical_render_matrix": True,
                                 "checkpoint_provenance": {model: {key: value for key, value in cache_audit["model_caches"][model].items()
                                                                    if key not in {"directory", "rows_by_request"}}
                                                              for model in MODEL_ORDER}},
            "statistics": {"bootstrap_method": "crossed preset-family × MIDI-performance cluster bootstrap; anchored distance pairs use distinct original identities sampled in proportion to bootstrap multiplicity products; variance shares use the full crossed resampled grid; identical cluster draws and sampled pair positions across arms",
                           "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_replicates": bootstrap_replicates,
                           "sampled_unordered_2x2_combinations_per_bootstrap_replicate": bootstrap_combinations,
                           "bootstrap_combination_sampling": "Monte Carlo estimate of the exhaustive 2x2 statistic; preset and MIDI pairs are sampled from their resampled clusters conditioned on distinct original identities, yielding multiplicity-product pair weights; full-cohort point summaries are exhaustive",
                           "paired_effect_estimator": "difference between observed exhaustive full-cohort point estimates; paired bootstrap distribution supplies the confidence interval only",
                           "tie_tolerance_cosine_distance": TIE_TOLERANCE,
                           "near_zero_preset_distance_ratio_denominator": NEAR_ZERO_DENOMINATOR_TOLERANCE,
                           "normalized_margin_epsilon": NORMALIZED_MARGIN_EPSILON,
                           "zero_norm_tolerance": ZERO_NORM_TOLERANCE,
                           "ratio_rule": "undefined if different-preset same-MIDI cosine distance is at or below the near-zero tolerance",
                           "preset_dominant_rule": "different-preset same-MIDI distance exceeds same-preset different-MIDI distance by more than tie tolerance",
                           "confidence_interval": "percentile 95% interval over crossed cluster resamples; intervals do not treat anchor observations as independent"},
            "compute": {"device": device_name, "dtype": "float32", "torch_version": torch.__version__,
                        "elapsed_seconds": time.perf_counter() - started,
                        "input_shapes_by_arm": {arm: point["input_shape"] for arm, point in arm_results.items()},
                        "distance_tensor_shapes": {"same_preset_pairwise": [len(cache_audit["preset_ids"]), len(cache_audit["midi_ids"]), len(cache_audit["midi_ids"])],
                                                    "same_midi_pairwise": [len(cache_audit["midi_ids"]), len(cache_audit["preset_ids"]), len(cache_audit["preset_ids"])]},
                        "preset_pair_chunk_size": chunk_preset_pairs},
            "difficult_preset_stratum": {"status": "not_available", "reason": "Existing source-family metadata maps each of the 129 presets to a singleton family and contains no independent difficulty label; no model-specific distance or retrieval score was used to select a subset."},
            "arms": arm_results, "paired_differences_vs_cir": paired}


def generate_figures(result: dict, output_dir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    arms = result["arms"]
    labels = list(arms)
    figure_paths = []
    metrics = [
        ("Median ratio", lambda arm: arm["quantiles"]["ratio"]["p50"]),
        ("Mean normalized margin", lambda arm: arm["mean_normalized_margin"]),
        ("Preset-dominant fraction (%)", lambda arm: 100 * arm["preset_dominant_fraction"]),
        ("Preset variation (%)", lambda arm: 100 * arm["variance_decomposition"]["shares"]["preset"]),
        ("MIDI variation (%)", lambda arm: 100 * arm["variance_decomposition"]["shares"]["midi"]),
        ("Interaction variation (%)", lambda arm: 100 * arm["variance_decomposition"]["shares"]["interaction"]),
    ]
    matrix = np.asarray([[metric(arms[name]) for name in labels] for _, metric in metrics], dtype=float).T
    color_values = matrix.copy()
    color_values[:, 0] = np.log10(np.maximum(color_values[:, 0], 1e-8))
    scaled = np.zeros_like(color_values)
    for column in range(color_values.shape[1]):
        low, high = color_values[:, column].min(), color_values[:, column].max()
        scaled[:, column] = (color_values[:, column] - low) / (high - low) if high > low else 0.5
    fig, ax = plt.subplots(figsize=(14.0, 8.5), constrained_layout=True)
    image = ax.imshow(scaled, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(metrics)), [name for name, _ in metrics], rotation=20, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(labels)), labels, fontsize=8)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            label = f"{value:.2f}" if column < 2 else f"{value:.1f}%"
            ax.text(column, row, label, ha="center", va="center", color="white" if scaled[row, column] < 0.55 else "black", fontsize=8)
    fig.colorbar(image, ax=ax, label="Relative scale within each metric column (cell labels show actual values)")
    ax.set_title("Preset-versus-performance sensitivity across all 11 representations")
    path = output_dir / "issue157_sensitivity_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figure_paths.append(path.name)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    for label, arm in arms.items():
        hist = arm["histograms"]
        ratio_edges = np.asarray(hist["ratio_log10_edges"])
        ratio_counts = np.asarray(hist["ratio_counts"], dtype=float)
        ratio_centers = np.sqrt(ratio_edges[:-1] * ratio_edges[1:])
        ratio_total = max(arm["ratio_defined_count"], 1)
        axes[0].plot(ratio_centers, ratio_counts / ratio_total, label=label, linewidth=1.25)
        margin_edges = np.asarray(hist["margin_edges"])
        margin_counts = np.asarray(hist["margin_counts"], dtype=float)
        axes[1].plot((margin_edges[:-1] + margin_edges[1:]) / 2,
                     margin_counts / arm["anchored_observations"], label=label, linewidth=1.25)
    axes[0].set_xscale("log")
    axes[0].axvline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Distance ratio: MIDI-change distance ÷ preset-change distance")
    axes[0].set_ylabel("Fraction of defined anchored observations per bin")
    axes[0].set_title("Ratio (below 1 means the preset change is larger)")
    axes[1].axvline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Normalized margin (positive means preset change is larger)")
    axes[1].set_ylabel("Fraction of all anchored observations per bin")
    axes[1].set_title("Normalized-margin distribution")
    axes[0].legend(fontsize=7, ncol=2)
    path = output_dir / "issue157_distance_distributions.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figure_paths.append(path.name)

    components = np.asarray([[100 * arms[label]["variance_decomposition"]["shares"][key]
                               for key in ("preset", "midi", "interaction")] for label in labels])
    fig, ax = plt.subplots(figsize=(12.5, 6.8), constrained_layout=True)
    bottom = np.zeros(len(labels))
    colors = ("#3569a8", "#e2a43a", "#6da56d")
    for column, (name, color) in enumerate(zip(("Preset", "MIDI performance", "Interaction"), colors, strict=True)):
        ax.barh(np.arange(len(labels)), components[:, column], left=bottom, label=name, color=color)
        bottom += components[:, column]
    ax.set_yticks(np.arange(len(labels)), labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of total squared variation (%)")
    ax.set_title("Where normalized embedding variation comes from")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    path = output_dir / "issue157_variance_decomposition.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figure_paths.append(path.name)

    low = np.asarray([[[0.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [2.0, 1.0]]], dtype=float)
    high = np.asarray([[[0.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [3.0, 0.0]]], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for ax, grid, title, shares in ((axes[0], low, "Similar MIDI shifts: little interaction", (80.0, 20.0, 0.0)),
                                    (axes[1], high, "Different MIDI shifts: interaction", (86.7, 6.7, 6.7))):
        for preset_index, (preset_label, color) in enumerate((("Preset A", "#3569a8"), ("Preset B", "#dc7444"))):
            first, second = grid[preset_index]
            ax.scatter(*first, color=color, s=55)
            ax.scatter(*second, color=color, marker="s", s=55)
            ax.annotate("", xy=second, xytext=first,
                        arrowprops={"arrowstyle": "->", "color": color, "lw": 2})
            ax.text(*(first + np.asarray([0.05, -0.08])), f"{preset_label}, MIDI 1", fontsize=8, color=color)
            ax.text(*(second + np.asarray([0.05, 0.04])), f"{preset_label}, MIDI 2", fontsize=8, color=color)
        ax.set_title(f"{title}\nPreset {shares[0]:.1f}% · MIDI {shares[1]:.1f}% · interaction {shares[2]:.1f}%")
        ax.set_xlabel("Representation dimension 1 (illustrative units)")
        ax.set_ylabel("Representation dimension 2 (illustrative units)")
        ax.grid(alpha=0.2)
        ax.set_aspect("equal", adjustable="datalim")
    path = output_dir / "issue157_interaction_geometry.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figure_paths.append(path.name)
    return figure_paths


def run_report(dataset_dir: Path, checkpoint_root: Path, report_path: Path, *, device: str,
               bootstrap_replicates: int = 1_000, bootstrap_combinations: int = 32_768,
               chunk_preset_pairs: int = 64) -> tuple[dict, list[str]]:
    audit = verify_embedding_cache(dataset_dir, checkpoint_root)
    result = analyze_cache(audit, device_name=device, bootstrap_replicates=bootstrap_replicates,
                           bootstrap_combinations=bootstrap_combinations, chunk_preset_pairs=chunk_preset_pairs)
    figure_paths = generate_figures(result, report_path.parent / "figures")
    result["figures"] = figure_paths
    result_path = report_path.parent / "issue157_sensitivity_results.json"
    report_path.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result, figure_paths

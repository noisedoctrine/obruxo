from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time


PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[2]
for candidate in (REPO / "research/data_generation", REPO / "research/training",
                  REPO / "research/modelling/basic_pitch", PACKAGE):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render/evaluate controlled Vital preset perturbations on real validation MIDI")
    parser.add_argument("--dataset", type=Path, required=True, help="prepared real validation dataset used by the main benchmark")
    parser.add_argument("--renderer-config", type=Path, required=True)
    parser.add_argument("--plugin-path", type=Path)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=("clap_music", "muq_mulan", "dac_44khz", "encodec_24khz"),
                        default=("clap_music", "dac_44khz"))
    parser.add_argument("--device", choices=("cpu", "cuda", "xpu"), default="cpu")
    return parser


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _cosine(left, right, np) -> float:
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def _cross_seed_keys(target: str, midi_id: str, variants: list[str], query_seed: str, gallery_seed: str):
    if query_seed == gallery_seed:
        raise ValueError("controlled retrieval requires a distinct independent gallery render seed")
    query_key = (target, midi_id, query_seed)
    gallery_keys = [(candidate, midi_id, gallery_seed) for candidate in variants]
    if query_key in gallery_keys:
        raise ValueError("query render cannot appear in its candidate gallery")
    return query_key, gallery_keys


def _select_diagnostic_performances(performances: list[dict]) -> list[dict]:
    chosen = sorted(performances, key=lambda item: item["midi_id"])
    midi_ids = {item["midi_id"] for item in chosen}
    if len(chosen) < 3 or len(midi_ids) != len(chosen):
        raise ValueError("controlled experiment needs at least three distinct real MIDI excerpts")
    return chosen


def _bootstrap_midi_mean(rows: list[dict], metric: str, *, seed: int = 0, replicates: int = 2000) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["midi_id"]].append(float(row[metric]))
    means = [sum(values) / len(values) for _, values in sorted(grouped.items())]
    if not means:
        return {"estimate": None, "lower_95": None, "upper_95": None, "midi_clusters": 0}
    rng, draws = random.Random(seed), []
    for _ in range(replicates):
        draws.append(sum(means[rng.randrange(len(means))] for _ in means) / len(means))
    draws.sort()
    return {"estimate": round(sum(means) / len(means), 6),
            "lower_95": round(draws[int(0.025 * (replicates - 1))], 6),
            "upper_95": round(draws[int(0.975 * (replicates - 1))], 6),
            "midi_clusters": len(means), "replicates": replicates, "seed": seed}


def _seeded_preset(preset, seed_index: int, vital_preset_type):
    value = preset.to_dict()
    if seed_index:
        for index, seed in enumerate(value["settings"]["random_values"]):
            seed["seed"] = (int(seed["seed"]) + seed_index * (104_729 + index * 7_919)) % 2**31
    return vital_preset_type(value, preset.schema)


def _spectral_descriptor(wav_path: Path, np):
    from scipy.io import wavfile

    sample_rate, audio = wavfile.read(wav_path)
    audio = np.asarray(audio)
    if np.issubdtype(audio.dtype, np.integer):
        limits = np.iinfo(audio.dtype)
        audio = audio.astype(np.float32) / max(abs(limits.min), limits.max)
    else:
        audio = audio.astype(np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.ndim != 1 or sample_rate < 32_000:
        raise ValueError("controlled audio descriptor requires mono or stereo Vital audio at 32 kHz or higher")
    fft_size, hop_size = 4096, 2048
    if audio.size < fft_size:
        audio = np.pad(audio, (0, fft_size - audio.size))
    frames = np.lib.stride_tricks.sliding_window_view(audio, fft_size)[::hop_size]
    frames = frames * np.hanning(fft_size).astype(np.float32)
    frame_rms = np.sqrt(np.mean(frames * frames, axis=1))
    active = frames[frame_rms > max(float(frame_rms.max()) * 0.01, 1e-8)]
    if not len(active):
        raise ValueError("controlled Vital render has no active spectral frames")
    power = np.square(np.abs(np.fft.rfft(active, axis=1)))
    power /= np.maximum(power.sum(axis=1, keepdims=True), 1e-12)
    frequencies = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)
    edges = np.geomspace(45.0, 16_000.0, 65)
    bands = [float(power[:, (frequencies >= low) & (frequencies < high)].sum(axis=1).mean())
             for low, high in zip(edges[:-1], edges[1:], strict=True)]
    vector = np.log(np.asarray(bands, dtype=np.float32) + 1e-12)
    vector -= vector.mean()
    norm = np.linalg.norm(vector)
    return vector / norm if norm else None


def run(args: argparse.Namespace) -> dict:
    import numpy as np
    import yaml
    from obruxo_data.midi import Performance
    from obruxo_data.render.base import FULL_CORPUS_RENDERER_ID, RenderRequest
    from obruxo_data.render.vita import VitalRenderer
    from obruxo_data.vital import VitalPreset
    from frozen_retrieval.core import _rank_metrics
    from frozen_retrieval.encoders import cache_feature, file_sha256, load_checkpoint_lock, load_adapter, resolve_checkpoint
    from frozen_retrieval.rendering import _accepted, _qa_warnings, _rss_bytes

    dataset = args.dataset.resolve(strict=True)
    summary = json.loads((dataset / "dataset_summary.private.json").read_text(encoding="utf-8"))
    source_root = Path(summary["source_root"]).resolve(strict=True)
    presets = _read_jsonl(dataset / "presets.private.jsonl")
    performances = _read_jsonl(dataset / "performances.private.jsonl")
    render_plan = _read_jsonl(dataset / "render_plan.private.jsonl")
    receipts = {row["request_id"]: row for row in _read_jsonl(dataset / "render_receipts.private.jsonl")}
    request_by_pair = {(row["state_id"], row["midi_id"]): row["request_id"] for row in render_plan}
    qa_candidates = json.loads((dataset / "candidate_sets.render_qualified.private.json").read_text(encoding="utf-8"))
    qa_eligible_states = {state_id for candidate in qa_candidates["candidate_sets"].values()
                          for state_id in candidate["state_ids"]}
    renderer_config = yaml.safe_load(args.renderer_config.read_text(encoding="utf-8"))
    if renderer_config.get("renderer_id") != FULL_CORPUS_RENDERER_ID:
        raise ValueError("controlled runs require the corpus-qualified Vital renderer")

    chosen = _select_diagnostic_performances(performances)

    eligible_presets = []
    for row in sorted(presets, key=lambda item: item["state_id"]):
        if row["state_id"] not in qa_eligible_states:
            continue
        if not all((request_id := request_by_pair.get((row["state_id"], midi["midi_id"]))) and
                   receipts.get(request_id, {}).get("accepted") for midi in chosen):
            continue
        path = source_root / row["relative_path"]
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["raw_sha256"]:
            raise ValueError("controlled experiment source preset changed after the main dataset was frozen")
        candidate = VitalPreset.load(path)
        routes = [route for route in candidate.to_dict()["settings"]["modulations"] if route.get("source")]
        lfo_routes = [route for route in routes if route["source"].startswith("lfo_") and
                      route["source"] + "_frequency" in candidate.to_dict()["settings"]]
        if lfo_routes:
            eligible_presets.append((len(candidate.validate().diagnostics), row["state_id"], row, candidate,
                                     lfo_routes[0]))
    if not eligible_presets:
        raise ValueError("no render-qualified preset with an active LFO route has accepted renders for the selected MIDI")
    _, _, selected_preset, original, route = min(eligible_presets, key=lambda item: (item[0], item[1]))
    original_validation_diagnostics = [item.to_dict() for item in original.validate().diagnostics]

    base_variants = {"original": VitalPreset(original.to_dict(), original.schema)}
    controls = {}
    for variant_name, parameter, delta in (("filter_cutoff_shift", "filter_1_cutoff", 0.20),
                                           ("amp_attack_shift", "env_1_attack", 0.20)):
        variant = VitalPreset(original.to_dict(), original.schema)
        before = variant.get_raw(parameter)
        after_norm = variant.get_normalized(parameter)
        after_norm = after_norm + delta if after_norm <= 1.0 - delta else after_norm - delta
        variant.set_normalized(parameter, after_norm)
        base_variants[variant_name] = variant
        controls[variant_name] = {"field": parameter, "before_raw": before, "after_raw": variant.get_raw(parameter),
                                  "before_normalized": round(original.get_normalized(parameter), 6),
                                  "after_normalized": round(after_norm, 6)}

    lfo_frequency = f"{route['source']}_frequency"
    lfo = VitalPreset(original.to_dict(), original.schema)
    lfo_before = lfo.get_raw(lfo_frequency)
    lfo_norm_before = lfo.get_normalized(lfo_frequency)
    lfo_norm_after = lfo_norm_before + 0.25 if lfo_norm_before <= 0.75 else lfo_norm_before - 0.25
    lfo.set_normalized(lfo_frequency, lfo_norm_after)
    base_variants["lfo_frequency_shift"] = lfo
    controls["lfo_frequency_shift"] = {"field": lfo_frequency, "route_source": route["source"],
        "route_destination": route["destination"], "before_raw": lfo_before,
        "after_raw": lfo.get_raw(lfo_frequency), "before_normalized": round(lfo_norm_before, 6),
        "after_normalized": round(lfo_norm_after, 6)}

    seed_ids = ("source", "independent")
    variants = {(name, seed_id): _seeded_preset(preset, seed_index, VitalPreset)
                for name, preset in base_variants.items()
                for seed_index, seed_id in enumerate(seed_ids)}

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    render_root = output / "renders"
    render_root.mkdir(exist_ok=True)
    renderer = VitalRenderer.from_config(args.renderer_config, plugin_path=args.plugin_path)
    if renderer.renderer_id != FULL_CORPUS_RENDERER_ID or renderer.max_workers != 1:
        raise ValueError("controlled experiment renderer differs from the qualified single-worker Vital profile")

    render_rows, accepted = [], {}
    for (variant_name, seed_id), preset in variants.items():
        original_settings = original.to_dict()["settings"]
        variant_settings = preset.to_dict()["settings"]
        changed_fields = {key for key in set(original_settings) | set(variant_settings)
                          if original_settings.get(key) != variant_settings.get(key)}
        expected_fields = set() if variant_name == "original" else {controls[variant_name]["field"]}
        if seed_id != "source":
            expected_fields.add("random_values")
        if changed_fields != expected_fields:
            raise ValueError(f"controlled variant {variant_name!r}/{seed_id} changed unexpected settings: {sorted(changed_fields)}")
        baseline_codes = Counter(json.dumps(item, sort_keys=True) for item in original_validation_diagnostics)
        variant_codes = Counter(json.dumps(item.to_dict(), sort_keys=True) for item in preset.validate().diagnostics)
        if variant_codes - baseline_codes:
            raise ValueError(f"controlled variant {variant_name!r}/{seed_id} introduced new Vital schema diagnostics")
        for midi in chosen:
            performance = Performance.from_dict(midi["excerpt"])
            request = RenderRequest(preset=preset, performance=performance, sample_rate=44_100, channels=2,
                tail_seconds=2.0, renderer_id=renderer.renderer_id)
            from obruxo_data.render.batch import run_batch
            run_batch(renderer, [request], render_root, workers=1, resume=True)
            metadata_path, wav_path = render_root / f"{request.request_id}.json", render_root / f"{request.request_id}.wav"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            qa_ok, qa_status = _accepted(metadata)
            row = {"variant": variant_name, "seed_id": seed_id, "midi_id": midi["midi_id"], "request_id": request.request_id,
                   "audio_sha256": file_sha256(wav_path), "accepted": qa_ok, "qa_status": qa_status,
                   "qa_warnings": _qa_warnings(metadata),
                   "wav_path": wav_path, "metadata": metadata}
            render_rows.append(row)
            if qa_ok:
                accepted[(variant_name, midi["midi_id"], seed_id)] = row
    (output / "render_manifest.private.jsonl").write_text("".join(json.dumps({key: value for key, value in row.items()
        if key not in {"wav_path", "metadata"}}, sort_keys=True) + "\n" for row in render_rows), encoding="utf-8")
    if len(accepted) != len(variants) * len(chosen):
        excluded = [{"variant": row["variant"], "seed_id": row["seed_id"], "midi_id": row["midi_id"],
                     "qa_status": row["qa_status"]}
                    for row in render_rows if not row["accepted"]]
        raise RuntimeError(f"controlled matrix requires every planned render to pass Vital QA: {excluded}")

    audio_vectors = {key: _spectral_descriptor(row["wav_path"], np) for key, row in accepted.items()}
    audio_repeat_rows, audio_perturbation_rows = [], defaultdict(list)
    for midi in chosen:
        midi_id = midi["midi_id"]
        repeat_similarity = _cosine(audio_vectors[("original", midi_id, "source")],
                                    audio_vectors[("original", midi_id, "independent")], np)
        audio_repeat_rows.append({"midi_id": midi_id, "distance": 1.0 - repeat_similarity})
        for name in list(base_variants)[1:]:
            similarity = _cosine(audio_vectors[("original", midi_id, "source")],
                                 audio_vectors[(name, midi_id, "source")], np)
            distance = 1.0 - similarity
            audio_perturbation_rows[name].append({"midi_id": midi_id, "distance": distance,
                "exceeds_repeat_distance": float(distance > 1.0 - repeat_similarity)})
    audio_change_diagnostics = {
        "descriptor": "cosine distance over the same normalized 64-bin log-frequency power descriptor used for hard-negative selection",
        "identical_preset_independent_seed_distance": _bootstrap_midi_mean(audio_repeat_rows, "distance", seed=6),
        "perturbations": {name: {
            "distance": _bootstrap_midi_mean(rows, "distance", seed=7),
            "fraction_exceeding_same_midi_repeat_distance": _bootstrap_midi_mean(
                rows, "exceeds_repeat_distance", seed=8)}
            for name, rows in audio_perturbation_rows.items()}}

    lock = load_checkpoint_lock(args.lock)
    model_results, feature_records = {}, []
    for model_name in args.models:
        checkpoint, lock_entry, checkpoint_digest = resolve_checkpoint(model_name, args.checkpoint_root, lock)
        adapter = load_adapter(model_name, checkpoint, lock_entry, checkpoint_digest,
                               checkpoint_root=args.checkpoint_root, device=args.device)
        if args.device == "cuda":
            adapter.torch.cuda.reset_peak_memory_stats()
        elif args.device == "xpu" and hasattr(adapter.torch.xpu, "reset_peak_memory_stats"):
            adapter.torch.xpu.reset_peak_memory_stats()
        feature_dir = output / "features" / model_name
        feature_dir.mkdir(parents=True, exist_ok=True)
        started, peak_rss = time.perf_counter(), _rss_bytes()
        vectors_by_tap = defaultdict(dict)
        for key, render in sorted(accepted.items()):
            feature = cache_feature(adapter, render["wav_path"], request_id=render["request_id"],
                                    render_metadata=render["metadata"], output=feature_dir, source_precision="fp32")
            with np.load(feature_dir / feature["cache_path"], allow_pickle=False) as arrays:
                for tap in arrays.files:
                    vectors_by_tap[tap][key] = arrays[tap].astype(np.float32, copy=False)
            feature_records.append({"model": model_name, "variant": key[0], "midi_id": key[1],
                                    "seed_id": key[2], "feature_id": feature["feature_id"],
                                    "feature_taps": sorted(arrays.files),
                                    "reused": feature["reused"]})
            current_rss = _rss_bytes()
            if current_rss is not None:
                peak_rss = max(peak_rss or current_rss, current_rss)

        variant_names, midi_ids = list(base_variants), [row["midi_id"] for row in chosen]
        pool_results = {}
        for tap, vectors in vectors_by_tap.items():
            cross_metrics, same_midi_metrics, repeat_rows = [], [], []
            cross_by_variant, same_by_variant = defaultdict(list), defaultdict(list)
            perturbation_rows = defaultdict(list)
            for midi_index, midi_id in enumerate(midi_ids):
                gallery_midi = midi_ids[(midi_index + 1) % len(midi_ids)]
                repeat_similarity = _cosine(vectors[("original", midi_id, "source")],
                                            vectors[("original", midi_id, "independent")], np)
                repeat_rows.append({"midi_id": midi_id, "distance": 1.0 - repeat_similarity})
                for target in variant_names:
                    query_key, same_keys = _cross_seed_keys(target, midi_id, variant_names, "source", "independent")
                    query = vectors[query_key]
                    same_scores = [(candidate, _cosine(query, vectors[key], np))
                                   for candidate, key in zip(variant_names, same_keys, strict=True)]
                    same_rank = {**_rank_metrics(same_scores, target), "midi_id": midi_id}
                    same_midi_metrics.append(same_rank)
                    same_by_variant[target].append(same_rank)

                    _, cross_keys = _cross_seed_keys(target, gallery_midi, variant_names, "source", "independent")
                    cross_scores = [(candidate, _cosine(query, vectors[key], np))
                                    for candidate, key in zip(variant_names, cross_keys, strict=True)]
                    cross_rank = {**_rank_metrics(cross_scores, target), "midi_id": midi_id}
                    cross_metrics.append(cross_rank)
                    cross_by_variant[target].append(cross_rank)

                for variant_name in variant_names[1:]:
                    similarity = _cosine(vectors[("original", midi_id, "source")],
                                         vectors[(variant_name, midi_id, "source")], np)
                    distance = 1.0 - similarity
                    perturbation_rows[variant_name].append({
                        "midi_id": midi_id, "distance": distance,
                        "exceeds_repeat_distance": float(distance > 1.0 - repeat_similarity)})

            pool_results[tap] = {
                "cross_midi_variant_retrieval": {
                    "queries": len(cross_metrics), "gallery_candidates": len(variant_names),
                    "recall_at_1": _mean([item["recall_at_1"] for item in cross_metrics]),
                    "recall_at_5": _mean([item["recall_at_5"] for item in cross_metrics]),
                    "mrr": _mean([item["mrr"] for item in cross_metrics]),
                    "confidence_intervals_by_midi": {metric: _bootstrap_midi_mean(cross_metrics, metric)
                                                      for metric in ("recall_at_1", "recall_at_5", "mrr")}},
                "same_midi_independent_seed_control": {
                    "queries": len(same_midi_metrics),
                    "recall_at_1": _mean([item["recall_at_1"] for item in same_midi_metrics]),
                    "mrr": _mean([item["mrr"] for item in same_midi_metrics]),
                    "confidence_intervals_by_midi": {metric: _bootstrap_midi_mean(same_midi_metrics, metric, seed=1)
                                                      for metric in ("recall_at_1", "mrr")}},
                "identical_preset_repeat_distance": _bootstrap_midi_mean(repeat_rows, "distance", seed=2),
                "per_target_variant": {name: {
                    "cross_midi_retrieval": {
                        "queries": len(cross_by_variant[name]),
                        "recall_at_1": _mean([item["recall_at_1"] for item in cross_by_variant[name]]),
                        "recall_at_5": _mean([item["recall_at_5"] for item in cross_by_variant[name]]),
                        "mrr": _mean([item["mrr"] for item in cross_by_variant[name]]),
                        "confidence_intervals_by_midi": {
                            metric: _bootstrap_midi_mean(cross_by_variant[name], metric, seed=3)
                            for metric in ("recall_at_1", "recall_at_5", "mrr")}},
                    "same_midi_independent_seed_control": {
                        "recall_at_1": _mean([item["recall_at_1"] for item in same_by_variant[name]]),
                        "mrr": _mean([item["mrr"] for item in same_by_variant[name]])},
                    **({"perturbation_distance": _bootstrap_midi_mean(perturbation_rows[name], "distance", seed=4),
                        "perturbation_exceeds_repeat_distance_fraction": _bootstrap_midi_mean(
                            perturbation_rows[name], "exceeds_repeat_distance", seed=5)} if name != "original" else {})}
                    for name in variant_names}}

        elapsed = time.perf_counter() - started
        peak_accelerator = None
        if args.device == "cuda":
            peak_accelerator = int(adapter.torch.cuda.max_memory_allocated())
        elif args.device == "xpu" and hasattr(adapter.torch.xpu, "max_memory_allocated"):
            peak_accelerator = int(adapter.torch.xpu.max_memory_allocated())
        model_results[model_name] = {"checkpoint_sha256": checkpoint_digest, "source_revision": lock_entry["revision"],
            "loader_revision": lock_entry["loader_revision"], "device": args.device, "frozen": True,
            "feature_count": len(accepted), "elapsed_seconds": round(elapsed, 3),
            "peak_process_rss_observed_bytes": peak_rss, "peak_accelerator_memory_bytes": peak_accelerator,
            "pool_results": pool_results}

    (output / "feature_manifest.private.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n"
        for row in feature_records), encoding="utf-8")
    result = {"version": 1, "experiment": "controlled_filter_envelope_lfo_perturbation_v2",
        "source_snapshot_id": summary["source_snapshot_id"], "source_split": summary["source_split"],
        "performance_source_split": summary.get("performance_source_split", summary["source_split"]),
        "reserved_split_access": "none",
        "renderer_id": renderer.renderer_id, "preset_state_id": selected_preset["state_id"],
        "preset_source_sha256": selected_preset["raw_sha256"], "midi_ids": [row["midi_id"] for row in chosen],
        "inherited_source_validation_diagnostics": {"count": len(original_validation_diagnostics),
            "codes": dict(Counter(item["code"] for item in original_validation_diagnostics)),
            "new_diagnostics_after_perturbation": 0},
        "midi_strata": [{"midi_id": row["midi_id"], "max_polyphony": row["stats"]["max_polyphony"],
                         "pitch_median": row["stats"]["pitch_median"], "duration_seconds": row["stats"]["duration_seconds"],
                         "note_duration_median_seconds": row["stats"]["note_duration_median_seconds"],
                         "velocity_min": row["stats"]["velocity_min"], "velocity_max": row["stats"]["velocity_max"],
                         "articulation": row["stats"]["articulation"]}
                        for row in chosen],
        "variants": list(base_variants), "controlled_fields": controls,
        "independent_render_seeds": list(seed_ids), "same_midi_query_gallery_overlap": False,
        "audio_change_diagnostics": audio_change_diagnostics,
        "render_requests": len(render_rows), "accepted_renders": len(accepted),
        "models": model_results,
        "confidence_intervals": f"percentile bootstrap over normalized MIDI identity clusters ({len(chosen)} clusters; 2,000 replicates)"}
    (output / "controlled_summary.private.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    args = _parser().parse_args()
    for name in ("dataset", "renderer_config", "checkpoint_root", "lock"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    if args.plugin_path is not None:
        args.plugin_path = args.plugin_path.resolve(strict=True)
    args.output = args.output.resolve()
    print(json.dumps(run(args), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

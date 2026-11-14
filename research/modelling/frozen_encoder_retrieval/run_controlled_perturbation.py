from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
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
    parser.add_argument("--models", nargs="+", choices=("clap_music", "dac_44khz"), default=("clap_music", "dac_44khz"))
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

    mono = [row for row in performances if row["stats"]["max_polyphony"] == 1]
    poly = [row for row in performances if row["stats"]["max_polyphony"] > 1]
    chosen = []
    for candidates in (
            sorted(mono, key=lambda item: (item["stats"]["pitch_median"], item["stats"]["note_duration_median_seconds"], item["midi_id"])),
            sorted(poly, key=lambda item: (-item["stats"]["pitch_median"], -item["stats"]["note_duration_median_seconds"], item["midi_id"])),
            sorted(mono, key=lambda item: (-item["stats"]["note_duration_median_seconds"], -item["stats"]["pitch_median"], item["midi_id"]))):
        candidate = next((row for row in candidates if row["midi_id"] not in {item["midi_id"] for item in chosen}), None)
        if candidate is not None:
            chosen.append(candidate)
    if len(chosen) < 3:
        raise ValueError("controlled experiment needs three different real validation MIDI excerpts")
    if not any(row["stats"]["velocity_min"] < row["stats"]["velocity_max"] for row in chosen):
        variable_velocity = [row for row in performances if row["stats"]["velocity_min"] < row["stats"]["velocity_max"]]
        if variable_velocity:
            chosen.append(max(variable_velocity, key=lambda row: (
                row["stats"]["velocity_max"] - row["stats"]["velocity_min"],
                row["stats"]["max_polyphony"], row["stats"]["pitch_median"], row["midi_id"])))

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
        if any(route.get("source") for route in candidate.to_dict()["settings"]["modulations"]):
            eligible_presets.append((len(candidate.validate().diagnostics), row["state_id"], row, candidate))
    if not eligible_presets:
        raise ValueError("no render-qualified validation preset with a live modulation route has accepted renders for the selected MIDI")
    _, _, selected_preset, original = min(eligible_presets, key=lambda item: (item[0], item[1]))
    original_validation_diagnostics = [item.to_dict() for item in original.validate().diagnostics]

    variants = {"original": VitalPreset(original.to_dict(), original.schema)}
    controls = {}
    for variant_name, parameter, delta in (("filter_cutoff_shift", "filter_1_cutoff", 0.20),
                                           ("amp_attack_shift", "env_1_attack", 0.20)):
        variant = VitalPreset(original.to_dict(), original.schema)
        before = variant.get_raw(parameter)
        after_norm = variant.get_normalized(parameter)
        after_norm = after_norm + delta if after_norm <= 1.0 - delta else after_norm - delta
        variant.set_normalized(parameter, after_norm)
        variants[variant_name] = variant
        controls[variant_name] = {"field": parameter, "before_raw": before, "after_raw": variant.get_raw(parameter),
                                  "before_normalized": round(original.get_normalized(parameter), 6),
                                  "after_normalized": round(after_norm, 6)}

    live_routes = [(index + 1, route) for index, route in enumerate(original.to_dict()["settings"]["modulations"])
                   if route.get("source")]
    slot, route = live_routes[0]
    amount_name = f"modulation_{slot}_amount"
    modulation = VitalPreset(original.to_dict(), original.schema)
    amount_before = modulation.get_raw(amount_name)
    amount_norm_before = modulation.get_normalized(amount_name)
    amount_norm_after = amount_norm_before + 0.25 if amount_norm_before <= 0.75 else amount_norm_before - 0.25
    modulation.set_normalized(amount_name, amount_norm_after)
    variants["modulation_amount_shift"] = modulation
    controls["modulation_amount_shift"] = {"field": amount_name, "route_source": route["source"],
        "route_destination": route["destination"], "before_raw": amount_before,
        "after_raw": modulation.get_raw(amount_name), "before_normalized": round(amount_norm_before, 6),
        "after_normalized": round(amount_norm_after, 6)}

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    render_root = output / "renders"
    render_root.mkdir(exist_ok=True)
    renderer = VitalRenderer.from_config(args.renderer_config, plugin_path=args.plugin_path)
    if renderer.renderer_id != FULL_CORPUS_RENDERER_ID or renderer.max_workers != 1:
        raise ValueError("controlled experiment renderer differs from the qualified single-worker Vital profile")

    render_rows, accepted = [], {}
    for variant_name, preset in variants.items():
        original_settings = original.to_dict()["settings"]
        variant_settings = preset.to_dict()["settings"]
        changed_fields = {key for key in set(original_settings) | set(variant_settings)
                          if original_settings.get(key) != variant_settings.get(key)}
        expected_fields = set() if variant_name == "original" else {controls[variant_name]["field"]}
        if changed_fields != expected_fields:
            raise ValueError(f"controlled variant {variant_name!r} changed unexpected settings: {sorted(changed_fields)}")
        baseline_codes = Counter(json.dumps(item, sort_keys=True) for item in original_validation_diagnostics)
        variant_codes = Counter(json.dumps(item.to_dict(), sort_keys=True) for item in preset.validate().diagnostics)
        if variant_codes - baseline_codes:
            raise ValueError(f"controlled variant {variant_name!r} introduced new Vital schema diagnostics")
        for midi in chosen:
            performance = Performance.from_dict(midi["excerpt"])
            request = RenderRequest(preset=preset, performance=performance, sample_rate=44_100, channels=2,
                tail_seconds=2.0, renderer_id=renderer.renderer_id)
            from obruxo_data.render.batch import run_batch
            run_batch(renderer, [request], render_root, workers=1, resume=True)
            metadata_path, wav_path = render_root / f"{request.request_id}.json", render_root / f"{request.request_id}.wav"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            qa_ok, qa_status = _accepted(metadata)
            row = {"variant": variant_name, "midi_id": midi["midi_id"], "request_id": request.request_id,
                   "audio_sha256": file_sha256(wav_path), "accepted": qa_ok, "qa_status": qa_status,
                   "qa_warnings": _qa_warnings(metadata),
                   "wav_path": wav_path, "metadata": metadata}
            render_rows.append(row)
            if qa_ok:
                accepted[(variant_name, midi["midi_id"])] = row
    (output / "render_manifest.private.jsonl").write_text("".join(json.dumps({key: value for key, value in row.items()
        if key not in {"wav_path", "metadata"}}, sort_keys=True) + "\n" for row in render_rows), encoding="utf-8")
    if len(accepted) != len(variants) * len(chosen):
        excluded = [{"variant": row["variant"], "midi_id": row["midi_id"], "qa_status": row["qa_status"]}
                    for row in render_rows if not row["accepted"]]
        raise RuntimeError(f"controlled matrix requires every planned render to pass Vital QA: {excluded}")

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
                                    "feature_id": feature["feature_id"], "feature_taps": sorted(arrays.files),
                                    "reused": feature["reused"]})
            current_rss = _rss_bytes()
            if current_rss is not None:
                peak_rss = max(peak_rss or current_rss, current_rss)

        variant_names, midi_ids = list(variants), [row["midi_id"] for row in chosen]
        pool_results = {}
        for tap, vectors in vectors_by_tap.items():
            cross_metrics, same_midi_metrics = [], []
            same_variant_cross_midi, changed_variant_same_midi = [], []
            baseline_contrast = defaultdict(list)
            cross_by_variant, same_by_variant = defaultdict(list), defaultdict(list)
            cross_cosine_by_variant, baseline_cosine_by_variant = defaultdict(list), defaultdict(list)
            for midi_index, midi_id in enumerate(midi_ids):
                gallery_midi = midi_ids[(midi_index + 1) % len(midi_ids)]
                for target in variant_names:
                    query = vectors[(target, midi_id)]
                    gallery_scores = [(candidate, _cosine(query, vectors[(candidate, gallery_midi)], np))
                                      for candidate in variant_names]
                    cross_rank = _rank_metrics(gallery_scores, target)
                    cross_metrics.append(cross_rank)
                    cross_by_variant[target].append(cross_rank)
                    cross_cosine_by_variant[target].append(dict(gallery_scores)[target])
                    same_scores = [(candidate, _cosine(query, vectors[(candidate, midi_id)], np)) for candidate in variant_names]
                    same_rank = _rank_metrics(same_scores, target)
                    same_midi_metrics.append(same_rank)
                    same_by_variant[target].append(same_rank)
                for variant_name in variant_names:
                    same_variant_cross_midi.append(_cosine(vectors[(variant_name, midi_id)],
                                                            vectors[(variant_name, gallery_midi)], np))
                for left_index, left in enumerate(variant_names):
                    for right in variant_names[left_index + 1:]:
                        changed_variant_same_midi.append(_cosine(vectors[(left, midi_id)], vectors[(right, midi_id)], np))
                for variant_name in variant_names[1:]:
                    similarity = _cosine(vectors[("original", midi_id)], vectors[(variant_name, midi_id)], np)
                    baseline_contrast[variant_name].append(similarity)
                    baseline_cosine_by_variant[variant_name].append(similarity)

            pool_results[tap] = {"cross_midi_variant_retrieval": {"queries": len(cross_metrics),
                    "recall_at_1": _mean([item["recall_at_1"] for item in cross_metrics]),
                    "recall_at_5": _mean([item["recall_at_5"] for item in cross_metrics]),
                    "mrr": _mean([item["mrr"] for item in cross_metrics]), "gallery_candidates": len(variant_names)},
                "same_midi_variant_control": {"queries": len(same_midi_metrics),
                    "recall_at_1": _mean([item["recall_at_1"] for item in same_midi_metrics]),
                    "mrr": _mean([item["mrr"] for item in same_midi_metrics])},
                "same_variant_different_midi_mean_cosine": _mean(same_variant_cross_midi),
                "different_variant_same_midi_mean_cosine": _mean(changed_variant_same_midi),
                "original_vs_perturbation_same_midi_mean_cosine": {name: _mean(values) for name, values in baseline_contrast.items()},
                "per_target_variant": {name: {"cross_midi_retrieval": {
                        "queries": len(cross_by_variant[name]),
                        "recall_at_1": _mean([item["recall_at_1"] for item in cross_by_variant[name]]),
                        "recall_at_5": _mean([item["recall_at_5"] for item in cross_by_variant[name]]),
                        "mrr": _mean([item["mrr"] for item in cross_by_variant[name]]),
                        "target_mean_cosine": _mean(cross_cosine_by_variant[name])},
                    "same_midi_control": {"recall_at_1": _mean([item["recall_at_1"] for item in same_by_variant[name]]),
                                          "mrr": _mean([item["mrr"] for item in same_by_variant[name]])},
                    "same_variant_different_midi_mean_cosine": _mean([
                        _cosine(vectors[(name, midi)], vectors[(name, midi_ids[(index + 1) % len(midi_ids)])], np)
                        for index, midi in enumerate(midi_ids)]),
                    "original_same_midi_cosine": 1.0 if name == "original" else _mean(baseline_cosine_by_variant[name])}
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
    result = {"version": 1, "experiment": "controlled_filter_envelope_modulation_perturbation_v1",
        "source_snapshot_id": summary["source_snapshot_id"], "source_split": "validation", "reserved_split_access": "none",
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
        "variants": list(variants), "controlled_fields": controls,
        "render_requests": len(render_rows), "accepted_renders": len(accepted),
        "models": model_results, "confidence_intervals": "not estimated; small controlled diagnostic set"}
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

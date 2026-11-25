from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import time

from .core import canonical_json
from .encoders import (CacheIdentityDriftError, cache_feature, file_sha256, load_adapter,
                       load_checkpoint_lock, resolve_checkpoint, resolve_source_checkout)
from .rendering import accepted_receipts, _rss_bytes
from .dataset import validate_dataset_manifests


def extract_dataset(dataset_dir: Path, model_name: str, checkpoint_root: Path, lock_path: Path, *, device: str = "cpu",
                    active_rms_target: float | None = None, progress=None) -> dict:
    dataset_dir = dataset_dir.resolve(strict=True)
    dataset_summary = json.loads((dataset_dir / "dataset_summary.private.json").read_text(encoding="utf-8"))
    validate_dataset_manifests(dataset_dir, dataset_summary)
    lock = load_checkpoint_lock(lock_path)
    checkpoint, entry, checkpoint_digest = resolve_checkpoint(model_name, checkpoint_root, lock)
    source_root = resolve_source_checkout(checkpoint_root, entry)
    adapter = load_adapter(model_name, checkpoint, entry, checkpoint_digest, checkpoint_root=checkpoint_root,
                           device=device, source_root=source_root)
    plan = {}
    with (dataset_dir / "render_plan.private.jsonl").open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                plan[row["request_id"]] = row
    receipts = accepted_receipts(dataset_dir)
    if active_rms_target is not None:
        evaluation = json.loads((dataset_dir / "evaluation.private.json").read_text(encoding="utf-8"))
        metric_path = dataset_dir / "episode_metrics.private.jsonl"
        anchor_arm = sorted(evaluation["available_arms"])[0]
        pairs = set()
        for line in metric_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("arm") == anchor_arm and row.get("condition") == "cross_midi_common_gallery":
                pairs.add((row["target_state_id"], row["query_midi_id"]))
        preset_ids = {state for state, _ in pairs}
        midi_ids = {midi for _, midi in pairs}
        if len(pairs) != 6_450 or len(preset_ids) != 129 or len(midi_ids) != 50:
            raise ValueError("active-RMS extraction requires the existing complete 129 by 50 common cohort")
        request_by_pair = {(row["state_id"], row["midi_id"]): row["request_id"] for row in plan.values()}
        selected_requests = {request_by_pair[pair] for pair in pairs}
        if not selected_requests.issubset(receipts):
            raise ValueError("active-RMS cohort includes a WAV without an accepted render receipt")
        receipts = {request_id: receipt for request_id, receipt in receipts.items() if request_id in selected_requests}
    output_root = dataset_dir / ("features_active_rms" if active_rms_target is not None else "features")
    output_dir = output_root / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    if device == "cuda":
        adapter.torch.cuda.reset_peak_memory_stats()
    elif device == "xpu" and hasattr(adapter.torch.xpu, "reset_peak_memory_stats"):
        adapter.torch.xpu.reset_peak_memory_stats()
    started = time.perf_counter()
    peak_rss = _rss_bytes()
    records = {}
    failure_rows = []
    counts = Counter()
    for completed, (request_id, receipt) in enumerate(sorted(receipts.items()), start=1):
        plan_row = plan.get(request_id)
        if plan_row is None:
            raise ValueError("render receipt is not present in the frozen render plan")
        wav_path = dataset_dir / receipt["wav_relative_path"]
        metadata_path = dataset_dir / receipt["metadata_relative_path"]
        if file_sha256(wav_path) != receipt["audio_sha256"]:
            raise ValueError("render audio changed after QA; feature cache is rejected")
        render_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if render_metadata.get("provenance", {}).get("renderer_id") != plan_row["renderer_id"]:
            raise ValueError("render metadata and plan renderer profiles differ")
        try:
            feature = cache_feature(adapter, wav_path, request_id=request_id, render_metadata=render_metadata,
                                    output=output_dir, source_precision="fp32",
                                    active_rms_target=active_rms_target)
        except CacheIdentityDriftError:
            raise
        except Exception as error:
            failure_rows.append({"request_id": request_id, "state_id": plan_row["state_id"],
                                 "failure_type": type(error).__name__, "failure_message": str(error)[:240]})
            counts["feature_failures"] += 1
            if progress and (completed % 32 == 0 or completed == len(receipts)):
                progress(completed, len(failure_rows))
            continue
        feature_metadata = json.loads((output_dir / f"{feature['feature_id']}.json").read_text(encoding="utf-8"))
        records[request_id] = {"request_id": request_id, "state_id": plan_row["state_id"],
                               "family_id": plan_row["family_id"], "midi_id": plan_row["midi_id"],
                               "excerpt_id": plan_row["excerpt_id"], "roles": plan_row["roles"],
                               "split": plan_row["split"], "feature_id": feature["feature_id"],
                               "cache_path": feature["cache_path"], "audio_sha256": feature["audio_sha256"],
                               "dimensions": feature_metadata["dimensions"], "source_kind": plan_row["source_kind"],
                               **({"normalization": feature_metadata["normalization"]}
                                  if active_rms_target is not None else {})}
        counts["features_reused" if feature["reused"] else "features_created"] += 1
        if progress and (completed % 32 == 0 or completed == len(receipts)):
            progress(completed, len(failure_rows))
        if active_rms_target is not None and (completed % 500 == 0 or completed == len(receipts)):
            print(f"{model_name}: completed {completed:,}/{len(receipts):,} active-RMS features.", flush=True)
        current_rss = _rss_bytes()
        if current_rss is not None:
            peak_rss = max(peak_rss or current_rss, current_rss)
    with (output_dir / "feature_manifest.private.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in records.values():
            stream.write(canonical_json(row) + "\n")
    with (output_dir / "feature_failures.private.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in failure_rows:
            stream.write(canonical_json(row) + "\n")
    elapsed = time.perf_counter() - started
    torch = adapter.torch
    device_peak = None
    if device == "cuda":
        device_peak = int(torch.cuda.max_memory_allocated())
    elif device == "xpu" and hasattr(torch.xpu, "max_memory_allocated"):
        device_peak = int(torch.xpu.max_memory_allocated())
    summary = {"version": 1, "model": model_name, "checkpoint_sha256": checkpoint_digest,
               "source_revision": entry["revision"], "loader_revision": entry["loader_revision"],
               "device": device, "precision": "float32", "frozen": True, "eval_mode": True,
               "input_render_count": len(receipts), "feature_count": len(records), "failure_count": len(failure_rows), "dimensions_by_variant":
               (next(iter(records.values()))["dimensions"] if records else {}), **dict(counts),
               "elapsed_seconds": round(elapsed, 3), "peak_process_rss_observed_bytes": peak_rss,
               "peak_accelerator_memory_bytes": device_peak,
               "preprocessing": ("stereo active-RMS normalization before mono downmix and scipy polyphase resample"
                                 if active_rms_target is not None else
                                 "stereo mean to mono; scipy polyphase resample; no loudness normalization"),
               "pooling": "official native clip embedding" if len(records) and model_name in {"clap_music", "muq_mulan", "cir_ast"}
               else "valid sequence frames: masked mean and 8 ordered bins of mean plus standard deviation"}
    if model_name == "vital_jepa_v7":
        from .vital_jepa_v7 import POOLING, PREPROCESSING

        summary.update(preprocessing=PREPROCESSING, pooling=POOLING)
    if active_rms_target is not None:
        from .normalization import ACTIVE_FLOOR_DB, NORMALIZATION_VERSION, WINDOW_SECONDS

        summary["normalization"] = {"version": NORMALIZATION_VERSION, "window_seconds": WINDOW_SECONDS,
                                    "active_floor_db": ACTIVE_FLOOR_DB,
                                    "target_active_rms": float(active_rms_target),
                                    "application": "one scalar per original stereo WAV before encoder preprocessing",
                                    "source_audio_unchanged": True}
    (output_dir / "extraction.latest.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return summary

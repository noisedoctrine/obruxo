from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import time

from .core import canonical_json
from .encoders import (CacheIdentityDriftError, cache_feature, file_sha256, load_adapter,
                       load_checkpoint_lock, resolve_checkpoint)
from .rendering import accepted_receipts, _rss_bytes
from .dataset import validate_dataset_manifests


def extract_dataset(dataset_dir: Path, model_name: str, checkpoint_root: Path, lock_path: Path, *, device: str = "cpu") -> dict:
    dataset_dir = dataset_dir.resolve(strict=True)
    dataset_summary = json.loads((dataset_dir / "dataset_summary.private.json").read_text(encoding="utf-8"))
    validate_dataset_manifests(dataset_dir, dataset_summary)
    lock = load_checkpoint_lock(lock_path)
    checkpoint, entry, checkpoint_digest = resolve_checkpoint(model_name, checkpoint_root, lock)
    source_root = None
    if entry.get("source_root"):
        source_root = (checkpoint_root.resolve(strict=True) / entry["source_root"]).resolve(strict=True)
    adapter = load_adapter(model_name, checkpoint, entry, checkpoint_digest, checkpoint_root=checkpoint_root,
                           device=device, source_root=source_root)
    plan = {}
    with (dataset_dir / "render_plan.private.jsonl").open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                plan[row["request_id"]] = row
    receipts = accepted_receipts(dataset_dir)
    output_dir = dataset_dir / "features" / model_name
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
    for request_id, receipt in sorted(receipts.items()):
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
                                    output=output_dir, source_precision="fp32")
        except CacheIdentityDriftError:
            raise
        except Exception as error:
            failure_rows.append({"request_id": request_id, "state_id": plan_row["state_id"],
                                 "failure_type": type(error).__name__})
            counts["feature_failures"] += 1
            continue
        feature_metadata = json.loads((output_dir / f"{feature['feature_id']}.json").read_text(encoding="utf-8"))
        records[request_id] = {"request_id": request_id, "state_id": plan_row["state_id"],
                               "family_id": plan_row["family_id"], "midi_id": plan_row["midi_id"],
                               "excerpt_id": plan_row["excerpt_id"], "roles": plan_row["roles"],
                               "split": plan_row["split"], "feature_id": feature["feature_id"],
                               "cache_path": feature["cache_path"], "audio_sha256": feature["audio_sha256"],
                               "dimensions": feature_metadata["dimensions"], "source_kind": plan_row["source_kind"]}
        counts["features_reused" if feature["reused"] else "features_created"] += 1
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
               "preprocessing": "stereo mean to mono; scipy polyphase resample; no loudness normalization",
               "pooling": "official native clip embedding" if len(records) and model_name in {"clap_music", "muq_mulan", "cir_ast"}
               else "valid sequence frames: masked mean and 8 ordered bins of mean plus standard deviation"}
    (output_dir / "extraction.latest.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return summary

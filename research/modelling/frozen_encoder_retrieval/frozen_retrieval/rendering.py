from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import queue
import threading
import time

from .core import canonical_json, digest
from .dataset import _safe_source, validate_dataset_manifests


RENDER_POLICY_ID = "vital_original_preset_cross_midi_excerpt_44100_stereo_2s-tail_v2"
QA_POLICY_ID = "finite_silent_reject_clipping_tail_warn_v2"


def render_plan_identity(row: dict) -> str:
    return digest({key: value for key, value in row.items() if key != "request"} | {"render_policy_id": RENDER_POLICY_ID})


def _plan_batches(plan_path: Path, performances_path: Path, source_root: Path, batch_size: int):
    from obruxo_data.midi import Performance
    from obruxo_data.render.base import RenderRequest
    from obruxo_training.identity import PresetIdentity

    performance_by_id = {}
    for line in performances_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            performance_by_id[item["midi_id"]] = item
    source_root = source_root.resolve(strict=True)
    raw_by_state, state_paths, verified_midi = {}, {}, {}

    with plan_path.open("r", encoding="utf-8") as stream:
        rows, requests = [], []
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            state_id = row["state_id"]
            if state_id not in raw_by_state:
                source = _safe_source(source_root, row["preset_relative_path"])
                raw = source.read_bytes()
                identity = PresetIdentity.from_bytes(raw)
                if (identity.raw, identity.state) != (row["preset_sha256"], state_id):
                    raise ValueError("original preset source bytes or identity changed after manifest freeze")
                raw_by_state[state_id] = raw.decode("utf-8-sig")
                state_paths[state_id] = (row["preset_relative_path"], row["preset_sha256"])
            elif state_paths[state_id] != (row["preset_relative_path"], row["preset_sha256"]):
                raise ValueError("one frozen preset state maps to multiple source identities")
            if row["source_kind"] == "newly_rendered_original_preset" and row["midi_id"] not in verified_midi:
                midi_source = _safe_source(source_root, row["midi_relative_path"])
                actual_midi_sha = hashlib.sha256(midi_source.read_bytes()).hexdigest()
                if actual_midi_sha != row["midi_sha256"]:
                    raise ValueError("original MIDI source bytes changed after manifest freeze")
                verified_midi[row["midi_id"]] = (row["midi_relative_path"], row["midi_sha256"])
            midi = performance_by_id.get(row["midi_id"])
            if midi is None or midi["excerpt_id"] != row["excerpt_id"]:
                raise ValueError("render request refers to a missing or changed frozen MIDI excerpt")
            if row["split"] not in {"train", "validation"}:
                raise ValueError("render plan attempts to consume the reserved test split")
            performance = Performance.from_dict(midi["excerpt"])
            request = RenderRequest(preset_json=raw_by_state[state_id], performance=performance,
                                    sample_rate=int(row["sample_rate"]), channels=int(row["channels"]),
                                    tail_seconds=float(row["tail_seconds"]), renderer_id=row["renderer_id"])
            if request.request_id != row["request_id"]:
                raise ValueError("reconstructed render request identity differs from the frozen request ID")
            rows.append(row)
            requests.append(request)
            if len(rows) == batch_size:
                yield rows, requests
                rows, requests = [], []
        if rows:
            yield rows, requests


def _rss_bytes() -> int | None:
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except ImportError:
        return None


def _qa_warnings(metadata: dict) -> list[str]:
    qa = metadata.get("qa", {})
    warnings = []
    if int(qa.get("clipping_count", 0)) > 0:
        warnings.append("amplitude_above_unit_range")
    if float(qa.get("tail_rms", 0.0)) > float(qa.get("tail_rms_warning", float("inf"))):
        warnings.append("tail_rms_above_warning")
    return warnings


def _accepted(metadata: dict) -> tuple[bool, str]:
    qa = metadata.get("qa", {})
    if not qa.get("finite") or qa.get("actual_frames") != qa.get("expected_frames"):
        return False, "render_qa_shape_or_finite_failure"
    if qa.get("actual_channels") != qa.get("expected_channels"):
        return False, "render_qa_channel_failure"
    if float(qa.get("rms", 0.0)) <= float(qa.get("silence_threshold", 1e-6)):
        return False, "render_qa_silence"
    warnings = _qa_warnings(metadata)
    if warnings:
        return True, "accepted_with_" + "_and_".join(warnings)
    return True, "accepted"


def render_dataset(dataset_dir: Path, renderer_config: Path, *, plugin_path: Path | None = None,
                   batch_size: int = 32, queue_depth: int = 2) -> dict:
    """Run the durable manifest through a bounded producer/serial Vital consumer."""
    from obruxo_data.render.batch import run_batch, _is_complete
    from obruxo_data.render.base import FULL_CORPUS_RENDERER_ID
    from obruxo_data.render.vita import VitalRenderer

    if batch_size < 1 or queue_depth < 1:
        raise ValueError("batch_size and queue_depth must be positive")
    dataset_dir = dataset_dir.resolve(strict=True)
    summary = json.loads((dataset_dir / "dataset_summary.private.json").read_text(encoding="utf-8"))
    validate_dataset_manifests(dataset_dir, summary)
    if summary.get("renderer_id") != FULL_CORPUS_RENDERER_ID:
        raise ValueError("benchmark render manifest does not use the corpus-qualified Vital profile")
    plan_path = dataset_dir / "render_plan.private.jsonl"
    if not plan_path.is_file():
        raise FileNotFoundError(plan_path)
    renderer = VitalRenderer.from_config(renderer_config, plugin_path=plugin_path)
    if renderer.renderer_id != summary["renderer_id"] or renderer.max_workers != 1:
        raise ValueError("renderer identity/worker count differs from the frozen manifest")
    render_root = dataset_dir / "renders" / RENDER_POLICY_ID
    render_root.mkdir(parents=True, exist_ok=True)
    journal = dataset_dir / "render_receipts.private.jsonl"
    previous = {}
    if journal.is_file():
        for line in journal.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                previous[row["request_id"]] = row
    work: queue.Queue = queue.Queue(maxsize=queue_depth)
    timings, worker_error = [], []

    def produce():
        try:
            for batch in _plan_batches(plan_path, dataset_dir / "performances.private.jsonl",
                                       Path(summary["source_root"]), batch_size):
                work.put(batch)
        except BaseException as error:
            worker_error.append(error)
        finally:
            work.put(None)

    producer = threading.Thread(target=produce, name="obruxo-render-plan-reader", daemon=True)
    producer.start()
    counts = Counter()
    peak_rss = _rss_bytes()
    current_rendered = current_reused = 0
    with journal.open("a", encoding="utf-8", newline="\n") as receipts:
        while True:
            item = work.get()
            if item is None:
                break
            rows, requests = item
            for row in rows:
                plan_identity = render_plan_identity(row)
                old = previous.get(row["request_id"])
                if old and old["plan_identity"] != plan_identity:
                    raise ValueError("render plan identity drift for an existing request; stale audio is not reused")
            for row, request in zip(rows, requests, strict=True):
                metadata_path = render_root / f"{request.request_id}.json"
                wav_path = render_root / f"{request.request_id}.wav"
                reused = _is_complete(wav_path, metadata_path, request.request_id)
                started = time.perf_counter()
                try:
                    run_batch(renderer, [request], render_root, workers=1, resume=True)
                except Exception as error:
                    timings.append(time.perf_counter() - started)
                    plan_identity = render_plan_identity(row)
                    failure = {"request_id": request.request_id, "plan_identity": plan_identity,
                               "render_policy_id": RENDER_POLICY_ID, "qa_policy_id": QA_POLICY_ID, "accepted": False,
                               "source_kind": row["source_kind"],
                               "qa_status": f"render_exception:{type(error).__name__}",
                               "error_type": type(error).__name__}
                    receipts.write(canonical_json(failure) + "\n")
                    receipts.flush()
                    previous[request.request_id] = failure
                    counts[failure["qa_status"]] += 1
                    peak_rss = max(filter(None, (peak_rss, _rss_bytes())), default=peak_rss)
                    continue
                timings.append(time.perf_counter() - started)
                current_reused += int(reused)
                current_rendered += int(not reused)
                peak_rss = max(filter(None, (peak_rss, _rss_bytes())), default=peak_rss)
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata.get("provenance", {}).get("request_id") != request.request_id:
                    raise ValueError("render provenance request identity changed")
                if metadata.get("provenance", {}).get("renderer_id") != summary["renderer_id"]:
                    raise ValueError("render provenance profile differs from the frozen renderer identity")
                accepted, qa_status = _accepted(metadata)
                receipt = {"request_id": request.request_id, "plan_identity": render_plan_identity(row),
                    "render_policy_id": RENDER_POLICY_ID, "qa_policy_id": QA_POLICY_ID,
                    "audio_sha256": hashlib.sha256(wav_path.read_bytes()).hexdigest(),
                    "source_kind": row["source_kind"],
                    "duration_seconds": metadata["qa"]["actual_frames"] / metadata["sample_rate"],
                    "renderer_id": metadata["provenance"]["renderer_id"], "qa_status": qa_status,
                    "accepted": accepted, "qa_warnings": _qa_warnings(metadata), "reused": reused,
                    "qa_metrics": {key: metadata["qa"].get(key) for key in
                                   ("clipping_count", "peak", "rms", "tail_rms", "tail_rms_warning")},
                    "wav_relative_path": wav_path.relative_to(dataset_dir).as_posix(),
                    "metadata_relative_path": metadata_path.relative_to(dataset_dir).as_posix()}
                receipts.write(canonical_json(receipt) + "\n")
                receipts.flush()
                previous[request.request_id] = receipt
                counts[qa_status] += 1
        if worker_error:
            raise RuntimeError("render manifest producer failed") from worker_error[0]
    producer.join()
    latest = {"render_policy_id": RENDER_POLICY_ID, "renderer_id": renderer.renderer_id,
              "qa_policy_id": QA_POLICY_ID,
              "requested": summary["render_requests"], "accepted": sum(1 for row in previous.values() if row.get("accepted")),
              "excluded": dict(sorted(Counter(row["qa_status"] for row in previous.values() if not row.get("accepted")).items())),
              "qa_warning_counts": dict(sorted(Counter(warning for row in previous.values() if row.get("accepted")
                                                       for warning in row.get("qa_warnings", [])).items())),
              "runs_rendered_this_run": current_rendered, "cache_reused_this_run": current_reused,
              "wall_seconds_this_run": round(sum(timings), 3), "request_count_this_run": len(timings),
              "batch_size": batch_size, "bounded_queue_depth": queue_depth,
              "renderer_workers": 1, "device": "CPU, verified Vital/DawDreamer", "process_rss_peak_bytes": peak_rss}
    (dataset_dir / "render_run.latest.json").write_text(json.dumps(latest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return latest


def accepted_receipts(dataset_dir: Path) -> dict[str, dict]:
    journal = dataset_dir / "render_receipts.private.jsonl"
    rows = {}
    if not journal.is_file():
        return rows
    for line in journal.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["request_id"]] = row
    stale = [row for row in rows.values() if row.get("qa_policy_id") != QA_POLICY_ID]
    if stale:
        raise ValueError("render receipts use a stale QA policy; rerun the resumable render stage to refresh them")
    return {key: value for key, value in rows.items() if value.get("accepted")}

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import threading
import time

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[2]
for directory in (PACKAGE, REPO / "research/data_generation", REPO / "research/training",
                  REPO / "research/modelling/basic_pitch"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from frozen_retrieval.core import MODEL_REGISTRY, canonical_json
from frozen_retrieval.encoders import (file_sha256, load_adapter, load_checkpoint_lock,
                                       resolve_checkpoint, resolve_source_checkout)
from frozen_retrieval.rendering import accepted_receipts, _rss_bytes

SAMPLE_SEED = 157
SAMPLE_COUNT = 128
TIMED_PASSES = 3
STRATA = ("4–6 s", "6–8 s", "8–10 s", "10 s")
COMMON_GALLERY_RECALL = {"basic_pitch_trunk": 21.9, "beats_iter3_plus": 62.2, "cir_ast": 81.4,
                         "clap_music": 58.9, "dac_44khz": 31.4, "encodec_24khz": 24.2,
                         "muq_mulan": 30.2}
SHORT_NAMES = {"basic_pitch_trunk": "Basic Pitch", "beats_iter3_plus": "BEATs", "cir_ast": "CIR",
               "clap_music": "CLAP", "dac_44khz": "DAC", "encodec_24khz": "EnCodec", "muq_mulan": "MuQ-MuLan"}
DEFAULT_DATASET = PACKAGE / "outputs" / "issue152-validation-train-midi-v3-50"
DEFAULT_CHECKPOINT_ROOT = PACKAGE / "local_checkpoints"
DEFAULT_OUTPUT = PACKAGE / "outputs" / "model-efficiency-benchmark"


def _duration_band(duration: float) -> str:
    if 4.0 <= duration < 6.0:
        return STRATA[0]
    if 6.0 <= duration < 8.0:
        return STRATA[1]
    if 8.0 <= duration < 10.0:
        return STRATA[2]
    if math.isclose(duration, 10.0, abs_tol=1e-6):
        return STRATA[3]
    raise ValueError("benchmark audio duration must fall between 4 and 10 seconds")


def _stratified_sample(candidates: list[dict], *, count: int = SAMPLE_COUNT, seed: int = SAMPLE_SEED) -> list[dict]:
    buckets = {band: [] for band in STRATA}
    for candidate in candidates:
        band = _duration_band(float(candidate["duration_seconds"]))
        buckets[band].append({**candidate, "duration_band": band})
    if count != len(STRATA) * (count // len(STRATA)) or count < len(STRATA):
        raise ValueError("sample count must divide evenly across the four duration strata")
    per_stratum = count // len(STRATA)
    empty = [band for band, rows in buckets.items() if len(rows) < per_stratum]
    if empty:
        raise ValueError(f"not enough accepted WAVs in duration stratum(s): {', '.join(empty)}")
    rng = random.Random(seed)
    selected = []
    for band in STRATA:
        selected.extend(rng.sample(sorted(buckets[band], key=lambda row: row["request_id"]), per_stratum))
    return selected


def _wav_candidates(dataset: Path) -> tuple[list[dict], dict]:
    from scipy.io import wavfile

    dataset = dataset.resolve(strict=True)
    receipts = accepted_receipts(dataset)
    candidates, source_rates, channel_counts, dtypes, skipped = [], Counter(), Counter(), Counter(), 0
    for request_id, receipt in sorted(receipts.items()):
        relative = receipt.get("wav_relative_path")
        if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
            skipped += 1
            continue
        path = (dataset / relative).resolve()
        if not path.is_relative_to(dataset) or not path.is_file():
            skipped += 1
            continue
        sample_rate, samples = wavfile.read(path, mmap=True)
        duration = samples.shape[0] / sample_rate
        if duration < 4.0 or duration > 10.0 + 1e-6:
            continue
        if not math.isclose(duration, float(receipt["duration_seconds"]), abs_tol=1 / sample_rate):
            raise ValueError("render receipt duration differs from the existing WAV header")
        channels = samples.shape[1] if samples.ndim == 2 else 1
        source_rates[str(sample_rate)] += 1
        channel_counts[str(channels)] += 1
        dtypes[str(samples.dtype)] += 1
        candidates.append({"request_id": request_id, "wav_relative_path": path.relative_to(dataset).as_posix(),
                           "audio_sha256": receipt["audio_sha256"], "duration_seconds": duration,
                           "duration_band": _duration_band(duration), "sample_rate": int(sample_rate),
                           "channels": int(channels), "dtype": str(samples.dtype)})
    profile = {"accepted_receipts": len(receipts), "eligible_existing_wavs": len(candidates),
               "skipped_or_ineligible_receipts": skipped, "source_sample_rates_hz": dict(sorted(source_rates.items())),
               "source_channels": dict(sorted(channel_counts.items())), "source_dtypes": dict(sorted(dtypes.items()))}
    return candidates, profile


def _percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def _summarize_measurements(records: list[dict]) -> dict:
    def summarize(rows: list[dict]) -> dict:
        per_record = [(row, statistics.median(row["latency_seconds"])) for row in rows if row["latency_seconds"]]
        forwards = [(row, statistics.median(row["forward_seconds"])) for row in rows if row["forward_seconds"]]
        wall_seconds = sum(sum(row["latency_seconds"]) for row in rows)
        audio_seconds = sum(float(row["duration_seconds"]) * len(row["latency_seconds"]) for row in rows)
        return {"recordings": len(per_record), "latency_median_ms": 1000 * statistics.median(value for _, value in per_record) if per_record else None,
                "latency_p90_ms": 1000 * _percentile([value for _, value in per_record], .9) if per_record else None,
                "forward_median_ms": 1000 * statistics.median(value for _, value in forwards) if forwards else None,
                "forward_p90_ms": 1000 * _percentile([value for _, value in forwards], .9) if forwards else None,
                "audio_seconds_per_wall_second": audio_seconds / wall_seconds if wall_seconds else None,
                "timed_calls": sum(len(row["latency_seconds"]) for row in rows)}

    overall = summarize(records)
    overall["duration_bands"] = {band: summarize([row for row in records if row["duration_band"] == band]) for band in STRATA}
    return overall


def _write_tradeoff_figure(results: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(7.4, 4.2), constrained_layout=True)
    offsets = {"basic_pitch_trunk": (7, 4), "beats_iter3_plus": (7, 5), "cir_ast": (7, 5),
               "clap_music": (7, 4), "dac_44khz": (-66, 5), "encodec_24khz": (7, -12),
               "muq_mulan": (7, 5)}
    for name, result in results["models"].items():
        if not result.get("all_inputs_processed"):
            continue
        latency = result["aggregate"]["latency_median_ms"]
        recall = COMMON_GALLERY_RECALL[name]
        axes.scatter(latency, recall, s=52, zorder=3)
        axes.annotate(SHORT_NAMES[name], (latency, recall), xytext=offsets[name],
                      textcoords="offset points", fontsize=8)
    axes.set_xscale("log")
    axes.set_xticks([50, 100, 250, 500, 1000, 2000], labels=["50", "100", "250", "500", "1,000", "2,000"])
    axes.set_xlabel("Median audio-to-embedding latency (ms; log scale)")
    axes.set_ylabel("Best tested original common-gallery Recall@1 (%)")
    axes.set_ylim(15, 88)
    axes.grid(True, which="major", alpha=.25)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _sync_xpu(torch) -> None:
    torch.xpu.synchronize()


def _forward_target(adapter):
    if adapter.name == "basic_pitch_trunk":
        return adapter.model, "forward_trunk", "contour trunk forward"
    if adapter.name == "beats_iter3_plus":
        return adapter.model, "extract_features", "BEATs feature forward"
    if adapter.name == "clap_music":
        return adapter.model, "get_audio_embedding_from_data", "official embedding wrapper (includes its input preparation)"
    if adapter.name == "muq_mulan":
        return adapter.model, "forward", "MuQ audio forward (includes model-native frontend)"
    if adapter.name == "cir_ast":
        return adapter.model, "compute_audio_embeddings", "official embedding wrapper (includes its input preparation)"
    if adapter.name == "vital_jepa_v7":
        return adapter.model, "forward", "local embedding wrapper (includes its log-mel frontend)"
    return adapter.model.encoder, "forward", "continuous codec encoder forward"


def _instrument_forward(adapter, torch, measurements: list[float]) -> tuple[bool, str, callable | None]:
    owner, attribute, boundary = _forward_target(adapter)
    original = getattr(owner, attribute, None)
    if not callable(original):
        return False, boundary, None

    def timed(*args, **kwargs):
        _sync_xpu(torch)
        started = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            _sync_xpu(torch)
            measurements.append(time.perf_counter() - started)

    try:
        setattr(owner, attribute, timed)
    except (AttributeError, TypeError, RuntimeError):
        return False, boundary, None

    def restore():
        setattr(owner, attribute, original)

    return True, boundary, restore


def _record_failure(rows: list[dict], model_name: str, request_id: str, pass_index: int, error: Exception):
    rows.append({"model": model_name, "request_id": request_id, "pass": pass_index,
                 "failure_type": type(error).__name__, "failure_message": str(error)[:240]})


def _reuse_completed_worker(path: Path, model_name: str, expected_ids: set[str], lock_entry: dict) -> dict | None:
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        result, records = raw["result"], raw["records"]
        if (result.get("model") != model_name or not result.get("all_inputs_processed")
                or result.get("input_count") != SAMPLE_COUNT or result.get("timed_passes") != TIMED_PASSES
                or result.get("batch_size") != 1 or result.get("precision") != "float32"
                or not str(result.get("device", "")).startswith("Intel(R) Arc")
                or result.get("checkpoint_sha256") != lock_entry.get("sha256")
                or result.get("checkpoint_revision") != lock_entry.get("revision")
                or result.get("loader_revision") != lock_entry.get("loader_revision")
                or {row["request_id"] for row in records} != expected_ids
                or any(len(row["latency_seconds"]) != TIMED_PASSES for row in records)
                or raw.get("failures") or len(raw.get("pass_order_request_ids", [])) != TIMED_PASSES
                or any(set(order) != expected_ids for order in raw["pass_order_request_ids"])):
            return None
        return result
    except (KeyError, OSError, TypeError, ValueError):
        return None


def _parse_python_paths(values: list[str]) -> dict[str, Path]:
    paths = {}
    for value in values:
        model_name, separator, raw_path = value.partition("=")
        if not separator or model_name not in MODEL_REGISTRY or not raw_path:
            raise ValueError("--pythonpath must use MODEL=existing-site-packages-directory")
        if model_name in paths:
            raise ValueError(f"--pythonpath was provided more than once for {model_name}")
        path = Path(raw_path).resolve(strict=True)
        if not path.is_dir():
            raise ValueError(f"--pythonpath for {model_name} must point to an existing directory")
        paths[model_name] = path
    return paths


def _run_worker(model_name: str, sample_path: Path, dataset: Path, checkpoint_root: Path, lock_path: Path,
                raw_path: Path) -> dict:
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"})
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    rows = sample["records"]
    if len(rows) != SAMPLE_COUNT or len({row["request_id"] for row in rows}) != SAMPLE_COUNT:
        raise ValueError("sample manifest must contain 128 distinct rendered recordings")

    import torch

    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError("the requested XPU is unavailable; CPU fallback is forbidden")
    torch.xpu.init()
    device_name = torch.xpu.get_device_name(0)
    torch.xpu.reset_peak_memory_stats()
    peak_rss = [_rss_bytes() or 0]
    stop_sampler = threading.Event()

    def sample_rss():
        while not stop_sampler.wait(.01):
            current = _rss_bytes()
            if current is not None:
                peak_rss[0] = max(peak_rss[0], current)

    sampler = threading.Thread(target=sample_rss, name="rss-sampler", daemon=True)
    sampler.start()
    lock = load_checkpoint_lock(lock_path)
    loading_started = time.perf_counter()
    checkpoint, lock_entry, checkpoint_digest = resolve_checkpoint(model_name, checkpoint_root, lock)
    source_root = resolve_source_checkout(checkpoint_root, lock_entry)
    adapter = load_adapter(model_name, checkpoint, lock_entry, checkpoint_digest, checkpoint_root=checkpoint_root,
                           device="xpu", source_root=source_root)
    _sync_xpu(torch)
    model_loading_seconds = time.perf_counter() - loading_started
    parameter_owner = next((target for target in (adapter.model, getattr(adapter.model, "model", None))
                            if callable(getattr(target, "parameters", None))), None)
    parameter_dtypes = {str(parameter.dtype) for parameter in parameter_owner.parameters()} if parameter_owner else set()
    if parameter_dtypes and parameter_dtypes != {"torch.float32"}:
        raise RuntimeError(f"locked model parameters are not all FP32: {sorted(parameter_dtypes)}")
    resident_xpu_bytes = int(torch.xpu.memory_allocated())

    forward_measurements: list[float] = []
    forward_available, forward_boundary, restore_forward = _instrument_forward(adapter, torch, forward_measurements)
    failures = []
    warmup_ids = []
    first_by_band = {band: next(row for row in rows if row["duration_band"] == band) for band in STRATA}
    try:
        for band in STRATA:
            row = first_by_band[band]
            try:
                adapter.extract(dataset / row["wav_relative_path"])
                warmup_ids.append(row["request_id"])
            except Exception as error:
                _record_failure(failures, model_name, row["request_id"], 0, error)
    finally:
        if restore_forward is not None:
            restore_forward()

    forward_measurements.clear()
    print(f"{model_name}: warm-up complete; starting three timed passes.", flush=True)
    pass_orders = []
    measurements = {row["request_id"]: {"request_id": row["request_id"], "duration_seconds": row["duration_seconds"],
                                         "duration_band": row["duration_band"], "latency_seconds": [], "forward_seconds": []}
                    for row in rows}
    forward_available, forward_boundary, restore_forward = _instrument_forward(adapter, torch, forward_measurements)
    try:
        for pass_index in range(1, TIMED_PASSES + 1):
            order = list(rows)
            random.Random(SAMPLE_SEED + pass_index).shuffle(order)
            pass_orders.append([row["request_id"] for row in order])
            for row in order:
                call_forward_start = len(forward_measurements)
                try:
                    _sync_xpu(torch)
                    started = time.perf_counter()
                    adapter.extract(dataset / row["wav_relative_path"])
                    _sync_xpu(torch)
                    elapsed = time.perf_counter() - started
                    record = measurements[row["request_id"]]
                    record["latency_seconds"].append(elapsed)
                    calls = forward_measurements[call_forward_start:]
                    if calls:
                        record["forward_seconds"].append(sum(calls))
                except Exception as error:
                    _record_failure(failures, model_name, row["request_id"], pass_index, error)
            print(f"{model_name}: completed timed pass {pass_index}/{TIMED_PASSES}.", flush=True)
    finally:
        if restore_forward is not None:
            restore_forward()
        _sync_xpu(torch)
        stop_sampler.set()
        sampler.join(timeout=2)

    peak_rss[0] = max(peak_rss[0], _rss_bytes() or 0)
    peak_xpu_bytes = int(torch.xpu.max_memory_allocated())
    per_record = list(measurements.values())
    aggregate = _summarize_measurements(per_record)
    successful_recordings = sum(len(row["latency_seconds"]) == TIMED_PASSES for row in per_record)
    all_inputs_processed = successful_recordings == SAMPLE_COUNT and not failures
    result = {"model": model_name, "display_name": MODEL_REGISTRY[model_name]["name"],
              "all_inputs_processed": all_inputs_processed, "input_count": SAMPLE_COUNT,
              "successful_recordings_all_passes": successful_recordings, "successful_timed_calls": aggregate["timed_calls"],
              "timed_passes": TIMED_PASSES, "warmup_recordings": len(warmup_ids), "warmup_failure_count": len(failures) - sum(
                  int(row["pass"] != 0) for row in failures), "failure_count": len(failures),
              "failure_types": dict(Counter(row["failure_type"] for row in failures)),
              "model_loading_seconds": model_loading_seconds, "resident_xpu_bytes_after_load": resident_xpu_bytes,
              "peak_process_rss_bytes": peak_rss[0], "peak_xpu_allocated_bytes": peak_xpu_bytes,
              "device": device_name, "precision": "float32", "batch_size": 1,
              "forward_timing_available": forward_available, "forward_timing_boundary": forward_boundary,
              "checkpoint_sha256": checkpoint_digest, "checkpoint_revision": lock_entry["revision"],
              "loader_revision": lock_entry["loader_revision"], "local_config_pins": {
                  name: {"revision": record["revision"], "sha256": record["sha256"]}
                  for name, record in lock_entry.get("local_configs", {}).items()},
              "runtime_versions": lock_entry["runtime_versions"], "aggregate": aggregate}
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps({"result": result, "records": per_record, "failures": failures,
                                    "pass_order_request_ids": pass_orders}, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return result


def _run_parent(args) -> int:
    dataset = args.dataset.resolve(strict=True)
    checkpoint_root = args.checkpoint_root.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    python_paths = _parse_python_paths(args.pythonpath)
    candidates, source_profile = _wav_candidates(dataset)
    selected = _stratified_sample(candidates, count=SAMPLE_COUNT, seed=SAMPLE_SEED)
    for row in selected:
        wav_path = dataset / row["wav_relative_path"]
        if file_sha256(wav_path) != row["audio_sha256"]:
            raise ValueError("selected rendered WAV differs from its accepted render receipt")
    profile = {**source_profile, "sample_count": SAMPLE_COUNT, "seed": SAMPLE_SEED,
               "duration_strata_counts": dict(Counter(row["duration_band"] for row in selected)),
               "duration_seconds": {"minimum": min(row["duration_seconds"] for row in selected),
                                    "median": statistics.median(row["duration_seconds"] for row in selected),
                                    "maximum": max(row["duration_seconds"] for row in selected)},
               "sample_source_rates_hz": dict(Counter(str(row["sample_rate"]) for row in selected)),
               "sample_channels": dict(Counter(str(row["channels"]) for row in selected)),
               "sample_dtypes": dict(Counter(row["dtype"] for row in selected))}
    sample_path = output / "sample.private.json"
    sample_path.write_text(json.dumps({"profile": profile, "records": selected}, sort_keys=True, indent=2) + "\n",
                           encoding="utf-8")

    lock_path = checkpoint_root / "checkpoint-lock.json"
    lock = load_checkpoint_lock(lock_path)
    expected_ids = {row["request_id"] for row in selected}
    model_results = {}
    worker_dir = output / "workers"
    for model_name in MODEL_REGISTRY:
        if model_name not in COMMON_GALLERY_RECALL:
            continue
        raw_path = worker_dir / f"{model_name}.private.json"
        log_path = worker_dir / f"{model_name}.private.log"
        reused = _reuse_completed_worker(raw_path, model_name, expected_ids, lock["models"][model_name])
        if reused is not None:
            model_results[model_name] = reused
            print(f"{model_name}: reused its completed isolated-process benchmark.", flush=True)
            continue
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", model_name,
                   "--sample-manifest", str(sample_path), "--dataset", str(dataset),
                   "--checkpoint-root", str(checkpoint_root), "--lock", str(lock_path), "--raw-output", str(raw_path)]
        worker_env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        if model_name in python_paths:
            worker_env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(python_paths[model_name]), worker_env.get("PYTHONPATH", ""))))
        completed = subprocess.run(command, capture_output=True, text=True, check=False, env=worker_env)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
        if completed.returncode == 0 and raw_path.is_file():
            model_results[model_name] = json.loads(raw_path.read_text(encoding="utf-8"))["result"]
            print(f"{model_name}: {model_results[model_name]['successful_recordings_all_passes']}/128 recordings succeeded in all three passes.", flush=True)
        else:
            model_results[model_name] = {"all_inputs_processed": False, "input_count": SAMPLE_COUNT,
                                         "failure_count": None,
                                         "failure": "worker failed; details are in the ignored private log"}
            print(f"{model_name}: benchmark worker failed; see ignored outputs for its private log.", flush=True)

    results = {"schema_version": 1, "method": {"seed": SAMPLE_SEED, "sample_count": SAMPLE_COUNT,
               "duration_strata_seconds": {STRATA[0]: [4, 6], STRATA[1]: [6, 8], STRATA[2]: [8, 10], STRATA[3]: [10, 10]},
               "equal_allocation_per_duration_stratum": SAMPLE_COUNT // len(STRATA), "timed_passes": TIMED_PASSES,
               "warmups_per_duration_stratum": 1, "batch_size": 1, "device_required": "XPU",
               "precision_required": "float32", "isolated_process_per_encoder": True,
               "timing": "synchronized per-recording wall time around existing adapter.extract; includes WAV read, mono conversion/resampling, adapter preprocessing, forward, output transfer and pooling; excludes load and warmup",
               "forward_timing": "synchronized around the existing model forward entry point where instrumentable; wrapper boundaries may include model-native preprocessing",
               "ram_sampling_interval_seconds": 0.01, "sample_profile": profile},
               "models": model_results}
    aggregate_path = args.aggregate_output.resolve()
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path = aggregate_path.parent / "figures" / "model_efficiency_vs_retrieval.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    _write_tradeoff_figure(results, figure_path)
    results["visualization"] = "figures/model_efficiency_vs_retrieval.png"
    aggregate_path.write_text(json.dumps(results, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    complete = all(row.get("all_inputs_processed") for row in model_results.values())
    print(f"Saved sanitized aggregate results to {aggregate_path}.", flush=True)
    return 0 if complete else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Standardized XPU inference-efficiency benchmark for frozen audio adapters")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--aggregate-output", type=Path, default=PACKAGE / "reports" / "model_efficiency_benchmark.json")
    parser.add_argument("--pythonpath", action="append", default=[], metavar="MODEL=DIR",
                        help="prepend an existing locked dependency site-packages directory for one model worker")
    parser.add_argument("--worker", choices=tuple(MODEL_REGISTRY))
    parser.add_argument("--sample-manifest", type=Path)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--raw-output", type=Path)
    args = parser.parse_args(argv)
    if args.worker:
        required = (args.sample_manifest, args.lock, args.raw_output)
        if any(value is None for value in required):
            parser.error("worker mode requires --sample-manifest, --lock and --raw-output")
        return 0 if _run_worker(args.worker, args.sample_manifest, args.dataset, args.checkpoint_root, args.lock,
                                args.raw_output) else 1
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())

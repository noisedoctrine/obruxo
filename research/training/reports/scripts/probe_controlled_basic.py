"""Render a small fixed-phase factorial and diagnose frozen basic-model separation."""

import argparse
from itertools import product
import json
from pathlib import Path
import sys
from time import perf_counter

REPOSITORY = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(REPOSITORY / "research/training"), str(REPOSITORY / "research/data_generation")]


def main():
    import numpy as np
    import torch
    from obruxo_data.hashing import canonical_sha256, sha256_bytes
    from obruxo_data.midi import Performance
    from obruxo_data.render.base import RenderRequest
    from obruxo_training.audit import write_json
    from obruxo_training.basic_direct import restore_direct
    from obruxo_training.benchmark import MemoryMonitor
    from obruxo_training.category_scale_refresh import cpu_settings
    from obruxo_training.checkpoints import atomic_torch_save
    from obruxo_training.codec import CONTINUOUS
    from obruxo_training.conditioned_bridge import read
    from obruxo_training.controlled_probes import FACTORS, factor_readouts, patch_prediction_stability, patch_readout, validate_grid
    from obruxo_training.features import tensor_digest
    from obruxo_training.listening_loudness import LoudnessMeter
    from obruxo_training.model import batch_features
    from obruxo_training.native_fit import digest
    from obruxo_training.performances import initial_bank
    from obruxo_training.pilot import PilotPool
    from obruxo_training.reproducibility import fixed_generator_phases
    from obruxo_training.stream import NativeRendererFactory, RenderWorker
    from probe_frozen_basic import representations
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fit", "initial", "pool", "reserved", "output", "native-deps", "meter-deps", "ffmpeg"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    sys.path[:0] = [str(args.native_deps), str(args.meter_deps)]
    import pyloudnorm
    fit, proof, reservation = read(args.fit / "plan.json"), read(args.fit / "verification.json"), read(args.reserved)
    paths = {"initial": args.initial, "parameter": args.fit / "diagnostic-parameter.pt", "spectral": args.fit / "diagnostic-spectral.pt"}
    expected = {"initial": fit["initial_file_sha256"], **{arm: proof["evidence_sha256"][f"diagnostic-{arm}.pt"] for arm in ("parameter", "spectral")}}
    pool = PilotPool.load(args.pool)
    if (not proof["verified"] or proof["plan_id"] != canonical_sha256(fit) or pool.pool_id != fit["pool_id"]
            or any(digest(path) != expected[name] for name, path in paths.items())):
        raise ValueError("controlled probes require unchanged verified basic models and pool")
    entries = {entry.state_id: entry for entry in pool.entries}
    states = [row["state_id"] for row in fit["training"]]
    if len(set(states)) != 12 or any(pool.manifest.by_state[state].split != "train" for state in states):
        raise ValueError("controlled probes use exactly the already-used twelve training patches")
    performances = []
    for values in product(*FACTORS.values()):
        pitch, velocity, duration = values
        performance = Performance(bpm=120)
        performance.add_note(pitch=pitch, velocity=velocity, start_tick=0, duration_ticks=duration)
        performance.end_tick = 3840
        performances.append({**dict(zip(FACTORS, values, strict=True)), "performance": performance.to_dict(),
                             "performance_id": canonical_sha256(performance.to_dict())})
    forbidden = {entry.performance_id for entry in initial_bank()}
    # The reserved performances are inspected only for identity exclusion, never rendered or scored.
    def reserved_ids(value):
        if isinstance(value, dict):
            identities = {canonical_sha256(value)} if "events" in value and "ticks_per_beat" in value else set()
            return identities | {item for key, item in value.items() if key == "performance_id" and isinstance(item, str)} | set().union(*(reserved_ids(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(reserved_ids(item) for item in value))
        return set()
    reserved = reserved_ids(reservation["performances"])
    if len(reserved) != 24:
        raise ValueError("controlled probe must exclude all 24 reserved confirmation performances")
    forbidden |= reserved
    if any(row["performance_id"] in forbidden for row in performances):
        raise ValueError("controlled grid overlaps a reserved performance")
    rows = [{"state_id": state, "categorical": list(entries[state].target.categorical), "continuous": list(entries[state].target.continuous), **performance}
            for state in states for performance in performances]
    validate_grid(rows)
    sources = ("reports/scripts/probe_controlled_basic.py", "reports/scripts/probe_frozen_basic.py", "obruxo_training/controlled_probes.py",
               "obruxo_training/representation_probes.py", "obruxo_training/features.py", "obruxo_training/stream.py", "obruxo_training/reproducibility.py")
    bound = {str(path): digest(path) for path in (*paths.values(), args.pool, args.reserved, args.fit / "verification.json")}
    meter = LoudnessMeter(args.ffmpeg)
    plan = {"format": "controlled_frozen_basic_probes_v1", "inputs_sha256": bound,
            "source_sha256_lf": {name: sha256_bytes((REPOSITORY / "research/training" / name).read_bytes().replace(b"\r\n", b"\n")) for name in sources},
            "patches": states, "observations": rows, "factors": FACTORS, "all_clips_seconds": 4.25, "preroll_frames": 32768,
            "expected_renderer_identity": fit["expected_renderer_identity"], "repeat_cell": {"pitch": 60, "velocity": 64, "duration_ticks": 384},
            "loudness": {"meter": meter.identity, "initial_target_lufs": -23., "maximum_spread_lu": .1, "true_peak_ceiling_dbtp": -1.,
                         "processing": "one_constant_gain_per_complete_clip_from_original_samples", "maximum_gain_solver_passes": 4},
            "ridge": .1, "null_seeds": list(range(5)), "factor_folds": 2, "factor_fold_unit": "whole_patch_stratified_by_source_categories",
            "patch_probe_split": "train_pitch60_test48and72_same12patches", "inverse_model_optimizer_updates": 0,
            "limits": {"native_requests": 156, "wall_seconds": 1200, "process_tree_bytes": 4 * 1024**3, "archive_bytes": 512 * 1024**2},
            "limitations": ["Twelve patches were already used in inverse training; this is a diagnostic, not final validation.",
                            "Velocity and oscillator level can both change gain; LUFS matching removes that dimension intentionally.",
                            "A readable factor is present but the decoder may not use it; fixed outputs alone could reflect collapse."]}
    if args.output.exists():
        if read(args.output / "plan.json") != json.loads(json.dumps(plan)) or (args.output / "summary.json").exists():
            raise ValueError("preserve a changed or completed controlled-probe run")
    else:
        args.output.mkdir(parents=True)
        write_json(args.output / "plan.json", plan)
    for directory in ("recordings", "attempts", "repeats"):
        (args.output / directory).mkdir(exist_ok=True)
    started, monitor, recordings = perf_counter(), MemoryMonitor(), []
    monitor.start()
    try:
        with cpu_settings(), RenderWorker(renderer_factory=NativeRendererFactory(renderer_id=pool.codec.contract.renderer_id), max_jobs=32) as worker:
            for index, row in enumerate(rows):
                path = args.output / "recordings" / f"{index:03d}.pt"
                receipt_path = path.with_suffix(".json")
                if path.exists() and receipt_path.exists():
                    record = read(receipt_path)
                    if digest(path) != record["file_sha256"] or record["index"] != index:
                        raise ValueError("saved controlled recording changed")
                    recordings.append(record)
                    continue
                if (args.output / "attempts" / f"{index:03d}.json").exists():
                    raise ValueError("preserve and inspect the incomplete native attempt before resuming")
                request = RenderRequest(preset_json=fixed_generator_phases(pool.codec.decode(entries[row["state_id"]].target)).decode(),
                           performance=Performance.from_dict(row["performance"]), tail_seconds=.25,
                           renderer_id=pool.codec.contract.renderer_id, preroll_frames=32768)
                audio, metadata = worker.render(request, timeout=30.)
                identity = {key: metadata.get("provenance", {}).get(key) for key in fit["expected_renderer_identity"]}
                record = {"index": index, "request_id": request.request_id, "metadata": metadata}
                write_json(args.output / "attempts" / f"{index:03d}.json", record)
                if (audio is None or metadata["status"] != "accepted" or identity != fit["expected_renderer_identity"]
                        or metadata["provenance"]["request_id"] != request.request_id):
                    raise ValueError("controlled source render failed acceptance or renderer identity")
                repeat = None
                if all(row[name] == value for name, value in plan["repeat_cell"].items()):
                    repeated, repeat_metadata = worker.render(request, timeout=30.)
                    repeat = {"metadata": repeat_metadata, "exact": repeated is not None and np.array_equal(audio, repeated)}
                    write_json(args.output / "repeats" / f"{index:03d}.json", repeat)
                    if not repeat["exact"] or repeat_metadata["status"] != "accepted":
                        raise ValueError("fixed-phase controlled repeat was not exact")
                atomic_torch_save(path, {"audio": torch.from_numpy(audio.copy()), "request_id": request.request_id})
                record.update(file_sha256=digest(path), audio_sha256=sha256_bytes(audio.tobytes()), before=meter.measure(audio), repeat=repeat)
                write_json(receipt_path, record)
                recordings.append(record)
                write_json(args.output / "progress.json", {"stage": "render", "recordings": len(recordings), "total": len(rows), "elapsed_seconds": perf_counter() - started})
                if perf_counter() - started > 1200:
                    raise ValueError("controlled probe exceeded wall allocation")
                if len(recordings) % 12 == 0:
                    print(json.dumps({"recordings": len(recordings), "total": len(rows), "elapsed_seconds": perf_counter() - started}), flush=True)
            target = min(-23., *(record["before"]["integrated_lufs"] - 1.1 - record["before"]["true_peak_upper_dbtp"] for record in recordings))
            independent = pyloudnorm.Meter(44100, filter_class="DeMan")
            initial, frontend, codec = restore_direct(torch.load(args.initial, map_location="cpu", weights_only=True))
            del initial
            extracted, processing = {"original": [], "matched": []}, []
            for index, (row, record) in enumerate(zip(rows, recordings, strict=True)):
                audio = torch.load(args.output / "recordings" / f"{index:03d}.pt", map_location="cpu", weights_only=True)["audio"].numpy()
                gain, after, matched = target - record["before"]["integrated_lufs"], None, None
                for iteration in range(4):
                    matched = (audio.astype(np.float64) * 10**(gain / 20)).astype(np.float32)
                    after = meter.measure(matched)
                    if abs(after["integrated_lufs"] - target) <= .025 and after["true_peak_upper_dbtp"] <= -1.:
                        break
                    gain += target - after["integrated_lufs"]
                checked = float(independent.integrated_loudness(matched.astype(np.float64)))
                if abs(checked - target) > .05 or abs(after["integrated_lufs"] - target) > .025 or after["true_peak_upper_dbtp"] > -1.:
                    raise ValueError("controlled matched clip misses independent LUFS/headroom tolerance")
                processing.append({"index": index, "gain_db": gain, "after": after, "independent_lufs": checked, "passes": iteration + 1})
                for condition, value in (("original", audio), ("matched", matched)):
                    extracted[condition].append({**row, "features": frontend(value)})
                write_json(args.output / "progress.json", {"stage": "features", "recordings": index + 1, "total": len(rows), "elapsed_seconds": perf_counter() - started})
            frontend.require_frozen()
            atomic_torch_save(args.output / "features.pt", extracted)
            write_json(args.output / "loudness.json", {"target_lufs": target, "clips": processing, "frontend": frontend.describe(),
                       "maximum_independent_spread_lu": max(row["independent_lufs"] for row in processing) - min(row["independent_lufs"] for row in processing)})
            result, active = {}, torch.tensor([codec.continuous_mask(entries[row["state_id"]].target) for row in rows])
            for model_name, path in paths.items():
                model, _front, _codec = restore_direct(torch.load(path, map_location="cpu", weights_only=True))
                for condition, observations in extracted.items():
                    branches = representations(model, observations)
                    if model_name == "initial":
                        branches.update(raw_mel=torch.stack([row["features"]["timbre"].flatten() for row in observations]),
                                        raw_basic_pitch=torch.stack([row["features"]["pitch"] for row in observations]))
                    measured = {name: {"factor_readouts": factor_readouts(features, rows), "patch_readout": patch_readout(features, rows),
                                      "mean_feature_variance": float(features.double().var(0, unbiased=False).mean())} for name, features in branches.items()}
                    with torch.inference_mode():
                        predictions = model(batch_features(observations))
                        categories = torch.stack([value.argmax(1) for value in predictions["categorical"]], 1)
                    result[f"{model_name}/{condition}"] = {"branches": measured, "prediction_stability": patch_prediction_stability(predictions["continuous"], categories, rows, active)}
                    print(json.dumps({"scored": f"{model_name}/{condition}", "timbre": {key: value["accuracy"] for key, value in measured["timbre"]["factor_readouts"].items()}}), flush=True)
                    write_json(args.output / "progress.json", {"stage": "probes", "scored": list(result), "elapsed_seconds": perf_counter() - started})
    finally:
        monitor.close()
    archive = sum(path.stat().st_size for path in args.output.rglob("*") if path.is_file())
    resources = {"elapsed_seconds": perf_counter() - started, "peak_process_tree_rss_bytes": monitor.peak_bytes, "rss_sampling_errors": monitor.errors, "archive_bytes_before_summary": archive}
    if (any(digest(Path(path)) != value for path, value in bound.items()) or resources["elapsed_seconds"] > 1200 or monitor.errors
            or not monitor.peak_bytes or monitor.peak_bytes > 4 * 1024**3 or archive > 510 * 1024**2):
        raise ValueError("controlled diagnostic changed its inputs or exceeded resource allocation")
    summary = {"plan_id": canonical_sha256(plan), "complete": True, "recordings": len(rows), "exact_repeats": 12, "native_requests": 156,
               "inverse_model_optimizer_updates": 0, "new_validation_or_test_used": False, "input_artifacts_unchanged": True, "results": result,
               "continuous_control_order": [row[0] for row in CONTINUOUS], "resources": resources, "features_sha256": digest(args.output / "features.pt"),
               "features_tensor_sha256": {condition: canonical_sha256([tensor_digest(row["features"]) for row in observations]) for condition, observations in extracted.items()},
               "qualified_for_release": False, "limitations": plan["limitations"]}
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "progress.json", {"complete": True, "resources": resources})
    print(json.dumps({"complete": True, "resources": resources}), flush=True)


if __name__ == "__main__":
    main()

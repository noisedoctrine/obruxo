"""Bounded training-time coverage diagnosis; plan first, then 36 native requests and no fitting."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import time

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def make_plan(source, diagnosis):
    from obruxo_data.hashing import canonical_sha256
    parent = json.loads((source / "plan.json").read_text())
    analysis = json.loads(diagnosis.read_text())
    if (analysis["plan_id"] != canonical_sha256(parent) or analysis["helper_sha256"] != digest(source / "helper.pt")
            or analysis["records_sha256"] != digest(source / "records.json")):
        raise ValueError("training diagnosis belongs to different source evidence")
    for name, expected in parent["source"].items():
        path = RESEARCH / ("data_generation/obruxo_data/render/vita.py" if name == "vita.py" else "training/obruxo_training/" + name)
        if digest(path) != expected:
            raise ValueError("frozen helper/renderer source changed")
    selected = []
    for group in ("attack", "decay", "release", "detune"):
        ids = sorted(row["pair_id"] for row in analysis["pairs"][group])
        if len(ids) < 3:
            raise ValueError("insufficient training pairs")
        for index in (0, len(ids) // 2, len(ids) - 1):
            pair_id = ids[index]
            recipes = [row for row in parent["training_recipes"] if row.get("pair_id") == pair_id]
            if len(recipes) != 2 or recipes[0]["performance"] != recipes[1]["performance"]:
                raise ValueError("expected one aligned original/changed pair")
            selected.append({"group": group, "pair_id": pair_id, "recipes": recipes})
    return {"format": "basic_spectral_time_coverage_v1", "source_plan_id": analysis["plan_id"],
            "source_modules": parent["source"], "script_sha256": digest(Path(__file__)), "diagnosis_sha256": digest(diagnosis),
            "helper_sha256": analysis["helper_sha256"], "records_sha256": analysis["records_sha256"],
            "codec": parent["codec"], "seed": parent["config"]["seed"], "pairs": selected,
            "selection": "first_middle_last_pair_id_per_attack_decay_release_detune_not_ranked_by_error",
            "measurements": "frozen_paired_change_loss_full_recording_and_original_sampled_times",
            "coverage": "per_resolution_mean_absolute_signed_log1p_change_and_nearest_sample_to_peak_response",
            "limits": {"qualification_attempts": 36, "wall_seconds": 900, "process_tree_bytes": 8 * 1024**3,
                       "archive_bytes": 10 * 1024**2},
            "optimizer_updates": 0, "qualification_changes": False, "test_split_used": False}


def change_errors(predicted, target):
    from obruxo_training.paired_spectral import paired_change_loss
    unchanged = {"magnitudes": tuple(value[:1].expand_as(value) for value in target["magnitudes"]),
                 "envelope": target["envelope"][:1].expand_as(target["envelope"])}
    return {"helper": float(paired_change_loss(predicted, target).item()),
            "no_change": float(paired_change_loss(unchanged, target).item())}


def coverage(full, sampled, full_times, sampled_times):
    import torch
    result = []
    for index, (name, values, sparse, times, sparse_times) in enumerate(zip(("fft512", "fft2048", "fft8192", "envelope"),
            (*full["magnitudes"], full["envelope"]), (*sampled["magnitudes"], sampled["envelope"]),
            full_times, sampled_times, strict=True)):
        valid = full["masks"][index][0] if index < 3 else full["envelope_mask"][0]
        values, times = values[..., valid], times[:, valid]
        delta = (torch.log1p(values[1] / .0001) - torch.log1p(values[0] / .0001)).abs()
        sparse_delta = (torch.log1p(sparse[1] / .0001) - torch.log1p(sparse[0] / .0001)).abs()
        response = delta.flatten(0, -2).mean(0)
        peak_time = times[0, response.argmax()]
        result.append({"resolution": name, "full_mean_change": float(delta.mean()), "sampled_mean_change": float(sparse_delta.mean()),
                       "peak_response_time_seconds": float(peak_time), "peak_response": float(response.max()),
                       "nearest_sample_to_peak_seconds": float((sparse_times[0] - peak_time).abs().min())})
    return result


def check_pair(pair, cache, positions, helper, adapter, session, seed):
    import numpy as np
    import torch
    from obruxo_data.hashing import canonical_sha256
    from obruxo_data.midi import Performance
    from obruxo_training.codec import PilotTarget
    from obruxo_training.paired_spectral import require_aligned_pairs
    from obruxo_training.spectral import SpectralTargets
    from obruxo_training.temporal_proxy import collate_frames, full_times, hard_controls, sample_frames

    records = [cache[positions[canonical_sha256(recipe)]] for recipe in pair["recipes"]]
    require_aligned_pairs(records)
    targets = [PilotTarget.from_dict(recipe["target"]) for recipe in pair["recipes"]]
    performance = Performance.from_dict(pair["recipes"][0]["performance"])
    audio, hashes = [], []
    for index in (0, 1, 0):
        label = f"{pair['group']}:{pair['pair_id']}:{len(hashes)}"
        rendered, receipt = session.render(targets[index], performance, "qualification", label)
        if rendered is None:
            raise RuntimeError("native coverage request failed; preserve receipt without replacement")
        expected = cache.rows[positions[canonical_sha256(pair["recipes"][index])]]["receipt"]["qa"]["audio_float32_sha256"]
        actual = receipt["qa"]["audio_float32_sha256"]
        if expected != actual:
            raise ValueError("rerender differs from the original fixed-phase training audio")
        hashes.append(actual)
        if len(audio) < 2:
            audio.append(rendered.copy())
    target = SpectralTargets()(torch.from_numpy(np.stack(audio)).permute(0, 2, 1))
    sparse, sparse_times = collate_frames(records)
    for index in range(2):
        one = {key: tuple(value[index:index + 1] for value in values) if isinstance(values, tuple) else values[index:index + 1]
               for key, values in target.items()}
        sampled, times = sample_frames(one, torch.Generator().manual_seed(seed + pair["pair_id"]))
        original = records[index]
        if (any(not torch.equal(a, b) for a, b in zip(times, original["times"], strict=True))
                or any(not torch.equal(a, b) for a, b in zip(sampled["magnitudes"], original["measurements"]["magnitudes"], strict=True))
                or not torch.equal(sampled["envelope"], original["measurements"]["envelope"])):
            raise ValueError("full rerender does not reproduce stored training frames exactly")
    times, controls = full_times(target), hard_controls(adapter, targets)
    played = torch.stack([record["performance"] for record in records])
    with torch.no_grad():
        full_error = change_errors(helper(controls, played, times), target)
        sparse_error = change_errors(helper(controls, played, sparse_times), sparse)
    return {"group": pair["group"], "pair_id": pair["pair_id"], "source_state_id": records[0]["source_state_id"],
            "full": full_error, "sampled": sparse_error, "coverage": coverage(target, sparse, times, sparse_times),
            "audio_sha256": hashes, "repeat_exact": hashes[0] == hashes[2], "stored_frames_exact": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    import torch
    from obruxo_data.hashing import canonical_sha256
    from obruxo_training.audit import write_json
    from obruxo_training.benchmark import MemoryMonitor
    from obruxo_training.temporal_proxy_data import NativeSession, RecordCache
    from obruxo_training.temporal_proxy_screen import load_temporal_proxy

    torch.set_num_threads(2)
    plan = make_plan(args.source, args.diagnosis)
    if not args.execute:
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output / "plan.json", plan)
        print(json.dumps({"plan_id": canonical_sha256(plan), "pairs": len(plan["pairs"]), "planned_requests": 36}))
        return
    if json.loads((args.output / "plan.json").read_text()) != plan or (args.output / "run.json").exists():
        raise ValueError("plan changed or this non-resumable diagnostic was already attempted")
    write_json(args.output / "run.json", {"started_unix": time(), "plan_id": canonical_sha256(plan)})
    helper, adapter = load_temporal_proxy(args.source / "helper.pt", require_qualified=False)
    cache = RecordCache(args.source, json.loads((args.source / "records.json").read_text()))
    positions = {row["recipe_id"]: i for i, row in enumerate(cache.rows)}
    monitor = MemoryMonitor()
    monitor.start()
    session = NativeSession(args.output, plan, adapter.codec, monitor)
    rows, failure = [], None
    try:
        for pair in plan["pairs"]:
            rows.append(check_pair(pair, cache, positions, helper, adapter, session, plan["seed"]))
            write_json(args.output / "cases.json", rows)
            session.check_limits()
            print(json.dumps({"completed_pairs": len(rows), "group": pair["group"], "full": rows[-1]["full"],
                              "sampled": rows[-1]["sampled"]}), flush=True)
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        monitor.close()
        elapsed = time() - session.started
        archive_bytes = sum(path.stat().st_size for path in args.output.rglob("*") if path.is_file())
        resources_ok = (monitor.errors == 0 and monitor.peak_bytes is not None and monitor.peak_bytes <= plan["limits"]["process_tree_bytes"]
                        and elapsed <= plan["limits"]["wall_seconds"] and archive_bytes < plan["limits"]["archive_bytes"])
        result = {"plan_id": canonical_sha256(plan), "complete": len(rows) == 12 and failure is None, "failure": failure,
                  "claim": "training_time_coverage_diagnosis_not_model_qualification", "cases": rows,
                  "native_requests": sum(row["event"] == "attempt" for row in session.events), "optimizer_updates": 0,
                  "elapsed_seconds": elapsed, "peak_process_tree_rss_bytes": monitor.peak_bytes, "rss_sampling_errors": monitor.errors,
                  "within_resource_limits": resources_ok, "archive_bytes_before_summary": archive_bytes,
                  "renderer_identity": session.renderer_identity, "test_split_used": False, "audio_archived": False}
        write_json(args.output / "summary.json", result)
    if not resources_ok:
        raise RuntimeError("diagnostic resource limit failed")


if __name__ == "__main__":
    main()

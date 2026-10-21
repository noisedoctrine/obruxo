"""Compare archived cold/settled inputs through identical initial weights; no audio or learning."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def require_pair_contract(cold, settled):
    keys = ("pool_id", "manifest_id", "codec", "frontend", "initial_file_sha256", "initial_model_sha256", "training")
    if any(cold[key] != settled[key] for key in keys) or len(cold["training"]) != 32:
        raise ValueError("paired input identity or complete 32-state recipes differ")
    if cold.get("render_profile", {}).get("preroll_frames", 0) != 0 or settled["render_profile"]["preroll_frames"] != 32768:
        raise ValueError("comparison requires the declared cold and settled profiles")
    for name in ("model.py", "features.py", "trainer.py", "reproducibility.py"):
        if cold["source"][name] != settled["source"][name]:
            raise ValueError("paired model/frontend/phase source differs")


def feature_change(cold, settled):
    if cold.shape != settled.shape or not np.isfinite(cold).all() or not np.isfinite(settled).all():
        raise ValueError("feature pair shape or finiteness changed")
    before, after = cold.astype(np.float64), settled.astype(np.float64)
    delta = after - before
    return {"relative_l2": float(np.linalg.norm(delta) / max(np.linalg.norm(before), 1e-12)),
            "mean_absolute_delta": float(np.abs(delta).mean()), "maximum_absolute_delta": float(np.abs(delta).max()),
            "exact_equal": bool(np.array_equal(cold, settled))}


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {"count": len(values), "mean": float(values.mean()), "median": float(np.median(values)),
            "p90": float(np.quantile(values, .9)), "maximum": float(values.max())}


def compare(args):
    import torch
    from obruxo_data.hashing import canonical_sha256
    from obruxo_training.benchmark import MemoryMonitor
    from obruxo_training.codec import PilotTarget
    from obruxo_training.features import tensor_digest, validate_features
    from obruxo_training.model import batch_features, predicted_targets
    from obruxo_training.trainer import restore_model
    started = perf_counter()
    torch.set_num_threads(2)
    roots = {"cold": args.cold, "settled": args.settled}
    audits = {name: json.loads(path.read_text()) for name, path in (("cold", args.cold_audit), ("settled", args.settled_audit))}
    if (not all(audits["cold"][key] for key in ("offline_prefix_replay_exact", "cached_records_hashed", "both_optimizer_states_exact", "gumbel_rng_exact"))
            or not all(audits["settled"][key] for key in ("complete", "offline_training_replay_exact", "cached_targets_and_inputs_rescored"))):
        raise ValueError("paired parents lack their required prior audits")
    plans, manifests, receipts = {}, {}, {}
    for name, root in roots.items():
        audit = audits[name]
        if any(digest(root / key) != value for key, value in audit["evidence_sha256"].items()):
            raise ValueError("parent evidence bytes changed")
        plans[name] = json.loads((root / "plan.json").read_text())
        if canonical_sha256(plans[name]) != audit["plan_id"]:
            raise ValueError("parent plan identity changed")
        manifests[name] = json.loads((root / "records.json").read_text())
        if set(manifests[name]) != {str(index) for index in range(32)}:
            raise ValueError("paired cache membership is incomplete")
        events = [json.loads(line) for line in (root / "renders.jsonl").read_text().splitlines()]
        receipts[name] = {row["label"]: row for row in events if row["event"] == "outcome" and row["label"].startswith("source:")}
    cold, settled = plans["cold"], plans["settled"]
    require_pair_contract(cold, settled)
    if digest(args.initial) != cold["initial_file_sha256"]:
        raise ValueError("initial weights differ from the paired plans")
    for name in ("model.py", "features.py", "trainer.py", "reproducibility.py"):
        path = RESEARCH / "training/obruxo_training" / name
        if sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != cold["source"][name]:
            raise ValueError("paired inference implementation changed")
    monitor = MemoryMonitor()
    monitor.start()
    try:
        model, frontend, codec = restore_model(torch.load(args.initial, map_location="cpu", weights_only=True))
        before = tensor_digest(model.state_dict())
        if before != cold["initial_model_sha256"] or frontend.describe() != cold["frontend"] or codec.to_dict() != cold["codec"]:
            raise ValueError("restored inference contracts differ")
        rows, native_identity = [], None
        for index, recipe in enumerate(cold["training"]):
            if perf_counter() - started > 300 or (monitor.peak_bytes or 0) > 8 * 1024**3:
                raise RuntimeError("saved-input comparison resource cap exceeded")
            records, predictions, files = {}, {}, {}
            for name, root in roots.items():
                entry = manifests[name][str(index)]
                path = root / "records" / entry["file"]
                if digest(path) != entry["sha256"] or entry["recipe_id"] != canonical_sha256(recipe):
                    raise ValueError("paired record hash or recipe changed")
                record = torch.load(path, map_location="cpu", weights_only=True)
                if (record["recipe_id"] != canonical_sha256(recipe)
                        or record.get("render_profile", {}).get("preroll_frames", 0) != plans[name].get("render_profile", {}).get("preroll_frames", 0)):
                    raise ValueError("paired record contract changed")
                for view in (0, 1):
                    receipt = receipts[name][f"source:{index}:{view}"]
                    if receipt["status"] != "accepted" or receipt["qa"]["audio_float32_sha256"] != record["audio_sha256"][view]:
                        raise ValueError("cached input audio differs from accepted source receipt")
                    identity = {key: receipt["provenance"][key] for key in ("engine_fingerprint", "backend_version", "renderer_id")}
                    if native_identity is not None and native_identity != identity:
                        raise ValueError("native engine/backend differs across input pairs")
                    native_identity = identity
                    if receipt["provenance"]["settings"].get("preroll_frames", 0) != plans[name].get("render_profile", {}).get("preroll_frames", 0):
                        raise ValueError("source receipt belongs to a different renderer profile")
                validate_features(record["features"])
                records[name], files[name] = record, entry["sha256"]
                with torch.no_grad():
                    output = model(batch_features([{"features": record["features"]}]))
                predictions[name] = {"continuous": output["continuous"][0].tolist(),
                    "categorical_logits": [value[0].tolist() for value in output["categorical"]],
                    "hard_target": predicted_targets(output, codec)[0].to_dict()}
            for view in (0, 1):
                pair = [receipts[name][f"source:{index}:{view}"] for name in roots]
                if any(pair[0][key] != pair[1][key] for key in ("source_sha256", "controlled_sha256")):
                    raise ValueError("source preset bytes differ across input pairs")
            rows.append({"state_id": recipe["state_id"], "recipe_id": canonical_sha256(recipe), "source_target": recipe["target"],
                "record_sha256": files, "source_audio_sha256": {name: row["audio_sha256"] for name, row in records.items()},
                "features": {name: feature_change(records["cold"]["features"][name].numpy(), records["settled"]["features"][name].numpy())
                             for name in ("timbre", "pitch")}, "predictions": predictions})
        if tensor_digest(model.state_dict()) != before:
            raise ValueError("comparison modified the initial model")
        frontend.require_frozen()
    finally:
        monitor.close()
    controls = []
    for index, (name, *_bounds) in enumerate(codec.continuous):
        enabled = [row for row in rows if codec.continuous_mask(PilotTarget.from_dict(row["source_target"]))[index]]
        errors = {profile: summary([abs(row["predictions"][profile]["hard_target"]["continuous"][index] - row["source_target"]["continuous"][index])
                                   for row in enabled]) for profile in roots}
        controls.append({"name": name, "source_active_states": len(enabled), "normalized_parameter_error": errors,
            "dense_output_absolute_change": summary([abs(row["predictions"]["settled"]["continuous"][index] - row["predictions"]["cold"]["continuous"][index]) for row in rows])})
    result = {"format": "paired_renderer_input_comparison_v1", "descriptive_only": True, "causal_claim": False,
        "native_requests": 0, "optimizer_updates": 0, "test_split_used": False, "states": 32, "rows": rows,
        "feature_relative_l2": {name: summary([row["features"][name]["relative_l2"] for row in rows]) for name in ("timbre", "pitch")},
        "controls": controls, "hard_category_changes": {name: sum(row["predictions"]["cold"]["hard_target"]["categorical"][index] != row["predictions"]["settled"]["hard_target"]["categorical"][index] for row in rows)
                                                      for index, (name, _values) in enumerate(codec.categorical)},
        "parent_plan_ids": {name: audit["plan_id"] for name, audit in audits.items()},
        "input_sha256": {"cold_audit": digest(args.cold_audit), "settled_audit": digest(args.settled_audit), "initial_weights": digest(args.initial)},
        "source_sha256": sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(), "initial_model_unchanged": True,
        "native_identity": native_identity, "same_source_patch_bytes_verified": True, "source_renderer_profiles_verified": True,
        "cold_input_audio_independently_rescored": False, "settled_input_audio_previously_rescored": True,
        "limitations": "archived cold features are hash/receipt verified, not remeasured from retained audio; parameter differences are not perceptual scores",
        "limits": {"wall_seconds": 300, "process_tree_bytes": 8 * 1024**3, "evidence_bytes": 64 * 1024**2},
        "elapsed_seconds": perf_counter() - started, "peak_process_tree_rss_bytes": monitor.peak_bytes, "rss_sampling_errors": monitor.errors}
    if (result["elapsed_seconds"] > 300 or monitor.errors or not monitor.peak_bytes or monitor.peak_bytes > 8 * 1024**3
            or len(json.dumps(result).encode()) > 64 * 1024**2):
        raise RuntimeError("saved-input comparison exceeds resource limits")
    result["within_resource_limits"] = True
    return result


def main():
    from obruxo_training.audit import write_json
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cold", "settled", "cold-audit", "settled-audit", "initial", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite comparison evidence")
    result = compare(args)
    write_json(args.output, result)
    print(json.dumps({"states": result["states"], "feature_relative_l2": result["feature_relative_l2"], "hard_category_changes": result["hard_category_changes"]}))


if __name__ == "__main__":
    main()

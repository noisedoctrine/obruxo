"""Compare saved recordings of the same training patches through one fixed model."""

import argparse
from collections import Counter
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


def pitches(performance):
    return sorted({event["data"][0] for event in performance["events"] if event["kind"] == "note_on" and event["data"][1] > 0})


def pattern_table(recipes, modes):
    counts = [[0] * 4 for _ in range(2)]
    for recipe, mode in zip(recipes, modes, strict=True):
        counts[recipe["target"]["categorical"][1]][mode] += 1
    return {"filter_off_then_on_by_held_staccato_melody_chord": counts,
            "pattern_parity_only_filter_accuracy": sum(counts[index % 2][index] for index in range(4)) / len(recipes)}


def make_plan(args):
    from obruxo_data.hashing import canonical_sha256
    full = json.loads((args.bootstrap / "summary.json").read_text())
    full_audit = json.loads(args.bootstrap_audit.read_text())
    old = json.loads(args.parent_audit.read_text())
    parent_plan = json.loads((args.parent / "plan.json").read_text())
    if (not full["preflight_passed"] or not full_audit["model_optimizer_batch_and_global_rng_exact"]
            or full_audit["plan_id"] != full["plan_id"] or full_audit["training_source_receipts_verified"] != 404
            or not old["complete"] or not old["offline_training_replay_exact"] or not old["cached_targets_and_inputs_rescored"]
            or old["plan"] != parent_plan or old["plan_id"] != canonical_sha256(parent_plan)
            or old["renderer_identity"] != full["training"]["renderer_identity"] or len(parent_plan["training"]) != 64):
        raise ValueError("input-note comparison needs the audited corrected-render parents")
    for root, audit in ((args.bootstrap, full_audit), (args.parent, old)):
        if any(digest(root / name) != expected for name, expected in audit["evidence_sha256"].items()):
            raise ValueError("parent evidence changed after its audit")
    for name in ("model.py", "features.py", "reproducibility.py", "temporal_proxy_data.py"):
        if sha256((RESEARCH / "training/obruxo_training" / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() != parent_plan["source"][name]:
            raise ValueError("paired frontend/model/phase or performance implementation changed")
    index = json.loads((args.parent / "records.json").read_text())
    if set(index) != {str(i) for i in range(64)}:
        raise ValueError("parent lacks its complete 64-state feature bank")
    fit = sorted(args.bootstrap.glob("fit-*"))[-1]
    return {"format": "basic_input_note_diagnosis_v1", "initial_sha256": digest(fit / "model.pt"),
            "bootstrap_checkpoint_sha256": digest(fit / "checkpoint.pt"), "bootstrap_audit_sha256": digest(args.bootstrap_audit),
            "parent_audit_sha256": digest(args.parent_audit), "parent_plan_id": old["plan_id"],
            "record_sha256": {key: row["sha256"] for key, row in index.items()},
            "source_sha256": sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
            "limits": {"wall_seconds": 300, "process_tree_bytes": 4 * 1024**3, "new_evidence_bytes": 64 * 1024**2},
            "native_requests": 0, "optimizer_updates": 0, "test_split_used": False}


def diagnose(args, plan):
    import torch
    from obruxo_data.hashing import canonical_sha256
    from obruxo_data.midi import Performance
    from obruxo_data.render.base import RenderRequest
    from obruxo_training.benchmark import MemoryMonitor
    from obruxo_training.codec import PilotTarget
    from obruxo_training.evaluate import load_evaluation_checkpoint
    from obruxo_training.features import tensor_digest
    from obruxo_training.model import batch_features, predicted_targets
    from obruxo_training.pilot import PilotPool
    from obruxo_training.reproducibility import fixed_generator_phases
    from obruxo_training.trainer import load_model
    started = perf_counter()
    monitor = MemoryMonitor()
    monitor.start()
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        fit = sorted(args.bootstrap.glob("fit-*"))[-1]
        pool = PilotPool.load(args.pool)
        saved, _, _, sampler = load_evaluation_checkpoint(fit / "checkpoint.pt", pool)
        model, frontend, codec = load_model(fit / "model.pt")
        old_plan = json.loads((args.parent / "plan.json").read_text())
        if codec.to_dict() != old_plan["codec"] or frontend.describe() != old_plan["frontend"]:
            raise ValueError("paired codec or frozen frontend differs")
        by_state = {item["state_id"]: item for item in saved["replay"]}
        index = json.loads((args.parent / "records.json").read_text())
        receipts = [json.loads(line) for line in (args.parent / "renders.jsonl").read_text().splitlines()]
        source = {row["label"]: row for row in receipts if row["event"] == "outcome" and row["phase"] == "training" and row["label"].startswith("source:")}
        cases, narrow_recipes, narrow_modes = [], [], []
        references = {entry.performance_id: i for i, entry in enumerate(codec.contract.performances()) if entry.role == "reference"}
        before = tensor_digest(model.state_dict())
        for i, recipe in enumerate(old_plan["training"]):
            item = by_state[recipe["state_id"]]
            if item["target"] != recipe["target"]:
                raise ValueError("paired source patch labels differ")
            path = args.parent / "records" / index[str(i)]["file"]
            if digest(path) != plan["record_sha256"][str(i)]:
                raise ValueError("paired source feature record changed")
            record = torch.load(path, map_location="cpu", weights_only=True)
            if record["recipe_id"] != canonical_sha256(recipe) or record["render_profile"] != old_plan["render_profile"]:
                raise ValueError("paired source recipe/profile differs")
            raw = codec.decode(PilotTarget.from_dict(recipe["target"]))
            request = RenderRequest(preset_json=fixed_generator_phases(raw).decode(), performance=Performance.from_dict(recipe["performances"][0]),
                                    renderer_id=codec.contract.renderer_id, tail_seconds=codec.contract.tail_seconds, preroll_frames=32768)
            receipt = source[f"source:{i}:0"]
            if (receipt["request_id"] != request.request_id or receipt["source_sha256"] != sha256(raw).hexdigest()
                    or record["audio_sha256"][0] != receipt["qa"]["audio_float32_sha256"] or receipt["status"] != "accepted"):
                raise ValueError("paired wide-note audio identity differs")
            own = sampler.recipe_at(item["sequence_id"])
            narrow_performance = own.render_request().performance.to_dict()
            narrow_recipes.append(recipe)
            narrow_modes.append(references[own.performance_id])
            predictions = {}
            with torch.no_grad():
                for name, features in (("initialization", item["features"]), ("sound_training", record["features"])):
                    predictions[name] = predicted_targets(model(batch_features([{"features": features}])), codec)[0]
            target = PilotTarget.from_dict(recipe["target"])
            mask = codec.continuous_mask(target)
            cases.append({"state_id": recipe["state_id"], "target": recipe["target"], "active": list(mask),
                          "initialization_pitches": pitches(narrow_performance), "sound_training_pitches": pitches(recipe["performances"][0]),
                          "initialization_performance_id": own.performance_id, "sound_training_performance_id": canonical_sha256(recipe["performances"][0]),
                          "features_sha256": {"initialization": tensor_digest(item["features"]), "sound_training": tensor_digest(record["features"])},
                          "predictions": {name: value.to_dict() for name, value in predictions.items()}})
        metrics = {}
        for name in ("initialization", "sound_training"):
            errors = [abs(row["predictions"][name]["continuous"][i] - row["target"]["continuous"][i])
                      for row in cases for i, active in enumerate(row["active"]) if active]
            metrics[name] = {"source_active_normalized_mae": float(np.mean(errors)),
                             "category_accuracy_by_head": [float(np.mean([row["predictions"][name]["categorical"][i] == row["target"]["categorical"][i] for row in cases])) for i in (0, 1)]}
        assert before == tensor_digest(model.state_dict())
        original = set(old_plan["training_population"]["original_state_ids"])
        retained = [row for row in old_plan["training"] if row["state_id"] in original]
        retained_modes = [i % 4 for i, row in enumerate(old_plan["training"]) if row["state_id"] in original]
        result = {"plan_id": canonical_sha256(plan), "metrics": metrics, "cases": cases,
                  "initialization_pattern_filter_table": pattern_table(narrow_recipes, narrow_modes),
                  "sound_training_pattern_filter_table": pattern_table(old_plan["training"], [i % 4 for i in range(64)]),
                  "retained_original_pattern_filter_table": pattern_table(retained, retained_modes),
                  "development_pattern_filter_table": pattern_table(old_plan["validation"], [i % 4 for i in range(len(old_plan["validation"]))]),
                  "pitch_counts": {name: dict(Counter(pitch for row in cases for pitch in row[name + "_pitches"])) for name in ("initialization", "sound_training")},
                  "elapsed_seconds": perf_counter() - started, "peak_process_tree_rss_bytes": monitor.peak_bytes,
                  "rss_sampling_errors": monitor.errors, "native_requests": 0, "optimizer_updates": 0,
                  "model_unchanged": True, "test_split_used": False, "qualified_for_release": False,
                  "limitations": "input pitch, duration and pattern vary together; parameter errors are not perceptual scores or proof of a single cause"}
        if result["elapsed_seconds"] > 300 or monitor.peak_bytes is None or monitor.peak_bytes > 4 * 1024**3 or monitor.errors:
            raise RuntimeError("input-note diagnosis resource limit failed")
        return result
    finally:
        monitor.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bootstrap", "bootstrap-audit", "parent", "parent-audit", "pool", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = make_plan(args)
    args.output.mkdir(parents=True, exist_ok=True)
    frozen = args.output / "plan.json"
    if frozen.exists() and json.loads(frozen.read_text()) != plan:
        raise ValueError("input-note diagnosis plan changed")
    frozen.write_text(json.dumps(plan, sort_keys=True, indent=2) + "\n")
    if args.execute:
        if (args.output / "summary.json").exists():
            raise ValueError("preserve the completed input-note diagnosis")
        result = diagnose(args, plan)
        (args.output / "summary.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
        print(json.dumps({key: value for key, value in result.items() if key not in {"cases", "pitch_counts"}}), flush=True)
    else:
        print(json.dumps({"executed": False, "plan": plan}), flush=True)

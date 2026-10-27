"""Read performance and source-control information from unchanged basic encoders."""

import argparse
import json
from pathlib import Path
from statistics import mean
import sys
from time import perf_counter

REPOSITORY = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(REPOSITORY / "research/training"), str(REPOSITORY / "research/data_generation")]


def cached_rows(cache, pool):
    from obruxo_data.hashing import canonical_sha256
    from obruxo_data.midi.events import Performance
    from obruxo_data.midi.timing import TempoMap
    from obruxo_training.conditioned_bridge import training_bank
    from obruxo_training.features import tensor_digest
    from obruxo_training.mixed_input_performances import MixedSampler
    sampler = MixedSampler(pool)
    if sampler.sampler_id != cache["sampler_id"] or sampler.observation_contract != cache["observation_contract"]:
        raise ValueError("probe cache changed its original observation schedule")
    bank = training_bank(cache, pool)
    targets = {entry.state_id: entry.target.to_dict() for entry in pool.entries}
    rows, observations = [], {}
    for item in sorted(bank.values(), key=lambda row: row["sequence_id"]):
        recipe = sampler.recipe_at(item["sequence_id"])
        if (recipe.recipe_id != item["recipe_id"] or recipe.state_id != item["state_id"]
                or recipe.performance_id != item["performance_id"] or item["target"] != targets[item["state_id"]]):
            raise ValueError("cached probe labels do not match their authoritative recipe")
        view = {0: 0, 1: 1, 2: 0, 3: 2}[recipe.input_view]
        key = item["state_id"], item["audio_sha256"]
        if key in observations:
            if observations[key] != (view, tensor_digest(item["features"])):
                raise ValueError("duplicate physical recording changed its view or features")
            continue
        performance = Performance.from_dict(json.loads(recipe.request_json)["performance"])
        if canonical_sha256(performance.to_dict()) != item["performance_id"]:
            raise ValueError("MIDI label identity changed")
        spans, timing = performance.note_spans(), TempoMap.from_performance(performance)
        labels = [mean(span.pitch for span in spans), mean(span.velocity for span in spans),
                  mean(float(timing.tick_to_seconds(span.end_tick) - timing.tick_to_seconds(span.start_tick)) for span in spans)]
        rows.append({"state_id": item["state_id"], "view": view, "recipe_id": item["recipe_id"], "performance_id": item["performance_id"],
                     "audio_sha256": item["audio_sha256"], "features_sha256": tensor_digest(item["features"]),
                     "continuous": item["target"]["continuous"], "categorical": item["target"]["categorical"], "performance": labels,
                     "features": item["features"]})
        observations[key] = view, tensor_digest(item["features"])
    if len(rows) != 1212 or len({row["state_id"] for row in rows}) != 404:
        raise ValueError("cached screen lost its full three-view training population")
    return rows


def representations(model, rows):
    import torch
    from obruxo_training.features import tensor_digest
    from obruxo_training.model import batch_features
    before = tensor_digest(model.state_dict())
    values = {name: [] for name in ("timbre", "performance", "mixed_hidden")}
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), 32):
            features = batch_features(rows[start:start + 32])
            branch = model.correction[0]
            timbre, performance = branch.timbre(features["timbre"]), branch.pitch(features["pitch"])
            hidden = model.correction[1](branch.hidden(torch.cat((timbre, performance), 1)))
            for name, value in zip(values, (timbre, performance, hidden), strict=True):
                values[name].append(value.clone())
    if tensor_digest(model.state_dict()) != before or any(parameter.grad is not None for parameter in model.parameters()):
        raise ValueError("probe changed a frozen model or accumulated gradients")
    return {name: torch.cat(parts) for name, parts in values.items()}


def main():
    import torch
    from obruxo_data.hashing import canonical_sha256
    from obruxo_training.audit import write_json
    from obruxo_training.basic_direct import restore_direct
    from obruxo_training.benchmark import MemoryMonitor
    from obruxo_training.category_scale_refresh import cpu_settings
    from obruxo_training.codec import CONTINUOUS, PilotTarget
    from obruxo_training.conditioned_bridge import read
    from obruxo_training.native_fit import digest
    from obruxo_training.pilot import PilotPool
    from obruxo_training.representation_probes import NULL_SEEDS, RIDGE, cached_readouts, cross_performance_retrieval
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fit", "initial", "features", "pool", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    fit, proof = read(args.fit / "plan.json"), read(args.fit / "verification.json")
    paths = {"initial": args.initial, "parameter": args.fit / "diagnostic-parameter.pt", "spectral": args.fit / "diagnostic-spectral.pt"}
    expected = {"initial": fit["initial_file_sha256"], **{arm: proof["evidence_sha256"][f"diagnostic-{arm}.pt"] for arm in ("parameter", "spectral")}}
    if (not proof["verified"] or not proof["offline_training_replay_exact"] or proof["plan_id"] != canonical_sha256(fit)
            or digest(args.features) != fit["input_cache_sha256"] or any(digest(path) != expected[name] for name, path in paths.items())):
        raise ValueError("probes require unchanged independently verified model artifacts")
    source_names = ("reports/scripts/probe_frozen_basic.py", "obruxo_training/representation_probes.py", "obruxo_training/model.py",
                    "obruxo_training/trainable_basic.py", "obruxo_training/mixed_input_performances.py", "obruxo_training/features.py")
    bound = {str(path): digest(path) for path in (*paths.values(), args.features, args.pool, args.fit / "verification.json")}
    sources = {name: __import__("hashlib").sha256((REPOSITORY / "research/training" / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for name in source_names}
    plan = {"format": "cached_frozen_basic_probes_v1", "inputs_sha256": bound, "source_sha256_lf": sources,
            "ridge": RIDGE, "null_seeds": list(NULL_SEEDS), "folds": 3, "fold_unit": "whole_patch_stratified_by_source_categories",
            "observations": 1212, "training_patches": 404, "duplicate_original_rows_removed": 404,
            "limits": {"wall_seconds": 600, "process_tree_bytes": 4 * 1024**3, "archive_bytes": 16 * 1024**2},
            "new_native_requests": 0, "inverse_model_optimizer_updates": 0,
            "limitations": ["All patches were previously seen by the inverse encoder; held-out means new to the linear probe only.",
                            "Cached performances change pitch, timing and velocity together; this is not an independent-factor causal test.",
                            "Linear readability does not prove decoder reliance; low readability does not prove absence of information."]}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "plan.json", plan)
    started, monitor, results = perf_counter(), MemoryMonitor(), {}
    monitor.start()
    try:
        with cpu_settings():
            pool = PilotPool.load(args.pool)
            rows = cached_rows(torch.load(args.features, map_location="cpu", weights_only=True), pool)
            active = torch.tensor([pool.codec.continuous_mask(PilotTarget(tuple(row["continuous"]), tuple(row["categorical"]))) for row in rows])
            write_json(args.output / "observations.json", [{key: value for key, value in row.items() if key != "features"} for row in rows])
            raw = {"raw_mel": torch.stack([row["features"]["timbre"].flatten() for row in rows]),
                   "raw_basic_pitch": torch.stack([row["features"]["pitch"] for row in rows])}
            for model_name, path in paths.items():
                model, _frontend, codec = restore_direct(torch.load(path, map_location="cpu", weights_only=True))
                if codec.to_dict() != pool.codec.to_dict():
                    raise ValueError("probe codec changed")
                branches = representations(model, rows)
                if model_name == "initial":
                    branches.update(raw)
                for branch, features in branches.items():
                    result = {"width": features.shape[1], "mean_feature_variance": float(features.double().var(0, unbiased=False).mean()),
                              "readouts": cached_readouts(features, rows, active), "patch_retrieval": cross_performance_retrieval(features, rows)}
                    results[f"{model_name}/{branch}"] = result
                    write_json(args.output / "progress.json", {"complete": False, "completed_branches": list(results), "elapsed_seconds": perf_counter() - started})
                    print(json.dumps({"branch": f"{model_name}/{branch}", "view_accuracy": result["readouts"]["performance_view"]["accuracy"],
                                      "patch_retrieval": {name: value["accuracy"] for name, value in result["patch_retrieval"].items()}}), flush=True)
                del model, branches
    finally:
        monitor.close()
    resources = {"elapsed_seconds": perf_counter() - started, "peak_process_tree_rss_bytes": monitor.peak_bytes, "rss_sampling_errors": monitor.errors}
    if (any(digest(Path(path)) != value for path, value in bound.items()) or resources["elapsed_seconds"] > 600
            or not monitor.peak_bytes or monitor.peak_bytes > 4 * 1024**3 or monitor.errors):
        raise ValueError("probe changed source evidence or exceeded its declared resource allocation")
    summary = {"plan_id": canonical_sha256(plan), "complete": True, "resources": resources, "results": results,
               "continuous_control_order": [value[0] for value in CONTINUOUS], "input_artifacts_unchanged": True,
               "new_native_requests": 0, "inverse_model_optimizer_updates": 0, "new_validation_or_test_used": False,
               "isolation_established": False, "qualified_for_release": False, "limitations": plan["limitations"]}
    write_json(args.output / "summary.json", summary)
    archive = sum(path.stat().st_size for path in args.output.rglob("*") if path.is_file())
    if archive > plan["limits"]["archive_bytes"]:
        raise ValueError("probe archive exceeded the declared cap")
    write_json(args.output / "progress.json", {"complete": True, "elapsed_seconds": resources["elapsed_seconds"], "archive_bytes": archive})
    print(json.dumps({"complete": True, "resources": resources, "archive_bytes": archive}), flush=True)


if __name__ == "__main__":
    main()

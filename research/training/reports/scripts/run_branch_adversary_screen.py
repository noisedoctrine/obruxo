"""One matched, cached encoder-only test of branch-specific note removal."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

REPOSITORY = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(REPOSITORY / "research/training"), str(REPOSITORY / "research/data_generation")]


def split_states(rows, excluded):
    groups = defaultdict(set)
    for row in rows:
        groups[tuple(row["categorical"])].add(row["state_id"])
    training = set()
    for states in groups.values():
        available = sorted(states - excluded, key=lambda state: hashlib.sha256(("branch_aux_v1:" + state).encode()).hexdigest())
        training.update(available[:2 * len(available) // 3])
    return training


def encode(encoder, rows):
    import torch
    with torch.no_grad():
        return torch.cat([encoder(torch.stack([row["features"]["timbre"] for row in rows[start:start + 64]])) for start in range(0, len(rows), 64)])


def fresh_mlp(features, targets, train, test):
    """New nonlinear adversary, fitted after encoder training with no access to test labels."""
    import torch
    from torch import nn
    from obruxo_training.representation_probes import regression_metrics
    center, scale = features[train].mean(0), features[train].std(0, unbiased=False)
    scale = torch.where(scale > 1e-8, scale, 1.)
    x, query = (features[train] - center) / scale, (features[test] - center) / scale
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(79)
        head = nn.Sequential(nn.Linear(768, 64), nn.GELU(), nn.Linear(64, 2))
        optimizer = torch.optim.Adam(head.parameters(), lr=.001)
        for _ in range(250):
            optimizer.zero_grad(set_to_none=True)
            loss = (head(x) - targets[train]).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
        with torch.no_grad():
            prediction = head(query)
    return {name: regression_metrics(targets[test, index], prediction[:, index], targets[train, index].mean().expand(int(test.sum())))
            for index, name in enumerate(("scaled_mean_midi", "scaled_mean_note_seconds"))}


def score(features, rows, training, active):
    import torch
    from torch.nn import functional as F
    from obruxo_training.branch_adversary import note_targets
    from obruxo_training.representation_probes import NULL_SEEDS, cross_performance_retrieval, regression_metrics, ridge_predict
    train = torch.tensor([row["state_id"] in training for row in rows])
    test = ~train
    performance = note_targets(rows).double()
    categories = torch.tensor([row["categorical"] for row in rows])
    controls = torch.tensor([row["continuous"] for row in rows]).double()
    target = torch.cat((performance, F.one_hot(categories[:, 0], 3), F.one_hot(categories[:, 1], 2)), 1).double()
    predictions = ridge_predict(features[train], target[train], features[test])
    readouts = {name: regression_metrics(performance[test, index], predictions[:, index], performance[train, index].mean().expand(int(test.sum())))
                for index, name in enumerate(("scaled_mean_midi", "scaled_mean_note_seconds"))}
    shuffled = []
    for seed in NULL_SEEDS:
        values, generator = target.clone(), torch.Generator().manual_seed(seed)
        for state in sorted(training):
            indices = [index for index, row in enumerate(rows) if row["state_id"] == state]
            order = torch.randperm(3, generator=generator).tolist()
            values[indices, :2] = target[[indices[index] for index in order], :2]
        shuffled.append(ridge_predict(features[train], values[train, :2], features[test]))
    for index, name in enumerate(readouts):
        readouts[name]["shuffled_training_mse"] = [float((value[:, index] - performance[test, index]).square().mean()) for value in shuffled]
    # Category controls are retained independently of performance-probe success.
    source_categories = {}
    for index, (name, section, classes) in enumerate((("unison", slice(2, 5), 3), ("filter", slice(5, 7), 2))):
        actual = categories[test, index]
        confusion = torch.bincount(actual * classes + predictions[:, section].argmax(1), minlength=classes**2).reshape(classes, classes)
        source_categories[name] = {"accuracy": float((predictions[:, section].argmax(1) == actual).double().mean()),
                                   "balanced_accuracy": float((confusion.diag().double() / confusion.sum(1)).mean()), "confusion": confusion.tolist()}
    source_continuous = []
    for index in range(9):
        selected, queried = train & active[:, index], test & active[:, index]
        value = ridge_predict(features[selected], controls[selected, index:index + 1], features[queried])[:, 0]
        source_continuous.append(regression_metrics(controls[queried, index], value, controls[selected, index].mean().expand(int(queried.sum()))))
    held_rows, held_features = [row for index, row in enumerate(rows) if test[index]], features[test]
    return {"linear_performance": readouts, "fresh_nonlinear_performance": fresh_mlp(features, performance.float(), train, test),
            "source_categories": source_categories, "source_continuous": source_continuous,
            "cross_performance_patch_retrieval": cross_performance_retrieval(held_features, held_rows),
            "normalized_spread": float(F.normalize(held_features, dim=1).std(0, unbiased=False).mean()),
            "zero_vectors": int((held_features.norm(dim=1) < 1e-12).sum()), "training_patches": len(training),
            "probe_test_patches": len({row["state_id"] for row in held_rows}), "probe_test_observations": len(held_rows)}


def direction(control, candidate):
    """Frozen screen rules; these cannot admit an inverse model or select a release."""
    from statistics import mean
    def mse(value):
        return value["rmse"] ** 2
    checks = {}
    for family in ("linear_performance", "fresh_nonlinear_performance"):
        for name in ("scaled_mean_midi", "scaled_mean_note_seconds"):
            checks[family + "/" + name + "/less_readable"] = mse(candidate[family][name]) >= 1.1 * mse(control[family][name])
    checks["source_categories_preserved"] = all(candidate["source_categories"][name]["balanced_accuracy"] >= value["balanced_accuracy"] - .05 for name, value in control["source_categories"].items())
    checks["source_continuous_preserved"] = mean(mse(value) for value in candidate["source_continuous"]) <= 1.1 * mean(mse(value) for value in control["source_continuous"])
    checks["patch_retrieval_preserved"] = mean(value["accuracy"] for value in candidate["cross_performance_patch_retrieval"].values()) >= mean(value["accuracy"] for value in control["cross_performance_patch_retrieval"].values()) - .05
    checks["no_measured_collapse"] = candidate["zero_vectors"] == 0 and candidate["normalized_spread"] >= .5 * control["normalized_spread"]
    return {"checks": checks, "promising_for_sound_trial": all(checks.values()), "inverse_reconstruction_tested": False}


def main():
    import torch
    from obruxo_data.hashing import canonical_sha256, sha256_bytes
    from obruxo_training.audit import write_json
    from obruxo_training.basic_direct import restore_direct
    from obruxo_training.benchmark import MemoryMonitor
    from obruxo_training.branch_adversary import SPEC, initialize, note_targets, pair_indices, restore, snapshot, step, warm_head
    from obruxo_training.category_scale_refresh import cpu_settings
    from obruxo_training.checkpoints import atomic_torch_save
    from obruxo_training.codec import PilotTarget
    from obruxo_training.conditioned_bridge import read
    from obruxo_training.controlled_probes import factor_readouts, patch_readout
    from obruxo_training.features import tensor_digest
    from obruxo_training.native_fit import digest
    from obruxo_training.paired_native_audit import equal_tree
    from obruxo_training.pilot import PilotPool
    from obruxo_training.trainer import capture_rng
    from probe_frozen_basic import cached_rows
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("initial", "features", "pool", "controlled", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    started, monitor = perf_counter(), MemoryMonitor()
    monitor.start()
    try:
        with cpu_settings():
            controlled_plan, controlled = read(args.controlled / "plan.json"), read(args.controlled / "summary.json")
            if (not controlled["complete"] or controlled["plan_id"] != canonical_sha256(controlled_plan)
                    or digest(args.controlled / "features.pt") != controlled["features_sha256"]
                    or digest(args.initial) != controlled_plan["inputs_sha256"][str(args.initial)]):
                raise ValueError("branch screen needs unchanged verified inputs and controlled evidence")
            pool = PilotPool.load(args.pool)
            rows = cached_rows(torch.load(args.features, map_location="cpu", weights_only=True), pool)
            model, frontend, codec = restore_direct(torch.load(args.initial, map_location="cpu", weights_only=True))
            original = tensor_digest(model.state_dict())
            if codec.to_dict() != pool.codec.to_dict() or original != "8ce131675f9216521ee716bce334f449b4524d6c154b1745193488d44c9a3c93":
                raise ValueError("branch screen must use the qualified untrained initializer")
            excluded = set(controlled_plan["patches"])
            training = split_states(rows, excluded)
            held = {row["state_id"] for row in rows} - training
            if len(excluded) != 12 or not excluded.issubset(held) or training & held:
                raise ValueError("all controlled patches must be excluded from auxiliary training")
            sources = ("obruxo_training/branch_adversary.py", "reports/scripts/run_branch_adversary_screen.py", "reports/scripts/probe_frozen_basic.py",
                       "obruxo_training/paired.py", "obruxo_training/representation_probes.py", "obruxo_training/controlled_probes.py")
            bound = {str(path): digest(path) for path in (args.initial, args.features, args.pool, args.controlled / "plan.json", args.controlled / "summary.json", args.controlled / "features.pt")}
            plan = {"format": "cached_branch_adversary_screen_v1", "inputs_sha256": bound, "objective": SPEC,
                    "source_sha256_lf": {name: sha256_bytes((REPOSITORY / "research/training" / name).read_bytes().replace(b"\r\n", b"\n")) for name in sources},
                    "training_states": sorted(training), "probe_test_states": sorted(held), "controlled_states_excluded": sorted(excluded),
                    "encoder_objective": "same_patch_cross_performance_bidirectional_contrastive_temperature_0.1",
                    "arms": {"paired_only": {"maximum_reversal": 0.}, "paired_adversarial": {"maximum_reversal": 1.}},
                    "updates_per_arm": 500, "detached_head_warmup_updates_per_arm": 100, "encoder_learning_rate": .0001, "head_learning_rate": .001,
                    "patch_pairs_per_update": 32, "reversal_ramp_updates": 100, "gradient_norm_cap": 5., "seed": 0,
                    "fresh_probes": {"ridge": .1, "mlp_hidden": 64, "mlp_updates": 250, "mlp_learning_rate": .001, "seed": 79},
                    "direction_rules": {"performance_mse_minimum_ratio_each_fresh_probe_and_factor": 1.1, "source_category_maximum_drop": .05,
                                        "source_continuous_mean_mse_maximum_ratio": 1.1, "patch_retrieval_maximum_drop": .05, "normalized_spread_minimum_ratio": .5},
                    "limits": {"wall_seconds": 1200, "process_tree_bytes": 4 * 1024**3, "archive_bytes": 64 * 1024**2},
                    "new_native_requests": 0, "inverse_sound_fit_updates": 0, "additional_inference_outputs": 0,
                    "limitations": ["Encoder-only directional screen; no spectral inverse training or reconstruction-quality claim.",
                                    "One seed, one fixed adversarial setting; this does not settle other separation methods.",
                                    "All patches were previously seen during baseline encoder initialization; probe-test states are held out from this new training only.",
                                    "Known MIDI summaries supervise only the temporary training head; inference still receives audio only."]}
            args.output.mkdir(parents=True, exist_ok=False)
            write_json(args.output / "plan.json", plan)
            write_json(args.output / "observations.json", [{key: value for key, value in row.items() if key != "features"} for row in rows])
            by_state = defaultdict(list)
            for index, row in enumerate(rows):
                if row["state_id"] in training:
                    by_state[row["state_id"]].append(index)
            mel = torch.stack([row["features"]["timbre"] for row in rows])
            targets = note_targets(rows)
            active = torch.tensor([codec.continuous_mask(PilotTarget(tuple(row["continuous"]), tuple(row["categorical"]))) for row in rows])
            enc = model.correction[0].timbre
            arms = {name: initialize(enc) for name in plan["arms"]}
            if not equal_tree(snapshot(arms["paired_only"]), snapshot(arms["paired_adversarial"])):
                raise ValueError("matched encoder arms have different initialization")
            initial_values = encode(enc, rows)
            generator = torch.Generator().manual_seed(21)
            for _ in range(100):
                indices = pair_indices(by_state, generator)
                for arm in arms.values():
                    warm_head(arm, initial_values[indices], targets[indices])
            if not equal_tree(snapshot(arms["paired_only"]), snapshot(arms["paired_adversarial"])):
                raise ValueError("detached warmup changed matched-arm identity")
            generator, trace, recovery = torch.Generator().manual_seed(0), [], None
            for index in range(500):
                indices = pair_indices(by_state, generator)
                result = {}
                for name, arm in arms.items():
                    coefficient = plan["arms"][name]["maximum_reversal"] * min(1., (index + 1) / 100)
                    result[name] = step(arm, mel[indices], targets[indices], coefficient=coefficient)
                trace.append({"step": index + 1, "indices_sha256": canonical_sha256(indices), "arms": result})
                if index + 1 == 250:
                    saved = {"arms": {name: snapshot(arm) for name, arm in arms.items()}, "sampler_rng": generator.get_state(), "rng": capture_rng("cpu")}
                    atomic_torch_save(args.output / "recovery-prefix.pt", saved)
                if index + 1 == 251:
                    saved = torch.load(args.output / "recovery-prefix.pt", map_location="cpu", weights_only=True)
                    if not equal_tree(saved["rng"], capture_rng("cpu")):
                        raise ValueError("encoder training unexpectedly changed the global RNG")
                    restored_generator = torch.Generator()
                    restored_generator.set_state(saved["sampler_rng"])
                    if indices != pair_indices(by_state, restored_generator):
                        raise ValueError("cold encoder recovery changed the next paired minibatch")
                    for name, arm in arms.items():
                        cold = initialize(enc)
                        restore(cold, saved["arms"][name])
                        actual = step(cold, mel[indices], targets[indices], coefficient=plan["arms"][name]["maximum_reversal"])
                        if actual != result[name] or not equal_tree(snapshot(cold), snapshot(arm)):
                            raise ValueError("cold encoder/heads/Adam recovery is not exact")
                    recovery = {"step": 251, "next_pair_indices_exact": True, "encoders_heads_adam_exact": True, "software_replay_updates_per_arm": 1}
                if (index + 1) % 100 == 0:
                    atomic_torch_save(args.output / "checkpoint.pt", {"plan_id": canonical_sha256(plan), "step": index + 1, "trace": trace,
                                      "arms": {name: snapshot(arm) for name, arm in arms.items()}, "sampler_rng": generator.get_state(), "rng": capture_rng("cpu")})
                    progress = {"stage": "encoder_fit", "updates": index + 1, "planned": 500, "elapsed_seconds": perf_counter() - started, "arms": result}
                    write_json(args.output / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
                if perf_counter() - started > 1200:
                    raise ValueError("cached branch screen exceeded its fixed wall allocation")
            result = {}
            controlled_features = torch.load(args.controlled / "features.pt", map_location="cpu", weights_only=True)
            for name, encoder in {"initial": enc, **{name: arm["encoder"] for name, arm in arms.items()}}.items():
                features = encode(encoder, rows)
                measured = score(features, rows, training, active)
                grid = {}
                for condition, observations in controlled_features.items():
                    values = encode(encoder, observations)
                    grid[condition] = {"factor_readouts": factor_readouts(values, observations), "patch_readout": patch_readout(values, observations)}
                result[name] = {"cached_heldout": measured, "controlled_grid": grid, "encoder_sha256": tensor_digest(encoder.state_dict())}
                if name != "initial":
                    atomic_torch_save(args.output / (name + "-encoder.pt"), {"format": "research_encoder_only_v1", "encoder": encoder.state_dict(), "plan_id": canonical_sha256(plan),
                                      "qualified_for_release": False, "not_an_inverse_model": True})
                print(json.dumps({"scored": name, "linear_performance": measured["linear_performance"], "nonlinear_performance": measured["fresh_nonlinear_performance"]}), flush=True)
            if (tensor_digest(model.state_dict()) != original or any(parameter.grad is not None for parameter in model.parameters())
                    or any(digest(Path(path)) != value for path, value in bound.items()) or recovery is None):
                raise ValueError("screen changed its initializer/input evidence or lost recovery qualification")
            frontend.require_frozen()
            budget = model.require_frozen()
    finally:
        monitor.close()
    resources = {"elapsed_seconds": perf_counter() - started, "peak_process_tree_rss_bytes": monitor.peak_bytes, "rss_sampling_errors": monitor.errors}
    archive = sum(path.stat().st_size for path in args.output.rglob("*") if path.is_file())
    if resources["elapsed_seconds"] > 1200 or not monitor.peak_bytes or monitor.peak_bytes > 4 * 1024**3 or monitor.errors or archive > 62 * 1024**2:
        raise ValueError("cached branch screen exceeded its declared allocation")
    summary = {"plan_id": canonical_sha256(plan), "complete": True, "resources": resources, "archive_bytes_before_summary": archive,
               "encoder_updates_per_arm": 500, "detached_head_warmup_updates_per_arm": 100, "recovery": recovery, "results": result,
               "direction": direction(result["paired_only"]["cached_heldout"], result["paired_adversarial"]["cached_heldout"]),
               "inverse_model_budget_unchanged": budget, "temporary_head_parameters": sum(parameter.numel() for parameter in arms["paired_only"]["head"].parameters()),
               "temporary_head_outputs": 2, "new_native_requests": 0, "inverse_sound_fit_updates": 0, "qualified_for_release": False,
               "input_artifacts_unchanged": True, "new_validation_or_test_used": False, "limitations": plan["limitations"]}
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "progress.json", {"complete": True, "resources": resources})
    print(json.dumps({"complete": True, "resources": resources, "direction": summary["direction"]}), flush=True)


if __name__ == "__main__":
    main()

"""One training-calibrated adversary comparison using the verified paired-only control."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean, median
import sys
from time import perf_counter

REPOSITORY = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(REPOSITORY / "research/training"), str(REPOSITORY / "research/data_generation")]


def main():
    import torch
    from obruxo_data.hashing import canonical_sha256, sha256_bytes
    from obruxo_training.audit import write_json
    from obruxo_training.basic_direct import restore_direct
    from obruxo_training.benchmark import MemoryMonitor
    from obruxo_training.branch_adversary import initialize, note_targets, pair_indices, snapshot, step, warm_head
    from obruxo_training.branch_calibration import SPEC, coefficient, measure
    from obruxo_training.category_scale_refresh import cpu_settings
    from obruxo_training.checkpoints import atomic_torch_save
    from obruxo_training.codec import PilotTarget
    from obruxo_training.conditioned_bridge import read, training_bank
    from obruxo_training.controlled_probes import factor_readouts, patch_readout
    from obruxo_training.features import tensor_digest
    from obruxo_training.native_fit import digest
    from obruxo_training.paired_native_audit import equal_tree
    from obruxo_training.pilot import PilotPool
    from obruxo_training.trainer import capture_rng
    from run_branch_adversary_screen import direction, encode, score
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prior", "prior-audit", "initial", "features", "pool", "controlled", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--mode", choices=("execute", "verify"), default="execute")
    parser.add_argument("--audit-output", type=Path)
    args = parser.parse_args()
    old, old_summary, proof, audit_plan = (read(path) for path in (args.prior / "plan.json", args.prior / "summary.json",
                                                               args.prior_audit / "verification.json", args.prior_audit / "plan.json"))
    if (not proof["verified"] or not proof["encoder_head_adam_rng_replay_exact"] or proof["fresh_cached_probes_reproduced"] != 2
            or old_summary["plan_id"] != canonical_sha256(old) or proof["screen_plan_id"] != old_summary["plan_id"]
            or any(digest(args.prior / name) != value for name, value in audit_plan["evidence_sha256"].items())
            or any(digest(path) != old["inputs_sha256"].get(str(path)) for path in (args.initial, args.features, args.pool, args.controlled / "features.pt"))
            or any(digest(Path(path)) != value for path, value in old["inputs_sha256"].items())
            or any(sha256_bytes((REPOSITORY / "research/training" / name).read_bytes().replace(b"\r\n", b"\n")) != value for name, value in old["source_sha256_lf"].items())):
        raise ValueError("calibration requires unchanged independently replayed parent evidence")
    source_names = ("reports/scripts/run_calibrated_branch_screen.py", "obruxo_training/branch_calibration.py")
    bound = {str(path): digest(path) for path in (args.initial, args.features, args.pool, args.controlled / "features.pt", args.prior / "observations.json",
             args.prior / "paired_only-encoder.pt", args.prior / "summary.json", args.prior_audit / "verification.json", args.prior_audit / "plan.json")}
    plan = {"format": "equal_gradient_branch_adversary_screen_v1", "parent_plan_id": old_summary["plan_id"], "inputs_sha256": bound,
            "source_sha256_lf": {name: sha256_bytes((REPOSITORY / "research/training" / name).read_bytes().replace(b"\r\n", b"\n")) for name in source_names},
            "inherited_source_sha256_lf": old["source_sha256_lf"], "calibration": SPEC, "encoder_updates": 500, "detached_head_warmup_updates": 100,
            "training_states": old["training_states"], "probe_test_states": old["probe_test_states"], "controlled_states_excluded": old["controlled_states_excluded"],
            "comparison": "one_calibrated_candidate_vs_exact_verified_paired_only_control", "control_retrained": False,
            "evaluation": old["fresh_probes"], "direction_rules": old["direction_rules"], "limits": old["limits"],
            "gradient_diagnostics": {"updates": [250, 500], "training_batches_per_checkpoint": 8, "seed": 73},
            "additional_inference_outputs": 0, "new_native_requests": 0, "inverse_sound_fit_updates": 0, "limitations": old["limitations"]}
    verifying = args.mode == "verify"
    wall_limit = 600 if verifying else 1200
    if verifying:
        if args.audit_output is None or read(args.output / "plan.json") != plan:
            raise ValueError("verification needs its own output and the unchanged calibrated plan")
        expected = read(args.output / "summary.json")
        evidence = {str(path.relative_to(args.output)): digest(path) for path in args.output.rglob("*") if path.is_file()}
        args.audit_output.mkdir(parents=True, exist_ok=False)
        write_json(args.audit_output / "plan.json", {"screen_plan_id": canonical_sha256(plan), "evidence_sha256": evidence,
                   "source_sha256_lf": plan["source_sha256_lf"], "new_native_requests": 0, "new_experimental_arms": 0})
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output / "plan.json", plan)
    started, monitor = perf_counter(), MemoryMonitor()
    monitor.start()
    try:
        with cpu_settings():
            pool = PilotPool.load(args.pool)
            bank = training_bank(torch.load(args.features, map_location="cpu", weights_only=True), pool)
            cached = {(row["state_id"], row["audio_sha256"]): row for row in bank.values()}
            rows, training = read(args.prior / "observations.json"), set(plan["training_states"])
            by_state = defaultdict(list)
            for index, row in enumerate(rows):
                item = cached[row["state_id"], row["audio_sha256"]]
                if tensor_digest(item["features"]) != row["features_sha256"] or item["performance_id"] != row["performance_id"]:
                    raise ValueError("calibrated observations differ from verified physical recordings")
                row["features"] = item["features"]
                if row["state_id"] in training:
                    by_state[row["state_id"]].append(index)
            if (len(rows) != 1212 or len(training) != 259 or len(set(plan["probe_test_states"])) != 145
                    or training & set(plan["probe_test_states"]) or not set(plan["controlled_states_excluded"]).issubset(plan["probe_test_states"])):
                raise ValueError("calibrated screen changed its whole-patch split")
            model, frontend, codec = restore_direct(torch.load(args.initial, map_location="cpu", weights_only=True))
            original = tensor_digest(model.state_dict())
            torch.manual_seed(0)
            enc, targets = model.correction[0].timbre, note_targets(rows)
            arm, mel = initialize(enc), torch.stack([row["features"]["timbre"] for row in rows])
            values, generator = encode(enc, rows), torch.Generator().manual_seed(21)
            for _ in range(100):
                indices = pair_indices(by_state, generator)
                warm_head(arm, values[indices], targets[indices])

            def gradient_batch():
                generator = torch.Generator().manual_seed(73)
                return [measure(arm, mel[indices], targets[indices]) for _ in range(8) for indices in [pair_indices(by_state, generator)]]

            measurements = gradient_batch()
            strength = coefficient(measurements)
            calibration = {"coefficient": strength, "training_batches": measurements,
                           "median_scaled_note_to_identity_norm_ratio": median(row["unscaled_note_to_identity_norm_ratio"] * strength for row in measurements),
                           "validation_used": False, "optimizer_updates_during_measurement": 0}
            if verifying:
                if calibration != read(args.output / "calibration.json"):
                    raise ValueError("training-only coefficient does not reproduce exactly")
                saved = torch.load(args.output / "checkpoint.pt", map_location="cpu", weights_only=True)
            else:
                write_json(args.output / "calibration.json", calibration)
                print(json.dumps({"training_only_coefficient": strength, "median_gradient_ratio": calibration["median_scaled_note_to_identity_norm_ratio"]}), flush=True)
            generator, trace, diagnostics = torch.Generator().manual_seed(0), [], []
            for index in range(500):
                indices = pair_indices(by_state, generator)
                result = step(arm, mel[indices], targets[indices], coefficient=strength * min(1., (index + 1) / 100))
                row = {"step": index + 1, "indices_sha256": canonical_sha256(indices), **result}
                if verifying and row != saved["trace"][index]:
                    raise ValueError("calibrated encoder/head/Adam trace does not replay exactly")
                trace.append(row)
                if index + 1 in (250, 500):
                    diagnostic = {"update": index + 1, "batches": gradient_batch()}
                    diagnostic["mean_scaled_note_to_identity_norm_ratio"] = mean(item["unscaled_note_to_identity_norm_ratio"] * strength for item in diagnostic["batches"])
                    diagnostics.append(diagnostic)
                if not verifying and (index + 1) % 100 == 0:
                    atomic_torch_save(args.output / "checkpoint.pt", {"plan_id": canonical_sha256(plan), "step": index + 1, "trace": trace,
                                      "arm": snapshot(arm), "coefficient": strength, "sampler_rng": generator.get_state(), "rng": capture_rng("cpu")})
                    write_json(args.output / "progress.json", {"stage": "fit", "updates": index + 1, "elapsed_seconds": perf_counter() - started})
                    print(json.dumps({"updates": index + 1, "losses": result}), flush=True)
                if perf_counter() - started > wall_limit:
                    raise ValueError("calibrated cached screen exceeded its frozen wall allocation")
            if verifying and (not equal_tree(snapshot(arm), saved["arm"]) or not equal_tree(generator.get_state(), saved["sampler_rng"])
                              or not equal_tree(capture_rng("cpu"), saved["rng"])):
                raise ValueError("calibrated final encoder/head/Adam/RNG differs")
            active = torch.tensor([codec.continuous_mask(PilotTarget(tuple(row["continuous"]), tuple(row["categorical"]))) for row in rows])
            measured = score(encode(arm["encoder"], rows), rows, training, active)
            grid = {}
            for condition, observations in torch.load(args.controlled / "features.pt", map_location="cpu", weights_only=True).items():
                values = encode(arm["encoder"], observations)
                grid[condition] = {"factor_readouts": factor_readouts(values, observations), "patch_readout": patch_readout(values, observations)}
            result = {"cached_heldout": measured, "controlled_grid": grid, "encoder_sha256": tensor_digest(arm["encoder"].state_dict())}
            decision = direction(old_summary["results"]["paired_only"]["cached_heldout"], measured)
            if verifying:
                if result != expected["result"] or diagnostics != read(args.output / "gradient_diagnostics.json") or decision != expected["direction"]:
                    raise ValueError("calibrated fresh probes, diagnostics or frozen decision differs")
                # The reused comparator must still match a freshly fitted readout.
                control = initialize(enc)["encoder"]
                control.load_state_dict(torch.load(args.prior / "paired_only-encoder.pt", map_location="cpu", weights_only=True)["encoder"])
                if score(encode(control, rows), rows, training, active) != old_summary["results"]["paired_only"]["cached_heldout"]:
                    raise ValueError("reused paired-only comparator no longer reproduces")
            else:
                write_json(args.output / "gradient_diagnostics.json", diagnostics)
                atomic_torch_save(args.output / "encoder.pt", {"format": "research_encoder_only_v1", "encoder": arm["encoder"].state_dict(),
                                  "plan_id": canonical_sha256(plan), "qualified_for_release": False, "not_an_inverse_model": True})
            frontend.require_frozen()
            if tensor_digest(model.state_dict()) != original or any(p.grad is not None for p in model.parameters()):
                raise ValueError("calibrated screen modified the original inverse model")
            budget = model.require_frozen()
    finally:
        monitor.close()
    resources = {"elapsed_seconds": perf_counter() - started, "peak_process_tree_rss_bytes": monitor.peak_bytes, "rss_sampling_errors": monitor.errors}
    if (resources["elapsed_seconds"] > wall_limit or not monitor.peak_bytes or monitor.peak_bytes > 4 * 1024**3 or monitor.errors
            or any(digest(Path(path)) != value for path, value in bound.items())):
        raise ValueError("calibrated run exceeded resources or changed its inputs")
    if verifying:
        if any(digest(args.output / name) != value for name, value in evidence.items()):
            raise ValueError("calibrated verification changed its source evidence")
        verification = {"verified": True, "screen_plan_id": canonical_sha256(plan), "calibration_encoder_head_adam_rng_exact": True,
                        "candidate_and_control_fresh_probes_exact": True, "software_replay_encoder_updates": 500,
                        "software_replay_head_warmup_updates": 100, "resources": resources, "new_native_requests": 0, "new_experimental_arms": 0,
                        "promising_for_sound_trial": decision["promising_for_sound_trial"], "qualified_for_release": False}
        write_json(args.audit_output / "verification.json", verification)
        print(json.dumps(verification), flush=True)
    else:
        archive = sum(path.stat().st_size for path in args.output.rglob("*") if path.is_file())
        if archive > 62 * 1024**2:
            raise ValueError("calibrated evidence archive exceeds its fixed allocation")
        summary = {"plan_id": canonical_sha256(plan), "complete": True, "resources": resources, "calibration": calibration,
                   "result": result, "direction": decision, "encoder_updates": 500, "head_warmup_updates": 100, "control_retrained": False,
                   "new_native_requests": 0, "inverse_sound_fit_updates": 0, "inverse_budget_unchanged": budget,
                   "archive_bytes_before_summary": archive, "qualified_for_release": False, "limitations": plan["limitations"]}
        write_json(args.output / "summary.json", summary)
        write_json(args.output / "progress.json", {"complete": True, "resources": resources})
        print(json.dumps({"complete": True, "resources": resources, "coefficient": strength, "direction": decision}), flush=True)


if __name__ == "__main__":
    main()

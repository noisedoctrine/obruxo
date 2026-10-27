"""Replay the cached note-removal screen and inspect its saved gradient magnitudes."""

import argparse
import json
from pathlib import Path
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
    from obruxo_training.branch_adversary import initialize, note_targets, pair_indices, restore, snapshot, step, warm_head
    from obruxo_training.category_scale_refresh import cpu_settings
    from obruxo_training.codec import PilotTarget
    from obruxo_training.conditioned_bridge import read, training_bank
    from obruxo_training.features import tensor_digest
    from obruxo_training.native_fit import digest
    from obruxo_training.paired import identity_terms
    from obruxo_training.paired_native_audit import equal_tree
    from obruxo_training.pilot import PilotPool
    from obruxo_training.trainer import capture_rng, restore_rng
    from run_branch_adversary_screen import direction, encode, score
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("screen", "initial", "features", "pool", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    plan, summary = read(args.screen / "plan.json"), read(args.screen / "summary.json")
    evidence = {str(path.relative_to(args.screen)): digest(path) for path in args.screen.rglob("*") if path.is_file()}
    if (not summary["complete"] or summary["plan_id"] != canonical_sha256(plan)
            or any(digest(Path(path)) != value for path, value in plan["inputs_sha256"].items())
            or any(sha256_bytes((REPOSITORY / "research/training" / name).read_bytes().replace(b"\r\n", b"\n")) != value for name, value in plan["source_sha256_lf"].items())):
        raise ValueError("cached screen source, inputs or plan changed")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "plan.json", {"screen_plan_id": summary["plan_id"], "evidence_sha256": evidence,
               "verifier_sha256_lf": sha256_bytes(Path(__file__).read_bytes().replace(b"\r\n", b"\n")),
               "limits": {"wall_seconds": 600, "process_tree_bytes": 4 * 1024**3}, "new_native_requests": 0})
    started, monitor = perf_counter(), MemoryMonitor()
    monitor.start()
    try:
        with cpu_settings():
            pool = PilotPool.load(args.pool)
            bank = training_bank(torch.load(args.features, map_location="cpu", weights_only=True), pool)
            cached = {(row["state_id"], row["audio_sha256"]): row for row in bank.values()}
            rows = read(args.screen / "observations.json")
            training = set(plan["training_states"])
            if (len(rows) != 1212 or len({row["state_id"] for row in rows}) != 404
                    or training & set(plan["probe_test_states"]) or not set(plan["controlled_states_excluded"]).issubset(plan["probe_test_states"])):
                raise ValueError("screen denominator or held-out patch split changed")
            by_state = {}
            for index, row in enumerate(rows):
                item = cached[row["state_id"], row["audio_sha256"]]
                if (tensor_digest(item["features"]) != row["features_sha256"] or item["performance_id"] != row["performance_id"]
                        or item["target"] != {name: row[name] for name in ("continuous", "categorical")}):
                    raise ValueError("screen row no longer matches its cached physical observation")
                row["features"] = item["features"]
                if row["state_id"] in training:
                    by_state.setdefault(row["state_id"], []).append(index)
            model, frontend, codec = restore_direct(torch.load(args.initial, map_location="cpu", weights_only=True))
            original = tensor_digest(model.state_dict())
            enc, targets = model.correction[0].timbre, note_targets(rows)
            mel = torch.stack([row["features"]["timbre"] for row in rows])
            arms = {name: initialize(enc) for name in plan["arms"]}
            generator, initial_values = torch.Generator().manual_seed(21), encode(enc, rows)
            for _ in range(100):
                indices = pair_indices(by_state, generator)
                for arm in arms.values():
                    warm_head(arm, initial_values[indices], targets[indices])
            final = torch.load(args.screen / "checkpoint.pt", map_location="cpu", weights_only=True)
            prefix = torch.load(args.screen / "recovery-prefix.pt", map_location="cpu", weights_only=True)
            if final["step"] != 500 or len(final["trace"]) != 500 or final["plan_id"] != canonical_sha256(plan):
                raise ValueError("screen did not complete the predeclared update schedule")
            # Module construction uses a fresh process RNG. Training itself uses
            # the explicit paired generator; restore the archived global state
            # before checking that the replay preserves it, as recovery does.
            restore_rng(prefix["rng"], "cpu")
            generator = torch.Generator().manual_seed(0)
            for index, expected in enumerate(final["trace"]):
                indices = pair_indices(by_state, generator)
                actual = {name: step(arm, mel[indices], targets[indices], coefficient=plan["arms"][name]["maximum_reversal"] * min(1., (index + 1) / 100)) for name, arm in arms.items()}
                if {"step": index + 1, "indices_sha256": canonical_sha256(indices), "arms": actual} != expected:
                    raise ValueError("cached screen training trajectory does not replay exactly")
                if index + 1 == 250 and (not equal_tree(generator.get_state(), prefix["sampler_rng"])
                        or not equal_tree(capture_rng("cpu"), prefix["rng"]) or not all(equal_tree(snapshot(arm), prefix["arms"][name]) for name, arm in arms.items())):
                    raise ValueError("screen prefix encoder/head/Adam/RNG differs")
            if (not equal_tree(generator.get_state(), final["sampler_rng"]) or not equal_tree(capture_rng("cpu"), final["rng"])
                    or not all(equal_tree(snapshot(arm), final["arms"][name]) for name, arm in arms.items())):
                raise ValueError("screen final encoder/head/Adam/RNG differs")
            print(json.dumps({"training_replay_exact": True, "updates_per_arm": 500}), flush=True)
            active = torch.tensor([codec.continuous_mask(PilotTarget(tuple(row["continuous"]), tuple(row["categorical"]))) for row in rows])
            for name, arm in arms.items():
                exported = torch.load(args.screen / (name + "-encoder.pt"), map_location="cpu", weights_only=True)
                if not exported["not_an_inverse_model"] or exported["qualified_for_release"] or not equal_tree(exported["encoder"], arm["encoder"].state_dict()):
                    raise ValueError("research encoder export changed or claims inverse qualification")
                if score(encode(arm["encoder"], rows), rows, training, active) != summary["results"][name]["cached_heldout"]:
                    raise ValueError("fresh held-out probe results do not reproduce")
            if direction(summary["results"]["paired_only"]["cached_heldout"], summary["results"]["paired_adversarial"]["cached_heldout"]) != summary["direction"]:
                raise ValueError("screen decision differs from its predeclared rules")
            # Diagnostic only: inspect fixed training minibatches at already saved weights.
            gradient_rows = []
            for checkpoint_name, checkpoint in (("prefix250", prefix), ("final500", final)):
                arm = initialize(enc)
                restore(arm, checkpoint["arms"]["paired_adversarial"])
                generator = torch.Generator().manual_seed(73)
                parameters = tuple(arm["encoder"].parameters())
                for batch in range(8):
                    indices = pair_indices(by_state, generator)
                    value = arm["encoder"](mel[indices])
                    identity = identity_terms(value)["identity_loss"]
                    note = (arm["head"](value, coefficient=1.) - targets[indices]).square().mean()
                    primary = torch.autograd.grad(identity, parameters, retain_graph=True)
                    nuisance = torch.autograd.grad(note, parameters)
                    a, b = torch.cat([value.flatten() for value in primary]).double(), torch.cat([value.flatten() for value in nuisance]).double()
                    gradient_rows.append({"checkpoint": checkpoint_name, "batch": batch, "identity_norm": float(a.norm()), "reversed_note_norm": float(b.norm()),
                                          "reversed_note_to_identity_norm_ratio": float(b.norm() / a.norm()), "gradient_cosine": float(a.dot(b) / (a.norm() * b.norm()))})
            frontend.require_frozen()
            if tensor_digest(model.state_dict()) != original or any(parameter.grad is not None for parameter in model.parameters()):
                raise ValueError("audit modified the original inverse model")
    finally:
        monitor.close()
    resources = {"elapsed_seconds": perf_counter() - started, "peak_process_tree_rss_bytes": monitor.peak_bytes, "rss_sampling_errors": monitor.errors}
    if (resources["elapsed_seconds"] > 600 or not monitor.peak_bytes or monitor.peak_bytes > 4 * 1024**3 or monitor.errors
            or any(digest(args.screen / name) != value for name, value in evidence.items())):
        raise ValueError("screen audit exceeded resources or changed its evidence")
    write_json(args.output / "gradient_magnitudes.json", gradient_rows)
    verified = {"verified": True, "screen_plan_id": summary["plan_id"], "encoder_head_adam_rng_replay_exact": True,
                "software_replay_encoder_updates_per_arm": 500, "software_replay_head_warmup_updates_per_arm": 100,
                "fresh_cached_probes_reproduced": 2, "fixed_gradient_diagnostic_batches": len(gradient_rows), "resources": resources,
                "new_native_requests": 0, "new_experimental_arms": 0, "inverse_sound_fit_updates": 0, "all_evidence_unchanged": True,
                "promising_for_sound_trial": summary["direction"]["promising_for_sound_trial"], "qualified_for_release": False}
    write_json(args.output / "verification.json", verified)
    print(json.dumps(verified), flush=True)


if __name__ == "__main__":
    main()

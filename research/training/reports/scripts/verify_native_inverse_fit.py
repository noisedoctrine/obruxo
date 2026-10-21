"""Replay matched training from logged native losses and audit the saved model decisions."""

import argparse
from collections import defaultdict
import copy
import json
from pathlib import Path
import sys

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


def equal_tree(first, second):
    import torch
    if isinstance(first, torch.Tensor):
        return isinstance(second, torch.Tensor) and torch.equal(first, second)
    if isinstance(first, dict):
        return isinstance(second, dict) and first.keys() == second.keys() and all(equal_tree(first[key], second[key]) for key in first)
    if isinstance(first, (list, tuple)):
        return type(first) is type(second) and len(first) == len(second) and all(equal_tree(a, b) for a, b in zip(first, second, strict=True))
    return first == second


def verify(source, pool, initial_path, retrieval_path, qualification_path):
    import torch
    from obruxo_data.hashing import canonical_sha256, sha256_bytes
    from obruxo_data.midi import Performance
    from obruxo_data.render.base import RenderRequest
    from obruxo_data.render.qa import audio_float32_sha256
    from obruxo_training.checkpoints import load_training_checkpoint
    from obruxo_training.codec import PilotTarget
    from obruxo_training.evaluate import nearest_training_target
    from obruxo_training.features import tensor_digest
    from obruxo_training.native_fit import audit_summary, digest, make_plan, matched_step, predict_candidates
    from obruxo_training.native_fit_report import learning_diagnostics, paired_comparison, public_cases, recovered_timeouts
    from obruxo_training.reproducibility import fixed_generator_phases
    from obruxo_training.spectral import SpectralTargets
    from obruxo_training.trainer import restore_model
    torch.set_num_threads(2)
    frozen = json.loads((source / "plan.json").read_text())
    plan = make_plan(pool, initial_path, retrieval_path, qualification_path, preroll_frames=frozen.get("render_profile", {}).get("preroll_frames", 0))
    summary = json.loads((source / "summary.json").read_text())
    if (plan != json.loads((source / "plan.json").read_text()) or summary["plan"] != plan
            or summary["plan_id"] != canonical_sha256(plan) or not summary["complete"] or summary["failure"] is not None):
        raise ValueError("run is incomplete or its frozen contracts changed")
    saved = torch.load(source / "checkpoint.pt", map_location="cpu", weights_only=True)
    interruptions = []
    for path in sorted((source / "interruptions").glob("*/checkpoint.pt")):
        snapshot = torch.load(path, map_location="cpu", weights_only=True)
        history = json.loads(path.with_name("summary.json").read_text())
        if (snapshot["plan_id"] != canonical_sha256(plan) or history["plan"] != plan
                or not 0 < snapshot["step"] < saved["step"] or snapshot["trace"] != saved["trace"][:snapshot["step"]]
                or history["elapsed_seconds"] > summary["elapsed_seconds"]):
            raise ValueError("interruption snapshot does not belong to the completed run")
        for name in ("renders.jsonl", "measurements.jsonl", "updates.jsonl"):
            if not (source / name).read_bytes().startswith(path.with_name(name).read_bytes()):
                raise ValueError("interruption ledger is not an unchanged prefix of the completed run")
        interruptions.append((path, snapshot, history))
    charges = [json.loads(line) for line in (source / "updates.jsonl").read_text().splitlines()]
    measurements = [json.loads(line) for line in (source / "measurements.jsonl").read_text().splitlines()]
    events = [json.loads(line) for line in (source / "renders.jsonl").read_text().splitlines()]
    attempts = {(row["phase"], row["attempt"]): row for row in events if row["event"] == "attempt"}
    outcomes = {(row["phase"], row["attempt"]): row for row in events if row["event"] == "outcome"}
    if len(attempts) != sum(row["event"] == "attempt" for row in events) or attempts.keys() != outcomes.keys():
        raise ValueError("duplicate or incomplete native attempt ledger")
    timeouts = recovered_timeouts(list(outcomes.values()))
    for key, receipt in outcomes.items():
        if (receipt["request_id"] != attempts[key]["request_id"]
                or receipt["renderer_identity"] != summary["renderer_identity"]):
            raise ValueError("native receipt acceptance or identity changed")

    def require_recorded_receipt(receipt):
        if outcomes.get((receipt["phase"], receipt["attempt"])) != receipt:
            raise ValueError("audit receipt differs from the native outcome ledger")
    initial = torch.load(initial_path, map_location="cpu", weights_only=True)
    incumbent, frontend, codec = restore_model(initial)
    models = {name: copy.deepcopy(incumbent).train() for name in ("parameter", "spectral")}
    optimizers = {name: torch.optim.Adam(model.parameters(), lr=plan["optimizer"]["learning_rate"]) for name, model in models.items()}
    rows = json.loads((source / "records.json").read_text())
    records = {}
    for index, recipe in enumerate(plan["training"]):
        row, path = rows[str(index)], source / "records" / f"{index:03d}.pt"
        if digest(path) != row["sha256"] or row["recipe_id"] != canonical_sha256(recipe):
            raise ValueError("training feature/target record changed")
        record = torch.load(path, map_location="cpu", weights_only=True)
        for view in (0, 1):
            audit = torch.load(source / "audits" / f"fit-{index:03d}-{view}.pt", map_location="cpu", weights_only=True)
            audio = audit["audio"].numpy()
            if audio_float32_sha256(audio) != record["audio_sha256"][view]:
                raise ValueError("training target does not match exact source audit repeat")
            original = [row for row in outcomes.values() if row["phase"] == "training" and row["label"] == f"source:{index}:{view}"
                        and row["qa"]["audio_float32_sha256"] == record["audio_sha256"][view]]
            if not original:
                raise ValueError("training source record has no matching native collection receipt")
            measured = SpectralTargets()(audit["audio"].T[None])
            if not equal_tree(measured, record["targets"][view]):
                raise ValueError("cached training spectra do not match native source audio")
            if view == 0 and not equal_tree(frontend(audio), record["features"]):
                raise ValueError("cached inverse input features do not match native source audio")
        # Keep small input features; full targets are verified, then released.
        records[index] = record["features"]
    by_attempt = defaultdict(list)
    receipts_by_request = defaultdict(list)
    for receipt in outcomes.values():
        receipts_by_request[receipt["request_id"]].append(receipt)

    def require_request(candidate, performance, receipt):
        raw = codec.decode(candidate)
        request = RenderRequest(preset_json=fixed_generator_phases(raw).decode(), performance=performance,
                                renderer_id=codec.contract.renderer_id, tail_seconds=codec.contract.tail_seconds,
                                preroll_frames=plan.get("render_profile", {}).get("preroll_frames", 0))
        if (request.request_id != receipt["request_id"] or sha256_bytes(raw) != receipt["source_sha256"]
                or sha256_bytes(request.preset_json.encode()) != receipt["controlled_sha256"]):
            raise ValueError("receipt does not reconstruct from the claimed patch/MIDI")

    for measured in measurements:
        step = measured["step"] - 1
        update = plan["updates"][step]
        performance = Performance.from_dict(plan["training"][update["state"]]["performances"][update["view"]])
        receipts = [row for row in receipts_by_request[measured["request_id"]]
                    if row["status"] == "accepted" and row["label"] == f"gradient:{step}:{measured['query']}"
                    and row["qa"]["audio_float32_sha256"] == measured["audio_sha256"]]
        if not receipts:
            raise ValueError("a logged training loss has no matching native receipt")
        require_request(PilotTarget.from_dict(measured["candidate"]), performance, receipts[0])
        by_attempt[measured["optimizer_attempt"]].append(measured)
    if (saved["plan_id"] != canonical_sha256(plan) or saved["step"] != 128 or len(saved["trace"]) != 128
            or len(charges) > plan["limits"]["optimizer_attempts"] or any(row["attempt"] != i for i, row in enumerate(charges))):
        raise ValueError("optimizer checkpoint or attempt count changed")
    torch.manual_seed(plan["optimizer"]["seed"])
    for step, committed in enumerate(saved["trace"]):
        update = plan["updates"][step]
        charge = {key: committed[key] for key in ("attempt", "step", "state", "view")}
        if (charges[committed["attempt"]] != charge or charge["step"] != step + 1
                or charge["state"] != update["state"] or charge["view"] != update["view"]):
            raise ValueError("committed update does not match the frozen selection or charge")
        point = by_attempt[committed["attempt"]]
        cursor = 0

        def evaluate(_index, candidate):
            nonlocal cursor
            measured = point[cursor]
            if measured["query"] != cursor or candidate.to_dict() != measured["candidate"]:
                raise ValueError("offline gradient requested a different patch")
            cursor += 1
            return measured["loss"]

        result = matched_step(models, optimizers, records[update["state"]], PilotTarget.from_dict(plan["training"][update["state"]]["target"]),
                              codec, evaluate, step, len(plan["updates"]))
        if cursor != len(point) or any(result[key] != committed[key] for key in result):
            raise ValueError("offline numerical update differs from training")
        for _path, snapshot, _history in interruptions:
            if snapshot["step"] != step + 1:
                continue
            if (not torch.equal(torch.get_rng_state(), snapshot["rng"]["torch"])
                    or any(not equal_tree(model.state_dict(), snapshot["models"][name])
                           or not equal_tree(optimizers[name].state_dict(), snapshot["optimizers"][name]) for name, model in models.items())):
                raise ValueError("replayed interruption checkpoint differs from its archived models/optimizers/RNG")
    for name, model in models.items():
        if not equal_tree(model.state_dict(), saved["models"][name]) or not equal_tree(optimizers[name].state_dict(), saved["optimizers"][name]):
            raise ValueError("replayed model/optimizer differs from the committed final checkpoint")
        exported = torch.load(source / f"diagnostic-{name}.pt", map_location="cpu", weights_only=True)
        if (not equal_tree(exported["model"], model.state_dict()) or exported["training_plan_id"] != canonical_sha256(plan)
                or exported["qualified_for_release"] or exported["codec"] != codec.to_dict()
                or exported.get("render_profile", {}) != plan.get("render_profile", {})):
            raise ValueError("diagnostic inference export differs from its actual trained model")
    if not torch.equal(torch.get_rng_state(), saved["rng"]["torch"]):
        raise ValueError("replayed Gumbel RNG differs from the committed checkpoint")
    all_models = {"initial": incumbent.eval(), **{name: model.eval() for name, model in models.items()}}
    retrieval = load_training_checkpoint(retrieval_path)["replay"]
    fit_cases = json.loads((source / "fit_cases.json").read_text())
    validation_cases = json.loads((source / "validation_cases.json").read_text()) if (source / "validation_cases.json").exists() else []
    for split, cases in (("training", fit_cases), ("validation", validation_cases)):
        candidates_by_state = {}
        for case in cases:
            recipe = next(row for row in plan[split] if row["state_id"] == case["state_id"])
            index, view = plan[split].index(recipe), case["view"]
            audit = torch.load(source / "audits" / f"{case['label']}.pt", map_location="cpu", weights_only=True)
            if audit["rows"] != case["rows"] or audit["source_receipt"] != case["source_receipt"] or not audit["repeat_exact"]:
                raise ValueError("saved native audit cases changed")
            if audio_float32_sha256(audit["audio"].numpy()) != audit["source_receipt"]["qa"]["audio_float32_sha256"]:
                raise ValueError("saved native audit source audio changed")
            if split == "training":
                candidates = predict_candidates(all_models, codec, records[index])
            elif view == 0:
                features = frontend(audit["audio"].numpy())
                candidates = {**predict_candidates(all_models, codec, features), "template": codec.default_target(),
                              "retrieval": nearest_training_target(features, retrieval)}
                candidates_by_state[recipe["state_id"]] = candidates
            else:
                candidates = candidates_by_state[recipe["state_id"]]
            performance = Performance.from_dict(recipe["performances"][view])
            require_recorded_receipt(audit["source_receipt"])
            require_request(PilotTarget.from_dict(recipe["target"]), performance, audit["source_receipt"])
            if split == "validation":
                repeat = audit["repeat_receipt"]
                require_recorded_receipt(repeat)
                require_request(PilotTarget.from_dict(recipe["target"]), performance, repeat)
                if repeat["qa"]["audio_float32_sha256"] != audit["source_receipt"]["qa"]["audio_float32_sha256"]:
                    raise ValueError("validation repeat hash changed")
            for name, candidate in candidates.items():
                if case["rows"][name]["candidate"] != candidate.to_dict():
                    raise ValueError("hard native audit patch differs from actual model output")
                require_recorded_receipt(case["rows"][name]["receipt"])
                require_request(candidate, performance, case["rows"][name]["receipt"])
    fit_result = audit_summary(fit_cases, plan, validation=False)
    validation_result = audit_summary(validation_cases, plan, validation=True) if fit_result["pass"] else None
    if fit_result != summary["fit_result"] or validation_result != summary["validation_result"]:
        raise ValueError("native quality gates changed")
    counts = {phase: sum(key[0] == phase for key in attempts) for phase in ("training", "qualification")}
    archive = sum(path.stat().st_size for path in source.rglob("*") if path.is_file())
    limits = plan["limits"]
    histories = [summary, *(history for _path, _snapshot, history in interruptions)]
    peak = max(history["peak_process_tree_rss_bytes"] for history in histories)
    sampling_errors = sum(history["rss_sampling_errors"] for history in histories)
    resources = (sampling_errors == 0 and 0 < peak <= limits["process_tree_bytes"]
                 and summary["elapsed_seconds"] <= limits["wall_seconds"] and archive <= limits["archive_bytes"]
                 and all(counts[phase] <= limits[f"{phase}_attempts"] for phase in counts))
    accepted = all(row["status"] == "accepted" for row in outcomes.values())
    passed = bool(fit_result["pass"] and validation_result and validation_result["pass"] and resources and accepted)
    if (passed != summary["matched_comparison_passed"] or resources != summary["within_resource_limits"]
            or counts != summary["render_attempts"] or accepted != summary["all_native_accepted"] or not summary["incumbent_artifacts_unchanged"]):
        raise ValueError("final qualification/resource decision changed")
    return {**summary, "retained_archive_bytes": archive, "optimizer_attempts": len(charges),
            "peak_process_tree_rss_bytes": peak, "rss_sampling_errors": sampling_errors,
            "last_segment_peak_process_tree_rss_bytes": summary["peak_process_tree_rss_bytes"],
            "recovered_infrastructure_timeouts": timeouts,
            "archived_interruption_checkpoints_replayed": [snapshot["step"] for _path, snapshot, _history in interruptions],
            "interruption_evidence_sha256": {str(path.parent.relative_to(source)): {name: digest(path.with_name(name)) for name in
                ("summary.json", "checkpoint.pt", "renders.jsonl", "updates.jsonl", "measurements.jsonl")} for path, _snapshot, _history in interruptions},
            "offline_training_replay_exact": True, "cached_targets_and_inputs_rescored": True,
            "hard_audit_patches_reconstructed": True, "candidate_audit_audio_rescored": False,
            "fit_cases": public_cases(fit_cases), "validation_cases": public_cases(validation_cases),
            "learning_diagnostics": learning_diagnostics(saved["trace"], fit_cases, plan["codec"]["categorical"]),
            "descriptive_validation_comparison": paired_comparison(validation_cases) if validation_cases else None,
            "final_model_sha256": {name: tensor_digest(model.state_dict()) for name, model in models.items()},
            "evidence_sha256": {name: digest(source / name) for name in ("summary.json", "checkpoint.pt", "records.json", "updates.jsonl", "measurements.jsonl", "renders.jsonl")}}


def main():
    from obruxo_training.audit import write_json
    from obruxo_training.pilot import PilotPool
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "pool", "initial", "retrieval", "qualification", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to replace verified evidence")
    result = verify(args.source, PilotPool.load(args.pool), args.initial, args.retrieval, args.qualification)
    write_json(args.output, result)
    print(json.dumps({"offline_training_replay_exact": True, "matched_comparison_passed": result["matched_comparison_passed"]}))


if __name__ == "__main__":
    main()

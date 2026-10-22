"""Verify a stopped native inverse run without restarting or re-rendering it."""

import argparse
from collections import defaultdict
import copy
import json
from pathlib import Path
import sys

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]

from verify_native_inverse_fit import equal_tree  # noqa: E402


def audit(source, pool, initial_path, retrieval_path, qualification_path, *, records_root=None, calibration_path=None):
    import torch
    from obruxo_data.hashing import canonical_sha256, sha256_bytes
    from obruxo_data.midi import Performance
    from obruxo_data.render.base import RenderRequest
    from obruxo_training.codec import CATEGORICAL, PilotTarget
    from obruxo_training.category_signal import scaled_logits
    from obruxo_training.features import tensor_digest, validate_features
    from obruxo_training.native_fit import digest, make_plan, matched_step
    from obruxo_training.native_inverse import native_reconstruction_loss
    from obruxo_training.native_fit_report import terminal_interruption
    from obruxo_training.reproducibility import fixed_generator_phases
    from obruxo_training.spectral_controls import BasicControlAdapter, temperature_at
    from obruxo_training.trainer import restore_model
    torch.set_num_threads(2)
    records_root = source if records_root is None else records_root
    frozen = json.loads((source / "plan.json").read_text())
    plan = make_plan(pool, initial_path, retrieval_path, qualification_path,
                     preroll_frames=frozen.get("render_profile", {}).get("preroll_frames", 0), category_calibration_path=calibration_path)
    summary = json.loads((source / "summary.json").read_text())
    if (plan != json.loads((source / "plan.json").read_text()) or summary["plan"] != plan
            or summary["plan_id"] != canonical_sha256(plan) or summary["complete"] or not summary["failure"]):
        raise ValueError("expected an unchanged, terminal interrupted run")
    saved = torch.load(source / "checkpoint.pt", map_location="cpu", weights_only=True)
    if (saved["plan_id"] != canonical_sha256(plan) or saved["step"] != len(saved["trace"])
            or saved.get("category_training") != plan.get("category_training")):
        raise ValueError("committed checkpoint contract changed")
    charges = [json.loads(line) for line in (source / "updates.jsonl").read_text().splitlines()]
    measured = [json.loads(line) for line in (source / "measurements.jsonl").read_text().splitlines()]
    events = [json.loads(line) for line in (source / "renders.jsonl").read_text().splitlines()]
    attempts = [row for row in events if row["event"] == "attempt"]
    outcomes = [row for row in events if row["event"] == "outcome"]
    if len(attempts) != len(outcomes):
        raise ValueError("incomplete native attempt/outcome ledger")
    for index, (attempt, outcome) in enumerate(zip(attempts, outcomes, strict=True)):
        if (attempt["attempt"] != index or outcome["attempt"] != index or attempt["request_id"] != outcome["request_id"]
                or outcome["renderer_identity"] != summary["renderer_identity"]):
            raise ValueError("native attempt identity changed")
    interrupted = terminal_interruption(outcomes)
    if any(row["attempt"] != i for i, row in enumerate(charges)):
        raise ValueError("optimizer charge sequence changed")
    incumbent, frontend, codec = restore_model(torch.load(initial_path, map_location="cpu", weights_only=True))
    models = {name: copy.deepcopy(incumbent).train() for name in ("parameter", "spectral")}
    optimizers = {name: torch.optim.Adam(model.parameters(), lr=plan["optimizer"]["learning_rate"]) for name, model in models.items()}
    manifest, features = json.loads((source / "records.json").read_text()), {}
    for index, recipe in enumerate(plan["training"]):
        row, path = manifest[str(index)], records_root / "records" / f"{index:03d}.pt"
        if row["recipe_id"] != canonical_sha256(recipe) or row["sha256"] != digest(path):
            raise ValueError("cached training record changed")
        record = torch.load(path, map_location="cpu", weights_only=True)
        validate_features(record["features"])
        features[index] = record["features"]
    by_attempt, by_request = defaultdict(list), defaultdict(list)
    for row in measured:
        by_attempt[row["optimizer_attempt"]].append(row)
    for row in outcomes:
        by_request[row["request_id"]].append(row)

    def require_request(candidate, performance, receipt):
        raw = codec.decode(candidate)
        request = RenderRequest(preset_json=fixed_generator_phases(raw).decode(), performance=performance,
                                renderer_id=codec.contract.renderer_id, tail_seconds=codec.contract.tail_seconds,
                                preroll_frames=plan.get("render_profile", {}).get("preroll_frames", 0))
        if (request.request_id != receipt["request_id"] or sha256_bytes(raw) != receipt["source_sha256"]
                or sha256_bytes(request.preset_json.encode()) != receipt["controlled_sha256"]):
            raise ValueError("native receipt cannot be reconstructed")

    for row in measured:
        update = plan["updates"][row["step"] - 1]
        performance = Performance.from_dict(plan["training"][update["state"]]["performances"][update["view"]])
        receipts = [receipt for receipt in by_request[row["request_id"]] if receipt["status"] == "accepted"
                    and receipt["label"] == f"gradient:{row['step'] - 1}:{row['query']}"
                    and receipt["qa"]["audio_float32_sha256"] == row["audio_sha256"]]
        if not receipts:
            raise ValueError("measured loss lacks an accepted native receipt")
        require_request(PilotTarget.from_dict(row["candidate"]), performance, receipts[0])
    torch.manual_seed(plan["optimizer"]["seed"])
    for step, trace in enumerate(saved["trace"]):
        update = plan["updates"][step]
        charge = {key: trace[key] for key in ("attempt", "step", "state", "view")}
        if (charge != charges[trace["attempt"]] or charge["step"] != step + 1
                or charge["state"] != update["state"] or charge["view"] != update["view"]):
            raise ValueError("committed point differs from frozen order or charge")
        point, cursor = by_attempt[trace["attempt"]], 0
        def evaluate(_index, candidate):
            nonlocal cursor
            row = point[cursor]
            if row["query"] != cursor or row["candidate"] != candidate.to_dict():
                raise ValueError("replayed point requests a different native patch")
            cursor += 1
            return row["loss"]
        result = matched_step(models, optimizers, features[update["state"]], PilotTarget.from_dict(plan["training"][update["state"]]["target"]),
                              codec, evaluate, step, len(plan["updates"]), category_training=plan.get("category_training"))
        if cursor != len(point) or any(result[key] != trace[key] for key in result):
            raise ValueError("native optimizer prefix cannot be replayed exactly")
    if not torch.equal(torch.get_rng_state(), saved["rng"]["torch"]):
        raise ValueError("prefix Gumbel RNG changed")
    for name, model in models.items():
        if not equal_tree(model.state_dict(), saved["models"][name]) or not equal_tree(optimizers[name].state_dict(), saved["optimizers"][name]):
            raise ValueError("prefix model or optimizer differs from committed checkpoint")
    hashes = {name: tensor_digest(model.state_dict()) for name, model in models.items()}
    pending = charges[-1]
    if pending["step"] != saved["step"] + 1:
        raise ValueError("terminal optimizer attempt does not follow the checkpoint")
    update = plan["updates"][saved["step"]]
    performance = Performance.from_dict(plan["training"][update["state"]]["performances"][update["view"]])
    prediction = models["spectral"]( {"timbre": features[update["state"]]["timbre"][None], "pitch": features[update["state"]]["pitch"][None]} )
    if plan.get("category_training"):
        prediction = {**prediction, "categorical": scaled_logits(prediction["categorical"], plan["category_training"]["scales"])}
    prefix, cursor, failing = by_attempt[pending["attempt"]], 0, None
    class FoundRejection(Exception):
        pass
    def pending_evaluate(_index, candidate):
        nonlocal cursor, failing
        if cursor < len(prefix):
            row = prefix[cursor]
            if row["candidate"] != candidate.to_dict():
                raise ValueError("pending accepted probe differs from the checkpoint")
            cursor += 1
            return row["loss"]
        require_request(candidate, performance, interrupted)
        failing = candidate
        raise FoundRejection()
    try:
        native_reconstruction_loss(prediction, BasicControlAdapter(codec), pending_evaluate,
                                   temperature=temperature_at(saved["step"], len(plan["updates"])))
    except FoundRejection:
        pass
    if failing is None:
        raise ValueError("rejected probe was not reconstructed")
    anchor = PilotTarget(tuple(prediction["continuous"].detach()[0].tolist()), tuple(prefix[0]["candidate"]["categorical"]))
    archive = sum(path.stat().st_size for path in source.rglob("*") if path.is_file())
    probe = {"candidate": failing.to_dict(), "actual_categories": {name: options[index]
             for (name, options), index in zip(CATEGORICAL, failing.categorical, strict=True)},
             "receipt": interrupted, "optimizer_step": pending["step"], "state_index": update["state"], "view": update["view"]}
    return {"format": "interrupted_native_inverse_prefix_audit_v2", "plan_id": canonical_sha256(plan),
            "initial_model_sha256": plan["initial_model_sha256"], "frozen_source": plan["source"],
            "committed_updates_per_arm": saved["step"], "optimizer_attempts": len(charges),
            "summary_reported_committed_updates_per_arm": summary["committed_updates_per_arm"],
            "offline_prefix_replay_exact": True, "both_optimizer_states_exact": True, "gumbel_rng_exact": True,
            "cached_records_hashed": True, "cached_audio_targets_independently_rescored": False,
            "accepted_native_requests": len(outcomes) - 1,
            "rejected_native_requests": int(interrupted["status"] == "rejected"),
            "native_error_requests": int(interrupted["status"] == "error"),
            "total_native_attempts": len(attempts), "terminal_probe_reconstructed": True, "interrupted_probe": probe,
            "rejected_probe_reconstructed": interrupted["status"] == "rejected",
            "rejected_probe": probe if interrupted["status"] == "rejected" else None,
            "pending_anchor": anchor.to_dict(), "model_sha256": hashes,
            "elapsed_seconds": summary["elapsed_seconds"], "peak_process_tree_rss_bytes": summary["peak_process_tree_rss_bytes"],
            "rss_sampling_errors": summary["rss_sampling_errors"], "retained_archive_bytes": archive,
            "within_resource_limits": summary["within_resource_limits"], "quality_evaluation_performed": False,
            "matched_comparison_passed": False, "qualified_for_release": False, "test_split_used": False,
            "evidence_sha256": {name: digest(source / name) for name in ("plan.json", "summary.json", "checkpoint.pt", "records.json", "updates.jsonl", "measurements.jsonl", "renders.jsonl")}}


def main():
    from obruxo_training.audit import write_json
    from obruxo_training.pilot import PilotPool
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "pool", "initial", "retrieval", "qualification", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--records-root", type=Path)
    parser.add_argument("--category-calibration", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to replace audited evidence")
    result = audit(args.source, PilotPool.load(args.pool), args.initial, args.retrieval, args.qualification,
                   records_root=args.records_root, calibration_path=args.category_calibration)
    write_json(args.output, result)
    print(json.dumps({"committed_updates_per_arm": result["committed_updates_per_arm"], "offline_prefix_replay_exact": True,
                      "terminal_probe_reconstructed": True, "terminal_status": result["interrupted_probe"]["receipt"]["status"]}))


if __name__ == "__main__":
    main()

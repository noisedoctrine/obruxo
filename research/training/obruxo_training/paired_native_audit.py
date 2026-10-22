"""Replay paired optimization and rescore every retained hard-model comparison."""

import copy
import json
import shutil
from time import perf_counter, time

import torch

from obruxo_data.hashing import canonical_sha256, sha256_bytes
from obruxo_data.midi import Performance
from obruxo_data.render.base import RenderRequest
from obruxo_data.render.qa import audio_float32_sha256

from . import category_scale_refresh as refresh
from .audit import write_json
from .benchmark import MemoryMonitor
from .checkpoints import load_training_checkpoint
from .codec import PilotTarget
from .evaluate import nearest_training_target
from .features import tensor_digest
from .native_fit import audit_summary, digest, predict_candidates, record_features
from .native_fit_report import development_comparison, public_cases
from .paired_native_evidence import native_ledger, rescore_waveform
from .paired_native_fit import make_plan
from .paired_native_step import matched_pair_step
from .paired_native_training import CHECKPOINT_FORMAT
from .reproducibility import fixed_generator_phases
from .spectral import SpectralTargets
from .trainer import capture_rng, restore_model, restore_rng


def equal_tree(first, second):
    if torch.is_tensor(first):
        return torch.is_tensor(second) and first.dtype == second.dtype and torch.equal(first, second)
    if isinstance(first, dict):
        return isinstance(second, dict) and first.keys() == second.keys() and all(equal_tree(first[key], second[key]) for key in first)
    if isinstance(first, (list, tuple)):
        return type(first) is type(second) and len(first) == len(second) and all(equal_tree(a, b) for a, b in zip(first, second, strict=True))
    return first == second


class IncompleteMeasurement(Exception):
    pass


def state_signature(value):
    if torch.is_tensor(value):
        return {"tensor_sha256": tensor_digest({"value": value}), "dtype": str(value.dtype), "shape": list(value.shape)}
    if isinstance(value, dict):
        return {str(key): state_signature(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [state_signature(item) for item in value]
    return value


def training_signature(models, optimizers):
    return canonical_sha256(state_signature({"models": {name: model.state_dict() for name, model in models.items()},
        "optimizers": {name: optimizer.state_dict() for name, optimizer in optimizers.items()}, "rng": capture_rng("cpu")}))


def verify(source, pool, bootstrap_root, bootstrap_audit_path, qualification_path, cache_path, choice_path, direction_source, *, prefix_plan=None):
    prefix = prefix_plan is not None
    plan = prefix_plan if prefix else make_plan(pool, bootstrap_root, bootstrap_audit_path, qualification_path, cache_path, choice_path, direction_source)
    summary = ({"elapsed_seconds": time() - json.loads((source / "run.json").read_text())["started_unix"]} if prefix else
               json.loads((source / "summary.json").read_text()))
    if (json.loads((source / "plan.json").read_text()) != plan or (not prefix and (summary["plan"] != plan
            or summary["plan_id"] != canonical_sha256(plan) or not summary["complete"] or summary["failure"] is not None))):
        raise ValueError("paired fit is incomplete or its frozen inputs/source changed")
    saved = torch.load(source / "checkpoint.pt", map_location="cpu", weights_only=True)
    if (saved["format"] != CHECKPOINT_FORMAT or saved["plan_id"] != canonical_sha256(plan)
            or not 1 <= saved["step"] <= len(plan["updates"]) or (not prefix and saved["step"] != len(plan["updates"]))
            or saved["production_training"] != plan["production_training"] or saved["step"] != len(saved["trace"])):
        raise ValueError("paired checkpoint contract differs from the fit plan")
    events = [json.loads(line) for line in (source / "renders.jsonl").read_text().splitlines()]
    attempts, outcomes, recovered = native_ledger(events, plan["expected_renderer_identity"])
    used = set()
    def require_receipt(receipt):
        bare = {key: value for key, value in receipt.items() if key != "retained_audio"}
        if bare != outcomes.get((receipt["phase"], receipt["attempt"])) or receipt["status"] != "accepted":
            raise ValueError("paired evidence cites an unrecorded native receipt")
    def require_request(candidate, performance, receipt):
        request = RenderRequest(preset_json=fixed_generator_phases(pool.codec.decode(candidate)).decode(), performance=performance,
                                renderer_id=pool.codec.contract.renderer_id, tail_seconds=pool.codec.contract.tail_seconds,
                                preroll_frames=plan["render_profile"]["preroll_frames"])
        if (receipt["request_id"] != request.request_id or receipt["source_sha256"] != sha256_bytes(pool.codec.decode(candidate))
                or receipt["controlled_sha256"] != sha256_bytes(request.preset_json.encode())):
            raise ValueError("paired native receipt differs from its patch or performance")
        identity = ("phase", "label", "request_id", "source_sha256", "controlled_sha256")
        for key, previous in outcomes.items():
            if previous["status"] == "accepted" and all(previous[name] == receipt[name] for name in identity):
                if previous["qa"]["audio_float32_sha256"] != receipt["qa"]["audio_float32_sha256"]:
                    raise ValueError("an identical paired request changed audio across recovery")
                used.add(key)
    initial = torch.load(bootstrap_root / "fit-000/model.pt", map_location="cpu", weights_only=True)
    incumbent, frontend, codec = restore_model(initial)
    models = {name: copy.deepcopy(incumbent).train() for name in ("parameter", "spectral")}
    optimizers = {name: torch.optim.Adam(model.parameters(), lr=plan["optimizer"]["learning_rate"]) for name, model in models.items()}
    torch.manual_seed(plan["optimizer"]["seed"])
    rows = json.loads((source / "records.json").read_text())
    records = {}
    required = {str(row["state"]) for update in plan["updates"][:saved["step"]] for row in update["examples"]}
    allowed = {str(index) for index in range(len(plan["training"]))}
    if not required.issubset(rows) or set(rows) - allowed or (not prefix and set(rows) != allowed):
        raise ValueError("paired source record coverage changed")
    for index in sorted(map(int, rows)):
        recipe = plan["training"][index]
        path = source / "records" / f"{index:03d}.pt"
        if rows[str(index)]["file"] != path.name or digest(path) != rows[str(index)]["sha256"]:
            raise ValueError("paired source record file changed")
        record = torch.load(path, map_location="cpu", weights_only=True)
        if (record["recipe_id"] != canonical_sha256(recipe) or rows[str(index)]["recipe_id"] != record["recipe_id"]
                or record["render_profile"] != plan["render_profile"]):
            raise ValueError("paired source record identity changed")
        if record["audio_sha256"] != recipe["source_audio_sha256"] or len(record["features"]) != 3 or len(record["targets"]) != 3:
            raise ValueError("paired source views changed")
        for view in range(3):
            if tensor_digest(record_features(record, view)) != recipe["source_feature_sha256"][view]:
                raise ValueError("paired cached input feature changed")
            matches = [receipt for receipt in outcomes.values() if receipt["status"] == "accepted" and receipt["phase"] == "training"
                       and receipt["label"] == f"source:{index}:{view}" and receipt["qa"]["audio_float32_sha256"] == record["audio_sha256"][view]]
            if not matches:
                raise ValueError("paired source feature has no accepted native origin")
            for receipt in matches:
                require_request(PilotTarget.from_dict(recipe["target"]), Performance.from_dict(recipe["performances"][view]), receipt)
        records[index] = record
    charges = [json.loads(line) for line in (source / "updates.jsonl").read_text().splitlines()]
    measured = [json.loads(line) for line in (source / "measurements.jsonl").read_text().splitlines()]
    by_attempt = {}
    for row in measured:
        by_attempt.setdefault(row["optimizer_attempt"], []).append(row)
    committed = {row["attempt"]: row for row in saved["trace"]}
    if len(committed) != len(saved["trace"]) or set(by_attempt) - set(range(len(charges))):
        raise ValueError("paired measurements/committed attempts changed")
    interrupted = []
    prefix_evidence = json.loads((source / "prefix-verification.json").read_text()) if not prefix and (source / "prefix-verification.json").exists() else None
    if prefix_evidence:
        if (prefix_evidence["checkpoint_file"] != "prefix-checkpoint.pt"
                or digest(source / "prefix-checkpoint.pt") != prefix_evidence["checkpoint_file_sha256"]):
            raise ValueError("paired early replay checkpoint changed")
        prefix_snapshot = torch.load(source / "prefix-checkpoint.pt", map_location="cpu", weights_only=True)
        if (prefix_snapshot["plan_id"] != canonical_sha256(plan) or prefix_snapshot["step"] != prefix_evidence["replayed_updates"]
                or prefix_snapshot["trace"] != saved["trace"][:prefix_snapshot["step"]]):
            raise ValueError("paired early replay checkpoint is not a committed prefix")
    snapshots = []
    for directory in sorted((source / "interruptions").glob("*")):
        manifest = json.loads((directory / "evidence.json").read_text())
        if any(digest(directory / name) != value for name, value in manifest.items()):
            raise ValueError("paired interruption evidence changed")
        history = json.loads((directory / "summary.json").read_text())
        if history["plan_id"] != canonical_sha256(plan) or history["elapsed_seconds"] > summary["elapsed_seconds"]:
            raise ValueError("paired interruption belongs to a different run")
        for name in ("renders.jsonl", "updates.jsonl", "measurements.jsonl"):
            if name in manifest and not (source / name).read_bytes().startswith((directory / name).read_bytes()):
                raise ValueError("paired interruption is not an unchanged ledger prefix")
        if "checkpoint.pt" in manifest:
            try:
                snapshot = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=True)
            except (RuntimeError, EOFError, OSError):
                snapshot = torch.load(directory / "checkpoint.previous.pt", map_location="cpu", weights_only=True)
            if snapshot["trace"] != saved["trace"][:snapshot["step"]]:
                raise ValueError("paired interruption checkpoint is not a committed prefix")
            snapshots.append(snapshot)
    for ordinal, charge in enumerate(charges):
        step = charge["step"] - 1
        if (charge["attempt"] != ordinal or step != len([key for key in committed if key < ordinal])
                or charge["examples"] != plan["updates"][step]["examples"]):
            raise ValueError("paired optimizer charge does not follow the committed schedule")
        examples = charge["examples"]
        measurements = by_attempt.get(ordinal, [])
        trial = {name: copy.deepcopy(model) for name, model in models.items()}
        trial_opts = {name: torch.optim.Adam(model.parameters(), lr=plan["optimizer"]["learning_rate"]) for name, model in trial.items()}
        for name in trial:
            trial_opts[name].load_state_dict(copy.deepcopy(optimizers[name].state_dict()))
        cursor, before_rng = 0, capture_rng("cpu")
        def evaluate(example, candidate):
            nonlocal cursor
            if cursor >= len(measurements):
                raise IncompleteMeasurement()
            observed = measurements[cursor]
            row = examples[example]
            if (observed["query"] != cursor or observed["step"] != step + 1 or observed["example"] != example
                    or any(observed[key] != row[key] for key in ("state", "input_view", "view")) or observed["candidate"] != candidate.to_dict()):
                raise ValueError("paired native gradient query differs from its replayed model")
            matching = [receipt for receipt in outcomes.values() if receipt["status"] == "accepted" and receipt["phase"] == "training"
                        and receipt["label"] == f"gradient:{step}:{example}:{cursor}" and receipt["request_id"] == observed["request_id"]
                        and receipt["qa"]["audio_float32_sha256"] == observed["audio_sha256"]]
            if not matching:
                raise ValueError("paired loss measurement has no accepted native receipt")
            for receipt in matching:
                require_request(candidate, Performance.from_dict(plan["training"][row["state"]]["performances"][row["view"]]), receipt)
            cursor += 1
            return observed["loss"]
        features = [record_features(records[row["state"]], row["input_view"]) for row in examples]
        targets = [PilotTarget.from_dict(plan["training"][row["state"]]["target"]) for row in examples]
        try:
            result = matched_pair_step(trial, trial_opts, features, targets, codec, evaluate, step, len(plan["updates"]), scales=plan["production_training"]["scales"])
        except IncompleteMeasurement:
            if ordinal in committed:
                raise ValueError("committed paired update has incomplete native measurements") from None
            result = None
        if cursor != len(measurements):
            raise ValueError("unconsumed paired native measurements")
        if ordinal in committed:
            if {**charge, **result} != committed[ordinal]:
                raise ValueError("paired model/gradient trace cannot be reconstructed")
            models, optimizers = trial, trial_opts
            if prefix_evidence and step + 1 == prefix_evidence["replayed_updates"]:
                if (prefix_evidence["plan_id"] != canonical_sha256(plan) or prefix_evidence["training_state_sha256"] != training_signature(models, optimizers)
                        or not prefix_evidence["offline_training_replay_exact"]):
                    raise ValueError("paired early end-to-end replay evidence changed")
            for snapshot in snapshots:
                if snapshot["step"] == step + 1 and (not all(equal_tree(models[name].state_dict(), snapshot["models"][name])
                        and equal_tree(optimizers[name].state_dict(), snapshot["optimizers"][name]) for name in models)
                        or not equal_tree(capture_rng("cpu"), snapshot["rng"])):
                    raise ValueError("paired interruption model/Adam/RNG state cannot be reconstructed")
        else:
            interrupted.append(ordinal)
            restore_rng(before_rng, "cpu")
    if (not all(equal_tree(models[name].state_dict(), saved["models"][name]) and equal_tree(optimizers[name].state_dict(), saved["optimizers"][name]) for name in models)
            or not equal_tree(capture_rng("cpu"), saved["rng"])):
        raise ValueError("paired final models, Adam or RNG do not reproduce exactly")
    if prefix:
        if used != {key for key, row in outcomes.items() if row["status"] == "accepted"}:
            raise ValueError("paired early replay contains an unaccounted native request")
        return {"plan_id": canonical_sha256(plan), "replayed_updates": saved["step"], "training_state_sha256": training_signature(models, optimizers),
                "offline_training_replay_exact": True, "checkpoint_file_sha256": digest(source / "checkpoint.pt"),
                "native_requests_added": 0, "optimizer_updates_added": 0, "qualified_for_release": False}
    if plan.get("prefix_check_updates") and prefix_evidence is None:
        raise ValueError("paired early end-to-end replay evidence is missing")
    all_models = {"initial": incumbent.eval(), **{name: model.eval() for name, model in models.items()}}
    retrieval = load_training_checkpoint(bootstrap_root / "fit-000/checkpoint.pt")["replay"]
    fit_cases = json.loads((source / "fit_cases.json").read_text())
    validation_cases = json.loads((source / "validation_cases.json").read_text()) if (source / "validation_cases.json").exists() else []
    compared, references = 0, {}
    model_ids = {name: tensor_digest(model.state_dict()) for name, model in all_models.items()}
    for split, cases in (("training", fit_cases), ("validation", validation_cases)):
        for case in cases:
            identity = case.get("case_id", case["state_id"])
            recipe = next(row for row in plan[split] if row.get("case_id", row["state_id"]) == identity)
            index, view = plan[split].index(recipe), case["view"]
            label = f"{'fit' if split == 'training' else 'validation'}-{index:03d}-{view}"
            if case["label"] != label:
                raise ValueError("paired comparison label changed")
            audit = torch.load(source / "audits" / f"{case['label']}.pt", map_location="cpu", weights_only=True)
            contract = {"plan_id": canonical_sha256(plan), "recipe": recipe, "view": view, "identity": model_ids}
            if (audit["contract"] != contract or audit["rows"] != case["rows"] or audit["source_receipt"] != case["source_receipt"]
                    or not audit["repeat_exact"] or case["repeat_exact"] != audit["repeat_exact"]):
                raise ValueError("paired hard comparison evidence changed")
            reference = audit["audio"].numpy()
            if audio_float32_sha256(reference) != audit["source_receipt"]["qa"]["audio_float32_sha256"]:
                raise ValueError("paired audit source audio changed")
            require_receipt(audit["source_receipt"])
            performance = Performance.from_dict(recipe["performances"][view])
            require_request(PilotTarget.from_dict(recipe["target"]), performance, audit["source_receipt"])
            if split == "training":
                record = records[index]
                if (audio_float32_sha256(reference) != record["audio_sha256"][view]
                        or not equal_tree(SpectralTargets()(audit["audio"].T[None]), record["targets"][view])
                        or tensor_digest(frontend(reference)) != tensor_digest(record_features(record, view))):
                    raise ValueError("paired cached source target/input cannot be reconstructed")
                candidates = predict_candidates(all_models, codec, record_features(record, view))
            else:
                require_receipt(audit["repeat_receipt"])
                require_request(PilotTarget.from_dict(recipe["target"]), performance, audit["repeat_receipt"])
                if audit["repeat_receipt"]["qa"]["audio_float32_sha256"] != audio_float32_sha256(reference):
                    raise ValueError("paired development source repeat changed")
                if view == 0:
                    features = frontend(reference)
                    references[identity] = {**predict_candidates(all_models, codec, features), "template": codec.default_target(),
                                            "retrieval": nearest_training_target(features, retrieval)}
                candidates = references[identity]
            if set(case["rows"]) != set(candidates) or audit["candidates"] != {name: value.to_dict() for name, value in candidates.items()}:
                raise ValueError("paired comparison candidate coverage changed")
            for name, candidate in candidates.items():
                row = case["rows"][name]
                if row["candidate"] != candidate.to_dict():
                    raise ValueError("paired hard patch differs from the actual model output")
                require_receipt(row["receipt"])
                require_request(candidate, performance, row["receipt"])
                if rescore_waveform(source, row["receipt"], reference, codec.contract.tail_seconds) != row["scores"]:
                    raise ValueError("paired sound score differs from its retained waveform")
                compared += 1
    fit_result = audit_summary(fit_cases, plan, validation=False)
    validation_result = audit_summary(validation_cases, plan, validation=True) if fit_result["pass"] else None
    if (not fit_result["complete"] or (validation_result is not None and not validation_result["complete"])
            or used != {key for key, row in outcomes.items() if row["status"] == "accepted"}):
        raise ValueError("paired comparison or native receipt coverage is incomplete")
    for name in models:
        exported = torch.load(source / f"diagnostic-{name}.pt", map_location="cpu", weights_only=True)
        restored, exported_frontend, exported_codec = restore_model(exported)
        if (tensor_digest(restored.state_dict()) != model_ids[name] or exported_codec.to_dict() != codec.to_dict()
                or tensor_digest(exported_frontend.basic_pitch.state_dict()) != tensor_digest(frontend.basic_pitch.state_dict())
                or exported["training_plan_id"] != canonical_sha256(plan) or exported["training_objective"] != name
                or exported["production_training"] != plan["production_training"] or exported["render_profile"] != plan["render_profile"]
                or exported["render_phase_controls"] != plan["render_phase_controls"] or exported["qualified_for_release"]):
            raise ValueError("paired diagnostic export differs from its recovered model/profile")
    archive = sum(path.stat().st_size for path in source.rglob("*") if path.is_file())
    counts = {phase: sum(key[0] == phase for key in attempts) for phase in ("training", "qualification")}
    resources = (summary["rss_sampling_errors"] == 0 and 0 < summary["peak_process_tree_rss_bytes"] <= plan["limits"]["process_tree_bytes"]
                 and summary["elapsed_seconds"] <= plan["limits"]["wall_seconds"] and archive + 1024**2 <= plan["limits"]["archive_bytes"]
                 and len(charges) <= plan["limits"]["optimizer_attempts"] and all(counts[phase] <= plan["limits"][f"{phase}_attempts"] for phase in counts))
    passed = bool(fit_result["pass"] and validation_result and validation_result["pass"] and resources)
    if (fit_result != summary["fit_result"] or validation_result != summary["validation_result"] or passed != summary["matched_comparison_passed"]
            or resources != summary["within_resource_limits"] or counts != summary["render_attempts"] or recovered != summary["recovered_timeouts"]
            or summary["all_native_accepted"] != all(row["status"] == "accepted" for row in outcomes.values())
            or not summary["native_receipts_resolved"] or not summary["incumbent_artifacts_unchanged"]):
        raise ValueError("paired quality/resource decision differs from reconstructed evidence")
    return {**summary, "offline_training_replay_exact": True, "comparison_waveforms_rescored": compared,
            "interrupted_optimizer_attempts_replayed": interrupted, "cached_targets_and_inputs_rescored": True,
            "interruption_checkpoints_replayed": [snapshot["step"] for snapshot in snapshots],
            "all_hard_requests_reconstructed": True, "retained_archive_bytes": archive,
            "fit_cases": public_cases(fit_cases), "validation_cases": public_cases(validation_cases),
            "descriptive_validation_comparison": development_comparison(validation_cases, plan.get("validation_panels")) if validation_cases else None,
            "final_model_sha256": {name: tensor_digest(model.state_dict()) for name, model in models.items()},
            "evidence_sha256": {name: digest(source / name) for name in ("plan.json", "summary.json", "checkpoint.pt", "records.json", "renders.jsonl", "updates.jsonl", "measurements.jsonl")},
            "native_requests_added": 0, "optimizer_updates_added": 0}


def verify_prefix(source, pool, bootstrap_root, plan):
    started, before_rng = perf_counter(), capture_rng("cpu")
    try:
        result = verify(source, pool, bootstrap_root, None, None, None, None, None, prefix_plan=plan)
    finally:
        restore_rng(before_rng, "cpu")
    result["elapsed_seconds"] = perf_counter() - started
    if result["replayed_updates"] != plan["prefix_check_updates"] or result["elapsed_seconds"] >= 300:
        raise ValueError("paired early end-to-end replay is incomplete or exceeds five minutes")
    checkpoint = source / "prefix-checkpoint.pt"
    if checkpoint.exists():
        if digest(checkpoint) != result["checkpoint_file_sha256"]:
            raise ValueError("preserve the existing early replay checkpoint")
    else:
        shutil.copy2(source / "checkpoint.pt", checkpoint)
    result["checkpoint_file"] = checkpoint.name
    write_json(source / "prefix-verification.json", result)
    return result


def run_verification(source, pool, *inputs):
    path = source / "verification.json"
    if path.exists():
        raise ValueError("paired verification already exists; preserve evidence")
    started, monitor = perf_counter(), MemoryMonitor()
    monitor.start()
    try:
        with refresh.cpu_settings():
            result = verify(source, pool, *inputs)
    finally:
        monitor.close()
    elapsed = perf_counter() - started
    within = (0 < elapsed < 1200 and monitor.peak_bytes is not None and 0 < monitor.peak_bytes < 4 * 1024**3 and monitor.errors == 0)
    result["verification_resources"] = {"elapsed_seconds": elapsed, "peak_process_tree_rss_bytes": monitor.peak_bytes,
                                        "rss_sampling_errors": monitor.errors, "within_resource_limits": within}
    result["verified"] = within
    write_json(path, result)
    if not within:
        raise ValueError("paired verification exceeded its twenty-minute/4-GiB allocation")
    return {key: result[key] for key in ("verified", "offline_training_replay_exact", "matched_comparison_passed", "comparison_waveforms_rescored", "verification_resources")}

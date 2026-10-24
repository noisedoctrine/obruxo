"""Matched paired sound training with production choices and retained comparison audio."""

import copy
import json
from pathlib import Path
import shutil
from time import perf_counter, time

import torch

from obruxo_data.hashing import canonical_sha256, sha256_bytes
from obruxo_data.midi import Performance, TempoMap

from . import category_scale_refresh as refresh
from .audit import write_json
from .balanced_native_fit import compose_plan
from .benchmark import MemoryMonitor
from .checkpoints import atomic_torch_save, load_training_checkpoint
from .conditioned_fit_policy import restore_policy
from .evaluate import nearest_training_target
from .features import tensor_digest
from .native_fit import FitRecords, audit_case, audit_summary, digest, make_plan as legacy_plan, predict_candidates, record_features
from .paired_native_evidence import RetainedAuditSession, native_ledger
from .paired_native_step import reciprocal_updates
from .paired_native_training import fit_pairs
from .paired_performance_objective import include_input_performance
from .production_controls import ProductionControlAdapter
from .production_native_audit import verify_screen
from .reproducibility import FULL_PHASE_CONTROL
from .temporal_proxy_data import NativeSession, category_balanced_entries
from .trainer import inference_state, restore_model
from .training_coverage import expand_training


def compose_paired_plan(pool, base, cache, scales):
    original = category_balanced_entries(pool, "train", 32)
    entries, _population = expand_training(pool, original)
    selected = {entry.state_id for entry in entries}
    filtered = {"replay": [row for row in cache["replay"] if row["state_id"] in selected]}
    plan = compose_plan(pool, base, filtered, {})
    plan.pop("category_training")
    updates = reciprocal_updates(len(plan["training"]))
    frames = {}
    for split in ("training", "validation"):
        frames[split] = []
        for recipe in plan[split]:
            for value in recipe["performances"]:
                performance = Performance.from_dict(value)
                frames[split].append(TempoMap.from_performance(performance).render_frame_count(performance.end_tick, pool.codec.contract.tail_seconds, 44100))
    candidate_bytes = 8 * (3 * sum(frames["training"]) + 5 * sum(frames["validation"]))
    count = 3 * len(frames["training"]) + 5 * len(frames["validation"])
    extra = candidate_bytes + count * 16384 + 8 * max(frames["training"] + frames["validation"]) + 32 * 1024**2
    storage = {**plan["storage_bound"], "candidate_waveform_bytes": candidate_bytes, "candidate_waveforms_maximum": count,
               "candidate_waveform_and_serialization_allowance_bytes": extra,
               "estimated_archive_upper_bound_bytes": plan["storage_bound"]["estimated_archive_upper_bound_bytes"] + extra}
    if storage["estimated_archive_upper_bound_bytes"] > plan["limits"]["archive_bytes"]:
        raise ValueError("paired comparison audio exceeds the existing archive allocation")
    return {**plan, "format": "paired_native_inverse_fit_v1", "updates": updates,
            "adapter": ProductionControlAdapter(pool.codec, scales=scales).contract(), "storage_bound": storage,
            "input_schedule": "one_original_and_one_reciprocal_wider_observation_per_update",
            "prefix_check_updates": 4,
            "production_training": {"forward": "unperturbed_argmax", "parameter_logits": "raw", "scales": list(scales)},
            "optimizer": {**plan["optimizer"], "batch_size": 2, "forward_microbatch_size": 1, "updates_per_arm": len(updates)},
            "limits": {**plan["limits"], "optimizer_attempts": len(updates) + 4},
            "recovery": "known_identical_RenderTimeout_only_with_original_deadline_and_all_attempts_charged",
            "comparison_audio": "content_addressed_float32_all_fit_and_development_candidates"}


def make_plan(pool, bootstrap_root, bootstrap_audit_path, qualification_path, cache_path, choice_path, direction_source, *, input_and_transfer=False):
    direction = verify_screen(direction_source, pool, bootstrap_root, bootstrap_audit_path, qualification_path, cache_path, choice_path)
    if (not direction["qualified_production_direction"] or direction["candidate_audio_rescored"] != 72
            or not direction["all_model_gradients_and_local_steps_reconstructed"] or not direction["all_requests_reconstructed"]):
        raise ValueError("paired sound fitting requires the reconstructed native production-direction qualification")
    initial, retrieval = bootstrap_root / "fit-000/model.pt", bootstrap_root / "fit-000/checkpoint.pt"
    base = legacy_plan(pool, initial, retrieval, qualification_path)
    if (base["initial_file_sha256"] != direction["plan"]["initial_sha256"]
            or base["initial_model_sha256"] != direction["plan"]["inference"]["model_sha256"]
            or digest(cache_path) != direction["plan"]["input_cache_sha256"]):
        raise ValueError("paired fit differs from its qualified model or input recordings")
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    plan = compose_paired_plan(pool, base, cache, direction["plan"]["adapter"]["fixed_scales"])
    if input_and_transfer:
        plan = include_input_performance(plan)
    names = ("paired_native_fit.py", "paired_native_step.py", "paired_native_training.py", "paired_native_evidence.py", "paired_native_audit.py",
             "paired_performance_objective.py", "balanced_native_fit.py", "balanced_sound_data.py", "native_fit_storage.py", "training_coverage.py", "native_fit_report.py")
    paths = {name: Path(__file__).with_name(name) for name in names}
    paths["reports/scripts/run_paired_native_fit.py"] = Path(__file__).parents[1] / "reports/scripts/run_paired_native_fit.py"
    plan["source"] = {**plan["source"], **direction["plan"]["source"],
                      **{name: sha256_bytes(path.read_bytes().replace(b"\r\n", b"\n")) for name, path in paths.items()}}
    plan.update(input_cache_sha256=digest(cache_path), expected_renderer_identity=direction["renderer_identity"],
                render_phase_controls=dict(FULL_PHASE_CONTROL),
                native_direction_plan_id=direction["plan_id"], native_direction_evidence_sha256=direction["evidence_sha256"])
    return plan


class PairedSession(NativeSession):
    def render(self, target, performance, phase, label):
        audio, receipt = super().render(target, performance, phase, label)
        if audio is not None and receipt["renderer_identity"] != self.plan["expected_renderer_identity"]:
            raise ValueError("paired fit renderer differs from its native qualification")
        return audio, receipt


def archive_interruption(output, plan):
    summary = json.loads((output / "summary.json").read_text())
    if summary["plan_id"] != canonical_sha256(plan) or summary["complete"] or summary["failure"] is None:
        raise ValueError("only an incomplete matching run may resume")
    root = output / "interruptions"
    root.mkdir(exist_ok=True)
    target = root / f"{len(list(root.iterdir())) + 1:03d}"
    target.mkdir()
    names = ("summary.json", "checkpoint.pt", "checkpoint.previous.pt", "renders.jsonl", "updates.jsonl", "measurements.jsonl", "records.json", "fit_cases.json", "validation_cases.json")
    for name in names:
        if (output / name).exists():
            shutil.copy2(output / name, target / name)
    write_json(target / "evidence.json", {name: digest(target / name) for name in names if (target / name).exists()})


def run_plan(pool, initial_path, retrieval_path, output, plan, *, execute=False, resume=False, input_validation=None, input_cache_path=None):
    plan_id = canonical_sha256(plan)
    if not execute:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "plan.json", plan)
        return {"plan_id": plan_id, "updates_per_arm": len(plan["updates"]), "maximum_native_requests": sum(plan["limits"][key] for key in ("training_attempts", "qualification_attempts"))}
    if json.loads((output / "plan.json").read_text()) != plan or (output / "run.json").exists() != resume:
        raise ValueError("paired plan changed or resume does not match this attempted run")
    if resume:
        archive_interruption(output, plan)
    else:
        write_json(output / "run.json", {"plan_id": plan_id, "started_unix": time(), "input_validation": input_validation})
    initial = torch.load(initial_path, map_location="cpu", weights_only=True)
    incumbent, frontend, codec = restore_model(initial)
    restore_policy(incumbent, initial)
    models = {name: copy.deepcopy(incumbent).train() for name in ("parameter", "spectral")}
    retrieval = load_training_checkpoint(retrieval_path)["replay"]
    monitor = MemoryMonitor()
    monitor.start()
    session = PairedSession(output, plan, codec, monitor)
    retained = RetainedAuditSession(session, output)
    records = FitRecords(output, plan, frontend, session)
    failure, trace, fit_cases, validation_cases = None, [], [], []
    fit_result, validation_result = None, None
    try:
        saved = None
        if resume and (output / "checkpoint.pt").exists():
            try:
                saved = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
            except (RuntimeError, EOFError, OSError):
                saved = torch.load(output / "checkpoint.previous.pt", map_location="cpu", weights_only=True)
        if saved:
            trace = list(saved["trace"])
        def committed(state):
            trace[:] = state["trace"]
            if state["step"] == plan.get("prefix_check_updates") and not (output / "prefix-verification.json").exists():
                from .paired_native_audit import verify_prefix
                verify_prefix(output, pool, initial_path.parent.parent, plan)
        trace = fit_pairs(models, records, session, output, plan, saved, after_step=committed)
        all_models = {"initial": incumbent.eval(), **{name: model.eval() for name, model in models.items()}}
        identities = {name: tensor_digest(model.state_dict()) for name, model in all_models.items()}
        if plan.get("fit_audit_cells"):
            from .conditioned_short_audit import collect_comparisons
            fit_cases = collect_comparisons(retained, output, plan, all_models, codec, records, input_cache_path, identities)
        for index, recipe in enumerate([] if plan.get("fit_audit_cells") else plan["training"]):
            record = records[index]
            for view in range(3):
                candidates = predict_candidates(all_models, codec, record_features(record, view))
                case, _ = audit_case(retained, output, f"fit-{index:03d}-{view}", recipe, view, lambda _audio: candidates,
                                     identities, expected_audio=record["audio_sha256"][view])
                fit_cases.append(case)
                write_json(output / "fit_cases.json", fit_cases)
                records.check_limits()
        if plan.get("fit_audit_cells"):
            from .conditioned_short_audit import short_summary
            fit_result = short_summary(fit_cases, plan, trace)
        else:
            fit_result = audit_summary(fit_cases, plan, validation=False)
        write_json(output / "fit_result.json", fit_result)
        if fit_result["pass"] and not plan.get("fit_audit_cells"):
            for index, recipe in enumerate(plan["validation"]):
                def provider(audio):
                    features = frontend(audio)
                    return {**predict_candidates(all_models, codec, features), "template": codec.default_target(),
                            "retrieval": nearest_training_target(features, retrieval)}
                reference, candidates = audit_case(retained, output, f"validation-{index:03d}-0", recipe, 0, provider, identities, repeat=True)
                transfer, _ = audit_case(retained, output, f"validation-{index:03d}-1", recipe, 1, lambda _audio: candidates, identities, repeat=True)
                validation_cases.extend((reference, transfer))
                write_json(output / "validation_cases.json", validation_cases)
                records.check_limits()
            validation_result = audit_summary(validation_cases, plan, validation=True)
        for name, model in models.items():
            export = inference_state(model, frontend, codec, pool.pool_id, preroll_frames=plan["render_profile"]["preroll_frames"])
            export.update(training_objective=name, training_plan_id=plan_id, qualified_for_release=False,
                          render_profile=plan["render_profile"], render_phase_controls=dict(FULL_PHASE_CONTROL), production_training=plan["production_training"])
            if hasattr(model, "conditioning_training_policy"):
                export["conditioning_training_policy"] = copy.deepcopy(model.conditioning_training_policy)
            atomic_torch_save(output / f"diagnostic-{name}.pt", export)
        frontend.require_frozen()
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        monitor.close()
        elapsed = time() - session.started
        archive = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
        attempts = [row for row in session.events if row["event"] == "attempt"]
        outcomes = [row for row in session.events if row["event"] == "outcome"]
        resolved, recovered = False, []
        try:
            _attempts, _outcomes, recovered = native_ledger(session.events, plan["expected_renderer_identity"])
            resolved = True
        except ValueError:
            pass
        resources = (monitor.errors == 0 and monitor.peak_bytes is not None and 0 < monitor.peak_bytes <= plan["limits"]["process_tree_bytes"]
                     and elapsed <= plan["limits"]["wall_seconds"] and archive + 1024**2 <= plan["limits"]["archive_bytes"])
        unchanged = digest(initial_path) == plan["initial_file_sha256"] and digest(retrieval_path) == plan["retrieval_file_sha256"]
        complete = failure is None and len(trace) == len(plan["updates"]) and fit_result is not None
        quality = fit_result and fit_result["pass"] and (bool(plan.get("fit_audit_cells")) or validation_result and validation_result["pass"])
        passed = bool(complete and quality and resolved and resources and unchanged)
        summary = {"plan_id": plan_id, "plan": plan, "complete": complete, "failure": failure, "fit_result": fit_result, "validation_result": validation_result,
                   "matched_comparison_passed": passed, "committed_updates_per_arm": len(trace), "elapsed_seconds": elapsed,
                   "peak_process_tree_rss_bytes": monitor.peak_bytes, "rss_sampling_errors": monitor.errors, "archive_bytes_before_summary": archive,
                   "render_attempts": {phase: sum(row["phase"] == phase for row in attempts) for phase in ("training", "qualification")},
                   "all_native_accepted": len(attempts) == len(outcomes) and all(row["status"] == "accepted" for row in outcomes),
                   "native_receipts_resolved": resolved, "recovered_timeouts": recovered, "within_resource_limits": resources,
                   "incumbent_artifacts_unchanged": unchanged, "renderer_identity": session.renderer_identity, "test_split_used": False, "qualified_for_release": False}
        write_json(output / "summary.json", summary)
    return {"complete": complete, "matched_comparison_passed": passed, "committed_updates_per_arm": len(trace)}


def run_fit(pool, bootstrap_root, bootstrap_audit_path, qualification_path, cache_path, choice_path, direction_source, output, *, execute=False, resume=False, input_and_transfer=False):
    started, monitor = perf_counter(), MemoryMonitor()
    monitor.start()
    with refresh.cpu_settings():
        try:
            plan = make_plan(pool, bootstrap_root, bootstrap_audit_path, qualification_path, cache_path, choice_path, direction_source, input_and_transfer=input_and_transfer)
        finally:
            monitor.close()
        validation = {"elapsed_seconds": perf_counter() - started, "peak_process_tree_rss_bytes": monitor.peak_bytes,
                      "rss_sampling_errors": monitor.errors, "native_requests": 0, "optimizer_updates": 0}
        if (validation["elapsed_seconds"] >= 600 or monitor.peak_bytes is None or monitor.peak_bytes > 8 * 1024**3 or monitor.errors):
            raise RuntimeError("paired fit input validation exceeded its ten-minute/8-GiB allocation")
        result = run_plan(pool, bootstrap_root / "fit-000/model.pt", bootstrap_root / "fit-000/checkpoint.pt", output, plan,
                          execute=execute, resume=resume, input_validation=validation)
        if not execute:
            write_json(output / "preparation.json", validation)
    return {**result, "input_validation": validation}

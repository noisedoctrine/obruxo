"""Finish comparisons after a confirmed process loss, without training again."""

from datetime import datetime, timezone
from pathlib import Path
import shutil
from time import perf_counter, time

import psutil
import torch

from obruxo_data.hashing import canonical_sha256, sha256_bytes
from obruxo_data.midi import Performance, TempoMap
from obruxo_data.render.base import RenderRequest

from .audit import write_json
from .basic_direct import export_direct
from .basic_direct_fit import SoundStore, lines, replay_training, require_charges
from .basic_direct_fit_audit import decisions, run_verification
from .basic_direct_fit_data import Inputs
from .benchmark import MemoryMonitor
from .category_scale_refresh import cpu_settings
from .checkpoints import atomic_torch_save
from .conditioned_bridge import read
from .features import tensor_digest
from .native_fit import append_durable, digest
from .paired_native_audit import equal_tree
from .paired_native_evidence import native_ledger, rescore_waveform
from .paired_native_fit import PairedSession
from .reproducibility import fixed_generator_phases
from .unison_fit_data import require_request
from .unison_sound_fit import collect_finals

LIMITS = {"wall_seconds": 1200, "process_tree_bytes": 4 * 1024**3, "archive_bytes": 2 * 1024**3}
ADAPTERS = {"basic_direct_fit_audit.py", "paired_native_evidence.py"}


def require_gone(identity):
    try:
        process = psutil.Process(identity["process_id"])
        if abs(process.create_time() - identity["process_created_unix"]) < .01 and process.is_running():
            raise ValueError("original training process is still live")
    except psutil.NoSuchProcess:
        pass


def source_manifest(source):
    return {path.relative_to(source).as_posix(): digest(path) for path in source.rglob("*") if path.is_file()}


def inspect(source, parent, bridge, cache):
    plan = read(source / "plan.json")
    identity = read(source / "service-start-001.json")
    require_gone(identity)
    if read(source / "run.json")["plan_id"] != canonical_sha256(plan) or identity["mode"] != "execute":
        raise ValueError("original execution identity or plan changed")
    if (source / "summary.json").exists() or (source / "service-001-completion.json").exists():
        raise ValueError("this completion handles only the confirmed receipt loss without a terminal report")
    if digest(cache) != plan["input_cache_sha256"] or digest(bridge / "model.pt") != plan["initial_file_sha256"]:
        raise ValueError("final completion source features or initializer changed")
    root, differences = Path(__file__).parents[1], {}
    for name, expected in plan["source"].items():
        path = root / name if "/" in name else Path(__file__).with_name(name)
        actual = sha256_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
        if actual != expected:
            if name not in ADAPTERS:
                raise ValueError("final completion cannot change numerical training or rendering code")
            differences[name] = {"historical": expected, "current": actual}
    inputs = Inputs(parent, bridge, cache, plan)
    inputs.validate()
    saved = torch.load(source / "checkpoint.pt", map_location="cpu", weights_only=True)
    require_charges(plan, lines(source / "updates.jsonl"), saved["trace"])
    if saved["step"] != len(plan["updates"]):
        raise ValueError("final-only completion requires all scheduled training updates")
    store = SoundStore(source, inputs)
    models, _optimizers = replay_training(inputs, store, saved, read(source / "prefix-verification.json"))
    if store.used_probes != set(store.probes):
        raise ValueError("full training replay did not consume every measured probe")
    if tensor_digest(models["parameter"].state_dict()) != plan["expected_parameter_model_sha256"]:
        raise ValueError("completed training changed its matched parameter control")
    events = lines(source / "renders.jsonl")
    if not events or events[-1]["event"] != "attempt" or events[-1]["phase"] != "qualification":
        raise ValueError("only one trailing unobserved final comparison may be resolved")
    pending = events[-1]
    attempts, outcomes, recovered = native_ledger(events[:-1], plan["expected_renderer_identity"])
    if recovered or (pending["phase"], pending["attempt"]) in attempts:
        raise ValueError("final completion cannot mix earlier failures with a receipt loss")
    if len(lines(source / "executions.jsonl")) != 1 or plan["limits"]["native_executions"] < 2:
        raise ValueError("no native completion execution remains")
    rows = lines(source / "finals.jsonl")
    order = [(index, arm) for index in range(len(plan["fit_audit_cells"])) for arm in models]
    if [(row["index"], row["arm"]) for row in rows] != order[:len(rows)] or len(rows) >= len(order):
        raise ValueError("retained final comparisons are not the exact planned prefix")
    for row in rows:
        index, arm = row["index"], row["arm"]
        cell = plan["fit_audit_cells"][index]
        with torch.no_grad():
            target = inputs.target(index, prediction=inputs.predict(models[arm].eval(), cell))
        receipt = row["receipt"]
        raw = {key: value for key, value in receipt.items() if key != "retained_audio"}
        if row["candidate"] != target.to_dict() or raw != outcomes.get(("qualification", receipt["attempt"])):
            raise ValueError("retained final prediction differs from the saved model or native ledger")
        performance = Performance.from_dict(cell["recipe"]["performances"][cell["playback_view"]])
        require_request(inputs.codec, plan, target, performance, receipt)
        reference, metadata = store.source(index)
        if metadata != row["source_audio"] or rescore_waveform(source, receipt, reference, inputs.codec.contract.tail_seconds) != row["scores"]:
            raise ValueError("retained final audio cannot be independently rescored")
    if len([row for row in outcomes.values() if row["phase"] == "qualification"]) != len(rows):
        raise ValueError("final native ledger contains an unexplained observed comparison")
    index, arm = order[len(rows)]
    cell = plan["fit_audit_cells"][index]
    with torch.no_grad():
        target = inputs.target(index, prediction=inputs.predict(models[arm], cell))
    raw = inputs.codec.decode(target)
    controlled = fixed_generator_phases(raw)
    performance = Performance.from_dict(cell["recipe"]["performances"][cell["playback_view"]])
    request = RenderRequest(preset_json=controlled.decode(), performance=performance, renderer_id=inputs.codec.contract.renderer_id,
                            tail_seconds=inputs.codec.contract.tail_seconds, preroll_frames=plan["render_profile"]["preroll_frames"])
    expected = {"event": "attempt", "phase": "qualification", "attempt": len(rows), "label": f"final:{index:03d}:{arm}",
                "request_id": request.request_id, "source_sha256": sha256_bytes(raw), "controlled_sha256": sha256_bytes(controlled)}
    if pending != expected:
        raise ValueError("lost receipt does not describe the next exact model comparison")
    missing = order[len(rows):]
    if pending["attempt"] + 1 + len(missing) > plan["limits"]["qualification_attempts"]:
        raise ValueError("final comparisons exceed the original qualification request cap")
    audio_bound = sum(8 * TempoMap.from_performance(Performance.from_dict(plan["fit_audit_cells"][i]["recipe"]["performances"][plan["fit_audit_cells"][i]["playback_view"]])).render_frame_count(
        Performance.from_dict(plan["fit_audit_cells"][i]["recipe"]["performances"][plan["fit_audit_cells"][i]["playback_view"]]).end_tick,
        inputs.codec.contract.tail_seconds, 44100) + 16384 for i, _arm in missing)
    archive = sum(path.stat().st_size for path in source.rglob("*") if path.is_file())
    bound = 2 * archive + audio_bound + 64 * 1024**2
    if bound > LIMITS["archive_bytes"]:
        raise ValueError("two-directory completion evidence exceeds the existing archive cap")
    return {"source_plan_id": canonical_sha256(plan), "source_directory": str(source.resolve()), "source_evidence_sha256": source_manifest(source),
            "process_identity": identity, "observed_at_utc": datetime.now(timezone.utc).isoformat(), "original_process_live": False,
            "pending_attempt": pending, "retained_comparisons": len(rows), "missing_comparisons": len(missing),
            "training_updates_per_arm": saved["step"], "training_model_adam_rng_replay_exact": True,
            "accepted_training_updates": sum(row["native_acceptance"]["accepted"] for row in saved["trace"]),
            "training_scalar_probes_replayed": len(store.probes), "source_archive_bytes": archive,
            "aggregate_archive_upper_bound_bytes": bound, "original_started_unix": read(source / "run.json")["started_unix"],
            "original_budget_passed": False, "original_peak_process_tree_rss_bytes": None,
            "original_terminal_resource_record_available": False, "compatible_adapter_source_differences": differences}


def verify_origin(plan, output):
    completion = plan["final_only_completion"]
    observation = read(output / "process-loss-observation.json")
    if canonical_sha256(observation) != completion["observation_evidence_sha256"] or observation != completion["observation"]:
        raise ValueError("frozen process-loss observation changed")
    require_gone(observation["process_identity"])
    source = Path(observation["source_directory"])
    if source_manifest(source) != observation["source_evidence_sha256"]:
        raise ValueError("immutable original fit evidence changed")
    original = read(source / "plan.json")
    pending = {key: value for key, value in observation["pending_attempt"].items() if key != "event"}
    expected_loss = {**pending, "observation_evidence_sha256": completion["observation_evidence_sha256"]}
    unchanged = {key: value for key, value in plan.items() if key not in ("format", "source", "limits", "final_only_completion")}
    expected = {key: value for key, value in original.items() if key not in ("format", "source", "limits")}
    if (canonical_sha256(original) != observation["source_plan_id"] or unchanged != expected
            or plan["limits"] != {**original["limits"], **LIMITS} or completion["known_process_losses"] != [expected_loss]
            or completion["experimental_training_updates"] != 0
            or completion["maximum_additional_native_requests"] != observation["missing_comparisons"]):
        raise ValueError("completion changed the original experiment or process-loss certificate")
    if (output / "renders.jsonl").exists() and not (output / "renders.jsonl").read_bytes().startswith((source / "renders.jsonl").read_bytes()):
        raise ValueError("completion changed the original native ledger prefix")
    if (output / "finals.jsonl").exists() and not (output / "finals.jsonl").read_bytes().startswith((source / "finals.jsonl").read_bytes()):
        raise ValueError("completion changed the original final-comparison prefix")
    for name in ("probes.jsonl", "updates.jsonl"):
        if (output / name).exists() and digest(output / name) != observation["source_evidence_sha256"][name]:
            raise ValueError("final-only completion changed training measurements or charges")
    if (output / "checkpoint.pt").exists():
        old = torch.load(source / "checkpoint.pt", map_location="cpu", weights_only=True)
        new = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
        if not equal_tree({key: value for key, value in old.items() if key != "plan_id"}, {key: value for key, value in new.items() if key != "plan_id"}):
            raise ValueError("final-only completion changed training tensors, Adam, RNG or trace")


def prepare(source, parent, bridge, cache, output):
    observation = inspect(source, parent, bridge, cache)
    old = read(source / "plan.json")
    identity = canonical_sha256(observation)
    pending = {key: value for key, value in observation["pending_attempt"].items() if key != "event"}
    loss = {**pending, "observation_evidence_sha256": identity}
    root = Path(__file__).parents[1]
    names = ("basic_direct_fit_audit.py", "paired_native_evidence.py", "final_fit_completion.py", "reports/scripts/complete_native_acceptance_fit.py")
    code = {name: sha256_bytes((root / name if "/" in name else Path(__file__).with_name(name)).read_bytes().replace(b"\r\n", b"\n")) for name in names}
    plan = {**old, "format": "native_accepted_final_only_completion_v1", "source": {**old["source"], **code},
            "limits": {**old["limits"], **LIMITS}, "final_only_completion": {"observation": observation,
                "observation_evidence_sha256": identity, "known_process_losses": [loss], "experimental_training_updates": 0,
                "maximum_additional_native_requests": observation["missing_comparisons"]}}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "process-loss-observation.json", observation)
    write_json(output / "plan.json", plan)
    return {"plan_id": canonical_sha256(plan), "observation": observation, "experimental_training_updates": 0}


def migrate(plan, output):
    verify_origin(plan, output)
    if any((output / name).exists() for name in ("run.json", "renders.jsonl", "checkpoint.pt")):
        raise ValueError("preserve attempted final completions; no second migration")
    source = Path(plan["final_only_completion"]["observation"]["source_directory"])
    for folder in ("source_audio", "comparison_audio"):
        shutil.copytree(source / folder, output / folder)
    for name in ("renders.jsonl", "probes.jsonl", "finals.jsonl", "updates.jsonl"):
        shutil.copyfile(source / name, output / name)
    interrupted = output / "interruption"
    interrupted.mkdir()
    for path in source.iterdir():
        if path.is_file():
            shutil.copyfile(path, interrupted / path.name)
    plan_id = canonical_sha256(plan)
    for name in ("checkpoint.pt", "prefix-checkpoint.pt"):
        saved = torch.load(source / name, map_location="cpu", weights_only=True)
        saved["plan_id"] = plan_id
        atomic_torch_save(output / name, saved)
    prefix = read(source / "prefix-verification.json")
    prefix.update(plan_id=plan_id, checkpoint_file_sha256=digest(output / "prefix-checkpoint.pt"))
    write_json(output / "prefix-verification.json", prefix)
    executions = lines(source / "executions.jsonl")
    if len(executions) != 1:
        raise ValueError("final completion requires the single original native execution")
    for row in executions:
        append_durable(output / "executions.jsonl", {**row, "plan_id": plan_id})
    append_durable(output / "executions.jsonl", {"ordinal": 2, "plan_id": plan_id, "resume": True, "started_unix": time()})
    write_json(output / "run.json", {"plan_id": plan_id, "started_unix": time(),
                                    "original_started_unix": read(source / "run.json")["started_unix"], "original_budget_passed": False})
    loss = plan["final_only_completion"]["known_process_losses"][0]
    append_durable(output / "renders.jsonl", {"event": "outcome", **loss, "status": "unobserved", "error_type": "ProcessLostBeforeReceipt"})
    verify_origin(plan, output)


def execute(plan, parent, bridge, cache, output):
    migrate(plan, output)
    inputs = Inputs(parent, bridge, cache, plan)
    inputs.validate()
    saved = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
    models, _optimizers = replay_training(inputs, SoundStore(output, inputs), saved, read(output / "prefix-verification.json"))
    started, monitor = perf_counter(), MemoryMonitor()
    monitor.start()
    session = PairedSession(output, plan, inputs.codec, monitor)
    store, failure, result = SoundStore(output, inputs, session), None, None
    try:
        collect_finals(inputs, store, models)
        result = decisions(inputs, store, saved["trace"])
        events = session.events[len(lines(Path(plan["final_only_completion"]["observation"]["source_directory"]) / "renders.jsonl")):]
        if (sum(row["event"] == "attempt" for row in events) != plan["final_only_completion"]["maximum_additional_native_requests"]
                or any(row["phase"] != "qualification" for row in events)):
            raise ValueError("final-only completion changed its native-request allocation")
        for arm, model in models.items():
            artifact = export_direct(model, inputs.frontend, inputs.codec, plan["pool_id"])
            artifact.update(training_objective=arm, training_plan_id=canonical_sha256(plan))
            atomic_torch_save(output / f"diagnostic-{arm}.pt", artifact)
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        monitor.close()
        archive = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
        aggregate = archive + plan["final_only_completion"]["observation"]["source_archive_bytes"]
        within = bool(time() - session.started <= LIMITS["wall_seconds"] and monitor.peak_bytes and monitor.peak_bytes <= LIMITS["process_tree_bytes"]
                      and not monitor.errors and aggregate + 1024**2 <= LIMITS["archive_bytes"])
        resolved, recovery = False, []
        try:
            _attempts, _outcomes, recovery = native_ledger(session.events, plan["expected_renderer_identity"], known_process_losses=plan["final_only_completion"]["known_process_losses"])
            resolved = True
        except ValueError:
            pass
        summary = {"plan_id": canonical_sha256(plan), "complete": failure is None and result is not None, "failure": failure, "result": result,
                   "committed_updates_per_arm": saved["step"], "native_receipts_resolved": resolved,
                   "recovered_timeouts": [row for row in recovery if row.get("kind") != "process_loss"],
                   "resolved_process_losses": [row for row in recovery if row.get("kind") == "process_loss"],
                   "elapsed_seconds": time() - session.started, "current_confirmation_seconds": perf_counter() - started,
                   "peak_process_tree_rss_bytes": monitor.peak_bytes, "rss_sampling_errors": monitor.errors,
                   "retained_archive_bytes_before_summary": archive, "aggregate_archive_bytes": aggregate, "within_resource_limits": within,
                   "render_attempts": {phase: sum(row["event"] == "attempt" and row["phase"] == phase for row in session.events) for phase in ("training", "qualification")},
                   "original_budget_passed": False, "original_peak_process_tree_rss_bytes": None,
                   "matched_comparison_passed": bool(result and result["pass"] and within and resolved), "qualified_for_release": False, "test_split_used": False}
        write_json(output / "summary.json", summary)
    verify_origin(plan, output)
    return summary


def run(source, parent, bridge, cache, output, *, mode="prepare"):
    with cpu_settings():
        if mode == "prepare":
            return prepare(source, parent, bridge, cache, output)
        plan = read(output / "plan.json")
        verify_origin(plan, output)
        root = Path(__file__).parents[1]
        for name, expected in plan["source"].items():
            path = root / name if "/" in name else Path(__file__).with_name(name)
            if sha256_bytes(path.read_bytes().replace(b"\r\n", b"\n")) != expected:
                raise ValueError("frozen final-completion implementation changed")
        if mode == "execute":
            return execute(plan, parent, bridge, cache, output)
        if mode == "verify":
            inputs = Inputs(parent, bridge, cache, plan)
            inputs.validate()
            return run_verification(inputs, output)
        raise ValueError("expected prepare, execute or verify")

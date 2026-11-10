"""Checkpointed native sound fitting of fourteen basic outputs with reference rehearsal."""

from collections import OrderedDict
import copy
import random
import shutil
from time import perf_counter, time

import numpy as np
import torch

from obruxo_data.hashing import canonical_sha256
from obruxo_data.midi import Performance

from . import category_scale_refresh as refresh
from .audit import write_json
from .basic_direct import export_direct, restore_direct
from .basic_direct_fit_data import Inputs, make_plan
from .basic_direct_fit_step import matched_step
from .benchmark import MemoryMonitor
from .batching import BatchConfig
from .checkpoints import atomic_torch_save
from .codec import PilotTarget
from .conditioned_bridge import read
from .features import tensor_digest
from .native_fit import append_durable, digest
from .native_acceptance import SPEC as ACCEPTANCE_SPEC, accept_update, capture
from .paired_native_audit import equal_tree, training_signature
from .paired_native_evidence import RetainedAuditSession, native_ledger
from .paired_native_fit import PairedSession
from .spectral import SpectralTargets, concatenate_targets, prepare_spectral_target, spectral_loss
from .trainer import capture_rng, restore_rng
from .unison_fit_data import require_request
from .unison_sound_fit import CostStore, collect_finals, lines


class SoundStore(CostStore):
    def __init__(self, output, inputs, native=None):
        super().__init__(output, inputs, RetainedAuditSession(native, output) if native else None)
        self.native = native
        self.batching = BatchConfig.from_plan(inputs.plan)
        self.probes = {}
        for row in lines(output / "probes.jsonl"):
            if row.get("batching") != self.batching.to_dict():
                raise ValueError("probe execution batching changed; use the recorded configuration and source revision")
            key = canonical_sha256(row["query"])
            if key in self.probes:
                raise ValueError("duplicate direct native probe")
            self.probes[key] = row
        self.used_probes, self.targets = set(), OrderedDict()
        self.measurements = SpectralTargets()

    def sound_target(self, index):
        cell = self.inputs.plan["fit_audit_cells"][index]
        sha = cell["recipe"]["source_audio_sha256"][cell["playback_view"]]
        if sha not in self.targets:
            audio, _metadata = self.source(index)
            self.targets[sha] = prepare_spectral_target(self.measurements(torch.from_numpy(audio.copy()).T[None]))
            while len(self.targets) > 4:
                self.targets.popitem(last=False)
        self.targets.move_to_end(sha)
        resident = self.inputs.bytes + sum(tensor_bytes(value) for value in self.targets.values()) + sum(tensor_bytes(value) for value in self.source_cache.values())
        if resident > self.inputs.plan["limits"]["replay_bytes"]:
            raise RuntimeError("direct fit resident source replay allocation exhausted")
        return self.targets[sha]

    def probe(self, step, observation, index, candidate):
        return self.probe_many([(step, observation, index, candidate)])[0]

    def probe_many(self, requests):
        queries = [{"step": step, "observation": observation, "index": index, "candidate": candidate.to_dict()}
                   for step, observation, index, candidate in requests]
        keys = [canonical_sha256(query) for query in queries]
        self.used_probes.update(keys)
        unique = {}
        for request, query, key in zip(requests, queries, keys, strict=True):
            unique.setdefault(key, (request, query, key))
        planned, pending, flushing = list(unique.values()), {}, False

        def flush():
            nonlocal flushing
            if not pending:
                return
            flushing = True
            groups = {}
            for position, (_request, query, _key) in enumerate(chunk):
                groups.setdefault(len(self.source(query["index"])[0]), []).append(position)
            losses = {}
            for positions in groups.values():
                if not any(position in pending for position in positions):
                    continue
                references = [self.sound_target(chunk[position][1]["index"]) for position in positions]
                frames = len(self.source(chunk[positions[0]][1]["index"])[0])
                audio = np.zeros((len(positions), frames, 2), dtype=np.float32)
                for row, position in enumerate(positions):
                    if position in pending:
                        audio[row] = pending[position][0]
                values = spectral_loss(self.measurements(torch.from_numpy(audio).transpose(1, 2)), concatenate_targets(references))["loss"].tolist()
                losses.update(zip(positions, values, strict=True))
            for position, (_audio, receipt) in pending.items():
                _request, query, key = chunk[position]
                value = losses[position]
                if not np.isfinite(value) or value < 0:
                    raise ValueError("native direct sound loss must be finite and nonnegative")
                _source, metadata = self.source(query["index"])
                row = {"query": query, "loss": value, "receipt": receipt, "source_audio": metadata, "batching": self.batching.to_dict()}
                append_durable(self.output / "probes.jsonl", row)
                self.probes[key] = row
            pending.clear()
            flushing = False

        # Include cached positions in planning so interruption/recovery cannot change a reduction layout.
        for start in range(0, len(planned), self.batching.spectral):
            chunk = planned[start:start + self.batching.spectral]
            try:
                for position, (request, query, key) in enumerate(chunk):
                    if key in self.probes:
                        continue
                    if self.native is None:
                        raise ValueError("offline replay is missing a measured native probe")
                    self.check_limits()
                    _step, _observation, index, candidate = request
                    cell = self.inputs.plan["fit_audit_cells"][index]
                    performance = Performance.from_dict(cell["recipe"]["performances"][cell["playback_view"]])
                    audio, receipt = self.native.render(candidate, performance, "training", probe_label(query))
                    if audio is None:
                        raise RuntimeError("direct sound probe was not accepted")
                    require_request(self.inputs.codec, self.inputs.plan, candidate, performance, receipt)
                    pending[position] = audio, receipt
                flush()
            except BaseException:
                # Only accepted rows are recorded; unrendered/cache slots are harmless discarded dummy measurements.
                if not flushing:
                    flush()
                raise
        return [self.probes[key]["loss"] for key in keys]

    def evaluate_many(self, step, observations, cell_lists, requests):
        physical = [(step, observations[observation], cell, patch) for observation, patch in requests for cell in cell_lists[observation]]
        values = iter(self.probe_many(physical))
        return [sum(next(values) for _cell in cell_lists[observation]) / len(cell_lists[observation]) for observation, _patch in requests]


def probe_label(query):
    return f"probe:{query['step']:03d}:{query['observation']}:{query['index']:03d}:{canonical_sha256(query)[:16]}"


def tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(map(tensor_bytes, value))
    return 0


def initialize(inputs):
    model, _frontend, _codec = restore_direct(torch.load(inputs.initial_path, map_location="cpu", weights_only=True))
    models = {arm: copy.deepcopy(model).train() for arm in ("parameter", "spectral")}
    optimizers = {arm: torch.optim.Adam(model.correction.parameters(), lr=inputs.plan["optimizer"]["learning_rate"]) for arm, model in models.items()}
    seed = inputs.plan["optimizer"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return models, optimizers


def step(models, optimizers, inputs, store, index):
    policy = inputs.plan.get("native_acceptance")
    if policy is not None and (policy != ACCEPTANCE_SPEC or inputs.plan.get("continuous_auxiliary") is not None):
        raise ValueError("native acceptance requires its fixed sound-only contract")
    update = inputs.plan["updates"][index]
    cell_lists = inputs.observations(update)
    observations = []
    for cells in cell_lists:
        cell = inputs.plan["fit_audit_cells"][cells[0]]
        observations.append({"cells": cells, "features": inputs.features[inputs.key(cell)],
                             "target": PilotTarget.from_dict(cell["recipe"]["target"]),
                             "baseline": inputs.cases[cells[0]]["rows"]["initial"]["scores"]["spectral"]})

    def evaluate(observation, candidate):
        costs = [store.probe(index + 1, observation, cell, candidate) for cell in cell_lists[observation]]
        return sum(costs) / len(costs)

    before = capture(inputs, models["spectral"], anchor_cells=[inputs.plan["fit_audit_cells"][cells[0]] for cells in cell_lists]) if policy is not None else None
    def evaluate_many(requests):
        return store.evaluate_many(index + 1, list(range(len(cell_lists))), cell_lists, requests)
    result = matched_step(models, optimizers, inputs.codec, observations, evaluate, index, len(inputs.plan["updates"]),
                          norm_cap=inputs.plan["optimizer"]["gradient_norm_cap"], continuous_auxiliary=inputs.plan.get("continuous_auxiliary"),
                          evaluate_many=evaluate_many, model_batch_size=BatchConfig.from_plan(inputs.plan).model)
    if before is not None:
        result["native_acceptance"] = accept_update(inputs, models["spectral"], before, cell_lists, observations, result, evaluate, evaluate_many=evaluate_many)
    return result


def replay_training(inputs, store, saved, prefix=None):
    models, optimizers = initialize(inputs)
    if saved["plan_id"] != canonical_sha256(inputs.plan) or saved["step"] != len(saved["trace"]):
        raise ValueError("direct checkpoint contract changed")
    for index, row in enumerate(saved["trace"]):
        result = step(models, optimizers, inputs, store, index)
        expected = {"step": index + 1, "update": inputs.plan["updates"][index], **result}
        if expected != {key: value for key, value in row.items() if key != "attempt"}:
            raise ValueError("direct native training trace cannot be reconstructed")
        if prefix and index + 1 == prefix["replayed_updates"] and training_signature(models, optimizers) != prefix["training_state_sha256"]:
            raise ValueError("direct early replay signature changed")
    if (not all(equal_tree(model.state_dict(), saved["models"][arm]) and equal_tree(optimizers[arm].state_dict(), saved["optimizers"][arm])
                for arm, model in models.items()) or not equal_tree(capture_rng("cpu"), saved["rng"])):
        raise ValueError("direct models, Adam or RNG do not reproduce exactly")
    return models, optimizers


def require_charges(plan, charges, trace):
    if len(charges) > plan["limits"]["optimizer_attempts"] or len(trace) > len(plan["updates"]):
        raise ValueError("direct update attempt allocation exhausted")
    steps = []
    for ordinal, charge in enumerate(charges):
        number = charge["step"]
        if (charge["attempt"] != ordinal or not 1 <= number <= len(plan["updates"])
                or charge["update"] != plan["updates"][number - 1] or number not in (len(steps), len(steps) + 1)):
            raise ValueError("direct optimizer attempt ledger changed")
        if number == len(steps) + 1:
            steps.append(number)
    if len(charges) - len(steps) > 1 or len(steps) not in (len(trace), len(trace) + 1):
        raise ValueError("direct fit permits at most one terminal interrupted attempt")
    for index, row in enumerate(trace):
        if row["step"] != index + 1 or row["attempt"] >= len(charges) or charges[row["attempt"]] != {key: row[key] for key in ("attempt", "step", "update")}:
            raise ValueError("direct committed update does not match its charge")


def train(inputs, store, output, saved=None):
    models, optimizers = initialize(inputs)
    trace = []
    if saved:
        if saved["plan_id"] != canonical_sha256(inputs.plan) or saved["step"] != len(saved["trace"]):
            raise ValueError("direct resume checkpoint changed")
        for arm, model in models.items():
            model.load_state_dict(saved["models"][arm])
            model.require_frozen()
            optimizers[arm].load_state_dict(saved["optimizers"][arm])
        restore_rng(saved["rng"], "cpu")
        trace = list(saved["trace"])
    charges = lines(output / "updates.jsonl")
    require_charges(inputs.plan, charges, trace)
    for index in range(len(trace), len(inputs.plan["updates"])):
        store.check_limits()
        if len(charges) >= inputs.plan["limits"]["optimizer_attempts"]:
            raise RuntimeError("direct optimizer attempt allocation exhausted")
        charge = {"attempt": len(charges), "step": index + 1, "update": inputs.plan["updates"][index]}
        append_durable(output / "updates.jsonl", charge)
        charges.append(charge)
        result = step(models, optimizers, inputs, store, index)
        trace.append({**charge, **result})
        snapshot = {"plan_id": canonical_sha256(inputs.plan), "step": index + 1, "trace": trace,
                    "models": {arm: model.state_dict() for arm, model in models.items()},
                    "optimizers": {arm: optimizer.state_dict() for arm, optimizer in optimizers.items()}, "rng": capture_rng("cpu")}
        atomic_torch_save(output / "checkpoint.pt", snapshot)
        if index + 1 == inputs.plan["prefix_check_updates"] and not (output / "prefix-verification.json").exists():
            before = capture_rng("cpu")
            try:
                replay_models, replay_opts = replay_training(inputs, SoundStore(output, inputs), torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True))
                expected = inputs.plan.get("expected_parameter_prefix_model_sha256")
                if expected and tensor_digest(replay_models["parameter"].state_dict()) != expected:
                    raise ValueError("auxiliary fit changed the matched parameter-control prefix")
                shutil.copyfile(output / "checkpoint.pt", output / "prefix-checkpoint.pt")
                proof = {"plan_id": canonical_sha256(inputs.plan), "replayed_updates": index + 1,
                    "training_state_sha256": training_signature(replay_models, replay_opts), "offline_training_replay_exact": True,
                    "checkpoint_file_sha256": digest(output / "prefix-checkpoint.pt"), "native_requests_added": 0, "experimental_optimizer_updates_added": 0}
                if inputs.plan.get("require_trainable_input_updates"):
                    from .trainable_basic_fit import check_input_learning
                    proof["input_branch_updates"] = check_input_learning(inputs, replay_models, replay_opts, prefix=True)
                write_json(output / "prefix-verification.json", proof)
            finally:
                restore_rng(before, "cpu")
        progress = {"phase": "fit", "updates": index + 1, "planned_updates": len(inputs.plan["updates"]),
                    "probes": len(store.probes), "losses": result["losses"], "clamping": result["clamping"]["spectral"]}
        if "native_acceptance" in result:
            accepted = result["native_acceptance"]
            progress["native_acceptance"] = {"accepted_updates": sum(row["native_acceptance"]["accepted"] for row in trace),
                                             "last_scale": accepted["scale"], "before_objective": accepted["before_objective"],
                                             "committed_objective": accepted["committed_objective"], "trials": len(accepted["trials"])}
        write_json(output / "progress.json", progress)
    return models, trace


def charge_execution(inputs, output, resume):
    plan, executions = inputs.plan, lines(output / "executions.jsonl")
    if (read(output / "plan.json") != plan or len(executions) >= plan["limits"]["native_executions"]
            or bool(executions) != resume or (output / "run.json").exists() != resume):
        raise ValueError("direct plan, execution or recovery allocation changed")
    if resume:
        old, events = read(output / "summary.json"), lines(output / "renders.jsonl")
        failures = [row for row in events if row["event"] == "outcome" and row["status"] != "accepted"]
        if (old["complete"] or not old["failure"] or not old["within_resource_limits"] or not failures or events[-1] != failures[-1]
                or any(row.get("error_type") != "RenderTimeout" or "qa" in row for row in failures)):
            raise ValueError("direct recovery requires a terminal known identical native timeout")
        root = output / "interruption"
        root.mkdir()
        for name in ("summary.json", "checkpoint.pt", "checkpoint.previous.pt", "updates.jsonl", "probes.jsonl", "finals.jsonl", "renders.jsonl"):
            if (output / name).exists():
                shutil.copy2(output / name, root / name)
    else:
        write_json(output / "run.json", {"plan_id": canonical_sha256(plan), "started_unix": time()})
    append_durable(output / "executions.jsonl", {"ordinal": len(executions) + 1, "plan_id": canonical_sha256(plan), "resume": resume, "started_unix": time()})


def execute(inputs, output, *, resume=False):
    from .basic_direct_fit_audit import decisions
    charge_execution(inputs, output, resume)
    monitor = MemoryMonitor()
    monitor.start()
    session = PairedSession(output, inputs.plan, inputs.codec, monitor)
    store = SoundStore(output, inputs, session)
    failure, result = None, None
    try:
        saved = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True) if (output / "checkpoint.pt").exists() else None
        models, trace = train(inputs, store, output, saved)
        expected = inputs.plan.get("expected_parameter_model_sha256")
        if expected and tensor_digest(models["parameter"].state_dict()) != expected:
            raise ValueError("auxiliary fit changed the matched parameter-control model")
        collect_finals(inputs, store, models)
        result = decisions(inputs, store, trace)
        for arm, model in models.items():
            artifact = export_direct(model, inputs.frontend, inputs.codec, inputs.plan["pool_id"])
            artifact.update(training_objective=arm, training_plan_id=canonical_sha256(inputs.plan))
            atomic_torch_save(output / f"diagnostic-{arm}.pt", artifact)
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        monitor.close()
        plan = inputs.plan
        archive = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
        elapsed = time() - session.started
        within = bool(monitor.peak_bytes and monitor.peak_bytes <= plan["limits"]["process_tree_bytes"] and not monitor.errors
                      and elapsed <= plan["limits"]["wall_seconds"] and archive + 1024**2 <= plan["limits"]["archive_bytes"])
        resolved, recovered = False, []
        try:
            _attempts, _outcomes, recovered = native_ledger(session.events, plan["expected_renderer_identity"])
            resolved = True
        except ValueError:
            pass
        committed = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)["step"] if (output / "checkpoint.pt").exists() else 0
        report = {"plan_id": canonical_sha256(plan), "complete": failure is None and result is not None, "failure": failure, "result": result,
                  "committed_updates_per_arm": committed, "native_receipts_resolved": resolved, "recovered_timeouts": recovered,
                  "elapsed_seconds": elapsed, "peak_process_tree_rss_bytes": monitor.peak_bytes, "rss_sampling_errors": monitor.errors,
                  "retained_archive_bytes_before_summary": archive, "within_resource_limits": within,
                  "render_attempts": {phase: sum(row["event"] == "attempt" and row["phase"] == phase for row in session.events) for phase in ("training", "qualification")},
                  "matched_comparison_passed": bool(result and result["pass"] and within and resolved), "qualified_for_release": False, "test_split_used": False}
        write_json(output / "summary.json", report)
    return report


def run(parent, bridge, recent, cache, pool, output, *, mode="prepare", batching=None):
    started, monitor = perf_counter(), MemoryMonitor()
    monitor.start()
    with refresh.cpu_settings():
        try:
            plan = make_plan(parent, bridge, recent, cache, pool, batching=batching)
            inputs = Inputs(parent, bridge, cache, plan)
            inputs.validate()
        finally:
            monitor.close()
        preparation = {"elapsed_seconds": perf_counter() - started, "peak_process_tree_rss_bytes": monitor.peak_bytes,
                       "rss_sampling_errors": monitor.errors, "native_requests": 0, "optimizer_updates": 0}
        if preparation["elapsed_seconds"] >= 600 or not monitor.peak_bytes or monitor.peak_bytes > 4 * 1024**3 or monitor.errors:
            raise ValueError("direct preparation exceeded ten minutes or 4 GiB")
        if mode == "prepare":
            output.mkdir(parents=True, exist_ok=False)
            write_json(output / "plan.json", plan)
            write_json(output / "preparation.json", preparation)
            return {"plan_id": canonical_sha256(plan), "preparation": preparation, "storage_bound": plan["storage_bound"]}
        if mode == "verify":
            from .basic_direct_fit_audit import run_verification
            return run_verification(inputs, output)
        if mode not in ("execute", "resume"):
            raise ValueError("unknown direct fitting mode")
        return execute(inputs, output, resume=mode == "resume")

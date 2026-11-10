"""On-demand finite-choice sound fitting with fixed continuous/filter predictions."""

from collections import OrderedDict
import copy
import json
import shutil
from time import perf_counter, time

import torch

from obruxo_data.hashing import canonical_sha256
from obruxo_data.midi import Performance
from obruxo_data.render.qa import audio_float32_sha256

from . import category_scale_refresh as refresh
from .audit import write_json
from .benchmark import MemoryMonitor
from .batching import BatchConfig, predict_cells
from .checkpoints import atomic_torch_save
from .conditioned_bridge import read
from .native_fit import append_durable, digest
from .paired_native_audit import equal_tree, training_signature
from .paired_native_evidence import RetainedAuditSession, WaveformAudit, native_ledger, rescore_waveform
from .paired_native_fit import PairedSession
from .paired_performance_objective import seed_external_streams
from .trainer import capture_rng, restore_rng
from .unison_choice import export_unison, measured_choice_loss, restore_unison
from .unison_choice_bridge import correction_gradients
from .unison_fit_data import Inputs, make_plan, require_request


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


class CostStore:
    def __init__(self, output, inputs, session=None):
        self.output, self.inputs, self.session = output, inputs, session
        self.costs = {(row["index"], row["choice"]): row for row in lines(output / "costs.jsonl")}
        self.finals = {(row["index"], row["arm"]): row for row in lines(output / "finals.jsonl")}
        if len(self.costs) != len(lines(output / "costs.jsonl")) or len(self.finals) != len(lines(output / "finals.jsonl")):
            raise ValueError("duplicate unison cost or final comparison")
        self.source_cache = OrderedDict()
        self.initial_rows = {}

    def check_limits(self):
        if self.session:
            self.session.check_limits()
        if sum(path.stat().st_size for path in self.output.rglob("*") if path.is_file()) + 32 * 1024**2 > self.inputs.plan["limits"]["archive_bytes"]:
            raise RuntimeError("unison evidence archive allocation exhausted")

    def source(self, index):
        cell = self.inputs.plan["fit_audit_cells"][index]
        sha = cell["recipe"]["source_audio_sha256"][cell["playback_view"]]
        path = self.output / "source_audio" / f"{sha}.pt"
        if sha not in self.source_cache:
            if path.exists():
                value = torch.load(path, map_location="cpu", weights_only=True)["audio"]
            else:
                if self.session is None:
                    raise ValueError("retained unison source waveform is missing")
                value, _receipt = self.inputs.source(index)
                path.parent.mkdir(exist_ok=True)
                atomic_torch_save(path, {"audio": value})
            if audio_float32_sha256(value.numpy()) != sha:
                raise ValueError("retained unison source audio changed")
            self.source_cache[sha] = value
            while len(self.source_cache) > 4:
                self.source_cache.popitem(last=False)
        self.source_cache.move_to_end(sha)
        return self.source_cache[sha].numpy(), {"file": path.relative_to(self.output).as_posix(), "sha256": digest(path)}

    def initial(self, index, *, audit=None):
        if index in self.initial_rows:
            return self.initial_rows[index]
        row = self.inputs.cases[index]["rows"]["initial"]
        receipt = row["receipt"]
        retained = receipt["retained_audio"]
        expected = f"comparison_audio/{receipt['qa']['audio_float32_sha256']}.pt"
        if retained["file"] != expected:
            raise ValueError("parent initial waveform path changed")
        path = self.output / expected
        if not path.exists():
            if self.session is None:
                raise ValueError("retained initial comparison waveform is missing")
            path.parent.mkdir(exist_ok=True)
            shutil.copyfile(self.inputs.parent / expected, path)
        reference, source = self.source(index)
        if audit is not None:
            audit.add(self.output, receipt, reference, self.inputs.codec.contract.tail_seconds, row["scores"])
        elif rescore_waveform(self.output, receipt, reference, self.inputs.codec.contract.tail_seconds) != row["scores"]:
            raise ValueError("parent initial sound score changed")
        result = {"index": index, "choice": row["candidate"]["categorical"][0], "candidate": row["candidate"],
                  "origin": "parent", "scores": row["scores"], "receipt": receipt, "source_audio": source}
        self.initial_rows[index] = result
        return result

    def render(self, index, candidate, phase, label):
        return self.render_many([(index, candidate, phase, label)])[0]

    def render_many(self, requests, *, on_result=None):
        from .paired_native_evidence import rescore_waveforms
        if self.session is None:
            raise ValueError("offline replay attempted a native render")
        size = BatchConfig.from_plan(self.inputs.plan).spectral
        pending, results, flushing = [], [], False
        def flush():
            nonlocal flushing
            if not pending:
                return
            flushing = True
            scores = rescore_waveforms(((self.output, row["receipt"], reference, self.inputs.codec.contract.tail_seconds)
                                       for row, reference in pending), batch_size=size)
            for (row, _reference), score in zip(pending, scores, strict=True):
                row = {**row, "scores": score}
                if on_result is not None:
                    on_result(row)
                results.append(row)
            pending.clear()
            flushing = False
        try:
            for index, candidate, phase, label in requests:
                self.check_limits()
                cell = self.inputs.plan["fit_audit_cells"][index]
                performance = Performance.from_dict(cell["recipe"]["performances"][cell["playback_view"]])
                audio, receipt = self.session.render(candidate, performance, phase, label)
                if audio is None:
                    raise RuntimeError("unison native render was not accepted")
                require_request(self.inputs.codec, self.inputs.plan, candidate, performance, receipt)
                reference, source = self.source(index)
                pending.append(({"index": index, "candidate": candidate.to_dict(), "origin": "new", "receipt": receipt, "source_audio": source}, reference))
                if len(pending) >= size:
                    flush()
        except BaseException:
            if not flushing:
                flush()
            raise
        flush()
        return results

    def cost(self, index, choice):
        key = index, choice
        candidate = self.inputs.target(index, choice)
        if key in self.costs:
            if self.costs[key]["candidate"] != candidate.to_dict():
                raise ValueError("cached unison cost patch changed")
            return self.costs[key]
        self.check_limits()
        initial = self.inputs.cases[index]["rows"]["initial"]
        row = self.initial(index) if candidate.to_dict() == initial["candidate"] else self.render(index, candidate, "training", f"cost:{index:03d}:{choice}")
        row = {**row, "choice": choice}
        append_durable(self.output / "costs.jsonl", row)
        self.costs[key] = row
        write_json(self.output / "progress.json", {"phase": "costs", "costs": len(self.costs), "planned_costs": 252, "updates": 0})
        return row

    def collect(self):
        for index, cell in enumerate(self.inputs.plan["fit_audit_cells"]):
            if cell["training_index"] is None:
                continue
            for choice in range(3):
                self.cost(index, choice)

    def vectors(self, examples):
        result = []
        for example in examples:
            recipe = self.inputs.plan["training"][example["state"]]
            own, transfer = (self.inputs.lookup[(recipe["state_id"], example["input_view"], playback)]
                             for playback in (example["input_view"], example["view"]))
            result.append([(self.costs[(own, choice)]["scores"]["spectral"] + self.costs[(transfer, choice)]["scores"]["spectral"]) / 2 for choice in range(3)])
        return result

    def signal(self):
        groups = {}
        for view in (1, 2):
            initial, oracle = [], []
            for state, recipe in enumerate(self.inputs.plan["training"]):
                vector = self.vectors([{"state": state, "input_view": view, "view": 0}])[0]
                index = self.inputs.lookup[(recipe["state_id"], view, view)]
                initial.append(vector[self.inputs.target(index).categorical[0]])
                oracle.append(min(vector))
            groups[str(view)] = {"initial_mean": sum(initial) / len(initial), "oracle_mean": sum(oracle) / len(oracle)}
            groups[str(view)]["available_gain"] = groups[str(view)]["initial_mean"] - groups[str(view)]["oracle_mean"]
        return {"groups": groups, "pass": all(row["available_gain"] > 0 for row in groups.values())}


def initialize(inputs):
    model, _front, _codec = restore_unison(torch.load(inputs.initial_path, map_location="cpu", weights_only=True))
    models = {name: copy.deepcopy(model).train() for name in ("parameter", "spectral")}
    optimizers = {name: torch.optim.Adam(network.correction.parameters(), lr=inputs.plan["optimizer"]["learning_rate"]) for name, network in models.items()}
    seed_external_streams(inputs.plan)
    torch.manual_seed(inputs.plan["optimizer"]["seed"])
    return models, optimizers


def step(models, optimizers, inputs, store, examples):
    cells = [inputs.plan["fit_audit_cells"][inputs.lookup[(inputs.plan["training"][row["state"]]["state_id"], row["input_view"], row["input_view"])]]
             for row in examples]
    for optimizer in optimizers.values():
        optimizer.zero_grad(set_to_none=True)
    predictions = {name: [inputs.predict(model, cell) for cell in cells] for name, model in models.items()}
    scores = {name: torch.cat([value["categorical"][0] for value in values]) for name, values in predictions.items()}
    target = torch.tensor([inputs.plan["training"][row["state"]]["target"]["categorical"][0] for row in examples])
    losses = {"parameter": torch.nn.functional.cross_entropy(scores["parameter"], target),
              "spectral": measured_choice_loss({"categorical": [scores["spectral"]]}, store.vectors(examples))}
    gradients = {}
    for name, loss in losses.items():
        if not torch.isfinite(loss):
            raise ValueError("nonfinite unison fitting objective")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(models[name].correction.parameters(), inputs.plan["optimizer"]["gradient_norm_cap"])
        if not torch.isfinite(norm):
            raise ValueError("nonfinite unison fitting gradient")
        gradients[name] = correction_gradients(models[name])
    for name, optimizer in optimizers.items():
        optimizer.step()
        models[name].require_frozen()
        with torch.no_grad():
            for cell in cells:
                inputs.predict(models[name], cell)
    return {"losses": {name: float(loss.detach()) for name, loss in losses.items()}, "gradients": gradients,
            "choices": {name: value.detach().argmax(-1).tolist() for name, value in scores.items()}, "continuous_filter_unchanged": True}


def replay_training(inputs, store, saved, prefix=None):
    models, optimizers = initialize(inputs)
    if saved["plan_id"] != canonical_sha256(inputs.plan) or saved["step"] != len(saved["trace"]):
        raise ValueError("unison checkpoint contract changed")
    for index, row in enumerate(saved["trace"]):
        expected = {"step": index + 1, "examples": inputs.plan["updates"][index]["examples"]}
        result = step(models, optimizers, inputs, store, expected["examples"])
        if {**expected, **result} != {key: value for key, value in row.items() if key != "attempt"}:
            raise ValueError("unison update trace cannot be reconstructed")
        if prefix and index + 1 == prefix["replayed_updates"] and training_signature(models, optimizers) != prefix["training_state_sha256"]:
            raise ValueError("unison early replay signature changed")
    if (not all(equal_tree(model.state_dict(), saved["models"][name]) and equal_tree(optimizers[name].state_dict(), saved["optimizers"][name])
                for name, model in models.items()) or not equal_tree(capture_rng("cpu"), saved["rng"])):
        raise ValueError("unison models, Adam or RNG do not reproduce exactly")
    return models, optimizers


def train(inputs, store, output, saved=None):
    models, optimizers = initialize(inputs)
    trace = []
    if saved:
        if saved["plan_id"] != canonical_sha256(inputs.plan) or saved["step"] != len(saved["trace"]):
            raise ValueError("unison resume checkpoint changed")
        for name, model in models.items():
            model.load_state_dict(saved["models"][name])
            model.require_frozen()
            optimizers[name].load_state_dict(saved["optimizers"][name])
        restore_rng(saved["rng"], "cpu")
        trace = list(saved["trace"])
    charges = lines(output / "updates.jsonl")
    if len(charges) > inputs.plan["limits"]["optimizer_attempts"] or any(row["attempt"] != index or row["step"] != index + 1
            or row["examples"] != inputs.plan["updates"][index]["examples"] for index, row in enumerate(charges)):
        raise ValueError("unison optimizer-attempt ledger changed")
    if len(charges) != len(trace):
        raise ValueError("unison interrupted optimization must have a committed checkpoint")
    for index in range(len(trace), len(inputs.plan["updates"])):
        store.check_limits()
        if len(charges) >= inputs.plan["limits"]["optimizer_attempts"]:
            raise RuntimeError("unison optimizer attempt allocation exhausted")
        examples = inputs.plan["updates"][index]["examples"]
        charge = {"attempt": len(charges), "step": index + 1, "examples": examples}
        append_durable(output / "updates.jsonl", charge)
        charges.append(charge)
        result = step(models, optimizers, inputs, store, examples)
        trace.append({**charge, **result})
        snapshot = {"plan_id": canonical_sha256(inputs.plan), "step": index + 1, "trace": trace,
                    "models": {name: model.state_dict() for name, model in models.items()},
                    "optimizers": {name: optimizer.state_dict() for name, optimizer in optimizers.items()}, "rng": capture_rng("cpu")}
        atomic_torch_save(output / "checkpoint.pt", snapshot)
        if index + 1 == inputs.plan["prefix_check_updates"] and not (output / "prefix-verification.json").exists():
            before = capture_rng("cpu")
            try:
                check_models, check_opts = replay_training(inputs, store, torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True))
                proof = {"plan_id": canonical_sha256(inputs.plan), "replayed_updates": index + 1, "training_state_sha256": training_signature(check_models, check_opts),
                         "offline_training_replay_exact": True, "native_requests_added": 0, "optimizer_updates_added": 0,
                         "checkpoint_file_sha256": digest(output / "checkpoint.pt")}
                shutil.copyfile(output / "checkpoint.pt", output / "prefix-checkpoint.pt")
                write_json(output / "prefix-verification.json", proof)
            finally:
                restore_rng(before, "cpu")
        write_json(output / "progress.json", {"phase": "fit", "updates": index + 1, "planned_updates": len(inputs.plan["updates"]), "losses": result["losses"]})
    return models, trace


def collect_finals(inputs, store, models):
    with torch.no_grad():
        banks = {arm: predict_cells(inputs, model.eval(), inputs.plan["fit_audit_cells"]) for arm, model in models.items()}
    requests, arms = [], []
    audit = WaveformAudit(BatchConfig.from_plan(inputs.plan).spectral)
    for index, cell in enumerate(inputs.plan["fit_audit_cells"]):
        store.initial(index, audit=audit)
        for arm, model in models.items():
            candidate = inputs.target(index, prediction=banks[arm][index])
            key = index, arm
            if key in store.finals:
                if store.finals[key]["candidate"] != candidate.to_dict():
                    raise ValueError("unison final comparison changed")
                continue
            requests.append((index, candidate, "qualification", f"final:{index:03d}:{arm}"))
            arms.append(arm)
    audit.finish()
    pending_arms = iter(arms)
    def publish(row):
        arm = next(pending_arms)
        row = {**row, "arm": arm}
        append_durable(store.output / "finals.jsonl", row)
        store.finals[row["index"], arm] = row
        write_json(store.output / "progress.json", {"phase": "finals", "updates": len(inputs.plan["updates"]), "comparisons": len(store.finals)})
    store.render_many(requests, on_result=publish)


def execute(inputs, output, *, resume=False):
    from .unison_fit_audit import decisions
    plan = inputs.plan
    if read(output / "plan.json") != plan:
        raise ValueError("unison fitting plan changed")
    executions = lines(output / "executions.jsonl")
    if len(executions) >= plan["limits"]["native_executions"] or bool(executions) != resume or (output / "run.json").exists() != resume:
        raise ValueError("unison execution/recovery allocation changed or exhausted")
    if resume:
        old = read(output / "summary.json")
        events = lines(output / "renders.jsonl")
        failures = [row for row in events if row["event"] == "outcome" and row["status"] != "accepted"]
        if (not old["failure"] or not old["within_resource_limits"] or not failures or events[-1] != failures[-1]
                or any(row.get("error_type") != "RenderTimeout" or "qa" in row for row in failures)):
            raise ValueError("unison fit permits only a terminal known native timeout recovery")
        root = output / "interruption"
        root.mkdir()
        for name in ("summary.json", "checkpoint.pt", "checkpoint.previous.pt", "updates.jsonl", "costs.jsonl", "finals.jsonl", "renders.jsonl"):
            if (output / name).exists():
                shutil.copyfile(output / name, root / name)
        write_json(root / "evidence.json", {path.name: digest(path) for path in root.iterdir() if path.is_file()})
    else:
        write_json(output / "run.json", {"plan_id": canonical_sha256(plan), "started_unix": time()})
    append_durable(output / "executions.jsonl", {"ordinal": len(executions) + 1, "resume": resume, "plan_id": canonical_sha256(plan), "started_unix": time()})
    monitor = MemoryMonitor()
    monitor.start()
    session = PairedSession(output, plan, inputs.codec, monitor)
    store = CostStore(output, inputs, RetainedAuditSession(session, output))
    signal, result, failure, trace = None, None, None, []
    try:
        store.collect()
        signal = store.signal()
        write_json(output / "signal.json", signal)
        if signal["pass"]:
            saved = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True) if (output / "checkpoint.pt").exists() else None
            models, trace = train(inputs, store, output, saved)
            collect_finals(inputs, store, models)
            result = decisions(inputs, store, signal)
            for name, model in models.items():
                artifact = export_unison(model, inputs.frontend, inputs.codec, plan["pool_id"])
                artifact.update(training_objective=name, training_plan_id=canonical_sha256(plan))
                atomic_torch_save(output / f"diagnostic-{name}.pt", artifact)
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        monitor.close()
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
        if (output / "checkpoint.pt").exists():
            committed = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)["step"]
        else:
            committed = 0
        complete = failure is None and signal is not None and (not signal["pass"] or result is not None)
        report = {"plan_id": canonical_sha256(plan), "complete": complete, "failure": failure, "signal": signal, "result": result,
                  "committed_updates_per_arm": committed, "native_receipts_resolved": resolved, "recovered_timeouts": recovered,
                  "elapsed_seconds": elapsed, "peak_process_tree_rss_bytes": monitor.peak_bytes, "rss_sampling_errors": monitor.errors,
                  "retained_archive_bytes_before_summary": archive, "within_resource_limits": within,
                  "render_attempts": {phase: sum(row["event"] == "attempt" and row["phase"] == phase for row in session.events) for phase in ("training", "qualification")},
                  "matched_comparison_passed": bool(complete and result and result["pass"] and within and resolved),
                  "qualified_for_release": False, "test_split_used": False}
        write_json(output / "summary.json", report)
    return report


def run(parent, bridge, cache, pool, output, *, mode="prepare"):
    started, monitor = perf_counter(), MemoryMonitor()
    monitor.start()
    with refresh.cpu_settings():
        try:
            plan = make_plan(parent, bridge, cache, pool)
            inputs = Inputs(parent, bridge, cache, plan)
            inputs.validate()
        finally:
            monitor.close()
        preparation = {"elapsed_seconds": perf_counter() - started, "peak_process_tree_rss_bytes": monitor.peak_bytes,
                       "rss_sampling_errors": monitor.errors, "native_requests": 0, "optimizer_updates": 0}
        if preparation["elapsed_seconds"] >= 600 or monitor.peak_bytes is None or monitor.peak_bytes > 4 * 1024**3 or monitor.errors:
            raise ValueError("unison preparation exceeded ten minutes or 4 GiB")
        if mode == "prepare":
            output.mkdir(parents=True, exist_ok=False)
            write_json(output / "plan.json", plan)
            write_json(output / "preparation.json", preparation)
            return {"plan_id": canonical_sha256(plan), "preparation": preparation, "storage_bound": plan["storage_bound"]}
        if mode == "verify":
            from .unison_fit_audit import run_verification
            return run_verification(inputs, output)
        if mode not in ("execute", "resume"):
            raise ValueError("unknown unison fitting mode")
        return execute(inputs, output, resume=mode == "resume")

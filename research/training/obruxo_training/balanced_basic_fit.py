"""One sound-trained candidate with identical musical coverage across three patches."""

import copy
import random
import shutil
from collections import Counter
from pathlib import Path
from time import time

import numpy as np
import torch
from obruxo_data.hashing import canonical_sha256, sha256_bytes
from obruxo_data.render.qa import audio_float32_sha256

from .audit import write_json
from .balanced_basic_data import validate_plan as validate_data
from .basic_direct import LOSS, REHEARSAL, restore_direct
from .basic_direct_fit import SoundStore, charge_execution, tensor_bytes
from .basic_direct_fit_data import Inputs, storage_bound
from .benchmark import MemoryMonitor
from .batching import BatchConfig, EXECUTION_BATCH_SOURCES, predict_cells, predict_features
from .checkpoints import atomic_torch_save
from .conditioned_bridge import read
from .features import tensor_digest, validate_features
from .inward_basic import export_inward, training_model
from .native_fit import append_durable, digest
from .native_acceptance import SPEC as ACCEPTANCE
from .paired_native_audit import equal_tree
from .paired_native_fit import PairedSession
from .shared_basic_auxiliary import step as shared_step
from .trainable_basic_fit import source_code
from .trainer import capture_rng, restore_rng
from .unison_sound_fit import lines

SPEC = {"format": "balanced_basic_native_sound_fit_v1", "training_indices": [2, 8, 11], "grid_order": [0, 3, 4, 1, 2, 5],
        "updates": 48, "learning_rate": .001, "gradient_norm_cap": 5., "seed": 0, "additional_outputs": 0,
        "old_comparisons": 104, "grid_comparisons": 36, "training_only_cells": 18, "input_bank": 74}
LIMITS = {"wall_seconds": 21600, "process_tree_bytes": 4 * 1024**3, "archive_bytes": 2 * 1024**3,
          "replay_bytes": 512 * 1024**2, "training_attempts": 7600, "qualification_attempts": 400,
          "native_executions": 2, "optimizer_attempts": 49}
SOURCES = ("balanced_basic_fit.py", "balanced_basic_audit.py", "reports/scripts/run_balanced_basic_fit.py", "reports/BALANCED_BASIC_FIT_CARD.md")


def schedule(updates):
    counts, result = Counter(), []
    for update in updates:
        state = update["examples"][0]["state"]
        if state not in SPEC["training_indices"] or {row["state"] for row in update["examples"]} != {state}:
            raise ValueError("balanced schedule must retain the preceding three-patch order")
        view = 3 + SPEC["grid_order"][counts[state] % 6]
        counts[state] += 1
        result.append({"examples": [{"state": state, "input_view": 0, "view": view},
                                    {"state": state, "input_view": view, "view": 0}], "references": list(update["references"])})
    if len(result) != 48 or any(counts[state] != 16 for state in SPEC["training_indices"]):
        raise ValueError("balanced fitting requires sixteen primary updates per patch")
    return result


def extend_cells(prior, data, records):
    cells, recipes = copy.deepcopy(prior["fit_audit_cells"]), {}
    if len(cells) != 104:
        raise ValueError("balanced fitting must preserve the original 104-cell panel")
    for state in SPEC["training_indices"]:
        recipe = copy.deepcopy(prior["training"][state])
        selected = [row for row in data["observations"] if row["training_index"] == state]
        if [row["grid_index"] for row in selected] != list(range(6)):
            raise ValueError("balanced source ordering changed")
        for row in selected:
            record = records[row["index"]]
            recipe["performances"].append(row["performance"])
            recipe["source_audio_sha256"].append(record["receipt"]["qa"]["audio_float32_sha256"])
            recipe["source_feature_sha256"].append(record["features_sha256"])
            recipe["source_recipe_ids"].append(canonical_sha256({"state_id": row["state_id"], "performance_id": row["performance_id"]}))
        recipes[state] = recipe
    for row in data["observations"]:
        for playback in (3 + row["grid_index"], 0):
            cells.append({"label": f"grid-{len(cells):03d}", "training_index": row["training_index"],
                          "recipe": recipes[row["training_index"]], "input_view": 3 + row["grid_index"], "playback_view": playback})
    for row in data["observations"]:
        cells.append({"label": f"train-grid-{len(cells):03d}", "training_index": row["training_index"],
                      "recipe": recipes[row["training_index"]], "input_view": 0, "playback_view": 3 + row["grid_index"]})
    if len(cells) != 158 or cells[:104] != prior["fit_audit_cells"]:
        raise ValueError("balanced grid altered the old comparisons or omitted reciprocal training cells")
    return cells


def code_identity(plan):
    code, _differences = source_code(plan, compatible=EXECUTION_BATCH_SOURCES)
    root = Path(__file__).parents[1]
    return {**code, **{name: sha256_bytes((root / name if "/" in name else Path(__file__).with_name(name)).read_bytes().replace(b"\r\n", b"\n")) for name in (*SOURCES, *sorted(EXECUTION_BATCH_SOURCES))}}


def prepare(paths, pool, output=None, *, batching=None):
    control, data = paths["control"], paths["data"]
    prior, proof, bank, data_proof = (read(path) for path in (control / "plan.json", control / "verification.json", data / "plan.json", data / "verification.json"))
    # Import a previously verified artifact; generation/recovery still checks its exact executable source.
    validate_data(bank, pool, verify_source=False)
    if (not proof["verified"] or not proof["complete"] or not proof["models_adam_calibration_rng_replay_exact"]
            or proof["plan_id"] != canonical_sha256(prior) or proof["qualified_for_release"] or proof["test_split_used"]
            or not data_proof["verified"] or data_proof["plan_id"] != canonical_sha256(bank)
            or data_proof["source_recordings"] != 18 or data_proof["exact_repeat_pairs"] != 6
            or bank["parent_fit_plan_id"] != proof["plan_id"] or pool.pool_id != prior["pool_id"]
            or digest(paths["bridge"] / "model.pt") != prior["initial_file_sha256"]
            or digest(paths["features"]) != prior["input_cache_sha256"]
            or prior["optimizer"] != {"learning_rate": SPEC["learning_rate"], "seed": SPEC["seed"], "gradient_norm_cap": SPEC["gradient_norm_cap"]}
            or any(bank[key] != prior[key] for key in ("render_profile", "render_phase_controls", "expected_renderer_identity"))):
        raise ValueError("balanced fitting requires the audited control and matching qualified sources")
    bound = dict(prior["inputs_sha256"])
    for name, expected in prior["input_files"].items():
        if name.startswith("parent/"):
            path = paths["parent"] / name.split("/", 1)[1]
            if digest(path) != expected:
                raise ValueError("original comparison parent changed")
            bound[str(path)] = expected
    for directory, verification in ((control, proof), (data, data_proof)):
        for name, expected in verification["evidence_sha256"].items():
            path = directory / name
            if digest(path) != expected:
                raise ValueError("audited control or source evidence changed")
            bound[str(path)] = expected
        bound[str(directory / "verification.json")] = digest(directory / "verification.json")
    for path, expected in list(bound.items()):
        if digest(Path(path)) != expected:
            raise ValueError("preceding initializer, features or original comparisons changed")
    records = {row["index"]: read(data / "records" / f"{row['index']:02d}.json") for row in bank["observations"]}
    cells = extend_cells(prior, bank, records)
    batching = batching or BatchConfig()
    plan = {**prior, "batching": batching.to_dict(), "format": SPEC["format"], "balanced_spec": SPEC, "paths": {key: str(value) for key, value in paths.items()},
            "loss": LOSS, "rehearsal": copy.deepcopy(REHEARSAL), "native_acceptance": copy.deepcopy(ACCEPTANCE),
            "inputs_sha256": bound, "source": {**prior["source"], **bank["source"]}, "limits": LIMITS,
            "updates": schedule(prior["updates"]), "fit_audit_cells": cells, "evaluation_indices": list(range(140)),
            "grid_comparisons": list(range(104, 140)), "maximum_native_requests": 8000, "planned_maximum_native_requests": 7418,
            "storage_bound": storage_bound(cells, pool.codec.contract.tail_seconds, batching=batching), "control_plan_id": proof["plan_id"],
            "data_plan_id": data_proof["plan_id"], "parameter_control_path": prior["comparison_fit"],
            "model_admission_allowed": False, "qualified_for_release": False}
    for key in ("comparison_spec", "auxiliary_split", "diagnostic_split", "diagnostic_states_excluded_for_sound_exposure", "comparison_fit"):
        plan.pop(key, None)
    plan["source"] = code_identity(plan)
    inputs = BalancedInputs(paths, plan)
    if len(inputs.features) != SPEC["input_bank"]:
        raise ValueError("balanced fitting requires the complete old and new input bank")
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "plan.json", plan)
    return plan


class BalancedInputs(Inputs):
    def __init__(self, paths, plan):
        old = {**plan, "fit_audit_cells": plan["fit_audit_cells"][:104]}
        super().__init__(paths["parent"], paths["bridge"], paths["features"], old)
        self.plan, self.paths, self.data = plan, paths, paths["data"]
        self.old_lookup = dict(self.lookup)
        rows = torch.load(self.data / "features.pt", map_location="cpu", weights_only=True)["rows"]
        self.grid_records = {}
        for row in rows:
            key = row["state_id"], row["grid_index"]
            record = read(self.data / "records" / f"{row['index']:02d}.json")
            if tensor_digest(row["features"]) != record["features_sha256"]:
                raise ValueError("balanced input features changed")
            validate_features(row["features"])
            self.grid_records[key] = row["index"]
            self.features[row["state_id"], 3 + row["grid_index"]] = row["features"]
        for cell in plan["fit_audit_cells"][104:]:
            key = self.key(cell)
            if key not in self.baselines:
                self.baselines[key] = None
            self.cases.append({"label": cell["label"], "rows": {"initial": {"scores": {"spectral": 0.}}}})
        missing = [key for key, value in self.baselines.items() if value is None]
        with torch.no_grad():
            rows = predict_features(self.model, [self.features[key] for key in missing], BatchConfig.from_plan(plan).model)
        self.baselines.update(zip(missing, rows, strict=True))
        self.lookup = {(cell["recipe"]["state_id"], cell["input_view"], cell["playback_view"]): index for index, cell in enumerate(plan["fit_audit_cells"])}
        self.bytes = sum(tensor_bytes(value) for value in self.features.values())
        if self.bytes > plan["limits"]["replay_bytes"] or len(self.lookup) != 158 or len(self.grid_records) != 18:
            raise ValueError("balanced inputs are incomplete or exceed replay memory")
        self.control_model, _front, _codec = restore_direct(torch.load(paths["control"] / "diagnostic-sound_only.pt", map_location="cpu", weights_only=True))
        self.control_finals = {(row["index"], row["arm"]): row for row in lines(paths["control"] / "finals.jsonl")}
        self.parameter_finals = {(row["index"], row["arm"]): row for row in lines(Path(plan["parameter_control_path"]) / "finals.jsonl")}

    def source(self, index):
        if index < 104:
            return super().source(index)
        cell = self.plan["fit_audit_cells"][index]
        state, view = cell["recipe"]["state_id"], cell["playback_view"]
        if view == 0:
            return super().source(self.old_lookup[state, 0, 0])
        record_index = self.grid_records[state, view - 3]
        path = self.data / "records" / f"{record_index:02d}.pt"
        saved, record = torch.load(path, map_location="cpu", weights_only=True), read(path.with_suffix(".json"))
        if digest(path) != record["file_sha256"] or audio_float32_sha256(saved["audio"].numpy()) != cell["recipe"]["source_audio_sha256"][view]:
            raise ValueError("balanced source waveform changed")
        return saved["audio"], record["receipt"]


def load_inputs(output, pool):
    plan = read(output / "plan.json")
    paths = {key: Path(value) for key, value in plan["paths"].items()}
    if plan != prepare(paths, pool, batching=BatchConfig.from_plan(plan)):
        raise ValueError("frozen balanced-fit source, inputs or allocation changed")
    return BalancedInputs(paths, plan)


class BalancedStore(SoundStore):
    def __init__(self, output, inputs, native=None):
        super().__init__(output, inputs, native)
        for index in inputs.plan["grid_comparisons"]:
            if (index, "initial") in self.finals:
                inputs.cases[index]["rows"]["initial"] = self.finals[index, "initial"]

    def initial(self, index, *, audit=None):
        if index < 104:
            return super().initial(index, audit=audit)
        if index not in self.inputs.plan["grid_comparisons"]:
            raise ValueError("training-only cells have no declared endpoint comparison")
        if (index, "initial") not in self.finals:
            prediction = self.inputs.baselines[self.inputs.key(self.inputs.plan["fit_audit_cells"][index])]
            patch = self.inputs.adapter.export(prediction["continuous"], prediction["categorical"])[0]
            row = {**self.render(index, patch, "qualification", f"grid-initial:{index}"), "arm": "initial"}
            append_durable(self.output / "finals.jsonl", row)
            self.finals[index, "initial"] = row
        row = self.finals[index, "initial"]
        self.inputs.cases[index]["rows"]["initial"] = row
        return row

    def collect_initial(self, indices):
        requests = []
        for index in indices:
            if (index, "initial") not in self.finals:
                prediction = self.inputs.baselines[self.inputs.key(self.inputs.plan["fit_audit_cells"][index])]
                patch = self.inputs.adapter.export(prediction["continuous"], prediction["categorical"])[0]
                requests.append((index, patch, "qualification", f"grid-initial:{index}"))
        def publish(row):
            row = {**row, "arm": "initial"}
            append_durable(self.output / "finals.jsonl", row)
            self.finals[row["index"], "initial"] = row
            self.inputs.cases[row["index"]]["rows"]["initial"] = row
        self.render_many(requests, on_result=publish)

    def control(self, index):
        key = index, "control"
        if key in self.finals:
            return self.finals[key]
        if index < 104:
            original = self.inputs.control_finals[index, "sound_only"]
            name = original["receipt"]["retained_audio"]["file"]
            destination = self.output / name
            destination.parent.mkdir(exist_ok=True)
            if not destination.exists():
                if self.native is None:
                    raise ValueError("retained audited control audio is missing")
                shutil.copyfile(self.inputs.paths["control"] / name, destination)
            _audio, source = self.source(index)
            row = {**original, "arm": "control", "origin": "audited_control", "source_audio": source}
        else:
            with torch.no_grad():
                prediction = self.inputs.predict(self.inputs.control_model, self.inputs.plan["fit_audit_cells"][index])
            patch = self.inputs.adapter.export(prediction["continuous"], prediction["categorical"])[0]
            row = {**self.render(index, patch, "qualification", f"grid-control:{index}"), "arm": "control"}
        append_durable(self.output / "finals.jsonl", row)
        self.finals[key] = row
        return row


def initialize(inputs):
    random.seed(SPEC["seed"])
    np.random.seed(SPEC["seed"])
    torch.manual_seed(SPEC["seed"])
    model = training_model(copy.deepcopy(inputs.model)).train()
    return model, torch.optim.Adam(model.correction.parameters(), lr=SPEC["learning_rate"])


def snapshot(inputs, model, optimizer, trace):
    return copy.deepcopy({"plan_id": canonical_sha256(inputs.plan), "step": len(trace), "trace": trace,
                          "model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": capture_rng("cpu")})


def train(inputs, store, *, output=None, saved=None, stop_after=None):
    model, optimizer = initialize(inputs)
    trace = []
    if saved is not None:
        if saved["plan_id"] != canonical_sha256(inputs.plan) or saved["step"] != len(saved["trace"]):
            raise ValueError("balanced checkpoint does not match the frozen run")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"], "cpu")
        trace = copy.deepcopy(saved["trace"])
    charges = lines(output / "updates.jsonl") if output else []
    for index in range(len(trace), len(inputs.plan["updates"]) if stop_after is None else stop_after):
        if output:
            store.check_limits()
            if len(charges) >= LIMITS["optimizer_attempts"]:
                raise ValueError("balanced optimizer attempt cap exhausted")
            charge = {"attempt": len(charges), "step": index + 1}
            append_durable(output / "updates.jsonl", charge)
            charges.append(charge)
        result, calibration = shared_step(inputs, store, {"balanced": model}, {"balanced": optimizer}, [], index, None)
        if calibration is not None or result["arms"]["balanced"]["coefficient"]:
            raise ValueError("balanced data fit unexpectedly enabled an auxiliary loss")
        trace.append(result)
        if output:
            state = snapshot(inputs, model, optimizer, trace)
            atomic_torch_save(output / "checkpoint.pt", state)
            if index + 1 == 2 and not (output / "prefix-verification.json").exists():
                _model, replay = train(inputs, BalancedStore(output, inputs), stop_after=2)
                if not equal_tree(state, replay):
                    raise ValueError("first balanced native updates do not replay exactly")
                restore_rng(state["rng"], "cpu")
                write_json(output / "prefix-verification.json", {"plan_id": state["plan_id"], "replayed_updates": 2,
                           "model_adam_rng_exact": True, "native_requests_added": 0})
            write_json(output / "progress.json", {"step": index + 1, "planned_updates": len(inputs.plan["updates"]),
                "sound_loss": result["arms"]["balanced"]["sound_loss"],
                "accepted_updates": sum(row["arms"]["balanced"]["acceptance"]["accepted"] for row in trace)})
    return model, snapshot(inputs, model, optimizer, trace)


def collect_endpoints(inputs, store, model):
    from .paired_native_evidence import WaveformAudit
    indices = inputs.plan["evaluation_indices"]
    with torch.no_grad():
        predictions = predict_cells(inputs, model, [inputs.plan["fit_audit_cells"][index] for index in indices])
        controls = predict_cells(inputs, inputs.control_model, [inputs.plan["fit_audit_cells"][index] for index in indices])
    requests, arms = [], []
    audit = WaveformAudit(BatchConfig.from_plan(inputs.plan).spectral)
    for index, prediction, control in zip(indices, predictions, controls, strict=True):
        store.initial(index, audit=audit)
        if index < 104:
            store.control(index)
        elif (index, "control") not in store.finals:
            patch = inputs.adapter.export(control["continuous"], control["categorical"])[0]
            requests.append((index, patch, "qualification", f"control:{index}"))
            arms.append("control")
        patch = inputs.adapter.export(prediction["continuous"], prediction["categorical"])[0]
        arms = ("balanced", "balanced_repeat") if index in inputs.plan["repeat_comparisons"] else ("balanced",)
        for arm in arms:
            if (index, arm) not in store.finals:
                requests.append((index, patch, "qualification", f"{arm}:{index}"))
                arms.append(arm)
    audit.finish()
    pending_arms = iter(arms)
    def publish(row):
        arm = next(pending_arms)
        row = {**row, "arm": arm}
        append_durable(store.output / "finals.jsonl", row)
        store.finals[row["index"], arm] = row
    store.render_many(requests, on_result=publish)


def execute(inputs, output, *, resume=False):
    from .balanced_basic_audit import results
    check = read(output / "software-check.json")
    if not check["passed"] or check["plan_id"] != canonical_sha256(inputs.plan):
        raise ValueError("real-initializer software check must pass before balanced native fitting")
    charge_execution(inputs, output, resume)
    monitor = MemoryMonitor()
    monitor.start()
    native = PairedSession(output, inputs.plan, inputs.codec, monitor)
    store, failure, result = BalancedStore(output, inputs, native), None, None
    try:
        store.collect_initial(inputs.plan["grid_comparisons"])
        saved = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True) if resume and (output / "checkpoint.pt").exists() else None
        model, _saved = train(inputs, store, output=output, saved=saved)
        collect_endpoints(inputs, store, model)
        result = results(inputs, store)
        write_json(output / "result.json", result)
        atomic_torch_save(output / "diagnostic-balanced.pt", export_inward(model, inputs.frontend, inputs.codec, inputs.plan["pool_id"]))
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        native.close()
        monitor.close()
        elapsed = time() - native.started
        archive = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
        requests = sum(row["event"] == "attempt" for row in native.events)
        saved = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True) if (output / "checkpoint.pt").exists() else None
        within = bool(monitor.peak_bytes and monitor.peak_bytes <= LIMITS["process_tree_bytes"] and not monitor.errors
                      and elapsed <= LIMITS["wall_seconds"] and archive <= LIMITS["archive_bytes"] and requests <= 8000)
        write_json(output / "summary.json", {"plan_id": canonical_sha256(inputs.plan), "complete": failure is None and result is not None,
            "failure": failure, "committed_updates": saved["step"] if saved else 0, "native_attempts": requests,
            "elapsed_seconds": elapsed, "peak_process_tree_rss_bytes": monitor.peak_bytes, "rss_sampling_errors": monitor.errors,
            "archive_bytes_before_summary": archive, "within_resource_limits": within, "qualified_for_release": False, "test_split_used": False})
    return read(output / "summary.json")

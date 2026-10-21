"""Frozen, matched basic inverse fitting with measured native sound-loss gradients."""

from collections import OrderedDict
import copy
import json
import os
from pathlib import Path
import platform
from time import time

import numpy as np
import torch

from obruxo_data.hashing import canonical_sha256, sha256_bytes
from obruxo_data.midi import Performance

from .audit import write_json
from .benchmark import MemoryMonitor
from .checkpoints import atomic_torch_save, load_training_checkpoint
from .codec import PilotCodec, PilotTarget
from .evaluate import audio_distance, nearest_training_target, restore_evaluation_checkpoint, retrieval_digest
from .features import tensor_digest
from .model import batch_features, loss_terms, predicted_targets
from .native_gradient import native_gradient_spec
from .native_gradient_screen import qualification_summary
from .native_inverse import native_reconstruction_loss
from .performances import initial_bank
from .spectral import SpectralTargets, spectral_loss, target_bytes
from .spectral_controls import BasicControlAdapter, temperature_at
from .temporal_proxy_data import NativeSession, category_balanced_entries, training_performance
from .trainer import capture_rng, inference_state, restore_model, restore_rng


def digest(path):
    return sha256_bytes(Path(path).read_bytes())


def require_qualification(report):
    plan = report["plan"]
    result = qualification_summary(report["cases"], plan)
    if (report["plan_id"] != canonical_sha256(plan) or plan["estimator"] != native_gradient_spec()
            or result != report["result"] or not result["direction_gates_passed"]
            or not report["qualified_native_gradient_path"] or not report["all_native_accepted"]
            or not report["within_resource_limits"] or not report["all_numerical_updates_reconstructed"]
            or not report["all_requests_reconstructed"] or report["test_split_used"]):
        raise ValueError("native estimator has not passed the required qualification")
    for name in ("native_gradient.py", "spectral.py", "spectral_controls.py", "reproducibility.py"):
        current = Path(__file__).with_name(name).read_bytes().replace(b"\r\n", b"\n")
        if sha256_bytes(current) != plan["source"][name]:
            raise ValueError("qualified estimator source changed")


def make_plan(pool, initial_path, retrieval_path, qualification_path):
    if type(pool.codec) is not PilotCodec:
        raise ValueError("native inverse fitting requires the exact basic codec")
    report = json.loads(qualification_path.read_text())
    require_qualification(report)
    initial = torch.load(initial_path, map_location="cpu", weights_only=True)
    model, frontend, codec = restore_model(initial)
    retrieval = load_training_checkpoint(retrieval_path)
    restore_evaluation_checkpoint(retrieval, pool)
    if (initial["training_pool_id"] != pool.pool_id or codec.to_dict() != pool.codec.to_dict()
            or tensor_digest(model.state_dict()) != tensor_digest(retrieval["model"])):
        raise ValueError("initial model and train-only retrieval checkpoint must match this pool")
    training = []
    reserved = {row.performance_id for row in initial_bank()}
    for index, entry in enumerate(category_balanced_entries(pool, "train", 32)):
        performances = [training_performance(0, index, (index + offset) % 4).to_dict() for offset in (0, 1)]
        if any(canonical_sha256(row) in reserved for row in performances):
            raise ValueError("training MIDI matches a reserved evaluation recipe")
        training.append({"state_id": entry.state_id, "target": entry.target.to_dict(), "performances": performances})
    validation = [{"state_id": entry.state_id, "target": entry.target.to_dict(),
                   "performances": [initial_bank()[offset + index % 4].performance().to_dict() for offset in (0, 4)]}
                  for index, entry in enumerate(category_balanced_entries(pool, "validation", 12))]
    rng = torch.Generator().manual_seed(0)
    updates = [{"state": int(index), "view": epoch % 2} for epoch in range(4) for index in torch.randperm(32, generator=rng)]
    names = ("native_fit.py", "native_gradient.py", "native_inverse.py", "spectral.py", "spectral_controls.py",
             "model.py", "features.py", "trainer.py", "evaluate.py", "temporal_proxy_data.py", "reproducibility.py", "checkpoints.py")
    paths = [Path(__file__).with_name(name) for name in names]
    paths += [Path(__file__).parents[2] / "data_generation/obruxo_data/render/vita.py",
              Path(__file__).parents[1] / "reports/scripts/run_native_inverse_fit.py"]
    return {"format": "matched_native_inverse_fit_v1", "pool_id": pool.pool_id, "manifest_id": pool.manifest.manifest_id,
            "codec": codec.to_dict(), "adapter": BasicControlAdapter(codec).contract(), "estimator": native_gradient_spec(),
            "initial_file_sha256": digest(initial_path), "initial_model_sha256": tensor_digest(model.state_dict()),
            "retrieval_file_sha256": digest(retrieval_path), "retrieval_digest": retrieval_digest(retrieval["replay"]),
            "qualification_file_sha256": digest(qualification_path), "qualification_plan_id": report["plan_id"],
            "frontend": frontend.describe(), "training": training, "validation": validation, "updates": updates,
            "optimizer": {"name": "Adam", "learning_rate": .0001, "gradient_norm_cap": 5., "seed": 0,
                          "updates_per_arm": 128, "batch_size": 1, "checkpoint_every": 1,
                          "spectral_parameter_anchor_weight": 0., "temperature_start": 1., "temperature_end": .5},
            "limits": {"training_attempts": 2896, "qualification_attempts": 432, "optimizer_attempts": 132,
                       "wall_seconds": 14400, "process_tree_bytes": 8 * 1024**3, "archive_bytes": 2 * 1024**3,
                       "replay_bytes": 512 * 1024**2, "replay_records": 512},
            "planned_maximum_requests": {"training_targets": 64, "gradient_probes": 2816, "fit_audit": 256,
                                         "validation_audit": 168, "total": 3304},
            "gates": {"fit_spectral_vs_initial_ratio": .9, "fit_spectral_vs_parameter_ratio": 1.,
                      "validation_mrstft_vs_parameter_ratio": .9, "validation_mrstft_vs_initial_ratio": .9,
                      "validation_vs_template_and_retrieval": "strictly_lower_mean_mrstft",
                      "validation_tail_p90_vs_initial_ratio": 1.1, "all_renders_accepted": True, "all_repeats_exact": True},
            "source": {path.name: sha256_bytes(path.read_bytes().replace(b"\r\n", b"\n")) for path in paths},
            "runtime": {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
                        "device": "cpu", "cpu_threads": 2, "precision": "float32", "native_workers": 1},
            "test_split_used": False, "checkpoint_selection": "final_only", "inference_input": "primary_audio_only"}


def append_durable(path, value):
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, allow_nan=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


class FitRecords:
    def __init__(self, output, plan, frontend, session):
        self.output, self.plan, self.frontend, self.session = output, plan, frontend, session
        self.directory = output / "records"
        self.directory.mkdir(exist_ok=True)
        self.path = output / "records.json"
        self.rows = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.cache, self.bytes = OrderedDict(), 0

    def check_limits(self):
        self.session.check_limits()
        archive = sum(path.stat().st_size for path in self.output.rglob("*") if path.is_file())
        if archive + 32 * 1024**2 > self.plan["limits"]["archive_bytes"]:
            raise RuntimeError("native inverse archive budget exhausted")

    def __getitem__(self, index):
        self.check_limits()
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index][0]
        recipe, path = self.plan["training"][index], self.directory / f"{index:03d}.pt"
        if str(index) in self.rows:
            if digest(path) != self.rows[str(index)]["sha256"]:
                raise ValueError("training record changed")
            record = torch.load(path, map_location="cpu", weights_only=True)
        else:
            targets, hashes, features = [], [], None
            for view, value in enumerate(recipe["performances"]):
                audio, receipt = self.session.render(PilotTarget.from_dict(recipe["target"]), Performance.from_dict(value),
                                                     "training", f"source:{index}:{view}")
                if audio is None:
                    raise RuntimeError("training source render rejected")
                targets.append(SpectralTargets()(torch.from_numpy(audio.copy()).T[None]))
                hashes.append(receipt["qa"]["audio_float32_sha256"])
                if view == 0:
                    features = self.frontend(audio)
            self.frontend.require_frozen()
            record = {"recipe_id": canonical_sha256(recipe), "features": features, "targets": targets, "audio_sha256": hashes}
            atomic_torch_save(path, record)
            self.rows[str(index)] = {"file": path.name, "sha256": digest(path), "recipe_id": record["recipe_id"]}
            write_json(self.path, self.rows)
        if record["recipe_id"] != canonical_sha256(recipe):
            raise ValueError("training recipe identity changed")
        size = sum(target_bytes(target) for target in record["targets"]) + sum(value.numel() * value.element_size() for value in record["features"].values())
        if size > self.plan["limits"]["replay_bytes"]:
            raise RuntimeError("one training record exceeds replay budget")
        while self.cache and (self.bytes + size > self.plan["limits"]["replay_bytes"] or len(self.cache) >= self.plan["limits"]["replay_records"]):
            _, (_, removed) = self.cache.popitem(last=False)
            self.bytes -= removed
        self.cache[index] = record, size
        self.bytes += size
        return record


def matched_step(models, optimizers, features, target, codec, evaluate, step, steps):
    """One shared observation, one update per arm; the spectral arm receives no labels."""
    losses, gradients = {}, {}
    adapter = BasicControlAdapter(codec)
    for name in ("parameter", "spectral"):
        model, optimizer = models[name], optimizers[name]
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch_features([{"features": features}]))
        if name == "parameter":
            loss = loss_terms(prediction, [target], codec)["loss"]
            estimates = []
        else:
            prediction["continuous"].retain_grad()
            for scores in prediction["categorical"]:
                scores.retain_grad()
            loss, estimates = native_reconstruction_loss(prediction, adapter, evaluate, temperature=temperature_at(step, steps))
        if not torch.isfinite(loss):
            raise ValueError("nonfinite inverse objective")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        if not torch.isfinite(norm):
            raise ValueError("nonfinite inverse gradients")
        if name == "spectral":
            gradients = {"continuous_l1": float(prediction["continuous"].grad.abs().sum()),
                         "category_l1": [float(scores.grad.abs().sum()) for scores in prediction["categorical"]]}
        optimizer.step()
        losses[name] = float(loss.detach())
    return {"losses": losses, "gradients": gradients, "gradient_queries": sum(row.queries for row in estimates),
            "estimates": [row.to_dict() for row in estimates]}


def fit_models(models, records, session, output, plan, saved=None, *, after_step=None):
    optimizers = {name: torch.optim.Adam(model.parameters(), lr=plan["optimizer"]["learning_rate"]) for name, model in models.items()}
    for model in models.values():
        model.train()
    trace = []
    if saved:
        if (saved["plan_id"] != canonical_sha256(plan) or saved["format"] != "matched_native_inverse_checkpoint_v1"
                or saved["step"] != len(saved["trace"])):
            raise ValueError("training checkpoint contract changed")
        for name, model in models.items():
            model.load_state_dict(saved["models"][name])
            optimizers[name].load_state_dict(saved["optimizers"][name])
        trace = saved["trace"]
        restore_rng(saved["rng"], "cpu")
    else:
        torch.manual_seed(plan["optimizer"]["seed"])
    charge_path = output / "updates.jsonl"
    charges = [json.loads(line) for line in charge_path.read_text().splitlines()] if charge_path.exists() else []
    if any(row["attempt"] != i for i, row in enumerate(charges)):
        raise ValueError("optimizer attempt ledger changed")
    for step, row in enumerate(trace):
        charge = {key: row[key] for key in ("attempt", "step", "state", "view")}
        if row["step"] != step + 1 or row["attempt"] >= len(charges) or charges[row["attempt"]] != charge:
            raise ValueError("checkpoint trace differs from charged optimizer attempts")
    for step in range(len(trace), len(plan["updates"])):
        records.check_limits()
        if len(charges) >= plan["limits"]["optimizer_attempts"]:
            raise RuntimeError("inverse optimizer-attempt budget exhausted")
        update = plan["updates"][step]
        index, view = update["state"], update["view"]
        record, recipe = records[index], plan["training"][index]
        charge = {"attempt": len(charges), "step": step + 1, "state": index, "view": view}
        append_durable(charge_path, charge)
        charges.append(charge)
        performance = Performance.from_dict(recipe["performances"][view])
        query = 0

        def evaluate(_index, candidate):
            nonlocal query
            audio, receipt = session.render(candidate, performance, "training", f"gradient:{step}:{query}")
            query += 1
            if audio is None:
                raise RuntimeError("native gradient training render rejected")
            value = float(spectral_loss(SpectralTargets()(torch.from_numpy(audio.copy()).T[None]), record["targets"][view])["loss"].item())
            append_durable(output / "measurements.jsonl", {"optimizer_attempt": charge["attempt"], "step": step + 1,
                           "query": query - 1, "candidate": candidate.to_dict(), "loss": value,
                           "request_id": receipt.get("request_id"), "audio_sha256": receipt["qa"]["audio_float32_sha256"]})
            return value

        result = matched_step(models, optimizers, record["features"], PilotTarget.from_dict(recipe["target"]), session.codec,
                              evaluate, step, len(plan["updates"]))
        trace.append({**charge, **result})
        state = {"format": "matched_native_inverse_checkpoint_v1", "plan_id": canonical_sha256(plan), "step": step + 1,
                 "models": {name: model.state_dict() for name, model in models.items()},
                 "optimizers": {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
                 "rng": capture_rng("cpu"), "trace": trace}
        atomic_torch_save(output / "checkpoint.pt", state)
        if after_step:
            after_step(state)
        if (step + 1) % 8 == 0:
            print(json.dumps({"stage": "train", "step": step + 1, "planned": len(plan["updates"]),
                              "losses": result["losses"], "gradient_queries": result["gradient_queries"]}), flush=True)
    return trace


def audit_case(session, output, label, recipe, view, candidate_provider, identity, *, expected_audio=None, repeat=False):
    path = output / "audits" / f"{label}.pt"
    path.parent.mkdir(exist_ok=True)
    contract = {"plan_id": canonical_sha256(session.plan), "recipe": recipe, "view": view, "identity": identity}
    saved = torch.load(path, map_location="cpu", weights_only=True) if path.exists() else {"contract": contract, "rows": {}}
    if saved["contract"] != contract:
        raise ValueError("native audit contract changed")
    performance = Performance.from_dict(recipe["performances"][view])

    def render(candidate, method):
        audio, receipt = session.render(candidate, performance, "qualification", f"{label}:{method}")
        if audio is None:
            raise RuntimeError("inverse audit render rejected")
        return audio, receipt

    if "audio" not in saved:
        audio, receipt = render(PilotTarget.from_dict(recipe["target"]), "source")
        if expected_audio is not None and receipt["qa"]["audio_float32_sha256"] != expected_audio:
            raise ValueError("training-source repeat changed")
        saved.update(audio=torch.from_numpy(audio.copy()), source_receipt=receipt, repeat_exact=expected_audio is not None)
        atomic_torch_save(path, saved)
    reference = saved["audio"].numpy()
    if sha256_bytes(reference.tobytes()) != saved["source_receipt"]["qa"]["audio_float32_sha256"]:
        raise ValueError("saved audit audio changed")
    candidates = candidate_provider(reference)
    encoded = {name: candidate.to_dict() for name, candidate in candidates.items()}
    if "candidates" in saved and saved["candidates"] != encoded:
        raise ValueError("audit predictions changed")
    saved["candidates"] = encoded
    if repeat and "repeat_receipt" not in saved:
        audio, receipt = render(PilotTarget.from_dict(recipe["target"]), "repeat")
        if not np.array_equal(reference, audio):
            raise ValueError("validation-source repeat changed")
        saved.update(repeat_receipt=receipt, repeat_exact=True)
        atomic_torch_save(path, saved)
    target = SpectralTargets()(torch.from_numpy(reference.copy()).T[None])
    tail = max(1, int(session.codec.contract.tail_seconds * 44100))
    reference_rms = np.sqrt(np.mean(reference[-tail:].astype(np.float64) ** 2, axis=0))
    for name, candidate in candidates.items():
        if name in saved["rows"]:
            continue
        audio, receipt = render(candidate, name)
        candidate_rms = np.sqrt(np.mean(audio[-tail:].astype(np.float64) ** 2, axis=0))
        scores = {"spectral": float(spectral_loss(SpectralTargets()(torch.from_numpy(audio.copy()).T[None]), target)["loss"].item()),
                  "mrstft": audio_distance(reference, audio)["mrstft"],
                  "tail_error": float(np.mean(np.abs(candidate_rms - reference_rms) / np.maximum(reference_rms, .001)))}
        saved["rows"][name] = {"scores": scores, "candidate": candidate.to_dict(), "receipt": receipt}
        atomic_torch_save(path, saved)
    return {"label": label, "state_id": recipe["state_id"], "view": view, "repeat_exact": saved["repeat_exact"],
            "source_receipt": saved["source_receipt"], "rows": saved["rows"]}, candidates


def audit_summary(cases, plan, *, validation):
    count = len(plan["validation" if validation else "training"])
    expected = {(entry["state_id"], view) for entry in plan["validation" if validation else "training"] for view in (0, 1)}
    actual = [(row["state_id"], row["view"]) for row in cases]
    complete = len(actual) == len(expected) and set(actual) == expected and all(row["repeat_exact"] for row in cases)
    groups = {}
    for view, role in ((0, "reference"), (1, "transfer" if validation else "additional_training_MIDI")):
        selected = [row for row in cases if row["view"] == view]
        names = ("initial", "parameter", "spectral", "template", "retrieval") if validation else ("initial", "parameter", "spectral")
        means = {name: {metric: float(np.mean([row["rows"][name]["scores"][metric] for row in selected])) if selected else None
                        for metric in ("spectral", "mrstft", "tail_error")} for name in names}
        passed = False
        tail = None
        if complete and len(selected) == count:
            if validation:
                tail = {name: float(np.quantile([row["rows"][name]["scores"]["tail_error"] for row in selected], .9))
                        for name in ("initial", "spectral")}
                passed = (means["spectral"]["mrstft"] <= .9 * means["parameter"]["mrstft"]
                          and means["spectral"]["mrstft"] <= .9 * means["initial"]["mrstft"]
                          and means["spectral"]["mrstft"] < min(means[name]["mrstft"] for name in ("template", "retrieval"))
                          and means["spectral"]["spectral"] <= means["parameter"]["spectral"]
                          and tail["spectral"] <= 1.1 * tail["initial"])
            else:
                passed = (means["spectral"]["spectral"] <= .9 * means["initial"]["spectral"]
                          and means["spectral"]["spectral"] <= means["parameter"]["spectral"]
                          and means["spectral"]["mrstft"] < means["initial"]["mrstft"])
        groups[role] = {"planned": count, "completed": len(selected), "means": means, "tail_p90": tail, "pass": bool(passed)}
    return {"complete": complete, "groups": groups, "pass": complete and all(row["pass"] for row in groups.values())}


@torch.no_grad()
def predict_candidates(models, codec, features):
    return {name: predicted_targets(model(batch_features([{"features": features}])), codec)[0] for name, model in models.items()}


def run_fit(pool, initial_path, retrieval_path, qualification_path, output, *, execute=False, resume=False):
    torch.set_num_threads(2)
    plan = make_plan(pool, initial_path, retrieval_path, qualification_path)
    plan_id = canonical_sha256(plan)
    if not execute:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "plan.json", plan)
        return {"plan_id": plan_id, "updates_per_arm": 128, "maximum_native_requests": 3328}
    if json.loads((output / "plan.json").read_text()) != plan:
        raise ValueError("frozen inverse plan changed")
    if (output / "run.json").exists() != resume:
        raise ValueError("use --resume only for this already attempted run")
    if not resume:
        write_json(output / "run.json", {"plan_id": plan_id, "started_unix": time()})
    initial = torch.load(initial_path, map_location="cpu", weights_only=True)
    incumbent, frontend, codec = restore_model(initial)
    models = {name: copy.deepcopy(incumbent).train() for name in ("parameter", "spectral")}
    retrieval = load_training_checkpoint(retrieval_path)["replay"]
    monitor = MemoryMonitor()
    monitor.start()
    session = NativeSession(output, plan, codec, monitor)
    records = FitRecords(output, plan, frontend, session)
    failure, fit_result, validation_result, trace = None, None, None, []
    fit_cases, validation_cases = [], []
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
        trace = fit_models(models, records, session, output, plan, saved, after_step=committed)
        all_models = {"initial": incumbent.eval(), **{name: model.eval() for name, model in models.items()}}
        model_ids = {name: tensor_digest(model.state_dict()) for name, model in all_models.items()}
        for index, recipe in enumerate(plan["training"]):
            record = records[index]
            candidates = predict_candidates(all_models, codec, record["features"])
            for view in (0, 1):
                case, _ = audit_case(session, output, f"fit-{index:03d}-{view}", recipe, view, lambda _audio: candidates,
                                     model_ids, expected_audio=record["audio_sha256"][view])
                fit_cases.append(case)
                write_json(output / "fit_cases.json", fit_cases)
                records.check_limits()
            print(json.dumps({"stage": "fit_audit", "completed": index + 1, "planned": 32}), flush=True)
        fit_result = audit_summary(fit_cases, plan, validation=False)
        write_json(output / "fit_result.json", fit_result)
        if fit_result["pass"]:
            for index, recipe in enumerate(plan["validation"]):
                def provider(audio):
                    features = frontend(audio)
                    return {**predict_candidates(all_models, codec, features), "template": codec.default_target(),
                            "retrieval": nearest_training_target(features, retrieval)}
                reference, candidates = audit_case(session, output, f"validation-{index:03d}-0", recipe, 0, provider, model_ids, repeat=True)
                transfer, _ = audit_case(session, output, f"validation-{index:03d}-1", recipe, 1, lambda _audio: candidates, model_ids, repeat=True)
                validation_cases.extend((reference, transfer))
                write_json(output / "validation_cases.json", validation_cases)
                records.check_limits()
                print(json.dumps({"stage": "validation", "completed": index + 1, "planned": 12}), flush=True)
            validation_result = audit_summary(validation_cases, plan, validation=True)
        for name, model in models.items():
            export = inference_state(model, frontend, codec, pool.pool_id)
            export.update(training_objective=name, training_plan_id=plan_id, qualified_for_release=False)
            atomic_torch_save(output / f"diagnostic-{name}.pt", export)
        frontend.require_frozen()
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        monitor.close()
        elapsed = time() - session.started
        archive = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
        attempts = [row for row in session.events if row["event"] == "attempt"]
        outcomes = [row for row in session.events if row["event"] == "outcome"]
        accepted = len(attempts) == len(outcomes) and all(row["status"] == "accepted" for row in outcomes)
        resources = (monitor.errors == 0 and monitor.peak_bytes is not None and monitor.peak_bytes <= plan["limits"]["process_tree_bytes"]
                     and elapsed <= plan["limits"]["wall_seconds"] and archive + 1024**2 <= plan["limits"]["archive_bytes"])
        unchanged = digest(initial_path) == plan["initial_file_sha256"] and digest(retrieval_path) == plan["retrieval_file_sha256"]
        complete = failure is None and len(trace) == 128 and fit_result is not None
        passed = bool(complete and fit_result["pass"] and validation_result and validation_result["pass"] and accepted and resources and unchanged)
        summary = {"plan_id": plan_id, "plan": plan, "complete": complete, "failure": failure,
                   "fit_result": fit_result, "validation_result": validation_result, "matched_comparison_passed": passed,
                   "committed_updates_per_arm": len(trace), "elapsed_seconds": elapsed, "peak_process_tree_rss_bytes": monitor.peak_bytes,
                   "rss_sampling_errors": monitor.errors, "archive_bytes_before_summary": archive,
                   "render_attempts": {phase: sum(row["phase"] == phase for row in attempts) for phase in ("training", "qualification")},
                   "all_native_accepted": accepted, "within_resource_limits": resources, "incumbent_artifacts_unchanged": unchanged,
                   "renderer_identity": session.renderer_identity, "test_split_used": False, "qualified_for_release": False}
        write_json(output / "summary.json", summary)
    return {"complete": complete, "matched_comparison_passed": passed, "committed_updates_per_arm": len(trace)}

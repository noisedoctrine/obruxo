"""Bounded real-Vital direction check at the accepted predictor's production outputs."""

from collections import Counter
import json
import os
from pathlib import Path
from time import perf_counter, time

import numpy as np
import torch

from obruxo_data.hashing import canonical_sha256, sha256_bytes
from obruxo_data.midi import Performance, TempoMap

from . import category_scale_refresh as refresh
from . import production_choice_check as choice_check
from .audit import write_json
from .balanced_sound_data import cached_training_records
from .benchmark import MemoryMonitor
from .checkpoints import atomic_torch_save
from .codec import PilotTarget
from .evaluate import audio_distance
from .features import tensor_digest
from .mixed_input_performances import MixedSampler
from .model import batch_features
from .production_controls import ProductionControlAdapter
from .production_direction import GATES, UPDATE, direction_summary, guided_prediction, raw_prediction
from .spectral import FFT_SIZES, MAX_SAMPLES, SAMPLE_RATE, SpectralTargets, spectral_loss
from .temporal_proxy_data import NativeSession, category_balanced_entries
from .trainer import load_model

LIMITS = {"qualification_attempts": 624, "wall_seconds": 2700, "process_tree_bytes": 8 * 1024**3,
          "archive_bytes": 512 * 1024**2, "planning_wall_seconds": 600, "optimizer_updates": 0}


def source_identity():
    names = ("production_direction.py", "production_native_screen.py", "production_native_audit.py", "balanced_sound_data.py")
    paths = {name: Path(__file__).with_name(name) for name in names}
    paths["reports/scripts/run_production_native_screen.py"] = Path(__file__).parents[1] / "reports/scripts/run_production_native_screen.py"
    return {**choice_check.source_identity(), **{name: sha256_bytes(path.read_bytes().replace(b"\r\n", b"\n")) for name, path in paths.items()}}


def selected_cases(pool, cache, model, adapter, proof):
    entries = category_balanced_entries(pool, "train", 12)
    if Counter(entry.target.categorical for entry in entries) != Counter({(voices, filtered): 2 for voices in range(3) for filtered in range(2)}):
        raise ValueError("native choice check needs two training states in each basic category")
    selected = {entry.state_id for entry in entries}
    selected_cache = {"replay": [item for item in cache["replay"] if item["state_id"] in selected]}
    records = cached_training_records(pool, entries, selected_cache, MixedSampler(pool))
    bank = {item["recipe_id"]: item for item in cache["replay"]}
    proven = {item["recipe_id"]: item for item in proof["initial_training_inputs"]}
    cases, features, counts = [], {}, Counter()
    for index, (entry, record) in enumerate(zip(entries, records, strict=True)):
        wider_view = 1 + counts[entry.target.categorical] % 2
        counts[entry.target.categorical] += 1
        for role, source, target in (("original_input", 0, wider_view), ("wider_input", wider_view, 0)):
            recipe = record["source_recipe_ids"][source]
            item = bank[recipe]
            if (recipe not in proven or tensor_digest(item["features"]) != proven[recipe]["features_sha256"]
                    or item["audio_sha256"] != proven[recipe]["audio_sha256"]):
                raise ValueError("native direction input lacks the accepted production-choice proof")
            with torch.no_grad():
                prediction = model(batch_features([{"features": item["features"]}]))
            performance = Performance.from_dict(record["performances"][target])
            samples = TempoMap.from_performance(performance).render_frame_count(performance.end_tick, pool.codec.contract.tail_seconds, SAMPLE_RATE)
            if not max(FFT_SIZES) // 2 < samples <= MAX_SAMPLES:
                raise ValueError("native direction performance is outside the qualified spectral length")
            cases.append({"index": index, "role": role, "state_id": entry.state_id, "source": entry.target.to_dict(),
                          "input_view": source, "target_view": target, "input_recipe_id": recipe,
                          "input_features_sha256": tensor_digest(item["features"]), "input_audio_sha256": item["audio_sha256"],
                          "target_recipe_id": record["source_recipe_ids"][target], "target_audio_sha256": record["source_audio_sha256"][target],
                          "performance": performance.to_dict(), "source_samples": samples, "raw_prediction": raw_prediction(prediction),
                          "production_patch": adapter.export(prediction["continuous"], prediction["categorical"])[0].to_dict()})
            features[index, role] = item["features"]
    return cases, features


def make_plan(pool, bootstrap_root, bootstrap_audit_path, qualification_path, input_cache_path, choice_path):
    checked = choice_check.validate_check(choice_path, pool, bootstrap_root, bootstrap_audit_path, qualification_path, input_cache_path)
    proof = json.loads(choice_path.read_text())
    initial_path = bootstrap_root / proof["plan"]["initial_artifact"]
    if (refresh.digest(choice_path) != checked["report_sha256"] or refresh.digest(initial_path) != proof["plan"]["initial_sha256"]
            or refresh.digest(input_cache_path) != proof["plan"]["input_cache_sha256"]
            or refresh.digest(qualification_path) != proof["plan"]["qualification_sha256"]):
        raise ValueError("accepted native-direction inputs changed after verification")
    model, _frontend, codec = load_model(initial_path)
    model.eval()
    if tensor_digest(model.state_dict()) != proof["plan"]["inference"]["model_sha256"]:
        raise ValueError("native direction must start from the accepted initializer")
    cache = torch.load(input_cache_path, map_location="cpu", weights_only=True)
    adapter = ProductionControlAdapter(codec, scales=checked["scales"])
    cases, features = selected_cases(pool, cache, model, adapter, proof)
    bound = sum(case["source_samples"] * 2 * 4 * 2 * 4 + 256 * 1024 for case in cases) + 32 * 1024**2
    if bound > LIMITS["archive_bytes"]:
        raise ValueError("native production-choice source archive exceeds its planned cap")
    plan = {"format": "basic_production_native_direction_v1", "pool_id": pool.pool_id, "manifest_id": pool.manifest.manifest_id,
            "codec": codec.to_dict(), "initial_sha256": proof["plan"]["initial_sha256"], "inference": proof["plan"]["inference"],
            "input_cache_sha256": proof["plan"]["input_cache_sha256"], "choice_proof": checked,
            "physical_qualification_sha256": refresh.digest(qualification_path), "adapter": adapter.contract(),
            "render_profile": proof["plan"]["render_profile"], "render_phase_controls": proof["plan"]["render_phase_controls"],
            "expected_renderer_identity": json.loads(qualification_path.read_text())["renderer_identity"],
            "cases": cases, "source": source_identity(), "runtime": refresh.runtime_identity(), "update": UPDATE.copy(), "gates": GATES.copy(),
            "limits": LIMITS.copy(), "archive_estimate_bytes": bound, "test_split_used": False, "development_split_used": False,
            "selection": "two_train_states_per_category_original_and_category_balanced_held_or_alternate_reciprocal_inputs"}
    return plan, model, features


def qualify_case(case, session, model, features, adapter):
    measure, rows = SpectralTargets(), []
    performance = Performance.from_dict(case["performance"])

    def render(candidate, label):
        audio, receipt = session.render(candidate, performance, "qualification", f"{case['index']}:{case['role']}:{label}")
        if audio is None or receipt["renderer_identity"] != session.plan["expected_renderer_identity"]:
            raise RuntimeError("native production-choice render rejected or renderer changed")
        rows.append({"label": label, "candidate": candidate.to_dict(), "request_id": receipt["request_id"],
                     "audio_sha256": receipt["qa"]["audio_float32_sha256"]})
        return audio

    source = PilotTarget.from_dict(case["source"])
    reference, repeated = render(source, "source"), render(source, "repeat")
    if (not np.array_equal(reference, repeated) or len(reference) != case["source_samples"]
            or rows[0]["audio_sha256"] != case["target_audio_sha256"]):
        raise RuntimeError("fixed-phase source differs from its repeat or audited recording")
    tensor_path = session.output / "sources" / f"{case['index']:02d}-{case['role']}.pt"
    tensor_path.parent.mkdir(exist_ok=True)
    atomic_torch_save(tensor_path, {"audio": torch.from_numpy(reference.copy())})
    target, scores = measure(torch.from_numpy(reference.copy()).T[None]), {}

    def score(candidate, label):
        audio = render(candidate, label)
        result = {"spectral": float(spectral_loss(measure(torch.from_numpy(audio.copy()).T[None]), target)["loss"]),
                  "mrstft": audio_distance(reference, audio)["mrstft"]}
        rows[-1]["scores"] = result
        if label in ("start", "continuous", "joint"):
            path = session.output / "audits" / f"{case['index']:02d}-{case['role']}-{label}.pt"
            path.parent.mkdir(exist_ok=True)
            atomic_torch_save(path, {"audio": torch.from_numpy(audio.copy())})
            rows[-1].update(audio_tensor=path.relative_to(session.output).as_posix(), audio_tensor_sha256=refresh.digest(path))
        return result

    def evaluate(index, candidate):
        if index != 0:
            raise ValueError("native direction requires one independently measured prediction")
        label = "start" if len(rows) == 2 else f"probe-{len(rows) - 3:02d}"
        result = score(candidate, label)
        if label == "start":
            scores["start"] = result
        return result["spectral"]

    direction = guided_prediction(model, features, adapter, evaluate)
    if direction["raw_prediction"] != case["raw_prediction"] or direction["production_patch"] != case["production_patch"]:
        raise ValueError("actual native prediction differs from the frozen model outputs")
    for method, candidate in direction["candidates"].items():
        scores[method] = score(PilotTarget.from_dict(candidate), method)
    return {"index": case["index"], "role": case["role"], "state_id": case["state_id"], "scores": scores,
            "repeat_exact": True, "direction": direction, "renders": rows,
            "source_tensor": tensor_path.relative_to(session.output).as_posix(), "source_tensor_sha256": refresh.digest(tensor_path),
            "category_changed": direction["candidates"]["joint"]["categorical"] != direction["production_patch"]["categorical"]}


def run_screen(pool, bootstrap_root, bootstrap_audit_path, qualification_path, input_cache_path, choice_path, output, *, execute=False):
    if (output / "run.json").exists() or (not execute and (output / "preparation.json").exists()):
        raise ValueError("native production-choice check already attempted; preserve evidence")
    started, monitor = perf_counter(), MemoryMonitor()
    monitor.start()
    try:
        with refresh.cpu_settings():
            plan, model, features = make_plan(pool, bootstrap_root, bootstrap_audit_path, qualification_path, input_cache_path, choice_path)
            planning = {"elapsed_seconds": perf_counter() - started, "peak_process_tree_rss_bytes": monitor.peak_bytes,
                        "rss_sampling_errors": monitor.errors, "native_requests": 0, "optimizer_updates": 0}
            if (planning["elapsed_seconds"] >= LIMITS["planning_wall_seconds"] or monitor.peak_bytes is None
                    or monitor.peak_bytes > LIMITS["process_tree_bytes"] or monitor.errors):
                raise RuntimeError("native production-choice planning exceeded its resource limits")
            if output.exists():
                if json.loads((output / "plan.json").read_text()) != plan:
                    raise ValueError("native production-choice plan changed")
            else:
                output.mkdir(parents=True)
                write_json(output / "plan.json", plan)
            if not execute:
                write_json(output / "preparation.json", planning)
                return {"plan_id": canonical_sha256(plan), "executed": False, "cases": len(plan["cases"]), "planning": planning}
            with (output / "run.json").open("x", encoding="utf-8", newline="\n") as handle:
                json.dump({"started_unix": time(), "plan_id": canonical_sha256(plan), "planning": planning}, handle, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            session = NativeSession(output, plan, pool.codec, monitor)
            return execute_cases(output, plan, model, features, session, monitor)
    finally:
        monitor.close()


def execute_cases(output, plan, model, features, session, monitor):
    adapter = ProductionControlAdapter(session.codec, scales=plan["adapter"]["fixed_scales"])
    cases, failure = [], None
    try:
        for case in plan["cases"]:
            cases.append(qualify_case(case, session, model, features[case["index"], case["role"]], adapter))
            write_json(output / "cases.json", cases)
            session.check_limits()
            if sum(path.stat().st_size for path in output.rglob("*") if path.is_file()) + 1024**2 > LIMITS["archive_bytes"]:
                raise RuntimeError("native production-choice archive cap exhausted")
            print(json.dumps({"completed": len(cases), "planned": len(plan["cases"]), "role": case["role"]}), flush=True)
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        elapsed = time() - session.started
        archive = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
        attempts = sum(row["event"] == "attempt" for row in session.events)
        outcomes = [row for row in session.events if row["event"] == "outcome"]
        accepted = attempts == len(outcomes) and all(row["status"] == "accepted" for row in outcomes)
        resources = (monitor.errors == 0 and monitor.peak_bytes is not None and 0 < monitor.peak_bytes <= LIMITS["process_tree_bytes"]
                     and elapsed <= LIMITS["wall_seconds"] and archive + 1024**2 <= LIMITS["archive_bytes"]
                     and attempts <= LIMITS["qualification_attempts"])
        result = direction_summary(cases, plan)
        unchanged = tensor_digest(model.state_dict()) == plan["inference"]["model_sha256"]
        qualified = result["direction_gates_passed"] and accepted and resources and unchanged and failure is None
        summary = {"plan": plan, "plan_id": canonical_sha256(plan), "cases": cases, "result": result, "failure": failure,
                   "qualified_production_direction": bool(qualified), "elapsed_seconds": elapsed, "peak_process_tree_rss_bytes": monitor.peak_bytes,
                   "rss_sampling_errors": monitor.errors, "native_attempts": attempts, "all_native_accepted": accepted,
                   "within_resource_limits": resources, "archive_bytes_before_summary": archive, "renderer_identity": session.renderer_identity,
                   "weights_unchanged": unchanged, "optimizer_updates": 0, "test_split_used": False, "qualified_for_release": False}
        write_json(output / "summary.json", summary)
    return {key: summary[key] for key in ("plan_id", "qualified_production_direction", "native_attempts", "elapsed_seconds")}

"""Run a bounded label-only learnability diagnostic on the balanced-bank setup."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter, time

import numpy as np
import torch

PROJECT_ROOT = Path(os.environ.get("OBRUXO_CODE_ROOT", Path(__file__).resolve().parents[4])).resolve()
RESEARCH = PROJECT_ROOT / "research"
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]

from obruxo_data.hashing import canonical_sha256, sha256_bytes  # noqa: E402
from obruxo_data.midi import Performance  # noqa: E402
from obruxo_data.render.qa import audio_float32_sha256  # noqa: E402
from obruxo_training.audit import write_json  # noqa: E402
from obruxo_training.balanced_basic_fit import initialize as initialize_balanced, load_inputs  # noqa: E402
from obruxo_training.basic_direct import REHEARSAL  # noqa: E402
from obruxo_training.benchmark import MemoryMonitor  # noqa: E402
from obruxo_training.category_scale_refresh import cpu_settings  # noqa: E402
from obruxo_training.conditioned_bridge import read  # noqa: E402
from obruxo_training.evaluate import audio_distance  # noqa: E402
from obruxo_training.features import tensor_digest  # noqa: E402
from obruxo_training.inward_basic import export_inward  # noqa: E402
from obruxo_training.model import batch_features, loss_terms  # noqa: E402
from obruxo_training.native_fit import digest  # noqa: E402
from obruxo_training.paired_native_fit import PairedSession  # noqa: E402
from obruxo_training.pilot import PilotPool  # noqa: E402
from obruxo_training.spectral import SpectralTargets, spectral_loss  # noqa: E402
from obruxo_training.trainer import capture_rng  # noqa: E402
from obruxo_training.trainable_basic import restore_trainable  # noqa: E402
from obruxo_training.unison_fit_data import require_request  # noqa: E402
from obruxo_training.unison_sound_fit import lines  # noqa: E402
from obruxo_training.checkpoints import atomic_torch_save  # noqa: E402
from obruxo_training.codec import PilotTarget  # noqa: E402

STEPS = 48
CPU_THREADS = 2
RENDER_WORKERS = 1
MAX_WALL_SECONDS = 1200
MAX_PROCESS_BYTES = 4 * 1024**3
MAX_NATIVE_RENDERS = 18
FIT_REFERENCE = {"continuous_mae": 0.06, "categorical_accuracy": 1.0, "final_loss_ratio": 0.05}


def prepare(fit_output: Path, pool_path: Path, output: Path, reuse_training: Path | None = None) -> dict:
    fit_output, pool_path, output = (path.resolve() for path in (fit_output, pool_path, output))
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    fit_plan = read(fit_output / "plan.json")
    verification, summary = read(fit_output / "verification.json"), read(fit_output / "summary.json")
    result = read(fit_output / "result.json")
    if (not verification.get("verified") or not verification.get("complete") or not summary.get("complete")
            or verification.get("plan_id") != canonical_sha256(fit_plan) or result.get("model_admitted")
            or fit_plan.get("test_split_used")):
        raise ValueError("the balanced sound fit must be complete, audited and development-only")
    if len(fit_plan["updates"]) != STEPS or len(fit_plan["grid_comparisons"]) != 36:
        raise ValueError("the diagnostic requires the frozen 48-update / 18-clip bank")

    data_path, bridge_path = Path(fit_plan["paths"]["data"]), Path(fit_plan["paths"]["bridge"])
    data_plan = read(data_path / "plan.json")
    files = [fit_output / name for name in ("plan.json", "verification.json", "summary.json", "result.json", "finals.jsonl")]
    files += [pool_path, bridge_path / "model.pt"]
    files += [data_path / name for name in ("plan.json", "verification.json", "features.pt")]
    files.extend(data_path / "records" / f"{row['index']:02d}.{suffix}"
                 for row in data_plan["observations"] for suffix in ("json", "pt"))
    code_commit = subprocess.check_output(["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True).strip()
    training_spec = {"balanced_examples": 18, "reference_examples_per_update": 2, "updates": STEPS,
                     "schedule": "the frozen balanced-fit update and reference schedule",
                     "loss": "masked source-parameter loss; no audio-loss or auxiliary term", "optimizer": "Adam",
                     "model_batch_size": 4, "learning_rate": .001, "gradient_norm_cap": 5., "cpu_threads": CPU_THREADS,
                     "device": "cpu", "precision": "float32", "trainable_parameters": 257982,
                     "frozen_parameters": 355806, "dense_outputs": 14, "paired_reference_weights": REHEARSAL}
    training_reuse = None
    if reuse_training is not None:
        previous = reuse_training.resolve()
        old_plan, old_result = read(previous / "plan.json"), read(previous / "result.json")
        outcomes = [row for row in lines(previous / "renders.jsonl") if row.get("event") == "outcome"]
        if (old_plan.get("code_commit") != code_commit or old_plan.get("fit_plan_id") != canonical_sha256(fit_plan)
                or old_plan.get("training") != training_spec or old_result.get("training", {}).get("updates") != STEPS
                or old_result.get("test_split_used") or old_result.get("model_admitted")
                or len(outcomes) != MAX_NATIVE_RENDERS
                or any(row.get("status") != "error" or row.get("message") != "Vita is not installed" for row in outcomes)):
            raise ValueError("reusable training must be the complete matching fit whose only render failure was missing Vita")
        model_path, curve_path = previous / "diagnostic-supervised.pt", previous / "training-curve.json"
        artifact = torch.load(model_path, map_location="cpu", weights_only=True)
        model, _frontend, _codec = restore_trainable(artifact)
        if tensor_digest(model.state_dict()) != artifact["model_sha256"]:
            raise ValueError("saved supervised candidate does not match its exported model identity")
        training_reuse = {"path": str(previous), "result_sha256": digest(previous / "result.json"),
                          "model_sha256": digest(model_path), "curve_sha256": digest(curve_path),
                          "previous_plan_id": old_plan["plan_id"], "model_tensor_sha256": artifact["model_sha256"]}
        files.extend((previous / "result.json", model_path, curve_path, previous / "renders.jsonl"))

    source_names = ("balanced_basic_fit.py", "basic_direct_fit_data.py", "basic_direct_fit_step.py", "basic_direct.py",
                    "trainable_basic.py", "inward_basic.py", "model.py", "production_controls.py", "paired_native_fit.py",
                    "temporal_proxy_data.py", "unison_fit_data.py", "spectral.py", "evaluate.py")
    source = {"script": sha256_bytes(Path(__file__).read_bytes().replace(b"\r\n", b"\n"))}
    for name in source_names:
        path = RESEARCH / "training/obruxo_training" / name
        source[name] = sha256_bytes(path.read_bytes().replace(b"\r\n", b"\n"))

    plan = {"format": "balanced_supervised_learnability_v1", "code_commit": code_commit,
            "code_root": str(PROJECT_ROOT), "fit_output": str(fit_output),
            "fit_plan_id": canonical_sha256(fit_plan), "fit_result_sha256": digest(fit_output / "result.json"),
            "pool_path": str(pool_path), "pool_id": fit_plan["pool_id"],
            "input_files": {str(path): digest(path) for path in files}, "source": source,
            "training": training_spec, "training_reuse": training_reuse,
            "render": {"native_requests": MAX_NATIVE_RENDERS, "workers": RENDER_WORKERS,
                       "renderer_identity": fit_plan["expected_renderer_identity"],
                       "render_profile": fit_plan["render_profile"], "phase_controls": fit_plan["render_phase_controls"]},
            "fit_reference": FIT_REFERENCE,
            "limits": {"wall_seconds": MAX_WALL_SECONDS, "process_tree_bytes": MAX_PROCESS_BYTES,
                       "native_renders": MAX_NATIVE_RENDERS},
            "test_split_used": False, "model_admission_allowed": False}
    plan["plan_id"] = canonical_sha256(plan)
    output.mkdir(parents=True)
    write_json(output / "plan.json", plan)
    return plan


def checked_plan(output: Path) -> dict:
    plan = read(output / "plan.json")
    plan_id = plan.pop("plan_id")
    if canonical_sha256(plan) != plan_id:
        raise ValueError("supervised-control plan changed")
    if PROJECT_ROOT != Path(plan["code_root"]) or subprocess.check_output(["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True).strip() != plan["code_commit"]:
        raise ValueError("the historical source checkout changed")
    plan["plan_id"] = plan_id
    for path, expected in plan["input_files"].items():
        if digest(path) != expected:
            raise ValueError(f"supervised-control input changed: {path}")
    for name, expected in plan["source"].items():
        path = Path(__file__) if name == "script" else RESEARCH / "training/obruxo_training" / name
        if sha256_bytes(path.read_bytes().replace(b"\r\n", b"\n")) != expected:
            raise ValueError(f"supervised-control code changed: {name}")
    return plan


def metrics(model, feature_rows, targets, codec) -> dict:
    with torch.no_grad():
        predictions = model(batch_features([{"features": row} for row in feature_rows]))
        terms = loss_terms(predictions, targets, codec)
    return {name: float(value.detach().cpu()) for name, value in terms.items()
            if isinstance(value, torch.Tensor) and value.ndim == 0}


def predict_rows(model, feature_rows):
    """Run a selected update as one caller-sized model batch, then expose row views."""
    values = model(batch_features([{"features": row} for row in feature_rows]))
    return [{"continuous": values["continuous"][index:index + 1],
             "categorical": [scores[index:index + 1] for scores in values["categorical"]]}
            for index in range(len(feature_rows))]


def execute(output: Path) -> dict:
    plan = checked_plan(output)
    if (output / "run.json").exists():
        raise FileExistsError("this diagnostic already has a run record; use a new output directory")
    started = perf_counter()
    started_unix = time()
    write_json(output / "run.json", {"plan_id": plan["plan_id"], "started_unix": started_unix})
    monitor = MemoryMonitor()
    monitor.start()
    session = None
    curve, rows, failure, final_fit, fit_seconds = [], [], None, None, 0.
    try:
        pool = PilotPool.load(Path(plan["pool_path"]))
        inputs = load_inputs(Path(plan["fit_output"]), pool)
        source_plan = inputs.plan
        if pool.pool_id != plan["pool_id"] or canonical_sha256(source_plan) != plan["fit_plan_id"]:
            raise ValueError("source pool or balanced plan changed")

        indexed_cells = [(index, source_plan["fit_audit_cells"][index]) for index in source_plan["grid_comparisons"]
                         if source_plan["fit_audit_cells"][index]["input_view"] == source_plan["fit_audit_cells"][index]["playback_view"]]
        if len(indexed_cells) != 18:
            raise ValueError("expected exactly one same-performance input per balanced source clip")
        cell_indices, cells = [index for index, _ in indexed_cells], [cell for _, cell in indexed_cells]
        feature_rows = [inputs.features[inputs.key(cell)] for cell in cells]
        targets = [PilotTarget.from_dict(cell["recipe"]["target"]) for cell in cells]
        initial_rows = {(row["index"], row["arm"]): row for row in lines(Path(plan["fit_output"]) / "finals.jsonl")}
        initial_scores = [initial_rows[(index, "initial")]["scores"] for index in cell_indices]

        if plan["training_reuse"]:
            previous = Path(plan["training_reuse"]["path"])
            artifact = torch.load(previous / "diagnostic-supervised.pt", map_location="cpu", weights_only=True)
            model, frontend, codec = restore_trainable(artifact)
            if (tensor_digest(model.state_dict()) != plan["training_reuse"]["model_tensor_sha256"]
                    or codec.to_dict() != inputs.codec.to_dict()):
                raise ValueError("reused supervised candidate differs from the frozen codec or model identity")
            curve = read(previous / "training-curve.json")
            final_fit = metrics(model, feature_rows, targets, inputs.codec)
            fit_seconds = read(previous / "result.json")["training"]["seconds"]
            atomic_torch_save(output / "diagnostic-supervised.pt", artifact)
        else:
            model, optimizer = initialize_balanced(inputs)
            curve = [{"step": 0, **metrics(model, feature_rows, targets, inputs.codec)}]
            weights = REHEARSAL["paired_weights"] + REHEARSAL["reference_weights"]
            if len(weights) != 4:
                raise ValueError("balanced schedule must provide two paired and two reference examples")
            fit_started = perf_counter()
            for step, update in enumerate(source_plan["updates"], start=1):
                observations = inputs.observations(update)
                selected = [source_plan["fit_audit_cells"][indices[0]] for indices in observations]
                if len(selected) != 4:
                    raise ValueError("the frozen update must fit the declared four-example model batch")
                predictions = predict_rows(model, [inputs.features[inputs.key(cell)] for cell in selected])
                losses = [loss_terms(prediction, [PilotTarget.from_dict(cell["recipe"]["target"])], inputs.codec)["loss"]
                          for cell, prediction in zip(selected, predictions, strict=True)]
                loss = sum(weight * value for weight, value in zip(weights, losses, strict=True))
                if not torch.isfinite(loss):
                    raise ValueError(f"nonfinite supervised loss at update {step}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.correction.parameters(), 5.)
                if not torch.isfinite(norm):
                    raise ValueError(f"nonfinite supervised gradient at update {step}")
                optimizer.step()
                model.require_frozen()
                curve.append({"step": step, "scheduled_loss": float(loss.detach()), "gradient_norm": float(norm),
                              **metrics(model, feature_rows, targets, inputs.codec)})
                if step % 8 == 0 or step == STEPS:
                    atomic_torch_save(output / "training-checkpoint.pt", {"step": step, "model": model.state_dict(),
                                      "optimizer": optimizer.state_dict(), "rng": capture_rng("cpu"), "curve": curve})
                    write_json(output / "training-curve.json", curve)
            model.eval()
            fit_seconds = perf_counter() - fit_started
            final_fit = metrics(model, feature_rows, targets, inputs.codec)
            atomic_torch_save(output / "diagnostic-supervised.pt",
                              export_inward(model, inputs.frontend, inputs.codec, pool.pool_id))

        session_plan = {**source_plan, "limits": {**source_plan["limits"], "wall_seconds": MAX_WALL_SECONDS,
                        "process_tree_bytes": MAX_PROCESS_BYTES, "qualification_attempts": MAX_NATIVE_RENDERS,
                        "training_attempts": 0}}
        session = PairedSession(output, session_plan, inputs.codec, monitor)
        with torch.no_grad():
            predictions = model(batch_features([{"features": row} for row in feature_rows]))
            patches = inputs.adapter.export(predictions["continuous"], predictions["categorical"])
        jobs = [(patch, Performance.from_dict(cell["recipe"]["performances"][cell["playback_view"]]),
                 "qualification", f"balanced-supervised:{index:02d}")
                for index, (cell, patch) in enumerate(zip(cells, patches, strict=True))]
        rendered = []
        for index, job in enumerate(jobs, start=1):
            rendered.append(session.render(*job))
            print(json.dumps({"stage": "vital_render", "completed": index, "planned": len(jobs)}), flush=True)
        measurement = SpectralTargets()
        for index, (cell, target, patch, pair, baseline) in enumerate(zip(cells, targets, patches, rendered, initial_scores, strict=True)):
            audio, receipt = pair
            if audio is None:
                raise RuntimeError(f"Vital rejected supervised prediction {index}")
            performance = Performance.from_dict(cell["recipe"]["performances"][cell["playback_view"]])
            require_request(inputs.codec, session_plan, patch, performance, receipt)
            reference_tensor, _source_receipt = inputs.source(cell_indices[index])
            reference = reference_tensor.numpy()
            source_hash = cell["recipe"]["source_audio_sha256"][cell["playback_view"]]
            if audio.shape != reference.shape or audio_float32_sha256(reference) != source_hash:
                raise ValueError("Vital prediction and balanced source audio do not align")
            reference_targets = measurement(torch.from_numpy(reference.copy()).T[None])
            candidate_targets = measurement(torch.from_numpy(audio.copy()).T[None])
            spectral = float(spectral_loss(candidate_targets, reference_targets)["loss"].item())
            mrstft = float(audio_distance(reference, audio)["mrstft"])
            rows.append({"cell_index": cell_indices[index], "training_index": cell["training_index"],
                         "source_sha256": source_hash, "prediction": patch.to_dict(), "target": target.to_dict(),
                         "spectral": spectral, "mrstft": mrstft, "initial_spectral": baseline["spectral"],
                         "initial_mrstft": baseline["mrstft"],
                         "render_audio_sha256": receipt["qa"]["audio_float32_sha256"], "receipt": receipt})
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
    finally:
        monitor.close()

    elapsed = perf_counter() - started
    if monitor.errors:
        failure = failure or "memory monitor reported sampling errors"
    if monitor.peak_bytes is None or monitor.peak_bytes > MAX_PROCESS_BYTES:
        failure = failure or "process-tree memory cap was exceeded or could not be measured"
    if elapsed > MAX_WALL_SECONDS:
        failure = failure or "20-minute wall-time cap was exceeded"
    if len(rows) != MAX_NATIVE_RENDERS:
        failure = failure or f"only {len(rows)} of {MAX_NATIVE_RENDERS} Native renders were scored"

    means = {name: float(np.mean([row[name] for row in rows])) for name in
             ("spectral", "mrstft", "initial_spectral", "initial_mrstft")} if rows else {}
    per_sound = {str(state): {name: float(np.mean([row[name] for row in rows if row["training_index"] == state]))
                              for name in ("spectral", "mrstft", "initial_spectral", "initial_mrstft")}
                 for state in sorted({row["training_index"] for row in rows})}
    fit_pass = bool(final_fit and final_fit["continuous_mae"] < FIT_REFERENCE["continuous_mae"]
                    and final_fit["categorical_accuracy"] == FIT_REFERENCE["categorical_accuracy"]
                    and final_fit["loss"] < FIT_REFERENCE["final_loss_ratio"] * curve[0]["loss"])
    result = {"format": "balanced_supervised_learnability_result_v1", "plan_id": plan["plan_id"],
              "complete": failure is None, "failure": failure, "verified_balanced_input_plan_id": plan["fit_plan_id"],
              "training": {"updates": len(curve) - 1, "balanced_examples": 18, "seconds": fit_seconds,
                           "curve": curve, "initial": curve[0] if curve else None, "final": final_fit,
                           "normalized_fit_reference_pass": fit_pass},
              "real_vital": {"renders": len(rows), "mean": means, "per_sound": per_sound, "examples": rows,
                             "improvement_over_initial": {name: 1 - means[name] / max(means[f"initial_{name}"], 1e-12)
                                                           for name in ("spectral", "mrstft")} if means else {}},
              "resources": {"wall_seconds": elapsed, "fit_seconds": fit_seconds,
                            "peak_process_tree_rss_bytes": monitor.peak_bytes, "rss_sampling_errors": monitor.errors,
                            "renders": len(rows)},
              "test_split_used": False, "qualified_for_release": False, "model_admitted": False}
    write_json(output / "training-curve.json", curve)
    write_json(output / "result.json", result)
    return result


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-output", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-training", type=Path)
    parser.add_argument("--mode", choices=("prepare", "execute"), default="prepare")
    args = parser.parse_args()
    with cpu_settings():
        result = prepare(args.fit_output, args.pool, args.output, args.reuse_training) if args.mode == "prepare" else execute(args.output)
    print(json.dumps({key: value for key, value in result.items() if key not in ("source", "input_files", "curve", "examples")}, indent=2))
    if args.mode == "execute" and not result["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

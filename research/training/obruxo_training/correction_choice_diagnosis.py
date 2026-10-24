"""Inspect actual category learning and retained single-choice sound measurements."""

import json
import math
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch

from obruxo_data.hashing import canonical_sha256, sha256_bytes

from . import category_scale_refresh as refresh
from .audit import write_json
from .benchmark import MemoryMonitor
from .codec import PilotTarget
from .conditioned_basic import restore_conditioned
from .conditioned_bridge import read
from .features import tensor_digest, validate_features
from .model import batch_features
from .native_fit import digest, record_features

LIMITS = {"wall_seconds": 300, "process_tree_bytes": 4 * 1024**3, "report_bytes": 32 * 1024**2}


def choice_row(estimate, head):
    costs, anchor = estimate["categorical"][head], estimate["anchor"]["categorical"][head]
    if (len(costs) != (3, 2)[head] or type(anchor) is not int or not 0 <= anchor < len(costs)
            or any(not math.isfinite(value) or value < 0 for value in costs) or costs[anchor] != estimate["value"]):
        raise ValueError("saved category costs differ from their measured anchor")
    best = min(range(len(costs)), key=lambda index: (costs[index], index != anchor, index))
    gain = costs[anchor] - costs[best]
    return {"head": head, "anchor": anchor, "best_single_edit": best, "measured_costs": costs,
            "lower_cost_edit_available": gain > 0, "absolute_gain": gain, "relative_gain": gain / max(costs[anchor], 1e-12)}


def source_contract(source, initial):
    proof = read(source / "verification.json")
    if (not proof["verified"] or not proof["offline_training_replay_exact"] or not proof["all_native_accepted"]
            or not proof["complete"] or proof["committed_updates_per_arm"] != 48 or proof["comparison_waveforms_rescored"] != 312
            or proof["matched_comparison_passed"] or proof["test_split_used"] or proof["plan"]["validation"]
            or proof["plan"]["format"] != "conditioned_basic_short_native_fit_v1"):
        raise ValueError("choice diagnosis requires the verified failed 48-update correction fit")
    files = {name: digest(source / name) for name in proof["evidence_sha256"]}
    if files != proof["evidence_sha256"] or digest(initial) != proof["plan"]["initial_file_sha256"]:
        raise ValueError("verified correction evidence or initializer changed")
    for index, row in read(source / "records.json").items():
        name = f"records/{int(index):03d}.pt"
        files[name] = digest(source / name)
        if files[name] != row["sha256"]:
            raise ValueError("verified correction recording changed")
    for arm in ("parameter", "spectral"):
        files[f"diagnostic-{arm}.pt"] = digest(source / f"diagnostic-{arm}.pt")
    files["verification.json"] = digest(source / "verification.json")
    training = Path(__file__).parents[1]
    code = {name: sha256_bytes((training / (name if "/" in name else f"obruxo_training/{name}")).read_bytes().replace(b"\r\n", b"\n"))
            for name in proof["plan"]["source"]}
    if code != proof["plan"]["source"]:
        raise ValueError("choice diagnosis changed the verified prediction or measurement implementation")
    return {"format": "saved_correction_choice_diagnosis_v1", "parent_plan_id": proof["plan_id"], "parent_files": files,
            "initial_file_sha256": digest(initial), "numerical_source": code,
            "diagnosis_source_sha256": sha256_bytes(Path(__file__).read_bytes().replace(b"\r\n", b"\n")), "limits": LIMITS,
            "native_requests": 0, "optimizer_updates": 0, "repeated_observations_are_not_independent_cases": True}


def summarize(predictions, choices):
    groups = []
    for model, view in sorted({(row["model"], row["view"]) for row in predictions}):
        selected = [row for row in predictions if row["model"] == model and row["view"] == view]
        groups.append({"model": model, "view": view, "states": len(selected),
                       "mean_active_control_mae": mean(row["active_control_mae"] for row in selected),
                       "mean_normalized_control_change": mean(row["control_change"] for row in selected),
                       "category_accuracy": [mean(row["category_correct"][head] for row in selected) for head in (0, 1)],
                       "category_changes_from_initial": [sum(row["category_changed"][head] for row in selected) for head in (0, 1)]})
    measured = []
    for head, view in sorted({(row["head"], row["input_view"]) for row in choices}):
        selected = [row for row in choices if (row["head"], row["input_view"]) == (head, view)]
        improving = [row for row in selected if row["lower_cost_edit_available"]]
        measured.append({"head": head, "input_view": view, "repeated_observations": len(selected), "lower_cost_edits": len(improving),
                         "mean_relative_gain_all_observations": mean(row["relative_gain"] for row in selected),
                         "mean_relative_gain_when_edit_available": mean(row["relative_gain"] for row in improving) if improving else None})
    return {"prediction_groups": groups, "measured_choice_groups": measured}


@torch.no_grad()
def inspect(source, initial):
    proof = read(source / "verification.json")
    plan = proof["plan"]
    models = {}
    paths = {"initial": initial, **{name: source / f"diagnostic-{name}.pt" for name in ("parameter", "spectral")}}
    for name, path in paths.items():
        models[name], _frontend, codec = restore_conditioned(torch.load(path, map_location="cpu", weights_only=True))
        expected = plan["initial_model_sha256"] if name == "initial" else proof["final_model_sha256"][name]
        if tensor_digest(models[name].state_dict()) != expected:
            raise ValueError("diagnostic prediction model changed")
        models[name].eval()
    predictions = []
    for index, recipe in enumerate(plan["training"]):
        record = torch.load(source / "records" / f"{index:03d}.pt", map_location="cpu", weights_only=True)
        if record["recipe_id"] != canonical_sha256(recipe):
            raise ValueError("diagnostic training recipe changed")
        target = PilotTarget.from_dict(recipe["target"])
        mask = codec.continuous_mask(target)
        for view in range(3):
            features = record_features(record, view)
            validate_features(features)
            if tensor_digest(features) != recipe["source_feature_sha256"][view]:
                raise ValueError("diagnostic training input changed")
            outputs = {name: model(batch_features([{"features": features}])) for name, model in models.items()}
            baseline = [int(value.argmax()) for value in outputs["initial"]["categorical"]]
            for name, output in outputs.items():
                categories = [int(value.argmax()) for value in output["categorical"]]
                continuous = output["continuous"][0].tolist()
                predictions.append({"model": name, "state_id": recipe["state_id"], "view": view,
                    "input_audio_sha256": record["audio_sha256"][view], "predicted_continuous": continuous, "predicted_categories": categories,
                    "active_control_mae": mean(abs(value - desired) for value, desired, active in zip(continuous, target.continuous, mask, strict=True) if active),
                    "control_change": float((output["continuous"] - outputs["initial"]["continuous"]).abs().mean()),
                    "category_correct": [int(a == b) for a, b in zip(categories, target.categorical, strict=True)],
                    "category_changed": [int(a != b) for a, b in zip(categories, baseline, strict=True)]})
    saved = torch.load(source / "checkpoint.pt", map_location="cpu", weights_only=True)
    choices = [{"step": update["step"], "state_id": plan["training"][example["state"]]["state_id"], "input_view": example["input_view"],
                **choice_row(estimate, head)} for update in saved["trace"]
               for example, estimate in zip(update["examples"], update["estimates"], strict=True) for head in (0, 1)]
    if len(predictions) != 108 or len(choices) != 192:
        raise ValueError("choice diagnosis lost training inputs or repeated native observations")
    return predictions, choices


def run(source, initial, output, *, execute=False):
    started, monitor = perf_counter(), MemoryMonitor()
    monitor.start()
    try:
        plan = source_contract(source, initial)
        if not execute:
            output.mkdir(parents=True, exist_ok=False)
            write_json(output / "plan.json", plan)
        else:
            if read(output / "plan.json") != plan or (output / "summary.json").exists():
                raise ValueError("preserve the existing diagnosis or changed frozen input")
            with refresh.cpu_settings():
                predictions, choices = inspect(source, initial)
            if source_contract(source, initial) != plan:
                raise ValueError("choice diagnosis changed parent evidence")
    finally:
        monitor.close()
    resources = {"elapsed_seconds": perf_counter() - started, "peak_process_tree_rss_bytes": monitor.peak_bytes, "rss_sampling_errors": monitor.errors}
    if resources["elapsed_seconds"] >= 300 or monitor.peak_bytes is None or monitor.peak_bytes > LIMITS["process_tree_bytes"] or monitor.errors:
        raise ValueError("choice diagnosis exceeded five minutes or 4 GiB")
    report = {"plan_id": canonical_sha256(plan), "native_requests": 0, "optimizer_updates": 0, "resources": resources, "qualified_for_release": False}
    if execute:
        report.update(parent_evidence_unchanged=True, prediction_rows=predictions, measured_choice_rows=choices, **summarize(predictions, choices))
    if len(json.dumps(report).encode("utf-8")) > LIMITS["report_bytes"]:
        raise ValueError("choice diagnosis exceeded its report allowance")
    write_json(output / ("summary.json" if execute else "preparation.json"), report)
    return {key: value for key, value in report.items() if key not in ("prediction_rows", "measured_choice_rows")}

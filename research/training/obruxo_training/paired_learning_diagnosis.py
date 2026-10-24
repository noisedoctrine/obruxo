"""Inspect saved feature sensitivity and paired gradients without new native work."""

import copy
from contextlib import contextmanager
import json
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import torch

from obruxo_data.hashing import canonical_sha256, sha256_bytes

from . import category_scale_refresh as refresh
from . import paired_native_step as pair
from .audit import write_json
from .benchmark import MemoryMonitor
from .codec import PilotTarget
from .features import validate_features
from .model import batch_features
from .native_fit import digest, record_features
from .native_gradient import measured_loss
from .paired_native_audit import equal_tree
from .paired_performance_objective import INPUT_TRANSFER, evaluation_views, seed_external_streams
from .trainer import capture_rng, restore_model


def alignment(names, first, second):
    groups = {}
    for name, a, b in zip(names, first, second, strict=True):
        if a is None and b is None:
            continue
        a = torch.zeros_like(b) if a is None else a
        b = torch.zeros_like(a) if b is None else b
        a, b = a.detach().double(), b.detach().double()
        values = groups.setdefault(name.split(".")[0], [0., 0., 0.])
        values[0] += float((a * b).sum())
        values[1] += float(a.square().sum())
        values[2] += float(b.square().sum())
    groups["all"] = [sum(row[index] for row in groups.values()) for index in range(3)]
    result = {}
    for name, (dot, first_squared, second_squared) in groups.items():
        a, b = first_squared**.5, second_squared**.5
        result[name] = {"cosine": max(-1., min(1., dot / (a * b))) if a and b else None,
                        "original_norm": a, "wider_norm": b, "wider_to_original_norm": b / a if a else None}
    return result


@contextmanager
def observe_gradients(model, rows):
    original = pair.native_reconstruction_loss
    names, parameters = zip(*model.named_parameters(), strict=True)
    def observed(prediction, adapter, evaluate, *, temperature):
        loss, estimates = original(prediction, adapter, evaluate, temperature=temperature)
        raw, logits = prediction["continuous"], prediction["categorical"]
        encoded = adapter(raw, logits, temperature=temperature, sample=True)
        gradients = []
        for index, estimate in enumerate(estimates):
            choices = (encoded[index:index + 1, 9:12], encoded[index:index + 1, 12:])
            component = measured_loss(raw[index:index + 1], choices, estimate)
            gradients.append(torch.autograd.grad(component, parameters, retain_graph=True, allow_unused=True))
        if len(gradients) != 2:
            raise ValueError("gradient diagnosis requires the two reciprocal observations")
        rows.append(alignment(names, *gradients))
        # retain_grad hooks also run during autograd.grad; leave the real step's trace untouched.
        for value in (raw, *logits):
            value.grad = None
        if any(parameter.grad is not None for parameter in parameters):
            raise ValueError("gradient observation accumulated a training parameter gradient")
        return loss, estimates
    pair.native_reconstruction_loss = observed
    try:
        yield
    finally:
        pair.native_reconstruction_loss = original


def replay_alignment(initial, codec, plan, records, measurements, saved):
    if plan.get("reconstruction_objective") != INPUT_TRANSFER or saved["step"] != len(plan["updates"]):
        raise ValueError("diagnosis requires a complete input-and-transfer checkpoint")
    models = {name: copy.deepcopy(initial).train() for name in ("parameter", "spectral")}
    optimizers = {name: torch.optim.Adam(model.parameters(), lr=plan["optimizer"]["learning_rate"]) for name, model in models.items()}
    seed_external_streams(plan)
    torch.manual_seed(plan["optimizer"]["seed"])
    rows, by_step = [], {}
    for row in measurements:
        by_step.setdefault(row["step"], []).append(row)
    if set(by_step) != set(range(1, len(plan["updates"]) + 1)) or len(saved["trace"]) != len(plan["updates"]):
        raise ValueError("saved measurement coverage changed")
    with observe_gradients(models["spectral"], rows):
        for step, update in enumerate(plan["updates"]):
            expected, cursor = saved["trace"][step], 0
            examples, measured = update["examples"], by_step[step + 1]
            if expected["attempt"] != step or expected["examples"] != examples:
                raise ValueError("diagnosis requires the uninterrupted declared update schedule")
            def evaluate(index, candidate):
                nonlocal cursor
                example, losses = examples[index], []
                for view in evaluation_views(plan, example):
                    if cursor >= len(measured):
                        raise ValueError("incomplete saved performance component")
                    row = measured[cursor]
                    if (row["query"] != cursor or row["example"] != index or row["optimizer_attempt"] != step
                            or row["view"] != view or row["candidate"] != candidate.to_dict()
                            or any(row[key] != example[key] for key in ("state", "input_view"))):
                        raise ValueError("saved native component differs from the replayed prediction")
                    losses.append(row["loss"])
                    cursor += 1
                return sum(losses) / len(losses)
            features = [record_features(records[row["state"]], row["input_view"]) for row in examples]
            targets = [PilotTarget.from_dict(plan["training"][row["state"]]["target"]) for row in examples]
            result = pair.matched_pair_step(models, optimizers, features, targets, codec, evaluate, step, len(plan["updates"]),
                                            scales=plan["production_training"]["scales"])
            result["gradient_queries"] = cursor
            if cursor != len(measured) or {**{key: expected[key] for key in ("attempt", "step", "examples")}, **result} != expected:
                raise ValueError("gradient observation changed the original update trace")
    if (any(not equal_tree(model.state_dict(), saved["models"][name]) or not equal_tree(optimizers[name].state_dict(), saved["optimizers"][name])
            for name, model in models.items()) or not equal_tree(capture_rng("cpu"), saved["rng"])):
        raise ValueError("gradient diagnosis did not reproduce final model/Adam/RNG exactly")
    return models, rows


@torch.no_grad()
def feature_sensitivity(models, codec, plan, records):
    rows = []
    for index, recipe in enumerate(plan["training"]):
        target = PilotTarget.from_dict(recipe["target"])
        mask = codec.continuous_mask(target)
        original = record_features(records[index], 0)
        for view in range(3):
            current = record_features(records[index], view)
            variants = {"recording": current}
            if view:
                variants.update(wide_timbre_original_pitch={"timbre": current["timbre"], "pitch": original["pitch"]},
                                original_timbre_wide_pitch={"timbre": original["timbre"], "pitch": current["pitch"]})
            for variant, features in variants.items():
                validate_features(features)
                for name, model in models.items():
                    prediction = model(batch_features([{"features": features}]))
                    continuous = prediction["continuous"][0].tolist()
                    categories = [int(scores[0].argmax()) for scores in prediction["categorical"]]
                    rows.append({"state_id": recipe["state_id"], "view": view, "variant": variant, "model": name,
                                 "active_control_mae": mean(abs(a - b) for a, b, active in zip(continuous, target.continuous, mask, strict=True) if active),
                                 "category_correct": [int(a == b) for a, b in zip(categories, target.categorical, strict=True)],
                                 "predicted_continuous": continuous, "predicted_categories": categories})
    return rows


def summarize(gradients, features):
    groups = {}
    for name in gradients[0]:
        valid = [row[name]["cosine"] for row in gradients if row[name]["cosine"] is not None]
        ratios = [row[name]["wider_to_original_norm"] for row in gradients if row[name]["wider_to_original_norm"] is not None]
        groups[name] = {"paired_updates": len(gradients), "nonzero_pairs": len(valid), "negative_cosine_pairs": sum(value < 0 for value in valid),
                        "median_cosine": median(valid) if valid else None, "median_wider_to_original_norm": median(ratios) if ratios else None}
    feature_groups = []
    for model, view, variant in sorted({(row["model"], row["view"], row["variant"]) for row in features}):
        selected = [row for row in features if (row["model"], row["view"], row["variant"]) == (model, view, variant)]
        feature_groups.append({"model": model, "view": view, "variant": variant, "states": len(selected),
                               "mean_active_control_mae": mean(row["active_control_mae"] for row in selected),
                               "category_accuracy": [mean(row["category_correct"][head] for row in selected) for head in (0, 1)]})
    return {"gradient_groups": groups, "feature_groups": feature_groups}


def source_hashes(source, initial):
    proof = json.loads((source / "verification.json").read_text())
    if (not proof["verified"] or not proof["offline_training_replay_exact"] or not proof["all_native_accepted"]
            or not proof["complete"] or proof["recovered_timeouts"] or proof["committed_updates_per_arm"] != 128
            or proof["plan"].get("reconstruction_objective") != INPUT_TRANSFER or proof["interrupted_optimizer_attempts_replayed"]
            or proof["interruption_checkpoints_replayed"]):
        raise ValueError("diagnosis requires the independently verified uninterrupted 128-update parent")
    files = {name: digest(source / name) for name in proof["evidence_sha256"]}
    if files != proof["evidence_sha256"] or digest(initial) != proof["plan"]["initial_file_sha256"]:
        raise ValueError("verified parent evidence or initializer changed")
    records = json.loads((source / "records.json").read_text())
    for index, row in records.items():
        name = f"records/{int(index):03d}.pt"
        files[name] = digest(source / name)
        if files[name] != row["sha256"]:
            raise ValueError("verified source feature file changed")
    files["verification.json"] = digest(source / "verification.json")
    names = ("paired_native_step.py", "paired_performance_objective.py", "production_controls.py", "native_inverse.py", "native_gradient.py",
             "model.py", "features.py", "codec.py", "trainer.py", "spectral_controls.py", "paired_native_audit.py", "native_fit.py",
             "category_scale_refresh.py", "benchmark.py")
    code = {name: sha256_bytes(Path(__file__).with_name(name).read_bytes().replace(b"\r\n", b"\n")) for name in names}
    if any(value != proof["plan"]["source"][name] for name, value in code.items()):
        raise ValueError("diagnostic replay differs from the verified numerical source")
    return {"parent_files": files, "initial_file_sha256": digest(initial), "numerical_source": code,
            "diagnosis_source_sha256": digest(Path(__file__)), "parent_plan_id": proof["plan_id"],
            "limits": {"wall_seconds": 300, "process_tree_bytes": 4 * 1024**3, "report_bytes": 64 * 1024**2}}


def run_diagnosis(source, initial, output, *, execute=False):
    source, initial, output = map(Path, (source, initial, output))
    started, monitor = perf_counter(), MemoryMonitor()
    monitor.start()
    try:
        frozen = source_hashes(source, initial)
        if not execute:
            output.mkdir(parents=True, exist_ok=False)
            write_json(output / "plan.json", frozen)
        else:
            if json.loads((output / "plan.json").read_text()) != frozen or (output / "summary.json").exists():
                raise ValueError("preserve the previous diagnosis or changed frozen input")
            with refresh.cpu_settings():
                saved = torch.load(source / "checkpoint.pt", map_location="cpu", weights_only=True)
                plan = json.loads((source / "plan.json").read_text())
                network, _frontend, codec = restore_model(torch.load(initial, map_location="cpu", weights_only=True))
                records = {}
                for index in range(len(plan["training"])):
                    record = torch.load(source / "records" / f"{index:03d}.pt", map_location="cpu", weights_only=True)
                    records[index] = {"features": [record_features(record, view).copy() for view in range(3)]}
                measured = [json.loads(line) for line in (source / "measurements.jsonl").read_text().splitlines()]
                models, gradients = replay_alignment(network, codec, plan, records, measured, saved)
                models = {"initial": network.eval(), **{name: model.eval() for name, model in models.items()}}
                features = feature_sensitivity(models, codec, plan, records)
            if source_hashes(source, initial) != frozen:
                raise ValueError("diagnosis changed immutable parent evidence")
    finally:
        monitor.close()
    resources = {"elapsed_seconds": perf_counter() - started, "peak_process_tree_rss_bytes": monitor.peak_bytes, "rss_sampling_errors": monitor.errors}
    if (resources["elapsed_seconds"] >= 300 or monitor.peak_bytes is None or monitor.peak_bytes > 4 * 1024**3 or monitor.errors):
        raise ValueError("saved-data diagnosis exceeds five minutes or four GiB")
    if not execute:
        report = {"plan_id": canonical_sha256(frozen), "new_native_requests": 0, "new_optimizer_updates": 0, "preparation_resources": resources}
        write_json(output / "preparation.json", report)
        return report
    report = {"plan_id": canonical_sha256(frozen), "parent_plan_id": frozen["parent_plan_id"], "parent_evidence_unchanged": True,
              "analysis_replay_exact": True, "replayed_updates_per_arm": len(gradients), "new_native_requests": 0, "new_optimizer_updates": 0,
              "test_split_used": False, "qualified_for_release": False, "resources": resources, **summarize(gradients, features),
              "gradient_rows": [{"step": index + 1, "state_id": plan["training"][plan["updates"][index]["examples"][0]["state"]]["state_id"],
                                 "wide_view": plan["updates"][index]["examples"][0]["view"], "groups": row} for index, row in enumerate(gradients)],
              "feature_rows": features}
    if len(json.dumps(report).encode()) > frozen["limits"]["report_bytes"]:
        raise ValueError("diagnostic report exceeds its declared size")
    write_json(output / "summary.json", report)
    return {key: value for key, value in report.items() if key not in ("gradient_rows", "feature_rows")}

"""Inspect audited final gaps and auxiliary-gradient alignment without new native work."""

from pathlib import Path
from statistics import median
from time import perf_counter

import torch

from obruxo_data.hashing import canonical_sha256, sha256_bytes

from . import category_scale_refresh as refresh
from .audit import write_json
from .basic_direct_fit import SoundStore, initialize, lines, step
from .basic_direct_fit_data import Inputs
from .benchmark import MemoryMonitor
from .codec import PilotTarget
from .conditioned_bridge import read
from .continuous_auxiliary import SPEC, paired_continuous_loss
from .model import batch_features
from .native_fit import digest
from .paired_native_audit import equal_tree
from .trainer import capture_rng

LIMITS = {"wall_seconds": 300, "process_tree_bytes": 4 * 1024**3, "archive_bytes": 64 * 1024**2}


def final_gaps(plan, finals):
    rows = []
    for recipe in plan["training"]:
        indices = [i for i, cell in enumerate(plan["fit_audit_cells"]) if cell["recipe"]["state_id"] == recipe["state_id"] and cell["input_view"] == 1]
        if len(indices) != 2 or {plan["fit_audit_cells"][i]["playback_view"] for i in indices} != {0, 1}:
            raise ValueError("held-input gap requires both own and transferred playback")
        a, b = (finals[(indices[0], arm)]["candidate"] for arm in ("spectral", "parameter"))
        if any(finals[(i, arm)]["candidate"] != candidate for i in indices for arm, candidate in (("spectral", a), ("parameter", b))):
            raise ValueError("the same inferred input patch changed across playback performances")
        delta = sum(finals[(i, "spectral")]["scores"]["spectral"] - finals[(i, "parameter")]["scores"]["spectral"] for i in indices) / 2
        rows.append({"state_id": recipe["state_id"], "combined_sound_gap": delta, "same_choices": a["categorical"] == b["categorical"],
                     "sound_choices": a["categorical"], "parameter_choices": b["categorical"], "source_choices": recipe["target"]["categorical"],
                     "continuous_differences": [x - y for x, y in zip(a["continuous"], b["continuous"], strict=True)]})
    ordered = sorted(rows, key=lambda row: (-row["combined_sound_gap"], row["state_id"]))
    return {"ranked_patches": ordered, "sound_worse_patches": sum(row["combined_sound_gap"] > 0 for row in rows),
            "sound_better_patches": sum(row["combined_sound_gap"] < 0 for row in rows),
            "net_gap_sum_over_patches": sum(row["combined_sound_gap"] for row in rows),
            "same_choice_gap_sum": sum(row["combined_sound_gap"] for row in rows if row["same_choices"]),
            "different_choice_gap_sum": sum(row["combined_sound_gap"] for row in rows if not row["same_choices"]),
            "same_choice_patches": sum(row["same_choices"] for row in rows)}


def output_alignment(prediction, target, native, codec):
    mask = prediction.new_tensor(codec.continuous_mask(target))
    label = 2 * (prediction.detach()[0] - prediction.new_tensor(target.continuous)) * mask / mask.sum()
    coefficients = prediction.new_tensor(native)
    return {"gradient_dot": float((label.double() * coefficients.double()).sum()),
            "native_norm": float(coefficients.double().norm()), "label_norm": float(label.double().norm())}


def inspect_gradients(inputs, source, saved):
    models, optimizers = initialize(inputs)
    store = SoundStore(source, inputs)
    neural, output = [], []
    for index, row in enumerate(saved["trace"]):
        before = capture_rng("cpu")
        observations = inputs.observations(inputs.plan["updates"][index])[:2]
        predictions, targets = [], []
        for observation, cells in enumerate(observations):
            cell = inputs.plan["fit_audit_cells"][cells[0]]
            prediction = models["spectral"](batch_features([{"features": inputs.features[inputs.key(cell)]}]))
            target = PilotTarget.from_dict(cell["recipe"]["target"])
            estimate = row["estimates"][observation]
            if prediction["continuous"].detach()[0].tolist() != estimate["anchor"]["continuous"]:
                raise ValueError("auxiliary analysis reached stale native controls")
            predictions.append(prediction)
            targets.append(target)
            output.append({"step": index + 1, "state_id": cell["recipe"]["state_id"], "input_view": cell["input_view"],
                           **output_alignment(prediction["continuous"], target, estimate["continuous"], inputs.codec)})
        auxiliary = paired_continuous_loss(predictions, targets, inputs.codec)
        values = torch.autograd.grad(auxiliary, tuple(models["spectral"].correction.parameters()))
        vector = torch.cat([value.detach().flatten().double() for value in values])
        if not equal_tree(before, capture_rng("cpu")):
            raise ValueError("auxiliary inspection changed randomness")
        result = step(models, optimizers, inputs, store, index)
        expected = {"step": index + 1, "update": inputs.plan["updates"][index], **result}
        if expected != {key: value for key, value in row.items() if key != "attempt"}:
            raise ValueError("auxiliary inspection changed the original training replay")
        native = torch.cat([p.grad.detach().flatten().double() for p in models["spectral"].correction.parameters()])
        denominator = float(native.norm() * vector.norm())
        native_norm = row["gradient_norms"]["spectral"]
        neural.append({"step": index + 1, "auxiliary_loss": float(auxiliary.detach()),
                       "gradient_cosine": float(torch.dot(native, vector)) / denominator if denominator else None,
                       "native_unclipped_norm": native_norm, "auxiliary_norm": float(vector.norm()),
                       "auxiliary_to_native_norm_ratio": float(vector.norm()) / native_norm if native_norm else None})
    if (not all(equal_tree(model.state_dict(), saved["models"][arm]) and equal_tree(optimizers[arm].state_dict(), saved["optimizers"][arm])
                for arm, model in models.items()) or not equal_tree(capture_rng("cpu"), saved["rng"])
            or store.used_probes != set(store.probes)):
        raise ValueError("inspected model, Adam, RNG or probe replay differs from the audited fit")
    return {"neural_steps": neural, "paired_output_observations": output, "offline_replayed_updates_per_arm": saved["step"],
            "scalar_native_probes_reused": len(store.used_probes), "model_adam_rng_replay_exact": True}


def make_plan(source, parent, bridge, features):
    proof, original = read(source / "verification.json"), read(source / "plan.json")
    if (not proof["verified"] or not proof["complete"] or not proof["offline_training_replay_exact"] or not proof["native_receipts_resolved"]
            or proof["comparison_waveforms_rescored"] != 312 or proof["committed_updates_per_arm"] != 48
            or proof["matched_comparison_passed"] or proof["test_split_used"] or proof["qualified_for_release"]):
        raise ValueError("gap diagnosis requires the complete audited direct fit and its unchanged gates")
    files = {f"fit/{name}": digest(source / name) for name in proof["evidence_sha256"]}
    if {name: files[f"fit/{name}"] for name in proof["evidence_sha256"]} != proof["evidence_sha256"]:
        raise ValueError("audited direct evidence changed")
    if (digest(parent / "fit_cases.json") != original["input_files"]["parent/fit_cases.json"]
            or digest(bridge / "model.pt") != original["initial_file_sha256"] or digest(features) != original["input_cache_sha256"]):
        raise ValueError("direct diagnostic source recordings, initializer or features changed")
    root = Path(__file__).parents[1]
    for name, value in original["source"].items():
        path = root / name if "/" in name else Path(__file__).with_name(name)
        if sha256_bytes(path.read_bytes().replace(b"\r\n", b"\n")) != value:
            raise ValueError("audited fitting implementation changed before gap inspection")
    names = ("direct_fit_gaps.py", "continuous_auxiliary.py", "reports/scripts/inspect_direct_fit_gaps.py")
    code = {name: sha256_bytes((root / name if "/" in name else Path(__file__).with_name(name)).read_bytes().replace(b"\r\n", b"\n")) for name in names}
    return {"format": "audited_direct_gap_inspection_v1", "fit_plan_id": proof["plan_id"], "fit_verification_sha256": digest(source / "verification.json"),
            "input_files": files, "initial_file_sha256": original["initial_file_sha256"], "input_cache_sha256": original["input_cache_sha256"],
            "parent_cases_sha256": digest(parent / "fit_cases.json"), "source": code, "limits": LIMITS,
            "auxiliary": SPEC, "native_requests": 0, "experimental_training_updates": 0, "replayed_updates_per_arm": 48,
            "output_observations": 96, "final_predictions": "all_twelve_held_inputs_and_both_playback_performances",
            "frozen_gates_unchanged": True, "qualified_for_release": False}


def run(source, parent, bridge, features, output, *, execute=False):
    with refresh.cpu_settings():
        plan = make_plan(source, parent, bridge, features)
        if not execute:
            output.mkdir(parents=True, exist_ok=False)
            write_json(output / "plan.json", plan)
            return {"plan_id": canonical_sha256(plan), "native_requests": 0, "experimental_training_updates": 0}
        if read(output / "plan.json") != plan or (output / "summary.json").exists():
            raise ValueError("preserve the frozen diagnosis and any completed result")
        started, monitor = perf_counter(), MemoryMonitor()
        monitor.start()
        try:
            inputs = Inputs(parent, bridge, features, read(source / "plan.json"))
            saved = torch.load(source / "checkpoint.pt", map_location="cpu", weights_only=True)
            finals = {(row["index"], row["arm"]): row for row in lines(source / "finals.jsonl")}
            gaps = final_gaps(inputs.plan, finals)
            alignment = inspect_gradients(inputs, source, saved)
        finally:
            monitor.close()
        cosines = [row["gradient_cosine"] for row in alignment["neural_steps"] if row["gradient_cosine"] is not None]
        ratios = [row["auxiliary_to_native_norm_ratio"] for row in alignment["neural_steps"] if row["auxiliary_to_native_norm_ratio"] is not None]
        output_rows = alignment["paired_output_observations"]
        resource = {"elapsed_seconds": perf_counter() - started, "peak_process_tree_rss_bytes": monitor.peak_bytes, "rss_sampling_errors": monitor.errors}
        within = bool(resource["elapsed_seconds"] < LIMITS["wall_seconds"] and monitor.peak_bytes and monitor.peak_bytes <= LIMITS["process_tree_bytes"] and not monitor.errors)
        result = {"plan_id": canonical_sha256(plan), "gaps": gaps, "alignment": alignment,
                  "alignment_counts": {"paired_outputs_positive": sum(row["gradient_dot"] > 0 for row in output_rows), "paired_outputs": len(output_rows),
                    "neural_positive": sum(value > 0 for value in cosines), "neural_negative": sum(value < 0 for value in cosines),
                    "neural_cosine_min": min(cosines), "neural_cosine_max": max(cosines),
                    "auxiliary_to_native_norm_ratio_median": median(ratios)},
                  "resources": resource, "within_resource_limits": within, "native_requests": 0, "experimental_training_updates": 0,
                  "frozen_gates_unchanged": True, "qualified_for_release": False}
        write_json(output / "summary.json", result)
        if sum(path.stat().st_size for path in output.rglob("*") if path.is_file()) > LIMITS["archive_bytes"] or not within:
            raise ValueError("gap inspection exceeded its resource allocation")
        if make_plan(source, parent, bridge, features) != plan:
            raise ValueError("gap inspection changed its source evidence")
        return {key: result[key] for key in ("plan_id", "alignment_counts", "resources", "within_resource_limits", "native_requests", "experimental_training_updates")}

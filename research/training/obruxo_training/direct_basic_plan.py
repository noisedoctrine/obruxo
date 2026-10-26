"""Pin three previously used training sounds for a direct-control component check."""

from pathlib import Path
import importlib.metadata
import platform

import numpy as np
import torch

from obruxo_data.hashing import canonical_sha256, sha256_bytes
from obruxo_data.midi import Performance, TempoMap
from obruxo_data.render.qa import audio_float32_sha256

from .basic_direct import restore_direct
from .conditioned_bridge import read
from .conditioned_short_audit import cached_features, feature_bank
from .codec import PilotTarget
from .direct_basic_control import SPEC
from .features import tensor_digest
from .model import batch_features, predicted_targets
from .native_fit import digest
from .native_fit_storage import target_tensor_bytes
from .paired_native_evidence import rescore_waveform
from .reproducibility import FULL_PHASE_CONTROL
from .trainable_basic_fit import source_code
from .unison_fit_data import require_request
from .unison_sound_fit import lines

LIMITS = {"wall_seconds": 14400, "process_tree_bytes": 4 * 1024**3, "archive_bytes": 2 * 1024**3,
          "replay_bytes": 512 * 1024**2, "training_attempts": 4584, "qualification_attempts": 432,
          "native_executions": 2, "optimizer_attempts_per_case": 33}
SOURCE_NAMES = ("direct_basic_control.py", "direct_basic_plan.py", "direct_basic_store.py", "direct_basic_fit.py", "direct_basic_audit.py")


def code_identity(prior):
    code, _compatible = source_code(prior)
    root = Path(__file__).parents[1]
    names = [*SOURCE_NAMES, "reports/scripts/run_direct_basic_control.py"]
    code.update({name: sha256_bytes((root / name if "/" in name else Path(__file__).with_name(name)).read_bytes().replace(b"\r\n", b"\n")) for name in names})
    return code


def runtime_identity():
    import vita
    return {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
            "vita": importlib.metadata.version("vita"), "vita_entry_sha256": digest(Path(vita.__file__)),
            "device": "cpu", "precision": "float32", "torch_threads": 2, "native_workers": 1}


def make_plan(fit, bridge, cache, pool, feedback, software):
    prior, proof = read(fit / "plan.json"), read(fit / "verification.json")
    bridge_plan, qualified = read(bridge / "plan.json"), read(bridge / "summary.json")
    listener, check = read(feedback), read(software)
    if (not proof["verified"] or not proof["complete"] or not proof["offline_training_replay_exact"]
            or proof["plan_id"] != canonical_sha256(prior) or proof["qualified_for_release"] or proof["test_split_used"]
            or not qualified["initialization_gradient_reference_recovery_passed"] or not qualified["parent_artifacts_unchanged"]
            or qualified["new_native_requests"] or qualified["experimental_training_updates"]
            or canonical_sha256(bridge_plan) != prior["bridge_plan_id"] or qualified["plan_id"] != prior["bridge_plan_id"]
            or qualified["model_file_sha256"] != digest(bridge / "model.pt")
            or digest(cache) != prior["input_cache_sha256"] or pool.pool_id != prior["pool_id"]
            or prior["render_phase_controls"] != FULL_PHASE_CONTROL or prior["render_profile"]["preroll_frames"] != 32768
            or not listener["completed_trials"] or not listener["tone_preferences_confounded_by_loudness"]
            or listener["qualified_for_model_selection"] or not check["passed"] or check["native_requests"] or check["experimental_updates"]):
        raise ValueError("direct fitting requires qualified initialization, unchanged training evidence, actual feedback and passing software checks")
    if any(digest(fit / name) != proof["evidence_sha256"].get(name) for name in ("plan.json", "finals.jsonl")):
        raise ValueError("verified fit plan or comparisons changed")
    code = code_identity(prior)
    if check["source_sha256"] != {name: code[name] for name in SOURCE_NAMES}:
        raise ValueError("direct software qualification belongs to different implementation bytes")
    model, _frontend, codec = restore_direct(torch.load(bridge / "model.pt", map_location="cpu", weights_only=True))
    if tensor_digest(model.state_dict()) != qualified["model_sha256"] or codec.to_dict() != pool.codec.to_dict():
        raise ValueError("qualified initializer model or codec changed")
    bank, cases, files = feature_bank(cache), [], {}
    finals = {(row["index"], row["arm"]): row for row in lines(fit / "finals.jsonl")}
    frames, residents = [], []
    for training_index, note, categories in ((2, 36, [0, 1]), (8, 61, [2, 0]), (11, 83, [2, 1])):
        recipe = prior["training"][training_index]
        if pool.manifest.by_state[recipe["state_id"]].split != "train" or recipe["target"]["categorical"] != categories:
            raise ValueError("direct check must retain the three declared training sounds")
        pitches = {event["data"][0] for event in recipe["performances"][1]["events"] if event["kind"] == "note_on" and event["data"][1] > 0}
        if pitches != {note}:
            raise ValueError("declared diagnostic register changed")
        features = cached_features(bank, recipe, 1)
        with torch.no_grad():
            initial = predicted_targets(model(batch_features([{"features": features}])), codec)[0]
        controls = []
        for view in (1, 0):
            indices = [i for i, cell in enumerate(prior["fit_audit_cells"]) if cell["training_index"] == training_index
                       and cell["input_view"] == 1 and cell["playback_view"] == view]
            if len(indices) != 1:
                raise ValueError("direct check requires the declared held/original comparison pair")
            row = finals[indices[0], "parameter"]
            source_file = f"source_audio/{recipe['source_audio_sha256'][view]}.pt"
            for name in (source_file, row["receipt"]["retained_audio"]["file"]):
                actual = digest(fit / name)
                if actual != proof["evidence_sha256"].get(name):
                    raise ValueError("direct source or parameter-control waveform changed")
                files[f"fit/{name}"] = actual
            saved = torch.load(fit / source_file, map_location="cpu", weights_only=True)
            audio = saved["audio"].numpy()
            performance = Performance.from_dict(recipe["performances"][view])
            count = TempoMap.from_performance(performance).render_frame_count(performance.end_tick, codec.contract.tail_seconds, 44100)
            if (set(saved) != {"audio"} or audio.dtype != np.float32 or audio.shape != (count, 2)
                    or audio_float32_sha256(audio) != recipe["source_audio_sha256"][view]):
                raise ValueError("direct source waveform does not match its frozen recipe")
            require_request(codec, prior, PilotTarget.from_dict(row["candidate"]), performance, row["receipt"])
            if rescore_waveform(fit, row["receipt"], audio, codec.contract.tail_seconds) != row["scores"]:
                raise ValueError("matched parameter-control comparison cannot be reproduced")
            controls.append(row)
            frames.append(count)
            residents.append(target_tensor_bytes(count) + 8 * count)
        if controls[0]["candidate"] != controls[1]["candidate"]:
            raise ValueError("parameter control must be one patch measured under both performances")
        cases.append({"training_index": training_index, "held_midi": note, "recipe": recipe,
                      "initial": initial.to_dict(), "parameter": controls, "views": [1, 0]})
    files.update({f"fit/{name}": digest(fit / name) for name in ("plan.json", "verification.json", "finals.jsonl")})
    files.update({f"bridge/{name}": digest(bridge / name) for name in ("plan.json", "summary.json", "model.pt")})
    files.update(features=digest(cache), feedback=digest(feedback), software=digest(software))
    storage = {"retained_initial_final_repeated_audio_bytes": 4 * 8 * sum(frames),
               "ledger_checkpoint_and_recovery_allowance_bytes": 128 * 1024**2,
               "resident_sources_and_targets_upper_bound_bytes": sum(sorted(residents, reverse=True)[:2]),
               "archive_guard_bytes": 32 * 1024**2}
    storage["estimated_archive_upper_bound_bytes"] = sum(storage.values())
    if storage["estimated_archive_upper_bound_bytes"] > LIMITS["archive_bytes"] or storage["resident_sources_and_targets_upper_bound_bytes"] > LIMITS["replay_bytes"]:
        raise ValueError("direct retained audio or resident targets exceed the fixed limits")
    return {"format": "direct_basic_control_check_v1", "source": code, "inputs": files, "pool_id": pool.pool_id,
            "fit_plan_id": proof["plan_id"], "bridge_plan_id": qualified["plan_id"], "codec": codec.to_dict(), "cases": cases,
            "optimizer": SPEC, "limits": LIMITS, "storage_bound": storage, "runtime": runtime_identity(),
            "planned_maximum_native_requests": 4200, "maximum_native_requests": 4608,
            "render_profile": prior["render_profile"], "render_phase_controls": prior["render_phase_controls"],
            "expected_renderer_identity": prior["expected_renderer_identity"], "test_split_used": False,
            "inverse_training_updates": 0, "model_admission_allowed": False,
            "trial_rejection": "silence_clipping_or_tail_only; other_invalid_audio_or_any_gradient_category_error_fails",
            "recovery": "at_most_one_known_identical_RenderTimeout; original_deadline_and_all_attempts_preserved",
            "objective": "equal_mean_of_original_raw_combined_spectral_loss_on_held_and_original_performances",
            "listening_feedback": listener}


def validate(plan, fit, bridge, cache, feedback, software):
    locations = {"fit": fit, "bridge": bridge}
    direct = {"features": cache, "feedback": feedback, "software": software}
    for name, expected in plan["inputs"].items():
        path = direct[name] if name in direct else locations[name.split("/", 1)[0]] / name.split("/", 1)[1]
        if digest(path) != expected:
            raise ValueError("frozen direct-control input bytes changed")
    if (code_identity(plan) != plan["source"] or runtime_identity() != plan["runtime"] or plan["optimizer"] != SPEC or plan["limits"] != LIMITS):
        raise ValueError("frozen direct-control code, runtime or allocation changed")

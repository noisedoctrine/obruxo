"""Bind the small unison fit to verified recordings, presets and source audio."""

from pathlib import Path

import torch

from obruxo_data.hashing import canonical_sha256, sha256_bytes
from obruxo_data.midi import Performance, TempoMap
from obruxo_data.render.base import RenderRequest
from obruxo_data.render.qa import audio_float32_sha256

from .conditioned_bridge import read
from .conditioned_short_audit import cached_features, feature_bank
from .features import tensor_digest
from .model import batch_features
from .native_fit import digest
from .native_fit_report import public_cases
from .paired_native_step import reciprocal_updates
from .production_controls import ProductionControlAdapter
from .reproducibility import fixed_generator_phases
from .unison_choice import COUNTS, SPEC, restore_unison

LIMITS = {"wall_seconds": 1200, "process_tree_bytes": 4 * 1024**3, "archive_bytes": 1024**3, "replay_bytes": 64 * 1024**2,
          "optimizer_attempts": 193, "native_executions": 2, "training_attempts": 176, "qualification_attempts": 224}


def require_request(codec, plan, target, performance, receipt):
    raw = codec.decode(target)
    controlled = fixed_generator_phases(raw)
    request = RenderRequest(preset_json=controlled.decode(), performance=performance, renderer_id=codec.contract.renderer_id,
                            tail_seconds=codec.contract.tail_seconds, preroll_frames=plan["render_profile"]["preroll_frames"])
    if (receipt["status"] != "accepted" or receipt["renderer_identity"] != plan["expected_renderer_identity"]
            or receipt["request_id"] != request.request_id or receipt["source_sha256"] != sha256_bytes(raw)
            or receipt["controlled_sha256"] != sha256_bytes(controlled)):
        raise ValueError("unison evidence differs from its patch, performance or renderer")


def storage_bound(cells, tail_seconds):
    frames, sources = [], {}
    for cell in cells:
        performance = Performance.from_dict(cell["recipe"]["performances"][cell["playback_view"]])
        count = TempoMap.from_performance(performance).render_frame_count(performance.end_tick, tail_seconds, 44100)
        frames.append(count)
        sources[cell["recipe"]["source_audio_sha256"][cell["playback_view"]]] = count
    values = {"three_choice_training_audio_bytes": 3 * 8 * sum(frames[:84]),
              "additional_initial_regression_audio_bytes": 8 * sum(frames[84:]),
              "fresh_two_arm_comparison_audio_bytes": 2 * 8 * sum(frames), "unique_source_audio_bytes": 8 * sum(sources.values()),
              "serialization_allowance_bytes": (len(frames) * 3 + 168 + len(sources)) * 16384,
              "checkpoint_export_interruption_reserve_bytes": 64 * 1024**2, "ledger_publication_reserve_bytes": 64 * 1024**2,
              "input_feature_allowance_bytes": 4 * 1024**2, "transient_waveform_allowance_bytes": 4 * 8 * max(frames),
              "archive_guard_bytes": 32 * 1024**2}
    total = sum(values.values())
    if total > LIMITS["archive_bytes"]:
        raise ValueError("planned unison evidence exceeds 1 GiB")
    return {**values, "estimated_archive_upper_bound_bytes": total, "candidate_waveforms_maximum": 480, "unique_sources": len(sources)}


def make_plan(parent, bridge, cache, pool):
    proof, qualified = read(parent / "verification.json"), read(bridge / "summary.json")
    prior, bridge_plan = proof["plan"], read(bridge / "plan.json")
    if (not proof["verified"] or not proof["offline_training_replay_exact"] or not proof["all_native_accepted"] or not proof["complete"]
            or proof["committed_updates_per_arm"] != 48 or proof["comparison_waveforms_rescored"] != 312
            or not qualified["initialization_and_gradient_access_passed"] or not qualified["parent_artifacts_unchanged"]
            or qualified["plan_id"] != canonical_sha256(bridge_plan) or qualified["model_budget"] != COUNTS
            or digest(bridge / "model.pt") != qualified["model_file_sha256"] or digest(cache) != prior["input_cache_sha256"]
            or prior["pool_id"] != pool.pool_id or prior["validation"] or prior["test_split_used"]):
        raise ValueError("unison fit requires the verified short fit, qualified bridge and unchanged cache/pool")
    files = {f"parent/{name}": digest(parent / name) for name in proof["evidence_sha256"]}
    if {name: files[f"parent/{name}"] for name in proof["evidence_sha256"]} != proof["evidence_sha256"]:
        raise ValueError("verified parent evidence changed")
    cases = read(parent / "fit_cases.json")
    if public_cases(cases) != proof["fit_cases"] or len(cases) != 104 or len(prior["training"]) != 12:
        raise ValueError("verified parent comparison cases changed")
    files["parent/fit_cases.json"] = digest(parent / "fit_cases.json")
    files["parent/verification.json"] = digest(parent / "verification.json")
    for case in cases:
        for name in (f"audits/{case['label']}.pt", case["rows"]["initial"]["receipt"]["retained_audio"]["file"]):
            files[f"parent/{name}"] = digest(parent / name)
    files.update({f"bridge/{name}": digest(bridge / name) for name in ("model.pt", "plan.json", "summary.json")})
    training_root = Path(__file__).parents[1]
    for name, value in bridge_plan["source"].items():
        path = training_root / name if "/" in name else Path(__file__).with_name(name)
        if sha256_bytes(path.read_bytes().replace(b"\r\n", b"\n")) != value:
            raise ValueError("qualified unison implementation changed")
    names = ("unison_fit_data.py", "unison_sound_fit.py", "unison_fit_audit.py", "paired_native_evidence.py", "paired_native_fit.py",
             "native_fit_report.py", "conditioned_short_audit.py", "benchmark.py", "checkpoints.py", "local_execution.py")
    code = {name: sha256_bytes(Path(__file__).with_name(name).read_bytes().replace(b"\r\n", b"\n")) for name in names}
    train_ids = {key for key, assignment in pool.manifest.by_state.items() if assignment.split == "train"}
    if not (set(prior["regression_state_ids"]) | {cell["recipe"]["state_id"] for cell in prior["fit_audit_cells"]}).issubset(train_ids):
        raise ValueError("unison fitting contains held-out states")
    return {"format": "independent_unison_sound_fit_v1", "input_files": files, "input_cache_sha256": digest(cache), "pool_id": pool.pool_id,
            "bridge_plan_id": qualified["plan_id"], "initial_file_sha256": qualified["model_file_sha256"], "source": {**bridge_plan["source"], **code},
            "training": prior["training"], "fit_audit_cells": prior["fit_audit_cells"], "regression_state_ids": prior["regression_state_ids"],
            "training_view_roles": prior["training_view_roles"], "render_profile": prior["render_profile"], "render_phase_controls": prior["render_phase_controls"],
            "expected_renderer_identity": prior["expected_renderer_identity"], "model_budget": COUNTS, "model_spec": SPEC,
            "updates": sum((reciprocal_updates(12, seed=seed) for seed in range(8)), []), "epoch_shuffle_seeds": list(range(8)),
            "optimizer": {"name": "Adam", "learning_rate": .001, "seed": 0, "gradient_norm_cap": 5., "updates_per_arm": 192},
            "reconstruction_objective": "mean_input_and_transfer_v1", "rng_initialization": "python_numpy_and_torch_from_optimizer_seed",
            "loss": "expected_measured_three_choice_sound_cost_temperature_one", "parameter_control": "unison_cross_entropy",
            "limits": LIMITS, "prefix_check_updates": 4, "planned_native_requests": {"training": 168, "qualification": 208, "total": 376},
            "storage_bound": storage_bound(prior["fit_audit_cells"], pool.codec.contract.tail_seconds),
            "gates": {"oracle_available_both_wide_views": True, "fraction_of_available_gain": .5, "six_groups_nonregression": True,
                      "original_32_mean_nonregression": True, "original_32_mrstft_p90_ratio": 1.1},
            "runtime": {"threads": 2, "device": "cpu", "precision": "float32", "native_workers": 1},
            "test_split_used": False, "model_admission_allowed": False}


class Inputs:
    def __init__(self, parent, bridge, cache, plan):
        self.parent, self.plan, self.initial_path = parent, plan, bridge / "model.pt"
        self.model, self.frontend, self.codec = restore_unison(torch.load(self.initial_path, map_location="cpu", weights_only=True))
        self.adapter = ProductionControlAdapter(self.codec, scales=SPEC["adapter_scales"])
        self.cases = read(parent / "fit_cases.json")
        bank = feature_bank(cache)
        self.features, self.baselines = {}, {}
        for cell in plan["fit_audit_cells"]:
            recipe, view = cell["recipe"], cell["input_view"]
            key = recipe["state_id"], view
            if key in self.features:
                continue
            self.features[key] = cached_features(bank, recipe, view)
            with torch.no_grad():
                self.baselines[key] = self.model(batch_features([{"features": self.features[key]}]))
        self.bytes = sum(value.numel() * value.element_size() for features in self.features.values() for value in features.values())
        if self.bytes > plan["limits"]["replay_bytes"]:
            raise ValueError("unison input features exceed replay allocation")
        self.lookup = {(cell["recipe"]["state_id"], cell["input_view"], cell["playback_view"]): index
                       for index, cell in enumerate(plan["fit_audit_cells"])}

    def key(self, cell):
        return cell["recipe"]["state_id"], cell["input_view"]

    def predict(self, model, cell):
        prediction = model(batch_features([{"features": self.features[self.key(cell)]}]))
        reference = self.baselines[self.key(cell)]
        if not torch.equal(prediction["continuous"], reference["continuous"]) or not torch.equal(prediction["categorical"][1], reference["categorical"][1]):
            raise ValueError("unison fit changed continuous or filter predictions")
        return prediction

    def target(self, index, choice=None, *, prediction=None):
        cell = self.plan["fit_audit_cells"][index]
        prediction = prediction if prediction is not None else self.baselines[self.key(cell)]
        logits = prediction["categorical"]
        if choice is not None:
            logits = [torch.nn.functional.one_hot(torch.tensor([choice]), 3).float(), logits[1]]
        return self.adapter.export(prediction["continuous"], logits)[0]

    def source(self, index):
        cell, case = self.plan["fit_audit_cells"][index], self.cases[index]
        recipe, playback = cell["recipe"], cell["playback_view"]
        audit = torch.load(self.parent / "audits" / f"{cell['label']}.pt", map_location="cpu", weights_only=True)
        if (case["label"] != cell["label"] or case["input_view"] != cell["input_view"] or case["playback_view"] != playback
                or audit["contract"]["recipe"] != recipe or audit["contract"]["view"] != playback
                or audit["rows"] != case["rows"] or audit["source_receipt"] != case["source_receipt"]):
            raise ValueError("parent unison source contract changed")
        audio = audit["audio"].numpy()
        if audio_float32_sha256(audio) != recipe["source_audio_sha256"][playback] or audio_float32_sha256(audio) != case["source_receipt"]["qa"]["audio_float32_sha256"]:
            raise ValueError("parent unison source waveform changed")
        from .codec import PilotTarget
        require_request(self.codec, self.plan, PilotTarget.from_dict(recipe["target"]), Performance.from_dict(recipe["performances"][playback]), case["source_receipt"])
        return audit["audio"], case["source_receipt"]

    def validate(self):
        previous = self.frontend.pitch_batch_size
        # Imported feature hashes must be checked in their recorded numerical layout.
        self.frontend.pitch_batch_size = self.plan.get("source_pitch_batch_size", 1)
        try:
            self._validate_sources()
        finally:
            self.frontend.pitch_batch_size = previous
        self.frontend.require_frozen()

    def _validate_sources(self):
        checked = set()
        for index, cell in enumerate(self.plan["fit_audit_cells"]):
            audio, _receipt = self.source(index)
            recipe, playback = cell["recipe"], cell["playback_view"]
            key = recipe["state_id"], playback
            if key not in checked:
                if tensor_digest(self.frontend(audio.numpy())) != recipe["source_feature_sha256"][playback]:
                    raise ValueError("retained unison source features cannot be reconstructed")
                checked.add(key)
            initial = self.cases[index]["rows"]["initial"]
            original = self.target(index, prediction=self.predict(self.model, cell))
            if initial["candidate"] != original.to_dict():
                raise ValueError("qualified unison initializer differs from retained initial prediction")
            performance = Performance.from_dict(recipe["performances"][playback])
            require_request(self.codec, self.plan, original, performance, initial["receipt"])
